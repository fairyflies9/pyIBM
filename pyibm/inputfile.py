"""The input2d-style parameter file for pyIBM (docs/user-guide.md describes every key).

The file looks like IB2d's input2d but the grammar is our own and deliberately loose:

    % a comment, also with #
    Fluid_Parameters { rho = 1.0  mu = 0.01 }
    Temporal_Information {
        Tfinal = 1.0
        dt = 1e-4
    }

Tokens are a section name followed by '{', then 'key = value' pairs in any layout (several per
line, one per line, braces on their own lines or not), then '}'. Values are numbers, quoted
strings, true/false, or bare words (taken as strings, so backend = mac works as well as
backend = "mac"). Quoted strings are taken literally: a backslash is an ordinary character, so
a Windows path that ends in a backslash reads back as written. Unknown sections or keys raise
ValueError naming them. Keys left out take the defaults of the dataclasses below; the fluid,
time and grid sections have no defaults and are required. Files are read as UTF-8, with or
without a byte-order mark.

Nothing in this file is derived from IB2d source; only the section and key names were chosen
to look familiar.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import numpy as np

from pyibm.grid import Grid

__all__ = [
    "FluidParams", "TimeParams", "GridParams", "SolverParams", "BCParams",
    "StructureParams", "OutputParams", "HooksParams", "Parameters",
    "tokenize", "parse_input_text", "parameters_from_sections", "validate_parameters",
    "read_input_file", "write_input_file", "format_input_file", "make_grid",
    "SECTION_NAMES", "BACKENDS", "TIME_SCHEMES", "COUPLINGS", "KERNELS", "BLOBS",
    "ADVECTIONS", "VISCOUS", "BLOB_INTERPOLATIONS", "MAC_PROJECTIONS", "BC_KINDS",
]


# ====================================================================== parameter dataclasses

@dataclass
class FluidParams:
    """rho [kg/m^2 in 2D], mu [N s/m^2]."""
    rho: float
    mu: float


@dataclass
class TimeParams:
    """Final time and time step [s]. t_n = n*dt is computed, never accumulated."""
    Tfinal: float
    dt: float

    @property
    def n_steps(self) -> int:
        """Number of steps to reach Tfinal (rounded; the driver may override with --steps)."""
        return int(round(self.Tfinal / self.dt))


@dataclass
class GridParams:
    """Cells Nx, Ny and box size Lx, Ly [m]; the domain is [0, Lx) x [0, Ly)."""
    Nx: int
    Ny: int
    Lx: float
    Ly: float

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny


BACKENDS = ("periodic_fft", "mac")
TIME_SCHEMES = ("midpoint", "pm1996")
COUPLINGS = ("delta", "blob")
KERNELS = ("peskin4", "cosine4", "roma3")
BLOBS = ("gaussian4", "compact")
ADVECTIONS = ("blend", "central", "upwind")
VISCOUS = ("implicit", "explicit")
BLOB_INTERPOLATIONS = ("streamfunction", "bicubic", "adjoint")
MAC_PROJECTIONS = ("nonincremental", "incremental")
BC_KINDS = ("periodic", "dirichlet", "outflow")


@dataclass
class SolverParams:
    """Back end, time scheme, coupling and their options (the Solver section).

    coupling = "blob" is the blob projection method of Cortez & Minion (2000).
    blob_interpolation picks how it gets the marker velocities: "streamfunction" (the default)
    takes the curl of the interpolated discrete stream function, which is exactly
    divergence-free; "bicubic" interpolates u and v directly, and the interpolated field is
    divergence-free only to O(h^2), so the area inside a closed membrane can drift slowly;
    "adjoint" is divergence-free and conserves energy, at a higher cost. mac_projection picks
    the pressure update of the MAC projection step. docs/numerics.md compares the choices.

    Cortez, R. & Minion, M. (2000). The blob projection method for immersed boundary problems.
    J. Comput. Phys. 161, 428-453.
    """
    backend: str = "periodic_fft"
    time_scheme: str = "midpoint"
    coupling: str = "delta"
    kernel: str = "peskin4"
    remove_mean_force: bool = False
    blob: str = "gaussian4"
    blob_delta: float = 3.0      # blob scale in units of dx
    patch: int = 0               # blob patch half-width in cells; 0 = automatic
    blob_interpolation: str = "streamfunction"   # blob only: "streamfunction", "bicubic", "adjoint"
    advection: str = "blend"     # mac only
    viscous: str = "implicit"    # mac only
    mac_projection: str = "nonincremental"       # mac only: "nonincremental" or "incremental"


@dataclass
class BCParams:
    """Kind of each side: "periodic", "dirichlet" or "outflow" (mac back end)."""
    left: str = "periodic"
    right: str = "periodic"
    bottom: str = "periodic"
    top: str = "periodic"


@dataclass
class StructureParams:
    """Structure file name (None = fluid only), optional ds, gravity on mass points."""
    name: str | None = None
    ds: float | None = None
    gravity_x: float = 0.0
    gravity_y: float = 0.0

    @property
    def gravity(self) -> tuple[float, float]:
        return (self.gravity_x, self.gravity_y)


@dataclass
class OutputParams:
    """Output cadence and switches."""
    print_dump: int = 100
    save_vtk: bool = True
    save_history: bool = True
    plot: bool = False
    output_dir: str = "viz"


@dataclass
class HooksParams:
    """The hooks file of the case, relative to the case directory. A missing hooks.py means
    no hooks; any other name must exist."""
    file: str = "hooks.py"


@dataclass
class Parameters:
    """Everything the input file holds."""
    fluid: FluidParams
    time: TimeParams
    grid: GridParams
    solver: SolverParams = field(default_factory=SolverParams)
    bc: BCParams = field(default_factory=BCParams)
    structure: StructureParams = field(default_factory=StructureParams)
    output: OutputParams = field(default_factory=OutputParams)
    hooks: HooksParams = field(default_factory=HooksParams)


# (section name in the file, attribute of Parameters, dataclass)
_SECTIONS = (
    ("Fluid_Parameters", "fluid", FluidParams),
    ("Temporal_Information", "time", TimeParams),
    ("Grid_Parameters", "grid", GridParams),
    ("Solver", "solver", SolverParams),
    ("Boundary_Conditions", "bc", BCParams),
    ("Lag_Structure_Info", "structure", StructureParams),
    ("Output_Info", "output", OutputParams),
    ("Hooks", "hooks", HooksParams),
)
SECTION_NAMES = tuple(s[0] for s in _SECTIONS)

# Expected kind of every key. We keep this explicit rather than reading the annotations
# because `from __future__ import annotations` turns them into strings.
_KINDS: dict[type, dict[str, str]] = {
    FluidParams: {"rho": "float", "mu": "float"},
    TimeParams: {"Tfinal": "float", "dt": "float"},
    GridParams: {"Nx": "int", "Ny": "int", "Lx": "float", "Ly": "float"},
    SolverParams: {"backend": "str", "time_scheme": "str", "coupling": "str", "kernel": "str",
                   "remove_mean_force": "bool", "blob": "str", "blob_delta": "float",
                   "patch": "int", "blob_interpolation": "str", "advection": "str",
                   "viscous": "str", "mac_projection": "str"},
    BCParams: {"left": "str", "right": "str", "bottom": "str", "top": "str"},
    StructureParams: {"name": "str", "ds": "float", "gravity_x": "float", "gravity_y": "float"},
    OutputParams: {"print_dump": "int", "save_vtk": "bool", "save_history": "bool",
                   "plot": "bool", "output_dir": "str"},
    HooksParams: {"file": "str"},
}


# ====================================================================== tokenizer

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>[%#][^\n]*)
  | (?P<string>"[^"\n]*"|'[^'\n]*')
  | (?P<punct>[{}=])
  | (?P<word>[^\s{}="'%#]+)
    """,
    re.VERBOSE,
)

_INT_RE = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class Token:
    kind: str       # "string", "punct" or "word"
    text: str       # raw text (quotes stripped for strings)
    line: int


def tokenize(text: str, source: str = "<input>") -> list[Token]:
    """Split the file into tokens, dropping whitespace and %/# comments.

    Strings keep their spaces and may contain % and #; a '%' or '#' anywhere else starts a
    comment that runs to the end of the line. Strings are literal: there are no escapes, so a
    backslash is an ordinary character, and a string cannot hold its own quote character (use
    the other quote). Braces and '=' are tokens on their own, so 'Name{Nx=16}' tokenizes like
    'Name { Nx = 16 }'. A leading byte-order mark is ignored.
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    tokens: list[Token] = []
    pos, line = 0, 1
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            snippet = text[pos:pos + 20].split("\n")[0]
            raise ValueError(f"{source} line {line}: cannot read {snippet!r} "
                             "(unterminated quote?)")
        kind = m.lastgroup
        chunk = m.group(0)
        if kind == "string":
            tokens.append(Token("string", chunk[1:-1], line))
        elif kind in ("punct", "word"):
            tokens.append(Token(kind, chunk, line))
        line += chunk.count("\n")
        pos = m.end()
    return tokens


def _value_of(tok: Token) -> Any:
    """Turn a value token into int, float, bool or str."""
    if tok.kind == "string":
        return tok.text
    s = tok.text
    if _INT_RE.match(s):
        return int(s)
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return float(s)
    except ValueError:
        return s            # a bare word is a string


# ====================================================================== parser

def parse_input_text(text: str, source: str = "<input>") -> dict[str, dict[str, Any]]:
    """Parse the file text into {section: {key: value}} without any knowledge of the schema.

    A section may appear more than once (its keys are merged); a key given twice in the same
    section is an error, as is a 'key = value' outside every section.
    """
    tokens = tokenize(text, source)
    sections: dict[str, dict[str, Any]] = {}
    i, n = 0, len(tokens)

    def fail(tok: Token | None, msg: str) -> ValueError:
        where = f"{source} line {tok.line}" if tok is not None else f"{source} end of file"
        return ValueError(f"{where}: {msg}")

    while i < n:
        tok = tokens[i]
        if tok.kind != "word":
            raise fail(tok, f"expected a section name, got {tok.text!r}")
        if i + 1 < n and tokens[i + 1].kind == "punct" and tokens[i + 1].text == "=":
            raise fail(tok, f"'{tok.text} = ...' is outside every section; put it inside "
                            "'Section { ... }'")
        if i + 1 >= n or tokens[i + 1].text != "{":
            raise fail(tok, f"expected '{{' after section name {tok.text!r}")
        name = tok.text
        body = sections.setdefault(name, {})
        i += 2
        while True:
            if i >= n:
                raise fail(None, f"section {name!r} is not closed with '}}'")
            tok = tokens[i]
            if tok.kind == "punct" and tok.text == "}":
                i += 1
                break
            if tok.kind == "punct":
                raise fail(tok, f"unexpected {tok.text!r} in section {name!r}")
            if tok.kind == "string":
                raise fail(tok, f"expected a key name in section {name!r}, got a quoted string")
            key = tok.text
            if i + 1 >= n or tokens[i + 1].text != "=" or tokens[i + 1].kind != "punct":
                raise fail(tok, f"expected '=' after key {key!r} in section {name!r}")
            if i + 2 >= n or tokens[i + 2].kind == "punct":
                raise fail(tok, f"missing value for key {key!r} in section {name!r}")
            if key in body:
                raise fail(tok, f"key {key!r} given twice in section {name!r}")
            body[key] = _value_of(tokens[i + 2])
            i += 3
    return sections


# ====================================================================== building Parameters

def _coerce(value: Any, kind: str, section: str, key: str) -> Any:
    """Check and convert a parsed value to the kind the dataclass expects."""
    what = f"{section}.{key}"
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"{what} must be true/false or 0/1, got {value!r}")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{what} must be an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{what} must be a number, got {value!r}")
        return float(value)
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError(f"{what} must be a string (quote it), got {value!r}")
        return value
    raise AssertionError(kind)


def parameters_from_sections(sections: dict[str, dict[str, Any]], source: str = "<input>"
                             ) -> Parameters:
    """Build Parameters from the parsed sections, applying defaults and checking keys.

    Unknown sections and keys raise ValueError naming them; a required key without a default
    that is missing raises ValueError naming section and key. Validation of the values as a
    whole happens in validate_parameters.
    """
    known = {name for name, _, _ in _SECTIONS}
    for name in sections:
        if name not in known:
            raise ValueError(f"{source}: unknown section {name!r}; known sections are "
                             + ", ".join(SECTION_NAMES))
    kwargs: dict[str, Any] = {}
    for name, attr, cls in _SECTIONS:
        body = sections.get(name, {})
        kinds = _KINDS[cls]
        for key in body:
            if key not in kinds:
                raise ValueError(f"{source}: unknown key {key!r} in section {name!r}; known keys "
                                 "are " + ", ".join(kinds))
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in body:
                values[f.name] = _coerce(body[f.name], kinds[f.name], name, f.name)
        missing = [f.name for f in fields(cls)
                   if f.name not in values and f.default is MISSING and f.default_factory is MISSING]
        if missing:
            raise ValueError(f"{source}: section {name!r} is missing required key(s) "
                             + ", ".join(missing))
        kwargs[attr] = cls(**values)
    return Parameters(**kwargs)


def validate_parameters(params: Parameters) -> Parameters:
    """Check the parameters as a whole and return them; a ValueError names the offending key.

    - dt > 0, Tfinal > 0, rho > 0, mu >= 0, Lx, Ly > 0, all finite;
    - Nx, Ny: an even integer >= 8 in a periodic direction (both directions on periodic_fft),
      any integer >= 4 in a direction with walls or outflow (the same rule as Grid);
    - every enumerated option is one of the known choices;
    - periodic_fft: all four sides periodic and dx == dy; blob needs periodic_fft;
    - mac: periodic sides come in opposite pairs; at most one outflow side;
    - ds > 0 when given, print_dump >= 1, blob_delta > 0, patch >= 0;
    - Hooks.file names a .py file (a name before the .py, too).
    The roma3 kernel on the periodic back end only warns, since the run still works; it has a
    checkerboard null space on the collocated periodic grid (docs/numerics.md, kernels).
    """
    fl, tm, gr, so, bc, st, out, hk = (params.fluid, params.time, params.grid, params.solver,
                                       params.bc, params.structure, params.output, params.hooks)
    if not (tm.dt > 0 and np.isfinite(tm.dt)):
        raise ValueError(f"Temporal_Information.dt must be finite and positive, got {tm.dt}")
    if not (tm.Tfinal > 0 and np.isfinite(tm.Tfinal)):
        raise ValueError(f"Temporal_Information.Tfinal must be finite and positive, "
                         f"got {tm.Tfinal}")
    if not (fl.rho > 0 and np.isfinite(fl.rho)):
        raise ValueError(f"Fluid_Parameters.rho must be finite and positive, got {fl.rho}")
    if not (fl.mu >= 0 and np.isfinite(fl.mu)):
        raise ValueError(f"Fluid_Parameters.mu must be finite and non-negative, got {fl.mu}")
    if not (gr.Lx > 0 and gr.Ly > 0 and np.isfinite(gr.Lx) and np.isfinite(gr.Ly)):
        raise ValueError(f"Grid_Parameters.Lx, Ly must be finite and positive, "
                         f"got {gr.Lx}, {gr.Ly}")
    for key, n in (("Nx", gr.Nx), ("Ny", gr.Ny)):
        if isinstance(n, bool) or not float(n).is_integer():
            raise ValueError(f"Grid_Parameters.{key} must be an integer, got {n}")

    def choice(section: str, key: str, value: str, options: tuple[str, ...]) -> None:
        if value not in options:
            raise ValueError(f"{section}.{key} must be one of {', '.join(options)}; got {value!r}")

    choice("Solver", "backend", so.backend, BACKENDS)
    choice("Solver", "time_scheme", so.time_scheme, TIME_SCHEMES)
    choice("Solver", "coupling", so.coupling, COUPLINGS)
    choice("Solver", "kernel", so.kernel, KERNELS)
    choice("Solver", "blob", so.blob, BLOBS)
    choice("Solver", "blob_interpolation", so.blob_interpolation, BLOB_INTERPOLATIONS)
    choice("Solver", "advection", so.advection, ADVECTIONS)
    choice("Solver", "viscous", so.viscous, VISCOUS)
    choice("Solver", "mac_projection", so.mac_projection, MAC_PROJECTIONS)
    for side in ("left", "right", "bottom", "top"):
        choice("Boundary_Conditions", side, getattr(bc, side), BC_KINDS)
    if not so.blob_delta > 0:
        raise ValueError(f"Solver.blob_delta must be positive, got {so.blob_delta}")
    if so.patch < 0:
        raise ValueError(f"Solver.patch must be >= 0 (0 = automatic), got {so.patch}")

    sides = (bc.left, bc.right, bc.bottom, bc.top)
    if so.backend == "periodic_fft":
        if any(s != "periodic" for s in sides):
            raise ValueError("the periodic_fft back end needs all four sides periodic; got "
                             f"left={bc.left}, right={bc.right}, bottom={bc.bottom}, top={bc.top}")
        periodic_x = periodic_y = True
    else:
        if (bc.left == "periodic") != (bc.right == "periodic"):
            raise ValueError("periodic must be set on both left and right")
        if (bc.bottom == "periodic") != (bc.top == "periodic"):
            raise ValueError("periodic must be set on both bottom and top")
        periodic_x, periodic_y = bc.left == "periodic", bc.bottom == "periodic"
    for key, n, periodic in (("Nx", gr.Nx, periodic_x), ("Ny", gr.Ny, periodic_y)):
        if periodic and (n < 8 or n % 2 != 0):
            raise ValueError(f"Grid_Parameters.{key} must be an even integer >= 8 in a periodic "
                             f"direction, got {n}")
        if not periodic and n < 4:
            raise ValueError(f"Grid_Parameters.{key} must be an integer >= 4 in a direction with "
                             f"walls or outflow, got {n}")
    if so.backend == "periodic_fft":
        if abs(gr.dx - gr.dy) > 1e-12 * max(gr.dx, gr.dy):
            raise ValueError(f"the periodic_fft back end needs dx == dy, got dx={gr.dx}, dy={gr.dy}")
        if so.kernel == "roma3":
            warnings.warn("the roma3 kernel has a checkerboard null space on the collocated "
                          "periodic grid; it is meant for the mac back end", stacklevel=2)
    elif sum(s == "outflow" for s in sides) > 1:
        raise ValueError("at most one outflow side is supported")
    if so.coupling == "blob" and so.backend != "periodic_fft":
        raise ValueError("coupling = \"blob\" requires backend = \"periodic_fft\"")

    if st.ds is not None and not st.ds > 0:
        raise ValueError(f"Lag_Structure_Info.ds must be positive, got {st.ds}")
    if out.print_dump < 1:
        raise ValueError(f"Output_Info.print_dump must be >= 1, got {out.print_dump}")
    base = ""                            # the file name without its folders
    if isinstance(hk.file, str):
        base = hk.file.replace("\\", "/").rsplit("/", 1)[-1]
    if not (base.endswith(".py") and base[:-3].strip()):
        raise ValueError(f"Hooks.file must name a Python file ending in .py, got {hk.file!r}")
    return params


def read_input_file(path: str) -> Parameters:
    """Read, build and validate an input file; returns Parameters. The file is read as UTF-8
    with an optional byte-order mark, which Windows editors and PowerShell often write."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read()
    sections = parse_input_text(text, source=str(path))
    params = parameters_from_sections(sections, source=str(path))
    return validate_parameters(params)


def make_grid(params: Parameters) -> Grid:
    """The Grid the parameters describe: periodic in x when left/right are periodic, in y when
    bottom/top are. The periodic back end's dx == dy check is done by validate_parameters."""
    g, bc = params.grid, params.bc
    return Grid(g.Lx, g.Ly, g.Nx, g.Ny,
                periodic_x=(bc.left == "periodic"), periodic_y=(bc.bottom == "periodic"))


# ====================================================================== writer

def _format_value(value: Any) -> str:
    """One value as input-file text, in a form the tokenizer reads back unchanged.

    numpy scalars (np.float64, np.int64, np.bool_, ...) are written as the Python values they
    hold; NumPy 2 would otherwise print them as 'np.float64(...)'. np.longdouble is rounded
    to a double, since that is what the reader gives back. Floats use repr, which
    reads back to the same double. A string is quoted with " unless it contains ", then
    with '; one that holds both quote characters or a line break cannot be written.
    """
    if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
        value = value.item()
    if isinstance(value, np.floating):   # .item() leaves np.longdouble as it is
        value = float(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            raise ValueError(f"cannot write a string containing a line break: {value!r}")
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
        raise ValueError(f"cannot write a string containing both quote characters: {value!r}")
    raise TypeError(f"cannot write a value of type {type(value).__name__}")


def format_input_file(params: Parameters, header: str | None = None) -> str:
    """Render Parameters as input-file text, every key written (None keys omitted)."""
    lines: list[str] = []
    if header:
        for ln in header.splitlines():
            lines.append(f"% {ln}" if ln else "%")
        lines.append("")
    for name, attr, cls in _SECTIONS:
        obj = getattr(params, attr)
        lines.append(f"{name} {{")
        for f in fields(cls):
            value = getattr(obj, f.name)
            if value is None:
                continue
            lines.append(f"    {f.name} = {_format_value(value)}")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def write_input_file(path: str, params: Parameters, header: str | None = None) -> None:
    """Write Parameters to path so that read_input_file(path) gives them back."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(format_input_file(params, header))
