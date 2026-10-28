"""Tests for pyibm.inputfile: grammar, defaults, validation and the writer round trip."""

import os

import numpy as np
import pytest

from pyibm import inputfile as inf
from pyibm.grid import Grid


SPEC_EXAMPLE = """
% comment  (also #)
Fluid_Parameters {
    rho = 1.0
    mu = 0.01
}
Temporal_Information { Tfinal = 1.0  dt = 1e-4 }       % several keys per line allowed
Grid_Parameters { Nx = 128  Ny = 128  Lx = 1.0  Ly = 1.0 }
Solver {
    backend = "periodic_fft"      % or "mac"
    time_scheme = "midpoint"      % or "pm1996"
    coupling = "delta"            % or "blob"
    kernel = "peskin4"            % "cosine4", "roma3"
    remove_mean_force = 0         % periodic back end: subtract the spatial mean of the force each step
    blob = "gaussian4"            % or "compact"; blob_delta = 3.0 (in units of dx); patch = 0 (auto)
    blob_interpolation = "bicubic"    % blob: "streamfunction" (default) or "bicubic"
    advection = "blend"           % mac: "blend" (Seibold gamma), "central", "upwind"
    viscous = "implicit"          % mac: "implicit" or "explicit"
    mac_projection = "incremental"    % mac: "nonincremental" (default) or "incremental"
}
Boundary_Conditions {             % mac only; each side: "periodic", "dirichlet", "outflow"
    left = "dirichlet"  right = "outflow"  bottom = "dirichlet"  top = "dirichlet"
}
Lag_Structure_Info { name = "plate"  ds = 0.0125  gravity_x = 0  gravity_y = 0 }
Output_Info { print_dump = 100  save_vtk = 1  save_history = 1  plot = 0  output_dir = "viz" }
Hooks { file = "hooks_oscillating.py" }
"""

MINIMAL = """
Fluid_Parameters { rho = 2  mu = 3 }
Temporal_Information { Tfinal = 0.5  dt = 0.001 }
Grid_Parameters { Nx = 16  Ny = 16  Lx = 0.4  Ly = 0.4 }
"""


def test_parse_spec_example():
    sections = inf.parse_input_text(SPEC_EXAMPLE)
    assert set(sections) == set(inf.SECTION_NAMES)
    assert sections["Temporal_Information"] == {"Tfinal": 1.0, "dt": 1e-4}
    assert sections["Grid_Parameters"]["Nx"] == 128 and isinstance(sections["Grid_Parameters"]["Nx"], int)
    assert sections["Solver"]["remove_mean_force"] == 0
    assert sections["Boundary_Conditions"]["right"] == "outflow"
    p = inf.parameters_from_sections(sections)      # the example mixes periodic_fft with walls,
    assert p.fluid.mu == 0.01                       # so build it without validating
    assert p.time.dt == 1e-4 and p.grid.Nx == 128 and p.grid.dx == pytest.approx(1 / 128)
    assert p.solver.remove_mean_force is False and p.solver.blob_delta == 3.0
    assert p.solver.blob_interpolation == "bicubic" and p.solver.mac_projection == "incremental"
    assert p.bc.right == "outflow"
    assert p.structure.name == "plate" and p.structure.ds == 0.0125 and p.structure.gravity == (0.0, 0.0)
    assert p.output.save_vtk is True and p.output.plot is False and p.output.output_dir == "viz"
    assert p.hooks == inf.HooksParams("hooks_oscillating.py")
    with pytest.raises(ValueError, match="periodic"):
        inf.validate_parameters(p)
    p.solver.backend = "mac"
    inf.validate_parameters(p)


def test_defaults(tmp_path):
    path = os.path.join(tmp_path, "input2d")
    with open(path, "w") as fh:
        fh.write(MINIMAL)
    p = inf.read_input_file(path)
    assert p.fluid == inf.FluidParams(2.0, 3.0)
    assert isinstance(p.fluid.rho, float) and isinstance(p.time.Tfinal, float)
    assert p.time.n_steps == 500
    assert p.solver == inf.SolverParams()
    assert p.solver.backend == "periodic_fft" and p.solver.time_scheme == "midpoint"
    assert p.solver.coupling == "delta" and p.solver.kernel == "peskin4"
    assert p.solver.remove_mean_force is False and p.solver.blob == "gaussian4"
    assert p.solver.patch == 0 and p.solver.advection == "blend" and p.solver.viscous == "implicit"
    assert p.solver.blob_interpolation == "streamfunction"
    assert p.solver.mac_projection == "nonincremental"
    assert p.bc == inf.BCParams("periodic", "periodic", "periodic", "periodic")
    assert p.structure == inf.StructureParams() and p.structure.name is None and p.structure.ds is None
    assert p.output == inf.OutputParams(100, True, True, False, "viz")
    assert p.hooks == inf.HooksParams() and p.hooks.file == "hooks.py"
    g = inf.make_grid(p)
    assert isinstance(g, Grid) and g.periodic_x and g.periodic_y
    assert g.Nx == 16 and g.dx == pytest.approx(0.025)


def test_comments_and_layout():
    text = """
    # hash comment
    Fluid_Parameters{rho=1 mu=0.5}   # no spaces needed
    Temporal_Information
    {
        Tfinal = 2 % percent comment after a value
        dt = 1e-2
    }
    Grid_Parameters { Nx = 8
        Ny = 8  Lx = 1  Ly = 1 }
    Output_Info { output_dir = "out % not a comment # either"  plot = true }
    Solver { backend = mac  kernel = 'roma3' }
    Boundary_Conditions { left = "dirichlet" right = "dirichlet" bottom = "dirichlet" top = "dirichlet" }
    """
    sections = inf.parse_input_text(text)
    assert sections["Fluid_Parameters"] == {"rho": 1, "mu": 0.5}
    assert sections["Temporal_Information"] == {"Tfinal": 2, "dt": 1e-2}
    assert sections["Output_Info"] == {"output_dir": "out % not a comment # either", "plot": True}
    assert sections["Solver"] == {"backend": "mac", "kernel": "roma3"}
    p = inf.validate_parameters(inf.parameters_from_sections(sections))
    assert p.time.Tfinal == 2.0 and p.output.output_dir == "out % not a comment # either"
    assert p.solver.backend == "mac" and p.solver.kernel == "roma3"
    g = inf.make_grid(p)
    assert not g.periodic_x and not g.periodic_y


def test_section_repeated_merges_and_empty_section():
    text = MINIMAL + "Solver { backend = \"mac\" }\nSolver { viscous = \"explicit\" }\nOutput_Info { }\n"
    text += "Boundary_Conditions { left = \"dirichlet\" right = \"dirichlet\" }\n"
    p = inf.validate_parameters(inf.parameters_from_sections(inf.parse_input_text(text)))
    assert p.solver.backend == "mac" and p.solver.viscous == "explicit"
    assert p.bc.left == "dirichlet" and p.bc.bottom == "periodic" and p.bc.top == "periodic"


def test_unknown_key_and_section():
    with pytest.raises(ValueError) as err:
        inf.parameters_from_sections(inf.parse_input_text(MINIMAL + "Solver { kernal = \"peskin4\" }"))
    assert "kernal" in str(err.value) and "Solver" in str(err.value)
    with pytest.raises(ValueError) as err:
        inf.parameters_from_sections(inf.parse_input_text(MINIMAL + "Fluid_Parameters { density = 1 }"))
    assert "density" in str(err.value)
    with pytest.raises(ValueError) as err:
        inf.parameters_from_sections(inf.parse_input_text(MINIMAL + "Fluid { rho = 1 }"))
    assert "Fluid" in str(err.value)


def test_missing_required_key_or_section():
    with pytest.raises(ValueError) as err:
        inf.parameters_from_sections(inf.parse_input_text(
            "Fluid_Parameters { rho = 1 }\nTemporal_Information { Tfinal = 1 dt = 0.1 }\n"
            "Grid_Parameters { Nx = 8 Ny = 8 Lx = 1 Ly = 1 }"))
    assert "mu" in str(err.value) and "Fluid_Parameters" in str(err.value)
    with pytest.raises(ValueError) as err:
        inf.parameters_from_sections(inf.parse_input_text(
            "Fluid_Parameters { rho = 1 mu = 1 }\nGrid_Parameters { Nx = 8 Ny = 8 Lx = 1 Ly = 1 }"))
    assert "Temporal_Information" in str(err.value)


@pytest.mark.parametrize("text, message", [
    ("Fluid_Parameters { rho = 1 rho = 2 }", "twice"),
    ("rho = 1", "outside"),
    ("Fluid_Parameters { rho = 1 ", "not closed"),
    ("Fluid_Parameters { rho 1 }", "expected '='"),
    ("Fluid_Parameters { rho = }", "missing value"),
    ("Fluid_Parameters rho = 1 }", "expected '{'"),
    ("Fluid_Parameters { rho = \"unterminated }", "unterminated"),
    ("{ rho = 1 }", "section name"),
    ("Fluid_Parameters { \"rho\" = 1 }", "quoted"),
])
def test_grammar_errors(text, message):
    with pytest.raises(ValueError, match=message):
        inf.parse_input_text(text)


def test_error_messages_carry_source_and_line():
    with pytest.raises(ValueError, match=r"input2d line 3"):
        inf.parse_input_text("\n\nFluid_Parameters { rho = 1 rho = 2 }", source="input2d")


def test_type_coercion():
    def build(extra):
        return inf.parameters_from_sections(inf.parse_input_text(MINIMAL + extra))

    p = build("Output_Info { print_dump = 32.0 }")
    assert p.output.print_dump == 32 and isinstance(p.output.print_dump, int)
    with pytest.raises(ValueError, match="integer"):
        build("Output_Info { print_dump = 32.5 }")
    with pytest.raises(ValueError, match="integer"):
        build("Output_Info { print_dump = true }")
    with pytest.raises(ValueError, match="number"):
        build("Solver { blob_delta = \"3\" }")
    with pytest.raises(ValueError, match="true/false"):
        build("Output_Info { save_vtk = 2 }")
    assert build("Output_Info { save_vtk = \"TRUE\" plot = False }").output.save_vtk is True
    assert build("Output_Info { plot = False }").output.plot is False
    with pytest.raises(ValueError, match="string"):
        build("Lag_Structure_Info { name = 2008 }")
    assert build("Lag_Structure_Info { name = plate2008 }").structure.name == "plate2008"
    assert build("Lag_Structure_Info { gravity_y = -9.81 }").structure.gravity == (0.0, -9.81)


def build_valid(**overrides):
    p = inf.parameters_from_sections(inf.parse_input_text(MINIMAL))
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        setattr(getattr(p, section), key, value)
    return p


@pytest.mark.parametrize("overrides, message", [
    ({"time.dt": 0.0}, "dt"),
    ({"time.dt": -1.0}, "dt"),
    ({"time.Tfinal": 0.0}, "Tfinal"),
    ({"fluid.rho": 0.0}, "rho"),
    ({"fluid.mu": -1.0}, "mu"),
    ({"grid.Nx": 15}, "even"),
    ({"grid.Ny": 4}, "even"),
    ({"grid.Nx": 16.5}, "integer"),
    # mac: even and >= 8 in a periodic direction, >= 4 in a direction with walls
    ({"solver.backend": "mac", "bc.bottom": "dirichlet", "bc.top": "dirichlet", "grid.Nx": 9}, "even"),
    ({"solver.backend": "mac", "bc.bottom": "dirichlet", "bc.top": "dirichlet", "grid.Nx": 6}, "even"),
    ({"solver.backend": "mac", "bc.left": "dirichlet", "bc.right": "dirichlet",
      "bc.bottom": "dirichlet", "bc.top": "dirichlet", "grid.Ny": 3}, ">= 4"),
    ({"grid.Lx": 0.0}, "Lx"),
    ({"grid.Ny": 32}, "dx == dy"),
    ({"bc.left": "dirichlet"}, "four sides periodic"),
    ({"solver.coupling": "blob", "solver.backend": "mac",
      "bc.left": "dirichlet", "bc.right": "dirichlet"}, "blob"),
    ({"solver.backend": "mac", "bc.left": "dirichlet"}, "both left and right"),
    ({"solver.backend": "mac", "bc.top": "outflow"}, "both bottom and top"),
    ({"solver.backend": "mac", "bc.left": "outflow", "bc.right": "outflow"}, "one outflow"),
    ({"solver.backend": "fft"}, "backend"),
    ({"solver.time_scheme": "rk4"}, "time_scheme"),
    ({"solver.kernel": "peskin6"}, "kernel"),
    ({"solver.blob": "gauss"}, "blob"),
    ({"solver.advection": "weno"}, "advection"),
    ({"solver.viscous": "cn"}, "viscous"),
    ({"solver.blob_interpolation": "quintic"}, "blob_interpolation"),
    ({"solver.mac_projection": "rotational"}, "mac_projection"),
    ({"bc.top": "wall"}, "top"),
    ({"solver.blob_delta": 0.0}, "blob_delta"),
    ({"solver.patch": -1}, "patch"),
    ({"structure.ds": 0.0}, "ds"),
    ({"output.print_dump": 0}, "print_dump"),
    ({"hooks.file": ""}, "Hooks.file"),
    ({"hooks.file": "hooks.txt"}, "Hooks.file"),
    ({"hooks.file": ".py"}, "Hooks.file"),
    ({"hooks.file": "   .py"}, "Hooks.file"),
    ({"hooks.file": "variants/.py"}, "Hooks.file"),
    ({"grid.Lx": float("inf")}, "Lx"),
    ({"grid.Ly": float("nan")}, "Ly"),
    ({"time.dt": float("inf")}, "dt"),
    ({"time.Tfinal": float("inf")}, "Tfinal"),
    ({"fluid.rho": float("inf")}, "rho"),
    ({"fluid.mu": float("nan")}, "mu"),
])
def test_validation_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        inf.validate_parameters(build_valid(**overrides))


def test_validation_accepts():
    inf.validate_parameters(build_valid())
    inf.validate_parameters(build_valid(**{"solver.coupling": "blob"}))
    p = build_valid(**{"solver.backend": "mac", "grid.Ny": 32, "bc.left": "dirichlet",
                       "bc.right": "outflow", "bc.bottom": "dirichlet", "bc.top": "dirichlet"})
    inf.validate_parameters(p)                                   # dx != dy is fine on mac
    g = inf.make_grid(p)
    assert not g.periodic_x and not g.periodic_y and g.shape_u() == (17, 32)
    p = build_valid(**{"solver.backend": "mac", "bc.bottom": "dirichlet", "bc.top": "dirichlet"})
    inf.validate_parameters(p)                                   # periodic in x only
    assert inf.make_grid(p).periodic_x and not inf.make_grid(p).periodic_y
    with pytest.warns(UserWarning, match="roma3"):
        inf.validate_parameters(build_valid(**{"solver.kernel": "roma3"}))
    # small and odd N are fine in a direction with walls or outflow (mac only)
    inf.validate_parameters(build_valid(**{"solver.backend": "mac", "bc.left": "dirichlet",
                                           "bc.right": "outflow", "bc.bottom": "dirichlet",
                                           "bc.top": "dirichlet", "grid.Nx": 5, "grid.Ny": 4}))
    inf.validate_parameters(build_valid(**{"solver.backend": "mac", "bc.bottom": "dirichlet",
                                           "bc.top": "dirichlet", "grid.Nx": 8, "grid.Ny": 7}))
    inf.validate_parameters(build_valid(**{"solver.blob_interpolation": "bicubic",
                                           "solver.coupling": "blob",
                                           "solver.mac_projection": "incremental",
                                           "hooks.file": "variants/hooks_fast.py"}))


def test_write_read_round_trip(tmp_path):
    p = build_valid(**{"solver.backend": "mac", "solver.remove_mean_force": True,
                       "solver.mac_projection": "incremental",
                       "solver.blob_interpolation": "bicubic",
                       "bc.left": "dirichlet", "bc.right": "outflow",
                       "structure.name": "my plate", "structure.ds": 1e-3, "structure.gravity_y": -9.81,
                       "output.output_dir": "viz/run 1", "output.print_dump": 7, "output.plot": True,
                       "time.dt": 1.2345678901234567e-5, "hooks.file": "hooks_oscillating.py"})
    p.bc.bottom = "dirichlet"
    p.bc.top = "dirichlet"
    path = os.path.join(tmp_path, "input2d")
    inf.write_input_file(path, p, header="written by the test\nsecond line")
    q = inf.read_input_file(path)
    assert q == p
    with open(path) as fh:
        text = fh.read()
    assert text.startswith("% written by the test\n% second line\n")
    assert 'name = "my plate"' in text and "remove_mean_force = true" in text
    assert 'mac_projection = "incremental"' in text and 'blob_interpolation = "bicubic"' in text
    assert 'Hooks {\n    file = "hooks_oscillating.py"\n}' in text
    # None keys are left out, so a fluid-only case round-trips too
    p2 = build_valid()
    inf.write_input_file(path, p2)
    with open(path) as fh:
        text = fh.read()
    assert "name" not in text.split("Lag_Structure_Info")[1].split("}")[0]
    assert inf.read_input_file(path) == p2


@pytest.mark.parametrize("value", [
    "viz\\run1\\",          # a Windows directory with a trailing backslash
    "C:\\out\\",
    "viz\\\\",              # an even number of trailing backslashes
    "a\\nb",                # backslash-n is two characters, not a line break
    'my "best" plate',      # holds a double quote: written with single quotes
    "Laura's plate",
    "",
])
def test_strings_round_trip_literally(tmp_path, value):
    p = build_valid(**{"output.output_dir": value})
    path = os.path.join(tmp_path, "input2d")
    inf.write_input_file(path, p)
    assert inf.read_input_file(path).output.output_dir == value


def test_strings_that_cannot_be_written():
    for bad in ('both " and \'', "two\nlines", "carriage\rreturn"):
        with pytest.raises(ValueError):
            inf.format_input_file(build_valid(**{"structure.name": bad}))


def test_numpy_scalars_are_written_as_plain_numbers(tmp_path):
    p = build_valid(**{"structure.name": np.str_("ring"), "structure.ds": np.float64(0.0123),
                       "time.dt": np.float64(1e-3) / 3, "fluid.mu": np.float32(0.25),
                       "grid.Nx": np.int64(16), "grid.Ny": np.int32(16),
                       "output.save_vtk": np.bool_(False), "output.print_dump": np.int64(5),
                       "solver.blob_delta": np.array(4.0), "fluid.rho": np.longdouble(1.5)})
    text = inf.format_input_file(p)
    assert "np." not in text and "ds = 0.0123" in text and "Nx = 16" in text
    path = os.path.join(tmp_path, "input2d")
    inf.write_input_file(path, p)
    q = inf.read_input_file(path)
    assert q.structure.ds == 0.0123 and q.time.dt == float(np.float64(1e-3) / 3)
    assert q.fluid.mu == float(np.float32(0.25)) and q.solver.blob_delta == 4.0
    assert q.fluid.rho == 1.5 and "rho = 1.5" in text
    assert q.grid.Nx == 16 and type(q.grid.Nx) is int and type(q.grid.Ny) is int
    assert q.output.save_vtk is False and q.output.print_dump == 5 and q.structure.name == "ring"


@pytest.mark.parametrize("first_line", ["% written with a byte-order mark\n", ""])
def test_input_file_with_byte_order_mark(tmp_path, first_line):
    # PowerShell 5.1 (Set-Content -Encoding utf8) and Notepad's "UTF-8 with BOM" write a BOM
    path = os.path.join(tmp_path, "input2d")
    with open(path, "w", encoding="utf-8-sig") as fh:
        fh.write(first_line + MINIMAL.lstrip())
    p = inf.read_input_file(path)
    assert p.grid.Nx == 16 and p.fluid.rho == 2.0
    assert inf.parse_input_text("\ufeff" + MINIMAL) == inf.parse_input_text(MINIMAL)


def test_tokenizer_details():
    toks = inf.tokenize('A{b=1 c="x y" d=-2.5e3}', source="s")
    assert [t.text for t in toks] == ["A", "{", "b", "=", "1", "c", "=", "x y", "d", "=", "-2.5e3", "}"]
    assert toks[7].kind == "string" and toks[10].kind == "word"
    assert inf.parse_input_text('A{b=1 c="x y" d=-2.5e3}') == {"A": {"b": 1, "c": "x y", "d": -2500.0}}
    assert inf.parse_input_text("A { b = +3  c = .5  d = 1.  e = TRUE  f = none }")["A"] == \
        {"b": 3, "c": 0.5, "d": 1.0, "e": True, "f": "none"}
    assert inf.parse_input_text("") == {}
    assert inf.parse_input_text("% only a comment\n# and another") == {}
    # strings are literal: no escapes, and the other quote character may appear inside
    assert inf.parse_input_text('A { p = "C:\\out\\"  q = \'say "hi"\'  r = "it\'s" }')["A"] == \
        {"p": "C:\\out\\", "q": 'say "hi"', "r": "it's"}
