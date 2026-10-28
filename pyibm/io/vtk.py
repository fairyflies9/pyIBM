"""Legacy ASCII VTK output for grid fields and Lagrangian points.

We write the legacy text format ("# vtk DataFile Version 3.0") because ParaView, VisIt and the
VTK library itself all read it, and because it needs nothing beyond NumPy.

Grid fields follow the package convention (docs/numerics.md, conventions): shape (nx, ny),
indexed [i, j] with i along x. A legacy STRUCTURED_POINTS dataset stores its point values with
x varying fastest, then y, then z, so the writers flatten with ``field.ravel(order="F")`` and
nothing else in the code has to know about it. The z dimension has one point and spacing 1.

Values are written with 17 significant digits, which reproduces every float64 exactly, so a
file written here and read back with ``read_structured_scalar`` is bit-identical to the array.

File names used by the driver (docs/user-guide.md, output):
``<output_dir>/<field>.<step:05d>.vtk`` for grid fields and ``<output_dir>/lag.<step:05d>.vtk``
for the Lagrangian points; see ``vtk_filename``. ``read_lagrangian_points`` reads the
positions back from a ``lag`` file and ``read_lagrangian_file`` also the links and the point
data.

For the MAC back end the driver writes cell-centred fields with origin ``(dx/2, dy/2)``; use
``mac_to_centres`` to average the face velocities to the centres first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

VTK_VERSION_LINE = "# vtk DataFile Version 3.0"
VTK_VERTEX = 1  # cell type ids from the VTK legacy file format specification
VTK_LINE = 3
_FMT = "%.17g"  # 17 significant digits round-trip float64 exactly


# ---------------------------------------------------------------------------- helpers
def vtk_filename(output_dir: str | os.PathLike, field: str, step: int) -> str:
    """Path of one dump: ``<output_dir>/<field>.<step:05d>.vtk`` (use ``field="lag"`` for the points)."""
    return os.path.join(os.fspath(output_dir), f"{field}.{int(step):05d}.vtk")


def _check_name(name: str) -> str:
    """VTK attribute names are single ASCII tokens; anything else breaks the readers."""
    if (not isinstance(name, str) or not name or not name.isascii()
            or any(c.isspace() for c in name)):
        raise ValueError(f"VTK attribute names must be non-empty ASCII without whitespace, got {name!r}")
    return name


def _triple(values, pad: float, what: str) -> tuple[float, float, float]:
    """Accept (x, y) or (x, y, z) and return (x, y, z), filling z with ``pad``."""
    v = [float(c) for c in np.asarray(values, dtype=float).ravel()]
    if len(v) == 2:
        v.append(pad)
    if len(v) != 3:
        raise ValueError(f"{what} must have 2 or 3 components, got {values!r}")
    return v[0], v[1], v[2]


def _open_for_writing(path):
    """Open ``path`` for ASCII text with LF line endings, creating the parent directory."""
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return open(path, "w", encoding="ascii", newline="\n")


def _write_structured_header(fh, title: str, shape: tuple[int, int], origin, spacing) -> None:
    """Header of a STRUCTURED_POINTS file up to and including the POINT_DATA line."""
    nx, ny = shape
    x0, y0, z0 = _triple(origin, 0.0, "origin")
    hx, hy, hz = _triple(spacing, 1.0, "spacing")
    if hx <= 0 or hy <= 0 or hz <= 0:
        raise ValueError(f"spacing must be positive, got {spacing!r}")
    fh.write(f"{VTK_VERSION_LINE}\n{title}\nASCII\nDATASET STRUCTURED_POINTS\n")
    fh.write(f"DIMENSIONS {nx} {ny} 1\n")
    fh.write("ORIGIN %s %s %s\n" % (_FMT % x0, _FMT % y0, _FMT % z0))
    fh.write("SPACING %s %s %s\n" % (_FMT % hx, _FMT % hy, _FMT % hz))
    fh.write(f"POINT_DATA {nx * ny}\n")


# ---------------------------------------------------------------------------- writers
def write_structured_scalar(path, name: str, field: np.ndarray, origin, spacing) -> str:
    """Write a scalar grid field as a legacy ASCII STRUCTURED_POINTS file.

    Parameters
    ----------
    path : output file; the parent directory is created if needed.
    name : attribute name shown in ParaView (one ASCII token, e.g. ``"vorticity"``).
    field : array of shape (nx, ny) indexed [i, j], value at ``(x0 + i*hx, y0 + j*hy)``.
    origin : ``(x0, y0)`` (a third component is accepted and taken as z0, default 0).
    spacing : ``(hx, hy)`` (third component hz, default 1).

    The file has ``DIMENSIONS nx ny 1``, ``ORIGIN x0 y0 0``, ``SPACING hx hy 1``, then
    ``POINT_DATA nx*ny`` / ``SCALARS name double 1`` / ``LOOKUP_TABLE default`` and one value per
    line, x fastest (``field.ravel(order="F")``), which is what the legacy format expects.
    Returns the path written.
    """
    name = _check_name(name)
    field = np.asarray(field, dtype=float)
    if field.ndim != 2:
        raise ValueError(f"field must be 2-D (nx, ny), got shape {field.shape}")
    with _open_for_writing(path) as fh:
        _write_structured_header(fh, f"pyIBM {name}", field.shape, origin, spacing)
        fh.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
        np.savetxt(fh, field.ravel(order="F"), fmt=_FMT)
    return os.fspath(path)


def write_structured_vector(path, name: str, fx: np.ndarray, fy: np.ndarray, origin, spacing) -> str:
    """Write a vector grid field (fx, fy) as a legacy ASCII STRUCTURED_POINTS file.

    Same header as ``write_structured_scalar``; the attribute block is ``VECTORS name double``
    followed by one ``fx fy 0`` triple per point, x fastest. ``fx`` and ``fy`` must have the same
    (nx, ny) shape and live at the same points (average MAC face values to the centres first,
    see ``mac_to_centres``). Returns the path written.
    """
    name = _check_name(name)
    fx = np.asarray(fx, dtype=float)
    fy = np.asarray(fy, dtype=float)
    if fx.ndim != 2 or fx.shape != fy.shape:
        raise ValueError(f"fx and fy must be 2-D arrays of the same shape, got {fx.shape} and {fy.shape}")
    xyz = np.zeros((fx.size, 3))
    xyz[:, 0] = fx.ravel(order="F")
    xyz[:, 1] = fy.ravel(order="F")
    with _open_for_writing(path) as fh:
        _write_structured_header(fh, f"pyIBM {name}", fx.shape, origin, spacing)
        fh.write(f"VECTORS {name} double\n")
        np.savetxt(fh, xyz, fmt=_FMT)
    return os.fspath(path)


def write_lagrangian_points(path, X: np.ndarray, connectivity=None, scalars=None) -> str:
    """Write Lagrangian points (and optional links) as a legacy ASCII UNSTRUCTURED_GRID file.

    Parameters
    ----------
    path : output file; the parent directory is created if needed.
    X : positions, shape (N, 2) in metres; written as ``POINTS N double`` with z = 0.
    connectivity : optional integer pairs (M, 2) of 0-based point indices, e.g. the ``i, j`` of
        the springs; each pair becomes a LINE cell (type 3) so ParaView draws the fibre.
    scalars : optional ``{name: array}`` of point data. A (N,) array is written as
        ``SCALARS name double 1``; a (N, 2) array (a force or velocity per point) as
        ``VECTORS name double`` with z = 0. Names must be single ASCII tokens.

    Every point also gets a VERTEX cell (type 1) so the points render even without links. The
    ``CELLS`` line counts ``N + M`` cells and ``2N + 3M`` integers (each vertex cell is
    ``1 i``, each line cell ``2 i j``). Returns the path written.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (N, 2), got {X.shape}")
    n = X.shape[0]

    if connectivity is None:
        conn = np.zeros((0, 2), dtype=int)
    else:
        conn = np.asarray(connectivity)
        if conn.size == 0:
            conn = np.zeros((0, 2), dtype=int)
        if conn.ndim != 2 or conn.shape[1] != 2 or not np.issubdtype(conn.dtype, np.integer):
            raise ValueError(f"connectivity must be integer pairs of shape (M, 2), got {conn.shape} {conn.dtype}")
        if conn.size and ((conn < 0).any() or (conn >= n).any()):
            raise ValueError("connectivity indices must lie in 0 .. N-1")
    m = conn.shape[0]

    data: list[tuple[str, np.ndarray]] = []
    for key, val in dict(scalars or {}).items():
        arr = np.asarray(val, dtype=float)
        if arr.shape != (n,) and arr.shape != (n, 2):
            raise ValueError(f"point data {key!r} must have shape ({n},) or ({n}, 2), got {arr.shape}")
        data.append((_check_name(key), arr))

    with _open_for_writing(path) as fh:
        fh.write(f"{VTK_VERSION_LINE}\npyIBM Lagrangian points\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {n} double\n")
        xyz = np.zeros((n, 3))
        xyz[:, :2] = X
        np.savetxt(fh, xyz, fmt=_FMT)
        fh.write(f"CELLS {n + m} {2 * n + 3 * m}\n")
        np.savetxt(fh, np.column_stack([np.ones(n, dtype=int), np.arange(n)]), fmt="%d")
        np.savetxt(fh, np.column_stack([np.full(m, 2, dtype=int), conn.astype(int)]), fmt="%d")
        fh.write(f"CELL_TYPES {n + m}\n")
        np.savetxt(fh, np.concatenate([np.full(n, VTK_VERTEX), np.full(m, VTK_LINE)]), fmt="%d")
        if data:
            fh.write(f"POINT_DATA {n}\n")
            for key, arr in data:
                if arr.ndim == 1:
                    fh.write(f"SCALARS {key} double 1\nLOOKUP_TABLE default\n")
                    np.savetxt(fh, arr, fmt=_FMT)
                else:
                    vec = np.zeros((n, 3))
                    vec[:, :2] = arr
                    fh.write(f"VECTORS {key} double\n")
                    np.savetxt(fh, vec, fmt=_FMT)
    return os.fspath(path)


def mac_to_centres(u: np.ndarray, v: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    """Average MAC face velocities to the cell centres for output.

    ``u`` lives on the vertical faces x = i*dx (shape ``grid.shape_u()``), ``v`` on the
    horizontal faces y = j*dy (shape ``grid.shape_v()``). The centre ((i+1/2) dx, (j+1/2) dy)
    is the midpoint of faces i and i+1, so

        uc[i, j] = (u[i, j] + u[i+1, j]) / 2,    vc[i, j] = (v[i, j] + v[i, j+1]) / 2,

    the usual second-order face-to-centre average. In a periodic direction face Nx is face 0
    (``np.roll``). Returns ``(uc, vc)`` of shape (Nx, Ny) at ``grid.xc, grid.yc``; write them
    with origin ``(dx/2, dy/2)``.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != grid.shape_u():
        raise ValueError(f"u must have shape {grid.shape_u()}, got {u.shape}")
    if v.shape != grid.shape_v():
        raise ValueError(f"v must have shape {grid.shape_v()}, got {v.shape}")
    if grid.periodic_x:
        uc = 0.5 * (u + np.roll(u, -1, axis=0))
    else:
        uc = 0.5 * (u[:-1, :] + u[1:, :])
    if grid.periodic_y:
        vc = 0.5 * (v + np.roll(v, -1, axis=1))
    else:
        vc = 0.5 * (v[:, :-1] + v[:, 1:])
    return uc, vc


# ---------------------------------------------------------------------------- reader
@dataclass(frozen=True)
class StructuredScalar:
    """What ``read_structured_scalar`` returns: the field as (nx, ny) [i, j] and its geometry."""

    name: str
    field: np.ndarray
    origin: tuple[float, float]
    spacing: tuple[float, float]


@dataclass(frozen=True)
class StructuredVector:
    """What ``read_structured_vector`` returns."""

    name: str
    fx: np.ndarray
    fy: np.ndarray
    origin: tuple[float, float]
    spacing: tuple[float, float]


def _read_structured(path):
    """Parse a 2-D STRUCTURED_POINTS file with one point attribute (SCALARS or VECTORS).

    Only what our own writers produce is supported; this exists for the tests and for quick
    checks, not as a general VTK reader. Returns (kind, name, dims, origin, spacing, values)
    with ``values`` the flat x-fastest data.
    """
    path = os.fspath(path)
    with open(path, "r", encoding="ascii") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 5 or not lines[0].startswith("# vtk DataFile Version"):
        raise ValueError(f"{path}: not a legacy VTK file")
    if lines[2].strip() != "ASCII":
        raise ValueError(f"{path}: only ASCII files are read")
    if lines[3].split() != ["DATASET", "STRUCTURED_POINTS"]:
        raise ValueError(f"{path}: not a STRUCTURED_POINTS dataset")

    dims = origin = spacing = npts = None
    k = 4
    while k < len(lines):
        tok = lines[k].split()
        k += 1
        if not tok:
            continue
        if tok[0] == "DIMENSIONS":
            dims = tuple(int(t) for t in tok[1:4])
        elif tok[0] == "ORIGIN":
            origin = tuple(float(t) for t in tok[1:4])
        elif tok[0] == "SPACING":
            spacing = tuple(float(t) for t in tok[1:4])
        elif tok[0] == "POINT_DATA":
            npts = int(tok[1])
            break
        else:
            raise ValueError(f"{path}: unexpected header line {lines[k - 1]!r}")
    if dims is None or origin is None or spacing is None or npts is None:
        raise ValueError(f"{path}: incomplete STRUCTURED_POINTS header")
    if len(dims) != 3 or dims[2] != 1:
        raise ValueError(f"{path}: only 2-D files (nz = 1) are read, got DIMENSIONS {dims}")
    if npts != dims[0] * dims[1]:
        raise ValueError(f"{path}: POINT_DATA {npts} does not match DIMENSIONS {dims}")

    while k < len(lines) and not lines[k].split():
        k += 1
    if k >= len(lines):
        raise ValueError(f"{path}: no point attribute found")
    tok = lines[k].split()
    k += 1
    if tok[0] == "SCALARS":
        kind, name = "scalar", tok[1]
        ncomp = int(tok[3]) if len(tok) > 3 else 1
        if ncomp != 1:
            raise ValueError(f"{path}: only one-component scalars are read")
        if k < len(lines) and lines[k].split()[:1] == ["LOOKUP_TABLE"]:
            k += 1
    elif tok[0] == "VECTORS":
        kind, name, ncomp = "vector", tok[1], 3
    else:
        raise ValueError(f"{path}: expected SCALARS or VECTORS, got {lines[k - 1]!r}")

    tokens = " ".join(lines[k:]).split()
    if len(tokens) < npts * ncomp:
        raise ValueError(f"{path}: expected {npts * ncomp} values, found {len(tokens)}")
    values = np.array(tokens[: npts * ncomp], dtype=float)
    return kind, name, dims, origin, spacing, values


def read_structured_scalar(path) -> StructuredScalar:
    """Read a file written by ``write_structured_scalar`` back into an (nx, ny) [i, j] array.

    The flat x-fastest data is reshaped with ``order="F"``, the inverse of the writer.
    ``origin`` and ``spacing`` come back as (x, y) pairs (z is dropped).
    """
    kind, name, dims, origin, spacing, values = _read_structured(path)
    if kind != "scalar":
        raise ValueError(f"{os.fspath(path)}: contains VECTORS, use read_structured_vector")
    field = values.reshape((dims[0], dims[1]), order="F")
    return StructuredScalar(name, field, (origin[0], origin[1]), (spacing[0], spacing[1]))


def read_structured_vector(path) -> StructuredVector:
    """Read a file written by ``write_structured_vector`` back into (fx, fy) arrays."""
    kind, name, dims, origin, spacing, values = _read_structured(path)
    if kind != "vector":
        raise ValueError(f"{os.fspath(path)}: contains SCALARS, use read_structured_scalar")
    xyz = values.reshape((dims[0] * dims[1], 3))
    fx = xyz[:, 0].reshape((dims[0], dims[1]), order="F")
    fy = xyz[:, 1].reshape((dims[0], dims[1]), order="F")
    return StructuredVector(name, fx, fy, (origin[0], origin[1]), (spacing[0], spacing[1]))


@dataclass(frozen=True)
class LagrangianPoints:
    """What ``read_lagrangian_file`` returns.

    X : positions (N, 2) (the z column is dropped).
    connectivity : the LINE cells as integer pairs (M, 2), shape (0, 2) when there are none.
    point_data : ``{name: array}``, (N,) for SCALARS and (N, 2) for VECTORS (z dropped), in
        file order; empty when the file has no POINT_DATA block.
    """

    X: np.ndarray
    connectivity: np.ndarray
    point_data: dict


def read_lagrangian_file(path) -> LagrangianPoints:
    """Read a file written by ``write_lagrangian_points`` (the driver's ``lag.NNNNN.vtk``).

    Only what our writer produces is supported: an ASCII UNSTRUCTURED_GRID with ``POINTS``,
    ``CELLS`` of VERTEX (type 1) and LINE (type 3), ``CELL_TYPES``, and an optional
    ``POINT_DATA`` block of one-component ``SCALARS`` and ``VECTORS``. Anything else raises
    ValueError naming the file. Values are read as written (17 digits), so the positions
    come back bit-identical.
    """
    path = os.fspath(path)
    with open(path, "r", encoding="ascii") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 5 or not lines[0].startswith("# vtk DataFile Version"):
        raise ValueError(f"{path}: not a legacy VTK file")
    if lines[2].strip() != "ASCII":
        raise ValueError(f"{path}: only ASCII files are read")
    if lines[3].split() != ["DATASET", "UNSTRUCTURED_GRID"]:
        raise ValueError(f"{path}: not an UNSTRUCTURED_GRID dataset")
    tokens = " ".join(lines[4:]).split()
    pos = 0

    def take(count: int, what: str) -> list[str]:
        nonlocal pos
        if pos + count > len(tokens):
            raise ValueError(f"{path}: the file ends inside the {what} block")
        out = tokens[pos:pos + count]
        pos += count
        return out

    X = cells = types = None
    n_points = None
    point_data: dict[str, np.ndarray] = {}
    while pos < len(tokens):
        key = take(1, "header")[0]
        if key == "POINTS":
            n_points = int(take(2, "POINTS")[0])
            X = np.array(take(3 * n_points, "POINTS"), dtype=float).reshape(n_points, 3)[:, :2]
        elif key == "CELLS":
            n_cells, size = (int(v) for v in take(2, "CELLS"))
            cells = (n_cells, np.array(take(size, "CELLS"), dtype=np.int64))
        elif key == "CELL_TYPES":
            n_types = int(take(1, "CELL_TYPES")[0])
            types = np.array(take(n_types, "CELL_TYPES"), dtype=np.int64)
        elif key == "POINT_DATA":
            if n_points is None or int(take(1, "POINT_DATA")[0]) != n_points:
                raise ValueError(f"{path}: POINT_DATA does not match POINTS")
        elif key in ("SCALARS", "VECTORS") and n_points is None:
            raise ValueError(f"{path}: {key} before POINTS")
        elif key == "SCALARS":
            name = take(2, "SCALARS")[0]                 # the name, then the data type
            if pos < len(tokens) and tokens[pos].isdigit():
                if int(take(1, "SCALARS")[0]) != 1:
                    raise ValueError(f"{path}: only one-component SCALARS are read")
            if pos < len(tokens) and tokens[pos] == "LOOKUP_TABLE":
                take(2, "LOOKUP_TABLE")
            point_data[name] = np.array(take(n_points, f"SCALARS {name}"), dtype=float)
        elif key == "VECTORS":
            name = take(2, "VECTORS")[0]
            vec = np.array(take(3 * n_points, f"VECTORS {name}"), dtype=float)
            point_data[name] = vec.reshape(n_points, 3)[:, :2]
        else:
            raise ValueError(f"{path}: unexpected keyword {key!r}")
    if X is None:
        raise ValueError(f"{path}: no POINTS block")

    pairs = np.zeros((0, 2), dtype=np.int64)
    if cells is not None:
        n_cells, flat = cells
        if types is None or len(types) != n_cells:
            raise ValueError(f"{path}: CELL_TYPES does not match CELLS")
        lines_found, k = [], 0
        for c in range(n_cells):
            if k >= len(flat):
                raise ValueError(f"{path}: CELLS holds fewer than {n_cells} cells")
            m = int(flat[k])
            if types[c] == VTK_LINE:
                if m != 2:
                    raise ValueError(f"{path}: a LINE cell with {m} points")
                lines_found.append(flat[k + 1:k + 3])
            elif types[c] != VTK_VERTEX:
                raise ValueError(f"{path}: cell type {int(types[c])} is not read")
            k += m + 1
        if lines_found:
            pairs = np.array(lines_found, dtype=np.int64).reshape(-1, 2)
    return LagrangianPoints(X, pairs, point_data)


def read_lagrangian_points(path) -> np.ndarray:
    """Positions (N, 2) from a ``lag.NNNNN.vtk`` file; ``read_lagrangian_file`` also returns
    the links and the point data (such as the driver's ``force``)."""
    return read_lagrangian_file(path).X
