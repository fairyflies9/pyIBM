"""Tests for pyibm.io: legacy VTK writers, the CSV history and the matplotlib snapshot.

The VTK checks compare the files line by line with the legacy format (version line, title,
ASCII, DATASET, DIMENSIONS/ORIGIN/SPACING, POINT_DATA, attribute header, value count) and
verify the x-fastest ordering by writing a field equal to the x coordinate.
"""

import os
import sys

import numpy as np
import pytest

from pyibm.grid import Grid
from pyibm.io import vtk
from pyibm.io.history import HistoryWriter, read_history

PYPLOT_LOADED_AT_START = "matplotlib.pyplot" in sys.modules


def _lines(path):
    with open(path, "r", encoding="ascii") as fh:
        return fh.read().splitlines()


def _index(lines, prefix):
    """Index of the first line starting with ``prefix``."""
    for k, line in enumerate(lines):
        if line.startswith(prefix):
            return k
    raise AssertionError(f"no line starting with {prefix!r}")


# ------------------------------------------------------------------ VTK structured points
def test_scalar_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    field = rng.standard_normal((12, 10)) * 1e3 + 1 / 3
    path = tmp_path / "sub" / "omega.00010.vtk"          # parent directory is created
    out = vtk.write_structured_scalar(path, "vorticity", field, (0.0, 0.0), (0.1, 0.2))
    assert os.path.exists(out)
    back = vtk.read_structured_scalar(path)
    assert back.name == "vorticity"
    assert back.field.shape == (12, 10)
    assert np.array_equal(back.field, field)                # 17 digits: bit-exact
    assert back.origin == (0.0, 0.0)
    assert back.spacing == (0.1, 0.2)


def test_scalar_file_structure(tmp_path):
    nx, ny = 6, 4
    field = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    path = tmp_path / "p.00000.vtk"
    vtk.write_structured_scalar(path, "pressure", field, (0.5, 0.25), (0.1, 0.2))
    lines = _lines(path)
    assert lines[0] == "# vtk DataFile Version 3.0"
    assert len(lines[1]) <= 256 and lines[1]                # title line
    assert lines[2] == "ASCII"
    assert lines[3] == "DATASET STRUCTURED_POINTS"
    assert lines[4].split() == ["DIMENSIONS", "6", "4", "1"]
    tok = lines[5].split()
    assert tok[0] == "ORIGIN" and [float(t) for t in tok[1:]] == [0.5, 0.25, 0.0]
    tok = lines[6].split()
    assert tok[0] == "SPACING" and [float(t) for t in tok[1:]] == [0.1, 0.2, 1.0]
    assert lines[7] == "POINT_DATA 24"
    assert lines[8] == "SCALARS pressure double 1"
    assert lines[9] == "LOOKUP_TABLE default"
    data = lines[10:]
    assert len(data) == 24 and all(len(d.split()) == 1 for d in data)
    with open(path, "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw                                 # LF line endings, ASCII
    raw.decode("ascii")


def test_scalar_x_fastest(tmp_path):
    """Write a field equal to x: the first nx values must run through x0 + i*hx, then repeat."""
    g = Grid(1.2, 0.8, 12, 8)
    Xm, Ym = g.meshgrid("nodes")
    path = tmp_path / "x.vtk"
    vtk.write_structured_scalar(path, "x", Xm, (0.0, 0.0), (g.dx, g.dy))
    lines = _lines(path)
    k = _index(lines, "LOOKUP_TABLE") + 1
    values = np.array(lines[k:], dtype=float)
    assert values.size == 12 * 8
    assert values[:12] == pytest.approx(g.x)               # x varies fastest
    assert values[12] == pytest.approx(0.0)                 # second row starts again at x = 0
    assert values[12:24] == pytest.approx(g.x)
    # the same file with y: constant over each block of nx values
    vtk.write_structured_scalar(path, "y", Ym, (0.0, 0.0), (g.dx, g.dy))
    lines = _lines(path)
    k = _index(lines, "LOOKUP_TABLE") + 1
    values = np.array(lines[k:], dtype=float).reshape(8, 12)   # (row j, column i) as stored
    assert np.allclose(values, g.y[:, None])
    # and the ravel order the writer uses is the inverse of the reader's reshape
    back = vtk.read_structured_scalar(path)
    assert np.array_equal(back.field, Ym)


def test_vector_file_and_round_trip(tmp_path):
    g = Grid(1.0, 0.5, 10, 8, periodic_x=False, periodic_y=False)
    Xm, Ym = g.meshgrid("centres")
    fx, fy = Xm.copy(), -Ym.copy()
    path = tmp_path / "vel.00003.vtk"
    vtk.write_structured_vector(path, "velocity", fx, fy, (g.dx / 2, g.dy / 2), (g.dx, g.dy))
    lines = _lines(path)
    assert lines[3] == "DATASET STRUCTURED_POINTS"
    assert lines[4].split() == ["DIMENSIONS", "10", "8", "1"]
    assert lines[7] == "POINT_DATA 80"
    assert lines[8] == "VECTORS velocity double"
    data = [d.split() for d in lines[9:]]
    assert len(data) == 80 and all(len(d) == 3 for d in data)
    triples = np.array(data, dtype=float)
    assert np.all(triples[:, 2] == 0.0)
    assert triples[:10, 0] == pytest.approx(g.xc)           # x fastest in the vector block too
    assert triples[:10, 1] == pytest.approx(-g.yc[0])
    back = vtk.read_structured_vector(path)
    assert back.name == "velocity"
    assert np.array_equal(back.fx, fx) and np.array_equal(back.fy, fy)
    assert back.origin == (g.dx / 2, g.dy / 2) and back.spacing == (g.dx, g.dy)
    with pytest.raises(ValueError):
        vtk.read_structured_scalar(path)                    # wrong reader for the file


def test_structured_validation(tmp_path):
    with pytest.raises(ValueError):
        vtk.write_structured_scalar(tmp_path / "a.vtk", "bad name", np.zeros((8, 8)), (0, 0), (1, 1))
    with pytest.raises(ValueError):
        vtk.write_structured_scalar(tmp_path / "a.vtk", "f", np.zeros(8), (0, 0), (1, 1))
    with pytest.raises(ValueError):
        vtk.write_structured_scalar(tmp_path / "a.vtk", "f", np.zeros((8, 8)), (0, 0), (0, 1))
    with pytest.raises(ValueError):
        vtk.write_structured_vector(tmp_path / "a.vtk", "f", np.zeros((8, 8)), np.zeros((8, 4)), (0, 0), (1, 1))


# ------------------------------------------------------------------ VTK Lagrangian points
def test_lagrangian_file_structure(tmp_path):
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    conn = np.array([[0, 1], [1, 2], [3, 4]])
    k = np.arange(5, dtype=float) * 10.0
    f = np.column_stack([np.arange(5.0), -np.arange(5.0)])
    path = tmp_path / "lag.00007.vtk"
    vtk.write_lagrangian_points(path, X, connectivity=conn, scalars={"stiffness": k, "force": f})
    lines = _lines(path)
    assert lines[0] == "# vtk DataFile Version 3.0"
    assert lines[2] == "ASCII"
    assert lines[3] == "DATASET UNSTRUCTURED_GRID"
    assert lines[4] == "POINTS 5 double"
    pts = np.array([l.split() for l in lines[5:10]], dtype=float)
    assert pts.shape == (5, 3) and np.array_equal(pts[:, :2], X) and np.all(pts[:, 2] == 0)
    assert lines[10] == "CELLS 8 19"                        # 5 vertices (2 ints) + 3 lines (3 ints)
    assert [l.split() for l in lines[11:16]] == [["1", str(i)] for i in range(5)]
    assert [l.split() for l in lines[16:19]] == [["2", "0", "1"], ["2", "1", "2"], ["2", "3", "4"]]
    assert lines[19] == "CELL_TYPES 8"
    assert lines[20:28] == ["1"] * 5 + ["3"] * 3           # VERTEX = 1, LINE = 3
    assert lines[28] == "POINT_DATA 5"
    assert lines[29] == "SCALARS stiffness double 1"
    assert lines[30] == "LOOKUP_TABLE default"
    assert np.array(lines[31:36], dtype=float) == pytest.approx(k)
    assert lines[36] == "VECTORS force double"
    vec = np.array([l.split() for l in lines[37:42]], dtype=float)
    assert np.array_equal(vec[:, :2], f) and np.all(vec[:, 2] == 0)
    assert len(lines) == 42


def test_lagrangian_minimal_and_validation(tmp_path):
    X = np.zeros((3, 2))
    path = tmp_path / "lag.vtk"
    vtk.write_lagrangian_points(path, X)
    lines = _lines(path)
    assert lines[4] == "POINTS 3 double"
    assert lines[8] == "CELLS 3 6"
    assert lines[12] == "CELL_TYPES 3"
    assert lines[13:16] == ["1", "1", "1"]
    assert len(lines) == 16                                 # no POINT_DATA block without scalars
    with pytest.raises(ValueError):
        vtk.write_lagrangian_points(path, X, connectivity=np.array([[0, 3]]))
    with pytest.raises(ValueError):
        vtk.write_lagrangian_points(path, X, connectivity=np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError):
        vtk.write_lagrangian_points(path, X, scalars={"k": np.zeros(4)})
    with pytest.raises(ValueError):
        vtk.write_lagrangian_points(path, np.zeros((3, 3)))


def test_lagrangian_round_trip(tmp_path):
    """read_lagrangian_file gives back what write_lagrangian_points wrote, bit for bit;
    read_lagrangian_points gives the positions alone."""
    rng = np.random.default_rng(5)
    X = rng.random((7, 2)) / 3.0
    conn = np.array([[0, 1], [1, 2], [5, 6]])
    k = rng.standard_normal(7) * 1e5
    f = rng.standard_normal((7, 2)) / 7.0
    path = tmp_path / "lag.00012.vtk"
    vtk.write_lagrangian_points(path, X, connectivity=conn, scalars={"stiffness": k, "force": f})
    back = vtk.read_lagrangian_file(path)
    assert np.array_equal(back.X, X)
    assert np.array_equal(back.connectivity, conn)
    assert list(back.point_data) == ["stiffness", "force"]
    assert np.array_equal(back.point_data["stiffness"], k)
    assert np.array_equal(back.point_data["force"], f)
    assert np.array_equal(vtk.read_lagrangian_points(path), X)
    # no links and no point data
    vtk.write_lagrangian_points(path, X)
    back = vtk.read_lagrangian_file(path)
    assert np.array_equal(back.X, X) and back.connectivity.shape == (0, 2)
    assert back.point_data == {}


def test_lagrangian_reader_refuses_other_files(tmp_path):
    vtk.write_structured_scalar(tmp_path / "p.vtk", "p", np.zeros((4, 4)), (0, 0), (1, 1))
    with pytest.raises(ValueError, match="UNSTRUCTURED_GRID"):
        vtk.read_lagrangian_file(tmp_path / "p.vtk")
    path = tmp_path / "lag.vtk"
    vtk.write_lagrangian_points(path, np.zeros((3, 2)), scalars={"k": np.ones(3)})
    lines = _lines(path)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")                  # one value short
    with pytest.raises(ValueError, match="ends inside"):
        vtk.read_lagrangian_file(path)


def test_vtk_filename():
    assert vtk.vtk_filename("viz", "omega", 7) == os.path.join("viz", "omega.00007.vtk")
    assert vtk.vtk_filename("viz", "lag", 12345) == os.path.join("viz", "lag.12345.vtk")


def test_mac_to_centres():
    # periodic: averaging cos(2 pi x/L) between faces gives cos(pi dx/L) cos(2 pi xc/L) exactly
    g = Grid.periodic_box(1.0, 1.0, 16, 16)
    Xu, _ = g.meshgrid("u")
    _, Yv = g.meshgrid("v")
    u = np.cos(2 * np.pi * Xu / g.Lx)
    v = np.sin(2 * np.pi * Yv / g.Ly)
    uc, vc = vtk.mac_to_centres(u, v, g)
    Xc, Yc = g.meshgrid("centres")
    assert uc.shape == (16, 16) and vc.shape == (16, 16)
    assert np.allclose(uc, np.cos(np.pi * g.dx / g.Lx) * np.cos(2 * np.pi * Xc / g.Lx))
    assert np.allclose(vc, np.cos(np.pi * g.dy / g.Ly) * np.sin(2 * np.pi * Yc / g.Ly))
    # walls: a linear profile is averaged exactly
    g = Grid(1.0, 0.5, 16, 8, periodic_x=False, periodic_y=False)
    Xu, _ = g.meshgrid("u")
    _, Yv = g.meshgrid("v")
    uc, vc = vtk.mac_to_centres(2.0 * Xu, 3.0 * Yv, g)
    Xc, Yc = g.meshgrid("centres")
    assert np.allclose(uc, 2.0 * Xc) and np.allclose(vc, 3.0 * Yc)
    with pytest.raises(ValueError):
        vtk.mac_to_centres(np.zeros((16, 8)), 3.0 * Yv, g)


# ------------------------------------------------------------------ history CSV
def test_history_round_trip_17_digits(tmp_path):
    path = tmp_path / "history.csv"
    cols = ["step", "t", "fx_body", "fy_body", "cfl"]
    rows = np.array([
        [0, 0.0, 0.1, 1 / 3, 0.5],
        [100, 100 * 1e-4, -2.0 / 7.0, 1e-300, 12345678901234567.0],
        [200, 0.02, np.pi * 1e20, -np.e * 1e-20, 0.7],
    ])
    with HistoryWriter(path, cols) as h:
        for r in rows:
            h.write(r)
    hist = read_history(path)
    assert isinstance(hist, np.recarray)
    assert hist.dtype.names == tuple(cols)
    assert len(hist) == 3
    for j, c in enumerate(cols):
        assert np.array_equal(hist[c], rows[:, j])          # exact, thanks to %.17g
    assert np.array_equal(hist.t, rows[:, 1])
    assert hist.step.astype(int).tolist() == [0, 100, 200]
    with open(path, "r", encoding="ascii") as fh:
        text = fh.read().splitlines()
    assert text[0] == "step,t,fx_body,fy_body,cfl"
    assert len(text) == 4
    assert "0.10000000000000001" in text[1]                 # 17 significant digits, not 0.1
    assert "0.33333333333333331" in text[1]


def test_history_flush_append_and_header_check(tmp_path):
    path = tmp_path / "h.csv"
    h = HistoryWriter(path, ("step", "t"))
    h.write([1, 0.5])
    assert len(read_history(path)) == 1                     # visible before close: flushed
    h.write({"t": 1.0, "step": 2})                          # mapping, any key order
    h.write(step=3, t=1.5)                                  # keywords
    h.close()
    h2 = HistoryWriter(path, ("step", "t"))                 # append, header not repeated
    h2.write([4, 2.0])
    h2.close()
    hist = read_history(path)
    assert hist.step.tolist() == [1, 2, 3, 4]
    assert hist.t.tolist() == [0.5, 1.0, 1.5, 2.0]
    with open(path, "r", encoding="ascii") as fh:
        assert fh.read().count("step,t") == 1
    with pytest.raises(ValueError):
        HistoryWriter(path, ("step", "t", "cfl"))           # different columns: refuse
    with pytest.raises(ValueError):
        HistoryWriter(tmp_path / "x.csv", ("a", "a"))
    with pytest.raises(ValueError):
        HistoryWriter(tmp_path / "x.csv", ("a,b",))
    with HistoryWriter(tmp_path / "y.csv", ("a", "b")) as h3:
        with pytest.raises(ValueError):
            h3.write([1.0])
        with pytest.raises(ValueError):
            h3.write({"a": 1.0, "c": 2.0})
        with pytest.raises(ValueError):
            h3.write([1.0, 2.0], a=1.0, b=2.0)


def test_history_empty_and_bad_rows(tmp_path):
    path = tmp_path / "e.csv"
    HistoryWriter(path, ("a", "b")).close()
    hist = read_history(path)
    assert hist.dtype.names == ("a", "b") and len(hist) == 0
    with open(path, "a", encoding="ascii", newline="\n") as fh:
        fh.write("1,2,3\n")
    with pytest.raises(ValueError):
        read_history(path)
    with pytest.raises(FileNotFoundError):
        read_history(tmp_path / "missing.csv")


# ------------------------------------------------------------------ snapshot
def test_snapshot_saves_png_headless(tmp_path):
    from matplotlib.figure import Figure
    from pyibm.io import plot

    if not PYPLOT_LOADED_AT_START:
        assert "matplotlib.pyplot" not in sys.modules       # we never go through pyplot
    g = Grid.periodic_box(1.0, 1.0, 16, 16)
    Xm, Ym = g.meshgrid("nodes")
    omega = np.sin(2 * np.pi * Xm) * np.cos(2 * np.pi * Ym)
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    X = np.column_stack([0.5 + 0.2 * np.cos(theta), 0.5 + 0.2 * np.sin(theta)])
    targets = X + 0.01
    path = tmp_path / "viz" / "snap.00000.png"
    fig = plot.snapshot(g, omega, X=X, targets=targets, path=path, title="t = 0")
    assert isinstance(fig, Figure)
    assert path.exists() and path.stat().st_size > 1000
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    ax = fig.axes[0]
    assert ax.get_title() == "t = 0"
    assert ax.get_xlim() == (0.0, 1.0) and ax.get_ylim() == (0.0, 1.0)
    if not PYPLOT_LOADED_AT_START:
        assert "matplotlib.pyplot" not in sys.modules


def test_snapshot_periodic_wrap_and_symmetric_limits():
    from pyibm.io import plot

    g = Grid.periodic_box(2.0, 1.0, 32, 16)
    rng = np.random.default_rng(3)
    omega = rng.standard_normal((32, 16))
    omega[5, 7] = 4.0                                       # the extreme sets the colour limit
    fig = plot.snapshot(g, omega)
    mesh = fig.axes[0].collections[0]
    assert mesh.get_clim() == (-4.0, 4.0)
    arr = np.asarray(mesh.get_array()).reshape(17, 33)      # rows along y, columns along x
    assert np.array_equal(arr[:16, :32], omega.T)           # transposed inside snapshot
    assert np.array_equal(arr[:, 32], arr[:, 0])            # wrapped column appended
    assert np.array_equal(arr[16, :], arr[0, :])            # wrapped row appended
    fig = plot.snapshot(g, omega, vmax=10.0)
    assert fig.axes[0].collections[0].get_clim() == (-10.0, 10.0)
    with pytest.raises(ValueError):
        plot.snapshot(g, omega.T)


def test_snapshot_walls_mixed_and_zero_field(tmp_path):
    from pyibm.io import plot

    g = Grid(1.0, 0.5, 16, 8, periodic_x=False, periodic_y=False)
    fig = plot.snapshot(g, np.zeros((16, 8)), path=tmp_path / "zero.png")
    mesh = fig.axes[0].collections[0]
    assert np.asarray(mesh.get_array()).size == 16 * 8      # nothing appended at walls
    assert mesh.get_clim() == (-1.0, 1.0)                   # all-zero field: fallback limits
    assert (tmp_path / "zero.png").exists()
    g = Grid(1.0, 0.5, 16, 8, periodic_x=True, periodic_y=False)
    fig = plot.snapshot(g, np.ones((16, 8)), location="centres")
    assert np.asarray(fig.axes[0].collections[0].get_array()).size == 17 * 8
    with pytest.raises(ValueError):
        plot.snapshot(g, np.ones((16, 8)), location="faces")
