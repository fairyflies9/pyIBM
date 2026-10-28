"""Lagrangian structures for pyIBM: data model, structure files, forces and the hooks loader.

A structure is a set of N Lagrangian points X (N, 2) with quadrature weights ds (N,),
connected by elements of six kinds:

- springs (Hookean or odd-power),
- damped springs (a spring plus a dashpot on the relative velocity),
- torsional beams (penalise the cross product of the two segments meeting at a point),
- non-invariant beams (penalise the second difference vector),
- targets (tethers to prescribed positions outside the fluid, optionally damped),
- masses (tethers to ghost points that carry mass and feel gravity).

Every elastic force is -dE/dX of an energy per unit fiber length and comes back as a force
density per unit length [N/m]; the coupling multiplies by ds when it spreads to the grid. All
stiffnesses are per unit length, and so is the mass m of a mass point, so k/m is an inverse
time squared whatever ds is. Separation vectors between points go through grid.min_image, so
elements that straddle a periodic boundary see the short image. The dashpots of the damped
springs and the target damping are the only terms that depend on velocity; they have no
energy.

The file formats are IB2d-like: whitespace-separated ASCII, the first line is the row count,
indices are 0-based. The column tables are in docs/user-guide.md and in the reader docstrings
below. Nothing here is copied from IB2d; we wrote the readers and writers from our own tables.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from pyibm.grid import Grid

__all__ = [
    "Springs", "DampedSprings", "Beams", "NonInvBeams", "Targets", "Masses", "Structure",
    "read_vertex", "read_spring", "read_d_spring", "read_beam", "read_noninv_beam",
    "read_target", "read_mass", "read_structure", "nearest_neighbour_ds",
    "write_vertex", "write_spring", "write_d_spring", "write_beam", "write_noninv_beam",
    "write_target", "write_mass", "write_structure",
    "spring_energy", "damped_spring_energy", "beam_energy", "noninv_beam_energy",
    "target_energy", "mass_energy", "structure_energy", "structure_energy_terms",
    "spring_forces", "damped_spring_forces", "beam_forces", "noninv_beam_forces",
    "target_forces", "mass_forces", "mass_tether_forces", "compute_forces", "advance_masses",
    "Hooks", "HOOK_NAMES", "DEFAULT_HOOKS_FILE", "load_hooks",
]


# ====================================================================== array coercion helpers

def _as_index_array(a, name: str) -> np.ndarray:
    """1-D integer index array (np.intp). Floats are accepted when they are whole numbers."""
    arr = np.asarray(a)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of indices, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.all(np.mod(arr, 1) == 0):
            raise ValueError(f"{name} must hold integers")
    return arr.astype(np.intp)


def _as_float_array(a, name: str, M: int) -> np.ndarray:
    """1-D float64 array of length M; a scalar is broadcast to M."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        return np.full(M, float(arr))
    if arr.shape != (M,):
        raise ValueError(f"{name} must have shape ({M},), got {arr.shape}")
    return np.array(arr, dtype=float, copy=True)


def _as_xy_array(a, name: str, M: int) -> np.ndarray:
    """(M, 2) float64 array (always a copy)."""
    arr = np.array(a, dtype=float, copy=True)
    if arr.shape != (M, 2):
        raise ValueError(f"{name} must have shape ({M}, 2), got {arr.shape}")
    return arr


def _check_indices(idx: np.ndarray, N: int, what: str, where: str) -> None:
    """Raise ValueError if any index is outside [0, N)."""
    if idx.size and (idx.min() < 0 or idx.max() >= N):
        raise ValueError(f"{where}: {what} index out of range: indices are 0-based and must lie "
                         f"in [0, {N}), found min {idx.min()} max {idx.max()}")


# ====================================================================== data model

@dataclass
class Springs:
    """Springs between points i and j.

    Energy per unit length E = k/(alpha+1) |L - rest|^(alpha+1) with L = |X_j - X_i|.
    alpha = 1 (default) is a Hookean spring, E = k/2 (L - rest)^2; alpha = 3 is the usual
    odd-power stiffening spring. We write the energy with |L - rest| so that it is well defined
    for every positive alpha; for odd alpha it is identical to (L - rest)^(alpha+1). The force
    k sgn(L - rest) |L - rest|^alpha is its derivative in all cases.
    """

    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    rest: np.ndarray
    alpha: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.i = _as_index_array(self.i, "i")
        self.j = _as_index_array(self.j, "j")
        M = len(self.i)
        if len(self.j) != M:
            raise ValueError("spring index arrays i and j must have the same length")
        self.k = _as_float_array(self.k, "k", M)
        self.rest = _as_float_array(self.rest, "rest", M)
        self.alpha = np.ones(M) if self.alpha is None else _as_float_array(self.alpha, "alpha", M)
        if np.any(self.i == self.j):
            raise ValueError("a spring connects a point to itself")
        if np.any(self.alpha <= 0):
            raise ValueError("spring alpha must be positive")

    def __len__(self) -> int:
        return len(self.i)


@dataclass
class DampedSprings:
    """Hookean springs (alpha = 1) with a dashpot of coefficient b on the relative velocity."""

    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    rest: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        self.i = _as_index_array(self.i, "i")
        self.j = _as_index_array(self.j, "j")
        M = len(self.i)
        if len(self.j) != M:
            raise ValueError("damped spring index arrays i and j must have the same length")
        self.k = _as_float_array(self.k, "k", M)
        self.rest = _as_float_array(self.rest, "rest", M)
        self.b = _as_float_array(self.b, "b", M)
        if np.any(self.i == self.j):
            raise ValueError("a damped spring connects a point to itself")

    def __len__(self) -> int:
        return len(self.i)


@dataclass
class Beams:
    """Torsional beams about q, between segments p->q and q->r.

    With A = X_r - X_q and B = X_q - X_p, c = A x B = ax*by - ay*bx and E = k/2 (c - curv)^2.
    curv is the preferred cross product C [m^2] (zero for a straight beam).

    Sign: c = |A||B| sin(theta_B - theta_A), where theta_A and theta_B are the directions of A
    and B. A left (counterclockwise) turn from p through q to r therefore gives c < 0, and a
    convex polygon numbered counterclockwise has curv < 0 at every beam.
    """

    p: np.ndarray
    q: np.ndarray
    r: np.ndarray
    k: np.ndarray
    curv: np.ndarray

    def __post_init__(self) -> None:
        self.p = _as_index_array(self.p, "p")
        self.q = _as_index_array(self.q, "q")
        self.r = _as_index_array(self.r, "r")
        M = len(self.p)
        if len(self.q) != M or len(self.r) != M:
            raise ValueError("beam index arrays p, q, r must have the same length")
        self.k = _as_float_array(self.k, "k", M)
        self.curv = _as_float_array(self.curv, "curv", M)
        if np.any((self.p == self.q) | (self.q == self.r) | (self.p == self.r)):
            raise ValueError("a beam must join three distinct points")

    def __len__(self) -> int:
        return len(self.p)


@dataclass
class NonInvBeams:
    """Non-invariant beams: E = k/2 |X_r - 2 X_q + X_p - (cx, cy)|^2.

    (cx, cy) is the preferred second-difference vector; it does not rotate with the body,
    hence the name.
    """

    p: np.ndarray
    q: np.ndarray
    r: np.ndarray
    k: np.ndarray
    cx: np.ndarray
    cy: np.ndarray

    def __post_init__(self) -> None:
        self.p = _as_index_array(self.p, "p")
        self.q = _as_index_array(self.q, "q")
        self.r = _as_index_array(self.r, "r")
        M = len(self.p)
        if len(self.q) != M or len(self.r) != M:
            raise ValueError("non-invariant beam index arrays p, q, r must have the same length")
        self.k = _as_float_array(self.k, "k", M)
        self.cx = _as_float_array(self.cx, "cx", M)
        self.cy = _as_float_array(self.cy, "cy", M)
        if np.any((self.p == self.q) | (self.q == self.r) | (self.p == self.r)):
            raise ValueError("a non-invariant beam must join three distinct points")

    def __len__(self) -> int:
        return len(self.p)


@dataclass
class Targets:
    """Target points: point idx is tethered to the prescribed position X with stiffness k and,
    when c > 0, damped towards the prescribed velocity V.

    The force per unit length on the point is k (X_target - X_i) + c (V_target - V_i), as in
    Miller & Peskin (2005), eq. 11. V_i is the point's own velocity, which compute_forces forms
    from the previous positions. k is per unit length [N/m^2] and c is a damping per unit
    length [N s/m^2]. c defaults to 0, the plain tether, and then V does not enter the force.
    V is zeros unless a hook sets it; with c > 0 a hook that moves the targets should also set
    V to their velocity, or the damping pulls the points towards rest.

    Miller, L. A. & Peskin, C. S. (2005). A computational fluid dynamics of 'clap and fling'
    in the smallest insects. J. Exp. Biol. 208, 195-212.
    """

    idx: np.ndarray
    k: np.ndarray
    X: np.ndarray
    V: np.ndarray | None = None
    c: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.idx = _as_index_array(self.idx, "idx")
        M = len(self.idx)
        self.k = _as_float_array(self.k, "k", M)
        self.X = _as_xy_array(self.X, "X", M)
        self.V = np.zeros((M, 2)) if self.V is None else _as_xy_array(self.V, "V", M)
        self.c = np.zeros(M) if self.c is None else _as_float_array(self.c, "c", M)
        if not np.all(np.isfinite(self.c) & (self.c >= 0.0)):
            raise ValueError("target damping c must be finite and >= 0")

    def __len__(self) -> int:
        return len(self.idx)


@dataclass
class Masses:
    """Mass points: point idx is tethered (stiffness k) to a ghost at Y with mass m (per unit
    length) and velocity V. The ghost is outside the fluid and obeys m dV/dt = -k (Y - X) + m g."""

    idx: np.ndarray
    k: np.ndarray
    m: np.ndarray
    Y: np.ndarray
    V: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.idx = _as_index_array(self.idx, "idx")
        M = len(self.idx)
        self.k = _as_float_array(self.k, "k", M)
        self.m = _as_float_array(self.m, "m", M)
        if np.any(self.m <= 0):
            raise ValueError("mass points need m > 0")
        self.Y = _as_xy_array(self.Y, "Y", M)
        self.V = np.zeros((M, 2)) if self.V is None else _as_xy_array(self.V, "V", M)

    def __len__(self) -> int:
        return len(self.idx)


@dataclass
class Structure:
    """One Lagrangian structure: positions, quadrature weights and the elements joining them.

    X0 holds the initial positions, so that hooks can prescribe motion relative to the initial
    shape. When it is not given it is a copy of X at construction (read_structure therefore
    gives the positions of the .vertex file). The force routines never use it.
    """

    name: str
    X: np.ndarray
    ds: np.ndarray
    springs: Springs | None = None
    damped_springs: DampedSprings | None = None
    beams: Beams | None = None
    noninv_beams: NonInvBeams | None = None
    targets: Targets | None = None
    masses: Masses | None = None
    gravity: tuple[float, float] = (0.0, 0.0)
    X0: np.ndarray | None = None

    def __post_init__(self) -> None:
        X = np.array(self.X, dtype=float, copy=True)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must have shape (N, 2), got {X.shape}")
        self.X = X
        N = len(X)
        self.ds = _as_float_array(self.ds, "ds", N)
        if np.any(self.ds <= 0):
            raise ValueError("ds must be positive")
        gx, gy = self.gravity
        self.gravity = (float(gx), float(gy))
        self.X0 = X.copy() if self.X0 is None else _as_xy_array(self.X0, "X0", N)
        where = f"structure {self.name!r}"
        if self.springs is not None:
            _check_indices(self.springs.i, N, "spring", where)
            _check_indices(self.springs.j, N, "spring", where)
        if self.damped_springs is not None:
            _check_indices(self.damped_springs.i, N, "damped spring", where)
            _check_indices(self.damped_springs.j, N, "damped spring", where)
        if self.beams is not None:
            for a in (self.beams.p, self.beams.q, self.beams.r):
                _check_indices(a, N, "beam", where)
        if self.noninv_beams is not None:
            for a in (self.noninv_beams.p, self.noninv_beams.q, self.noninv_beams.r):
                _check_indices(a, N, "non-invariant beam", where)
        if self.targets is not None:
            _check_indices(self.targets.idx, N, "target", where)
        if self.masses is not None:
            _check_indices(self.masses.idx, N, "mass", where)

    @property
    def n(self) -> int:
        """Number of Lagrangian points."""
        return len(self.X)


# ====================================================================== file readers

def _strip_comment(line: str) -> str:
    """Drop everything after a '#' or '%' (we allow comments although IB2d files have none)."""
    for ch in ("#", "%"):
        pos = line.find(ch)
        if pos >= 0:
            line = line[:pos]
    return line.strip()


def _read_table(path: str, ncols: int, n_optional: int = 0) -> np.ndarray:
    """Read a count-headed whitespace table into a float array of shape (count, ncols).

    The first non-blank line holds the row count and must equal the number of data rows;
    otherwise we raise ValueError naming the file. The last n_optional columns may be missing
    on any row; missing entries come back as NaN for the caller to default. The file is read
    as UTF-8 with an optional byte-order mark, which Windows editors and PowerShell often write.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = fh.read().splitlines()
    rows: list[tuple[int, list[str]]] = []
    for ln, line in enumerate(raw, start=1):
        s = _strip_comment(line)
        if s:
            rows.append((ln, s.split()))
    if not rows:
        raise ValueError(f"{path}: empty file; the first line must give the number of rows")
    ln0, head = rows[0]
    try:
        if len(head) != 1:
            raise ValueError
        count = int(float(head[0]))
        if count < 0 or float(head[0]) != count:
            raise ValueError
    except ValueError:
        raise ValueError(f"{path} line {ln0}: the first line must be the row count, got "
                         f"{' '.join(head)!r}") from None
    data = rows[1:]
    if len(data) != count:
        raise ValueError(f"{path}: header says {count} rows but the file holds {len(data)} data rows")
    out = np.full((count, ncols), np.nan)
    lo = ncols - n_optional
    for row, (ln, toks) in enumerate(data):
        if not (lo <= len(toks) <= ncols):
            want = f"{ncols}" if n_optional == 0 else f"{lo} to {ncols}"
            raise ValueError(f"{path} line {ln}: expected {want} columns, got {len(toks)}")
        try:
            out[row, :len(toks)] = [float(t) for t in toks]
        except ValueError:
            raise ValueError(f"{path} line {ln}: cannot parse {' '.join(toks)!r} as numbers") from None
    return out


def _int_column(col: np.ndarray, path: str, name: str) -> np.ndarray:
    if col.size and (not np.all(np.isfinite(col)) or not np.all(np.mod(col, 1) == 0)):
        raise ValueError(f"{path}: column {name!r} must hold integer indices")
    return col.astype(np.intp)


def read_vertex(path: str) -> np.ndarray:
    """Read a .vertex file (columns x y) into an (N, 2) array."""
    return _read_table(path, 2)


def read_spring(path: str, n_points: int | None = None) -> Springs:
    """Read a .spring file (columns i j k rest [alpha]); a missing alpha means 1."""
    t = _read_table(path, 5, n_optional=1)
    i = _int_column(t[:, 0], path, "i")
    j = _int_column(t[:, 1], path, "j")
    alpha = np.where(np.isnan(t[:, 4]), 1.0, t[:, 4])
    if n_points is not None:
        _check_indices(i, n_points, "spring", path)
        _check_indices(j, n_points, "spring", path)
    try:
        return Springs(i, j, t[:, 2], t[:, 3], alpha)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def read_d_spring(path: str, n_points: int | None = None) -> DampedSprings:
    """Read a .d_spring file (columns i j k rest b)."""
    t = _read_table(path, 5)
    i = _int_column(t[:, 0], path, "i")
    j = _int_column(t[:, 1], path, "j")
    if n_points is not None:
        _check_indices(i, n_points, "damped spring", path)
        _check_indices(j, n_points, "damped spring", path)
    try:
        return DampedSprings(i, j, t[:, 2], t[:, 3], t[:, 4])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def read_beam(path: str, n_points: int | None = None) -> Beams:
    """Read a .beam file (columns p q r k curv)."""
    t = _read_table(path, 5)
    p = _int_column(t[:, 0], path, "p")
    q = _int_column(t[:, 1], path, "q")
    r = _int_column(t[:, 2], path, "r")
    if n_points is not None:
        for a in (p, q, r):
            _check_indices(a, n_points, "beam", path)
    try:
        return Beams(p, q, r, t[:, 3], t[:, 4])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def read_noninv_beam(path: str, n_points: int | None = None) -> NonInvBeams:
    """Read a .nonInv_beam file (columns p q r k cx cy)."""
    t = _read_table(path, 6)
    p = _int_column(t[:, 0], path, "p")
    q = _int_column(t[:, 1], path, "q")
    r = _int_column(t[:, 2], path, "r")
    if n_points is not None:
        for a in (p, q, r):
            _check_indices(a, n_points, "non-invariant beam", path)
    try:
        return NonInvBeams(p, q, r, t[:, 3], t[:, 4], t[:, 5])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def read_target(path: str, X: np.ndarray) -> Targets:
    """Read a .target file (columns i k [c]); a missing damping c means 0.

    Target positions start at the vertex positions X[i] and target velocities at zero.
    """
    t = _read_table(path, 3, n_optional=1)
    idx = _int_column(t[:, 0], path, "i")
    _check_indices(idx, len(X), "target", path)
    c = np.where(np.isnan(t[:, 2]), 0.0, t[:, 2])
    try:
        return Targets(idx, t[:, 1], np.asarray(X, dtype=float)[idx], c=c)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def read_mass(path: str, X: np.ndarray) -> Masses:
    """Read a .mass file (columns i k m). Ghosts start at the vertex positions with zero velocity."""
    t = _read_table(path, 3)
    idx = _int_column(t[:, 0], path, "i")
    _check_indices(idx, len(X), "mass", path)
    try:
        return Masses(idx, t[:, 1], t[:, 2], np.asarray(X, dtype=float)[idx])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def nearest_neighbour_ds(X: np.ndarray, label: str = "vertices") -> float:
    """Median nearest-neighbour distance among the vertices.

    Used for ds when the input file gives none. The distance is plain Euclidean in the plane
    (the reader has no grid, so no periodic images). We warn if the largest nearest-neighbour
    distance exceeds twice the smallest, because one ds for every point is then a poor
    quadrature weight and the user should give ds or fix the spacing. Duplicate vertices make
    the rule meaningless, so a zero median is an error.
    """
    X = np.asarray(X, dtype=float)
    if len(X) < 2:
        raise ValueError(f"{label}: cannot infer ds from fewer than two vertices; give "
                         "Lag_Structure_Info { ds = ... } in the input file")
    dist, _ = cKDTree(X).query(X, k=2)
    nn = dist[:, 1]
    ds = float(np.median(nn))
    if not ds > 0:
        raise ValueError(f"{label}: the median nearest-neighbour distance is zero (duplicate "
                         "vertices); give ds explicitly")
    lo, hi = float(nn.min()), float(nn.max())
    ratio = hi / lo if lo > 0 else np.inf
    if ratio > 2.0:
        warnings.warn(f"{label}: nearest-neighbour distances range from {lo:.3g} to {hi:.3g} "
                      f"(ratio {ratio:.2f} > 2); using the median {ds:.3g} as ds for every point. "
                      "Consider giving ds explicitly.", stacklevel=2)
    return ds


def read_structure(case_dir: str, name: str, ds: float | None = None,
                   gravity: tuple[float, float] = (0.0, 0.0)) -> Structure:
    """Read <case_dir>/<name>.vertex and whichever element files exist beside it.

    ds: quadrature weight for every point; None means the median nearest-neighbour distance
    among the vertices (nearest_neighbour_ds). gravity acts on mass points only.
    """
    base = os.path.join(case_dir, name)
    vpath = base + ".vertex"
    if not os.path.isfile(vpath):
        raise FileNotFoundError(f"vertex file not found: {vpath}")
    X = read_vertex(vpath)
    N = len(X)
    if N == 0:
        raise ValueError(f"{vpath}: no vertices")
    if ds is None:
        ds_value = nearest_neighbour_ds(X, label=vpath)
    else:
        ds_value = float(ds)
        if not ds_value > 0:
            raise ValueError(f"ds must be positive, got {ds}")

    def optional(ext: str, reader: Callable, *args):
        p = base + ext
        return reader(p, *args) if os.path.isfile(p) else None

    return Structure(
        name=name,
        X=X,
        ds=np.full(N, ds_value),
        springs=optional(".spring", read_spring, N),
        damped_springs=optional(".d_spring", read_d_spring, N),
        beams=optional(".beam", read_beam, N),
        noninv_beams=optional(".nonInv_beam", read_noninv_beam, N),
        targets=optional(".target", read_target, X),
        masses=optional(".mass", read_mass, X),
        gravity=(float(gravity[0]), float(gravity[1])),
    )


# ====================================================================== file writers

def _write_table(path: str, columns: list[np.ndarray], is_int: list[bool]) -> None:
    """Write a count-headed table. Integers as %d, floats with 17 significant digits."""
    n = len(columns[0])
    fmt = ["%d" if b else "%.17g" for b in is_int]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{n}\n")
        if n:
            np.savetxt(fh, np.column_stack(columns), fmt=fmt, delimiter=" ")


def write_vertex(path: str, X: np.ndarray) -> None:
    """Write an (N, 2) array as a .vertex file."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (N, 2), got {X.shape}")
    _write_table(path, [X[:, 0], X[:, 1]], [False, False])


def write_spring(path: str, springs: Springs) -> None:
    """Write a .spring file. The alpha column is written only if some alpha differs from 1."""
    s = springs
    cols = [s.i, s.j, s.k, s.rest]
    ints = [True, True, False, False]
    if np.any(s.alpha != 1.0):
        cols.append(s.alpha)
        ints.append(False)
    _write_table(path, cols, ints)


def write_d_spring(path: str, damped: DampedSprings) -> None:
    """Write a .d_spring file (i j k rest b)."""
    d = damped
    _write_table(path, [d.i, d.j, d.k, d.rest, d.b], [True, True, False, False, False])


def write_beam(path: str, beams: Beams) -> None:
    """Write a .beam file (p q r k curv)."""
    b = beams
    _write_table(path, [b.p, b.q, b.r, b.k, b.curv], [True, True, True, False, False])


def write_noninv_beam(path: str, beams: NonInvBeams) -> None:
    """Write a .nonInv_beam file (p q r k cx cy)."""
    b = beams
    _write_table(path, [b.p, b.q, b.r, b.k, b.cx, b.cy], [True, True, True, False, False, False])


def write_target(path: str, targets: Targets) -> None:
    """Write a .target file (i k [c]). The damping column c is written only if some c differs
    from 0. Target positions and velocities are not stored; they restart at the vertices."""
    cols = [targets.idx, targets.k]
    ints = [True, False]
    if np.any(targets.c != 0.0):
        cols.append(targets.c)
        ints.append(False)
    _write_table(path, cols, ints)


def write_mass(path: str, masses: Masses) -> None:
    """Write a .mass file (i k m). Ghost positions and velocities are not stored."""
    _write_table(path, [masses.idx, masses.k, masses.m], [True, False, False])


def write_structure(case_dir: str, structure: Structure) -> list[str]:
    """Write <name>.vertex plus one file per element type present. Returns the paths written."""
    base = os.path.join(case_dir, structure.name)
    written = [base + ".vertex"]
    write_vertex(written[0], structure.X)
    for ext, el, writer in ((".spring", structure.springs, write_spring),
                            (".d_spring", structure.damped_springs, write_d_spring),
                            (".beam", structure.beams, write_beam),
                            (".nonInv_beam", structure.noninv_beams, write_noninv_beam),
                            (".target", structure.targets, write_target),
                            (".mass", structure.masses, write_mass)):
        if el is not None:
            writer(base + ext, el)
            written.append(base + ext)
    return written


# ====================================================================== geometry helpers

def _accumulate(f: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """f[idx] += vals with repeated indices summed (np.bincount per component)."""
    if idx.size == 0:
        return
    N = f.shape[0]
    f[:, 0] += np.bincount(idx, weights=vals[:, 0], minlength=N)
    f[:, 1] += np.bincount(idx, weights=vals[:, 1], minlength=N)


def _segments(X: np.ndarray, i: np.ndarray, j: np.ndarray, grid: Grid):
    """Minimum-image separations d = X_j - X_i, lengths L and unit vectors e (zero where L = 0)."""
    d = grid.min_image(X[j] - X[i])
    L = np.hypot(d[:, 0], d[:, 1])
    safe = np.where(L > 0.0, L, 1.0)
    e = d / safe[:, None]
    e[L == 0.0] = 0.0
    return d, L, e


def _beam_vectors(X: np.ndarray, beams, grid: Grid):
    """A = X_r - X_q and B = X_q - X_p with minimum image, as (M,) component arrays."""
    A = grid.min_image(X[beams.r] - X[beams.q])
    B = grid.min_image(X[beams.q] - X[beams.p])
    return A[:, 0], A[:, 1], B[:, 0], B[:, 1]


# ====================================================================== energies

def spring_energy(X: np.ndarray, springs: Springs, grid: Grid) -> float:
    """Spring energy per unit length, E = sum k/(alpha+1) |L - rest|^(alpha+1)."""
    _, L, _ = _segments(X, springs.i, springs.j, grid)
    a1 = springs.alpha + 1.0
    return float(np.sum(springs.k / a1 * np.abs(L - springs.rest) ** a1))


def damped_spring_energy(X: np.ndarray, damped: DampedSprings, grid: Grid) -> float:
    """Elastic energy of the damped springs, sum k/2 (L - rest)^2 (the dashpot has none)."""
    _, L, _ = _segments(X, damped.i, damped.j, grid)
    return float(np.sum(0.5 * damped.k * (L - damped.rest) ** 2))


def beam_curvature(X: np.ndarray, beams: Beams, grid: Grid) -> np.ndarray:
    """Cross product c = (X_r - X_q) x (X_q - X_p) per beam, the quantity curv is compared to."""
    ax, ay, bx, by = _beam_vectors(X, beams, grid)
    return ax * by - ay * bx


def beam_energy(X: np.ndarray, beams: Beams, grid: Grid) -> float:
    """Torsional beam energy per unit length, E = sum k/2 (c - curv)^2 with c the cross product."""
    c = beam_curvature(X, beams, grid)
    return float(np.sum(0.5 * beams.k * (c - beams.curv) ** 2))


def noninv_beam_difference(X: np.ndarray, beams: NonInvBeams, grid: Grid) -> np.ndarray:
    """D = X_r - 2 X_q + X_p - (cx, cy) per beam, built from minimum-image segments (M, 2)."""
    A = grid.min_image(X[beams.r] - X[beams.q])
    B = grid.min_image(X[beams.q] - X[beams.p])
    D = A - B
    D[:, 0] -= beams.cx
    D[:, 1] -= beams.cy
    return D


def noninv_beam_energy(X: np.ndarray, beams: NonInvBeams, grid: Grid) -> float:
    """Non-invariant beam energy per unit length, E = sum k/2 |D|^2."""
    D = noninv_beam_difference(X, beams, grid)
    return float(np.sum(0.5 * beams.k * np.sum(D * D, axis=1)))


def target_energy(X: np.ndarray, targets: Targets, grid: Grid) -> float:
    """Target tether energy per unit length, E = sum k/2 |X_target - X_i|^2 (minimum image).

    The damping term c (V_target - V_i) has no energy and is not included.
    """
    d = grid.min_image(targets.X - X[targets.idx])
    return float(np.sum(0.5 * targets.k * np.sum(d * d, axis=1)))


def mass_energy(X: np.ndarray, masses: Masses, grid: Grid) -> float:
    """Mass tether energy per unit length, E = sum k/2 |Y - X_i|^2 (minimum image).

    The ghosts' kinetic and gravitational energies are not included; this is the part whose
    gradient with respect to X is the force on the fluid points.
    """
    d = grid.min_image(masses.Y - X[masses.idx])
    return float(np.sum(0.5 * masses.k * np.sum(d * d, axis=1)))


def structure_energy_terms(structure: Structure, grid: Grid, X: np.ndarray | None = None
                           ) -> dict[str, float]:
    """Energy per unit length of each element type present, keyed by field name."""
    X = structure.X if X is None else np.asarray(X, dtype=float)
    terms: dict[str, float] = {}
    if structure.springs is not None:
        terms["springs"] = spring_energy(X, structure.springs, grid)
    if structure.damped_springs is not None:
        terms["damped_springs"] = damped_spring_energy(X, structure.damped_springs, grid)
    if structure.beams is not None:
        terms["beams"] = beam_energy(X, structure.beams, grid)
    if structure.noninv_beams is not None:
        terms["noninv_beams"] = noninv_beam_energy(X, structure.noninv_beams, grid)
    if structure.targets is not None:
        terms["targets"] = target_energy(X, structure.targets, grid)
    if structure.masses is not None:
        terms["masses"] = mass_energy(X, structure.masses, grid)
    return terms


def structure_energy(structure: Structure, grid: Grid, X: np.ndarray | None = None) -> float:
    """Total elastic energy per unit length E(X). Its negative gradient is compute_forces
    without the velocity terms (X_prev = None and zero target velocities, or no damping).
    X defaults to structure.X."""
    return float(sum(structure_energy_terms(structure, grid, X).values()))


# ====================================================================== forces

def spring_forces(X: np.ndarray, springs: Springs, grid: Grid) -> np.ndarray:
    """Spring forces -dE/dX: f_i += k sgn(L - rest) |L - rest|^alpha e,
    f_j -= the same, e = (X_j - X_i)/L. Elements with L = 0 contribute nothing."""
    f = np.zeros_like(X)
    _, L, e = _segments(X, springs.i, springs.j, grid)
    stretch = L - springs.rest
    mag = springs.k * np.sign(stretch) * np.abs(stretch) ** springs.alpha
    fi = mag[:, None] * e
    _accumulate(f, springs.i, fi)
    _accumulate(f, springs.j, -fi)
    return f


def damped_spring_forces(X: np.ndarray, damped: DampedSprings, grid: Grid,
                         V: np.ndarray | None = None) -> np.ndarray:
    """Hookean spring force plus the dashpot f_i += b (V_j - V_i), f_j -= the same.
    V (N, 2) are the point velocities; None means zero.

    The dashpot acts on the full relative velocity vector, so it damps rotation of the link as
    well as stretching. Its power -b |V_j - V_i|^2 is never positive.
    """
    f = np.zeros_like(X)
    _, L, e = _segments(X, damped.i, damped.j, grid)
    fi = (damped.k * (L - damped.rest))[:, None] * e
    if V is not None:
        V = np.asarray(V, dtype=float)
        fi = fi + damped.b[:, None] * (V[damped.j] - V[damped.i])
    _accumulate(f, damped.i, fi)
    _accumulate(f, damped.j, -fi)
    return f


def beam_forces(X: np.ndarray, beams: Beams, grid: Grid) -> np.ndarray:
    """Torsional beam forces, the exact -dE/dX of E = k/2 (c - curv)^2:
    with g = k (c - curv), f_r += -g (by, -bx), f_p += -g (ay, -ax), f_q += g (ay + by, -(ax + bx)).
    The three sum to zero."""
    f = np.zeros_like(X)
    ax, ay, bx, by = _beam_vectors(X, beams, grid)
    g = beams.k * (ax * by - ay * bx - beams.curv)
    fr = -g[:, None] * np.stack([by, -bx], axis=1)
    fp = -g[:, None] * np.stack([ay, -ax], axis=1)
    fq = g[:, None] * np.stack([ay + by, -(ax + bx)], axis=1)
    _accumulate(f, beams.r, fr)
    _accumulate(f, beams.p, fp)
    _accumulate(f, beams.q, fq)
    return f


def noninv_beam_forces(X: np.ndarray, beams: NonInvBeams, grid: Grid) -> np.ndarray:
    """Non-invariant beam forces -dE/dX of E = k/2 |D|^2:
    f_r += -k D, f_p += -k D, f_q += 2 k D."""
    f = np.zeros_like(X)
    kD = beams.k[:, None] * noninv_beam_difference(X, beams, grid)
    _accumulate(f, beams.r, -kD)
    _accumulate(f, beams.p, -kD)
    _accumulate(f, beams.q, 2.0 * kD)
    return f


def target_forces(X: np.ndarray, targets: Targets, grid: Grid,
                  V: np.ndarray | None = None) -> np.ndarray:
    """Target tether and damping, f_i += k (X_target - X_i) + c (V_target - V_i)
    (Miller & Peskin 2005, eq. 11), with the minimum image on the position difference.

    V (N, 2) are the point velocities; None means zero, so the damping then pulls towards
    V_target alone. With every c = 0 (the default) the damping term is skipped.
    """
    f = np.zeros_like(X)
    d = grid.min_image(targets.X - X[targets.idx])
    ft = targets.k[:, None] * d
    if np.any(targets.c != 0.0):
        dV = targets.V if V is None else targets.V - np.asarray(V, dtype=float)[targets.idx]
        ft = ft + targets.c[:, None] * dV
    _accumulate(f, targets.idx, ft)
    return f


def mass_tether_forces(X: np.ndarray, masses: Masses, grid: Grid | None = None) -> np.ndarray:
    """Per-element tether force k (Y - X_i) on the fluid points, shape (M, 2). The ghost feels
    the opposite. grid = None skips the minimum image (plain difference)."""
    d = masses.Y - np.asarray(X, dtype=float)[masses.idx]
    if grid is not None:
        d = grid.min_image(d)
    return masses.k[:, None] * d


def mass_forces(X: np.ndarray, masses: Masses, grid: Grid) -> np.ndarray:
    """Mass tether f_i += k (Y - X_i) with minimum image."""
    f = np.zeros_like(X)
    _accumulate(f, masses.idx, mass_tether_forces(X, masses, grid))
    return f


def compute_forces(structure: Structure, grid: Grid, t: float, dt: float,
                   X_prev: np.ndarray | None = None, X: np.ndarray | None = None) -> np.ndarray:
    """Total Lagrangian force density f (N, 2) [N/m] at the current configuration: springs,
    damped springs, torsional beams, non-invariant beams, targets and mass tethers.

    The elastic part of each is -dE/dX of its energy per unit length. The velocity terms, the
    dashpots of the damped springs and the target damping, use the point velocities
    V = min_image(X - X_prev)/dt (V = 0 when X_prev is None). X_prev, when given, must have
    the shape of X even if no term needs the velocities.

    t is not used here: time enters only through the hooks (update_targets etc.), which the
    driver calls before this. X overrides structure.X when the driver wants the forces at a
    half-step position without touching the structure.
    """
    X = structure.X if X is None else np.asarray(X, dtype=float)
    if X.shape != structure.X.shape:
        raise ValueError(f"X must have shape {structure.X.shape}, got {X.shape}")
    if X_prev is not None:
        X_prev = np.asarray(X_prev, dtype=float)
        if X_prev.shape != X.shape:
            raise ValueError(f"X_prev must have shape {X.shape}, got {X_prev.shape}")
    damped_targets = structure.targets is not None and bool(np.any(structure.targets.c != 0.0))
    V = None
    if X_prev is not None and (structure.damped_springs is not None or damped_targets):
        if dt <= 0:
            raise ValueError("dt must be positive to form velocities from X_prev")
        V = grid.min_image(X - X_prev) / dt
    f = np.zeros_like(X)
    if structure.springs is not None:
        f += spring_forces(X, structure.springs, grid)
    if structure.damped_springs is not None:
        f += damped_spring_forces(X, structure.damped_springs, grid, V)
    if structure.beams is not None:
        f += beam_forces(X, structure.beams, grid)
    if structure.noninv_beams is not None:
        f += noninv_beam_forces(X, structure.noninv_beams, grid)
    if structure.targets is not None:
        f += target_forces(X, structure.targets, grid, V)
    if structure.masses is not None:
        f += mass_forces(X, structure.masses, grid)
    return f


def advance_masses(structure: Structure, dt: float, f_tether: np.ndarray | None = None,
                   grid: Grid | None = None, X: np.ndarray | None = None) -> None:
    """One symplectic Euler step of the ghost mass points:
    V += dt (-k (Y - X_i)/m + g), then Y += dt V, in place on structure.masses.

    f_tether (M, 2) is the tether force k (Y - X_i) on the fluid points (mass_tether_forces);
    the ghosts feel minus that. When it is None we compute it from X (default structure.X)
    and structure.masses.Y, with the minimum image if a grid is given. The driver chooses
    the substep lengths; this routine does exactly one step of length dt.
    """
    masses = structure.masses
    if masses is None:
        return
    if f_tether is None:
        f_tether = mass_tether_forces(structure.X if X is None else X, masses, grid)
    f_tether = np.asarray(f_tether, dtype=float)
    if f_tether.shape != masses.Y.shape:
        raise ValueError(f"f_tether must have shape {masses.Y.shape}, got {f_tether.shape}")
    g = np.asarray(structure.gravity, dtype=float)
    masses.V += dt * (-f_tether / masses.m[:, None] + g)
    masses.Y += dt * masses.V


# ====================================================================== hooks

HOOK_NAMES = ("update_targets", "update_springs", "update_beams", "update_noninv_beams",
              "update_masses", "lagrangian_force", "external_force", "initial_velocity",
              "bc_left", "bc_right", "bc_bottom", "bc_top", "on_output")


@dataclass
class Hooks:
    """The optional user functions of a case's hooks file; None when absent.

    Signatures (docs/user-guide.md says when the driver calls each one):

    - update_targets, update_springs, update_beams, update_noninv_beams, update_masses:
      (t, dt, s: Structure) -> None, called with the time at which the forces are evaluated;
    - lagrangian_force(t, dt, s, X) -> f: an extra force density per unit length [N/m] of
      shape (N, 2) at the positions X, added to the structure forces at the same positions
      and time level;
    - external_force(t, grid, u, v) -> (Fu, Fv): a body force density on the u and v layouts;
    - initial_velocity(grid, layout_u, layout_v) -> (u, v): arrays of shapes layout_u.shape
      and layout_v.shape, called once when the Simulation is built;
    - bc_left, bc_right: (y, t) -> (u, v), and bc_bottom, bc_top: (x, t) -> (u, v), the wall
      velocities of the MAC back end;
    - on_output(step, t, state) -> None, for diagnostics at every output step.
    """

    update_targets: Callable | None = None
    update_springs: Callable | None = None
    update_beams: Callable | None = None
    update_noninv_beams: Callable | None = None
    update_masses: Callable | None = None
    lagrangian_force: Callable | None = None
    external_force: Callable | None = None
    initial_velocity: Callable | None = None
    bc_left: Callable | None = None
    bc_right: Callable | None = None
    bc_bottom: Callable | None = None
    bc_top: Callable | None = None
    on_output: Callable | None = None
    path: str | None = None
    module: ModuleType | None = field(default=None, repr=False, compare=False)

    def defined(self) -> list[str]:
        """Names of the hooks that are present."""
        return [name for name in HOOK_NAMES if getattr(self, name) is not None]


DEFAULT_HOOKS_FILE = "hooks.py"

# Top-level modules that load_hooks never takes out of sys.modules, whatever files a case
# directory holds: the standard library and the packages pyibm itself runs on. Importing them
# again would give the hooks a second copy, with its own classes, of a package already in use.
_NEVER_REIMPORTED = (frozenset(sys.builtin_module_names) | frozenset(sys.stdlib_module_names)
                     | frozenset({"pyibm", "numpy", "scipy", "matplotlib"}))


def _local_module_names(directory: str) -> set[str]:
    """Top-level names that 'import name' finds in directory: every name.py and every
    subdirectory with an __init__.py."""
    names: set[str] = set()
    try:
        entries = os.listdir(directory)
    except OSError:
        return names
    for entry in entries:
        full = os.path.join(directory, entry)
        stem, ext = os.path.splitext(entry)
        if ext == ".py" and stem.isidentifier() and os.path.isfile(full):
            names.add(stem)
        elif entry.isidentifier() and os.path.isfile(os.path.join(full, "__init__.py")):
            names.add(entry)
    return names


def _loaded_from(module, dirs: list[str]) -> bool:
    """True if the module's file, or for a package one of its search paths, lies in one of
    dirs (absolute and normcased)."""
    locations = []
    if getattr(module, "__file__", None):
        locations.append(module.__file__)
    try:
        locations.extend(getattr(module, "__path__", None) or [])
    except TypeError:
        pass
    for loc in locations:
        loc = os.path.normcase(os.path.abspath(loc))
        if any(loc == d or loc.startswith(d + os.sep) for d in dirs):
            return True
    return False


def load_hooks(case_dir: str, filename: str = DEFAULT_HOOKS_FILE) -> Hooks:
    """Load <case_dir>/<filename> and pick out the functions named in HOOK_NAMES.

    A missing hooks.py gives a Hooks with every entry None. Any other file name must exist,
    since it was asked for by name (Hooks { file = ... } in the input file); a missing one
    raises FileNotFoundError.

    The file is loaded under a module name made from its absolute path, so the hooks of
    different cases never shadow each other. While it runs, its directory (and case_dir, if
    that is a different directory) is first on sys.path, so it can import helper modules that
    sit beside it. Those helpers are imported fresh on every call and are taken out of
    sys.modules again afterwards, and whatever sys.modules held under their names before the
    call is put back. So a second case with its own helper.py, or the same case after its
    helper was edited, gets its own file and never a copy cached by an earlier case. The hooks
    module keeps its own references to the helpers, so the hook functions go on working.
    A helper here is a name.py file, or a package folder, directly in one of those
    directories. Anything else the hooks file imports (numpy, the standard library, a package
    from a virtual environment that happens to sit inside the case folder) is imported the
    usual way and stays loaded.

    No bytecode is written while the hooks file runs. Python checks a cached .pyc against the
    size of the source and its modification time in whole seconds, so a .pyc from one load
    could stand in for a same-length edit made within the same second, as in a sweep that
    rewrites a helper and builds the next Simulation straight away. (A .pyc written by some
    other import of the helper, from a Python session started in the case folder, say, is
    still used when it matches.)

    Three consequences. A helper imported inside a hook function, at call time, is not found,
    because the case directory is no longer on sys.path then; import helpers at the top of the
    hooks file. Standard-library modules and the packages pyibm runs on (pyibm, numpy,
    scipy, matplotlib) are never imported again, so case files should not use those names.
    And a helper package is isolated only if it has an __init__.py: a folder without one (a
    namespace package) is still imported fresh for each case, but it is not set aside
    beforehand, so a module of the same name that is already imported hides it.
    """
    path = os.path.join(case_dir, filename)
    if not os.path.isfile(path):
        if os.path.normpath(filename) == DEFAULT_HOOKS_FILE:
            return Hooks(path=None, module=None)
        raise FileNotFoundError(f"hooks file not found: {os.path.abspath(path)} "
                                "(named by Hooks { file = ... } in the input file)")
    abspath = os.path.abspath(path)
    digest = hashlib.sha1(abspath.encode("utf-8")).hexdigest()[:12]
    modname = f"pyibm_hooks_{digest}"
    spec = importlib.util.spec_from_file_location(modname, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load hooks from {abspath}")
    module = importlib.util.module_from_spec(spec)

    search: list[str] = []
    for d in (os.path.dirname(abspath), os.path.abspath(case_dir)):
        if d not in search:
            search.append(d)
    search_nc = [os.path.normcase(d) for d in search]
    local: set[str] = set()
    for d in search:
        local |= _local_module_names(d)
    local -= _NEVER_REIMPORTED

    def is_local(name: str) -> bool:
        return name.partition(".")[0] in local

    def from_case_folder(name: str) -> bool:
        # a package whose top-level folder sits directly in a search directory, such as a
        # helper package without __init__.py; packages found through other sys.path entries
        # (even ones inside case_dir, like a .venv) are not helpers
        top = name.partition(".")[0]
        if top in _NEVER_REIMPORTED:
            return False
        return _loaded_from(sys.modules[name],
                            [os.path.join(d, os.path.normcase(top)) for d in search_nc])

    # Set aside what sys.modules holds under the case's module names (another case's helper,
    # or a copy the user imported), so that the hooks file imports its own files.
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if is_local(name)}
    before = set(sys.modules)
    sys.modules[modname] = module
    sys.path[:0] = search
    importlib.invalidate_caches()        # so helper files written since the last import are seen
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True       # no .pyc that a later same-second edit could match
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(modname, None)
        raise
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
        for d in search:
            try:
                sys.path.remove(d)
            except ValueError:
                pass
        for name in list(sys.modules):
            if name == modname:
                continue
            if is_local(name) or (name not in before and from_case_folder(name)):
                del sys.modules[name]
        sys.modules.update(saved)
    hooks = Hooks(path=abspath, module=module)
    for name in HOOK_NAMES:
        fn = getattr(module, name, None)
        if fn is not None and not callable(fn):
            raise TypeError(f"{abspath}: {name} must be a function, got {type(fn).__name__}")
        setattr(hooks, name, fn)
    return hooks
