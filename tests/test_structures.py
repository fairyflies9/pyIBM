"""Tests for pyibm.structures: forces, structure files and the hooks loader.

Every force is checked against a central finite difference of its energy on random
configurations in a periodic box, so that the minimum image is exercised; the force code and
the energy code are separate routines, so the check does not depend on the implementation.
"""

import os
import sys
import textwrap
import types

import numpy as np
import pytest

from pyibm.grid import Grid
from pyibm import structures as st


# ---------------------------------------------------------------------------- helpers

def fd_gradient(energy, X, eps=1e-6):
    """Central finite difference dE/dX, shape (N, 2)."""
    g = np.zeros_like(X)
    for n in range(X.shape[0]):
        for c in range(2):
            Xp = X.copy()
            Xm = X.copy()
            Xp[n, c] += eps
            Xm[n, c] -= eps
            g[n, c] = (energy(Xp) - energy(Xm)) / (2.0 * eps)
    return g


def assert_force_is_minus_gradient(force, energy, X, rtol=1e-6):
    f = force(X)
    g = fd_gradient(energy, X)
    scale = np.max(np.abs(f))
    assert scale > 0.0, "degenerate test configuration"
    assert np.max(np.abs(f + g)) <= rtol * scale


@pytest.fixture
def grid():
    return Grid.periodic_box(1.0, 1.0, 16, 16)


def random_pairs(rng, N, M):
    i = rng.integers(0, N, size=M)
    j = (i + rng.integers(1, N, size=M)) % N      # j != i
    return i, j


def random_triples(rng, N, M):
    p = rng.integers(0, N, size=M)
    q = (p + rng.integers(1, N - 1, size=M)) % N
    r = (q + rng.integers(1, N - 1, size=M)) % N
    r = np.where(r == p, (r + 1) % N, r)
    r = np.where(r == q, (r + 1) % N, r)
    assert np.all((p != q) & (q != r) & (p != r))
    return p, q, r


# ---------------------------------------------------------------------------- springs

@pytest.mark.parametrize("alpha", [1.0, 3.0, 2.0, 1.5])
def test_spring_force_matches_energy(grid, alpha):
    rng = np.random.default_rng(1)
    N, M = 12, 20
    X = rng.uniform(0.0, 1.0, size=(N, 2))          # spread over the whole periodic box
    i, j = random_pairs(rng, N, M)
    springs = st.Springs(i, j, rng.uniform(0.5, 5.0, M), rng.uniform(0.05, 0.4, M), alpha)
    assert_force_is_minus_gradient(lambda Y: st.spring_forces(Y, springs, grid),
                                   lambda Y: st.spring_energy(Y, springs, grid), X)


def test_spring_force_mixed_alpha_and_net_zero(grid):
    rng = np.random.default_rng(2)
    N, M = 10, 15
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    i, j = random_pairs(rng, N, M)
    alpha = rng.choice([1.0, 3.0, 2.0], size=M)
    springs = st.Springs(i, j, rng.uniform(0.5, 5.0, M), rng.uniform(0.05, 0.4, M), alpha)
    assert_force_is_minus_gradient(lambda Y: st.spring_forces(Y, springs, grid),
                                   lambda Y: st.spring_energy(Y, springs, grid), X)
    f = st.spring_forces(X, springs, grid)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-13)


def test_spring_uses_minimum_image(grid):
    # points 0.1 apart across the x boundary: the short image is 0.1, not 0.9
    X = np.array([[0.05, 0.5], [0.95, 0.5]])
    springs = st.Springs([0], [1], [2.0], [0.04])
    f = st.spring_forces(X, springs, grid)
    # L = 0.1, stretch 0.06, magnitude 0.12 along e = (-1, 0) from point 0 towards point 1's image
    assert np.allclose(f[0], [-0.12, 0.0]) and np.allclose(f[1], [0.12, 0.0])
    assert st.spring_energy(X, springs, grid) == pytest.approx(0.5 * 2.0 * 0.06 ** 2)
    # the same configuration shifted and wrapped gives the same forces
    rng = np.random.default_rng(3)
    N, M = 9, 12
    Xr = rng.uniform(0.0, 1.0, size=(N, 2))
    i, j = random_pairs(rng, N, M)
    sp = st.Springs(i, j, 1.0, 0.2, 3.0)
    shifted = grid.wrap(Xr + np.array([0.37, 0.61]))
    assert np.allclose(st.spring_forces(Xr, sp, grid), st.spring_forces(shifted, sp, grid), atol=1e-12)


def test_spring_at_rest_and_zero_length(grid):
    X = np.array([[0.2, 0.2], [0.5, 0.2], [0.2, 0.2]])
    springs = st.Springs([0, 0], [1, 2], [1.0, 1.0], [0.3, 0.1])
    f = st.spring_forces(X, springs, grid)
    assert np.allclose(f, 0.0)             # first at rest length; second has L = 0 and is skipped
    assert np.all(np.isfinite(f))


# ---------------------------------------------------------------------------- damped springs

def test_damped_spring_elastic_part_matches_energy(grid):
    rng = np.random.default_rng(4)
    N, M = 10, 14
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    i, j = random_pairs(rng, N, M)
    d = st.DampedSprings(i, j, rng.uniform(0.5, 5.0, M), rng.uniform(0.05, 0.4, M), rng.uniform(0, 1, M))
    assert_force_is_minus_gradient(lambda Y: st.damped_spring_forces(Y, d, grid),
                                   lambda Y: st.damped_spring_energy(Y, d, grid), X)
    assert np.allclose(st.damped_spring_forces(X, d, grid).sum(axis=0), 0.0, atol=1e-13)


def test_damped_spring_dashpot(grid):
    rng = np.random.default_rng(5)
    N, M = 8, 10
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    i, j = random_pairs(rng, N, M)
    d = st.DampedSprings(i, j, 1.0, 0.2, rng.uniform(0.1, 2.0, M))
    V = rng.normal(size=(N, 2))
    f_damp = st.damped_spring_forces(X, d, grid, V) - st.damped_spring_forces(X, d, grid)
    expected = np.zeros_like(X)
    for m in range(M):
        expected[i[m]] += d.b[m] * (V[j[m]] - V[i[m]])
        expected[j[m]] -= d.b[m] * (V[j[m]] - V[i[m]])
    assert np.allclose(f_damp, expected, atol=1e-13)
    assert np.sum(f_damp * V) <= 0.0            # the dashpot only dissipates
    # through compute_forces: V = min_image(X - X_prev)/dt, including a wrap across the boundary
    # (point 0 sits just right of x = 0 and moves right, so X_prev[0] wraps to x = 0.9995)
    dt = 1e-3
    X = X.copy()
    V = V.copy()
    X[0] = [0.0005, 0.5]
    V[0] = [1.0, 0.0]
    X_prev = grid.wrap(X - dt * V)
    assert X_prev[0, 0] > 0.9
    s = st.Structure("d", X, 0.1, damped_springs=d)
    f_total = st.compute_forces(s, grid, 0.0, dt, X_prev=X_prev)
    assert np.allclose(f_total, st.damped_spring_forces(X, d, grid, V), atol=1e-10)
    plain = st.damped_spring_forces(X, d, grid, (X - X_prev) / dt)      # no minimum image
    assert not np.allclose(f_total, plain, atol=1e-6)
    assert np.allclose(st.compute_forces(s, grid, 0.0, dt), st.damped_spring_forces(X, d, grid))
    with pytest.raises(ValueError, match="X_prev"):
        st.compute_forces(s, grid, 0.0, dt, X_prev=X_prev[:3])


# ---------------------------------------------------------------------------- torsional beams

def test_beam_force_matches_energy(grid):
    rng = np.random.default_rng(6)
    N, M = 12, 10
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    p, q, r = random_triples(rng, N, M)
    beams = st.Beams(p, q, r, rng.uniform(0.5, 5.0, M), rng.uniform(-0.05, 0.05, M))
    assert_force_is_minus_gradient(lambda Y: st.beam_forces(Y, beams, grid),
                                   lambda Y: st.beam_energy(Y, beams, grid), X)
    f = st.beam_forces(X, beams, grid)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-13)


def test_beam_force_zero_at_preferred_curvature(grid):
    rng = np.random.default_rng(7)
    N, M = 9, 6
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    p, q, r = random_triples(rng, N, M)
    probe = st.Beams(p, q, r, 1.0, 0.0)
    curv = st.beam_curvature(X, probe, grid)
    beams = st.Beams(p, q, r, rng.uniform(0.5, 5.0, M), curv)
    assert np.allclose(st.beam_forces(X, beams, grid), 0.0, atol=1e-15)
    assert st.beam_energy(X, beams, grid) == pytest.approx(0.0, abs=1e-30)


def test_straight_beam_zero_force(grid):
    X = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    beams = st.Beams([0], [1], [2], [100.0], [0.0])
    assert np.allclose(st.beam_forces(X, beams, grid), 0.0, atol=1e-14)
    X = np.array([[0.1, 0.5], [0.4, 0.5], [0.7, 0.5]])
    assert np.allclose(st.beam_forces(X, beams, grid), 0.0, atol=1e-14)
    # a beam straddling the periodic boundary in x is still straight
    X = np.array([[0.9, 0.5], [0.0, 0.5], [0.1, 0.5]])
    assert np.allclose(st.beam_forces(X, beams, grid), 0.0, atol=1e-14)


def test_beam_sign_convention(grid):
    # p, q, r with a left (counterclockwise) turn: c = A x B < 0 (here -0.01). With curv = 0
    # the force on r pushes it back towards the straight line (towards -y here) and the forces
    # sum to zero.
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]])
    beams = st.Beams([0], [1], [2], [1.0], [0.0])
    A = X[2] - X[1]
    B = X[1] - X[0]
    c = A[0] * B[1] - A[1] * B[0]
    assert c == pytest.approx(-0.01)
    g = 1.0 * c
    f = st.beam_forces(X, beams, grid)
    assert np.allclose(f[2], -g * np.array([B[1], -B[0]]))
    assert np.allclose(f[0], -g * np.array([A[1], -A[0]]))
    assert np.allclose(f[1], g * np.array([A[1] + B[1], -(A[0] + B[0])]))
    assert f[2, 1] < 0.0
    # c = |A||B| sin(theta_B - theta_A): a convex polygon numbered counterclockwise has c < 0
    # at every beam, and the same polygon numbered clockwise has c > 0
    n = 12
    theta = 2.0 * np.pi * np.arange(n) / n
    ring = np.column_stack([0.5 + 0.2 * np.cos(theta), 0.5 + 0.2 * np.sin(theta)])
    idx = np.arange(n)
    ring_beams = st.Beams((idx - 1) % n, idx, (idx + 1) % n, 1.0, 0.0)
    c_ccw = st.beam_curvature(ring, ring_beams, grid)
    ds = 2.0 * 0.2 * np.sin(np.pi / n)                         # chord length
    assert np.all(c_ccw < 0.0)
    assert np.allclose(c_ccw, -ds ** 2 * np.sin(2.0 * np.pi / n))
    assert np.all(st.beam_curvature(ring[::-1], ring_beams, grid) > 0.0)


# ---------------------------------------------------------------------------- non-invariant beams

def test_noninv_beam_force_matches_energy(grid):
    rng = np.random.default_rng(8)
    N, M = 12, 10
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    p, q, r = random_triples(rng, N, M)
    beams = st.NonInvBeams(p, q, r, rng.uniform(0.5, 5.0, M), rng.uniform(-0.1, 0.1, M),
                           rng.uniform(-0.1, 0.1, M))
    assert_force_is_minus_gradient(lambda Y: st.noninv_beam_forces(Y, beams, grid),
                                   lambda Y: st.noninv_beam_energy(Y, beams, grid), X)
    f = st.noninv_beam_forces(X, beams, grid)
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-13)


def test_noninv_beam_zero_at_preferred_shape(grid):
    rng = np.random.default_rng(9)
    N, M = 8, 5
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    p, q, r = random_triples(rng, N, M)
    D = st.noninv_beam_difference(X, st.NonInvBeams(p, q, r, 1.0, 0.0, 0.0), grid)
    beams = st.NonInvBeams(p, q, r, 3.0, D[:, 0], D[:, 1])
    assert np.allclose(st.noninv_beam_forces(X, beams, grid), 0.0, atol=1e-15)
    straight = st.NonInvBeams([0], [1], [2], [1.0], 0.0, 0.0)
    Xs = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    assert np.allclose(st.noninv_beam_forces(Xs, straight, grid), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------- targets and masses

def test_target_force_matches_energy(grid):
    rng = np.random.default_rng(10)
    N, M = 10, 7
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    idx = rng.choice(N, size=M, replace=False)
    targets = st.Targets(idx, rng.uniform(1.0, 10.0, M), rng.uniform(0.0, 1.0, size=(M, 2)))
    assert_force_is_minus_gradient(lambda Y: st.target_forces(Y, targets, grid),
                                   lambda Y: st.target_energy(Y, targets, grid), X)
    # across the periodic boundary the tether pulls towards the near image
    X = np.array([[0.02, 0.5]])
    t = st.Targets([0], [3.0], [[0.98, 0.5]])
    assert np.allclose(st.target_forces(X, t, grid), [[3.0 * (-0.04), 0.0]])
    # c defaults to zero, and then the target velocity does not enter the force
    assert np.array_equal(t.c, [0.0])
    t.V[:] = [[5.0, -2.0]]
    assert np.allclose(st.target_forces(X, t, grid, V=np.array([[1.0, 1.0]])), [[-0.12, 0.0]])


def test_target_damping(grid):
    # f_i = k (X_target - X_i) + c (V_target - V_i), Miller & Peskin 2005 eq. 11, with
    # V_i = min_image(X_i - X_prev_i)/dt formed in compute_forces
    rng = np.random.default_rng(14)
    N, M = 8, 5
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    X[0] = [0.0005, 0.3]                       # moves right across x = 0 during the last step
    idx = np.array([0, 2, 3, 5, 7])
    k = rng.uniform(1.0, 10.0, M)
    c = np.array([0.5, 0.0, 2.0, 1.0, 0.25])
    Xt = grid.wrap(X[idx] + rng.normal(scale=0.01, size=(M, 2)))
    Vt = rng.normal(size=(M, 2))
    V = rng.normal(size=(N, 2))
    V[0] = [1.0, 0.0]
    dt = 1e-3
    X_prev = grid.wrap(X - dt * V)
    assert X_prev[0, 0] > 0.9
    targets = st.Targets(idx, k, Xt, Vt, c)
    s = st.Structure("t", X, 0.1, targets=targets)
    expected = np.zeros_like(X)
    expected[idx] = k[:, None] * grid.min_image(Xt - X[idx]) + c[:, None] * (Vt - V[idx])
    f = st.compute_forces(s, grid, 0.0, dt, X_prev=X_prev)
    assert np.allclose(f, expected, atol=1e-9)
    assert np.allclose(st.target_forces(X, targets, grid, V), expected, atol=1e-12)
    # no previous positions: the point velocities are taken as zero
    expected0 = np.zeros_like(X)
    expected0[idx] = k[:, None] * grid.min_image(Xt - X[idx]) + c[:, None] * Vt
    assert np.allclose(st.compute_forces(s, grid, 0.0, dt), expected0, atol=1e-12)
    # the damping only dissipates when the targets are at rest
    targets.V[:] = 0.0
    f_damp = st.target_forces(X, targets, grid, V) - st.target_forces(X, targets, grid)
    assert np.sum(f_damp * V) < 0.0
    # the elastic energy is unchanged by c, and is still the potential of the tether part
    assert st.target_energy(X, targets, grid) == pytest.approx(
        st.target_energy(X, st.Targets(idx, k, Xt), grid))
    with pytest.raises(ValueError, match="damping"):
        st.Targets([0], [1.0], [[0.0, 0.0]], c=[-1.0])
    with pytest.raises(ValueError):
        st.compute_forces(s, grid, 0.0, 0.0, X_prev=X_prev)          # dt must be positive
    # X_prev of the wrong shape is an error even when no term needs the velocities
    s0 = st.Structure("t0", X[:3], 0.1, targets=st.Targets([0, 2], [1.0, 2.0], X[[0, 2]]))
    with pytest.raises(ValueError, match="X_prev"):
        st.compute_forces(s0, grid, 0.0, dt, X_prev=np.zeros((5, 2)))
    assert np.array_equal(st.compute_forces(s0, grid, 0.0, dt, X_prev=X[:3]),
                          st.compute_forces(s0, grid, 0.0, dt))


def test_target_file_damping_column(tmp_path):
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    path = os.path.join(tmp_path, "w.target")
    # two columns: c = 0, and the writer keeps two columns
    st.write_target(path, st.Targets([0, 2], [5.0, 6.0], X[[0, 2]]))
    with open(path) as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "2" and all(len(ln.split()) == 2 for ln in lines[1:])
    t = st.read_target(path, X)
    assert np.array_equal(t.c, [0.0, 0.0]) and np.array_equal(t.k, [5.0, 6.0])
    # some c != 0: the third column is written and read back
    st.write_target(path, st.Targets([0, 2], [5.0, 6.0], X[[0, 2]], c=[0.0, 0.125]))
    with open(path) as fh:
        lines = fh.read().splitlines()
    assert all(len(ln.split()) == 3 for ln in lines[1:])
    t = st.read_target(path, X)
    assert np.array_equal(t.c, [0.0, 0.125]) and np.array_equal(t.X, X[[0, 2]])
    # c on some rows only, as for the spring alpha column
    with open(path, "w") as fh:
        fh.write("3\n0 1.0\n1 2.0 0.5\n2 3.0\n")
    assert np.array_equal(st.read_target(path, X).c, [0.0, 0.5, 0.0])
    with open(path, "w") as fh:
        fh.write("1\n0 1.0 -0.5\n")
    with pytest.raises(ValueError, match="w.target"):
        st.read_target(path, X)


def test_mass_force_matches_energy(grid):
    rng = np.random.default_rng(11)
    N, M = 10, 6
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    idx = rng.choice(N, size=M, replace=False)
    masses = st.Masses(idx, rng.uniform(1.0, 10.0, M), rng.uniform(0.5, 2.0, M),
                       rng.uniform(0.0, 1.0, size=(M, 2)))
    assert_force_is_minus_gradient(lambda Y: st.mass_forces(Y, masses, grid),
                                   lambda Y: st.mass_energy(Y, masses, grid), X)


def test_advance_masses_symplectic_euler(grid):
    X = np.array([[0.5, 0.5], [0.2, 0.2]])
    masses = st.Masses([0, 1], [4.0, 2.0], [2.0, 0.5], [[0.52, 0.5], [0.2, 0.19]], [[0.0, 0.1], [0.0, 0.0]])
    s = st.Structure("m", X, 0.1, masses=masses, gravity=(0.0, -9.81))
    V0 = masses.V.copy()
    Y0 = masses.Y.copy()
    dt = 1e-3
    st.advance_masses(s, dt, grid=grid)
    acc = -masses.k[:, None] * (Y0 - X) / masses.m[:, None] + np.array([0.0, -9.81])
    V1 = V0 + dt * acc
    assert np.allclose(masses.V, V1)
    assert np.allclose(masses.Y, Y0 + dt * V1)
    # passing the tether force explicitly gives the same step
    masses.V[:] = V0
    masses.Y[:] = Y0
    st.advance_masses(s, dt, f_tether=st.mass_tether_forces(X, masses, grid))
    assert np.allclose(masses.V, V1) and np.allclose(masses.Y, Y0 + dt * V1)
    # the ghosts feel minus the force the fluid points feel
    f = st.mass_forces(X, masses, grid)
    assert np.allclose(f, st.mass_tether_forces(X, masses, grid))
    # no masses: a no-op
    st.advance_masses(st.Structure("n", X, 0.1), dt)


# ---------------------------------------------------------------------------- the whole structure

def make_full_structure(rng, N=14):
    X = rng.uniform(0.0, 1.0, size=(N, 2))
    i, j = random_pairs(rng, N, 12)
    p, q, r = random_triples(rng, N, 8)
    idx_t = rng.choice(N, size=4, replace=False)
    idx_m = rng.choice(N, size=3, replace=False)
    return st.Structure(
        "full", X, 0.05,
        springs=st.Springs(i, j, rng.uniform(0.5, 5, 12), rng.uniform(0.05, 0.4, 12),
                           rng.choice([1.0, 3.0], size=12)),
        damped_springs=st.DampedSprings(j, i, rng.uniform(0.5, 5, 12), rng.uniform(0.05, 0.4, 12), 0.3),
        beams=st.Beams(p, q, r, rng.uniform(0.5, 5, 8), rng.uniform(-0.05, 0.05, 8)),
        noninv_beams=st.NonInvBeams(r, q, p, rng.uniform(0.5, 5, 8), 0.01, -0.02),
        targets=st.Targets(idx_t, 10.0, rng.uniform(0, 1, size=(4, 2)), c=0.5),
        masses=st.Masses(idx_m, 5.0, 1.0, rng.uniform(0, 1, size=(3, 2))),
        gravity=(0.0, -1.0),
        X0=X,
    )


def test_compute_forces_is_minus_gradient_of_structure_energy(grid):
    rng = np.random.default_rng(12)
    s = make_full_structure(rng)
    assert_force_is_minus_gradient(lambda Y: st.compute_forces(s, grid, 0.0, 1e-3, X=Y),
                                   lambda Y: st.structure_energy(s, grid, X=Y), s.X)
    terms = st.structure_energy_terms(s, grid)
    assert set(terms) == {"springs", "damped_springs", "beams", "noninv_beams", "targets", "masses"}
    assert st.structure_energy(s, grid) == pytest.approx(sum(terms.values()))
    f = st.compute_forces(s, grid, 0.0, 1e-3)
    parts = (st.spring_forces(s.X, s.springs, grid) + st.damped_spring_forces(s.X, s.damped_springs, grid)
             + st.beam_forces(s.X, s.beams, grid) + st.noninv_beam_forces(s.X, s.noninv_beams, grid)
             + st.target_forces(s.X, s.targets, grid) + st.mass_forces(s.X, s.masses, grid))
    assert np.allclose(f, parts, atol=1e-13)
    assert f.shape == (s.n, 2) and f.dtype == np.float64
    with pytest.raises(ValueError):
        st.compute_forces(s, grid, 0.0, 1e-3, X=s.X[:3])


def test_forces_on_a_non_periodic_grid():
    g = Grid(1.0, 1.0, 16, 16, periodic_x=False, periodic_y=False)
    X = np.array([[0.05, 0.5], [0.95, 0.5]])
    springs = st.Springs([0], [1], [1.0], [0.5])
    f = st.spring_forces(X, springs, g)
    assert np.allclose(f[0], [0.4, 0.0])          # L = 0.9, no image
    assert_force_is_minus_gradient(lambda Y: st.spring_forces(Y, springs, g),
                                   lambda Y: st.spring_energy(Y, springs, g), X)


def test_dataclass_validation():
    with pytest.raises(ValueError):
        st.Springs([0], [0], [1.0], [0.1])              # self-connection
    with pytest.raises(ValueError):
        st.Springs([0, 1], [1], [1.0], [0.1])           # length mismatch
    with pytest.raises(ValueError):
        st.Springs([0], [1], [1.0], [0.1], [0.0])       # alpha must be positive
    with pytest.raises(ValueError):
        st.Beams([0], [1], [0], [1.0], [0.0])
    with pytest.raises(ValueError):
        st.Masses([0], [1.0], [0.0], [[0.0, 0.0]])      # m > 0
    with pytest.raises(ValueError):
        st.Structure("s", np.zeros((3, 2)), 0.1, springs=st.Springs([0], [3], [1.0], [0.1]))
    with pytest.raises(ValueError):
        st.Structure("s", np.zeros((3, 2)), 0.0)
    s = st.Structure("s", [[0.0, 0.0], [1.0, 1.0]], 0.25)
    assert s.ds.shape == (2,) and np.all(s.ds == 0.25) and s.n == 2
    # X0 defaults to a copy of the initial positions, not a view of X
    assert np.array_equal(s.X0, [[0.0, 0.0], [1.0, 1.0]])
    s.X[0] = [0.5, 0.5]
    assert np.array_equal(s.X0[0], [0.0, 0.0])
    s = st.Structure("s", [[0.0, 0.0], [1.0, 1.0]], 0.25, X0=[[0.1, 0.0], [1.0, 1.1]])
    assert np.array_equal(s.X0, [[0.1, 0.0], [1.0, 1.1]])
    with pytest.raises(ValueError):
        st.Structure("s", np.zeros((3, 2)), 0.1, X0=np.zeros((2, 2)))
    sp = st.Springs(np.array([0.0, 1.0]), [1, 2], 2.0, 0.1)       # whole floats accepted as indices
    assert sp.i.dtype == np.intp and len(sp) == 2 and np.all(sp.alpha == 1.0)


# ---------------------------------------------------------------------------- files

def test_write_read_round_trip(tmp_path):
    rng = np.random.default_rng(13)
    s = make_full_structure(rng)
    written = st.write_structure(str(tmp_path), s)
    assert len(written) == 7 and all(os.path.isfile(p) for p in written)
    r = st.read_structure(str(tmp_path), "full", ds=0.05, gravity=(0.0, -1.0))
    assert r.name == "full" and r.gravity == (0.0, -1.0)
    assert np.array_equal(r.X, s.X) and np.array_equal(r.X0, s.X) and np.allclose(r.ds, 0.05)
    for a in ("i", "j", "k", "rest", "alpha"):
        assert np.array_equal(getattr(r.springs, a), getattr(s.springs, a))
    for a in ("i", "j", "k", "rest", "b"):
        assert np.array_equal(getattr(r.damped_springs, a), getattr(s.damped_springs, a))
    for a in ("p", "q", "r", "k", "curv"):
        assert np.array_equal(getattr(r.beams, a), getattr(s.beams, a))
    for a in ("p", "q", "r", "k", "cx", "cy"):
        assert np.array_equal(getattr(r.noninv_beams, a), getattr(s.noninv_beams, a))
    assert np.array_equal(r.targets.idx, s.targets.idx) and np.array_equal(r.targets.k, s.targets.k)
    assert np.array_equal(r.targets.X, s.X[s.targets.idx]) and np.all(r.targets.V == 0.0)
    assert np.array_equal(r.targets.c, s.targets.c) and np.all(r.targets.c == 0.5)
    assert np.array_equal(r.masses.idx, s.masses.idx) and np.array_equal(r.masses.k, s.masses.k)
    assert np.array_equal(r.masses.m, s.masses.m)
    assert np.array_equal(r.masses.Y, s.X[s.masses.idx]) and np.all(r.masses.V == 0.0)
    # the spring file carried the alpha column because some alpha are 3
    with open(os.path.join(tmp_path, "full.spring")) as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "12" and all(len(ln.split()) == 5 for ln in lines[1:])


def test_spring_file_optional_alpha(tmp_path):
    path = os.path.join(tmp_path, "s.spring")
    st.write_spring(path, st.Springs([0, 1], [1, 2], [1.0, 2.0], [0.1, 0.2]))
    with open(path) as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "2" and all(len(ln.split()) == 4 for ln in lines[1:])
    sp = st.read_spring(path)
    assert np.all(sp.alpha == 1.0) and np.array_equal(sp.j, [1, 2])
    # a file with alpha on some rows only
    with open(path, "w") as fh:
        fh.write("3\n0 1 1.0 0.1\n1 2 2.0 0.2 3\n2 0 1.5 0.3\n")
    sp = st.read_spring(path, n_points=3)
    assert np.array_equal(sp.alpha, [1.0, 3.0, 1.0])
    with pytest.raises(ValueError):
        st.read_spring(path, n_points=2)      # index 2 out of range


def test_header_mismatch_and_bad_rows(tmp_path):
    bad = os.path.join(tmp_path, "bad.vertex")
    with open(bad, "w") as fh:
        fh.write("3\n0.0 0.0\n0.1 0.0\n")
    with pytest.raises(ValueError) as err:
        st.read_vertex(bad)
    assert "bad.vertex" in str(err.value) and "3" in str(err.value)
    with pytest.raises(ValueError):
        st.read_structure(str(tmp_path), "bad", ds=0.1)
    with open(bad, "w") as fh:
        fh.write("2\n0.0 0.0 0.0\n0.1 0.0\n")
    with pytest.raises(ValueError, match="columns"):
        st.read_vertex(bad)
    with open(bad, "w") as fh:
        fh.write("2\n0.0 abc\n0.1 0.0\n")
    with pytest.raises(ValueError, match="numbers"):
        st.read_vertex(bad)
    with open(bad, "w") as fh:
        fh.write("x y\n0.0 0.0\n")
    with pytest.raises(ValueError, match="row count"):
        st.read_vertex(bad)
    with pytest.raises(FileNotFoundError):
        st.read_structure(str(tmp_path), "nothing", ds=0.1)


def test_index_range_and_zero_based(tmp_path):
    st.write_vertex(os.path.join(tmp_path, "p.vertex"), np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]))
    with open(os.path.join(tmp_path, "p.spring"), "w") as fh:
        fh.write("1\n1 3 1.0 0.1\n")              # 1-based style: index 3 is out of range
    with pytest.raises(ValueError) as err:
        st.read_structure(str(tmp_path), "p", ds=0.1)
    assert "p.spring" in str(err.value) and "0-based" in str(err.value)
    with open(os.path.join(tmp_path, "p.spring"), "w") as fh:
        fh.write("1\n0 1.5 1.0 0.1\n")
    with pytest.raises(ValueError, match="integer"):
        st.read_structure(str(tmp_path), "p", ds=0.1)
    os.remove(os.path.join(tmp_path, "p.spring"))
    with open(os.path.join(tmp_path, "p.target"), "w") as fh:
        fh.write("1\n-1 1.0\n")
    with pytest.raises(ValueError, match="p.target"):
        st.read_structure(str(tmp_path), "p", ds=0.1)


def test_files_allow_comments_and_blank_lines(tmp_path):
    path = os.path.join(tmp_path, "c.vertex")
    with open(path, "w") as fh:
        fh.write("# a plate\n2   % two points\n\n0.0 0.0  # first\n0.1 0.0\n\n")
    assert np.array_equal(st.read_vertex(path), [[0.0, 0.0], [0.1, 0.0]])


def test_files_with_byte_order_mark(tmp_path):
    # PowerShell 5.1 (Set-Content -Encoding utf8) and Notepad's "UTF-8 with BOM" write a BOM
    def write_bom(name, text):
        with open(os.path.join(tmp_path, name), "w", encoding="utf-8-sig") as fh:
            fh.write(text)

    write_bom("b.vertex", "3\n0.0 0.0\n0.1 0.0\n0.2 0.0\n")
    write_bom("b.spring", "# springs, with a comment first\n2\n0 1 1.0 0.1\n1 2 1.0 0.1\n")
    write_bom("b.target", "% targets\n1\n0 10.0\n")
    s = st.read_structure(str(tmp_path), "b", ds=0.1)
    assert s.n == 3 and np.allclose(s.X[:, 0], [0.0, 0.1, 0.2])
    assert np.array_equal(s.springs.j, [1, 2]) and np.array_equal(s.targets.k, [10.0])


def test_ds_rule(tmp_path):
    X = np.column_stack([np.arange(10) * 0.1, np.zeros(10)])
    st.write_vertex(os.path.join(tmp_path, "line.vertex"), X)
    s = st.read_structure(str(tmp_path), "line")
    assert np.allclose(s.ds, 0.1)
    s = st.read_structure(str(tmp_path), "line", ds=0.03)
    assert np.allclose(s.ds, 0.03)
    assert st.nearest_neighbour_ds(X) == pytest.approx(0.1)
    # one far-away point: ratio of nearest-neighbour distances > 2 warns, the median wins
    Xw = np.vstack([X, [[5.0, 5.0]]])
    with pytest.warns(UserWarning, match="ratio"):
        ds = st.nearest_neighbour_ds(Xw)
    assert ds == pytest.approx(0.1)
    with pytest.raises(ValueError):
        st.nearest_neighbour_ds(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        st.nearest_neighbour_ds(np.zeros((3, 2)))      # duplicates: median distance 0
    with pytest.raises(ValueError):
        st.read_structure(str(tmp_path), "line", ds=-1.0)
    # an explicit ds does not touch the vertices, so duplicates are then allowed
    st.write_vertex(os.path.join(tmp_path, "dup.vertex"), np.zeros((3, 2)))
    assert st.read_structure(str(tmp_path), "dup", ds=0.1).n == 3


# ---------------------------------------------------------------------------- hooks

HOOKS_A = """
import numpy as np
from helper import OFFSET

TAG = "A"

def update_targets(t, dt, s):
    s.targets.X[:, 0] = s.X0[s.targets.idx, 0] + OFFSET * t

def bc_left(y, t):
    return np.ones_like(y) * t, np.zeros_like(y)

def not_a_hook():
    return 1
"""

HOOKS_B = """
TAG = "B"

def update_targets(t, dt, s):
    s.targets.X[:, 1] += 1.0

def external_force(t, grid, u, v):
    return u * 0, v * 0 + 1.0

def on_output(step, t, state):
    pass
"""


def test_load_hooks_from_two_cases(tmp_path):
    a = tmp_path / "caseA"
    b = tmp_path / "caseB"
    a.mkdir()
    b.mkdir()
    (a / "hooks.py").write_text(textwrap.dedent(HOOKS_A))
    (a / "helper.py").write_text("OFFSET = 2.0\n")
    (b / "hooks.py").write_text(textwrap.dedent(HOOKS_B))
    ha = st.load_hooks(str(a))
    hb = st.load_hooks(str(b))
    assert ha.defined() == ["update_targets", "bc_left"]
    assert hb.defined() == ["update_targets", "external_force", "on_output"]
    assert ha.update_springs is None and hb.bc_left is None
    assert ha.update_targets is not hb.update_targets
    assert ha.module.TAG == "A" and hb.module.TAG == "B"
    assert ha.module is not hb.module and ha.path.endswith("hooks.py")
    # the hooks really run, and see the sibling helper module of their own case
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    s = st.Structure("h", X, 0.1, targets=st.Targets([1], [1.0], X[[1]]), X0=X)
    ha.update_targets(0.5, 0.1, s)
    assert np.allclose(s.targets.X, [[0.3 + 1.0, 0.4]])
    hb.update_targets(0.5, 0.1, s)
    assert np.allclose(s.targets.X, [[1.3, 1.4]])
    u, v = ha.bc_left(np.linspace(0, 1, 5), 2.0)
    assert np.allclose(u, 2.0) and np.allclose(v, 0.0)
    # loading A again gives a fresh module with the same behaviour
    ha2 = st.load_hooks(str(a))
    assert ha2.update_targets is not ha.update_targets and ha2.module.TAG == "A"


def test_load_hooks_missing_and_invalid(tmp_path):
    h = st.load_hooks(str(tmp_path))
    assert h.defined() == [] and h.module is None and h.path is None
    assert all(getattr(h, name) is None for name in st.HOOK_NAMES)
    (tmp_path / "hooks.py").write_text("update_targets = 3\n")
    with pytest.raises(TypeError, match="update_targets"):
        st.load_hooks(str(tmp_path))
    (tmp_path / "hooks.py").write_text("raise RuntimeError('boom')\n")
    with pytest.raises(RuntimeError, match="boom"):
        st.load_hooks(str(tmp_path))


HOOKS_HELPER = """
import helper

def update_targets(t, dt, s):
    s.targets.X[:, 0] = helper.VALUE

def value():
    return helper.VALUE
"""


def make_case(root, name, hooks_text, files=None):
    case = root / name
    case.mkdir()
    (case / "hooks.py").write_text(textwrap.dedent(hooks_text))
    for rel, text in (files or {}).items():
        target = case / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(text))
    return case


def test_load_hooks_helpers_do_not_leak_between_cases(tmp_path):
    # two copies of a case that differ only in their helper.py, loaded in one process, as in
    # a parameter sweep from a notebook
    a = make_case(tmp_path, "caseA", HOOKS_HELPER, {"helper.py": "VALUE = 1.0\n"})
    b = make_case(tmp_path, "caseB", HOOKS_HELPER, {"helper.py": "VALUE = 5.0\n"})
    ha = st.load_hooks(str(a))
    assert "helper" not in sys.modules
    hb = st.load_hooks(str(b))
    assert "helper" not in sys.modules
    assert ha.module.value() == 1.0 and hb.module.value() == 5.0
    X = np.array([[0.1, 0.2]])
    s = st.Structure("h", X, 0.1, targets=st.Targets([0], [1.0], X))
    hb.update_targets(0.0, 0.1, s)
    assert s.targets.X[0, 0] == 5.0
    ha.update_targets(0.0, 0.1, s)              # A's functions still see A's helper
    assert s.targets.X[0, 0] == 1.0
    # an edited helper is read again on the next load (a different length, so that a .pyc
    # written in the same second cannot be taken for the new file)
    (a / "helper.py").write_text("VALUE = 2.0   # edited\n")
    assert st.load_hooks(str(a)).module.value() == 2.0
    assert ha.module.value() == 1.0             # the earlier load keeps its own copy


def test_load_hooks_isolates_case_packages_and_restores_sys_modules(tmp_path):
    hooks_text = """
    from tools.shapes import RADIUS
    import helper

    def value():
        return RADIUS, helper.VALUE
    """
    a = make_case(tmp_path, "caseA", hooks_text, {"tools/__init__.py": "",
                                                    "tools/shapes.py": "RADIUS = 0.1\n",
                                                    "helper.py": "VALUE = 1\n"})
    b = make_case(tmp_path, "caseB", hooks_text, {"tools/__init__.py": "",
                                                    "tools/shapes.py": "RADIUS = 0.25\n",
                                                    "helper.py": "VALUE = 2\n"})
    # a module the user imported under one of the case's names is set aside and put back
    mine = types.ModuleType("helper")
    mine.VALUE = -1
    sys.modules["helper"] = mine
    try:
        assert st.load_hooks(str(a)).module.value() == (0.1, 1)
        assert sys.modules["helper"] is mine
        assert st.load_hooks(str(b)).module.value() == (0.25, 2)
        assert sys.modules["helper"] is mine
    finally:
        sys.modules.pop("helper", None)
    assert not any(name == "tools" or name.startswith("tools.") for name in sys.modules)


def test_load_hooks_never_reimports_stdlib_or_pyibm(tmp_path):
    # case files named like a standard-library module or like pyibm itself must not replace
    # the modules already in use: the hooks get the real ones
    hooks_text = """
    import random
    import pyibm.structures
    from pyibm.structures import Structure
    """
    case = make_case(tmp_path, "case", hooks_text, {"random.py": "FAKE = True\n",
                                                     "pyibm/__init__.py": "FAKE = True\n",
                                                     "pyibm/structures.py": "Structure = None\n"})
    import random
    import pyibm
    h = st.load_hooks(str(case))
    assert h.module.random is random and not hasattr(random, "FAKE")
    assert h.module.Structure is st.Structure
    assert sys.modules["pyibm"] is pyibm and sys.modules["pyibm.structures"] is st


def test_load_hooks_named_file_and_new_hooks(tmp_path):
    case = make_case(tmp_path, "case", "def bc_right(y, t):\n    return y * 0, y * 0\n", {
        "hooks_fast.py": """
            import numpy as np
            import helper

            def lagrangian_force(t, dt, s, X):
                return np.full_like(X, helper.VALUE)

            def initial_velocity(grid, layout_u, layout_v):
                return np.zeros(layout_u.shape), np.zeros(layout_v.shape)
            """,
        "helper.py": "VALUE = 3.0\n",
        "variants/hooks_v.py": """
            import helper
            import vhelper

            def external_force(t, grid, u, v):
                return u * 0 + helper.VALUE, v * 0 + vhelper.VALUE
            """,
        "variants/vhelper.py": "VALUE = 7.0\n",
    })
    assert "lagrangian_force" in st.HOOK_NAMES and "initial_velocity" in st.HOOK_NAMES
    assert st.load_hooks(str(case)).defined() == ["bc_right"]
    h = st.load_hooks(str(case), filename="hooks_fast.py")
    assert h.defined() == ["lagrangian_force", "initial_velocity"]
    assert h.path == os.path.abspath(case / "hooks_fast.py")
    X = np.zeros((4, 2))
    assert np.array_equal(h.lagrangian_force(0.0, 0.1, None, X), np.full((4, 2), 3.0))
    # a hooks file in a subdirectory finds the helpers beside it and those of the case
    hv = st.load_hooks(str(case), filename=os.path.join("variants", "hooks_v.py"))
    Fu, Fv = hv.external_force(0.0, None, np.zeros(2), np.zeros(2))
    assert np.all(Fu == 3.0) and np.all(Fv == 7.0)
    assert "helper" not in sys.modules and "vhelper" not in sys.modules
    # a file asked for by name has to exist; a missing hooks.py just means no hooks
    with pytest.raises(FileNotFoundError, match="hooks_typo.py"):
        st.load_hooks(str(case), filename="hooks_typo.py")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert st.load_hooks(str(empty), filename="hooks.py").defined() == []


def test_load_hooks_keeps_modules_found_elsewhere_on_sys_path(tmp_path, monkeypatch):
    # A virtual environment inside the case folder: a package the hooks import from it is a
    # normal installed package, not a case helper, so it stays loaded and is the one the
    # rest of the program gets.
    case = make_case(tmp_path, "case", "import fakelib\n",
                     {".venv/site/fakelib/__init__.py": "VALUE = 1\n"})
    monkeypatch.syspath_prepend(str(case / ".venv" / "site"))
    monkeypatch.delitem(sys.modules, "fakelib", raising=False)
    h = st.load_hooks(str(case))
    try:
        assert sys.modules["fakelib"] is h.module.fakelib
    finally:
        sys.modules.pop("fakelib", None)
    # A case folder that holds the standard library itself (an ancestor of the Python
    # install, say): a standard-library package first imported by the hooks stays loaded.
    stdlib = os.path.dirname(os.__file__)
    if not os.path.isfile(os.path.join(stdlib, "pydoc_data", "topics.py")):
        pytest.skip("this Python has no pydoc_data package")
    for name in [n for n in sys.modules if n.partition(".")[0] == "pydoc_data"]:
        monkeypatch.delitem(sys.modules, name)
    hooks = tmp_path / "hooks_std.py"
    hooks.write_text("import pydoc_data.topics\n")
    h = st.load_hooks(stdlib, filename=str(hooks))
    assert sys.modules["pydoc_data.topics"] is h.module.pydoc_data.topics
    assert sys.modules["pydoc_data"] is h.module.pydoc_data


def test_load_hooks_reads_a_helper_edited_in_the_same_second(tmp_path, monkeypatch):
    # With bytecode writing on (the Python default) a .pyc of the first helper.py would match
    # a same-length edit with the same modification time, and the old value would come back.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    case = make_case(tmp_path, "case", HOOKS_HELPER, {"helper.py": "VALUE = 1.0\n"})
    helper = case / "helper.py"
    st1 = os.stat(helper)
    assert st.load_hooks(str(case)).module.value() == 1.0
    helper.write_text("VALUE = 2.0\n")
    os.utime(helper, ns=(st1.st_atime_ns, st1.st_mtime_ns))
    assert st.load_hooks(str(case)).module.value() == 2.0
    assert not (case / "__pycache__").exists()
    assert sys.dont_write_bytecode is False


def test_load_hooks_isolates_namespace_helper_packages(tmp_path):
    # a helper package without __init__.py is imported fresh for each case as well
    hooks_text = """
    from tools.shapes import RADIUS

    def value():
        return RADIUS
    """
    a = make_case(tmp_path, "caseA", hooks_text, {"tools/shapes.py": "RADIUS = 0.1\n"})
    b = make_case(tmp_path, "caseB", hooks_text, {"tools/shapes.py": "RADIUS = 0.25\n"})
    assert st.load_hooks(str(a)).module.value() == 0.1
    assert st.load_hooks(str(b)).module.value() == 0.25
    assert not any(name == "tools" or name.startswith("tools.") for name in sys.modules)
