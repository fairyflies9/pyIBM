"""Tests for pyibm.kernels: kernel identities, stencil construction, wrap-around, walls.

Each identity is checked against the kernel's definition, not against the implementation:
the even/odd sum rule (eq. 14.63), the partition of unity (eq. 14.66, which follows from
14.63), the first moment (eq. 14.64) and the sum of squares (eq. 14.65) of Peskin & McQueen
1996 section 14.5, evaluated at random off-grid positions. The full references are in the
``pyibm.kernels`` module docstring.
"""

import math

import numpy as np
import pytest

from pyibm.kernels import Cosine4, Kernel, Peskin4, Roma3, get_kernel, stencil_1d

KERNELS = ["peskin4", "cosine4", "roma3"]


def _unwrapped_offsets(kernel, s):
    """Node offsets j (relative to s) that stencil_1d is specified to use, built independently."""
    if kernel.width % 2 == 0:
        i0 = np.floor(s).astype(int)
        return i0[:, None] + np.arange(-(kernel.width // 2 - 1), kernel.width // 2 + 1)[None, :]
    i0 = np.floor(s + 0.5).astype(int)
    return i0[:, None] + np.arange(-(kernel.width // 2), kernel.width // 2 + 1)[None, :]


# ---------------------------------------------------------------------------- kernels
def test_get_kernel_names_and_properties():
    k = get_kernel("peskin4")
    assert isinstance(k, Peskin4) and isinstance(k, Kernel)
    assert k.name == "peskin4" and k.half_width == 2.0 and k.width == 4
    k = get_kernel("cosine4")
    assert isinstance(k, Cosine4) and k.name == "cosine4" and k.half_width == 2.0 and k.width == 4
    k = get_kernel("roma3")
    assert isinstance(k, Roma3) and k.name == "roma3" and k.half_width == 1.5 and k.width == 3
    assert get_kernel().name == "peskin4"                   # default
    assert get_kernel(" Peskin4 ").name == "peskin4"        # case/whitespace tolerant
    with pytest.raises(ValueError, match="unknown delta kernel"):
        get_kernel("peskin6")
    with pytest.raises(NotImplementedError):
        Kernel().phi(np.zeros(3))


def test_phi_known_values_symmetry_and_support():
    p4, c4, r3 = get_kernel("peskin4"), get_kernel("cosine4"), get_kernel("roma3")
    # values at the nodes, from the closed forms
    assert p4.phi(0.0) == pytest.approx(0.5) and p4.phi(1.0) == pytest.approx(0.25)
    assert p4.phi(2.0) == 0.0 and p4.phi(2.5) == 0.0
    assert c4.phi(0.0) == pytest.approx(0.5) and c4.phi(1.0) == pytest.approx(0.25)
    assert c4.phi(2.0) == 0.0
    assert r3.phi(0.0) == pytest.approx(2.0 / 3.0) and r3.phi(1.0) == pytest.approx(1.0 / 6.0)
    assert r3.phi(1.5) == 0.0 and r3.phi(0.5) == pytest.approx(0.5)
    r = np.linspace(-3, 3, 601)
    for k in (p4, c4, r3):
        v = k.phi(r)
        assert v.shape == r.shape and v.dtype == np.float64
        assert np.allclose(v, k.phi(-r))                          # even
        assert np.all(v[np.abs(r) >= k.half_width] == 0.0)        # zero outside support
        assert np.all(v[np.abs(r) < k.half_width] > 0.0)          # positive inside
        assert np.all(np.isfinite(k.phi(np.array([1.0, 2.0, 0.5, 1.5]))))   # clamped radicands
    # continuity at the branch points
    eps = 1e-9
    assert abs(p4.phi(1 - eps) - p4.phi(1 + eps)) < 1e-6
    assert abs(r3.phi(0.5 - eps) - r3.phi(0.5 + eps)) < 1e-6
    assert abs(r3.phi(1.5 - eps)) < 1e-4


@pytest.mark.parametrize("name", KERNELS)
def test_partition_of_unity(name):
    """sum_j phi(r - j) = 1 for every r: sum of stencil weights times h equals 1 (eq. 14.66)."""
    kernel = get_kernel(name)
    rng = np.random.default_rng(1)
    x0, h, n = 0.37, 0.0125, 4000
    X = x0 + rng.uniform(0.0, n * h, 10_000)
    idx, w = stencil_1d(kernel, X, x0, h, n, periodic=True)
    assert idx.shape == (10_000, kernel.width) and w.shape == idx.shape
    assert np.max(np.abs(w.sum(axis=1) * h - 1.0)) < 1e-12


def test_even_odd_sum_rule():
    """sum_{j even} phi(r - j) = sum_{j odd} phi(r - j) = 1/2 for every r (eq. 14.63) holds for
    peskin4 and cosine4. roma3 keeps only the total (eq. 14.66): at an integer r one of its
    two sums is 2/3 and the other 1/3, which is why it is meant for the MAC grid only."""
    rng = np.random.default_rng(6)
    integers = np.arange(-3.0, 4.0)
    s = np.concatenate([rng.uniform(-10, 10, 10_000), integers, integers + 0.5])
    on_node = np.isin(s, integers)
    for name in KERNELS:
        kernel = get_kernel(name)
        j = _unwrapped_offsets(kernel, s)          # every node inside the support of each s
        w = kernel.phi(j - s[:, None])
        even = np.where(np.mod(j, 2) == 0, w, 0.0).sum(axis=1)
        odd = np.where(np.mod(j, 2) == 1, w, 0.0).sum(axis=1)
        assert np.max(np.abs(even + odd - 1.0)) < 1e-12, name
        if name == "roma3":
            assert np.max(np.abs(even - 0.5)) > 0.1
            assert np.allclose(np.abs(even - odd)[on_node], 1.0 / 3.0, rtol=0.0, atol=1e-12)
        else:
            assert np.max(np.abs(even - 0.5)) < 1e-12, name
            assert np.max(np.abs(odd - 0.5)) < 1e-12, name


def test_first_moment_condition():
    """sum_j (x_j - X) phi_j = 0 (eq. 14.64) holds for peskin4 and roma3, not for cosine4."""
    rng = np.random.default_rng(2)
    h = 0.03
    X = rng.uniform(5 * h, 95 * h, 10_000)
    # (x_j - X) w_j = (j - s) phi_j, so this sum is the first moment in cells
    for name in ("peskin4", "roma3"):
        kernel = get_kernel(name)
        idx, w = stencil_1d(kernel, X, 0.0, h, 100, periodic=False)
        moment = ((idx * h - X[:, None]) * w).sum(axis=1)
        assert np.max(np.abs(moment)) < 1e-12
    kernel = get_kernel("cosine4")
    idx, w = stencil_1d(kernel, X, 0.0, h, 100, periodic=False)
    moment = ((idx * h - X[:, None]) * w).sum(axis=1)
    assert np.max(np.abs(moment)) > 1e-2                           # about 0.021 at worst


def test_sum_of_squares():
    """sum_j phi(r - j)^2 = 3/8 for peskin4 (eq. 14.65); also 3/8 for cosine4, 1/2 for roma3."""
    rng = np.random.default_rng(3)
    s = rng.uniform(-10, 10, 10_000)
    for name, C in (("peskin4", 3.0 / 8.0), ("cosine4", 3.0 / 8.0), ("roma3", 0.5)):
        kernel = get_kernel(name)
        j = _unwrapped_offsets(kernel, s)
        sumsq = (kernel.phi(j - s[:, None]) ** 2).sum(axis=1)
        assert np.max(np.abs(sumsq - C)) < 1e-12, name


# ---------------------------------------------------------------------------- stencils
def test_stencil_node_selection():
    h = 0.1
    # 4-point: i0 = floor(s), nodes i0-1..i0+2, distances from the unwrapped node
    idx, w = stencil_1d(get_kernel("peskin4"), np.array([0.53]), 0.0, h, 64, periodic=False)
    assert idx.tolist() == [[4, 5, 6, 7]]
    r = (idx[0] * h - 0.53) / h
    assert np.allclose(w[0], get_kernel("peskin4").phi(r) / h)
    # a shifted origin: s = (0.53 - 0.05)/0.1 = 4.8
    idx, _ = stencil_1d(get_kernel("peskin4"), np.array([0.53]), 0.05, h, 64, periodic=False)
    assert idx.tolist() == [[3, 4, 5, 6]]
    # on a node: i0 = 5, the last node sits at distance 2h with zero weight
    idx, w = stencil_1d(get_kernel("peskin4"), np.array([0.5]), 0.0, h, 64, periodic=False)
    assert idx.tolist() == [[4, 5, 6, 7]]
    assert np.allclose(w[0] * h, [0.25, 0.5, 0.25, 0.0])
    # 3-point: i0 = round(s), nodes i0-1..i0+1
    idx, w = stencil_1d(get_kernel("roma3"), np.array([0.53, 0.57]), 0.0, h, 64, periodic=False)
    assert idx.tolist() == [[4, 5, 6], [5, 6, 7]]
    assert np.allclose(w.sum(axis=1) * h, 1.0)
    # exact half: rounds up; the node that differs between rounding rules has zero weight
    idx, w = stencil_1d(get_kernel("roma3"), np.array([0.55]), 0.0, h, 64, periodic=False)
    assert idx.tolist() == [[5, 6, 7]]
    assert w[0, -1] == 0.0 and np.allclose(w[0] * h, [0.5, 0.5, 0.0])
    assert idx.dtype.kind == "i" and w.dtype == np.float64


@pytest.mark.parametrize("name", KERNELS)
def test_periodic_wrap_at_both_edges(name):
    """Near x = 0 and x = L the periodic stencil equals the unwrapped stencil of a shifted
    point (4 cells inward, far from any edge) with its indices shifted back and wrapped."""
    kernel = get_kernel(name)
    rng = np.random.default_rng(4)
    h, n = 0.025, 16
    L = n * h
    left = rng.uniform(0.0, 1.5 * h, 200)              # support crosses x = 0
    right = L - rng.uniform(0.0, 1.5 * h, 200)         # support crosses x = L
    for X, shift in ((left, 4), (right, -4)):
        idx_p, w_p = stencil_1d(kernel, X, 0.0, h, n, periodic=True)
        idx_u, w_u = stencil_1d(kernel, X + shift * h, 0.0, h, n, periodic=False)
        assert np.array_equal(idx_p, (idx_u - shift) % n)
        # compare phi values (w*h): a weight at the edge of the support is tiny, so a relative
        # test there would only measure round-off in the shifted coordinate
        assert np.allclose(w_p * h, w_u * h, rtol=0.0, atol=1e-13)
        assert np.all(idx_p >= 0) and np.all(idx_p < n)
        assert np.max(np.abs(w_p.sum(axis=1) * h - 1.0)) < 1e-12
    # a point past the box edge gives the same stencil as its image inside the box
    X = np.array([-0.3 * h, L + 0.7 * h, 3 * L + 0.2 * h])
    idx_a, w_a = stencil_1d(kernel, X, 0.0, h, n, periodic=True)
    idx_b, w_b = stencil_1d(kernel, np.mod(X, L), 0.0, h, n, periodic=True)
    assert np.array_equal(idx_a, idx_b) and np.allclose(w_a, w_b, rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("name", KERNELS)
def test_nonperiodic_wall_raises(name):
    kernel = get_kernel(name)
    h, n = 0.1, 11                          # nodes at 0, 0.1, ..., 1.0 (like u-faces, Nx + 1)
    msg = "Lagrangian point within kernel support of a non-periodic boundary"
    for bad in (0.0, 0.03, 1.0, 0.96, -0.3, 1.4):
        with pytest.raises(ValueError, match=msg):
            stencil_1d(kernel, np.array([bad]), 0.0, h, n, periodic=False)
    # one bad point among good ones still raises
    with pytest.raises(ValueError, match=msg):
        stencil_1d(kernel, np.array([0.5, 0.6, 0.02]), 0.0, h, n, periodic=False)
    # safely inside: no error, indices in range, weights sum to 1
    idx, w = stencil_1d(kernel, np.array([0.21, 0.5, 0.79]), 0.0, h, n, periodic=False)
    assert idx.min() >= 0 and idx.max() <= n - 1
    assert np.allclose(w.sum(axis=1) * h, 1.0)
    # the same coordinate is fine when the direction is periodic
    stencil_1d(kernel, np.array([0.02]), 0.0, h, n, periodic=True)
    # the exact contract is on the index set: the stencil must not need a node beyond the wall.
    # 4-point: s < 1 or s >= n-2 raises.  3-point: s < 1/2 or s >= n-3/2 raises.
    if kernel.width == 4:
        edge_bad, edge_ok = (0.09, 0.9), (0.1, 0.89)
    else:
        edge_bad, edge_ok = (0.04, 0.97), (0.06, 0.94)
    for bad in edge_bad:
        with pytest.raises(ValueError, match=msg):
            stencil_1d(kernel, np.array([bad]), 0.0, h, n, periodic=False)
    idx, w = stencil_1d(kernel, np.array(edge_ok), 0.0, h, n, periodic=False)
    assert idx.min() >= 0 and idx.max() <= n - 1 and np.allclose(w.sum(axis=1) * h, 1.0)


def test_stencil_input_validation():
    kernel = get_kernel("peskin4")
    with pytest.raises(ValueError):
        stencil_1d(kernel, np.zeros((3, 2)), 0.0, 0.1, 10, True)
    with pytest.raises(ValueError, match="non-finite"):
        stencil_1d(kernel, np.array([0.5, np.nan]), 0.0, 0.1, 10, True)
    with pytest.raises(ValueError):
        stencil_1d(kernel, np.array([0.5]), 0.0, 0.0, 10, True)
    idx, w = stencil_1d(kernel, np.zeros(0), 0.0, 0.1, 10, False)      # no points: no error
    assert idx.shape == (0, 4) and w.shape == (0, 4)


# ---------------------------------------------------------------------------- 2008 MATLAB
def _matlab_cosine_stencil_1d(X, dx, M):
    """Literal transcription of the index/weight arithmetic of forcespread.m (2008) along one
    axis: x1 = ceil(X/dx); nodes x1+1-k, k = 0..3; the if-chain wrap; the cosine weight with
    the unwrapped node. Returns a list of (0-based array index, weight) pairs."""
    x1 = math.ceil(X / dx)
    pairs = []
    for k in range(4):
        ii = x1 + 1 - k
        if ii == -1:
            ii = M - 1
        if ii == -2:
            ii = M - 2
        if ii == -3:
            ii = M - 3
        if ii == M:
            ii = 0
        if ii == M + 1:
            ii = 1
        if ii == M + 2:
            ii = 2
        w = (1 / (4 * dx)) * (1 + math.cos((math.pi / (2 * dx)) * (dx * (x1 + 1 - k) - X)))
        pairs.append((ii, w))
    return pairs


def test_cosine4_reproduces_matlab_2008_stencil():
    """cosine4 with stencil_1d gives the same nodes and weights as the 2008 MATLAB loop."""
    kernel = get_kernel("cosine4")
    M, dx = 16, 0.4 / 16
    rng = np.random.default_rng(5)
    points = [0.01, 0.39, 0.2137, 0.3999, 0.0001, 0.11] + list(rng.uniform(0.0, 0.4, 50))
    for X in points:
        ref = sorted(_matlab_cosine_stencil_1d(X, dx, M))
        idx, w = stencil_1d(kernel, np.array([X]), 0.0, dx, M, periodic=True)
        ours = sorted(zip(idx[0].tolist(), w[0].tolist()))
        assert [i for i, _ in ours] == [i for i, _ in ref], X
        assert np.allclose([v for _, v in ours], [v for _, v in ref], rtol=1e-12, atol=1e-13), X
    # exactly on a node the MATLAB set is {i0+1, i0, i0-1, i0-2} and ours {i0-1 .. i0+2}; the
    # two differ by one node at distance 2 dx, which carries weight exactly zero in both.
    X = 4 * dx
    ref = {i: v for i, v in _matlab_cosine_stencil_1d(X, dx, M) if v != 0.0}
    idx, w = stencil_1d(kernel, np.array([X]), 0.0, dx, M, periodic=True)
    ours = {int(i): float(v) for i, v in zip(idx[0], w[0]) if v != 0.0}
    assert set(ours) == set(ref) and all(abs(ours[i] - ref[i]) < 1e-12 for i in ref)
