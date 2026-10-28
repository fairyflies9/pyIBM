"""Physical invariants, checked without looking at how the code computes things.

Each test states a property of the continuous problem (momentum balance, the energy balance,
Laplace's law, mirror symmetry, Galilean invariance, incompressibility of a closed membrane)
or a property any consistent discretisation must have (convergence under refinement, two
different discretisations of the same problem agreeing), and measures it with quantities we
compute here: box momentum as a sum over the grid, kinetic energy as a sum of squares,
dissipation from our own differences, enclosed area by the shoelace formula, a steady Stokes
solve written out in this file. The package is used only to build a case and to step it.

Numbers quoted in the docstrings were measured with this file on 2026-10-26, and again on
2026-10-27 wherever the package changed them: the blob coupling's default interpolation became
the divergence-free "streamfunction" one (pyibm.coupling.blob), the MAC back end now reports
the rotational pressure, and the driver adds the membrane pressure p_F to the blob pressure.
Slow tests (``pytest -m slow``) carry the finest grids and the Richardson studies.

Set-up shared by most tests: unit box, rho = 1, a closed chain of Hookean springs on an
ellipse with semi-axes 0.2 and 0.125 centred in the box. The spring constant is k = K/ds^2
for a fixed K so the continuum limit does not change when we refine. The rest length is
either 0 (a pressurised membrane, like a constant tension) or the chord of the circle of the
same area (the ellipse relaxes to an unstressed circle). Lagrangian points: about two per cell
along the membrane unless said otherwise. dt is proportional to h when we refine.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyibm.coupling.blob import BlobProjection, get_blob
from pyibm.coupling.delta import DeltaCoupling
from pyibm.driver import Simulation
from pyibm.fluid.periodic_fft import periodic_symbols
from pyibm.grid import Grid, layout_collocated
from pyibm.inputfile import (BCParams, FluidParams, GridParams, OutputParams, Parameters,
                             SolverParams, StructureParams, TimeParams)
from pyibm.io.vtk import mac_to_centres
from pyibm.structures import (Hooks, Springs, Structure, Targets, compute_forces,
                              structure_energy)

# The springs here are stiff on purpose (k = K/ds^2); the driver's tether-number rule of thumb
# fires for them although every run below is stable.
pytestmark = pytest.mark.filterwarnings("ignore:tether number:RuntimeWarning")

PERIODIC = dict(left="periodic", right="periodic", bottom="periodic", top="periodic")
CLOSED = dict(left="dirichlet", right="dirichlet", bottom="dirichlet", top="dirichlet")
A_ELL, B_ELL = 0.2, 0.125


# ====================================================================== helpers

def make_params(N, dt, mu, backend="periodic_fft", scheme="midpoint", coupling="delta",
                bc=None, remove_mean_force=False, blob="gaussian4", blob_delta=4.0,
                advection="central") -> Parameters:
    """Unit box, rho = 1, N x N cells. Central advection on MAC so it adds no dissipation."""
    return Parameters(
        fluid=FluidParams(rho=1.0, mu=mu), time=TimeParams(Tfinal=dt, dt=dt),
        grid=GridParams(Nx=N, Ny=N, Lx=1.0, Ly=1.0),
        solver=SolverParams(backend=backend, time_scheme=scheme, coupling=coupling,
                            kernel="peskin4", remove_mean_force=remove_mean_force, blob=blob,
                            blob_delta=blob_delta, advection=advection),
        bc=BCParams(**(PERIODIC if bc is None else bc)), structure=StructureParams(),
        output=OutputParams(print_dump=10**9, save_vtk=False, save_history=False))


def ellipse(n, cx=0.5, cy=0.5, a=A_ELL, b=B_ELL, c2=0.0) -> np.ndarray:
    """n points at equal parameter steps; c2 adds c2*cos(2 theta) to y (an egg shape)."""
    th = 2 * np.pi * np.arange(n) / n
    return np.column_stack([cx + a * np.cos(th), cy + b * np.sin(th) + c2 * np.cos(2 * th)])


def spring_ring(X, K, rest) -> tuple[float, Springs]:
    """Closed chain of Hookean springs, k = K/ds^2 (fixed continuum limit), ds = mean link."""
    n = len(X)
    i = np.arange(n)
    j = (i + 1) % n
    d = X[j] - X[i]
    ds = float(np.mean(np.hypot(d[:, 0], d[:, 1])))
    return ds, Springs(i, j, np.full(n, K / ds**2), np.full(n, float(rest)))


def relaxing_ellipse(n, K=1.0, rest_zero=False, **kw) -> Structure:
    """Ellipse whose springs rest at the circle of equal area (or at zero length)."""
    X = ellipse(n, **kw)
    rest = 0.0 if rest_zero else 2 * np.sqrt(A_ELL * B_ELL) * np.sin(np.pi / n)
    ds, sp = spring_ring(X, K, rest)
    return Structure("ellipse", X, np.full(n, ds), springs=sp)


def make_sim(params: Parameters, structure: Structure | None) -> Simulation:
    return Simulation(params, ".", hooks=Hooks(), structure=structure, quiet=True)


def polygon_area(X) -> float:
    x, y = X[:, 0], X[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def box_momentum(sim) -> np.ndarray:
    """rho * sum(u) dx dy for both back ends (periodic: every stored value is one cell)."""
    g = sim.grid
    rho = sim.params.fluid.rho
    return rho * g.dx * g.dy * np.array([np.sum(sim.solver.u), np.sum(sim.solver.v)])


def grad_sq_sum(sim, u, v) -> float:
    """sum |grad_h u|^2 (times 1, the dx dy cancels the 1/h^2), by our own differences.

    Periodic: forward differences with wrap (equals -<u, L_h u> by summation by parts).
    Closed box at rest (MAC): differences between neighbouring faces, plus the half-cell
    differences to the no-slip wall, (2 u)^2/2 per wall-adjacent tangential value."""
    g = sim.grid
    if sim.backend == "periodic_fft" or g.periodic_x:
        tot = 0.0
        for a in (u, v):
            tot += np.sum((np.roll(a, -1, 0) - a) ** 2) + np.sum((np.roll(a, -1, 1) - a) ** 2)
        return float(tot)
    tot = (np.sum(np.diff(u, axis=0) ** 2) + np.sum(np.diff(u, axis=1) ** 2)
           + 2 * np.sum(u[:, 0] ** 2) + 2 * np.sum(u[:, -1] ** 2))
    tot += (np.sum(np.diff(v, axis=1) ** 2) + np.sum(np.diff(v, axis=0) ** 2)
            + 2 * np.sum(v[0, :] ** 2) + 2 * np.sum(v[-1, :] ** 2))
    return float(tot)


def max_speed(sim) -> float:
    if sim.backend == "periodic_fft":
        return float(np.max(np.hypot(sim.solver.u, sim.solver.v)))
    uc, vc = mac_to_centres(sim.solver.u, sim.solver.v, sim.grid)
    return float(np.max(np.hypot(uc, vc)))


def rate(e_coarse, e_fine, ratio=2.0) -> float:
    return float(np.log(abs(e_coarse / e_fine)) / np.log(ratio))


# ====================================================================== 1. momentum

def tethered_ring(n=64):
    """A ring of zero-rest-length springs (so it moves) with two points tethered to displaced
    targets, so the structure puts a net force on the fluid."""
    X = ellipse(n, a=0.3, b=0.15)
    ds, sp = spring_ring(X, 1.0, 0.0)
    idx = np.array([0, n // 4])
    tg = Targets(idx, np.full(2, 50.0), X[idx] + np.array([[0.03, 0.01], [0.02, -0.02]]))
    return Structure("ring", X, np.full(n, ds), springs=sp, targets=tg)


MOMENTUM_CASES = [
    # backend, scheme, coupling, remove_mean_force
    ("periodic_fft", "midpoint", "delta", False),
    ("periodic_fft", "midpoint", "delta", True),
    ("periodic_fft", "midpoint", "blob", False),
    ("periodic_fft", "midpoint", "blob", True),
    ("mac", "midpoint", "delta", False),
    ("mac", "pm1996", "delta", False),
    ("periodic_fft", "pm1996", "delta", True),
    ("periodic_fft", "pm1996", "blob", True),
]


@pytest.mark.parametrize("backend,scheme,coupling,remove_mean", MOMENTUM_CASES)
def test_momentum_change_is_dt_times_total_force(backend, scheme, coupling, remove_mean):
    """Doubly periodic box: P^{n+1} - P^n = dt sum(F) hx hy every step, and with
    remove_mean_force P stays constant. Measured residual <= 3.4e-18 against
    dt sum|F| h^2 = 1.2e-2 to 1.5e-2 for every case here (midpoint FFT delta and blob, MAC with
    all sides periodic in both schemes, where the flux-form advection telescopes, and pm1996 FFT
    with remove_mean_force, which holds the box mean although upwinding is not conservative).
    The spread force also carries exactly the Lagrangian total, sum F h^2 = sum f ds (5e-16)."""
    p = make_params(32, 2e-3, 0.02, backend=backend, scheme=scheme, coupling=coupling,
                    remove_mean_force=remove_mean, blob="compact")
    sim = make_sim(p, tethered_ring())
    g = sim.grid
    for _ in range(15):
        P0 = box_momentum(sim)
        sim.step()
        P1 = box_momentum(sim)
        F = np.array([np.sum(sim.Fu), np.sum(sim.Fv)]) * g.dx * g.dy
        scale = sim.dt * (np.sum(np.abs(sim.Fu)) + np.sum(np.abs(sim.Fv))) * g.dx * g.dy
        total_f = np.sum(sim.f * sim.structure.ds[:, None], axis=0)
        assert np.max(np.abs(F - total_f)) < 1e-12 * np.max(np.abs(total_f)) + 1e-14
        assert np.max(np.abs(F)) > 1e-2                  # the test is not trivially zero
        expected = np.zeros(2) if remove_mean else sim.dt * F
        assert np.max(np.abs(P1 - P0 - expected)) < 1e-12 * scale


def test_momentum_pm1996_periodic_upwind_is_not_conservative_but_small():
    """pm1996 on the FFT back end uses non-conservative upwind advection (docs/numerics.md),
    so without remove_mean_force the momentum balance holds only up to the advection defect:
    at most 2.4e-6 per step, 1.5e-4 of dt sum|F| h^2, for delta and 1.3e-6 (1.1e-4) for
    blob."""
    for coupling in ("delta", "blob"):
        p = make_params(32, 2e-3, 0.02, scheme="pm1996", coupling=coupling, blob="compact")
        sim = make_sim(p, tethered_ring())
        g = sim.grid
        worst = 0.0
        for _ in range(15):
            P0 = box_momentum(sim)
            sim.step()
            F = np.array([np.sum(sim.Fu), np.sum(sim.Fv)]) * g.dx * g.dy
            scale = sim.dt * (np.sum(np.abs(sim.Fu)) + np.sum(np.abs(sim.Fv))) * g.dx * g.dy
            worst = max(worst, np.max(np.abs(box_momentum(sim) - P0 - sim.dt * F)) / scale)
        assert worst < 2e-3


# ====================================================================== 2. energy

def energy_run(N, dt, mu, T, backend="periodic_fft", bc=None):
    """Release the stretched ellipse (K = 1) from rest. Returns W0, the balance residual
    (W(T) - W0 + D)/W0 with W = KE + ds E_elastic and D the CN viscous dissipation
    mu dt sum |grad ubar|^2, ubar = (u^n + u^{n+1})/2, the largest one-step change of W/W0,
    and the largest KE/W0 reached."""
    sim = make_sim(make_params(N, dt, mu, backend=backend, bc=bc), relaxing_ellipse(2 * N))
    ds = sim.structure.ds[0]

    def W():
        return sim.solver.kinetic_energy() + ds * structure_energy(sim.structure, sim.grid)

    W0 = W()
    Ws, D, ke = [W0], 0.0, 0.0
    for _ in range(int(round(T / dt))):
        u0, v0 = sim.solver.u.copy(), sim.solver.v.copy()
        sim.step()
        D += mu * dt * grad_sq_sum(sim, 0.5 * (u0 + sim.solver.u), 0.5 * (v0 + sim.solver.v))
        Ws.append(W())
        ke = max(ke, sim.solver.kinetic_energy())
    Ws = np.array(Ws)
    return W0, (Ws[-1] - W0 + D) / W0, float(np.max(np.diff(Ws))) / W0, ke / W0


@pytest.mark.parametrize("backend", ["periodic_fft", "mac"])
@pytest.mark.parametrize("mu", [0.0, 0.01])
def test_energy_balance_midpoint_is_second_order_in_dt(backend, mu):
    """Midpoint scheme, delta coupling, doubly periodic (MAC with all sides periodic):
    KE + elastic energy + viscous dissipation is conserved up to an error that falls as dt^2.
    The kinetic energy reaches 94 % (mu = 0) and 54 % (mu = 0.01) of W0, so this is a real
    exchange. Measured residuals at dt = 4e-3, 2e-3, 1e-3 (32^2, t = 0.5):
      FFT mu=0:    +2.30e-3, +5.50e-4, +1.35e-4 (rates 2.06, 2.03)
      FFT mu=0.01: +1.21e-5, +2.51e-6, +5.59e-7 (rates 2.26, 2.17)
      MAC mu=0:    +2.14e-3, +5.10e-4, +1.24e-4 (rates 2.07, 2.03)
      MAC mu=0.01: +3.57e-6, +4.12e-7, +3.72e-8 (rates 3.1, 3.5)
    With mu = 0 the total creeps up by O(dt^2) (the scheme is not exactly energy
    conserving: X_h is formed with u^n and the advection uses u_h instead of ubar), so
    "non-increasing" holds only with mu > 0, where every single step decreases W."""
    res = [energy_run(32, dt, mu, 0.5, backend=backend) for dt in (4e-3, 2e-3, 1e-3)]
    r = [x[1] for x in res]
    assert min(x[3] for x in res) > (0.9 if mu == 0 else 0.5)
    assert rate(r[0], r[1]) > 1.8 and rate(r[1], r[2]) > 1.8
    assert abs(r[2]) < (3e-4 if mu == 0 else 1e-6)
    if mu > 0:
        assert all(x[2] < 0 for x in res)               # W decreases at every step


def power_ratio(N, coupling, interpolation="streamfunction"):
    """Power the fluid receives, sum F.u h^2, over the power the structure gives,
    sum f ds . U, for the initial stretched ellipse in the steady Stokes flow its own force
    drives (helper steady_stokes below). The delta coupling's spread and interpolation are
    adjoint, so the ratio is 1. The blob projection pairs a blob of width delta with an
    interpolation of the grid velocity; only interpolation="adjoint" has that identity."""
    g = Grid.periodic_box(1.0, 1.0, N, N)
    st = relaxing_ellipse(2 * N)
    f = compute_forces(st, g, 0.0, 1.0)
    lay = layout_collocated(g)
    if coupling == "delta":
        dc = DeltaCoupling("peskin4")
        F, G = dc.spread_vector(f, st.X, st.ds, lay, lay)
        u, v = steady_stokes(F, G, g)
        U = dc.interpolate_vector(u, v, st.X, lay, lay)
    else:
        bp = BlobProjection(g, get_blob(coupling), 4.0 * g.dx, interpolation=interpolation)
        F, G = bp.projected_force(f, st.X, st.ds)
        u, v = steady_stokes(F, G, g)
        U = bp.interpolate(u, v, st.X)
    return float(np.sum(F * u + G * v) * g.dx * g.dy / np.sum(f * U * st.ds[:, None]))


def test_power_identity_delta_exact_blob_first_order():
    """Fluid power / structure power at 32, 64, 128, 256: delta 1 to round-off; with the
    blob's default interpolation ("streamfunction") gaussian4 (4h) 0.64, 0.78, 0.92, 0.97 and
    compact (4h) 0.67, 0.83, 0.92, 0.96, i.e. the mismatch is O(delta/R) and falls about as h
    (with "bicubic": 0.63, 0.77, 0.91, 0.96 and 0.64, 0.82, 0.91, 0.96). This is why
    the blob coupling has no energy balance: released from rest with mu = 0 (32^2, t = 0.5)
    the total energy changes by -22 % (gaussian4) and -12 % (compact) at dt = 2e-3 and 1e-3
    alike, while the delta coupling's error is 2e-3 and falls as dt^2. interpolation="adjoint"
    is the adjoint of the projection, and its ratio is 1 to 6e-16."""
    for N in (32, 128):
        assert abs(power_ratio(N, "delta") - 1.0) < 1e-12
    for blob in ("gaussian4", "compact"):
        r = [power_ratio(N, blob) for N in (32, 64, 128)]
        assert 0.5 < r[0] < r[1] < r[2] < 1.0
        assert rate(1 - r[1], 1 - r[2]) > 0.8
        for N in (32, 128):
            assert abs(power_ratio(N, blob, "adjoint") - 1.0) < 1e-12


def test_energy_balance_mac_closed_box_is_first_order():
    """Same release in a closed box (no-slip walls), mu = 0.01: the residual falls only as
    dt (3.7e-5, 1.7e-5, 8.4e-6; rates 1.1, 1.0), the O(dt) splitting error of the
    non-incremental projection with Dirichlet walls (pyibm.fluid.mac). W still decreases at
    every step."""
    res = [energy_run(32, dt, 0.01, 0.5, backend="mac", bc=CLOSED) for dt in (4e-3, 2e-3, 1e-3)]
    r = [x[1] for x in res]
    assert 0.8 < rate(r[0], r[1]) < 1.6 and 0.8 < rate(r[1], r[2]) < 1.6
    assert 0 < r[2] < 2e-5
    assert all(x[2] < 0 for x in res)


# ====================================================================== 3. Laplace's law

R_LAP, T_LAP, MU_LAP = 0.25, 1.0, 1.0


def pressurised_circle(N, backend, coupling="delta", blob="gaussian4"):
    """Circle of radius R with zero-rest-length springs of tension T = k L ds, fluid at rest,
    two points per cell. Returns the simulation and the discrete Laplace jump: total inward
    force over the circumference, N_pts |f| ds/(2 pi R) (-> T/R)."""
    dt = 0.025 / N
    p = make_params(N, dt, MU_LAP, backend=backend, coupling=coupling, blob=blob,
                    bc=CLOSED if backend == "mac" else None)
    n = int(round(2 * 2 * np.pi * R_LAP * N))
    n += n % 2
    X = ellipse(n, a=R_LAP, b=R_LAP)
    L = 2 * R_LAP * np.sin(np.pi / n)
    ds = 2 * np.pi * R_LAP / n
    i = np.arange(n)
    st = Structure("circle", X, np.full(n, ds),
                   springs=Springs(i, (i + 1) % n, np.full(n, T_LAP / (L * ds)), np.zeros(n)))
    sim = make_sim(p, st)
    f = compute_forces(st, sim.grid, 0.0, dt)
    return sim, float(np.sum(np.hypot(f[:, 0], f[:, 1])) * ds / (2 * np.pi * R_LAP))


def laplace_measure(sim, inner, outer):
    """Mean pressure inside r < inner minus the mean over outer[0] < r < outer[1]."""
    Xg, Yg = sim.grid.meshgrid("nodes" if sim.backend == "periodic_fft" else "centres")
    r = np.hypot(Xg - 0.5, Yg - 0.5)
    p = sim.solver.p
    return float(p[r < inner].mean() - p[(r > outer[0]) & (r < outer[1])].mean())


def laplace_run(N, backend, coupling="delta", T=0.05):
    sim, dp_disc = pressurised_circle(N, backend, coupling)
    A0 = polygon_area(sim.structure.X)
    for _ in range(int(round(T / sim.dt))):
        sim.step()
    jump = laplace_measure(sim, R_LAP / 2, (R_LAP + 0.1, 0.45))
    return (jump - dp_disc) / dp_disc, max_speed(sim), (polygon_area(sim.structure.X) - A0) / A0


@pytest.mark.parametrize("backend", ["periodic_fft", "mac"])
def test_laplace_law_static_membrane(backend):
    """A pressurised circle in fluid at rest (mu = 1, t = 0.05, MAC in a closed box) stays at
    rest up to a spurious flow that falls with h, and the pressure jump between the centre
    (r < R/2) and an annulus away from the membrane (R + 0.1 < r < 0.45) is T/R.
    Measured, 32^2, 64^2 (128^2 in the slow test):
      FFT: jump error -9.5e-4, -1.6e-4 (-2.0e-5); max|u| mu/T 1.7e-3, 1.1e-3 (6.5e-4);
           area drift -2.5e-5, -1.3e-5 (-6.6e-6)
      MAC: jump error -4.6e-5, -1.7e-5 (-1.7e-6); max|u| mu/T 3.7e-4, 2.3e-4 (1.3e-4);
           area drift -1.2e-6, -5.6e-7 (-2.9e-7)
    The spurious speed falls like h^0.6-0.8 and the volume leak like h. (The MAC jump is that
    of the rotational pressure the back end reports, see pyibm.fluid.mac.)"""
    e32, u32, a32 = laplace_run(32, backend)
    e64, u64, a64 = laplace_run(64, backend)
    assert abs(e32) < 3e-3 and abs(e64) < 5e-4 and abs(e64) < abs(e32)
    assert u32 < 3e-3 * T_LAP / MU_LAP and u64 < 0.8 * u32
    assert -1e-4 < a32 < 0 and -1e-4 < a64 < 0 and abs(a64) < 0.7 * abs(a32)


def test_laplace_law_blob_has_almost_no_spurious_flow():
    """With the blob coupling the pressurised circle is at rest to round-off of the force
    quadrature: max|u| mu/T = 1.5e-6 (32^2), 4.3e-6 (64^2) against 1.7e-3 and 1.1e-3 for the
    delta coupling on the same FFT back end, and the area drifts by 7e-11 against 1e-5. The
    regularised force of a uniform tension is an exact gradient, so the projection removes all
    of it (Cortez & Minion 2000)."""
    for N in (32, 64):
        _, u, a = laplace_run(N, "periodic_fft", coupling="blob")
        assert u < 1e-5 * T_LAP / MU_LAP
        assert abs(a) < 1e-7


def regularised_jump(sim, inner, outer, blob="gaussian4", blob_delta=4.0):
    """The Laplace jump the blob coupling should report, as a fraction of T/R: the regularised
    pressure (T/R) chi_R * phi_delta (the indicator of the disc convolved with the blob),
    averaged over the same windows as laplace_measure. The convolution is radial; we tabulate
    it by a midpoint rule in polar coordinates over the disc and interpolate to the nodes."""
    phi, delta = get_blob(blob), blob_delta * sim.grid.dx
    nr = nt = 200
    rho = (np.arange(nr) + 0.5) * R_LAP / nr
    th = (np.arange(nt) + 0.5) * 2 * np.pi / nt
    RHO, TH = np.meshgrid(rho, th, indexing="ij")
    yx, yy, w = RHO * np.cos(TH), RHO * np.sin(TH), RHO * (R_LAP / nr) * (2 * np.pi / nt)
    rr = np.linspace(0.0, 0.5, 251)
    prof = np.array([np.sum(phi.phi(np.hypot(r - yx, yy), delta) * w) for r in rr])
    Xg, Yg = sim.grid.meshgrid("nodes")
    r = np.hypot(Xg - 0.5, Yg - 0.5)
    fr = np.interp(r, rr, prof)
    return float(fr[r < inner].mean() - fr[(r > outer[0]) & (r < outer[1])].mean())


def blob_pressure_jump(N):
    """Jump in solver.p after 5 steps of the pressurised circle with the blob coupling
    (gaussian4, 4h), the discrete T/R, and the regularised jump on the same windows."""
    sim, dp_disc = pressurised_circle(N, "periodic_fft", coupling="blob")
    for _ in range(5):
        sim.step()
    inner, outer = R_LAP / 2, (R_LAP + 0.1, 0.45)
    reg = dp_disc * regularised_jump(sim, inner, outer)
    return laplace_measure(sim, inner, outer), dp_disc, reg


def test_blob_solver_pressure_shows_laplace_jump():
    """The fluid receives only P F~, so the driver adds p_F (L p_F = div F~) to solver.p and
    the blob pressure shows the membrane. It is the regularised pressure, smeared over the
    blob: on 32^2 the blob (delta = 0.125) reaches into both windows and the jump is 4.347,
    8.7 % above the discrete T/R = 3.999 but 2.0e-3 from the regularised jump 4.339; on 64^2
    it is 4.025 (6.3e-3 above T/R, 1.6e-4 from the regularised 4.024); on 128^2 (slow test)
    4.0019 (4.7e-4 above T/R). Before the driver added p_F the jump was 0."""
    jump, dp_disc, reg = blob_pressure_jump(32)
    assert abs(jump - reg) < 5e-3 * reg and jump > 1.05 * dp_disc
    jump, dp_disc, reg = blob_pressure_jump(64)
    assert abs(jump - dp_disc) < 1e-2 * dp_disc and abs(jump - reg) < 1e-3 * reg


@pytest.mark.slow
def test_blob_solver_pressure_shows_laplace_jump_128():
    jump, dp_disc, _ = blob_pressure_jump(128)
    assert abs(jump - dp_disc) < 1e-3 * dp_disc


def test_mac_pressure_is_sharp_next_to_the_membrane():
    """The MAC back end reports the rotational pressure rho (phi - (nu dt/2) L_h phi), so the
    jump stays sharp right next to the membrane. With the windows r < R - 3h and r > R + 3h the
    jump error is -5.5e-5, -8.5e-6, -2.7e-6 on 32^2, 64^2, 128^2, and the pressure inside has
    a standard deviation of 7.5e-5, 5.0e-5, 3.8e-5 of T/R. The FFT back end in the same windows
    gives +1.0e-3, +3.1e-4, +1.3e-4. With p = rho*phi, which the MAC back end reported before,
    the error was -3.3e-3, -5.0e-3, -6.2e-3, growing because dt ~ h makes sqrt(nu dt) grow in
    cells."""
    sim, dp_disc = pressurised_circle(32, "mac")
    for _ in range(int(round(0.05 / sim.dt))):
        sim.step()
    h = sim.grid.dx
    jump = laplace_measure(sim, R_LAP - 3 * h, (R_LAP + 3 * h, 0.45))
    assert abs(jump - dp_disc) < 3e-4 * dp_disc
    Xg, Yg = sim.grid.meshgrid("centres")
    inside = sim.solver.p[np.hypot(Xg - 0.5, Yg - 0.5) < R_LAP - 3 * h]
    assert np.std(inside) < 5e-4 * dp_disc


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["periodic_fft", "mac"])
def test_laplace_law_static_membrane_128(backend):
    """128^2: FFT jump error -2.0e-5, max|u| mu/T 6.5e-4, area drift -6.6e-6;
    MAC -1.7e-6, 1.3e-4, -2.9e-7 (see test_laplace_law_static_membrane)."""
    e64, u64, a64 = laplace_run(64, backend)
    e128, u128, a128 = laplace_run(128, backend)
    assert abs(e128) < abs(e64) and abs(e128) < 1e-4
    assert u128 < 0.8 * u64
    assert abs(a128) < 0.7 * abs(a64)


# ====================================================================== 4. mirror symmetry

def mirror_symmetric_flow(sim, amp=0.2):
    """Discretely divergence-free flow from psi(Lx - x, y) = -psi(x, y), so u is odd and v
    even about x = 1/2; psi vanishes on every wall of the unit box."""
    g = sim.grid

    def psi(x, y):
        return amp * np.sin(2 * np.pi * x) * (np.sin(2 * np.pi * y) + 0.25 * np.sin(4 * np.pi * y))

    if sim.backend == "periodic_fft":
        X, Y = g.meshgrid("nodes")
        P, h = psi(X, Y), g.dx
        sim.solver.u[:] = (np.roll(P, -1, 1) - np.roll(P, 1, 1)) / (2 * h)
        sim.solver.v[:] = -(np.roll(P, -1, 0) - np.roll(P, 1, 0)) / (2 * h)
    else:                                       # psi on the cell corners
        xc = np.arange(g.Nx + 1) * g.dx
        yc = np.arange(g.Ny + 1) * g.dy
        P = psi(*np.meshgrid(xc, yc, indexing="ij"))
        u = (P[:, 1:] - P[:, :-1]) / g.dy
        v = -(P[1:, :] - P[:-1, :]) / g.dx
        sim.solver.u[:] = u[:sim.solver.u.shape[0]]
        sim.solver.v[:] = v[:, :sim.solver.v.shape[1]]


def mirror_defect(sim, perm):
    """Largest |u + u(mirror)|, |v - v(mirror)| over max speed, and |X - mirror X| in cells."""
    g, u, v = sim.grid, sim.solver.u, sim.solver.v
    Nx = g.Nx
    if sim.backend == "periodic_fft" or g.periodic_x:
        um = -u[(Nx - np.arange(Nx)) % Nx]
    else:
        um = -u[::-1]                                       # faces x = i h, i = 0..Nx
    vm = v[(Nx - np.arange(Nx)) % Nx] if sim.backend == "periodic_fft" else v[::-1]
    scale = max(np.max(np.abs(u)), np.max(np.abs(v)))
    X = sim.structure.X
    Xm = X[perm].copy()
    Xm[:, 0] = g.Lx - Xm[:, 0]
    return (max(np.max(np.abs(u - um)), np.max(np.abs(v - vm))) / scale,
            float(np.max(np.abs(g.min_image(X - Xm)))) / g.dx)


SYMMETRY_CASES = [(b, bc, s, c)
                  for b, bc, c in (("periodic_fft", None, "delta"), ("periodic_fft", None, "blob"),
                                   ("mac", PERIODIC, "delta"), ("mac", CLOSED, "delta"))
                  for s in ("midpoint", "pm1996")]


def egg_case(backend, bc, scheme, coupling, n=64, perturb=0.0):
    p = make_params(32, 2e-3, 0.01, backend=backend, bc=bc, scheme=scheme, coupling=coupling,
                    advection="blend")
    X = ellipse(n, cy=0.45, a=0.22, b=0.14, c2=0.04)        # mirror symmetric, not in y
    ds, sp = spring_ring(X, 0.5, 0.0)
    X[3, 0] += perturb
    sim = make_sim(p, Structure("egg", X, np.full(n, ds), springs=sp))
    mirror_symmetric_flow(sim)
    perm = (n // 2 - np.arange(n)) % n                       # theta -> pi - theta
    return sim, perm


@pytest.mark.parametrize("backend,bc,scheme,coupling", SYMMETRY_CASES)
def test_mirror_symmetry_is_kept_to_round_off(backend, bc, scheme, coupling):
    """An egg-shaped ring and a flow that are mirror images of themselves about x = 1/2 stay
    so: after 100 steps (max|u| ~ 1.2-1.5, cfl ~ 0.2) the defect is 3e-15 .. 2e-14 of the
    speed and 2e-14 .. 6e-14 cells in the positions, for both back ends (MAC periodic and
    closed), both schemes, both couplings (blend advection on MAC)."""
    sim, perm = egg_case(backend, bc, scheme, coupling)
    assert max(mirror_defect(sim, perm)) < 1e-13
    for _ in range(100):
        sim.step()
    du, dX = mirror_defect(sim, perm)
    assert du < 1e-12 and dX < 1e-12


def test_mirror_symmetry_defect_detects_a_broken_symmetry():
    """Control: moving one marker by 1e-6 cells breaks the symmetry visibly (the measure is
    not blind)."""
    sim, perm = egg_case("periodic_fft", None, "midpoint", "delta", perturb=1e-6 / 32)
    for _ in range(20):
        sim.step()
    du, dX = mirror_defect(sim, perm)
    assert dX > 1e-7 and du > 1e-9


# ====================================================================== 5. Galilean invariance

def galilean_run(N, coupling, T=0.125, U=(1.0, 0.5)):
    """Run the relaxing ellipse with the fluid at rest (A) and with a uniform velocity U added
    (B), U chosen so that U T is a whole number of cells. Compare B shifted back by U T with
    A: max marker difference in cells, and the L2 and max velocity difference relative to A."""
    h = 1.0 / N
    dt = 0.032 * h
    steps = int(round(T / dt))
    T = steps * dt
    shift = np.round(np.array(U) * T / h).astype(int)
    U = shift * h / T
    sims = []
    for add in (False, True):
        sim = make_sim(make_params(N, dt, 0.01, coupling=coupling), relaxing_ellipse(2 * N))
        if add:
            sim.solver.u[:] += U[0]
            sim.solver.v[:] += U[1]
        sims.append(sim)
    A, B = sims
    for _ in range(steps):
        A.step()
        B.step()
    dX = A.grid.min_image(B.structure.X - U * T - A.structure.X)
    uB = np.roll(B.solver.u - U[0], (-shift[0], -shift[1]), axis=(0, 1))
    vB = np.roll(B.solver.v - U[1], (-shift[0], -shift[1]), axis=(0, 1))
    d = np.hypot(uB - A.solver.u, vB - A.solver.v)
    s = np.hypot(A.solver.u, A.solver.v)
    return (float(np.max(np.abs(dX))) / h, float(np.sqrt(np.mean(d**2) / np.mean(s**2))),
            float(np.max(d) / np.max(s)))


@pytest.mark.parametrize("coupling", ["delta", "blob"])
def test_galilean_invariance_periodic(coupling):
    """Adding U = (1, 0.5) to the initial state of the relaxing ellipse (FFT back end, t =
    0.125) gives the same relative motion up to a spatial discretisation error (halving dt
    changes nothing). Measured on 32^2, 64^2:
      markers, max difference in cells:   delta 0.061, 0.026   blob 0.024, 0.027
      velocity, L2 relative difference:   delta 0.154, 0.058   blob 0.046, 0.024
      velocity, max relative difference:  delta 0.157, 0.080   blob 0.072, 0.024
    In absolute terms the marker difference falls about as h^2 (delta 1.9e-3 -> 4.1e-4; at
    t = 0.25 on 32, 64, 128: 1.5e-3, 3.6e-4, 1.0e-4, while the L2 velocity difference goes
    0.10, 0.024, 0.0084). The largest velocity differences sit within a cell or two of the
    membrane, where the delta kernel's dependence on the sub-cell position of the markers
    shows (it is not translation invariant); there the max difference stalls near 5-7 % for
    the delta coupling on 64^2 and 128^2."""
    d32 = galilean_run(32, coupling)
    d64 = galilean_run(64, coupling)
    assert d32[0] < 0.1 and d64[0] < 0.1
    assert d64[1] < 0.6 * d32[1] and d64[1] < 0.1
    assert d32[2] < 0.25 and d64[2] < 0.12


# ====================================================================== 6. volume

def area_run(N, coupling, T, rest_zero, blob="gaussian4"):
    dt = 0.032 / N
    sim = make_sim(make_params(N, dt, 0.01, coupling=coupling, blob=blob),
                   relaxing_ellipse(4 * N, rest_zero=rest_zero))
    A0 = polygon_area(sim.structure.X)
    for _ in range(int(round(T / dt))):
        sim.step()
    return (polygon_area(sim.structure.X) - A0) / A0


@pytest.mark.parametrize("rest_zero", [True, False], ids=["pressurised", "relaxing"])
def test_area_of_closed_membrane_delta_vs_blob(rest_zero):
    """Relative area change of the ellipse after t = 0.1 (four markers per cell, mu = 0.01),
    delta against blob (gaussian4, delta = 4h), on 32^2 and 64^2. Measured:
      pressurised (rest length 0): delta -8.4e-4, -5.1e-4 (rate 0.7); blob -3.6e-5, -2.8e-6
      relaxing (unstressed circle): delta +4.4e-3, +5.3e-4; blob -2.5e-5, -2.1e-5
    At t = 0.25 (slow test) the pressurised membrane gives delta -9.1e-3, -5.7e-3, -3.4e-3 on
    32, 64, 128 (rates 0.67, 0.75), gaussian4 -1.2e-5, -3.3e-6, +7.8e-7, compact (4h) -1.4e-5,
    +7.5e-6, -1.1e-6; the relaxing one delta +1.6e-3, +6.1e-4, +2.8e-4 (rates 1.4, 1.1),
    gaussian4 -8.2e-5, -2.8e-6, -1.4e-6. For the delta coupling halving dt or doubling the
    markers changes none of these, so they are spatial errors. The blob's default interpolation
    moves the markers with an exactly divergence-free velocity (the curl of an interpolated
    stream function), so it keeps the area 20 to 200 times better than the delta function here,
    although the springs carry tangential forces too. With interpolation="bicubic"
    (pressurised +8.5e-4, +1.0e-3, relaxing -8.0e-5, +8.7e-4 at t = 0.1) the blob is at best
    a few times better than the delta function, because for tangential forces it leaks at
    first order (see the static leak tests)."""
    e = {(c, N): area_run(N, c, 0.1, rest_zero) for c in ("delta", "blob") for N in (32, 64)}
    for v in e.values():
        assert abs(v) < 6e-3
    for N in (32, 64):
        assert abs(e["blob", N]) * 10 < abs(e["delta", N])
    if rest_zero:
        assert e["delta", 64] < 0 and abs(e["delta", 64]) < abs(e["delta", 32])


def equal_arc_ellipse(n, a, b, cx=0.5, cy=0.5) -> np.ndarray:
    """n points equally spaced in arc length on an ellipse."""
    t = np.linspace(0.0, 2 * np.pi, 20001)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(a * np.cos(t)), np.diff(b * np.sin(t))))])
    tk = np.interp(np.arange(n) * s[-1] / n, s, t)
    return np.column_stack([cx + a * np.cos(tk), cy + b * np.sin(tk)])


def uniform_tension_area_run(N, coupling, T=0.25):
    """The tutorial-03 / Cortez & Minion sec. 4.1 set-up: ellipse 0.2 x 0.25, points equally
    spaced in arc length, zero-rest-length springs with k ds^2 = sigma = 0.2 (so the force is
    the curvature force sigma kappa n, no tangential part), nu = 0.005, four points per cell,
    gaussian4 at delta = 4h for the blob. Returns the relative area change at T."""
    dt = 0.032 / N
    X = equal_arc_ellipse(4 * N, 0.2, 0.25)
    ds, sp = spring_ring(X, 0.2, 0.0)
    sim = make_sim(make_params(N, dt, 0.005, coupling=coupling),
                   Structure("ellipse", X, np.full(len(X), ds), springs=sp))
    A0 = polygon_area(X)
    for _ in range(int(round(T / dt))):
        sim.step()
    return (polygon_area(sim.structure.X) - A0) / A0


def test_area_uniform_tension_blob_is_much_better():
    """With a uniform tension (pure normal force) the blob coupling keeps the area far better:
    relative area change at t = 0.25 on 32^2 and 64^2:
      delta     -3.4e-3, -2.1e-3 (rate 0.7)
      gaussian4 +6.6e-6, -1.7e-6 (500 and 1300 times smaller; with interpolation="bicubic"
                it is +3.5e-5 on both grids)
    The same ellipse with semi-axes 0.2 x 0.125: delta -4.4e-3, -3.1e-3, gaussian4 -1.1e-5,
    +1.1e-5, compact (4h) +3.7e-5, +1.2e-5."""
    for N in (32, 64):
        d = uniform_tension_area_run(N, "delta")
        b = uniform_tension_area_run(N, "blob")
        assert -6e-3 < d < -1e-3
        assert abs(b) * 100 < abs(d)


def steady_stokes(F, G, g, mu=1.0):
    """Steady Stokes flow on the periodic node grid, mu L_h u = -P F: the D0 projection of the
    periodic back end, the tight five-point Laplacian, the k = 0 mode dropped."""
    sym = periodic_symbols(g.Nx, g.Ny, g.dx)
    Fh, Gh = np.fft.fft2(F), np.fft.fft2(G)
    q = (sym.s1 * Fh + sym.s2 * Gh) * sym.inv_s_sq
    lam = -mu * sym.lap
    inv = np.zeros_like(lam)
    np.divide(1.0, lam, out=inv, where=lam > 0)
    return (np.real(np.fft.ifft2((Fh - sym.s1 * q) * inv)),
            np.real(np.fft.ifft2((Gh - sym.s2 * q) * inv)))


def steady_stokes_leak(N, coupling, force, blob_delta=4.0, interpolation="streamfunction"):
    """dA/dt of the exact ellipse (4 markers per cell) in the steady Stokes flow (mu = 1)
    driven by a membrane force: 'normal' f = -sigma kappa n (constant tension sigma = 0.2,
    the Cortez & Minion 2000 sec. 4.1 force) or 'tangential' f = 0.2 sin(2 theta) t.
    The grid force comes from the package (spread or blob projection, with the blob's
    ``interpolation`` for the marker velocity); the Stokes solve (mu L_h u = -P F with the
    solver's D0 projection and the tight Laplacian, k = 0 dropped) and dA/dt = sum U.n ds
    (exact arc-length weights) are written here. Returns dA/dt/(A max|U|)."""
    g = Grid.periodic_box(1.0, 1.0, N, N)
    n = 4 * N
    th = 2 * np.pi * np.arange(n) / n
    X = ellipse(n)
    xp, yp = -A_ELL * np.sin(th), B_ELL * np.cos(th)
    xpp, ypp = -A_ELL * np.cos(th), -B_ELL * np.sin(th)
    speed = np.hypot(xp, yp)
    ds = speed * 2 * np.pi / n
    nout = np.column_stack([yp, -xp]) / speed[:, None]
    tang = np.column_stack([xp, yp]) / speed[:, None]
    if force == "normal":
        kappa = (xp * ypp - yp * xpp) / speed**3
        f = -0.2 * kappa[:, None] * nout
    else:
        f = 0.2 * np.sin(2 * th)[:, None] * tang
    lay = layout_collocated(g)
    if coupling == "delta":
        dc = DeltaCoupling("peskin4")
        F, G = dc.spread_vector(f, X, ds, lay, lay)
    else:
        bp = BlobProjection(g, get_blob(coupling), blob_delta * g.dx, interpolation=interpolation)
        F, G = bp.projected_force(f, X, ds)
    u, v = steady_stokes(F, G, g)
    U = dc.interpolate_vector(u, v, X, lay, lay) if coupling == "delta" else bp.interpolate(u, v, X)
    return float(np.sum(np.sum(U * nout, axis=1) * ds)) / (np.pi * A_ELL * B_ELL) / np.max(np.abs(U))


def test_static_leak_normal_force_blob_converges_fast():
    """Leak rate of the exact ellipse in steady Stokes flow under a constant-tension
    (curvature) force, dA/dt/(A max|U|), at 32, 64, 128, 256:
      delta (peskin4):                +2.8e-3, -4.2e-3, -2.8e-3, -1.5e-3 (first order at best)
      gaussian4 (4h), bicubic:        +2.2e-2, +3.9e-3, +3.4e-4, -2.0e-5 (rates 2.5, 3.5, 4.1)
      compact (4h), bicubic:          +1.7e-2, +2.0e-3, +6.8e-5, -5.4e-5
      gaussian4 (4h), streamfunction: -9.7e-4, -1.2e-4, +3.2e-5, -1.3e-6
      compact (4h), streamfunction:   -1.4e-3, -1.2e-4, +4.6e-5, -1.8e-6
    With interpolation="bicubic" the blob is worse than the delta function on 32^2 (the blob
    is as wide as the minor semi-axis), equal on 64^2, 8 times better on 128^2 and 75 times
    better on 256^2: it needs delta well below the radius of curvature. The default
    "streamfunction" interpolation is divergence-free, so only the quadrature of the boundary
    integral is left: it is better than the delta function on every grid, 60 to 90 times on
    128^2 and 870 to 1200 times on 256^2."""
    Ns = (32, 64, 128, 256)
    d = {N: steady_stokes_leak(N, "delta", "normal") for N in Ns}
    b = {N: steady_stokes_leak(N, "gaussian4", "normal", interpolation="bicubic") for N in Ns}
    assert abs(b[32]) > abs(d[32])                                           # coarse: worse
    assert rate(b[64], b[128]) > 2.5 and rate(b[128], b[256]) > 2.5
    assert abs(b[128]) * 5 < abs(d[128]) and abs(b[256]) * 30 < abs(d[256])
    assert abs(d[256]) < abs(d[128]) < abs(d[64])
    for blob in ("gaussian4", "compact"):
        sf = {N: steady_stokes_leak(N, blob, "normal") for N in Ns}
        assert all(abs(sf[N]) < abs(d[N]) for N in Ns)
        assert abs(sf[128]) * 30 < abs(d[128]) and abs(sf[256]) * 300 < abs(d[256])


def test_static_leak_tangential_force():
    """Same measure for a purely tangential force, at 32, 64, 128, 256:
      delta:                     6.3e-2, 4.3e-2, 2.3e-2, 1.2e-2 (about first order)
      gaussian4, bicubic:        1.0e-2, 1.4e-2, 9.9e-3, 5.5e-3 (about first order)
      gaussian4, streamfunction: 5.1e-5, 5.0e-4, -1.1e-4, -5.2e-6
      compact, streamfunction:   2.0e-3, 9.2e-4, -2.1e-4, -1.2e-5
    With interpolation="bicubic" both couplings leak at about first order with similar sizes,
    so for spring networks, which carry tangential forces, that option gives no large gain in
    area conservation. The default divergence-free interpolation leaks 40 to 2300 times less
    than the delta function from 64^2 on."""
    d = [steady_stokes_leak(N, "delta", "tangential") for N in (64, 128, 256)]
    b = [steady_stokes_leak(N, "gaussian4", "tangential", interpolation="bicubic")
         for N in (64, 128, 256)]
    assert 0.5 < rate(d[1], d[2]) < 1.3 and 0.4 < rate(b[1], b[2]) < 1.3
    assert 0.2 < abs(b[2] / d[2]) < 2.0
    for blob in ("gaussian4", "compact"):
        sf = [steady_stokes_leak(N, blob, "tangential") for N in (64, 128, 256)]
        assert all(abs(x) * 20 < abs(y) for x, y in zip(sf, d))


@pytest.mark.slow
def test_area_pressurised_membrane_to_128():
    """t = 0.25: delta -9.1e-3, -5.7e-3, -3.4e-3 on 32, 64, 128 (rates 0.67, 0.75);
    gaussian4 (4h) -1.2e-5, -3.3e-6, +7.8e-7."""
    d = [area_run(N, "delta", 0.25, True) for N in (32, 64, 128)]
    b128 = area_run(128, "blob", 0.25, True)
    assert all(x < 0 for x in d)
    assert 0.4 < rate(d[0], d[1]) < 1.2 and 0.4 < rate(d[1], d[2]) < 1.2
    assert abs(b128) * 100 < abs(d[2])


# ====================================================================== 7. convergence

def ellipse_positions(N, T, coupling="delta", blob="gaussian4", backend="periodic_fft"):
    """Markers of the relaxing ellipse (2N markers, dt = 0.032 h) at time T; every
    (N/16)-th marker is returned so grids can be compared at the same material points."""
    dt = 0.032 / N
    sim = make_sim(make_params(N, dt, 0.01, backend=backend, coupling=coupling, blob=blob),
                   relaxing_ellipse(2 * N))
    for _ in range(int(round(T / dt))):
        sim.step()
    return sim.structure.X[::N // 16].copy()


def richardson(Ns, T, **kw):
    X = [ellipse_positions(N, T, **kw) for N in Ns]
    e = [float(np.max(np.hypot(*(X[i] - X[i + 1]).T))) for i in range(len(X) - 1)]
    return e, [rate(e[i], e[i + 1]) for i in range(len(e) - 1)]


def test_convergence_relaxing_ellipse_delta_coarse():
    """Max marker difference between successive grids at t = 0.5 (displacement 0.039):
    16 -> 32 -> 64: 6.0e-3, 1.8e-3 (rate 1.70, still pre-asymptotic; the slow test continues
    to 128 where the rate is 0.96)."""
    e, r = richardson((16, 32, 64), 0.5)
    assert e[1] < e[0] and 1.0 < r[0] < 2.5


@pytest.mark.slow
def test_convergence_relaxing_ellipse_delta():
    """32 -> 64 -> 128 at t = 0.5: 1.8e-3, 9.4e-4, rate 0.96; continuing to 256 (about 4
    minutes, not run here) gives 4.0e-4 and rate 1.24. About first order, as expected for the
    delta-function IB method."""
    e, r = richardson((32, 64, 128), 0.5)
    assert 0.7 < r[0] < 1.4


@pytest.mark.slow
@pytest.mark.parametrize("blob", ["gaussian4", "compact"])
def test_convergence_relaxing_ellipse_blob(blob):
    """Blob, delta = 4h, max marker difference between successive grids at t = 0.5:
      gaussian4: 4.9e-3, 3.4e-3 (rate 0.52); to 256: 1.1e-3 (rate 1.66)
      compact:   2.4e-3, 1.3e-3 (rate 0.84); to 256: 6.9e-4 (rate 0.96)
    against 1.8e-3, 9.4e-4, 4.0e-4 for the delta coupling (default "streamfunction"
    interpolation). With delta tied to h the blob thickens the membrane by O(delta/R)
    (delta = R/2.5 on 32^2) and the shape relaxes differently (the cos 2 theta amplitude of
    the radius about the centroid at t = 0.5 on 64^2, a least-squares fit over all markers,
    is 1.6e-3 with gaussian4 against 3.6e-3 with delta), so up
    to 256^2 the blob runs are not in the asymptotic range. (Cortez & Minion 2000 report
    rates of 2.5-3 for a curvature force on 256-576 grids; this case has springs, which carry
    tangential forces, and grids up to 256^2.)"""
    e, r = richardson((32, 64, 128), 0.5, coupling="blob", blob=blob)
    assert e[1] < e[0] and r[0] > 0.3


# ====================================================================== 8. back ends agree

def test_backends_agree_on_the_relaxing_ellipse():
    """The same ellipse in the doubly periodic unit box on the FFT back end and on MAC with
    all sides periodic (delta coupling, midpoint, t = 0.2): max marker difference 5.3e-4
    (0.017 h) on 32^2 and 3.9e-4 (0.025 h) on 64^2; 1.5e-4 (0.019 h) on 128^2 in the slow
    test, so about first order (0.93 over the two doublings), against a displacement of
    0.024."""
    for N in (32, 64):
        Xf = ellipse_positions(N, 0.2)
        Xm = ellipse_positions(N, 0.2, backend="mac")
        assert float(np.max(np.hypot(*(Xf - Xm).T))) < 0.05 / N


@pytest.mark.slow
def test_backends_agree_on_the_relaxing_ellipse_128():
    d = {}
    for N in (32, 128):
        d[N] = float(np.max(np.hypot(*(ellipse_positions(N, 0.2)
                                       - ellipse_positions(N, 0.2, backend="mac")).T)))
    assert d[128] < 0.05 / 128 and rate(d[32], d[128], 4.0) > 0.6
