"""The time loop: build a case from its parameters, step it, write diagnostics and output.

A run is

    params = read_input_file("input2d")
    sim = Simulation(params, case_dir)      # grid, fluid solver, coupling, structure, hooks
    sim.step()                              # one step of the chosen scheme, t = n*dt
    sim.output()                            # history row, VTK files, snapshot, on_output hook

and ``run(case_dir)`` is the loop around that (docs/user-guide.md describes the case folder,
docs/numerics.md the schemes). Two time schemes:

- ``midpoint`` (Peskin 2002 section 7, the default on both back ends): move the boundary to
  the half step with the current velocity, evaluate the forces there, advance the fluid with
  the half-step velocity in the advection term, then move the boundary over the full step
  with the half-step velocity. Second order in time.
- ``pm1996`` (Peskin & McQueen 1996 eqs. 14.55-14.61, the 2008 MATLAB code): forces at X^n,
  one first-order fluid step, then move the boundary with u^{n+1}. First order.

Time is ``t = n*dt``, computed, never accumulated. Hooks that prescribe motion are called with
the time at which the forces are evaluated: ``t + dt/2`` for the midpoint scheme and ``t`` for
pm1996. Positions are wrapped into the box in periodic directions after every move.

Mass points carry a ghost ``Y`` with velocity ``V`` outside the fluid, tethered to the fluid
point by ``k (Y - X)``: in the midpoint scheme the ghost is moved to the half step with
``V^n``, the tether force is taken there, and then ``V^{n+1} = V^n + dt a_h``,
``Y^{n+1} = Y^n + dt V_h`` with ``a_h = -k (Y_h - X_h)/m + g`` and ``V_h = V^n + (dt/2) a_h``;
in pm1996 it is one symplectic Euler step (``advance_masses``).

History. Diagnostics are written every ``print_dump`` steps, at step 0, and at the last step of
a run, so the history always ends with the final state. The columns (``HISTORY_COLUMNS``):

- ``step, t``: the step number n and t = n dt of the state the row describes;
- ``t_force``: the time level of ``fx_body, fy_body``. The force in a row is the force the
  fluid received during the last step, which the scheme evaluates at ``t - dt/2`` (midpoint, at
  the half-step positions) or ``t - dt`` (pm1996, at X^{n-1}); row 0 holds the force at the
  initial positions and t_force = 0. The same force is ``Simulation.f`` and the ``force``
  point data of ``lag.NNNNN.vtk``, so in those files the force lags the positions by the same
  half or whole step;
- ``fx_body, fy_body``: the force the fluid puts on the body, ``-sum(F) hx hy`` of the spread
  structure force (springs, beams, targets, mass tethers and the ``lagrangian_force`` hook; the
  ``external_force`` hook is not part of it);
- ``max_target_error_over_dx``: the largest distance between a target and its point, in cells,
  with the targets refreshed to t by ``update_targets`` when that hook exists, so the error
  compares positions at the same instant;
- ``cfl, mean_u, mean_v, kinetic_energy, max_speed`` of the fluid at t.

Pressure. ``solver.p`` (and ``pressure.NNNNN.vtk``) is the solver's pressure. With the blob
coupling the fluid receives only the divergence-free part P F~ of the regularised force
(Cortez & Minion 2000 eq. 11), so the solver's pressure holds no trace of the membrane; after
every step the driver adds p_F with L p_F = div F~ (``BlobProjection.force_pressure``) from the
same force and positions the step used. The blob pressure is therefore the regularised
pressure: a membrane under tension shows its Laplace jump, smeared over the width of the blob
on either side of it.

Checks. ``Simulation`` validates the parameters (``validate_parameters``) whether they come from
an input file or were built in code. Every step ends with a check that the velocity, the
positions and the ghost masses are finite; a blow-up raises ``FloatingPointError`` naming the
step, so a failed run is never reported as a success (the command line then exits with code
1). On the MAC back end every Lagrangian point must stay at least the kernel's half-width (2
cells for peskin4 and cosine4, 1.5 for roma3) from a non-periodic side; a point closer than
that would put part of its force on the wall faces, where the solver overwrites it, so we
raise ``ValueError`` instead. Two warnings: ``cfl > 1`` once per run, and the tether rule of
thumb below.

Tether rule of thumb. Explicit fibre forces limit the time step of the IB method through the
stiffness (Stockie & Wetton 1999 analyse this for a stretched fibre). For target points and
mass tethers we measure the stiffness with the tether number ``k dt^2/(rho dx)`` and warn when
it is above the largest value at which our own test runs stayed stable (``TETHER_STABLE``). The
test: a ring of radius 0.125 in the unit periodic box, 64^2, rho = 1, mu = 0.01, peskin4,
started in a uniform stream u = 1 and held by fixed target points, or by ghost masses of
m = 1 kg/m, every point tethered; k raised at fixed dt. Targets ran 3000 steps at 10 and blew
up within 65 steps at 12, with both schemes and both back ends (ds = dx/2, dt = 2e-3; with
ds = dx or dt = 1e-3 we only checked that 8 is stable and 16 is not). Mass tethers ran 3000
steps at 7 and blew up after 360-690 steps at 8, with both schemes (one pm1996 run with
ds = dx was still stable at 8 after 1000 steps). So the two schemes happen to share the
values. It is a reason to check dt or k, not a stability limit: the limit depends on the
set-up, and we measured it in one (the cylinder wake example runs its targets stably at 3.8
and 7.5; Lai & Peskin 2000, Table I, use 9.95). Springs and beams are not checked: along a chain
the stiffest modes are at the marker scale and barely reach the fluid, so their limits are
much higher (the swimmer example puts the spring limit between 20 and 80).

Output files go to ``<case_dir>/<output_dir>``: ``history.csv``, ``velocity``, ``vorticity``,
``pressure`` and ``lag.NNNNN.vtk``, and ``snapshot.NNNNN.png``. The first output of a
Simulation removes the files of those kinds that an earlier run left in that folder
(``history.csv`` only if its header is a pyIBM history; any other ``history.csv`` makes us
raise when ``save_history`` is on, and is left alone when it is off), whatever ``save_vtk``,
``plot`` and ``save_history`` are set to, so the folder never mixes two runs. A second
``run()`` on the same Simulation continues the run and appends to the history; a step whose
history row was written is not written again, even if its output failed after the row.

References: C. S. Peskin, The immersed boundary method, Acta Numerica 11 (2002) 479-517.
C. S. Peskin and D. M. McQueen, Fluid dynamics of the heart and its valves, chapter 14 of
Othmer, Adler, Lewis and Dallon (eds.), Case Studies in Mathematical Modeling (Prentice Hall,
1996), pp. 313-342. R. Cortez and M. Minion, The blob projection method for immersed boundary
problems, J. Comput. Phys. 161 (2000) 428-453. J. M. Stockie and B. R. Wetton, Analysis of
stiffness in the immersed boundary method and implications for time-stepping schemes,
J. Comput. Phys. 154 (1999) 41-64. M.-C. Lai and C. S. Peskin, An immersed boundary method
with formal second-order accuracy and reduced numerical viscosity, J. Comput. Phys. 160 (2000)
705-719.
"""

from __future__ import annotations

import os
import re
import time
import types
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyibm.coupling.blob import BlobProjection, get_blob
from pyibm.coupling.delta import DeltaCoupling
from pyibm.fluid.mac import BoundaryConditions, MACSolver
from pyibm.fluid.periodic_fft import PeriodicFFTSolver
from pyibm.grid import Grid
from pyibm.inputfile import Parameters, read_input_file, validate_parameters
from pyibm.io import vtk as vtkio
from pyibm.io.history import HistoryWriter
from pyibm.kernels import get_kernel
from pyibm.structures import (Hooks, Structure, advance_masses, compute_forces, load_hooks,
                              mass_tether_forces, read_structure)

__all__ = ["HISTORY_COLUMNS", "TETHER_STABLE", "RunResult", "Simulation", "run",
           "build_grid", "build_solver", "build_coupling"]

HISTORY_COLUMNS = ("step", "t", "t_force", "fx_body", "fy_body", "max_target_error_over_dx",
                   "cfl", "mean_u", "mean_v", "kinetic_energy", "max_speed")

# The history header of earlier pyIBM versions (before t_force), so a rerun into an old output
# folder recognises its history.csv as ours and replaces it.
_EARLIER_HISTORY_COLUMNS = ("step", "t", "fx_body", "fy_body", "max_target_error_over_dx",
                            "cfl", "mean_u", "mean_v", "kinetic_energy", "max_speed")

# Largest tether number k dt^2/(rho dx) at which our test runs stayed stable, per element and
# time scheme (module docstring). Above it the driver warns once. Read-only.
TETHER_STABLE = types.MappingProxyType({
    ("target", "midpoint"): 10.0,
    ("target", "pm1996"): 10.0,
    ("mass tether", "midpoint"): 7.0,
    ("mass tether", "pm1996"): 7.0,
})

# Output files the driver writes, apart from history.csv: removed from the output folder at
# the first output of a Simulation (module docstring).
_OUTPUT_FILE_RE = re.compile(r"^(?:(?:velocity|vorticity|pressure|lag)\.\d{5,}\.vtk"
                             r"|snapshot\.\d{5,}\.png)$")

_UPDATE_HOOKS = ("update_targets", "update_springs", "update_beams", "update_noninv_beams",
                 "update_masses")


# ====================================================================== builders

def build_grid(params: Parameters) -> Grid:
    """The grid the parameters describe: ``Grid.periodic_box`` for the FFT back end,
    otherwise periodic in x when left/right are periodic and in y when bottom/top are."""
    g, bc = params.grid, params.bc
    if params.solver.backend == "periodic_fft":
        return Grid.periodic_box(g.Lx, g.Ly, g.Nx, g.Ny)
    return Grid(g.Lx, g.Ly, g.Nx, g.Ny,
                periodic_x=(bc.left == "periodic"), periodic_y=(bc.bottom == "periodic"))


def build_solver(params: Parameters, grid: Grid, hooks: Hooks | None = None):
    """The fluid back end: ``PeriodicFFTSolver`` or ``MACSolver`` (with the Solver keys
    advection, viscous and mac_projection) whose ``BoundaryConditions`` are wired to the case's
    ``bc_left``, ``bc_right``, ``bc_bottom`` and ``bc_top`` hooks (``None`` means a wall at
    rest)."""
    fl, tm, so, bc = params.fluid, params.time, params.solver, params.bc
    if so.backend == "periodic_fft":
        return PeriodicFFTSolver(grid, fl.rho, fl.mu, tm.dt, remove_mean_force=so.remove_mean_force)
    if so.backend == "mac":
        hooks = hooks or Hooks()
        bcs = BoundaryConditions(bc.left, bc.right, bc.bottom, bc.top,
                                 bc_left=hooks.bc_left, bc_right=hooks.bc_right,
                                 bc_bottom=hooks.bc_bottom, bc_top=hooks.bc_top)
        return MACSolver(grid, fl.rho, fl.mu, tm.dt, bcs, advection=so.advection,
                         viscous=so.viscous, projection=so.mac_projection)
    raise ValueError(f"unknown backend {so.backend!r}")


def build_coupling(params: Parameters, grid: Grid):
    """``DeltaCoupling(kernel)`` or ``BlobProjection(grid, blob, blob_delta*dx, patch,
    interpolation=blob_interpolation)`` (``patch = 0`` in the input file means the automatic
    patch width)."""
    so = params.solver
    if so.coupling == "delta":
        return DeltaCoupling(get_kernel(so.kernel))
    if so.coupling == "blob":
        patch = None if so.patch == 0 else int(so.patch)
        return BlobProjection(grid, get_blob(so.blob), so.blob_delta * grid.dx, patch,
                              interpolation=so.blob_interpolation)
    raise ValueError(f"unknown coupling {so.coupling!r}")


# ====================================================================== result

@dataclass
class RunResult:
    """What ``run`` returns: the history as a record array (columns ``HISTORY_COLUMNS``),
    the final solver (``u``, ``v``, ``p``), structure (positions ``X``, targets, masses),
    grid, parameters and the wall time of the loop in seconds. ``simulation`` is the
    ``Simulation`` itself, with the last Lagrangian force ``f`` and the last spread force
    ``Fu, Fv`` on it (at the time level ``t_force``)."""

    history: np.recarray
    solver: Any
    structure: Structure | None
    grid: Grid
    parameters: Parameters
    wall_time: float
    steps: int
    t: float
    output_dir: str
    simulation: "Simulation" = field(repr=False)


# ====================================================================== the simulation

class Simulation:
    """A case ready to step: grid, fluid solver, coupling, structure and hooks.

    Parameters
    ----------
    params : Parameters
        From ``read_input_file`` or built in code; validated here with
        ``validate_parameters`` either way (ValueError naming the key).
    case_dir : str
        Directory holding the structure files and the hooks file; output goes to
        ``<case_dir>/<output_dir>``.
    hooks : Hooks, optional
        Defaults to ``load_hooks(case_dir, params.hooks.file)``.
    structure : Structure, optional
        Defaults to ``read_structure`` of ``params.structure.name`` in ``case_dir``; a case
        without a structure name is a fluid-only run.
    output_dir : str, optional
        Overrides ``params.output.output_dir``; an absolute path is used as is. The input
        file's output_dir follows the same rules (an absolute path or one with ``..`` is
        honoured), except that it may not be the case directory itself, so the output never
        mixes with the case files.
    quiet : bool
        Suppress the printed diagnostic lines.

    Attributes
    ----------
    n : the number of steps taken; ``t = n*dt``.
    f : the last Lagrangian force density (N, 2) [N/m], None before the first evaluation.
    Fu, Fv : the last spread structure force on the solver's u and v layouts [N/m^2].
    t_force : the time level of f, Fu and Fv (module docstring, History).
    p_F : with the blob coupling, the membrane pressure of the last force evaluation, added to
        ``solver.p`` after every step (at step 0 ``solver.p`` is zero for both couplings);
        None for the delta coupling or a fluid-only run.
    X_prev : the positions the velocities of the damped springs and damped targets are
        formed from, V = (X - X_prev)/dt: the previous half-step positions for the midpoint
        scheme, X^{n-1} for pm1996. In both schemes that velocity is centred about half a step
        before the time at which the force is evaluated.
    """

    def __init__(self, params: Parameters, case_dir: str = ".", hooks: Hooks | None = None,
                 structure: Structure | None = None, output_dir: str | None = None,
                 quiet: bool = False) -> None:
        validate_parameters(params)
        self.params = params
        self.case_dir = os.fspath(case_dir)
        self.quiet = bool(quiet)
        self.dt = float(params.time.dt)
        self.scheme = params.solver.time_scheme
        self.backend = params.solver.backend

        out = params.output.output_dir if output_dir is None else os.fspath(output_dir)
        self.output_dir = os.path.normpath(os.path.join(self.case_dir, out))
        if output_dir is None and (os.path.normcase(os.path.abspath(self.output_dir))
                                   == os.path.normcase(os.path.abspath(self.case_dir))):
            raise ValueError(f"Output_Info.output_dir = {out!r} is the case directory itself, "
                             "so the output would mix with the case files; name a folder "
                             "such as \"viz\"")

        self.n = 0
        self.grid = build_grid(params)
        self.hooks = load_hooks(self.case_dir, params.hooks.file) if hooks is None else hooks
        self.solver = build_solver(params, self.grid, self.hooks)
        self.coupling = build_coupling(params, self.grid)
        self._apply_initial_velocity()
        st = params.structure
        if structure is None and st.name:
            structure = read_structure(self.case_dir, st.name, ds=st.ds, gravity=st.gravity)
        self.structure = structure
        if self.structure is not None:
            self.structure.X = self.grid.wrap(self.structure.X)
            self._check_wall_distance(self.structure.X)

        self.X_prev: np.ndarray | None = None
        self.f: np.ndarray | None = None
        self.Fu = np.zeros(self.solver.layout_u.shape)
        self.Fv = np.zeros(self.solver.layout_v.shape)
        self.t_force = 0.0
        self.p_F: np.ndarray | None = None

        self._writer: HistoryWriter | None = None
        self._rows: list[tuple[float, ...]] = []
        self._last_output_step = -1
        self._cleaned = False
        self._warned_cfl = False
        self._check_tether_number()

    # ------------------------------------------------------------------ time
    @property
    def t(self) -> float:
        """Current time ``n*dt``."""
        return self.n * self.dt

    # ------------------------------------------------------------------ coupling adapters
    def _spread(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lagrangian force density f (N, 2) -> grid force (Fu, Fv) on the solver layouts:
        delta-function spreading (Peskin & McQueen 1996 eq. 14.58) or the blob projection
        P F~ (Cortez & Minion 2000 eqs. 11 and 13). For the blob the membrane pressure p_F
        (L p_F = div F~) comes from the same patch evaluation and is kept in ``self.p_F``."""
        if isinstance(self.coupling, BlobProjection):
            Fx, Fy, self.p_F = self.coupling.projected_force_and_pressure(f, X, ds)
            return Fx, Fy
        return self.coupling.spread_vector(f, X, ds, self.solver.layout_u, self.solver.layout_v)

    def _interp(self, u: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Grid velocity -> Lagrangian velocity (N, 2): the delta kernel (Peskin & McQueen 1996
        eq. 14.61), or for the blob coupling the interpolation chosen by the Solver key
        blob_interpolation (``BlobProjection.interpolate``)."""
        if isinstance(self.coupling, BlobProjection):
            return self.coupling.interpolate(u, v, X)
        return self.coupling.interpolate_vector(u, v, X, self.solver.layout_u, self.solver.layout_v)

    def _add_force_pressure(self) -> None:
        """Blob coupling: solver.p = solver.p + p_F, a new array. The blob runs only on the FFT
        back end, which never reads p back, so this changes the reported pressure only."""
        if self.p_F is not None and self.structure is not None:
            self.solver.p = self.solver.p + self.p_F

    # ------------------------------------------------------------------ hooks
    def _apply_initial_velocity(self) -> None:
        """The ``initial_velocity(grid, layout_u, layout_v) -> (u, v)`` hook, once, at build.

        The arrays are copied into the solver's u and v as given; the first step's projection
        removes any discrete gradient part, so a field that is not divergence-free on the grid
        loses that part in the first step."""
        fn = self.hooks.initial_velocity
        if fn is None:
            return
        lu, lv = self.solver.layout_u, self.solver.layout_v
        u0, v0 = fn(self.grid, lu, lv)
        u0 = np.asarray(u0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        if u0.shape != lu.shape or v0.shape != lv.shape:
            raise ValueError(
                f"initial_velocity must return arrays of shapes {lu.shape} and {lv.shape} "
                f"(the u and v layouts), got {u0.shape} and {v0.shape}")
        if not (np.all(np.isfinite(u0)) and np.all(np.isfinite(v0))):
            raise ValueError("initial_velocity returned non-finite values")
        self.solver.u[...] = u0
        self.solver.v[...] = v0

    def _call_update_hooks(self, t: float) -> None:
        """``update_targets``, ``update_springs``, ... with the time of the force evaluation."""
        s = self.structure
        if s is None:
            return
        for name in _UPDATE_HOOKS:
            fn = getattr(self.hooks, name)
            if fn is not None:
                fn(t, self.dt, s)

    def _add_external_force(self, Fu: np.ndarray, Fv: np.ndarray, t: float
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Add the ``external_force(t, grid, u, v)`` hook, a body force density on the
        solver's u and v layouts, to the spread force."""
        fn = self.hooks.external_force
        if fn is None:
            return Fu, Fv
        Eu, Ev = fn(t, self.grid, self.solver.u, self.solver.v)
        Eu = np.asarray(Eu, dtype=float)
        Ev = np.asarray(Ev, dtype=float)
        if Eu.shape != self.solver.layout_u.shape or Ev.shape != self.solver.layout_v.shape:
            raise ValueError(
                f"external_force must return arrays of shapes {self.solver.layout_u.shape} and "
                f"{self.solver.layout_v.shape} (the u and v layouts), got {Eu.shape} and {Ev.shape}")
        return Fu + Eu, Fv + Ev

    # ------------------------------------------------------------------ forces
    def _structure_force(self, t: float, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hooks at ``t``, then ``f = compute_forces`` at the positions X (and the ghosts)
        plus the ``lagrangian_force(t, dt, s, X)`` hook, and its spread ``(Fu, Fv)``.
        Returns ``(f, Fu, Fv)`` and records them with ``t_force = t``."""
        s = self.structure
        self._call_update_hooks(t)
        f = compute_forces(s, self.grid, t, self.dt, X_prev=self.X_prev, X=X)
        fn = self.hooks.lagrangian_force
        if fn is not None:
            extra = np.asarray(fn(t, self.dt, s, X), dtype=float)
            if extra.shape != f.shape:
                raise ValueError(f"lagrangian_force must return an array of shape {f.shape} "
                                 f"(one force per unit length per point), got {extra.shape}")
            f = f + extra
        Fu, Fv = self._spread(f, X, s.ds)
        self.f, self.Fu, self.Fv, self.t_force = f, Fu, Fv, float(t)
        return f, Fu, Fv

    def _zero_force(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.solver.layout_u.shape), np.zeros(self.solver.layout_v.shape)

    # ------------------------------------------------------------------ one step
    def step(self) -> None:
        """Advance from t = n dt to (n+1) dt with the chosen scheme, then check that the new
        state is finite (``FloatingPointError`` otherwise)."""
        if self.scheme == "midpoint":
            self._step_midpoint()
        else:
            self._step_pm1996()
        self.n += 1
        self._check_finite()

    def _step_midpoint(self) -> None:
        """Peskin 2002 section 7 (eqs. 7.8-7.12):

        1. U = interp(u^n, X^n); X_h = wrap(X^n + (dt/2) U); ghosts Y_h = Y^n + (dt/2) V^n.
        2. hooks with t + dt/2.
        3. f at (X_h, Y_h) plus lagrangian_force; F = spread(f, X_h) (or the blob projection);
           F += external_force.
        4. u_h = half_step(F); full_step(F, u_h); blob: p += p_F.
        5. U_h = interp(u_h, X_h); X^{n+1} = wrap(X^n + dt U_h); ghosts with a_h = -k(Y_h - X_h)/m + g:
           V_h = V^n + (dt/2) a_h, Y^{n+1} = Y^n + dt V_h, V^{n+1} = V^n + dt a_h.
        6. X_prev = X_h.
        """
        s, g, dt, t = self.structure, self.grid, self.dt, self.t
        t_half = t + 0.5 * dt
        if s is not None:
            X_n = s.X.copy()
            U = self._interp(self.solver.u, self.solver.v, X_n)
            X_h = g.wrap(X_n + 0.5 * dt * U)
            if not np.all(np.isfinite(X_h)):
                raise self._blow_up()
            self._check_wall_distance(X_h)
            s.X = X_h
            if s.masses is not None:
                Y_n, V_n = s.masses.Y.copy(), s.masses.V.copy()
                s.masses.Y = Y_n + 0.5 * dt * V_n
            _, Fu, Fv = self._structure_force(t_half, X_h)
        else:
            Fu, Fv = self._zero_force()
            self.t_force = t_half
        Fu, Fv = self._add_external_force(Fu, Fv, t_half)

        u_h, v_h = self.solver.half_step(Fu, Fv, t)
        self.solver.full_step(Fu, Fv, u_h, v_h, t)
        self._add_force_pressure()

        if s is not None:
            U_h = self._interp(u_h, v_h, X_h)
            s.X = g.wrap(X_n + dt * U_h)
            if s.masses is not None:
                m = s.masses
                a_h = -mass_tether_forces(X_h, m, g) / m.m[:, None] + np.asarray(s.gravity)
                V_h = V_n + 0.5 * dt * a_h
                m.Y = Y_n + dt * V_h
                m.V = V_n + dt * a_h
            self.X_prev = X_h
            self._check_wall_distance(s.X)

    def _step_pm1996(self) -> None:
        """Peskin & McQueen 1996 eqs. 14.55-14.61:

        1. hooks with t; f at X^n plus lagrangian_force; F = spread(f, X^n);
           F += external_force.
        2. solver.step(F): upwind advection, backward Euler, projection; blob: p += p_F.
        3. U = interp(u^{n+1}, X^n); X^{n+1} = wrap(X^n + dt U); ghosts by symplectic Euler.

        This is the order of the 2008 MATLAB loop (moveplate, target_force, forcespread,
        fluid_solve, move_boundary), which is why the regression test reproduces it.
        """
        s, g, dt, t = self.structure, self.grid, self.dt, self.t
        if s is not None:
            X_n = s.X
            _, Fu, Fv = self._structure_force(t, X_n)
        else:
            Fu, Fv = self._zero_force()
            self.t_force = t
        Fu, Fv = self._add_external_force(Fu, Fv, t)

        self.solver.step(Fu, Fv, t)
        self._add_force_pressure()

        if s is not None:
            U = self._interp(self.solver.u, self.solver.v, X_n)
            X_new = g.wrap(X_n + dt * U)
            if s.masses is not None:
                advance_masses(s, dt, f_tether=mass_tether_forces(X_n, s.masses, g))
            self.X_prev = X_n.copy()
            s.X = X_new
            self._check_wall_distance(s.X)

    # ------------------------------------------------------------------ checks
    def _check_finite(self) -> None:
        """Raise FloatingPointError if the velocity, the positions or the ghosts are not
        finite: the run has blown up and every later row would be NaN."""
        ok = bool(np.all(np.isfinite(self.solver.u)) and np.all(np.isfinite(self.solver.v)))
        s = self.structure
        if ok and s is not None:
            ok = bool(np.all(np.isfinite(s.X)))
            if ok and s.masses is not None:
                ok = bool(np.all(np.isfinite(s.masses.Y)) and np.all(np.isfinite(s.masses.V)))
        if not ok:
            raise self._blow_up()

    def _blow_up(self) -> FloatingPointError:
        return FloatingPointError(
            f"non-finite solution at step {self.n} (t = {self.t:.6g}): the run has blown up; "
            "reduce dt or the stiffness")

    def _check_wall_distance(self, X: np.ndarray) -> None:
        """MAC back end with the delta coupling: every point at least the kernel's half-width
        (in cells) from each non-periodic side. Closer, the kernel reaches the wall faces,
        whose values the solver overwrites, so part of the force would be lost."""
        if self.backend != "mac" or not isinstance(self.coupling, DeltaCoupling):
            return
        g = self.grid
        reach = float(self.coupling.kernel.half_width)
        X = np.asarray(X, dtype=float)
        if X.size == 0 or not np.all(np.isfinite(X)):
            return                       # a non-finite position is _check_finite's to report
        sides = []
        if not g.periodic_x:
            sides += [("left", X[:, 0] / g.dx), ("right", (g.Lx - X[:, 0]) / g.dx)]
        if not g.periodic_y:
            sides += [("bottom", X[:, 1] / g.dy), ("top", (g.Ly - X[:, 1]) / g.dy)]
        for side, dist in sides:
            k = int(np.argmin(dist))
            if not dist[k] >= reach - 1e-9:
                raise ValueError(
                    f"Lagrangian point {k} at ({X[k, 0]:.6g}, {X[k, 1]:.6g}) is {dist[k]:.3g} "
                    f"cells from the {side} side at step {self.n} (t = {self.t:.6g}); on the "
                    f"MAC back end every point must stay at least {reach:g} cells (the "
                    f"{self.coupling.kernel.name} kernel's half-width) from a non-periodic "
                    "side, or part of its force lands on the wall faces and is lost")

    def _check_tether_number(self) -> None:
        """Warn once if ``k dt^2/(rho dx)`` of the stiffest target or mass tether is above
        ``TETHER_STABLE`` for this time scheme: a rule of thumb (module docstring)."""
        s = self.structure
        if s is None:
            return
        rho, dx = self.params.fluid.rho, self.grid.dx
        for label, el in (("target", s.targets), ("mass tether", s.masses)):
            if el is None or len(el) == 0:
                continue
            k_max = float(np.max(el.k))
            number = k_max * self.dt ** 2 / (rho * dx)
            limit = TETHER_STABLE[(label, self.scheme)]
            if number > limit:
                warnings.warn(
                    f"tether number k dt^2/(rho dx) = {number:.3g} for the stiffest {label} "
                    f"(k = {k_max:.3g}) is above {limit:g}, the largest value at which our "
                    f"{self.scheme} test runs stayed stable; this is a rule of thumb, not a "
                    "stability limit: if the run misbehaves, reduce dt or k",
                    RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------ diagnostics
    def max_target_error_over_dx(self) -> float:
        """Largest distance between a target and its point in cells (0 without targets).

        The targets are refreshed to the current time with ``update_targets`` when that hook
        exists, so a prescribed motion is compared at the same instant as the positions.
        Hooks are functions of absolute time (docs/user-guide.md, hooks), so the extra call
        has no side effect beyond setting ``targets.X``.
        """
        s = self.structure
        if s is None or s.targets is None or len(s.targets) == 0:
            return 0.0
        if self.hooks.update_targets is not None:
            self.hooks.update_targets(self.t, self.dt, s)
        d = self.grid.min_image(s.targets.X - s.X[s.targets.idx])
        return float(np.max(np.hypot(d[:, 0], d[:, 1])) / self.grid.dx)

    def _max_speed(self) -> float:
        if self.backend == "periodic_fft":
            return float(np.max(np.hypot(self.solver.u, self.solver.v)))
        uc, vc = vtkio.mac_to_centres(self.solver.u, self.solver.v, self.grid)
        return float(np.max(np.hypot(uc, vc)))

    def diagnostics(self) -> dict[str, float]:
        """The history columns at the current state (see the module docstring). Raises
        FloatingPointError if the state is not finite."""
        self._check_finite()
        g = self.grid
        cfl = float(self.solver.cfl())
        ke = float(self.solver.kinetic_energy())
        if not (np.isfinite(cfl) and np.isfinite(ke)):
            raise FloatingPointError(
                f"non-finite CFL number or kinetic energy at step {self.n} (t = {self.t:.6g}): "
                "the run has blown up; reduce dt or the stiffness")
        if cfl > 1.0 and not self._warned_cfl:
            self._warned_cfl = True
            warnings.warn(f"CFL number {cfl:.3g} > 1 at step {self.n} (t = {self.t:.6g}); "
                          "reduce dt", RuntimeWarning, stacklevel=2)
        mean_u, mean_v = self.solver.mean_velocity()
        return {
            "step": float(self.n),
            "t": self.t,
            "t_force": float(self.t_force),
            "fx_body": float(-np.sum(self.Fu) * g.dx * g.dy + 0.0),   # + 0.0 drops -0.0
            "fy_body": float(-np.sum(self.Fv) * g.dx * g.dy + 0.0),
            "max_target_error_over_dx": self.max_target_error_over_dx(),
            "cfl": cfl,
            "mean_u": float(mean_u),
            "mean_v": float(mean_v),
            "kinetic_energy": ke,
            "max_speed": self._max_speed(),
        }

    # ------------------------------------------------------------------ output
    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _clean_output_dir(self) -> None:
        """Once per Simulation, before its first output: remove the VTK and PNG files of the
        kinds the driver writes (exact names only) and a pyIBM history.csv left by an earlier
        run. A history.csv that is not a pyIBM history is never deleted: with save_history on
        we raise instead, with it off we leave it alone. The cleanup is marked done only once
        it has run, so a retry after that error cleans the folder."""
        if os.path.isdir(self.output_dir):
            history = os.path.join(self.output_dir, "history.csv")
            ours = os.path.isfile(history) and _is_pyibm_history(history)
            if os.path.isfile(history) and not ours and self.params.output.save_history:
                raise ValueError(
                    f"{history} exists and is not a pyIBM history (its header is not "
                    f"{','.join(HISTORY_COLUMNS)}); move it, or choose another output_dir")
            for name in os.listdir(self.output_dir):
                path = os.path.join(self.output_dir, name)
                if _OUTPUT_FILE_RE.match(name) and os.path.isfile(path):
                    os.remove(path)
            if ours:
                os.remove(history)
        self._cleaned = True

    def _history_writer(self) -> HistoryWriter | None:
        """The CSV writer, opened on first use. The first output of a Simulation starts a
        fresh file (``_clean_output_dir`` has removed the old one); later opens, such as a
        second ``run()``, append to it."""
        if not self.params.output.save_history:
            return None
        if self._writer is None:
            self._ensure_output_dir()
            self._writer = HistoryWriter(os.path.join(self.output_dir, "history.csv"),
                                         HISTORY_COLUMNS)
        return self._writer

    def _write_vtk(self) -> None:
        """Velocity, vorticity and pressure as STRUCTURED_POINTS on the nodes (FFT back end,
        origin 0) or the cell centres (MAC, origin (dx/2, dy/2), u and v averaged there),
        and the Lagrangian points with the spring links and the force per unit length (at
        the time level t_force, see the module docstring)."""
        g, sol, step, out = self.grid, self.solver, self.n, self.output_dir
        spacing = (g.dx, g.dy)
        if self.backend == "periodic_fft":
            origin = (0.0, 0.0)
            u, v = sol.u, sol.v
        else:
            origin = (0.5 * g.dx, 0.5 * g.dy)
            u, v = vtkio.mac_to_centres(sol.u, sol.v, g)
        vtkio.write_structured_vector(vtkio.vtk_filename(out, "velocity", step), "velocity",
                                      u, v, origin, spacing)
        vtkio.write_structured_scalar(vtkio.vtk_filename(out, "vorticity", step), "vorticity",
                                      sol.vorticity(), origin, spacing)
        vtkio.write_structured_scalar(vtkio.vtk_filename(out, "pressure", step), "pressure",
                                      sol.p, origin, spacing)
        s = self.structure
        if s is not None:
            links = []
            if s.springs is not None and len(s.springs):
                links.append(np.column_stack([s.springs.i, s.springs.j]))
            if s.damped_springs is not None and len(s.damped_springs):
                links.append(np.column_stack([s.damped_springs.i, s.damped_springs.j]))
            conn = np.concatenate(links, axis=0).astype(int) if links else None
            scalars = {"force": self.f} if self.f is not None else None
            vtkio.write_lagrangian_points(vtkio.vtk_filename(out, "lag", step), s.X, conn, scalars)

    def _write_snapshot(self) -> None:
        """PNG of the vorticity with the boundary, ``<output_dir>/snapshot.<step:05d>.png``."""
        from pyibm.io.plot import snapshot   # matplotlib, imported only when plotting

        s = self.structure
        location = "nodes" if self.backend == "periodic_fft" else "centres"
        path = os.path.join(self.output_dir, f"snapshot.{self.n:05d}.png")
        snapshot(self.grid, self.solver.vorticity(), X=None if s is None else s.X,
                 targets=None if s is None or s.targets is None else s.targets,
                 path=path, title=f"step {self.n}, t = {self.t:.5g} s", location=location)

    def output(self) -> dict[str, float]:
        """Diagnostics row (printed unless quiet, appended to history.csv), VTK files,
        snapshot, and the ``on_output(step, t, state)`` hook with this Simulation as the
        state. Returns the diagnostics.

        At step 0 with no force yet, we first evaluate the structure force at the initial
        positions, so row 0 holds it whether it comes from ``run()`` or from a call by hand.
        The first call of a Simulation clears the output folder of an earlier run's files
        (module docstring). A step counts as written once its history row is recorded: if the
        VTK files, the snapshot or on_output then raise, a later ``run()`` continues from this
        step without writing the row again."""
        if self.n == 0 and self.f is None and self.structure is not None:
            self._check_finite()          # a NaN position is reported as a blow-up
            self._structure_force(self.t, self.structure.X)
        if not self._cleaned:
            self._clean_output_dir()
        d = self.diagnostics()
        writer = self._history_writer()
        if writer is not None:
            writer.write(d)
        self._rows.append(tuple(d[c] for c in HISTORY_COLUMNS))
        self._last_output_step = self.n
        if self.params.output.save_vtk:
            self._ensure_output_dir()
            self._write_vtk()
        if self.params.output.plot:
            self._ensure_output_dir()
            self._write_snapshot()
        if not self.quiet:
            print(f"step {self.n:7d}  t {d['t']:.6e}  fx {d['fx_body']:+.4e}  fy {d['fy_body']:+.4e}"
                  f"  err/dx {d['max_target_error_over_dx']:.3e}  cfl {d['cfl']:.3f}"
                  f"  KE {d['kinetic_energy']:.4e}  max|u| {d['max_speed']:.4e}", flush=True)
        if self.hooks.on_output is not None:
            self.hooks.on_output(self.n, self.t, self)
        return d

    def history(self) -> np.recarray:
        """The rows written so far as a record array with the ``HISTORY_COLUMNS`` fields."""
        data = np.array(self._rows, dtype=float).reshape(-1, len(HISTORY_COLUMNS))
        return np.rec.fromarrays([data[:, j] for j in range(len(HISTORY_COLUMNS))],
                                 names=list(HISTORY_COLUMNS))

    def close(self) -> None:
        """Close history.csv. A later output (a second ``run()``) reopens it and appends."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # ------------------------------------------------------------------ the loop
    def run(self, steps: int | None = None) -> RunResult:
        """Output at the current step (unless it was already written, as when a second
        ``run()`` continues the first), then ``steps`` steps (default ``Tfinal/dt`` rounded)
        with output every ``print_dump`` steps and at the last step. history.csv is closed
        when the loop ends, also when it ends with an exception. Returns a RunResult whose
        history holds every row of this Simulation, including those of earlier runs."""
        n_steps = self.params.time.n_steps if steps is None else int(steps)
        if n_steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        dump = self.params.output.print_dump
        n_end = self.n + n_steps
        if not self.quiet:
            so, gp = self.params.solver, self.params.grid
            name = self.structure.name if self.structure is not None else "(fluid only)"
            print(f"pyibm: {os.path.abspath(self.case_dir)}  structure {name}  "
                  f"{so.backend}/{so.time_scheme}/{so.coupling}"
                  f"{'/' + so.kernel if so.coupling == 'delta' else '/' + so.blob}  "
                  f"grid {gp.Nx}x{gp.Ny}  dt {self.dt:.6g}  steps {n_steps}  output {self.output_dir}",
                  flush=True)
        start = time.perf_counter()
        try:
            if self._last_output_step != self.n:
                self.output()
            while self.n < n_end:
                self.step()
                if self.n % dump == 0 or self.n == n_end:
                    self.output()
        finally:
            self.close()
        wall = time.perf_counter() - start
        if not self.quiet:
            per = 1e3 * wall / n_steps if n_steps else 0.0
            print(f"pyibm: {n_steps} steps in {wall:.2f} s ({per:.1f} ms/step), t = {self.t:.6g}",
                  flush=True)
        return RunResult(history=self.history(), solver=self.solver, structure=self.structure,
                         grid=self.grid, parameters=self.params, wall_time=wall, steps=n_steps,
                         t=self.t, output_dir=self.output_dir, simulation=self)


def _is_pyibm_history(path: str) -> bool:
    """True if the file is empty or its first line is a pyIBM history header (this version's
    or the one before t_force was added)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\r\n")
    return header in ("", ",".join(HISTORY_COLUMNS), ",".join(_EARLIER_HISTORY_COLUMNS))


# ====================================================================== entry point

def run(case_dir: str, input_file: str = "input2d", steps: int | None = None,
        quiet: bool = False, output_dir: str | None = None) -> RunResult:
    """Run the case in ``case_dir`` with ``<case_dir>/<input_file>``.

    ``steps`` overrides the number of steps (``Tfinal/dt``); output goes to
    ``<case_dir>/<output_dir>`` (``Output_Info { output_dir }`` unless given here); a non-quiet
    run prints one diagnostic line per output. Returns a ``RunResult``. A run that blows up
    raises ``FloatingPointError``.
    """
    case_dir = os.fspath(case_dir)
    path = input_file if os.path.isabs(input_file) else os.path.join(case_dir, input_file)
    with warnings.catch_warnings():
        # Simulation validates the parameters again and repeats this warning; say it once.
        warnings.filterwarnings("ignore", message="the roma3 kernel")
        params = read_input_file(path)
    sim = Simulation(params, case_dir, output_dir=output_dir, quiet=quiet)
    return sim.run(steps)
