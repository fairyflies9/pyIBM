# Regression reference data

`heaving_plate_2008_stiff1.mat` and `heaving_plate_2008_stiff1e4.mat` were produced by
`generate_matlab_reference.m` in MATLAB R2024a on 2026-10-26. The script runs the original,
unmodified 2008 MATLAB functions in `matlab-original/move-plate-example/` (parameters from
`in_fluid.m`: 16 x 16 grid on a 0.4 m box, mu = 3, rho = 2, dt = dx^2/nu, cosine kernel,
Peskin & McQueen 1996 first-order scheme) for 100 steps with the target stiffness given in
the file name, plotting removed, and saves the full state after steps 1, 2, 3, 5, 10, 25, 50
and 100.

Each file holds (load with `scipy.io.loadmat(path, simplify_cells=True)`):

- `params`: M, N, Q, dx, ds, dt, nu, mu, rho, length, width, time_final, stiff, nsteps, save_steps
- `X0`: initial plate points (Q x 2)
- `a, b, c, d`: the Fourier symbol arrays built by `moving_plate.m` lines 72-96
- `step_NNNN`: n, time (after the step), u, v, press, f1, f2, w1, w2 (all M x N, indexed
  [i, j] with i along x), X (plate after the move), XT (targets used in that step), fb
  (target forces per unit length used in that step)

The pyIBM regression test (`tests/test_matlab_regression.py`) runs the `pm1996` scheme with
the `cosine4` kernel on the same inputs and compares u, v, X, the pressure and the forces at
every checkpoint to round-off.
`stiff = 1` is the value in the 2008 code. With it the plate hardly moves, so pyIBM also
includes a run with `stiff = 1e4`, a stiffer tether that keeps the plate on its prescribed
path.
