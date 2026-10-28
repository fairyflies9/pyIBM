# pyIBM examples

Sixteen cases under `examples/`, each in its own folder with a `README.md` (physics,
parameters, how to run it, what to expect, with the numbers), the input and structure
files, a `generate.py` that writes them, an `analyze.py` that compares the run with the
reference, and a test file `tests/examples/test_<name>.py`. Each one either reproduces a
known answer (an exact solution, a published computation, or the 2008 MATLAB code) or says
plainly that it is a demonstration. The numbers below are copied from the example READMEs,
which give the details and the conditions.

Run any of them from the repository root:

```
python -m pyibm run examples/<name>                       # the default input2d
python -m pyibm run examples/<name> --input input2d_<variant>
python examples/<name>/analyze.py
```

## Summary

"Back end" is `periodic_fft` (doubly periodic, FFT) or `mac` (staggered grid with walls,
inflow, outflow or periodic sides); "coupling" is `delta` (discrete delta function) or
`blob` (blob projection), or none for a fluid-only case. Run times are wall times on a
laptop for the default input unless stated; they vary by 1.5 times or more with what else
the machine is doing.

| example | what it shows | back end, coupling | reference | how well it agrees | run time |
|---|---|---|---|---|---|
| [`heaving_plate_2008`](../examples/heaving_plate_2008) | a plate of target points pushed sideways through very viscous fluid: the case of the 2008 MATLAB code, and the regression anchor | `periodic_fft`, `delta`; `pm1996` with `cosine4` as in 2008, and `midpoint` with `peskin4` (`input2d_midpoint`) | the unmodified 2008 MATLAB code, run to produce `tests/data/*.mat` | velocity, pressure, forces and positions at eight checkpoints to about 1e-15 relative (the test demands 1e-10); the second-order run holds the plate within 0.016 cells | 0.1-0.2 s per input |
| [`taylor_green`](../examples/taylor_green) | the decaying vortex array; orders of accuracy of both back ends | `periodic_fft` and `mac`, fluid only | exact solution | max error 2.1e-4 at 64^2 on both back ends; energy decay rate equal to the grid rate to five digits (0.08 % below the exact rate, the five-point Laplacian's symbol error); spatial order 2.00; temporal order 2.00 (`midpoint`) and 1.0-1.1 (`pm1996`) | 0.55 s (`periodic_fft`), 1.5 s (`mac`); `analyze.py` with all tables 59 s |
| [`lid_driven_cavity`](../examples/lid_driven_cavity) | steady cavity flow at Re 100, 400 and 1000; an oscillating lid as a demonstration | `mac`, central advection, fluid only | Ghia, Ghia & Shin 1982, Tables I and II | largest difference on the u / v centrelines in lid units: 0.0038 / 0.0087 (Re 100, 64^2), 0.0021 / 0.0048 (Re 400, 128^2), 0.0033 / 0.0119 (Re 1000, 128^2) | 13 s (Re 100), 4 min (Re 400), 7 min (Re 1000), 25 s (oscillating) |
| [`stokes_second_problem`](../examples/stokes_second_problem) | a wall sliding back and forth under fluid at rest: the viscous term and time-varying wall data | `mac`, fluid only | exact solution (Stokes 1851) | largest error 1.92e-2, 5.48e-3, 1.45e-3 of the wall speed at 3.2, 6.4, 12.8 cells per penetration depth: second order | 2.1-3.6 s per input |
| [`womersley_channel`](../examples/womersley_channel) | oscillating pressure-driven flow in a channel, driven through an inflow/outflow pair or a body force | `mac`, fluid only | exact channel solution (the plane version of Womersley 1955) | alpha 3 with inflow/outflow: 1.34e-2 of the core speed (6.7e-3 away from the wall cells), 6.7e-3 (3.4e-3) with the incremental projection; periodic body-force runs 6.9e-3; the pressure gradient within 0.50 % of the forcing (0.19 % incremental) | 2 s (alpha 3) to 25 s (alpha 10); all five inputs about 50 s |
| [`cylinder_wake`](../examples/cylinder_wake) | the vortex street behind a ring of target points at Re 100 and 200 | `mac`, `delta` (`peskin4`), targets | Lai & Peskin 2000, Tables I and II | Re 100 on D/dx = 12: St 0.1765 against 0.165 (+7.0 %), mean C_D 1.5705 against 1.4473 (+8.5 %), lift amplitude 0.3603 against 0.3299 (+9.2 %); Re 200: St 0.2042 against 0.190 (+7.5 %). Forces converged in the tether stiffness; about half of the excess at D/dx = 12 comes from the narrow box. Within about 10 %, not a converged reproduction | 1.6 min (D/dx = 8), 8 min (D/dx = 12, each Re), 18 min (D/dx = 16) |
| [`flapping_filament`](../examples/flapping_filament) | a heavy flexible filament flapping like a flag in a uniform stream: springs, beams, mass points and a pinned end | `mac`, `delta` | Huang, Shin & Sung 2007, Fig. 13 | fine run (32 segments) over 10 < t < 15: period 3.26 against 3.23 (1 %), amplitude 0.35 against 0.29 and still growing at t = 15, so not converged; the coarse run (16 segments) overshoots, 0.46 and 3.74 against 0.33 and 3.18 over 20 < t < 30 | coarse 5.5-8 min to t = 30; fine 63 min to t = 15 (about 2.1 h to t = 30) |
| [`flapping_plate_re64`](../examples/flapping_plate_re64) | a rigid plate hovering with the tilted hybrid stroke at Re 64 | `periodic_fft` (`remove_mean_force`) and `mac`, `delta`, targets | Jones et al. 2015, Figs. 11A, 12A and 21 | cycle-mean vertical force coefficient 1.05 (`mac`) and 1.09 (`periodic_fft`) against 1.02 +- 0.03 interpolated to Re 64; C_V/C_T 0.268 and 0.275 against 0.29; the time series has the paper's shape; the plate lags its targets by up to 0.13 cells | 3.5 min (`periodic_fft`), 17 min (`mac`), machine shared |
| [`clap_and_fling`](../examples/clap_and_fling) | two wings rotating and translating apart (the fling) at Re 8 and 128 | `periodic_fft` and `mac`, `delta`, damped targets at Re 128 | Miller & Peskin 2005, Figs. 9 and 11 | landmarks within 0.03 of the stroke of where the paper has them; Re 128 peaks during rotation 36-42 % high, troughs within 20 %, end values within 11 %; Re 8 within 12 % except the drag rotation peak (+44 %). The gap between the wings is 4.3 cells here against 10 in the paper, which is why the rotation peaks are high | about 4 min per periodic input, 21 min (`mac`), machine shared |
| [`jellyfish_bell`](../examples/jellyfish_bell) | a bell of springs and beams squeezed by muscle springs at and near its natural frequency | `periodic_fft`, `delta` | Hoover & Miller 2015, Figs. 2, 6 and 7 | free-vibration frequency 0.84 against 0.86 1/s (the stiffness is calibrated to it, times 430; the force times 118); amplitude of the bell oscillation 0.25 against 0.24 at f = 0.5 and 1.2-1.5 times the paper's from f = 0.75 to 1.25; the bell does not swim at this resolution (net drift -0.025 L/s over ten cycles, against the paper's 0.31 L/s forward by the fifth cycle) | 5.2 min (ten cycles), 2.2-4.7 min for the other frequencies, machine shared |
| [`heart_tube`](../examples/heart_tube) | valveless pumping in a loop of tubing pinched near one end | `periodic_fft` (`remove_mean_force`), `delta`, targets | Jung & Peskin 2001, Figs. 3 and 8 | with the paper's scheme and time step, the net flow over the last five of 20 cycles is +0.84 cm^2/s against +0.6 (T = 0.325 s), -1.16 against -1.6 (T = 0.21 s) and -0.015 against about 0 (T = 1.34 s): the paper's directions, magnitudes within 40 %; the midpoint runs are still spinning up after 20 cycles | 64-96 s for 20 cycles at T = 0.325 and 0.21 s, about 3 min at T = 1.34 s, machine shared |
| [`swimmer`](../examples/swimmer) | a free filament that swims by a travelling curvature wave set through `update_beams` | `periodic_fft`, `delta` | demonstration; kinematics of Carling, Williams & Bowtell 1998 | swims head first at about 0.09 of the wave speed (fine grid) and 0.03 (coarse), following the prescribed curvature to 5 % RMS; measured stability limit of the springs between 20 and 80 in the tether number | 103 s (fine), 31 s (coarse) |
| [`sedimenting_ring`](../examples/sedimenting_ring) | an elastic ring with mass points falling under gravity in a closed box | `mac`, `delta`, mass points | demonstration (internal consistency) | the fluid carries the ring's weight, `fy_body/(M g)` = 0.9999; area drift 9.5e-4; a rigid circle dragged at the same speed feels 0.83 of the ring's weight | 2.8 min (fine, machine shared), 36 s (coarse), 99 s (rigid reference) |
| [`rubberband`](../examples/rubberband) | an ellipse of springs relaxing to a circle, with both couplings | `periodic_fft`, `delta` and `blob` | demonstration | relative area change at t = 2 s: -5.7e-4 (delta), +2.8e-5 (blob); the difference comes from the marker interpolation (see the README) | 2.3 s (delta), 5.4 s (blob) |
| [`blob_ellipse`](../examples/blob_ellipse) | the perturbed ellipse under constant tension | `periodic_fft`, `blob` and `delta` | Cortez & Minion 2000, section 4.1, Figs. 4-8 | relative area error at t = 1: +2.5e-6 (largest 5.7e-6), inside the paper's +-1e-5; delta -6.9e-3 (the paper's IB run with the same projection: -1.4e-2) | 33 s (blob), 5.4 s (delta) |
| [`forced_circle`](../examples/forced_circle) | a circle whose right half is pulled along a prescribed ellipse; the error trend with the time step | `periodic_fft` (`remove_mean_force`), `delta` and `blob` | Fovargue 2012, section 5.4.0.1, Fig. 5.6 (trend only: the membrane model and stiffnesses are our own) | from dt = dx/8 on the velocity error falls at orders 1.5-2.3 for both couplings, the second order of the midpoint scheme; her errors are far smaller (her scheme is fourth order, ours second) | 10 s (delta), 50 s (blob); the time-step study 8.4 min |

## Notes for reading the numbers

- **Forces have their own time level.** A history row holds the force of the last step,
  evaluated at `t_force` (`t - dt/2` for the midpoint scheme, `t - dt` for `pm1996`), not at
  the row's `t`. Compare forces at `t_force`. In `heaving_plate_2008`, a `pm1996` run and a
  midpoint run differ row for row by 1.9 % of the peak force, and at the same `t_force` by
  0.49 %, which is the first-order error of `pm1996`.
- **The MAC pressure belongs to the middle of the step.** It comes from the projection of the
  step that ends at `t`. In `womersley_channel`, a least-squares fit of `-dp/dx` to
  `K cos(omega (t - lag))` gives a lag of 0.64 `dt` (non-incremental) and 0.56 `dt`
  (incremental) with the midpoint scheme; `analyze.py` compares it at `t - dt/2`.
- **Target lag.** Examples with target points report the stiffness next to the largest
  target error in cells and say so when it is above 0.1 cells.
- **Variants.** Most examples have several inputs in one folder (`input2d_<variant>`), with
  output in `viz/<variant>/`. `examples/lid_driven_cavity/input2d_oscillating` uses its own
  hooks file through `Hooks { file = "hooks_oscillating.py" }`:
  `python -m pyibm run examples/lid_driven_cavity --input input2d_oscillating`.
- **Tests.** Every example has smoke runs in the default test suite and its full
  comparison as a slow test (`python -m pytest -m slow tests/examples/test_<name>.py`).

## References

- Carling, J., Williams, T. L. and Bowtell, G. (1998). Self-propelled anguilliform swimming:
  simultaneous solution of the two-dimensional Navier-Stokes equations and Newton's laws of
  motion. Journal of Experimental Biology 201, 3143-3166.
- Cortez, R. and Minion, M. (2000). The blob projection method for immersed boundary
  problems. Journal of Computational Physics 161, 428-453.
- Fovargue, L. (2012). Addressing the computational cost of the immersed boundary method
  through multi-implicit and multi-rate strategies with time parallelism. PhD thesis,
  University of North Carolina at Chapel Hill.
- Ghia, U., Ghia, K. N. and Shin, C. T. (1982). High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics
  48, 387-411.
- Hoover, A. and Miller, L. (2015). A numerical study of the benefits of driving jellyfish
  bells at their natural frequency. Journal of Theoretical Biology 374, 13-25.
- Huang, W.-X., Shin, S. J. and Sung, H. J. (2007). Simulation of flexible filaments in a
  uniform flow by the immersed boundary method. Journal of Computational Physics 226,
  2206-2228.
- Jones, S. K., Laurenza, R., Hedrick, T. L., Griffith, B. E. and Miller, L. A. (2015). Lift
  vs. drag based mechanisms for vertical force production in the smallest flying insects.
  Journal of Theoretical Biology 384, 105-120.
- Jung, E. and Peskin, C. S. (2001). Two-dimensional simulations of valveless pumping using
  the immersed boundary method. SIAM Journal on Scientific Computing 23, 19-45.
- Lai, M.-C. and Peskin, C. S. (2000). An immersed boundary method with formal second-order
  accuracy and reduced numerical viscosity. Journal of Computational Physics 160, 705-719.
- Miller, L. A. and Peskin, C. S. (2005). A computational fluid dynamics of 'clap and fling'
  in the smallest insects. Journal of Experimental Biology 208, 195-212.
- Miller, L. A. and Peskin, C. S. (2009). Flexible clap and fling in tiny insect flight.
  Journal of Experimental Biology 212, 3076-3090.
- Peskin, C. S. and McQueen, D. M. (1996). Fluid dynamics of the heart and its valves. In
  Othmer, H. G., Adler, F. R., Lewis, M. A. and Dallon, J. C. (eds.), Case Studies in
  Mathematical Modeling: Ecology, Physiology, and Cell Biology, chapter 14, pp. 313-342.
  Prentice Hall.
- Seibold, B. (2008). A compact and fast Matlab code solving the incompressible Navier-Stokes
  equations on rectangular domains. Course notes for MIT 18.086,
  http://www-math.mit.edu/18086.
- Stokes, G. G. (1851). On the effect of the internal friction of fluids on the motion of
  pendulums. Transactions of the Cambridge Philosophical Society 9, part II, 8-106.
- Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow and viscous
  drag in arteries when the pressure gradient is known. Journal of Physiology 127, 553-563.
