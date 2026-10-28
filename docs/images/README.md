# Pictures for the project page

Every picture here comes from `make_images.py` in this folder:

    python docs/images/make_images.py --jobs 17

It runs each example listed below from its committed input file with `pyibm.Simulation`
(nothing in the input is changed; the history and any VTK output go to a scratch folder,
`--work`, by default `pyibm-images` in the system temp folder), saves the vorticity, the
velocity and the structure positions at the listed times, and then draws the PNG files. With
`--draw-only` it draws again from the saved frames. On our 24-core laptop the whole script
took 5.5 minutes with all 17 runs at once, so no single run took longer. Each run stops at
its last listed time; the lines below say where that is short of the input's own end time.

Common to all field pictures: vorticity is the solver's own (`solver.vorticity()`, on the
grid nodes for the FFT back end and the cell centres for MAC), drawn with bilinear
interpolation between the grid values on a dark diverging colour map whose zero is the
background colour (blue clockwise, red to gold counterclockwise). The colour limits are
fixed per picture and the strongest vorticity, next to the structures, is saturated on
purpose. Structures are the Lagrangian points joined in index order, drawn as light lines (1 to 1.8 pt on the
example cards, 3 to 3.6 pt on the banner and the social preview, so the bodies stand out) (a gap between consecutive points of more than 2.5 times the median spacing starts a
new line, which separates the bodies, and a line whose two ends are that close is closed).
"t" is the simulated time and "step" the time step it was saved at; the windows and colour
limits of every card are in `make_images.py`.

- `hero.png` (1800 x 640): four vorticity panels. Cylinder wake, `cylinder_wake/input2d`, Re
  100, t = 30 (step 9600; the run stops at t = 32 of the input's 100), window x 1.83 to 15.97,
  y 1.95 to 6.05, limits +-3. Clap and fling, `clap_and_fling/input2d`, Re 128, the end of
  the fling, tau = 6 (t = 5 s, step 2500), window x 0.19 to 0.81, y 0.43 to 0.61, limits +-6.
  Flapping plate, `flapping_plate_re64/input2d`, mid-upstroke of the third cycle (t = 0.08999
  s, step 17600), window x 0.376 to 0.864, y 0.07 to 0.87, limits +-900. Jellyfish bell,
  `jellyfish_bell/input2d_fine` run for two of its ten cycles only, the expanded bell at
  t = 2.03 s (step 20349), window x 1.25 to 2.75, y 0.77 to 3.23, limits +-9.
- `social-preview.png` (1280 x 640): "pyIBM" and the tagline above the same cylinder wake,
  flapping plate and jellyfish frames and limits as the banner, in the windows x 2.53 to
  11.28, y 1.5 to 6.5 (cylinder), x 0.38 to 0.86, y 0.07 to 0.87 (plate) and x 1.22 to 2.78,
  y 0.7 to 3.3 (jellyfish).

Example cards in `examples/`, 640 x 400 each:

- `blob_ellipse.png`: the perturbed ellipse with the blob projection (`blob_ellipse/input2d`,
  128 x 128) at t = 0.2 and 0.6 s, the starting shape faint; limits +-6.
- `clap_and_fling.png`: both wings at tau = 2 (t = 1.666 s, step 833, tau = 1.9992) and at
  tau = 6 (t = 5 s, step 2500), `clap_and_fling/input2d`, Re 128; limits +-6.
- `cylinder_wake.png`: the vortex street at t = 30 (step 9600; the run stops at t = 32 of
  the input's 100), `cylinder_wake/input2d`, Re 100, x from 1.6 to 15.6 D, the full 8 D
  height; limits +-3.
- `flapping_filament.png`: `flapping_filament/input2d_coarse` (96 x 64, 16 segments) at t =
  26 (step 32500; the run stops there, the input goes to 30), the filament at t = 22, 23, 24
  and 25 faint; limits +-6. We use the coarse input because the default one is far too slow
  to reach the flapping state within a few minutes (its README gives the timings).
- `flapping_plate_re64.png`: `flapping_plate_re64/input2d` at mid-downstroke (t = 0.07363 s,
  step 14400) and mid-upstroke (t = 0.08999 s, step 17600) of the third cycle; the dashed line
  joins the plate centres at the two ends of the stroke (steps 16000 and 19200); limits +-600.
- `forced_circle.png`: `forced_circle/input2d` (delta coupling, 128 x 128) at t = 0.5 and
  1.0 s, the starting circle faint; limits +-40.
- `heart_tube.png`: the speed |u| of `heart_tube/input2d` at t = 6.5 s (step 6660, the end
  of the 20th drive cycle) on the warm half of the colour map, 0 to 12 cm/s. We show the
  speed because the vorticity of this run is dominated by grid-scale structure next to the
  tethered walls, which hides the flow.
- `heaving_plate_2008.png`: `heaving_plate_2008/input2d` at step 100 (t = 0.0417 s), one
  colour square per grid node (no interpolation) and the 16 plate points; limits +- the
  largest |vorticity| of that step.
- `jellyfish_bell.png`: `jellyfish_bell/input2d_fine` (h = L/32), run for two of its ten
  cycles only, the bell contracted at 1.25 cycles (t = 1.45 s, step 14535) and expanded at
  1.75 cycles (t = 2.03 s, step 20349); limits +-4.
- `lid_driven_cavity.png`: vorticity and streamlines of `lid_driven_cavity/input2d` (Re 100,
  64 x 64) at t = 18 s (step 2880) and `input2d_re1000` (Re 1000, 128 x 128) at t = 70 s
  (step 22400), the end of each input; limits +-5; the gold edge is the moving lid.
- `rubberband.png`: `rubberband/input2d` (delta coupling) at t = 0.1 and 1 s, the starting
  ellipse faint; limits +-1.
- `sedimenting_ring.png`: the whole box of `sedimenting_ring/input2d` at t = 1, 2 and 3 s
  (steps 4000, 8000, 12000), the starting ring faint; limits +-6.
- `stokes_second_problem.png`: velocity profiles of `stokes_second_problem/input2d` at the
  eight phases t = 4, 4.125, ..., 4.875 s (steps 1600 to 1950), the cell-centre u averaged
  over the 8 periodic columns (dots, y up to 5 d), against `stokes_exact.velocity_finite_depth`
  (lines).
- `swimmer.png`: `swimmer/input2d` at t = 4 s (step 26667), window x 0.85 to 2.95, y 0.3 to
  1.2 around the body; limits +-8.
- `taylor_green.png`: vorticity and streamlines of `taylor_green/input2d` at t = 0 and 0.5 s
  (step 250); both panels share the limits +- the largest |vorticity| at t = 0, so the decay
  shows as fading colour.
- `womersley_channel.png`: velocity profiles of `womersley_channel/input2d` at t = 3, 3.125,
  ..., 3.875 s (steps 1200 to 1550), the mean of the two cell-centre columns either side of
  x = Lx/2 (dots), against `womersley_exact.velocity` with H = 0.6 (lines).
