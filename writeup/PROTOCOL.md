# Pre-declared experiment protocol

Committed BEFORE running the experiments it governs. Any deviation found
necessary during execution will be recorded here in a "Deviations" section
with the reason, and reported in the writeup — including outright failures.

## Shared configuration

- Physics: the Track 02 cold plate, oil coolant, `ColdPlate(nx=32, ny=16,
  fluid="oil")`, all defaults (`pin = 2e4 Pa`, `t_in = 285 K`,
  `q_peak = 1.5e8 W/m^3` as the amplitude scale). Design `gamma`: the density
  filter applied to a fixed pseudo-random field, `rng(seed).uniform(0.2, 0.8)`
  — a generic partially blocked plate, not an optimized design.
- Camera: sensor 96x64, homography `default_homography(tilt=0.25)`,
  `t_ambient = 295 K`, `half_fov_tan = 0.45`. True camera parameters:
  `psf_sigma = 1.2 px`, `gain = 25 counts/(W m^-2 sr^-1)`, `offset = 500
  counts`. True emissivity: base 0.95 with a rectangular low-e patch (0.75)
  covering x in [0.55, 0.85], y in [0.25, 0.75] fractions of the plate.
- Noise model for synthetic measurements: additive iid Gaussian in counts,
  applied once to the measured image, never inside the gradient path.
  At gain 25 and dL/dT ~ 0.9 (W m^-2 sr^-1)/K near 300 K, 1 count of noise
  is roughly 0.045 K of NETD — sigma_noise = 2 counts is a realistic
  uncooled-microbolometer noise floor.
- Fixed-point tolerance 1e-9 during optimization, 1e-11 for verification and
  final re-scoring. Adjoint GMRES rtol 1e-8.
- Seeds: measurement noise seed 42; every rng that shapes the problem (gamma,
  eps pattern) seed 0. No seed shopping: these were chosen before any run.

## Gate G0 — end-to-end gradient check (precondition for everything)

`scripts/verify_e2e_gradient.py --nx 16 --ny 8`, seed 0: central finite
differences of the full pixels -> camera -> coupled equilibrium -> q chain
over a 6-point eps sweep. PASS iff best relative error < 1e-4. If it fails,
no experiment runs until it passes.

## Experiment A — camera self-calibration (inverse rendering)

Unknowns: emissivity map (32x16), psf_sigma, gain, offset. Known: the true
temperature field (one coupled solve at the nominal chip source), pose,
ambient.

- Parametrization: eps = 0.5 + 0.49 * tanh(z) (keeps eps in (0.01, 0.99));
  sigma, gain, offset unconstrained in log/linear space (sigma via softplus).
- Prior: total-variation-like smoothness lambda_tv * mean(Lap(eps)^2) with
  lambda_tv = 3e-2 (chosen on the noise-free case before running the noisy
  arms; the same value is then used for every noise level).
- Optimizer: Adam, lr 0.05 on z, 0.02 on the scalars, 400 iterations.
- Init: eps = 0.9 uniform, sigma = 0.8, gain = 20, offset = 400.
- Arms: noise sigma in {0, 1, 2, 5, 10} counts.
- Outcome metrics (reported for every arm, success or not):
  RMSE(eps_rec - eps_true), |sigma_rec - sigma_true|, relative errors of
  gain and offset, final rendering loss.
- Declared identifiability caveat (must appear in the writeup): a global
  gain <-> emissivity rescaling is only broken by the reflected-ambient term
  and the offset; where T approaches T_ambient the problem degenerates. We
  therefore report the gain*eps product error alongside the individual
  errors, and treat individual-parameter errors honestly as prior-resolved.

## Experiment B — HEADLINE: seeing through the camera into the physics

Unknown: the volumetric heat-source map q(x, y) (32x16 = 512 values).
Known: camera parameters and emissivity (the Experiment A setting at their
TRUE values — calibration error propagation is out of scope and said so),
gamma, all bulk physics parameters.

- Ground truth: `two_blob_source` — Gaussian blobs at (0.42, 0.06) amplitude
  1.5*q_scale, width 0.05, and (0.62, 0.06) amplitude 0.9*q_scale, width
  0.04, q_scale = 1.5e8 W/m^3. Chosen off the symmetry axes of the chip band
  used in Track 02; the recovery is never initialized from it.
- Measurement: ONE rendered image of the equilibrium at q_true, noise sigma =
  2 counts, seed 42.
- Parametrization: q = q_scale * softplus(z) (nonnegativity built in),
  z init = softplus^{-1}(0.3) uniform (a flat warm plate, no spatial hint).
- Prior: smoothness lambda_s * mean(Lap(q/q_scale)^2), lambda_s = 1e-4
  (set on a preliminary noise-free run at most 50 iterations long, before
  the reported runs; same value for both arms).
- Optimizer: Adam lr 0.1 on z, 250 iterations, warm-started fixed points.
- Arms:
  1. **coupled** — gradients through the full two-way equilibrium
     (`one_way=False`), forward physics coupled.
  2. **one-way physics** — the frozen-viscosity model: the SAME inverse
     machinery but with `coupling_scale = 0` physics (viscosity pinned at
     mu(T_ref), so flow does not respond to temperature). Its forward model
     is wrong for the data (which came from the coupled physics); the bias
     of its recovered source measures what ignoring the coupling costs.
     Reported with the identical prior, optimizer, iterations, and data.
- Outcome metrics (both arms): relative L2 error of q_rec, amplitude ratio,
  centroid shift in cells, total-power ratio, final data loss; convergence
  curves. The COUPLED-vs-ONE-WAY comparison is reported whatever its sign.
  If the one-way arm recovers the source just as well, that is the result
  we publish.
- Success criteria (declared now): coupled arm rel_l2 < 0.5 and centroid
  shift < 1.5 cells at sigma_noise = 2. No criterion for the one-way arm —
  it is a measurement, not a target.

## Experiment C (stretch, only if A and B pass their gates)

Design-from-appearance: optimize gamma so the RENDERED image matches an
apparent-hotspot specification. Protocol to be added here (and committed)
before it runs; if it is absent from the writeup, it was not attempted.

## Reporting rule

Every number in the README/writeup must come from a JSON artifact in
figures/ written by the script that produced it, carrying its configuration
(grid, seeds, noise, priors, iteration counts). Failures and deviations are
reported in the same place as successes.
