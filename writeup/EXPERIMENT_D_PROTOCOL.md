# Experiment D protocol — unseen-scene calibration risk

This document separates method development from final evidence. Seeds 0–2 are
development scenes and may be used to debug or tune the fixed method. The
twelve bank seeds 101–112 are held out: no method choice may use their results.
The bank is run only after the final protocol and runner are committed.

## Question

Across unseen component-fault scenes, can the calibrated camera fit a held-out
view at the declared sensor-noise scale and recover a useful diagnosis? If a
camera model inside a declared calibration-tolerance family also passes that
absolute fit gate, does it move the diagnosis by a practically material amount?

## Fixed problem bank

- Twelve scenes, seeds 101–112; each has 2–4 Gaussian component faults with
  randomized position, width, and amplitude inside the known chip footprint.
- Truth is generated on a 64×32 coupled-physics grid; inversion is 32×16.
  Coarse targets are 2×2 area averages of the fine-grid source, not samples of
  the inversion model.
- Material topology is fixed across scenes from problem seed 73. Each scene has
  an independent noise draw derived from its scene seed.
- The camera records two training poses (homography tilts +0.22 and −0.18) and
  one held-out pose (+0.05). Gaussian sensor noise is σ=2 counts per pixel.
- Evaluation uses a pose-fixed interior plate ROI: sensor pixels whose mapped
  plate coordinates are at least 1.5 coarse cells from every plate boundary.
  The same mask is used by both arms. Mixed silhouette/background pixels are
  excluded because a cell-centred 32×16 surface and a 64×32 surface represent
  that sub-pixel geometric edge differently; the claim concerns thermography
  over the resolved plate surface, not anti-aliasing of its silhouette.
- Before fault-scene inversion, a known uniform 0.4·Q_SCALE-per-cell reference
  source and one +0.1·Q_SCALE perturbation at each of the 20 admissible chip
  cells are solved on both grids. Their fine-minus-coarse rendered-pixel
  discrepancies at all three fixed poses form a frozen local 20-column
  Jacobian from source to observation. It is applied after the camera for every
  arm and bank scene. The correction's direct source VJP is added to the
  camera-to-coarse-model implicit-adjoint gradient.
  No random fault scene or bank observation contributes to this calibration.

## Fixed inverse method

- The unknown source is non-negative and restricted to cell centres inside the
  known physical chip footprint. All cells outside that support are exactly
  zero. This is a component-fault prior, not a truth-location oracle: positions,
  count, widths, amplitudes, and total power remain unknown.
- L-BFGS-B optimizes a softplus parameterization from a flat 0.02·Q_SCALE
  initialization, with TV weight 3e-4. Maximum 500 iterations / 1500 function
  evaluations, gradient tolerance 1e-7, and relative function tolerance 1e-6.
- The best evaluated iterate and final SciPy iterate are both stored. A run is
  called converged only when SciPy status is zero; otherwise it is an endpoint.
- Every scene first scores a discretization oracle: the exact 2×2-area-averaged
  true source through the coarse model. This is diagnostic, not an optimizer
  arm, and establishes whether the absolute pixel gate is representable at all.
- Snapshots are retained whenever a new best iterate is found after at least ten
  evaluations, enabling a final animation from the reported experiment.

## Camera arms

1. `full`: the generating camera (PSF 1.2 px, gain 25, offset 500, ambient
   295 K, tan-half-FOV 0.45, declared pose homographies).
2. `mismatch`: the same radiometry and optics, but a fixed +0.03 plate-fraction
   x translation in the sensor-to-plate homography (0.6 mm on the 20 mm plate).

The mismatch is not called plausible merely because it is close to the full
arm. Plausibility is evaluated independently on the held-out pose.

## Fixed metrics and gates

For each arm and scene report training and held-out pixel RMS in counts and
noise sigmas; held-out residual mean; maximum horizontal/vertical lag-1
residual autocorrelation; source relative L2; centroid error in cells and mm;
total-power error; peak-location error in mm; and source Wasserstein distance
in mm.

An arm has an **absolute plausible held-out fit** only if all hold:

- held-out pixel RMS ≤ 2.0 noise sigmas (4 counts);
- absolute held-out residual mean ≤ 0.25 noise sigmas (0.5 counts);
- absolute horizontal and vertical lag-1 residual autocorrelation ≤ 0.10.

A diagnosis is operationally useful when centroid error ≤1.0 mm, peak error
≤1.5 mm, and total-power relative error ≤20%. A mismatch is materially harmful
when it passes the same absolute fit gate yet increases centroid or Wasserstein
error by at least 0.5 mm, peak error by at least 1.0 mm, or total-power error by
at least 10 percentage points.

## Bank reporting

No bank scene or failed arm is excluded. Report median, IQR, 90th percentile,
worst case, paired win count, and a deterministic 10,000-resample paired
bootstrap 95% interval for full-minus-mismatch diagnostic error. The grand-prize
calibration-risk claim is accepted only if the calibrated arm passes the
absolute fit gate on at least 10/12 scenes and a materially harmful mismatch
passes that gate on at least 6/12 scenes. Otherwise the result is reported as a
negative finding and the submission falls back to the verified composition
claim.

## Development amendment D1 — interior ROI

The first full-resolution development run (seed 0, calibrated arm) reduced
training half-MSE from 116,193 to 56.15 by evaluation 175 but had plateaued at
about 10.6 counts RMS. Inspection against Experiment C's stored residuals
showed the dominant coherent error on the projected plate boundary. The run was
stopped without producing a result artifact. The interior ROI above was then
fixed before any bank seed was run. No optimizer, source, or bank setting was
changed. All subsequent development and bank results must use and store the ROI.

## Development amendment D2 — calibrated multi-fidelity correction

After D1, the exact area-averaged source oracle was measured at 51.25 counts
training RMS and 51.21 counts held-out RMS on development seed 0. This proves
the original 4-count gate was outside the representable coarse-model family;
optimizing harder could only bias the source to absorb discretization error.
The known-load multi-fidelity correction above was therefore fixed before any
bank run. The failed oracle JSON/NPZ is retained in `figures/experiment_d/`.
The absolute gate, held-out seeds, camera arms, and diagnostic thresholds are
unchanged.

## Development amendment D3 — affine rather than additive transfer

The D2 additive transfer reduced the oracle to 6.16 counts but the optimized
development scene plateaued near 5.97 counts RMS. That shows the remaining grid
error is response-dependent, not a constant temperature offset. The same two
known calibration states are therefore used as an affine transfer, with its
per-cell slope included in the VJP. It was fixed before any bank scene and then
superseded by the more expressive amendments below.

## Development amendment D4 — spatial discrepancy Jacobian

The two-state affine oracle reached 7.26 counts RMS, worse than the additive
model, proving one scalar response per output cell cannot represent how source
location interacts with the grid. The final calibration therefore measures one
temperature-discrepancy response per admissible source cell. This 20-column
linear map is small, independently generated, reusable across all scenes, and
fully differentiated. The bank remains untouched and the 4-count gate remains
unchanged.

## Development amendment D5 — local tangent calibration

The first D4 implementation used unit-load secants from zero and produced a
277-count oracle: superposing large secants through the nonlinear viscosity-flow
equilibrium is not a Jacobian. The final calibration is explicitly local around
a 0.4·Q_SCALE-per-cell reference, close to the development scene's mean source
coefficient, with +0.1·Q_SCALE perturbations. Runtime uses `(q−q_ref)/delta`;
the VJP uses the same delta. This correction was fixed before any bank scene.

## Development amendment D6 — calibrate the measured observation

The local temperature-space tangent produced a 6.23-count oracle. The remaining
error comes from rendering fine and coarse cell lattices, so the final tangent
is measured in pixel space at the two training poses and held-out pose. This is
the quantity used by the loss and the absolute gate. Its direct pixel-to-source
VJP is included exactly. Physics calibration loads, reference, delta, bank,
thresholds, and camera arms are unchanged.

## Development amendment D7 — isolate pose-calibration risk

The original compound mismatch plateaued near 5.5 counts training RMS and was
therefore visibly rejectable under the unchanged absolute gate. The final
mismatch isolates a 0.6 mm plate-plane pose translation with otherwise correct
radiometry and optics. This targets the practitioner question directly: can a
pose error be absorbed by a shifted fault while pixels still validate? The
development calibrated arm already reached 1.989 counts held-out RMS, 0.060 mm
centroid error, and 0.154 mm Wasserstein error. Its post-noise-floor evaluations
changed the objective by less than 0.1%, so the relative-function tolerance is
set to 1e-6 before the bank to avoid spending hundreds of evaluations on the
sixth decimal of a noise-limited objective. The best iterate remains stored.
