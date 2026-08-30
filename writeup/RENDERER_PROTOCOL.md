# Experiment C protocol — renderer necessity under discretisation mismatch

This protocol and its runner are committed before the first result-producing
run. It is a diagnosis-informed follow-up to Experiments A/B, not represented
as part of their earlier preregistration.

## Question

Does the differentiable LWIR renderer change the inferred physical cause, or
does it merely make a prettier image?

## Fixed design

- Truth uses a 64×32 coupled oil-cooling solve; inversion uses 32×16. The
  coarse material field is replicated onto the fine grid, while temperature
  and source are solved independently. This prevents identical-discretisation
  self-inversion.
- One 96×64 observation is rendered by the complete camera: spatial
  emissivity, reflected 295 K ambient, oblique homography, σ=1.2 px Gaussian
  PSF, cos⁴ vignetting with tan(half-FOV)=0.45, gain 25, offset 500, and
  Gaussian noise σ=2 counts with seed 42.
- Every arm uses the same coarse coupled physics, measurement, source
  parameterisation, flat 0.05·Q_SCALE initial condition, TV prior λ=0.003,
  analytic adjoint, L-BFGS-B optimizer, and 250-iteration budget.

## Camera arms

1. `full`: correct complete camera.
2. `blackbody`: ε=1 everywhere, removing emissivity and reflected ambient.
3. `no_psf`: σ=0.05 px, effectively removing optical blur.
4. `no_vignetting`: tan(half-FOV)=0.
5. `calibration_mismatch`: σ=0.9, gain +5%, offset +10 counts, ambient +3 K,
   and tan(half-FOV)=0.50.

No arm may be removed after seeing its result. Optimizer non-convergence is
reported from the stored SciPy status; endpoint losses are not called floors.

## Metrics and interpretation fixed before running

Report for every arm: source relative L2, centroid shift in coarse-grid cells,
total-power ratio, peak-amplitude ratio, pixel RMS in counts, optimizer status,
and ratios/differences against `full`.

The renderer is declared **diagnostically load-bearing** if at least one
simplified arm both:

- produces a pixel RMS no more than 3× the full arm (a superficially plausible
  image fit), and
- worsens source relative L2 by at least 25%, shifts the centroid by at least
  0.5 cells, or biases total power by at least 10 percentage points relative
  to the full arm.

If no simplified arm meets both conditions, the renderer-necessity claim is
rejected. All arms and the rejection are still published.
