# Inverse rendering through a multiphysics equilibrium

**Tesseract Hackathon 2026 — Track 05: Differentiable graphics & rendering**
Author: il-miscusi · Apache-2.0 · Repository: `tesseract-inverse-thermography`

## 1. Problem

A cold plate cools an electronics assembly. Somewhere in it, a hidden
volumetric heat source q(x, y) — a die dissipating where it should not — heats
the coolant, the coolant's viscosity changes, the flow redistributes, and the
surface temperature field settles into an equilibrium. A thermal camera looks
at that surface and records digital counts. The question is the inverse one:
**from one noisy LWIR image, where is the heat coming from?**

This is an inverse *rendering* problem whose scene parameters live behind a
physical equilibrium. The forward map is

```
q  ──(coupled flow–heat–viscosity equilibrium)──►  T*(q)
T* ──(radiometric camera model)────────────────►  counts
```

and neither half is optional. Counts are not temperatures: emissivity,
reflected ambient, blur, vignetting, gain and offset all sit between the field
and the pixel. And the surface temperature is not the source: conjugate
transport smears, advects, and couples it. Recovering q by gradient descent on
a pixel loss requires differentiating through both — the renderer *is* the
measurement model.

## 2. The composition, and why the boundaries are genuine

The forward map is composed from four Tesseracts, each differentiated by a
different mechanism:

| component | language / framework | derivative |
|---|---|---|
| thermal-camera renderer | JAX | autodiff VJP |
| Darcy/Brinkman flow | Fortran | **hand-derived discrete adjoint** |
| heat transport | JAX | implicit function theorem at the fixed point |
| viscosity closure | PyTorch | autodiff tape |

These are not one framework wearing four hats. Fortran has no tape at all —
its adjoint is derived by hand from the discrete equations and verified against
finite differences. The JAX and PyTorch tapes are mutually invisible. And the
equilibrium temperature T* is defined only implicitly, as the fixed point of
T = Heat(γ, Flow(γ, Viscosity(T)); q), so even a single-framework
implementation could not tape through it; the adjoint must come from the
implicit function theorem. Tesseract's uniform `apply` /
`vector_jacobian_product` contract is what makes these four derivative
mechanisms composable as one chain rule. Without it there is no slower
fallback — the pixels→q gradient does not exist.

## 3. The renderer

The camera Tesseract is physically based, not a colormap:

- **Radiometry.** Spectral radiance is integrated over the 8–14 µm LWIR band
  by fixed-node quadrature of Planck's law, so dL/dT carries the correct
  band-limited temperature sensitivity (~0.9 W m⁻² sr⁻¹ per K near 300 K)
  rather than a linearised proxy.
- **Surface model.** Grey opaque surfaces: at-sensor radiance is
  ε·L(T) + (1−ε)·L(T_ambient). The reflected-ambient term is not a nuisance —
  it is the physics that partially breaks the gain↔emissivity ambiguity in
  Experiment A.
- **Geometry.** The plate is projected through a homography (tilted-camera
  perspective) with differentiable bilinear sampling.
- **Optics.** A Gaussian PSF whose width σ is itself a differentiable
  parameter, and cos⁴ falloff vignetting.
- **Sensor.** Radiance → counts through gain and offset. Measurement noise
  (additive Gaussian in counts; 2 counts ≈ 0.045 K NETD, a realistic uncooled
  microbolometer floor) is applied to data once, outside the gradient path.

Every stage has its own unit gate: Planck monotonicity, emissivity limits,
homography round trips, PSF normalisation, sampling, vignetting, and a
temperature-VJP finite-difference check — 18 fast tests, run by `make judge`
and by CI on every push.

## 4. The adjoint chain

Let J be the pixel loss on rendered counts C = R(T*, θ) with camera
parameters θ, where T* satisfies the fixed point F(T*, q) = 0 of the coupled
equilibrium. The gradient the optimiser needs is

```
dJ/dq = (∂F/∂q)ᵀ λ,   where   (∂F/∂T)ᵀ λ = −Rᵀ_T (∂J/∂C)
```

The right-hand side is the renderer's VJP, pulled back from pixels to the
temperature field by JAX. The linear solve for λ is matrix-free GMRES, and
every matvec inside it crosses the Fortran, JAX, and PyTorch boundaries
through their Tesseract VJP endpoints — the hand-written Fortran adjoint
supplies its factor, the PyTorch tape its own. Finally (∂F/∂q)ᵀ contracts λ
onto the source grid. No component ever sees another's internals.

**Gate G0 (precondition for everything).** The complete pixels→q derivative is
checked against central finite differences of the entire stack on a 16×8
physics grid and 48×32 sensor: best directional relative error **3.6×10⁻⁷**
against a declared 10⁻⁴ threshold. No experiment ran before this gate passed.

## 5. Pre-registered protocol and results

The protocol (`writeup/PROTOCOL.md`) was committed before any reported run and
fixes grids, seeds, noise levels, priors, optimizer settings, and success
criteria. Its reporting rule binds every number below to a JSON artifact in
`figures/` written by the producing script. The original run is governed by
the protocol's Deviations section; the rerun is declared under Amendment v2.

### Experiment A — camera self-calibration

From one image of a known temperature field: recover the 32×16 emissivity map,
PSF width, gain, and offset, at noise σ ∈ {0, 1, 2, 5, 10} counts
(`figures/experiment_a.json`).

- Gain recovered to **0.20–0.21 % relative error** at every noise level.
- The declared identifiability caveat held: a global gain↔emissivity
  rescaling is only broken by reflected ambient and offset, so individual PSF
  and offset estimates from a single image stayed weakly identified
  (σ absolute error 0.91 px; offset relative error 19.0 %), while the
  gain·ε product error stayed at 8.7 %. We report this as prior-resolved
  rather than claiming per-parameter success the data cannot support.

### Experiment B — source recovery through the physics

From one image at noise σ = 2 counts: recover a 512-value non-negative
two-hotspot source map, initialised from a flat featureless plate. Two arms
share data, prior, optimizer, and iteration budget: **coupled** (gradients
through the live two-way equilibrium) and **one-way** (viscosity frozen at
μ(T_ref), a deliberately wrong forward model for data that came from the
coupled physics).

**v1 — an honest failure.** The first registered run missed its pre-declared
target: coupled relative L2 error 0.9706 (criterion < 0.5) with a 6.85-cell
centroid shift. It still recovered total source power to a 1.041 ratio, versus
1.364 for the one-way arm (one-way L2 0.9814) — evidence that the coupled
adjoint sees the energy balance correctly even when 250 iterations of a
smoothness-only prior cannot localise two blobs from one blurred image. We
report the failure at full volume because a pre-registered criterion that can
only be met is not a criterion.

**v2 — the amended protocol.** Amendment v2 (recorded in PROTOCOL.md before
the rerun) revises the recovery configuration; the
data, seeds, and two-arm design are unchanged. Final numbers, from
`figures/experiment_b_v2.json`:

- Coupled arm: relative L2 error **0.1370**, amplitude ratio 1.115, final
  data loss 2.022 against a noise floor of 2.0.
- Coupled centroid shift: **0.02 cells**.
- Frozen-viscosity arm: relative L2 error **0.2457** (1.79x the coupled
  error), centroid shift 0.17 cells, final recorded data loss 3.339, 65%
  above the noise floor. Both arms reached the 250-iteration L-BFGS-B limit;
  these are budget-matched endpoints, not claimed convergence floors. This
  comparison changes the forward model and its derivative together.
- Total-power ratio: coupled **1.000**, one-way 1.017.
- Verdict: the amended pre-declared criteria (coupled rel L2 < 0.5 AND
  centroid shift < 1.5 cells) **PASS**; the end-to-end FD gradient gate was
  re-verified on the final configuration (best relative error 3.6e-07).

The coupled-vs-one-way comparison is reported whatever its sign; if the wrong
forward model recovers the source just as well, that is the published result.

### Experiment C — is the renderer load-bearing?

This diagnosis-informed follow-up was committed before its first run in
`RENDERER_PROTOCOL.md`. It removes the identical-discretization inverse crime:
the observation comes from a 64×32 coupled solve while all recoveries invert at
32×16. Physics, data, prior, initialization, optimizer, and budget are fixed;
only the assumed camera changes across full, blackbody, no-PSF, no-vignetting,
and modest-calibration-mismatch arms.

The calibrated renderer ends at 13.95 counts pixel RMS, 0.78-cell centroid
error, and 0.885 total-power ratio. The modest mismatch ends at 29.82 counts
RMS — within the predeclared 3× plausibility envelope — but shifts the centroid
to 2.60 cells and power to 0.807. Its additional **1.82-cell** error passes the
fixed diagnostic-necessity criterion. The renderer is therefore load-bearing
for the inferred location even when the image-space fit is not catastrophically
bad.

The ablation also bounds the claim. Blackbody and no-vignetting assumptions
are visibly rejectable (54.32 and 142.42 counts RMS). Removing PSF blur fits
nearly as well as full (14.51 counts RMS) and improves source L2 from 1.248 to
1.007. We therefore claim calibration sensitivity, not that every stage is
individually necessary. Four of five arms hit their 250-iteration limit; only
no-vignetting met the optimizer convergence condition.

## 6. Real-world relevance

Thermography-based fault localisation is a working industrial practice —
finding the overheating die, the delaminated thermal interface, the blocked
coolant channel — and its accuracy is limited today by exactly the two things
this entry makes differentiable: the camera's radiometric transfer (emissivity
uncertainty is the dominant error in quantitative thermography) and the
conduction/convection path between the fault and the visible surface. The same
composition pattern — a radiometric differentiable renderer as measurement
model over a differentiable physical equilibrium — transfers directly to melt
pool pyrometry in additive manufacturing and to active-thermography NDT.

## 7. Limitations

- **Synthetic measurements.** Experiment B uses the same camera and grid for
  truth and inversion. Experiment C removes the grid identity and deliberately
  mismatches the camera in four arms, but a real calibrated sensor remains
  future work.
- **Single image, smoothness prior.** v1 shows that one blurred LWIR image
  with a generic prior under-determines a 512-parameter source. Multi-view or
  transient imaging would condition the problem far better.
- **Calibration error propagation is sampled, not exhaustive.** Experiment C
  tests one fixed modest mismatch; a distribution over calibration posteriors
  would be needed for uncertainty bounds.
- **Grid scale.** 32×16 physics grids keep every CI run under minutes; the
  adjoint machinery is matrix-free and does not depend on this size.

## 8. Reproducibility

`make judge` runs the submission audit, all fast gates, and a syntax sweep with
one PASS/FAIL verdict; `make verify` reruns the in-process end-to-end
finite-difference gate. `make verify-containers` builds all four Tesseract
images, serves them, and repeats a small complete pixels-to-source gradient
check through the HTTP/container boundaries: best relative error 1.14e-07,
PASS in 67 seconds. Its JSON receipt records image IDs and source commit. CI
compiles the Fortran solver from source and runs the fast judged surface on
every push. All dependencies are pinned exactly. Provenance of the physics
core copied from the same author's Track 02 repository is recorded in
`NOTICE.md`.
