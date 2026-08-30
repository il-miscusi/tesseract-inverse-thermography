# Demo video script — "See the heat. Invert the physics." (≤ 5:00)

All result numbers below are bound to `figures/experiment_b_v2.json` and
`figures/e2e_gradient_check.json`. Target runtime 4:30 at a calm reading pace
(~140 wpm).

Screen recording of the landing page + figures + one live terminal. No
webcam needed. Music optional and quiet.

---

## Shot 1 — Cold open on the rendered frame (0:00–0:25)

**Visuals:** `figures/hero.png`, cropped to panel 3 (the noisy camera frame)
filling the screen. Hold 3 s in silence, then the title fades in:
"See the heat. Invert the physics."

**VO:**
> This is a thermal image of a liquid-cooled chip plate. It isn't a
> photograph of temperature — no thermal image is. It's Planck radiance
> through emissivity, a lens, and a sensor, in digital counts.
> Somewhere behind this frame, two hot spots are driving the whole scene.
> The camera never sees them. We're going to recover them anyway — with
> one gradient.

## Shot 2 — The forward chain (0:25–1:15)

**Visuals:** `figures/chain.png`, slow pan left to right along the forward
row. Badge colors call out the frameworks as they're named.

**VO:**
> The physics is a coupled equilibrium that no single autodiff framework
> can trace. Viscosity comes from a PyTorch closure. The coolant flow is a
> compiled Fortran Darcy solver whose adjoint was derived by hand. Heat
> transport is JAX. Temperature feeds viscosity feeds flow feeds
> temperature — a genuine two-way fixed point.
> On top sits the fourth Tesseract: a physically based LWIR camera, written
> in JAX. Planck radiance integrated over eight to fourteen microns,
> grey-body emissivity with reflected ambient, a homography onto the
> sensor, a differentiable point-spread function, vignetting, gain and
> offset. Temperature field in, digital counts out — and every stage
> differentiable.

## Shot 3 — The reverse path (1:15–2:00)

**Visuals:** `chain.png` again, now the bottom row; trace the cyan reverse
arrows right to left. Overlay the two equations when the VO reaches them.

**VO:**
> Ask for the gradient of a pixel loss with respect to the heat source, and
> here is what actually runs. The camera's VJP pulls the loss back to the
> temperature field. The implicit function theorem turns the fixed point
> into a linear adjoint system, solved matrix-free with GMRES — every
> matvec is a chain of three VJPs across three containers: JAX, Fortran,
> PyTorch. One final heat-transport VJP lands on q.
> No framework ever sees the whole chain. Only VJPs cross the boundaries.
> We checked the composition against finite differences end to end:
> relative error 3.6 times 10 to the minus 7.

## Shot 4 — Live recovery (2:00–3:10)

**Visuals:** `figures/recovery.mp4` full-screen. Let it play; it carries
the shot. Caption: "illustrative 16×8 recovery · noise-free · Adam · gradient
through renderer + coupled physics".

**VO:**
> First, an illustrative small-grid recovery: hide the source, render one
> noise-free frame, and descend.
> Left, the truth the optimizer has never seen. Middle, its current belief,
> starting from a flat guess. Right, the image residual — the only thing it
> is ever told.
> Watch the residual drain as the belief localises. Each iteration is a full
> coupled solve, a render, and an adjoint solve. In the final noisy 32-by-16
> experiment, the implicit adjoint averaged seven matrix-free matvecs per
> gradient: 2,220 across the coupled recovery.

## Shot 5 — The coupling is load-bearing (3:10–4:10)

**Visuals:** `figures/recovery_convergence.png`, then the results table on
`docs/index.html` (scroll slowly).

**VO:**
> Does the multiphysics actually matter? We ran the identical recovery a
> second time with the industry-standard shortcut: viscosity frozen, one-way
> physics — same data, same optimizer, only the gradient changed.
> The coupled adjoint reaches the two-count noise floor; the one-way fit stays
> 65 percent above it. It recovers the source with relative L-two error 0.1370
> against 0.2457 one-way — 1.79 times worse — while ignoring the coupling
> biases total recovered power 1.7 percent high.
> The protocol, seeds, and success criteria were committed before the run —
> what you see is what the pre-registration produced.

## Shot 6 — Close (4:10–4:40)

**Visuals:** `figures/radiometry.png` for 5 s (the ablation strip), then
the landing page hero with the repo URL on screen.

**VO:**
> The renderer is physics, not a colormap: switch off emissivity, the blur,
> or the vignetting, and the counts move — which is exactly why the camera
> can also calibrate itself: emissivity map, PSF width, gain and offset,
> all recovered from a single frame in experiment A.
> Four Tesseracts, three languages, one gradient. Code, protocol, and every
> figure you've seen regenerate with one make target. Thanks for watching.

---

**Recording notes**
- Record at 1920×1080; the dark theme is the grade, don't color-correct.
- `make animation` regenerates `recovery.mp4` before capture.
- Keep total ≤ 5:00 hard; Shot 4 is the flex — trim the hold there first.
