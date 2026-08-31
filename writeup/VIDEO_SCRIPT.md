# Demo video script — "See the heat. Invert the physics." (≤ 5:00)

All result numbers below are bound to the frozen Experiment D bank,
`figures/experiment_b_v2.json`, and the two gradient receipts. Target runtime
4:30 at a calm reading pace
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
> matvec is a chain of three container VJPs: JAX, Fortran,
> PyTorch. One final heat-transport VJP lands on q.
> No framework ever sees the whole chain. Only VJPs cross the boundaries.
> We checked the composition against finite differences end to end:
> relative error 3.6 times 10 to the minus 7.
> We also built and served all four images and repeated the complete check
> across real container boundaries: 1.14 times 10 to the minus 7.

## Shot 4 — Live recovery (2:00–3:10)

**Visuals:** `figures/recovery.mp4` full-screen. Let it play; it carries
the shot. Caption: "frozen unseen scene 101 · calibrated arm · real L-BFGS-B
trajectory · held-out residual".

**VO:**
> This is one of the twelve frozen bank recoveries: seed 101, opened only
> after the method and gates were fixed. Hide the source, render two noisy
> training views, and descend.
> Left, the truth the optimizer has never seen. Middle, its current belief,
> starting from a flat guess. Right, the final residual on a third camera
> view that the optimizer never sees.
> Watch the belief localise. Each iteration is a full
> coupled solve, a render, and an adjoint solve. In the final noisy 32-by-16
> experiment, the implicit adjoint averaged seven matrix-free matvecs per
> gradient: 2,220 across the coupled recovery.

## Shot 5 — Frozen-bank generalization (3:10–4:05)

**Visuals:** `figures/experiment_d_generalization.png`, then the frozen-bank
section on `docs/index.html` (scroll slowly).

**VO:**
> Now the test that matters: twelve unseen fault scenes, fine-grid truth,
> coarse-grid inversion, and an independently held-out view. The calibrated
> system gives an operationally useful diagnosis on all twelve and passes
> every absolute pixel check on ten. Median held-out RMS is 2.014 counts at a
> two-count noise floor; median centroid error is 0.088 millimetres; median
> total-power error is 0.153 percent.
> Every scene and every failed gate is here. Nothing is filtered after the run.

## Shot 6 — A result that can say no (4:05–4:35)

**Visuals:** stay on the paired power-error panel, then reveal the verdict line.

**VO:**
> A four-percent emissivity error increases inferred power error on all twelve
> paired scenes: median 4.42 percentage points, with a bootstrap interval from
> 3.28 to 4.88. But our stronger preregistered claim required six scenes to be
> both independently plausible and at least five points harmful. Only one did.
> So that claim is not accepted. A differentiable system should make scientific
> claims testable — including claims the evidence rejects.

## Shot 7 — Close (4:40–4:55)

**Visuals:** `figures/radiometry.png` for 5 s (the ablation strip), then
the landing page hero with the repo URL on screen.

**VO:**
> The renderer is physics, not a colormap. Four Tesseracts, three native
> derivative regimes, one implicit adjoint, and one gradient from pixels to
> the hidden source. Code, frozen protocol, raw bank, and every
> figure you've seen regenerate with one make target. Thanks for watching.

---

**Recording notes**
- Record at 1920×1080; the dark theme is the grade, don't color-correct.
- `make animation` regenerates `recovery.mp4` before capture.
- Keep total ≤ 5:00 hard; Shot 4 is the flex — trim the hold there first.

---

# 90-second cut — record this one if only one gets recorded

Same visuals inventory, three beats, ~215 words at 140 wpm ≈ 1:30.

## Beat 1 — Composition depth (0:00–0:30)

**Visuals:** hero.png panel 3 (3 s hold, title fade), then chain.png pan.

**VO:**
> This is a thermal image of a chip plate — Planck radiance through a lens
> and a sensor, not a temperature map. Behind it, two hidden hot spots.
> We recover them with one gradient, through physics no single autodiff
> framework can trace: a PyTorch viscosity closure, a compiled Fortran flow
> solver with a hand-derived adjoint, JAX heat transport in a two-way fixed
> point, and a physically based LWIR camera on top. Four Tesseracts, three
> native derivative regimes, one implicit adjoint — verified against finite
> differences to one part in ten million, through four served containers.

## Beat 2 — Generalization under audit (0:30–1:00)

**Visuals:** recovery.gif (real run, scene 101), then the Experiment D panel.

**VO:**
> Then we audited it. Twelve frozen unseen scenes, fine-grid truth,
> coarse-grid inversion, an independently held-out camera pose. Useful
> diagnosis on all twelve; held-out residuals at the sensor noise floor on
> ten. Every scene, every gate, in the shipped JSON — nothing filtered
> after the run.

## Beat 3 — A result that can say no (1:00–1:30)

**Visuals:** paired power-error panel, reveal the verdict line, close on
repo URL.

**VO:**
> A four-percent emissivity error slips past every residual check and biases
> every power estimate — all twelve scenes, confidence interval three point
> three to four point nine points. And when our stronger preregistered claim
> missed its own threshold, we shipped the rejection. A differentiable
> system should make claims testable — including the ones the evidence
> turns down. Code, protocol, raw bank: one make target regenerates it all.
