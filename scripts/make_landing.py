# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Build the dependency-free GitHub Pages landing page from result artifacts.

    python3 scripts/make_landing.py \
        --results figures/experiment_b.json \
        --expa figures/experiment_a.json \
        --gradcheck figures/e2e_gradient_check.json

Copies the figure assets into docs/assets and writes docs/index.html.
Re-run after regenerating figures against the final experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"

REPO_URL = "https://github.com/il-miscusi/tesseract-inverse-thermography"

FIGS = ("hero.png", "chain.png", "recovery_convergence.png",
        "radiometry.png", "renderer_necessity.png",
        "experiment_d_generalization.png", "recovery.gif")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/experiment_b.json")
    ap.add_argument("--expa", default="figures/experiment_a.json")
    ap.add_argument("--gradcheck", default="figures/e2e_gradient_check.json")
    ap.add_argument("--renderer-results", default="figures/experiment_c_renderer.json")
    ap.add_argument("--container-check", default="figures/container_e2e_gradient_check.json")
    ap.add_argument("--experiment-d", default="figures/experiment_d/experiment_d_summary.json")
    ap.add_argument("--experiment-e", default="figures/experiment_e_factorial.json")
    args = ap.parse_args()

    b = json.loads((ROOT / args.results).read_text())
    a = json.loads((ROOT / args.expa).read_text())
    g = json.loads((ROOT / args.gradcheck).read_text())
    c = json.loads((ROOT / args.renderer_results).read_text())
    container = json.loads((ROOT / args.container_check).read_text())
    d = json.loads((ROOT / args.experiment_d).read_text())
    e = json.loads((ROOT / args.experiment_e).read_text())
    rc, ro = b["results"]["coupled"], b["results"]["one_way"]

    ASSETS.mkdir(parents=True, exist_ok=True)
    have = {}
    for name in FIGS:
        src = ROOT / "figures" / name
        if src.exists():
            shutil.copyfile(src, ASSETS / name)
            have[name] = True
        else:
            print(f"  note: figures/{name} missing, section skipped")
            have[name] = False

    arm0 = a["arms"][0]
    loss_gap = ro["final_data_loss"] / rc["final_data_loss"]
    cf = c["results"]["full"]
    cm = c["results"]["calibration_mismatch"]
    centroid_excess = cm["centroid_shift_cells"] - cf["centroid_shift_cells"]
    dfull = d["arms"]["full"]
    dpair = d["paired_diagnostic_error"]["total_power_relative_error"]
    dci = [100 * value for value in dpair["median_bootstrap_95_ci"]]

    gif_html = ('<figure><img src="assets/recovery.gif" alt="Recovery animation" loading="lazy">'
                '<figcaption><b>Recorded run</b> · frozen unseen scene 101 (Experiment D bank), '
                'calibrated arm &mdash; per-iteration snapshots from '
                '<code>figures/experiment_d/bank/bank_seed_101.npz</code>, not a demo.'
                '</figcaption></figure>'
                if have["recovery.gif"] else "")

    rows = "".join(
        f"<tr><td>{label}</td><td class='num c'>{cv}</td><td class='num o'>{ov}</td></tr>"
        for label, cv, ov in [
            ("relative L<sub>2</sub> error of q&#770;", fmt(rc["rel_l2"]), fmt(ro["rel_l2"])),
            ("peak amplitude ratio (1 = ideal)", fmt(rc["amplitude_ratio"]),
             fmt(ro["amplitude_ratio"])),
            ("centroid shift [cells]", fmt(rc["centroid_shift_cells"], 2),
             fmt(ro["centroid_shift_cells"], 2)),
            ("total recovered power ratio (1 = ideal)", fmt(rc["total_power_ratio"]),
             fmt(ro["total_power_ratio"])),
            ("final data loss [counts&sup2;]", f"{rc['final_data_loss']:.3g}",
             f"{ro['final_data_loss']:.3g}"),
            ("adjoint GMRES matvecs", str(rc["adjoint_matvecs_total"]),
             str(ro["adjoint_matvecs_total"])),
        ])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inverse Thermography — Tesseract Hackathon 2026, Track 05</title>
  <meta name="description" content="A physically based differentiable LWIR thermal-camera renderer, composed with a Fortran/JAX/PyTorch coupled equilibrium: recover the hidden heat source from one noisy image.">
  <meta property="og:title" content="See the heat. Invert the physics.">
  <meta property="og:description" content="A differentiable thermal camera in JAX, welded to a multi-framework coupled PDE equilibrium. One noisy image in, the hidden heat source out.">
  <meta property="og:image" content="https://il-miscusi.github.io/tesseract-inverse-thermography/assets/hero.png">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,700&family=Source+Sans+3:wght@400;600;700&display=swap">
  <style>
    :root {{ color-scheme:light; --bg:#fdfcfa; --ink:#1c1a17; --muted:#514a3f; --faint:#6b6459;
            --line:#d9d2c7; --accent:#8a3324; --panel:#0f0d13; --panel-line:#2a2530; --panel-muted:#b8b0a2; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; background:var(--bg); color:var(--ink);
           font:16px/1.65 "Source Sans 3","Helvetica Neue",Arial,sans-serif; }}
    a {{ color:var(--accent); }}
    .wrap {{ width:min(1080px, calc(100% - 3rem)); margin:auto; }}
    nav {{ display:flex; justify-content:space-between; align-items:baseline; padding:1.7rem 0 1.2rem;
          border-bottom:3px double var(--ink); }}
    nav b {{ letter-spacing:.18em; text-transform:uppercase; font-size:.8rem; font-weight:600; }}
    nav div a {{ color:var(--faint); text-decoration:none; margin-left:1.6rem; font-size:.85rem; letter-spacing:.06em; }}
    nav div a.repo {{ color:var(--ink); font-weight:600; }}
    header {{ padding:4.2rem 0 2.6rem; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.76rem; }}
    h1 {{ margin:.8rem 0 1.4rem; font-family:Newsreader,Georgia,"Times New Roman",serif; font-weight:500;
         font-size:clamp(2.4rem,5.5vw,4.2rem); line-height:1.08; letter-spacing:-.015em; max-width:900px; }}
    .lede {{ font-family:Newsreader,Georgia,serif; color:var(--muted); font-size:clamp(1.15rem,2vw,1.4rem);
            line-height:1.5; max-width:800px; }}
    .lede b {{ color:var(--ink); font-weight:700; }}
    .actions {{ display:flex; flex-wrap:wrap; gap:.85rem; margin-top:2.1rem; }}
    .button {{ display:inline-block; padding:.78rem 1.5rem; font-size:.94rem; font-weight:600;
              text-decoration:none; border:1px solid var(--ink); color:var(--ink); }}
    .button.primary {{ background:var(--ink); color:var(--bg); }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); padding:2.2rem 0; border-bottom:1px solid var(--line); }}
    .metric {{ border-left:1px solid var(--line); padding:.3rem 1.5rem; }}
    .metric .big {{ font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(1.6rem,3vw,2.5rem); line-height:1.1; }}
    .metric .lbl {{ color:var(--faint); font-size:.84rem; margin-top:.45rem; line-height:1.45; }}
    section {{ padding:3.4rem 0 .6rem; }}
    .fig-eyebrow {{ font-size:.75rem; letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
                   font-weight:700; margin:0 0 .7rem; }}
    h2 {{ margin:0 0 .9rem; font-family:Newsreader,Georgia,serif; font-weight:500;
         font-size:clamp(1.55rem,3vw,2.15rem); line-height:1.2; letter-spacing:-.01em; }}
    p {{ max-width:760px; color:var(--muted); }}
    p b {{ color:var(--ink); }}
    p.note {{ color:var(--faint); font-size:.9rem; }}
    figure {{ margin:2rem 0 2.4rem; border:1px solid var(--line); background:var(--panel); padding:1.2rem; }}
    figure img {{ width:100%; display:block; }}
    figcaption {{ color:var(--panel-muted); font-size:.85rem; line-height:1.5; margin-top:.8rem;
                 padding-top:.7rem; border-top:1px solid var(--panel-line); }}
    figcaption b {{ color:#fdfcfa; }}
    table {{ border-collapse:collapse; width:100%; max-width:760px; margin:1.4rem 0; font-size:.93rem; }}
    th, td {{ text-align:left; padding:.55rem .7rem; border-bottom:1px solid var(--line); }}
    th {{ color:var(--faint); font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em;
         border-bottom:2px solid var(--ink); }}
    td.num {{ font-variant-numeric:tabular-nums; text-align:right; }}
    td.c {{ color:var(--ink); font-weight:600; }} td.o {{ color:var(--faint); }}
    footer {{ border-top:3px double var(--ink); margin-top:4rem; padding:1.7rem 0 2.6rem;
             color:var(--faint); font-size:.85rem; }}
    code {{ font-size:.85em; }}
    @media (max-width:820px) {{ .metrics {{ grid-template-columns:1fr 1fr; row-gap:1.4rem; }} nav div {{ display:none; }} }}
  </style>
</head>
<body>
<div class="wrap">
  <nav>
    <b>Tesseract Hackathon 2026 · Track 05 · Differentiable graphics &amp; rendering</b>
    <div>
      <a href="#chain">The chain</a>
      <a href="#generalization">Evidence</a>
      <a href="#renderer">The renderer</a>
      <a class="repo" href="{REPO_URL}">Repository &rarr;</a>
    </div>
  </nav>

  <header>
    <div class="eyebrow">Inverse rendering through a multiphysics equilibrium</div>
    <h1>See the heat. Invert the physics.</h1>
    <p class="lede">A thermal camera never sees temperature, it sees band-integrated
      Planck radiance through emissivity, optics, and a sensor. We built that camera as a
      <b>differentiable JAX Tesseract</b> and welded it to a coupled
      <b>Fortran&thinsp;/&thinsp;JAX&thinsp;/&thinsp;PyTorch</b> cold-plate equilibrium.
      Then we asked one noisy rendered frame a question no framework can answer alone:
      <b>which part of the chip is overheating?</b> The gradient that answers it runs from
      pixels, back through the renderer, through an implicit-function-theorem adjoint
      spanning three native derivative regimes and hand-written Fortran, to the volumetric heat source.</p>
    <div class="actions">
      <a class="button primary" href="{REPO_URL}">Explore the repository</a>
      <a class="button" href="{REPO_URL}/blob/main/output/pdf/tesseract_inverse_thermography.pdf">Read the paper</a>
    </div>
  </header>

  <div class="metrics">
    <div class="metric"><div class="big">{g['best_rel_err']:.1e}</div>
      <div class="lbl">end-to-end gradient vs finite differences (pixels &rarr; q), verdict {g['verdict']}</div></div>
    <div class="metric"><div class="big">{dfull['operationally_useful_count']}/12</div>
      <div class="lbl">operationally useful diagnoses on the frozen unseen-scene bank</div></div>
    <div class="metric"><div class="big">{100 * dpair['median']:.2f} pp</div>
      <div class="lbl">paired median power-error increase from only 4% emissivity error; 12/12 same direction</div></div>
    <div class="metric"><div class="big">{container['best_rel_err']:.1e}</div>
      <div class="lbl">finite-difference error through four served containers · {container['seconds']:.0f} s · {container['verdict']}</div></div>
  </div>

  <section>
    <div class="fig-eyebrow">Figure 1 · The inversion</div>
    <h2>One noisy frame, back to the source</h2>
    <figure>
      <img src="assets/hero.png" alt="True source, coupled temperature field, rendered thermal image, recovered source">
      <figcaption><b>Fig. 1</b> · Left to right: the hidden heat source; the coupled flow–viscosity–temperature
        equilibrium it drives; the noisy LWIR frame the camera records (digital counts); and the
        source recovered by gradient descent through the whole chain.</figcaption>
    </figure>
  </section>

  <section>
    <div class="fig-eyebrow">Figure 2 · Why Tesseract</div>
    <h2 id="chain">One operational gradient across native stacks</h2>
    <p>The temperature field solves a fixed point spanning a PyTorch viscosity closure, a
       Fortran Darcy solver with a pen-and-paper adjoint, and a JAX heat-transport model.
       The camera renders that equilibrium to counts. Differentiating the composition uses
       only each component's VJP endpoint: GMRES solves the adjoint system matrix-free, one
       three-component VJP chain per matvec, then one heat-transport VJP lands the gradient
       on q. A separate receipt builds and serves all four images, then verifies the complete
       derivative through real HTTP/container boundaries at {container['best_rel_err']:.1e}
       relative error.</p>
    <figure><img src="assets/chain.png" alt="Differentiable chain diagram">
      <figcaption><b>Fig. 2</b> · The forward render path in muted ink; the reverse pixels-to-source
        gradient path through the camera VJP and the implicit-function-theorem adjoint.</figcaption></figure>
  </section>

  <section>
    <div class="fig-eyebrow">Figure 3 · Pre-registered evidence</div>
    <h2 id="generalization">Frozen-bank evidence, including the negative verdict</h2>
    <p>Twelve fault scenes stayed sealed until the method, two-view training,
       held-out third view, absolute pixel gate, and 4% emissivity mismatch were fixed.
       Truth runs at 64&times;32 and inversion at 32&times;16. The calibrated arm is
       operationally useful on <b>{dfull['operationally_useful_count']}/12</b> scenes and
       passes every held-out residual check on <b>{dfull['absolute_plausible_fit_count']}/12</b>.
       Median held-out RMS is {dfull['metrics']['holdout_rms_counts']['median']:.3f} counts
       at a 2-count noise floor; median centroid error is
       {dfull['metrics']['centroid_error_mm']['median']:.3f}&nbsp;mm.</p>
    <p>A fixed 4% emissivity underestimate increases absolute total-power error on
       <b>{dpair['full_lower_error_count']}/12</b> paired scenes. The median increase is
       <b>{100 * dpair['median']:.2f} percentage points</b> (deterministic paired-bootstrap
       95% interval {dci[0]:.2f}&ndash;{dci[1]:.2f}). The preregistered stronger claim is
       <b>not accepted</b>: only {d['materially_harmful_plausible_mismatch_count']}/12,
       short of 6/12, were both independently plausible and at least five points harmful.
       No scene or failed arm is excluded.</p>
    <figure><img src="assets/experiment_d_generalization.png" alt="Paired unseen-scene source power error and held-out residuals">
      <figcaption><b>Fig. 3</b> · Experiment D. Every point is a frozen scene; violet mismatch markers fail
        at least one independent plausibility condition, even when RMS alone is below 2&sigma;.</figcaption></figure>
  </section>

  <section>
    <div class="fig-eyebrow">Figure 4 · Mechanism</div>
    <h2 id="results">Mechanism check: the coupling is load-bearing</h2>
    <p>Two recoveries share data, prior, optimizer, and budget. The second uses a
       deliberately simplified frozen-viscosity forward model and its matching derivative;
       the comparison therefore measures model mismatch, not a gradient-only intervention.</p>
    <table>
      <tr><th>metric</th><th style="text-align:right">coupled adjoint</th><th style="text-align:right">one-way</th></tr>
      {rows}
    </table>
    <p class="note">Grid {b['grid'][0]}&times;{b['grid'][1]} plate, {b['sensor'][0]}&times;{b['sensor'][1]}
       sensor, &sigma;<sub>noise</sub> = {b['noise_counts']} counts, seeds pinned; protocol committed
       before the run (writeup/PROTOCOL.md). Declared criterion: {b['success_criteria']['declared']} —
       met: {b['success_criteria']['met']}. Both arms reached the declared 250-iteration
       limit, so the table reports budget-matched endpoints rather than convergence floors.</p>
    <p><b>Factorial mechanism check.</b> Experiment E keeps the coupled forward
       values but drops only the implicit feedback term in reverse. Source L<sub>2</sub>
       becomes {e['coupled_truncated']['final']['rel_l2']:.3f}, versus
       {rc['rel_l2']:.3f} with the exact implicit adjoint and {ro['rel_l2']:.3f}
       with frozen forward physics. Data loss is
       {e['coupled_truncated']['final']['data_loss']:.3f}, between exact-coupled
       {rc['final_data_loss']:.3f} and frozen-forward {ro['final_data_loss']:.3f}.
       This separates the effects: forward fidelity dominates pixel fit; exact
       gradient fidelity improves the inferred source shape.</p>
    <figure><img src="assets/recovery_convergence.png" alt="Convergence, coupled vs one-way">
      <figcaption><b>Fig. 4</b> · Budget-matched convergence of the coupled-adjoint and one-way arms.</figcaption></figure>
    {gif_html}
  </section>

  <section>
    <div class="fig-eyebrow">Figure 5 · Calibration sensitivity</div>
    <h2 id="necessity">The renderer changes the diagnosis</h2>
    <p>Experiment C removes the same-grid inverse crime: observations come from a
       {c['truth_grid'][0]}&times;{c['truth_grid'][1]} coupled solve and inversion runs at
       {c['inverse_grid'][0]}&times;{c['inverse_grid'][1]}. With the calibrated renderer,
       centroid error is {cf['centroid_shift_cells']:.2f} cells. A modest camera mismatch
       (PSF 0.9 vs 1.2 px, gain +5%, offset +10 counts, ambient +3 K, and a small FOV error)
       still fits within {cm['pixel_rms_counts'] / cf['pixel_rms_counts']:.2f}&times; the
       calibrated RMS, but moves the inferred hotspot an additional {centroid_excess:.2f}
       cells. Both fits remain far above the {c['noise_counts']:.0f}-count noise scale,
       so this is calibration-sensitivity evidence, not a plausible-fit claim.</p>
    <p class="note">The result is deliberately mixed: blackbody and no-vignetting models
       are visibly rejectable, while removing PSF blur does not worsen localization and
       improves coarse-grid source L2. The evidence supports calibration sensitivity, not
       a claim that every renderer stage is indispensable.</p>
    <figure><img src="assets/renderer_necessity.png" alt="Renderer calibration mismatch changes recovered hotspot">
      <figcaption><b>Fig. 5</b> · Calibrated versus mismatched camera models on an independent discretization.</figcaption></figure>
  </section>

  <section>
    <div class="fig-eyebrow">Figure 6 · The measurement model</div>
    <h2 id="renderer">The renderer is physics, not a colormap</h2>
    <p>Planck spectral radiance integrated over the 8&ndash;14&nbsp;&mu;m LWIR band by
       Gauss&ndash;Legendre quadrature, grey-body emission plus reflected ambient
       (Kirchhoff), a sensor-to-plate homography with differentiable bilinear sampling, a
       Gaussian PSF whose width is itself a parameter, cos<sup>4</sup> vignetting, then gain
       and offset to digital counts. Every stage moves the measured image; every stage is
       differentiable. Experiment&nbsp;A tests single-frame calibration and exposes its
       identifiability limit: gain is within {100 * arm0['gain_rel_err']:.1f}% at zero noise,
       but emissivity RMSE remains {arm0['eps_rmse']:.3f} and PSF/offset are not recovered.</p>
    <figure><img src="assets/radiometry.png" alt="Planck curve and per-stage ablation strip">
      <figcaption><b>Fig. 6</b> · Band-integrated Planck radiometry and the per-stage ablation strip.</figcaption></figure>
  </section>

  <footer>
    Tesseract Hackathon 2026 &middot; Track 05 &middot;
    <a href="{REPO_URL}">source</a> &middot;
    <a href="{REPO_URL}/blob/main/output/pdf/tesseract_inverse_thermography.pdf">paper</a> &middot;
    <a href="{REPO_URL}/blob/main/writeup/EXPERIMENT_D_PROTOCOL.md">frozen-bank protocol</a> &middot;
    Apache-2.0. Every figure on this page regenerates from committed artifacts with
    <code>make summarize-experiment-d figures renderer-figure landing</code>.
  </footer>
</div>
</body>
</html>
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html)
    print(f"wrote {DOCS / 'index.html'}")


if __name__ == "__main__":
    main()
