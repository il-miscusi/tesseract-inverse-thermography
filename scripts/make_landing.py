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
        "radiometry.png", "recovery.gif")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/experiment_b.json")
    ap.add_argument("--expa", default="figures/experiment_a.json")
    ap.add_argument("--gradcheck", default="figures/e2e_gradient_check.json")
    args = ap.parse_args()

    b = json.loads((ROOT / args.results).read_text())
    a = json.loads((ROOT / args.expa).read_text())
    g = json.loads((ROOT / args.gradcheck).read_text())
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

    gif_html = (f'<img src="assets/recovery.gif" alt="Recovery animation" loading="lazy">'
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
  <style>
    :root {{ color-scheme: dark; --bg:#0b0e14; --panel:#11151f; --ink:#e8ecf3; --muted:#9aa7b8;
            --line:#232a38; --accent:#4cc9f0; --amber:#f4a259; --hot:#ff7a3c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 80% -10%, #1e1420 0, transparent 40rem), var(--bg);
           color:var(--ink); font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
    a {{ color:var(--accent); }}
    .wrap {{ width:min(1080px, calc(100% - 2.4rem)); margin:auto; }}
    nav {{ display:flex; justify-content:space-between; align-items:center; padding:1.1rem 0;
          border-bottom:1px solid var(--line); }}
    nav b {{ letter-spacing:.1em; text-transform:uppercase; font-size:.8rem; color:var(--hot); }}
    nav div a {{ color:var(--muted); text-decoration:none; margin-left:1.1rem; font-size:.88rem; }}
    header {{ padding:4.5rem 0 2.5rem; }}
    .eyebrow {{ color:var(--hot); text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.76rem; }}
    h1 {{ margin:.6rem 0 1rem; font-size:clamp(2.2rem,5.5vw,4.4rem); line-height:1.02; letter-spacing:-.04em; max-width:900px; }}
    .lede {{ color:var(--muted); font-size:1.08rem; max-width:760px; }}
    .lede b {{ color:var(--ink); }}
    figure {{ margin:2.2rem 0; }}
    figure img {{ width:100%; border:1px solid var(--line); border-radius:10px; display:block; }}
    figcaption {{ color:var(--muted); font-size:.85rem; margin-top:.55rem; }}
    h2 {{ font-size:1.5rem; letter-spacing:-.02em; margin:3.2rem 0 .8rem; }}
    p {{ max-width:760px; }} p.note {{ color:var(--muted); font-size:.9rem; }}
    table {{ border-collapse:collapse; width:100%; max-width:760px; margin:1.2rem 0; font-size:.92rem; }}
    th, td {{ text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-weight:600; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }}
    td.num {{ font-variant-numeric:tabular-nums; text-align:right; }}
    td.c {{ color:var(--accent); }} td.o {{ color:var(--amber); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:1rem; margin:1.6rem 0; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; }}
    .card .big {{ font-size:1.7rem; font-weight:700; letter-spacing:-.03em; color:var(--accent); }}
    .card .lbl {{ color:var(--muted); font-size:.82rem; margin-top:.2rem; }}
    footer {{ border-top:1px solid var(--line); margin-top:4rem; padding:1.6rem 0 2.4rem; color:var(--muted); font-size:.85rem; }}
  </style>
</head>
<body>
<div class="wrap">
  <nav>
    <b>Track 05 · Differentiable graphics &amp; rendering</b>
    <div>
      <a href="#chain">the chain</a>
      <a href="#results">results</a>
      <a href="#renderer">the renderer</a>
      <a href="{REPO_URL}">repository</a>
    </div>
  </nav>

  <header>
    <div class="eyebrow">Tesseract Hackathon 2026</div>
    <h1>See the heat.<br>Invert the physics.</h1>
    <p class="lede">A thermal camera never sees temperature — it sees band-integrated
      Planck radiance through emissivity, optics, and a sensor. We built that camera as a
      <b>differentiable JAX Tesseract</b> and welded it to a coupled
      <b>Fortran&thinsp;/&thinsp;JAX&thinsp;/&thinsp;PyTorch</b> cold-plate equilibrium.
      Then we asked one noisy rendered frame a question no framework can answer alone:
      <b>which part of the chip is overheating?</b> The gradient that answers it runs from
      pixels, back through the renderer, through an implicit-function-theorem adjoint
      spanning three AD systems and hand-written Fortran, to the volumetric heat source.</p>
  </header>

  <figure>
    <img src="assets/hero.png" alt="True source, coupled temperature field, rendered thermal image, recovered source">
    <figcaption>Left to right: the hidden heat source; the coupled flow–viscosity–temperature
      equilibrium it drives; the noisy LWIR frame the camera records (digital counts); and the
      source recovered by gradient descent through the whole chain.</figcaption>
  </figure>

  <div class="cards">
    <div class="card"><div class="big">{g['best_rel_err']:.1e}</div>
      <div class="lbl">end-to-end gradient vs finite differences (pixels &rarr; q), verdict {g['verdict']}</div></div>
    <div class="card"><div class="big">{rc['adjoint_matvecs_total']}</div>
      <div class="lbl">adjoint GMRES matvecs across {b['iters']} iterations — each one a VJP chain over three containers</div></div>
    <div class="card"><div class="big">{loss_gap:,.0f}&times;</div>
      <div class="lbl">final image-space loss gap: coupled adjoint vs frozen-viscosity gradient, same data</div></div>
    <div class="card"><div class="big">4</div>
      <div class="lbl">Tesseracts: Fortran Darcy flow, PyTorch viscosity closure, JAX heat transport, JAX thermal camera</div></div>
  </div>

  <h2 id="chain">The gradient no single framework can trace</h2>
  <p>The temperature field solves a fixed point spanning a PyTorch viscosity closure, a
     Fortran Darcy solver with a pen-and-paper adjoint, and a JAX heat-transport model.
     The camera renders that equilibrium to counts. Differentiating the composition uses
     only each component's VJP endpoint: GMRES solves the adjoint system matrix-free, one
     three-container VJP chain per matvec, then one heat-transport VJP lands the gradient
     on q.</p>
  <figure><img src="assets/chain.png" alt="Differentiable chain diagram"></figure>

  <h2 id="results">Results: the coupling is load-bearing</h2>
  <p>Two recoveries, identical data, identical optimizer. The only difference is whether
     the gradient respects the two-way viscosity&ndash;flow&ndash;temperature coupling or
     freezes it (the standard one-way approximation).</p>
  <table>
    <tr><th>metric</th><th style="text-align:right">coupled adjoint</th><th style="text-align:right">one-way</th></tr>
    {rows}
  </table>
  <p class="note">Grid {b['grid'][0]}&times;{b['grid'][1]} plate, {b['sensor'][0]}&times;{b['sensor'][1]}
     sensor, &sigma;<sub>noise</sub> = {b['noise_counts']} counts, seeds pinned; protocol committed
     before the run (writeup/PROTOCOL.md). Declared criterion: {b['success_criteria']['declared']} —
     met: {b['success_criteria']['met']}.</p>
  <figure><img src="assets/recovery_convergence.png" alt="Convergence, coupled vs one-way"></figure>
  {gif_html}

  <h2 id="renderer">The renderer is physics, not a colormap</h2>
  <p>Planck spectral radiance integrated over the 8&ndash;14&nbsp;&mu;m LWIR band by
     Gauss&ndash;Legendre quadrature, grey-body emission plus reflected ambient
     (Kirchhoff), a sensor-to-plate homography with differentiable bilinear sampling, a
     Gaussian PSF whose width is itself a parameter, cos<sup>4</sup> vignetting, then gain
     and offset to digital counts. Every stage moves the measured image; every stage is
     differentiable. Camera parameters are recoverable too: experiment&nbsp;A re-estimates
     emissivity, PSF width, gain and offset from one image (emissivity RMSE
     {arm0['eps_rmse']:.3f}, gain to {100 * arm0['gain_rel_err']:.1f}% at zero noise).</p>
  <figure><img src="assets/radiometry.png" alt="Planck curve and per-stage ablation strip"></figure>

  <footer>
    Tesseract Hackathon 2026 &middot; Track 05 &middot;
    <a href="{REPO_URL}">source</a> &middot;
    <a href="{REPO_URL}/blob/main/writeup/PROTOCOL.md">pre-registered protocol</a> &middot;
    Apache-2.0. Every figure on this page regenerates from committed artifacts with
    <code>make figures landing</code>.
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
