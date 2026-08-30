# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Publication figures for the inverse-thermography submission.

Everything is parameterised on the artifact paths, so regenerating against the
final experiment run is one command:

    python3 scripts/make_figures.py \
        --fields figures/experiment_b_fields.npz \
        --results figures/experiment_b.json \
        --expa figures/experiment_a.json \
        --outdir figures

Outputs: hero.png, chain.png, recovery_convergence.png, radiometry.png.

The hero's flow streamlines need a velocity field.  If the fields NPZ carries
``ux``/``uy`` they are used directly; otherwise, when the coupler stack is
importable (COUPLER_INPROCESS=1 and DARCY_SOLVER_BIN set), ONE step of the
already-converged equilibrium reconstructs them (mu = N(T*), u = F(gamma, mu));
failing both, the panel falls back to material-density contours and says so.
The radiometry ablation strip re-renders the true temperature field through the
thermal-camera Tesseract's own JAX renderer with stages toggled; without JAX it
degrades to the Planck panel alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle
from figstyle import (ACCENT, ACCENT2, GRID, INK, MUTED, PANEL, TRUTH,
                      colorbar)

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ field ops
def show_field(ax, f, cmap, vmin=None, vmax=None):
    """Plate fields are indexed [ix, iy]; draw with x horizontal, y up."""
    return ax.imshow(np.asarray(f).T, origin="lower", cmap=cmap,
                     vmin=vmin, vmax=vmax, aspect="auto",
                     extent=[0, 1, 0, 1], interpolation="nearest")


def strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


def reconstruct_flow(fields) -> tuple[np.ndarray, np.ndarray] | None:
    """(ux, uy) on the plate grid, or None when it cannot be had cheaply."""
    if "ux" in fields and "uy" in fields:
        return np.asarray(fields["ux"]), np.asarray(fields["uy"])
    try:
        from coupler import ColdPlate
        from coupler.session import coupled_session

        nx, ny = fields["T_true"].shape
        plate = ColdPlate(nx=nx, ny=ny, fluid="oil")
        with coupled_session(plate) as system:
            out = system.step(np.asarray(fields["gamma"]),
                              np.asarray(fields["T_true"]))
        return np.asarray(out["ux"]), np.asarray(out["uy"])
    except Exception as e:  # noqa: BLE001 - any failure means "fall back"
        print(f"  flow reconstruction unavailable ({type(e).__name__}: {e}); "
              "hero falls back to material contours")
        return None


# ------------------------------------------------------------------ 1. hero
def make_hero(fields, results, out: Path) -> None:
    figstyle.use()
    q_true = np.asarray(fields["q_true"]) / 1e6      # W/m^3 -> MW/m^3
    q_rec = np.asarray(fields["q_coupled"]) / 1e6
    T = np.asarray(fields["T_true"])
    y = np.asarray(fields["y_meas"])
    flow = reconstruct_flow(fields)

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4))
    qmax = max(q_true.max(), q_rec.max())

    ax = axes[0]
    im = show_field(ax, q_true, figstyle.CMAP_SOURCE, vmin=0, vmax=qmax)
    ax.set_title("hidden heat source  $q(x,y)$", pad=8)
    colorbar(fig, im, ax, "MW m$^{-3}$")

    ax = axes[1]
    im = show_field(ax, T, figstyle.CMAP_TEMP)
    if flow is not None:
        ux, uy = flow
        nx, ny = ux.shape
        X, Y = np.meshgrid((np.arange(nx) + .5) / nx, (np.arange(ny) + .5) / ny,
                           indexing="ij")
        ax.streamplot(X.T, Y.T, ux.T, uy.T, color=ACCENT, linewidth=0.7,
                      density=0.9, arrowsize=0.7)
        ax.set_title("coupled equilibrium  $T^*$ + coolant flow", pad=8)
    else:
        g = np.asarray(fields["gamma"])
        ax.contour(np.linspace(0, 1, g.shape[0]), np.linspace(0, 1, g.shape[1]),
                   g.T, levels=4, colors=ACCENT, linewidths=0.6, alpha=0.8)
        ax.set_title("coupled equilibrium  $T^*$ + material $\\gamma$", pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    colorbar(fig, im, ax, "K")

    ax = axes[2]
    im = ax.imshow(y.T, origin="lower", cmap=figstyle.CMAP_THERMAL,
                   aspect="auto", extent=[0, 1, 0, 1], interpolation="nearest")
    ax.set_title("what the camera sees (96$\\times$64, noisy)", pad=8)
    colorbar(fig, im, ax, "counts")

    ax = axes[3]
    # its own scale: the recovery is judged by shape and location, and a
    # shared scale would hide a low-amplitude reconstruction entirely
    im = show_field(ax, q_rec, figstyle.CMAP_SOURCE, vmin=0)
    r = results["results"]["coupled"]
    ax.set_title("recovered source (coupled adjoint, own scale)", pad=8)
    ax.text(0.03, 0.05,
            f"rel $L_2$ {r['rel_l2']:.2f} · centroid {r['centroid_shift_cells']:.1f} cells",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=GRID))
    colorbar(fig, im, ax, "MW m$^{-3}$")

    for ax in axes:
        strip_axes(ax)
    for i in range(3):
        fig.text(0.253 + 0.247 * i, 0.5, "$\\rightarrow$", fontsize=16,
                 color=MUTED, ha="center", va="center")
    fig.suptitle(
        "One noisy thermal image $\\rightarrow$ the heat source that made it:  gradient descent "
        "through a differentiable LWIR camera and a Fortran/JAX/PyTorch equilibrium",
        fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ 2. chain
def make_chain(results, out: Path) -> None:
    """The differentiable chain, drawn to match coupler/thermography.py:
    forward q -> [N -> F -> H fixed point] -> camera -> counts; reverse
    dJ/dcounts -> camera VJP -> GMRES on (I - dG/dT)^T (one VJP_H·VJP_F·VJP_N
    chain per matvec) -> heat VJP wrt q."""
    figstyle.use()
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, title, badge, badge_color, lines=()):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6,rounding_size=1.2",
                     fc=PANEL, ec=GRID, lw=1.1))
        ax.text(x + 1.4, y + h - 2.6, title, ha="left", fontsize=9.5,
                color=INK, weight="bold")
        ax.text(x + w - 1.2, y + 2.0, badge, ha="right", va="center",
                fontsize=6.5, color="#0b0e14", weight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc=badge_color, ec="none"))
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.2 - 3.2 * i, ln, ha="center",
                    fontsize=7.4, color=MUTED)

    def arrow(p, q, color, style="-", lw=1.4, rad=0.0, label=None, dy=1.6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                     color=color, lw=lw, linestyle=style,
                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((p[0] + q[0]) / 2, max(p[1], q[1]) + dy, label,
                    ha="center", fontsize=7.2, color=color)

    # the coupled loop: three Tesseracts around a fixed point
    box(4, 22, 17, 13, "viscosity closure", "PyTorch", "#ee4c2c",
        ["$\\mu = N(T)$", "autograd VJP"])
    box(25, 22, 17, 13, "Darcy flow", "Fortran", "#734f96",
        ["$u = F(\\gamma,\\mu)$", "hand-derived adjoint"])
    box(46, 22, 17, 13, "heat transport", "JAX", "#2a7de1",
        ["$T = H(\\gamma,u;\\,q)$", "autodiff VJP"])
    box(70, 22, 26, 13, "thermal camera", "JAX", "#2a7de1",
        ["Planck 8–14 μm · emissivity · homography",
         "PSF · vignetting · gain/offset"])

    # forward arrows
    arrow((21, 28.5), (25, 28.5), INK)
    arrow((42, 28.5), (46, 28.5), INK)
    arrow((63, 28.5), (70, 28.5), INK, label="$T^*$")
    # fixed-point feedback H -> N (arcs below the loop)
    arrow((54, 21.4), (13, 21.4), MUTED, rad=-0.28, lw=1.1)
    ax.text(33.5, 13.6, "fixed point  $T = G(\\gamma,T;q) = H(\\gamma, F(\\gamma, N(T)); q)$",
            ha="center", fontsize=8, color=MUTED)
    # inputs / outputs
    ax.text(1.5, 37.8, "$q(x,y)$", fontsize=10, color=TRUTH, weight="bold")
    arrow((6, 37.5), (52, 35.6), TRUTH, rad=-0.18, lw=1.3)
    ax.text(98.5, 28.5, "counts", fontsize=9, color=INK, ha="left", va="center")
    arrow((96, 28.5), (98, 28.5), INK)

    # reverse path
    rc = ACCENT
    ax.text(98.5, 8.0, "$J$ (pixel loss)", fontsize=8.5, color=rc, ha="left")
    arrow((97, 9.5), (90, 9.5), rc)
    box(60, 4, 30, 9, "camera VJP", "JAX", "#2a7de1",
        ["$\\partial J/\\partial T^*$ from one pixel cotangent"])
    box(22, 4, 34, 9, "implicit-function-theorem adjoint", "GMRES", ACCENT,
        ["$(I - \\partial G/\\partial T)^\\top \\lambda = \\partial J/\\partial T^*$",
         "matrix-free: VJP$_H$ · VJP$_F$ · VJP$_N$ per matvec"])
    arrow((60, 8.5), (56, 8.5), rc)
    arrow((22, 8.5), (14, 8.5), rc,
          label="$\\partial J/\\partial q = (\\partial H/\\partial q)^\\top\\lambda$",
          dy=2.2)
    ax.text(6.5, 8.5, "$\\nabla_{\\!q} J$", fontsize=10, color=rc,
            weight="bold", ha="center", va="center")

    mv = results["results"]["coupled"]["adjoint_matvecs_total"]
    iters = results["iters"]
    ax.text(39, 0.6,
            f"{mv} adjoint matvecs over {iters} optimizer iterations "
            f"(≈{mv / iters:.1f} per gradient) · no framework ever traces the whole chain",
            ha="center", fontsize=7.6, color=MUTED)
    ax.set_title("The gradient no single AD framework can trace", fontsize=12.5, pad=14)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------ 3. recovery convergence
def make_convergence(fields, results, out: Path) -> None:
    figstyle.use()
    hc = np.asarray(fields["hist_coupled"])
    ho = np.asarray(fields["hist_oneway"])
    rc = results["results"]["coupled"]
    ro = results["results"]["one_way"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.semilogy(hc, color=ACCENT, lw=1.6, label="coupled adjoint")
    ax1.semilogy(ho, color=ACCENT2, lw=1.6, label="one-way (frozen viscosity)")
    ax1.set_xlabel("Adam iteration")
    ax1.set_ylabel("objective  $\\frac{1}{2}$ mean$(r^2)$ + smoothness  [counts$^2$]")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper right")
    gap = ro["final_data_loss"] / rc["final_data_loss"]
    ax1.annotate(f"final data loss:\n{gap:.0f}× apart",
                 xy=(len(hc) - 1, hc[-1]), xytext=(0.55, 0.45),
                 textcoords="axes fraction", fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax1.set_title("Same data, same optimizer — only the physics in the gradient differs")

    labels = ["rel $L_2$", "centroid shift\n(cells)", "total power\nratio"]
    cv = [rc["rel_l2"], rc["centroid_shift_cells"], rc["total_power_ratio"]]
    ov = [ro["rel_l2"], ro["centroid_shift_cells"], ro["total_power_ratio"]]
    x = np.arange(3)
    ax2.bar(x - 0.19, cv, 0.36, color=ACCENT, label="coupled")
    ax2.bar(x + 0.19, ov, 0.36, color=ACCENT2, label="one-way")
    ax2.axhline(1.0, color=TRUTH, lw=0.9, ls="--")
    ax2.text(2.42, 1.0, "ideal\npower", fontsize=6.8, color=TRUTH, va="center")
    ax2.set_xticks(x, labels, fontsize=7.6)
    ax2.set_ylabel("source-recovery metric  [–]")
    ax2.grid(True, axis="y", alpha=0.5)
    ax2.legend(fontsize=7.5)
    ax2.set_title("Recovery error by arm")
    fig.suptitle("Recovering $q$ from one image: coupled vs one-way gradients",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------- 4. radiometry
def make_radiometry(fields, results, out: Path) -> None:
    figstyle.use()
    cam = results["camera"]
    T = np.asarray(fields["T_true"])
    eps = np.asarray(fields["eps"])
    t_lo, t_hi = float(T.min()), float(T.max())

    try:
        cam_dir = ROOT / "tesseracts" / "thermal-camera"
        if str(cam_dir) not in sys.path:
            sys.path.insert(0, str(cam_dir))
        import _render as R
        have_jax = True
    except Exception as e:  # noqa: BLE001
        print(f"  JAX renderer unavailable ({e}); radiometry gets the Planck panel only")
        have_jax = False

    if have_jax:
        fig = plt.figure(figsize=(12.8, 6.4))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.3)
        axP = fig.add_subplot(gs[0, :2])
        axE = fig.add_subplot(gs[0, 2:])
        strip = [fig.add_subplot(gs[1, i]) for i in range(4)]
    else:
        fig, axP = plt.subplots(figsize=(7, 4.2))
        axE = None
        strip = []

    # Planck band radiance vs temperature
    Ts = np.linspace(200, 500, 400)
    if have_jax:
        L = np.asarray(R.band_radiance(Ts))
    else:
        lam = np.linspace(8e-6, 14e-6, 512)
        c1, c2 = 1.191042972e-16, 1.438776877e-2
        B = c1 / (lam[None, :] ** 5
                  * np.expm1(c2 / (lam[None, :] * Ts[:, None])))
        L = np.trapezoid(B, lam, axis=1)
    axP.plot(Ts, L, color=ACCENT, lw=1.8)
    axP.axvspan(t_lo, t_hi, color=ACCENT, alpha=0.14)
    axP.text((t_lo + t_hi) / 2, L.max() * 0.06,
             f"plate operating range\n{t_lo:.0f}–{t_hi:.0f} K",
             ha="center", fontsize=7.5, color=INK)
    axP.set_xlabel("temperature  $T$  [K]")
    axP.set_ylabel("band radiance $L_{8\\mathrm{-}14\\mu m}(T)$  [W m$^{-2}$ sr$^{-1}$]")
    axP.grid(True, alpha=0.5)
    axP.set_title("Planck radiance, Gauss–Legendre integrated over the LWIR band")

    if have_jax:
        import jax.numpy as jnp
        Hm = R.default_homography()
        common = dict(Hm=jnp.asarray(Hm), t_ambient=295.0, n_u=96, n_v=64)
        sig, g, o = cam["sigma"], cam["gain"], cam["offset"]

        # emissivity map panel
        im = show_field(axE, eps, "cividis")
        axE.set_title("surface emissivity map $\\epsilon(x,y)$ — what makes\n"
                      "counts ≠ temperature", fontsize=9.5)
        strip_axes(axE)
        colorbar(fig, im, axE, "ε  [–]")

        variants = [
            ("full model", dict(eps=eps, psf_sigma=sig, vig=0.45)),
            ("$\\epsilon = 1$ (no emissivity)", dict(eps=np.ones_like(eps), psf_sigma=sig, vig=0.45)),
            ("no PSF blur", dict(eps=eps, psf_sigma=0.05, vig=0.45)),
            ("no vignetting", dict(eps=eps, psf_sigma=sig, vig=0.0)),
        ]
        imgs = []
        for _, v in variants:
            img = np.asarray(R.render(
                jnp.asarray(T), jnp.asarray(v["eps"]), jnp.asarray(v["psf_sigma"]),
                jnp.asarray(g), jnp.asarray(o),
                half_fov_tan=v["vig"], **common))
            imgs.append(img)
        vmin = min(i.min() for i in imgs)
        vmax = max(i.max() for i in imgs)
        for ax, (name, _), img in zip(strip, variants, imgs):
            im = ax.imshow(img.T, origin="lower", cmap=figstyle.CMAP_THERMAL,
                           vmin=vmin, vmax=vmax, aspect="auto",
                           extent=[0, 1, 0, 1], interpolation="nearest")
            ax.set_title(name, fontsize=8.6)
            if name != "full model":
                d = float(np.abs(img - imgs[0]).max())
                ax.text(0.03, 0.05, f"max |Δ| = {d:.0f} counts",
                        transform=ax.transAxes, fontsize=7, color=INK,
                        bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=GRID))
            strip_axes(ax)
        colorbar(fig, im, strip[-1], "counts")
        fig.suptitle("The renderer is physics, not a colormap: each optical stage "
                     "moves the measured counts — and each is differentiable",
                     fontsize=12, y=0.97)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="figures/experiment_b_fields.npz")
    ap.add_argument("--results", default="figures/experiment_b.json")
    ap.add_argument("--expa", default="figures/experiment_a.json")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", default=None,
                    help="hero | chain | convergence | radiometry")
    args = ap.parse_args()

    fields = np.load(ROOT / args.fields)
    results = json.loads((ROOT / args.results).read_text())
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "hero": lambda: make_hero(fields, results, outdir / "hero.png"),
        "chain": lambda: make_chain(results, outdir / "chain.png"),
        "convergence": lambda: make_convergence(fields, results,
                                                outdir / "recovery_convergence.png"),
        "radiometry": lambda: make_radiometry(fields, results,
                                              outdir / "radiometry.png"),
    }
    for name, job in jobs.items():
        if args.only and name != args.only:
            continue
        job()


if __name__ == "__main__":
    main()
