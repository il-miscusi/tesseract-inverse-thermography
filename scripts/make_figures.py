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
from figstyle import (ACCENT, ACCENT2, FAINT, GRID, INK, MUTED, PANEL, TRUTH,
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
    """One cinematic strip: the inversion story, left to right."""
    figstyle.use()
    q_true = np.asarray(fields["q_true"]) / 1e6      # W/m^3 -> MW/m^3
    q_rec = np.asarray(fields["q_coupled"]) / 1e6
    T = np.asarray(fields["T_true"])
    y = np.asarray(fields["y_meas"])
    flow = reconstruct_flow(fields)
    r = results["results"]["coupled"]
    qmax = max(q_true.max(), q_rec.max())

    # The true blobs sit in the chip band at the base of the plate
    # (two_blob_source y_frac 0.06): crop both source panels to that band so
    # the panel shows structure, not empty domain.  The crop is labelled.
    band = max(4, int(round(0.375 * q_true.shape[1])))

    fig = plt.figure(figsize=(16.4, 5.6))
    gs = fig.add_gridspec(1, 4, left=0.025, right=0.955, top=0.735,
                          bottom=0.155, wspace=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    def show_band(ax, f, **kw):
        return ax.imshow(np.asarray(f)[:, :band].T, origin="lower",
                         cmap=figstyle.CMAP_SOURCE, aspect="auto",
                         extent=[0, 1, 0, 1], interpolation="nearest", **kw)

    titles = [
        ("hidden heat source", "$q(x,y)$ · chip band"),
        ("coupled equilibrium", "$T^*$ with coolant flow"),
        ("what the camera sees", "96×64 LWIR counts, noisy"),
        ("recovered source", "coupled adjoint · chip band"),
    ]

    ax = axes[0]
    show_band(ax, q_true, vmin=0, vmax=qmax)
    ax.text(0.965, 0.90, "scale shared with d", transform=ax.transAxes,
            fontsize=7, color=MUTED, ha="right")

    ax = axes[1]
    imT = show_field(ax, T, figstyle.CMAP_TEMP)
    if flow is not None:
        ux, uy = flow
        nx, ny = ux.shape
        X, Y = np.meshgrid((np.arange(nx) + .5) / nx, (np.arange(ny) + .5) / ny,
                           indexing="ij")
        ax.streamplot(X.T, Y.T, ux.T, uy.T, color=ACCENT, linewidth=0.7,
                      density=0.9, arrowsize=0.7)
    else:
        g = np.asarray(fields["gamma"])
        ax.contour(np.linspace(0, 1, g.shape[0]), np.linspace(0, 1, g.shape[1]),
                   g.T, levels=4, colors=ACCENT, linewidths=0.6, alpha=0.75)
        titles[1] = ("coupled equilibrium", "$T^*$ with material $\\gamma$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    figstyle.inset_cbar(fig, imT, ax, "K")

    ax = axes[2]
    imY = ax.imshow(y.T, origin="lower", cmap=figstyle.CMAP_THERMAL,
                    aspect="auto", extent=[0, 1, 0, 1], interpolation="nearest")
    figstyle.inset_cbar(fig, imY, ax, "counts")

    ax = axes[3]
    # shared scale with the truth panel: the v2 recovery holds amplitude
    # (ratio ~1.1), so the honest comparison is also the strongest one
    imQ = show_band(ax, q_rec, vmin=0, vmax=qmax)
    figstyle.stamp(ax, f"rel $L_2$ {r['rel_l2']:.2f} · "
                       f"centroid {r['centroid_shift_cells']:.1f} cells",
                   x=0.05, y=0.08, color=INK, edge=ACCENT)
    figstyle.inset_cbar(fig, imQ, ax, "MW m$^{-3}$")

    for i, (ax, (t, sub)) in enumerate(zip(axes, titles)):
        strip_axes(ax)
        figstyle.panel_letter(ax, "abcd"[i], x=0.0, y=1.16)
        ax.text(0.055, 1.155, t, transform=ax.transAxes, fontsize=11,
                fontweight="medium", color=INK, va="bottom")
        ax.text(0.055, 1.045, sub, transform=ax.transAxes, fontsize=8.5,
                color=MUTED, va="bottom")

    # subtle labeled connectors between stages, in the footer band
    stage = ["physics  ·  flow–heat fixed point", "render  ·  LWIR camera",
             "invert  ·  $\\nabla_{\\!q}J$ by adjoint"]
    for i in range(3):
        x0 = 0.025 + (0.955 - 0.025) * (i + 0.86) / 4
        x1 = 0.025 + (0.955 - 0.025) * (i + 1.14) / 4
        y0 = 0.083
        col = figstyle.GLOW if i == 2 else MUTED
        ar = FancyArrowPatch((x0, y0), (x1, y0), transform=fig.transFigure,
                             arrowstyle="-|>", mutation_scale=11,
                             color=col, lw=1.2, shrinkA=0, shrinkB=0)
        fig.add_artist(ar)
        fig.text((x0 + x1) / 2, y0 - 0.045, stage[i], fontsize=8,
                 color=col, ha="center", va="top")

    figstyle.headline(
        fig,
        "One noisy thermal image is enough to find the heat source that made it",
        "gradient descent through a differentiable LWIR camera and a "
        "Fortran / JAX / PyTorch flow–heat equilibrium — pixels to source, "
        "end to end",
        x=0.025, y=0.975, sub_dy=0.068, size=16.5)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ 2. chain
def make_chain(results, out: Path) -> None:
    """The differentiable chain, drawn to match coupler/thermography.py:
    forward q -> [N -> F -> H fixed point] -> camera -> counts; reverse
    dJ/dcounts -> camera VJP -> GMRES on (I - dG/dT)^T (one VJP_H·VJP_F·VJP_N
    chain per matvec) -> heat VJP wrt q."""
    import matplotlib.patheffects as pe

    figstyle.use()
    fig, ax = plt.subplots(figsize=(16.0, 7.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.845, bottom=0.075)
    ax.set_xlim(0, 100); ax.set_ylim(0, 59); ax.axis("off")

    GLOW = figstyle.GLOW

    def chip(x, y, text, color, fs=6.4):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color="#0b0a10", weight="bold",
                bbox=dict(boxstyle="round,pad=0.32", fc=color, ec="none",
                          alpha=0.92))

    def box(x, y, w, h, title, lines=(), edge=GRID, lw=1.1, title_color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.5,rounding_size=1.4",
                     fc=PANEL, ec=edge, lw=lw))
        ax.text(x + w / 2, y + h - 2.9, title, ha="center", fontsize=10,
                color=title_color, weight="bold")
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.6 - 3.3 * i, ln, ha="center",
                    fontsize=7.6, color=MUTED)

    def arrow(p, q, color, lw=1.4, rad=0.0, label=None, dy=1.7,
              glow=False, ls="-", fs=7.6):
        a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                            color=color, lw=lw, linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}",
                            capstyle="round")
        if glow:
            a.set_path_effects([
                pe.Stroke(linewidth=lw + 6, foreground=color, alpha=0.13),
                pe.Stroke(linewidth=lw + 2.6, foreground=color, alpha=0.28),
                pe.Normal(),
            ])
        ax.add_patch(a)
        if label:
            ax.text((p[0] + q[0]) / 2, max(p[1], q[1]) + dy, label,
                    ha="center", fontsize=fs, color=color)

    # ---- forward render path (muted ink), top band ----------------------
    fy, fh = 33, 14
    ax.text(2.8, fy + fh + 3.2, "forward render", fontsize=8.5, color=MUTED,
            ha="left", style="italic")
    box(6, fy, 16.5, fh, "viscosity closure",
        ["$\\mu = N(T)$", "autograd VJP"])
    chip(20.2, fy + 2.1, "PyTorch", "#ee6c4c")
    box(27.5, fy, 16.5, fh, "Darcy flow",
        ["$u = F(\\gamma,\\mu)$", "hand-derived adjoint"])
    chip(41.7, fy + 2.1, "Fortran", "#a08bd0")
    box(49, fy, 16.5, fh, "heat transport",
        ["$T = H(\\gamma,u;\\,q)$", "autodiff VJP"])
    chip(63.2, fy + 2.1, "JAX", "#5c9de1")
    box(71.5, fy, 22.5, fh, "thermal camera",
        ["Planck 8–14 μm · emissivity", "homography · PSF · vignette",
         "gain / offset"])
    chip(91.7, fy + 2.1, "JAX", "#5c9de1")

    fmid = fy + fh / 2
    ax.text(2.6, fmid + 3.2, "$q(x,y)$", fontsize=11, color=TRUTH,
            weight="bold", ha="center")
    arrow((2.6, fmid + 1.8), (5.4, fmid + 0.4), TRUTH, rad=0.15, lw=1.3)
    arrow((22.5, fmid), (27.5, fmid), MUTED)
    arrow((44, fmid), (49, fmid), MUTED)
    arrow((65.5, fmid), (71.5, fmid), MUTED, label="$T^*$", dy=1.2, fs=8.5)
    arrow((94, fmid), (97.6, fmid), MUTED)
    ax.text(97.9, fmid - 3.0, "pixels", fontsize=9.5, color=INK, ha="right",
            weight="bold")
    # fixed-point feedback H -> N, arced above the loop
    arrow((57, fy + fh + 0.6), (14, fy + fh + 0.6), FAINT, rad=0.16, lw=1.0,
          ls=(0, (4, 3)))
    ax.text(35.5, fy + fh + 8.6,
            "fixed point   $T = G(\\gamma,T;q) = H(\\gamma,\\,F(\\gamma,\\,N(T));\\,q)$",
            ha="center", fontsize=8.6, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.35", fc=figstyle.BG, ec="none"))

    # ---- reverse gradient path (luminous amber), bottom band — the claim
    ry, rh = 8, 13
    ax.text(97.5, ry + rh + 3.4, "reverse — pixels back to source",
            fontsize=8.5, color=GLOW, ha="right", style="italic")
    ax.text(97.9, ry + rh / 2 + 3.6, "$J$  (pixel loss)", fontsize=9,
            color=GLOW, ha="right")
    box(63, ry, 26, rh, "camera VJP",
        ["one pixel cotangent in,", "$\\partial J/\\partial T^*$ out"],
        edge=GLOW, lw=1.3, title_color=GLOW)
    chip(86.8, ry + 2.0, "JAX", "#5c9de1")
    box(21, ry, 36, rh, "implicit-function-theorem adjoint",
        ["$(I - \\partial G/\\partial T)^{\\!\\top}\\lambda = \\partial J/\\partial T^*$",
         "matrix-free GMRES · VJP$_H$ · VJP$_F$ · VJP$_N$ per matvec"],
        edge=GLOW, lw=1.3, title_color=GLOW)
    chip(53.6, ry + 2.0, "GMRES", ACCENT)

    rmid = ry + rh / 2
    arrow((96.5, rmid + 2.2), (89, rmid), GLOW, rad=0.18, lw=1.8, glow=True)
    arrow((63, rmid), (57, rmid), GLOW, lw=1.8, glow=True)
    arrow((21, rmid), (12.2, rmid), GLOW, lw=1.8, glow=True,
          label="$(\\partial H/\\partial q)^{\\!\\top}\\lambda$", dy=2.0, fs=8)
    ax.text(7.2, rmid, "$\\nabla_{\\!q} J$", fontsize=13, color=GLOW,
            weight="bold", ha="center", va="center")
    # the drop from pixels down into the reverse band
    arrow((97.6, fmid - 4.5), (97.6, ry + rh + 6.5), GLOW, lw=1.6, glow=True)

    mv = results["results"]["coupled"]["adjoint_matvecs_total"]
    evaluations = results["optimizer"]["coupled"]["nfev"]
    ax.text(50, 1.2,
            f"{mv} adjoint matvecs over {evaluations} objective evaluations "
            f"(≈{mv / evaluations:.1f} per gradient) · "
            "no framework ever traces the whole chain",
            ha="center", fontsize=8.2, color=FAINT)

    figstyle.headline(
        fig, "The gradient no single AD framework can trace",
        "three languages around a flow–heat fixed point, one differentiable "
        "camera — and an adjoint that carries pixels back to the hidden source",
        x=0.012, y=0.985, sub_dy=0.048, size=16.5)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4),
                                   gridspec_kw={"width_ratios": [1.6, 1],
                                                "wspace": 0.28})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.735, bottom=0.135)
    ax1.semilogy(hc, color=ACCENT, lw=1.8, solid_capstyle="round",
                 label="coupled adjoint")
    ax1.semilogy(ho, color=ACCENT2, lw=1.8, solid_capstyle="round",
                 label="one-way (frozen viscosity)")
    ax1.legend(frameon=False, fontsize=8.6, loc="upper right",
               handlelength=1.6)
    ax1.set_xlabel("L-BFGS-B objective evaluation")
    ax1.set_ylabel("objective  $\\frac{1}{2}$ mean$(r^2)$ + TV prior  [counts$^2$]")
    ax1.grid(True, alpha=0.45)
    ax1.spines[["top", "right"]].set_visible(False)
    gap = ro["final_data_loss"] / rc["final_data_loss"]
    ax1.annotate(f"final recorded data loss:  {gap:.2f}× apart",
                 xy=(len(hc) - 1, hc[-1]), xytext=(0.32, 0.58),
                 textcoords="axes fraction", fontsize=8.2, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax1.set_title("Optimization trace — same data, same recovery setup",
                  loc="left", fontsize=10.5, pad=10)

    labels = ["rel $L_2$", "centroid shift\n(cells)", "total power\nratio"]
    cv = [rc["rel_l2"], rc["centroid_shift_cells"], rc["total_power_ratio"]]
    ov = [ro["rel_l2"], ro["centroid_shift_cells"], ro["total_power_ratio"]]
    x = np.arange(3)
    ax2.bar(x - 0.19, cv, 0.36, color=ACCENT, label="coupled")
    ax2.bar(x + 0.19, ov, 0.36, color=ACCENT2, label="one-way")
    ax2.axhline(1.0, color=TRUTH, lw=0.9, ls="--")
    ax2.text(0.55, 1.035, "ideal power", fontsize=7, color=TRUTH,
             va="bottom", ha="center")
    ax2.set_xticks(x, labels, fontsize=7.8)
    ax2.set_ylim(0, 1.28)
    ax2.set_ylabel("source-recovery metric  [–]")
    ax2.grid(True, axis="y", alpha=0.45)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=7.8, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 1.0))
    ax2.set_title("Recovery error by arm", loc="left", fontsize=10.5, pad=10)
    figstyle.panel_letter(ax1, "a", x=-0.075, y=1.03)
    figstyle.panel_letter(ax2, "b", x=-0.13, y=1.03)
    figstyle.headline(
        fig, "Recovering $q$ from one image: coupled vs frozen-viscosity models",
        "the coupled adjoint holds amplitude and placement; freezing the "
        "viscosity loop degrades both",
        x=0.03, y=0.975, sub_dy=0.062, size=13.5)
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
        fig = plt.figure(figsize=(13.6, 7.4))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.52,
                              wspace=0.3, top=0.80, bottom=0.06,
                              left=0.06, right=0.965)
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
    axP.grid(True, alpha=0.45)
    axP.spines[["top", "right"]].set_visible(False)
    axP.set_title("Planck radiance, Gauss–Legendre integrated over the LWIR band",
                  loc="left", fontsize=10, pad=9)

    if have_jax:
        import jax.numpy as jnp
        Hm = R.default_homography()
        common = dict(Hm=jnp.asarray(Hm), t_ambient=295.0, n_u=96, n_v=64)
        sig, g, o = cam["sigma"], cam["gain"], cam["offset"]

        # emissivity map panel
        im = show_field(axE, eps, "cividis")
        axE.set_title("surface emissivity map $\\epsilon(x,y)$ — what makes "
                      "counts ≠ temperature", loc="left", fontsize=10, pad=9)
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
                figstyle.stamp(ax, f"max |Δ| = {d:.0f} counts", x=0.04,
                               y=0.06, fontsize=7)
            strip_axes(ax)
        figstyle.inset_cbar(fig, im, strip[-1], "counts")
        figstyle.panel_letter(axP, "a", x=-0.09, y=1.04)
        figstyle.panel_letter(axE, "b", x=-0.03, y=1.04)
        figstyle.panel_letter(strip[0], "c", x=-0.03, y=1.06)
        figstyle.headline(
            fig, "The renderer is physics, not a colormap",
            "each optical stage moves the measured counts — and each is "
            "differentiable, so the inversion can see through all of them",
            x=0.06, y=0.975, sub_dy=0.045, size=15)
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
