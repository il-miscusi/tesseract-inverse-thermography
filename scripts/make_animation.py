# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Animate the source recovery: truth | current estimate | shrinking residual.

Two ways in:

  * ``--snapshots path.npz`` — per-iteration snapshots from a real run.
    Contract: ``q_true (nx,ny)``, ``q_snaps (K,nx,ny)``, ``resid_snaps
    (K,n_u,n_v)``, ``losses (K,)``, ``iters (K,)``.
  * ``--demo`` — no snapshots on disk yet: run a small noise-free recovery
    in-process (16x8 plate, 48x32 sensor, ~60 iterations, minutes on a
    laptop; needs COUPLER_INPROCESS=1 and DARCY_SOLVER_BIN) and write the
    snapshots NPZ itself, then animate it.

Outputs ``figures/recovery.mp4`` (ffmpeg, when present) and
``figures/recovery.gif`` (PIL, always; kept under 10 MB for README embedding).

    python3 scripts/make_animation.py --demo
    python3 scripts/make_animation.py --snapshots figures/recovery_snapshots.npz
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle
from figstyle import ACCENT, GRID, INK, MUTED, PANEL, colorbar

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ demo run
def run_demo(out_npz: Path, *, nx=16, ny=8, n_u=48, n_v=32, iters=60,
             q_scale=1.5e8) -> Path:
    """A fast, noise-free, coupled recovery that records every iteration."""
    from coupler import ColdPlate, DensityFilter
    from coupler.camera import camera_session
    from coupler.session import coupled_session
    from coupler.thermography import (Adam, ThermographyForward,
                                      smoothness_penalty, softplus,
                                      softplus_grad, two_blob_source)

    rng = np.random.default_rng(0)
    plate = ColdPlate(nx=nx, ny=ny, fluid="oil")
    filt = DensityFilter(plate.shape, plate.filter_radius_cells)
    gamma = filt.forward(rng.uniform(0.2, 0.8, plate.shape))
    eps = 0.94 - 0.18 * (np.arange(nx * ny).reshape(nx, ny) % 7 == 3)
    q_true = two_blob_source(plate.shape, q_scale)

    cam_params = dict(n_u=n_u, n_v=n_v, t_ambient=295.0)
    q_snaps, resid_snaps, losses, its = [], [], [], []
    with coupled_session(plate) as system, camera_session(cam_params) as cam:
        fwd = ThermographyForward(
            system=system, camera=cam, gamma=gamma, eps=eps,
            psf_sigma=1.2, gain=25.0, offset=500.0,
            t_init=plate.t_in, fp_tol=1e-9, fp_maxiter=150,
        )
        y_meas = fwd.render(fwd.solve(q_true))  # noise-free demo measurement
        z = np.full(plate.shape, np.log(np.expm1(0.3)))
        opt = Adam(lr=0.1)
        lam_s = 1e-4
        fwd._T_warm = None
        for it in range(iters):
            q = q_scale * softplus(z)
            data_loss, grad_q, info = fwd.loss_and_grad_q(q, y_meas)
            sp, sp_grad = smoothness_penalty(q / q_scale)
            g_z = (grad_q + lam_s * sp_grad / q_scale) * q_scale * softplus_grad(z)
            q_snaps.append(q)
            resid_snaps.append(info["counts"] - y_meas)
            losses.append(data_loss + lam_s * sp)
            its.append(it)
            z = opt.step(z, g_z)
            if it % 10 == 0:
                print(f"  demo it {it:3d}  loss {losses[-1]:.4e}")

    np.savez(out_npz, q_true=q_true, q_snaps=np.asarray(q_snaps),
             resid_snaps=np.asarray(resid_snaps),
             losses=np.asarray(losses), iters=np.asarray(its))
    print(f"wrote {out_npz}")
    return out_npz


# ------------------------------------------------------------------ rendering
def render_frames(snap, frame_dir: Path) -> list[Path]:
    figstyle.use()
    q_true = np.asarray(snap["q_true"]) / 1e6
    q_snaps = np.asarray(snap["q_snaps"]) / 1e6
    resid = np.asarray(snap["resid_snaps"])
    losses = np.asarray(snap["losses"])
    its = np.asarray(snap["iters"])

    qmax = max(q_true.max(), q_snaps.max())
    rmax = float(np.abs(resid[0]).max())
    paths = []
    for k in range(len(its)):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
        ax = axes[0]
        im = ax.imshow(q_true.T, origin="lower", cmap=figstyle.CMAP_SOURCE,
                       vmin=0, vmax=qmax, aspect="auto", extent=[0, 1, 0, 1])
        ax.set_title("true source $q$", pad=6)
        colorbar(fig, im, ax, "MW m$^{-3}$")

        ax = axes[1]
        im = ax.imshow(q_snaps[k].T, origin="lower", cmap=figstyle.CMAP_SOURCE,
                       vmin=0, vmax=qmax, aspect="auto", extent=[0, 1, 0, 1])
        ax.set_title(f"recovered $q$ — iteration {its[k]}", pad=6)
        colorbar(fig, im, ax, "MW m$^{-3}$")

        ax = axes[2]
        im = ax.imshow(resid[k].T, origin="lower", cmap=figstyle.CMAP_DIVERGING,
                       vmin=-rmax, vmax=rmax, aspect="auto", extent=[0, 1, 0, 1])
        ax.set_title("image residual (counts)", pad=6)
        ax.text(0.03, 0.05, f"loss {losses[k]:.3e}", transform=ax.transAxes,
                fontsize=7.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=GRID))
        colorbar(fig, im, ax, "counts")

        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle("Gradient descent through renderer + coupled physics",
                     fontsize=11.5, y=1.02, color=INK)
        fig.tight_layout()
        p = frame_dir / f"frame_{k:04d}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    # hold the final frame so the endpoint reads
    paths.extend([paths[-1]] * 8)
    return paths


def encode(paths: list[Path], mp4: Path, gif: Path, fps: int = 8) -> None:
    if shutil.which("ffmpeg"):
        list_file = paths[0].parent / "frames.txt"
        list_file.write_text("".join(
            f"file '{p}'\nduration {1 / fps}\n" for p in paths))
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
             "-r", str(fps), str(mp4)],
            check=True, capture_output=True)
        print(f"wrote {mp4}")
    else:
        print("ffmpeg not found; skipping mp4 (gif still produced)")

    from PIL import Image
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
              for p in paths]
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    size_mb = gif.stat().st_size / 1e6
    print(f"wrote {gif} ({size_mb:.1f} MB)")
    if size_mb >= 10:
        # halve the resolution until it embeds
        while size_mb >= 10:
            frames = [f.resize((f.width * 3 // 4, f.height * 3 // 4))
                      for f in frames]
            frames[0].save(gif, save_all=True, append_images=frames[1:],
                           duration=int(1000 / fps), loop=0, optimize=True)
            size_mb = gif.stat().st_size / 1e6
        print(f"downscaled gif to {size_mb:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=None,
                    help="per-iteration NPZ from a real run")
    ap.add_argument("--demo", action="store_true",
                    help="generate snapshots with a fast in-process recovery")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=60, help="demo iterations")
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.snapshots:
        snap_path = ROOT / args.snapshots
    elif args.demo:
        snap_path = run_demo(outdir / "recovery_snapshots.npz", iters=args.iters)
    else:
        default = outdir / "recovery_snapshots.npz"
        if not default.exists():
            ap.error("no --snapshots given and no figures/recovery_snapshots.npz; "
                     "run with --demo to generate one")
        snap_path = default

    snap = np.load(snap_path)
    with tempfile.TemporaryDirectory() as td:
        paths = render_frames(snap, Path(td))
        encode(paths, outdir / "recovery.mp4", outdir / "recovery.gif",
               fps=args.fps)


if __name__ == "__main__":
    main()
