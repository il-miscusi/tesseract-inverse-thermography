# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Small-grid sweep harness for Experiment B fixes (diagnostic, not results).

Each run: 16x8 plate, 48x32 camera, coupled physics, configurable init /
optimizer / prior.  Prints metrics; wall clock is the budget, so keep iters low.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coupler import ColdPlate, DensityFilter
from coupler.camera import camera_session
from coupler.session import coupled_session
from coupler.thermography import (
    Adam,
    ThermographyForward,
    smoothness_penalty,
    softplus,
    softplus_grad,
    source_metrics,
    total_variation_penalty,
    two_blob_source,
)
from experiment_a import true_emissivity

Q_SCALE = 1.5e8
SIGMA_TRUE, GAIN_TRUE, OFFSET_TRUE = 1.2, 25.0, 500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--cam", type=int, nargs=2, default=(48, 32))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr-final", type=float, default=None,
                    help="cosine-decay target; None = constant lr")
    ap.add_argument("--init", type=float, default=0.05,
                    help="flat init amplitude in units of Q_SCALE")
    ap.add_argument("--lam-smooth", type=float, default=0.0)
    ap.add_argument("--lam-tv", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--label", type=str, default="sweep")
    args = ap.parse_args()

    plate = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil")
    filt = DensityFilter(plate.shape, plate.filter_radius_cells)
    gamma = filt.forward(np.random.default_rng(0).uniform(0.2, 0.8, plate.shape))
    eps = true_emissivity(plate.shape)
    q_true = two_blob_source(plate.shape, Q_SCALE)
    cam_params = dict(n_u=args.cam[0], n_v=args.cam[1], t_ambient=295.0)

    with coupled_session(plate) as system, camera_session(cam_params) as cam:
        fwd = ThermographyForward(
            system=system, camera=cam, gamma=gamma, eps=eps,
            psf_sigma=SIGMA_TRUE, gain=GAIN_TRUE, offset=OFFSET_TRUE,
            t_init=plate.t_in, fp_tol=1e-9, fp_maxiter=150,
        )
        st = fwd.solve(q_true)
        clean = fwd.render(st)
        y = clean + (np.random.default_rng(42).normal(0, args.noise, clean.shape)
                     if args.noise else 0.0)

        z = np.full(plate.shape, np.log(np.expm1(args.init)))
        opt = Adam(lr=args.lr)
        fwd._T_warm = None
        t0 = time.time()
        for it in range(args.iters):
            if args.lr_final is not None:
                opt.lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (
                    1 + np.cos(np.pi * it / max(args.iters - 1, 1)))
            q = Q_SCALE * softplus(z)
            dl, gq, info = fwd.loss_and_grad_q(q, y, one_way=False)
            g = gq.copy()
            if args.lam_smooth:
                _, spg = smoothness_penalty(q / Q_SCALE)
                g += args.lam_smooth * spg / Q_SCALE
            if args.lam_tv:
                _, tvg = total_variation_penalty(q / Q_SCALE)
                g += args.lam_tv * tvg / Q_SCALE
            z = opt.step(z, g * Q_SCALE * softplus_grad(z))
            if it % 50 == 0 or it == args.iters - 1:
                m = source_metrics(Q_SCALE * softplus(z), q_true)
                print(f"[{args.label}] it {it:4d} data {dl:.4e} rel_l2 {m['rel_l2']:.4f} "
                      f"amp {m['amplitude_ratio']:.3f} cen {m['centroid_shift_cells']:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        m = source_metrics(Q_SCALE * softplus(z), q_true)
        print(f"[{args.label}] FINAL data {dl:.4e} rel_l2 {m['rel_l2']:.4f} "
              f"amp {m['amplitude_ratio']:.3f} cen {m['centroid_shift_cells']:.2f} "
              f"power {m['total_power_ratio']:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
