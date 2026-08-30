# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Diagnostic probes for Experiment B's failed recovery (not a results script).

Probe 1: noise-free recovery on a small grid, many iterations -> identifiability
         ceiling with the current optimizer.
Probe 2: init AT the true source -> does the optimizer stay?  (separates
         "optimizer walks away" from "cannot get there".)
Probe 3: gradient magnitude audit: data term vs prior term at flat init.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    two_blob_source,
)

Q_SCALE = 1.5e8
LAMBDA_S = 1e-4
SIGMA_TRUE, GAIN_TRUE, OFFSET_TRUE = 1.2, 25.0, 500.0


def run_probe(nx, ny, iters, *, lam=LAMBDA_S, lr=0.1, init="flat",
              noise=0.0, cam=(48, 32), label=""):
    plate = ColdPlate(nx=nx, ny=ny, fluid="oil")
    filt = DensityFilter(plate.shape, plate.filter_radius_cells)
    gamma = filt.forward(np.random.default_rng(0).uniform(0.2, 0.8, plate.shape))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_a import true_emissivity
    eps = true_emissivity(plate.shape)
    q_true = two_blob_source(plate.shape, Q_SCALE)
    cam_params = dict(n_u=cam[0], n_v=cam[1], t_ambient=295.0)
    with coupled_session(plate) as system, camera_session(cam_params) as camc:
        fwd = ThermographyForward(
            system=system, camera=camc, gamma=gamma, eps=eps,
            psf_sigma=SIGMA_TRUE, gain=GAIN_TRUE, offset=OFFSET_TRUE,
            t_init=plate.t_in, fp_tol=1e-9, fp_maxiter=150,
        )
        st = fwd.solve(q_true)
        clean = fwd.render(st)
        y = clean + (np.random.default_rng(42).normal(0, noise, clean.shape)
                     if noise else 0.0)

        if init == "flat":
            z = np.full(plate.shape, np.log(np.expm1(0.3)))
        else:  # truth
            z = np.log(np.expm1(np.maximum(q_true / Q_SCALE, 1e-6)))
        opt = Adam(lr=lr)
        fwd._T_warm = None
        t0 = time.time()
        for it in range(iters):
            q = Q_SCALE * softplus(z)
            dl, gq, info = fwd.loss_and_grad_q(q, y, one_way=False)
            sp, spg = smoothness_penalty(q / Q_SCALE)
            gz = (gq + lam * spg / Q_SCALE) * Q_SCALE * softplus_grad(z)
            if it == 0:
                print(f"  [{label}] it0: data_loss {dl:.4e}  "
                      f"|gq*Qs| max {np.abs(gq*Q_SCALE).max():.3e}  "
                      f"|prior gz| max {np.abs(lam*spg*softplus_grad(z)).max():.3e}")
            z = opt.step(z, gz)
            if it % 100 == 0 or it == iters - 1:
                m = source_metrics(Q_SCALE * softplus(z), q_true)
                print(f"  [{label}] it {it:4d}  data {dl:.4e}  rel_l2 {m['rel_l2']:.4f} "
                      f"amp {m['amplitude_ratio']:.3f}  ({time.time()-t0:.0f}s)")
        q_fin = Q_SCALE * softplus(z)
        m = source_metrics(q_fin, q_true)
        print(f"  [{label}] FINAL data {dl:.4e} rel_l2 {m['rel_l2']:.4f} "
              f"amp {m['amplitude_ratio']:.3f} centroid {m['centroid_shift_cells']:.2f}")
        return m, dl


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        print("PROBE 1: 16x8, noise-free, 600 iters, flat init")
        run_probe(16, 8, 600, label="p1")
    if which in ("all", "2"):
        print("PROBE 2: 16x8, noise-free, 200 iters, TRUTH init")
        run_probe(16, 8, 200, init="truth", label="p2")
