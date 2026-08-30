# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Experiment B v2 — amended-protocol rerun of the headline source recovery.

The v1 run (figures/experiment_b.json, kept in the record) FAILED its declared
gate: rel_l2 0.9706 against a target of 0.5.  Diagnosis and the amended,
re-pre-declared protocol are in writeup/PROTOCOL.md ("Amendment v2").  In
brief: v1's flat init carried ~10x the true total power, its 250 Adam
iterations ended with the loss still falling (final pixel RMS 61 counts
against a 2-count noise floor), and the L2-on-Laplacian prior penalises
exactly the sharp hotspot rims the experiment wants back.

v2 changes ONLY the recovery (init 0.05*q_scale, Adam lr 0.2 with cosine
decay, TV prior, more iterations); measurement, physics, seeds, camera, noise
and metrics are identical to v1.  Results go to NEW files —
figures/experiment_b_v2.json / _fields.npz — never over v1's.

Run:  make experiment-b-v2
"""

from __future__ import annotations

import argparse
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
    softplus,
    softplus_grad,
    source_metrics,
    total_variation_penalty,
    two_blob_source,
)

# ---- unchanged from v1 (writeup/PROTOCOL.md, shared configuration) ----------
Q_SCALE = 1.5e8
SIGMA_TRUE, GAIN_TRUE, OFFSET_TRUE = 1.2, 25.0, 500.0
NOISE_COUNTS = 2.0
NOISE_SEED = 42
PROBLEM_SEED = 0

# ---- amended recovery settings (PROTOCOL.md Amendment v2) -------------------
LAMBDA_TV = 3e-3    # set on the 16x8 noise-free sweep before this ran
INIT_FRAC = 0.05    # flat init in units of Q_SCALE (v1's 0.3 carried ~10x the true power)
LR_MAX, LR_MIN = 0.2, 0.02   # cosine decay over the run


def recover(fwd: ThermographyForward, y_meas, shape, *, iters: int,
            one_way: bool, verbose: bool = True) -> dict:
    """Adam on z with q = Q_SCALE * softplus(z), TV prior on q/Q_SCALE."""
    z = np.full(shape, np.log(np.expm1(INIT_FRAC)))
    opt = Adam(lr=LR_MAX)
    history, matvecs = [], 0
    fwd._T_warm = None
    for it in range(iters):
        opt.lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (
            1 + np.cos(np.pi * it / max(iters - 1, 1)))
        q = Q_SCALE * softplus(z)
        data_loss, grad_q, info = fwd.loss_and_grad_q(q, y_meas, one_way=one_way)
        matvecs += info["adjoint_matvecs"]
        tv, tv_grad = total_variation_penalty(q / Q_SCALE)
        loss = data_loss + LAMBDA_TV * tv
        g_q = grad_q + LAMBDA_TV * tv_grad / Q_SCALE
        g_z = g_q * Q_SCALE * softplus_grad(z)
        z = opt.step(z, g_z)
        history.append({"iter": it, "loss": loss, "data_loss": data_loss})
        if verbose and (it % 25 == 0 or it == iters - 1):
            print(f"    it {it:4d}  loss {loss:.6e}  data {data_loss:.6e}  "
                  f"fp_iters {info['state'].iterations}", flush=True)
    return {"q": Q_SCALE * softplus(z), "history": history,
            "adjoint_matvecs_total": matvecs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--out", type=str, default="figures/experiment_b_v2.json")
    args = ap.parse_args()

    rng_problem = np.random.default_rng(PROBLEM_SEED)
    plate_coupled = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil")
    plate_oneway = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil", coupling_scale=0.0)
    filt = DensityFilter(plate_coupled.shape, plate_coupled.filter_radius_cells)
    gamma = filt.forward(rng_problem.uniform(0.2, 0.8, plate_coupled.shape))

    from experiment_a import true_emissivity
    eps = true_emissivity(plate_coupled.shape)
    q_true = two_blob_source(plate_coupled.shape, Q_SCALE)

    t0 = time.time()
    results = {}
    cam_params = dict(n_u=96, n_v=64, t_ambient=295.0)

    # measurement: IDENTICAL to v1 (coupled physics, one image, one noise draw)
    with coupled_session(plate_coupled) as system, camera_session(cam_params) as cam:
        fwd = ThermographyForward(
            system=system, camera=cam, gamma=gamma, eps=eps,
            psf_sigma=SIGMA_TRUE, gain=GAIN_TRUE, offset=OFFSET_TRUE,
            t_init=plate_coupled.t_in, fp_tol=1e-9, fp_maxiter=150,
        )
        st_true = fwd.solve(q_true)
        clean = fwd.render(st_true)
        y_meas = clean + np.random.default_rng(NOISE_SEED).normal(
            0.0, NOISE_COUNTS, clean.shape
        )
        print(f"measurement: T in [{st_true.T.min():.2f}, {st_true.T.max():.2f}] K, "
              f"counts in [{clean.min():.0f}, {clean.max():.0f}]", flush=True)

        print("arm 1: COUPLED", flush=True)
        rec = recover(fwd, y_meas, plate_coupled.shape,
                      iters=args.iters, one_way=False)
        results["coupled"] = {
            **source_metrics(rec["q"], q_true),
            "final_data_loss": rec["history"][-1]["data_loss"],
            "adjoint_matvecs_total": rec["adjoint_matvecs_total"],
        }
        q_coupled = rec["q"]
        hist_coupled = rec["history"]
        st_rec = fwd.solve(q_coupled)
        rendered_rec = fwd.render(st_rec)
        T_rec = st_rec.T

    with coupled_session(plate_oneway) as system1, camera_session(cam_params) as cam1:
        fwd1 = ThermographyForward(
            system=system1, camera=cam1, gamma=gamma, eps=eps,
            psf_sigma=SIGMA_TRUE, gain=GAIN_TRUE, offset=OFFSET_TRUE,
            t_init=plate_oneway.t_in, fp_tol=1e-9, fp_maxiter=150,
        )
        print("arm 2: ONE-WAY (frozen viscosity)", flush=True)
        rec1 = recover(fwd1, y_meas, plate_oneway.shape,
                       iters=args.iters, one_way=True)
        results["one_way"] = {
            **source_metrics(rec1["q"], q_true),
            "final_data_loss": rec1["history"][-1]["data_loss"],
            "adjoint_matvecs_total": rec1["adjoint_matvecs_total"],
        }
        q_oneway = rec1["q"]
        hist_oneway = rec1["history"]

    bias = {
        "rel_l2_coupled": results["coupled"]["rel_l2"],
        "rel_l2_one_way": results["one_way"]["rel_l2"],
        "one_way_over_coupled_l2": results["one_way"]["rel_l2"]
        / max(results["coupled"]["rel_l2"], 1e-300),
        "amplitude_ratio_coupled": results["coupled"]["amplitude_ratio"],
        "amplitude_ratio_one_way": results["one_way"]["amplitude_ratio"],
        "total_power_ratio_coupled": results["coupled"]["total_power_ratio"],
        "total_power_ratio_one_way": results["one_way"]["total_power_ratio"],
    }
    for arm in ("coupled", "one_way"):
        r = results[arm]
        print(f"{arm:8s}: rel_l2 {r['rel_l2']:.4f}  amp {r['amplitude_ratio']:.3f}  "
              f"centroid shift {r['centroid_shift_cells']:.2f} cells  "
              f"power {r['total_power_ratio']:.3f}", flush=True)

    root = Path(__file__).resolve().parents[1]
    np.savez(
        root / "figures" / "experiment_b_v2_fields.npz",
        q_true=q_true, q_coupled=q_coupled, q_oneway=q_oneway,
        y_meas=y_meas, clean=clean, rendered_rec=rendered_rec,
        residual=rendered_rec - y_meas,
        T_true=st_true.T, T_rec=T_rec, gamma=gamma, eps=eps,
        hist_coupled=np.asarray([h["loss"] for h in hist_coupled]),
        hist_coupled_data=np.asarray([h["data_loss"] for h in hist_coupled]),
        hist_oneway=np.asarray([h["loss"] for h in hist_oneway]),
        hist_oneway_data=np.asarray([h["data_loss"] for h in hist_oneway]),
    )
    out = root / args.out
    out.write_text(json.dumps({
        "protocol": "writeup/PROTOCOL.md Amendment v2 (v1 failure kept in figures/experiment_b.json)",
        "grid": [args.nx, args.ny],
        "sensor": [cam_params["n_u"], cam_params["n_v"]],
        "iters": args.iters,
        "prior": {"type": "tv", "lambda_tv": LAMBDA_TV},
        "init_frac": INIT_FRAC,
        "lr": {"max": LR_MAX, "min": LR_MIN, "schedule": "cosine"},
        "q_scale": Q_SCALE,
        "noise_counts": NOISE_COUNTS,
        "noise_seed": NOISE_SEED,
        "problem_seed": PROBLEM_SEED,
        "camera": {"sigma": SIGMA_TRUE, "gain": GAIN_TRUE, "offset": OFFSET_TRUE},
        "results": results,
        "bias": bias,
        "success_criteria": {
            "declared": "coupled rel_l2 < 0.5 and centroid shift < 1.5 cells "
                        "(PROTOCOL.md Amendment v2, declared before this run)",
            "met": bool(results["coupled"]["rel_l2"] < 0.5
                        and results["coupled"]["centroid_shift_cells"] < 1.5),
        },
        "seconds": time.time() - t0,
    }, indent=2) + "\n")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
