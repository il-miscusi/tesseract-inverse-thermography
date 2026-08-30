# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Experiment C: is the differentiable LWIR renderer load-bearing?

Observations come from the complete renderer and a 2x finer coupled-physics
grid. Every recovery uses the same coarse coupled physics, source prior,
initial condition, optimizer, data, and budget. Only the assumed camera model
changes. See writeup/RENDERER_PROTOCOL.md, committed before the first run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from coupler import ColdPlate, DensityFilter
from coupler.camera import camera_session
from coupler.session import coupled_session
from coupler.thermography import source_metrics, two_blob_source
from experiment_a import true_emissivity
from experiment_b_v2 import (GAIN_TRUE, INIT_FRAC, LAMBDA_TV, NOISE_COUNTS,
                             NOISE_SEED, OFFSET_TRUE, PROBLEM_SEED, Q_SCALE,
                             SIGMA_TRUE, recover)


def arm_definitions() -> dict[str, dict]:
    """Declared camera assumptions; physics and recovery never vary by arm."""
    return {
        "full": {},
        "blackbody": {"blackbody": True},
        "no_psf": {"psf_sigma": 0.05},
        "no_vignetting": {"half_fov_tan": 0.0},
        "calibration_mismatch": {
            "psf_sigma": 0.9,
            "gain": 26.25,
            "offset": 510.0,
            "t_ambient": 298.0,
            "half_fov_tan": 0.50,
        },
    }


def continuous_fields(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coarse gamma, emissivity, and source from the declared seed."""
    plate = ColdPlate(nx=nx, ny=ny, fluid="oil")
    raw = np.random.default_rng(PROBLEM_SEED).uniform(0.2, 0.8, plate.shape)
    gamma = DensityFilter(plate.shape, plate.filter_radius_cells).forward(raw)
    return gamma, true_emissivity(plate.shape), two_blob_source(plate.shape, Q_SCALE)


def evaluate_criterion(results: dict[str, dict]) -> dict:
    """Apply the protocol's fixed plausible-fit and wrong-diagnosis gates."""
    full = results["full"]
    arms = {}
    for name, r in results.items():
        if name == "full":
            continue
        plausible = r["pixel_rms_counts"] <= 3.0 * full["pixel_rms_counts"]
        wrong_diagnosis = (
            r["rel_l2"] >= 1.25 * full["rel_l2"]
            or r["centroid_shift_cells"] - full["centroid_shift_cells"] >= 0.5
            or abs(r["total_power_ratio"] - full["total_power_ratio"]) >= 0.10
        )
        arms[name] = {
            "plausible_image_fit": bool(plausible),
            "materially_wrong_diagnosis": bool(wrong_diagnosis),
            "passes_both": bool(plausible and wrong_diagnosis),
        }
    passing = [name for name, verdict in arms.items() if verdict["passes_both"]]
    return {
        "diagnostically_load_bearing": bool(passing),
        "passing_arms": passing,
        "arms": arms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--truth-scale", type=int, default=2)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--arms", nargs="+", default=list(arm_definitions()))
    ap.add_argument("--out", default="figures/experiment_c_renderer.json")
    args = ap.parse_args()

    declared = arm_definitions()
    unknown = sorted(set(args.arms) - set(declared))
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)}")

    t0 = time.time()
    gamma, eps, q_true = continuous_fields(args.nx, args.ny)
    scale = args.truth_scale
    truth_shape = (args.nx * scale, args.ny * scale)
    truth_plate = ColdPlate(nx=truth_shape[0], ny=truth_shape[1], fluid="oil")
    gamma_truth = np.repeat(np.repeat(gamma, scale, axis=0), scale, axis=1)
    eps_truth = true_emissivity(truth_shape)
    q_truth_hi = two_blob_source(truth_shape, Q_SCALE)
    sensor = {"n_u": 96, "n_v": 64, "t_ambient": 295.0,
              "half_fov_tan": 0.45}

    # Independent-discretisation observation: 64x32 truth by default, 32x16 inversion.
    with coupled_session(truth_plate) as truth_system, camera_session(sensor) as truth_cam:
        truth_system.heat.params["q_source"] = q_truth_hi
        truth_state = truth_system.solve(gamma_truth, t_init=truth_plate.t_in,
                                         tol=1e-9, maxiter=180)
        if not truth_state.converged:
            raise RuntimeError("fine-grid truth solve did not converge")
        clean = truth_cam.apply(truth_state.T, eps_truth, SIGMA_TRUE,
                                GAIN_TRUE, OFFSET_TRUE)
    y_meas = clean + np.random.default_rng(NOISE_SEED).normal(
        0.0, NOISE_COUNTS, clean.shape
    )

    results: dict[str, dict] = {}
    fields: dict[str, np.ndarray] = {
        "q_true": q_true,
        "q_truth_hi": q_truth_hi,
        "T_truth_hi": truth_state.T,
        "gamma": gamma,
        "gamma_truth": gamma_truth,
        "eps": eps,
        "eps_truth": eps_truth,
        "clean": clean,
        "y_meas": y_meas,
    }

    for name in args.arms:
        spec = declared[name]
        plate = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil")
        cam_fixed = {
            **sensor,
            "t_ambient": spec.get("t_ambient", sensor["t_ambient"]),
            "half_fov_tan": spec.get("half_fov_tan", sensor["half_fov_tan"]),
        }
        eps_arm = np.ones_like(eps) if spec.get("blackbody") else eps
        sigma = spec.get("psf_sigma", SIGMA_TRUE)
        gain = spec.get("gain", GAIN_TRUE)
        offset = spec.get("offset", OFFSET_TRUE)
        print(f"arm: {name}", flush=True)
        with coupled_session(plate) as system, camera_session(cam_fixed) as cam:
            from coupler.thermography import ThermographyForward

            fwd = ThermographyForward(
                system=system, camera=cam, gamma=gamma, eps=eps_arm,
                psf_sigma=sigma, gain=gain, offset=offset,
                t_init=plate.t_in, fp_tol=1e-9, fp_maxiter=150,
            )
            rec = recover(fwd, y_meas, plate.shape, iters=args.iters,
                          one_way=False)
            st = fwd.solve(rec["q"])
            rendered = fwd.render(st)

        metrics = source_metrics(rec["q"], q_true)
        residual = rendered - y_meas
        results[name] = {
            **metrics,
            "pixel_rms_counts": float(np.sqrt(np.mean(residual**2))),
            "final_data_loss": float(rec["history"][-1]["data_loss"]),
            "optimizer": rec["lbfgs"],
            "adjoint_matvecs_total": rec["adjoint_matvecs_total"],
            "camera_assumption": spec,
        }
        fields[f"q_{name}"] = rec["q"]
        fields[f"rendered_{name}"] = rendered
        fields[f"residual_{name}"] = residual
        fields[f"history_{name}"] = np.asarray([h["loss"] for h in rec["history"]])
        print(f"  rel_l2={metrics['rel_l2']:.4f}  "
              f"centroid={metrics['centroid_shift_cells']:.3f} cells  "
              f"power={metrics['total_power_ratio']:.3f}  "
              f"pixel_rms={results[name]['pixel_rms_counts']:.3f}", flush=True)

    full = results.get("full")
    if full:
        for name, r in results.items():
            r["rel_l2_over_full"] = r["rel_l2"] / max(full["rel_l2"], 1e-300)
            r["centroid_excess_over_full_cells"] = (
                r["centroid_shift_cells"] - full["centroid_shift_cells"]
            )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_name(out.stem + "_fields.npz"), **fields)
    payload = {
        "protocol": "writeup/RENDERER_PROTOCOL.md",
        "inverse_grid": [args.nx, args.ny],
        "truth_grid": list(truth_shape),
        "sensor": [sensor["n_u"], sensor["n_v"]],
        "optimizer": {"method": "L-BFGS-B", "maxiter": args.iters},
        "prior": {"type": "tv", "lambda_tv": LAMBDA_TV},
        "init_frac": INIT_FRAC,
        "noise_counts": NOISE_COUNTS,
        "noise_seed": NOISE_SEED,
        "problem_seed": PROBLEM_SEED,
        "truth_camera": {"psf_sigma": SIGMA_TRUE, "gain": GAIN_TRUE,
                         "offset": OFFSET_TRUE, **sensor},
        "results": results,
        "criterion": evaluate_criterion(results),
        "seconds": time.time() - t0,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} ({payload['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
