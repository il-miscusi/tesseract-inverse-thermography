# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Experiment D: unseen-scene, independent-grid calibration-risk study."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance_nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tesseracts" / "thermal-camera"))

from _render import default_homography
from coupler import ColdPlate, DensityFilter
from coupler.camera import camera_session
from coupler.session import coupled_session
from coupler.thermography import (gradient_wrt_q, softplus, softplus_grad,
                                 source_metrics, total_variation_penalty)
from experiment_a import true_emissivity
from experiment_b_v2 import GAIN_TRUE, NOISE_COUNTS, OFFSET_TRUE, Q_SCALE, SIGMA_TRUE

BANK_SEEDS = tuple(range(101, 113))
DEV_SEEDS = (0, 1, 2)
PROBLEM_SEED = 73
TRAIN_TILTS = (0.22, -0.18)
HOLDOUT_TILT = 0.05
LAMBDA_TV = 3e-4
INIT_FRAC = 0.02


def chip_mask(shape: tuple[int, int], plate: ColdPlate) -> np.ndarray:
    """Cell-centre mask for the known component footprint."""
    x = (np.arange(shape[0]) + 0.5) / shape[0]
    y = (np.arange(shape[1]) + 0.5) / shape[1]
    return ((x[:, None] >= plate.chip_x[0]) & (x[:, None] <= plate.chip_x[1])
            & (y[None, :] >= plate.chip_y[0]) & (y[None, :] <= plate.chip_y[1]))


def block_average(field: np.ndarray, scale: int) -> np.ndarray:
    nx, ny = field.shape
    if nx % scale or ny % scale:
        raise ValueError("field shape must be divisible by scale")
    return field.reshape(nx // scale, scale, ny // scale, scale).mean(axis=(1, 3))


def interior_pixel_mask(homography: np.ndarray, field_shape: tuple[int, int],
                        sensor_shape: tuple[int, int], margin_cells: float = 1.5) -> np.ndarray:
    """Pose-fixed ROI excluding the grid-dependent plate silhouette."""
    n_u, n_v = sensor_shape
    u = (np.arange(n_u) + 0.5) / n_u
    v = (np.arange(n_v) + 0.5) / n_v
    U, V = np.meshgrid(u, v, indexing="ij")
    H = np.asarray(homography, float)
    w = H[2, 0] * U + H[2, 1] * V + H[2, 2]
    x = (H[0, 0] * U + H[0, 1] * V + H[0, 2]) / w
    y = (H[1, 0] * U + H[1, 1] * V + H[1, 2]) / w
    mx, my = margin_cells / field_shape[0], margin_cells / field_shape[1]
    return (x >= mx) & (x <= 1.0 - mx) & (y >= my) & (y <= 1.0 - my)


def fault_scene(shape: tuple[int, int], seed: int, plate: ColdPlate) -> tuple[np.ndarray, list[dict]]:
    """Continuous random component faults, clipped to the physical chip."""
    rng = np.random.default_rng(seed)
    x = (np.arange(shape[0]) + 0.5) / shape[0]
    y = (np.arange(shape[1]) + 0.5) / shape[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    q = np.zeros(shape)
    blobs = []
    for _ in range(int(rng.integers(2, 5))):
        cx = float(rng.uniform(plate.chip_x[0] + 0.02, plate.chip_x[1] - 0.02))
        cy = float(rng.uniform(plate.chip_y[0] + 0.015, plate.chip_y[1] - 0.015))
        wx = float(rng.uniform(0.018, 0.045))
        wy = float(rng.uniform(0.012, 0.035))
        amp = float(rng.uniform(0.65, 1.55))
        q += amp * Q_SCALE * np.exp(-0.5 * (((X - cx) / wx) ** 2 + ((Y - cy) / wy) ** 2))
        blobs.append({"cx": cx, "cy": cy, "wx": wx, "wy": wy, "amplitude": amp})
    return q * chip_mask(shape, plate), blobs


def residual_diagnostics(residual: np.ndarray, mask: np.ndarray | None = None) -> dict:
    r = np.asarray(residual, float)
    valid = np.ones_like(r, dtype=bool) if mask is None else np.asarray(mask, bool)
    values = r[valid]
    mean = float(values.mean())
    centred = r - mean

    def corr(a, b, pair_mask):
        av, bv = a[pair_mask], b[pair_mask]
        denom = float(np.sqrt(np.sum(av**2) * np.sum(bv**2)))
        return float(np.sum(av * bv) / max(denom, 1e-300))

    hmask = valid[1:, :] & valid[:-1, :]
    vmask = valid[:, 1:] & valid[:, :-1]

    return {
        "rms_counts": float(np.sqrt(np.mean(values**2))),
        "mean_counts": mean,
        "lag1_horizontal": corr(centred[1:, :], centred[:-1, :], hmask),
        "lag1_vertical": corr(centred[:, 1:], centred[:, :-1], vmask),
        "pixels": int(valid.sum()),
    }


def physical_metrics(q_rec: np.ndarray, q_true: np.ndarray, plate: ColdPlate) -> dict:
    base = source_metrics(q_rec, q_true)
    nx, ny = q_true.shape
    coords = np.stack(np.meshgrid((np.arange(nx) + 0.5) * plate.dx * 1e3,
                                  (np.arange(ny) + 0.5) * plate.dy * 1e3,
                                  indexing="ij"), axis=-1).reshape(-1, 2)
    wr = np.maximum(q_rec.ravel(), 0.0)
    wt = np.maximum(q_true.ravel(), 0.0)
    wasserstein = float(wasserstein_distance_nd(coords, coords, u_weights=wr, v_weights=wt))
    pr = np.unravel_index(int(np.argmax(q_rec)), q_rec.shape)
    pt = np.unravel_index(int(np.argmax(q_true)), q_true.shape)
    peak_mm = float(np.hypot((pr[0] - pt[0]) * plate.dx,
                             (pr[1] - pt[1]) * plate.dy) * 1e3)
    # Convert the existing Euclidean cell-index centroid metric using the
    # conservative larger cell dimension; also store exact physical centroids.
    def centroid_mm(q):
        w = np.maximum(q, 0.0)
        m = max(float(w.sum()), 1e-300)
        return np.asarray([(w * coords[:, 0].reshape(q.shape)).sum() / m,
                           (w * coords[:, 1].reshape(q.shape)).sum() / m])
    centroid_mm = float(np.linalg.norm(centroid_mm(q_rec) - centroid_mm(q_true)))
    return {**base, "centroid_error_mm": centroid_mm,
            "peak_error_mm": peak_mm, "wasserstein_mm": wasserstein,
            "total_power_rel_error": abs(base["total_power_ratio"] - 1.0)}


class MultiViewForward:
    def __init__(self, system, cameras, masks, gamma, eps, psf_sigma, gain,
                 offset, t_init, temperature_correction=None):
        self.system, self.cameras = system, cameras
        self.masks = masks
        self.gamma, self.eps = gamma, eps
        self.psf_sigma, self.gain, self.offset = psf_sigma, gain, offset
        self.t_init, self._T_warm = t_init, None
        self.temperature_correction = (np.zeros_like(gamma) if temperature_correction is None
                                       else np.asarray(temperature_correction, float))

    def solve(self, q):
        self.system.heat.params["q_source"] = np.asarray(q, float)
        st = self.system.solve(self.gamma, T0=self._T_warm, tol=1e-9,
                               maxiter=180, t_init=self.t_init)
        if not st.converged:
            raise RuntimeError("coupled fixed point did not converge")
        self._T_warm = st.T
        return st

    def render(self, st):
        observed_T = st.T + self.temperature_correction
        return [cam.apply(observed_T, self.eps, self.psf_sigma, self.gain, self.offset)
                for cam in self.cameras]

    def loss_and_grad_q(self, q, measurements):
        st = self.solve(q)
        images = self.render(st)
        dJ_dT = np.zeros_like(st.T)
        loss = 0.0
        for cam, mask, image, measured in zip(self.cameras, self.masks, images, measurements):
            residual = image - measured
            loss += 0.5 * float(np.mean(residual[mask]**2)) / len(images)
            cot = np.zeros_like(residual)
            cot[mask] = residual[mask] / (len(images) * int(mask.sum()))
            observed_T = st.T + self.temperature_correction
            dJ_dT += cam.vjp(observed_T, self.eps, self.psf_sigma, self.gain,
                             self.offset, cot, wrt=("T",))["T"]
        grad, matvecs, ok = gradient_wrt_q(self.system, self.gamma, st, dJ_dT)
        if not ok:
            raise RuntimeError("implicit adjoint GMRES did not converge")
        return loss, grad, {"state": st, "images": images, "matvecs": matvecs}


def recover(fwd: MultiViewForward, measurements, mask, maxiter: int) -> dict:
    support = np.flatnonzero(mask.ravel())
    z0 = np.full(support.size, np.log(np.expm1(INIT_FRAC)))
    history, snapshots = [], []
    state = {"nfev": 0, "matvecs": 0, "best_loss": np.inf,
             "best_q": None, "last_snapshot_eval": -10}

    def unpack(z):
        q = np.zeros(mask.size)
        q[support] = Q_SCALE * softplus(z)
        return q.reshape(mask.shape)

    def fg(z):
        q = unpack(z)
        data_loss, grad_q, info = fwd.loss_and_grad_q(q, measurements)
        tv, tv_grad = total_variation_penalty(q / Q_SCALE)
        loss = data_loss + LAMBDA_TV * tv
        gq = grad_q + LAMBDA_TV * tv_grad / Q_SCALE
        grad_z = gq.ravel()[support] * Q_SCALE * softplus_grad(z)
        state["nfev"] += 1
        state["matvecs"] += info["matvecs"]
        history.append({"evaluation": state["nfev"], "loss": float(loss),
                        "data_loss": float(data_loss),
                        "gradient_norm": float(np.linalg.norm(grad_z))})
        if loss < state["best_loss"]:
            state["best_loss"], state["best_q"] = float(loss), q.copy()
            if state["nfev"] - state["last_snapshot_eval"] >= 10:
                snapshots.append(q.copy())
                state["last_snapshot_eval"] = state["nfev"]
        if state["nfev"] == 1 or state["nfev"] % 25 == 0:
            print(f"    eval {state['nfev']:4d} loss={loss:.5e} data={data_loss:.5e}", flush=True)
        return loss, grad_z

    result = minimize(fg, z0, jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "maxfun": 3 * maxiter,
                               "gtol": 1e-7, "ftol": 1e-12, "maxls": 30})
    q_final = unpack(result.x)
    if not snapshots or not np.array_equal(snapshots[-1], state["best_q"]):
        snapshots.append(state["best_q"].copy())
    return {"q_best": state["best_q"], "q_final": q_final,
            "history": history, "snapshots": snapshots,
            "matvecs": state["matvecs"],
            "optimizer": {"nit": int(result.nit), "nfev": int(state["nfev"]),
                          "status": int(result.status), "message": str(result.message),
                          "converged": bool(result.status == 0)}}


def camera_spec(arm: str) -> dict:
    if arm == "full":
        return {"psf_sigma": SIGMA_TRUE, "gain": GAIN_TRUE,
                "offset": OFFSET_TRUE, "t_ambient": 295.0,
                "half_fov_tan": 0.45, "tilt_delta": 0.0}
    if arm == "mismatch":
        return {"psf_sigma": 1.1, "gain": 25.25, "offset": 502.0,
                "t_ambient": 296.0, "half_fov_tan": 0.46,
                "tilt_delta": 0.015}
    raise ValueError(f"unknown arm {arm}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("dev", "bank"), required=True)
    parser.add_argument("--arms", nargs="+", default=("full", "mismatch"))
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--truth-scale", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--oracle-only", action="store_true",
                        help="score the area-averaged true source, skip optimization")
    parser.add_argument("--out-dir", default="figures/experiment_d")
    args = parser.parse_args()
    allowed = DEV_SEEDS if args.stage == "dev" else BANK_SEEDS
    if args.seed not in allowed:
        raise SystemExit(f"seed {args.seed} is not in declared {args.stage} seeds {allowed}")
    for arm in args.arms:
        camera_spec(arm)

    t0 = time.time()
    scale = args.truth_scale
    plate = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil")
    truth_plate = ColdPlate(nx=args.nx * scale, ny=args.ny * scale, fluid="oil")
    raw = np.random.default_rng(PROBLEM_SEED).uniform(0.2, 0.8, plate.shape)
    gamma = DensityFilter(plate.shape, plate.filter_radius_cells).forward(raw)
    gamma_truth = np.repeat(np.repeat(gamma, scale, axis=0), scale, axis=1)
    q_truth, blobs = fault_scene(truth_plate.shape, args.seed, truth_plate)
    q_target = block_average(q_truth, scale)
    eps = true_emissivity(plate.shape)
    eps_truth = true_emissivity(truth_plate.shape)
    support = chip_mask(plate.shape, plate)
    sensor = {"n_u": 96, "n_v": 64}

    train_truth_params = [{**sensor, "homography": default_homography(t)} for t in TRAIN_TILTS]
    hold_truth_params = {**sensor, "homography": default_homography(HOLDOUT_TILT)}
    train_masks = [interior_pixel_mask(p["homography"], plate.shape,
                                      (sensor["n_u"], sensor["n_v"]))
                   for p in train_truth_params]
    holdout_mask = interior_pixel_mask(hold_truth_params["homography"], plate.shape,
                                       (sensor["n_u"], sensor["n_v"]))
    with ExitStack() as stack:
        truth_system = stack.enter_context(coupled_session(truth_plate))
        truth_cams = [stack.enter_context(camera_session({**p, "t_ambient": 295.0,
                      "half_fov_tan": 0.45})) for p in train_truth_params]
        hold_truth_cam = stack.enter_context(camera_session({**hold_truth_params,
                      "t_ambient": 295.0, "half_fov_tan": 0.45}))
        truth_system.heat.params["q_source"] = q_truth
        truth_state = truth_system.solve(gamma_truth, t_init=truth_plate.t_in,
                                         tol=1e-9, maxiter=180)
        if not truth_state.converged:
            raise RuntimeError("fine-grid truth solve did not converge")
        clean_train = [cam.apply(truth_state.T, eps_truth, SIGMA_TRUE,
                                 GAIN_TRUE, OFFSET_TRUE) for cam in truth_cams]
        clean_holdout = hold_truth_cam.apply(truth_state.T, eps_truth, SIGMA_TRUE,
                                             GAIN_TRUE, OFFSET_TRUE)
        q_cal_truth = 0.5 * Q_SCALE * chip_mask(truth_plate.shape, truth_plate)
        truth_system.heat.params["q_source"] = q_cal_truth
        calibration_truth_state = truth_system.solve(
            gamma_truth, T0=truth_state.T, t_init=truth_plate.t_in,
            tol=1e-9, maxiter=180)
        if not calibration_truth_state.converged:
            raise RuntimeError("fine-grid calibration solve did not converge")
    rng_noise = np.random.default_rng(10_000 + args.seed)
    measured_train = [im + rng_noise.normal(0.0, NOISE_COUNTS, im.shape)
                      for im in clean_train]
    measured_holdout = clean_holdout + rng_noise.normal(0.0, NOISE_COUNTS,
                                                         clean_holdout.shape)

    q_cal_coarse = block_average(q_cal_truth, scale)
    with coupled_session(plate) as calibration_system:
        calibration_system.heat.params["q_source"] = q_cal_coarse
        calibration_coarse_state = calibration_system.solve(
            gamma, t_init=plate.t_in, tol=1e-9, maxiter=180)
        if not calibration_coarse_state.converged:
            raise RuntimeError("coarse-grid calibration solve did not converge")
    temperature_correction = (block_average(calibration_truth_state.T, scale)
                              - calibration_coarse_state.T)

    # Discretization oracle: even the exact area-averaged source may not be
    # representable by the coarse physics. This separates that floor from
    # optimizer error and is stored for every final scene.
    with ExitStack() as stack:
        oracle_system = stack.enter_context(coupled_session(plate))
        oracle_cams = [stack.enter_context(camera_session({**sensor,
                       "homography": default_homography(t),
                       "t_ambient": 295.0, "half_fov_tan": 0.45}))
                       for t in TRAIN_TILTS]
        oracle_hold_cam = stack.enter_context(camera_session({**sensor,
                          "homography": default_homography(HOLDOUT_TILT),
                          "t_ambient": 295.0, "half_fov_tan": 0.45}))
        oracle_fwd = MultiViewForward(oracle_system, oracle_cams, train_masks,
                                      gamma, eps, SIGMA_TRUE, GAIN_TRUE,
                                      OFFSET_TRUE, plate.t_in,
                                      temperature_correction)
        oracle_state = oracle_fwd.solve(q_target)
        oracle_train = oracle_fwd.render(oracle_state)
        oracle_holdout = oracle_hold_cam.apply(
            oracle_state.T + temperature_correction, eps, SIGMA_TRUE,
            GAIN_TRUE, OFFSET_TRUE)
    oracle_train_values = np.concatenate([(r - y)[m] for r, y, m in
                                          zip(oracle_train, measured_train, train_masks)])
    oracle_hold_diag = residual_diagnostics(oracle_holdout - measured_holdout,
                                            holdout_mask)
    oracle = {"train_rms_counts": float(np.sqrt(np.mean(oracle_train_values**2))),
              "holdout_residual": oracle_hold_diag,
              "holdout_rms_noise_sigmas": oracle_hold_diag["rms_counts"] / NOISE_COUNTS}
    print(f"coarse-source oracle: train={oracle['train_rms_counts']:.3f} "
          f"holdout={oracle_hold_diag['rms_counts']:.3f} counts", flush=True)

    results, arrays = {}, {"q_truth": q_truth, "q_target": q_target,
                           "gamma": gamma, "support": support,
                           "temperature_correction": temperature_correction,
                           "q_calibration": q_cal_coarse,
                           "train_masks": np.asarray(train_masks),
                           "holdout_mask": holdout_mask,
                           "measured_holdout": measured_holdout,
                           "clean_holdout": clean_holdout}
    for arm in (() if args.oracle_only else args.arms):
        spec = camera_spec(arm)
        print(f"arm {arm}", flush=True)
        with ExitStack() as stack:
            system = stack.enter_context(coupled_session(plate))
            cams = [stack.enter_context(camera_session({**sensor,
                    "homography": default_homography(t + spec["tilt_delta"]),
                    "t_ambient": spec["t_ambient"],
                    "half_fov_tan": spec["half_fov_tan"]})) for t in TRAIN_TILTS]
            hold_cam = stack.enter_context(camera_session({**sensor,
                    "homography": default_homography(HOLDOUT_TILT + spec["tilt_delta"]),
                    "t_ambient": spec["t_ambient"],
                    "half_fov_tan": spec["half_fov_tan"]}))
            fwd = MultiViewForward(system, cams, train_masks, gamma, eps, spec["psf_sigma"],
                                   spec["gain"], spec["offset"], plate.t_in,
                                   temperature_correction)
            rec = recover(fwd, measured_train, support, args.maxiter)
            st = fwd.solve(rec["q_best"])
            rendered_train = fwd.render(st)
            rendered_holdout = hold_cam.apply(st.T + temperature_correction, eps,
                                               spec["psf_sigma"],
                                               spec["gain"], spec["offset"])
        train_values = np.concatenate([(r - y)[m]
                                       for r, y, m in zip(rendered_train, measured_train,
                                                          train_masks)])
        holdout_residual = rendered_holdout - measured_holdout
        train_diag = {"rms_counts": float(np.sqrt(np.mean(train_values**2))),
                      "mean_counts": float(train_values.mean()),
                      "pixels": int(train_values.size)}
        hold_diag = residual_diagnostics(holdout_residual, holdout_mask)
        metrics = physical_metrics(rec["q_best"], q_target, plate)
        plausible = (hold_diag["rms_counts"] <= 2.0 * NOISE_COUNTS
                     and abs(hold_diag["mean_counts"]) <= 0.25 * NOISE_COUNTS
                     and abs(hold_diag["lag1_horizontal"]) <= 0.10
                     and abs(hold_diag["lag1_vertical"]) <= 0.10)
        useful = (metrics["centroid_error_mm"] <= 1.0
                  and metrics["peak_error_mm"] <= 1.5
                  and metrics["total_power_rel_error"] <= 0.20)
        results[arm] = {**metrics, "train_residual": train_diag,
                        "holdout_residual": hold_diag,
                        "holdout_rms_noise_sigmas": hold_diag["rms_counts"] / NOISE_COUNTS,
                        "absolute_plausible_fit": bool(plausible),
                        "operationally_useful_diagnosis": bool(useful),
                        "optimizer": rec["optimizer"],
                        "adjoint_matvecs_total": rec["matvecs"],
                        "camera_assumption": spec}
        arrays.update({f"q_{arm}": rec["q_best"],
                       f"q_final_{arm}": rec["q_final"],
                       f"rendered_holdout_{arm}": rendered_holdout,
                       f"residual_holdout_{arm}": holdout_residual,
                       f"history_{arm}": np.asarray([h["loss"] for h in rec["history"]]),
                       f"snapshots_{arm}": np.asarray(rec["snapshots"])})
        print(f"  holdout={hold_diag['rms_counts']:.3f} counts "
              f"centroid={metrics['centroid_error_mm']:.3f} mm "
              f"W1={metrics['wasserstein_mm']:.3f} mm plausible={plausible}", flush=True)

    if "full" in results and "mismatch" in results:
        full, mismatch = results["full"], results["mismatch"]
        harmful = (mismatch["absolute_plausible_fit"] and (
            mismatch["centroid_error_mm"] - full["centroid_error_mm"] >= 0.5
            or mismatch["wasserstein_mm"] - full["wasserstein_mm"] >= 0.5
            or mismatch["peak_error_mm"] - full["peak_error_mm"] >= 1.0
            or mismatch["total_power_rel_error"] - full["total_power_rel_error"] >= 0.10))
    else:
        harmful = False

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.stage}_seed_{args.seed}"
    np.savez(out_dir / f"{stem}.npz", **arrays)
    payload = {"protocol": "writeup/EXPERIMENT_D_PROTOCOL.md",
               "stage": args.stage, "seed": args.seed,
               "inverse_grid": list(plate.shape),
               "truth_grid": list(truth_plate.shape),
               "train_tilts": list(TRAIN_TILTS), "holdout_tilt": HOLDOUT_TILT,
               "noise_counts": NOISE_COUNTS, "problem_seed": PROBLEM_SEED,
               "source_blobs": blobs, "support_cells": int(support.sum()),
               "coarse_source_oracle": oracle, "results": results,
               "materially_harmful_plausible_mismatch": bool(harmful),
               "seconds": time.time() - t0}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_dir / f'{stem}.json'} ({payload['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
