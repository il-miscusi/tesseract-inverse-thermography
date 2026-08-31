#!/usr/bin/env python3
"""Run the missing coupled-forward / truncated-gradient factorial arm."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from coupler import ColdPlate, DensityFilter
from coupler.camera import camera_session
from coupler.session import coupled_session
from coupler.thermography import (
    ThermographyForward,
    softplus,
    softplus_grad,
    source_metrics,
    total_variation_penalty,
    two_blob_source,
)
from experiment_a import true_emissivity
from experiment_b_v2 import (
    GAIN_TRUE,
    INIT_FRAC,
    LAMBDA_TV,
    NOISE_COUNTS,
    NOISE_SEED,
    OFFSET_TRUE,
    PROBLEM_SEED,
    Q_SCALE,
    SIGMA_TRUE,
)


def objective(fwd, measured, shape):
    state = {"nfev": 0, "best_loss": float("inf"), "best_q": None, "last_grad_norm": None}
    history = []

    def fg(z_flat):
        z = z_flat.reshape(shape)
        q = Q_SCALE * softplus(z)
        data_loss, grad_q, _ = fwd.loss_and_grad_q(q, measured, one_way=True)
        tv, tv_grad = total_variation_penalty(q / Q_SCALE)
        loss = data_loss + LAMBDA_TV * tv
        grad_z = (grad_q + LAMBDA_TV * tv_grad / Q_SCALE) * Q_SCALE * softplus_grad(z)
        state["nfev"] += 1
        state["last_grad_norm"] = float(np.linalg.norm(grad_z))
        history.append({"evaluation": state["nfev"], "loss": float(loss),
                        "data_loss": float(data_loss),
                        "gradient_norm": state["last_grad_norm"]})
        if loss < state["best_loss"]:
            state["best_loss"], state["best_q"] = float(loss), q.copy()
        if state["nfev"] == 1 or state["nfev"] % 25 == 0:
            print(f"  eval {state['nfev']:4d} loss={loss:.6e} data={data_loss:.6e}", flush=True)
        return loss, grad_z.ravel()

    z0 = np.full(shape, np.log(np.expm1(INIT_FRAC)))
    result = minimize(fg, z0.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 250, "maxfun": 750})
    q_final = Q_SCALE * softplus(result.x.reshape(shape))
    return result, q_final, state["best_q"], history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/experiment_e_factorial.json"))
    args = parser.parse_args()
    prior = json.loads((ROOT / "figures/experiment_b_v2.json").read_text())
    expected = {"grid": [32, 16], "sensor": [96, 64], "noise_counts": 2.0,
                "noise_seed": 42, "problem_seed": 0}
    for key, value in expected.items():
        if prior[key] != value:
            raise SystemExit(f"Experiment B v2 {key} changed: {prior[key]} != {value}")

    plate = ColdPlate(nx=32, ny=16, fluid="oil")
    gamma = DensityFilter(plate.shape, plate.filter_radius_cells).forward(
        np.random.default_rng(PROBLEM_SEED).uniform(0.2, 0.8, plate.shape)
    )
    eps = true_emissivity(plate.shape)
    q_true = two_blob_source(plate.shape, Q_SCALE)
    camera_params = {"n_u": 96, "n_v": 64, "t_ambient": 295.0}
    started = time.time()

    with coupled_session(plate) as system, camera_session(camera_params) as camera:
        fwd = ThermographyForward(system, camera, gamma, eps, SIGMA_TRUE,
                                  GAIN_TRUE, OFFSET_TRUE, plate.t_in,
                                  fp_tol=1e-9, fp_maxiter=150)
        clean = fwd.render(fwd.solve(q_true))
        measured = clean + np.random.default_rng(NOISE_SEED).normal(
            0.0, NOISE_COUNTS, clean.shape
        )

        z0 = np.full(plate.shape, np.log(np.expm1(INIT_FRAC)))
        q0 = Q_SCALE * softplus(z0)
        _, g_exact, _ = fwd.loss_and_grad_q(q0, measured, one_way=False)
        fwd._T_warm = None
        _, g_truncated, _ = fwd.loss_and_grad_q(q0, measured, one_way=True)
        cosine = float(np.vdot(g_exact, g_truncated) /
                       (np.linalg.norm(g_exact) * np.linalg.norm(g_truncated)))
        relative_gradient_error = float(
            np.linalg.norm(g_truncated - g_exact) / np.linalg.norm(g_exact)
        )
        fwd._T_warm = None
        result, q_final, q_best, history = objective(fwd, measured, plate.shape)

    payload = {
        "protocol": "writeup/FACTORIAL_PROTOCOL.md",
        "reuses": "figures/experiment_b_v2.json",
        "configuration": expected,
        "initial_gradient": {"cosine_exact_vs_truncated": cosine,
                             "relative_l2_error": relative_gradient_error},
        "coupled_truncated": {
            "final": {**source_metrics(q_final, q_true),
                      "data_loss": history[-1]["data_loss"]},
            "best_evaluated": {**source_metrics(q_best, q_true),
                               "regularized_loss": min(h["loss"] for h in history),
                               "data_loss": history[int(np.argmin([h["loss"] for h in history]))]["data_loss"]},
            "optimizer": {"status": int(result.status), "message": str(result.message),
                          "converged": bool(result.status == 0), "nit": int(result.nit),
                          "nfev": len(history),
                          "final_gradient_norm": history[-1]["gradient_norm"]},
        },
        "reference_arms": prior["results"],
        "history": history,
        "seconds": time.time() - started,
    }
    output = ROOT / args.out
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"initial_gradient": payload["initial_gradient"],
                      "coupled_truncated": payload["coupled_truncated"]}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
