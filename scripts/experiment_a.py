# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Experiment A — camera self-calibration by inverse rendering.

Given ONE noisy synthetic thermal image of a KNOWN temperature field (one
coupled solve at the nominal chip source), recover the per-pixel emissivity
map, the PSF width, and the sensor gain and offset by Adam on the rendering
loss.  Protocol: writeup/PROTOCOL.md (committed before this ran).

This is the component-level inverse problem: it exercises the renderer's VJPs
w.r.t. every camera unknown, before Experiment B asks the physics to join in.

Run:  python3 scripts/experiment_a.py
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
from coupler.thermography import Adam, smoothness_penalty, softplus, softplus_grad


LAMBDA_TV = 3e-2  # protocol: chosen on the noise-free case, reused everywhere


def true_emissivity(shape) -> np.ndarray:
    """Base 0.95 with a rectangular low-e (bare metal) patch of 0.75."""
    nx, ny = shape
    eps = np.full(shape, 0.95)
    x0, x1 = int(0.55 * nx), int(0.85 * nx)
    y0, y1 = int(0.25 * ny), int(0.75 * ny)
    eps[x0:x1, y0:y1] = 0.75
    return eps


def calibrate(cam, T, y_meas, shape, *, iters: int, seed: int) -> dict:
    """Recover (eps, sigma, gain, offset) from one image by Adam."""
    # parametrization per protocol
    z = np.zeros(shape)  # eps = 0.5 + 0.49*tanh(z);  z=0 -> eps=0.5
    z[:] = np.arctanh((0.9 - 0.5) / 0.49)  # init eps = 0.9 uniform
    s_z = np.log(np.expm1(0.8))            # sigma = softplus(s_z), init 0.8
    gain = 20.0
    offset = 400.0

    ad_z = Adam(lr=0.05)
    ad_s = Adam(lr=0.02)
    ad_g = Adam(lr=0.02)
    ad_o = Adam(lr=0.02)

    history = []
    for it in range(iters):
        eps = 0.5 + 0.49 * np.tanh(z)
        sigma = softplus(np.asarray(s_z))
        counts = cam.apply(T, eps, sigma, gain, offset)
        r = counts - y_meas
        data_loss = 0.5 * float(np.mean(r**2))
        tv, tv_grad = smoothness_penalty(eps)
        loss = data_loss + LAMBDA_TV * tv

        cot = r / r.size
        g = cam.vjp(T, eps, sigma, gain, offset, cot)
        g_eps = g["eps"] + LAMBDA_TV * tv_grad
        g_z = g_eps * 0.49 * (1.0 - np.tanh(z) ** 2)
        g_s = float(g["psf_sigma"]) * float(softplus_grad(np.asarray(s_z)))
        # scalar step sizes: normalise by typical magnitudes so one lr fits all
        z = ad_z.step(z, g_z)
        s_z = float(ad_s.step(np.asarray([s_z]), np.asarray([g_s]))[0])
        gain = float(ad_g.step(np.asarray([gain]), np.asarray([float(g["gain"])]))[0])
        offset = float(ad_o.step(np.asarray([offset]), np.asarray([float(g["offset"])]))[0])
        history.append(loss)

    eps = 0.5 + 0.49 * np.tanh(z)
    return dict(eps=eps, sigma=float(softplus(np.asarray(s_z))), gain=gain,
                offset=offset, history=history)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--out", type=str, default="figures/experiment_a.json")
    args = ap.parse_args()

    plate = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil")
    filt = DensityFilter(plate.shape, plate.filter_radius_cells)
    rng_problem = np.random.default_rng(0)
    gamma = filt.forward(rng_problem.uniform(0.2, 0.8, plate.shape))
    eps_true = true_emissivity(plate.shape)
    SIGMA_TRUE, GAIN_TRUE, OFFSET_TRUE = 1.2, 25.0, 500.0

    t0 = time.time()
    with coupled_session(plate) as system, camera_session(
        dict(n_u=96, n_v=64, t_ambient=295.0)
    ) as cam:
        st = system.solve(gamma, tol=1e-11, maxiter=200, t_init=plate.t_in)
        if not st.converged:
            raise RuntimeError("forward physics did not converge")
        T = st.T
        print(f"physics: T in [{T.min():.2f}, {T.max():.2f}] K "
              f"({st.iterations} fixed-point iters)")

        clean = cam.apply(T, eps_true, SIGMA_TRUE, GAIN_TRUE, OFFSET_TRUE)
        arms = []
        for noise in args.noise:
            rng_noise = np.random.default_rng(42)
            y = clean + rng_noise.normal(0.0, noise, clean.shape)
            rec = calibrate(cam, T, y, plate.shape, iters=args.iters, seed=0)
            arm = {
                "noise_counts": noise,
                "eps_rmse": float(np.sqrt(np.mean((rec["eps"] - eps_true) ** 2))),
                "sigma_abs_err": abs(rec["sigma"] - SIGMA_TRUE),
                "gain_rel_err": abs(rec["gain"] - GAIN_TRUE) / GAIN_TRUE,
                "offset_rel_err": abs(rec["offset"] - OFFSET_TRUE) / OFFSET_TRUE,
                # the declared degeneracy check: gain*eps product error
                "gain_eps_rel_err": float(np.mean(np.abs(
                    rec["gain"] * rec["eps"] - GAIN_TRUE * eps_true
                ) / (GAIN_TRUE * eps_true))),
                "final_loss": rec["history"][-1],
                "loss_first": rec["history"][0],
                "sigma": rec["sigma"],
                "gain": rec["gain"],
                "offset": rec["offset"],
            }
            arms.append(arm)
            print(f"noise {noise:5.1f}: eps RMSE {arm['eps_rmse']:.4f}  "
                  f"sigma err {arm['sigma_abs_err']:.4f}  "
                  f"gain rel {arm['gain_rel_err']:.4f}  "
                  f"offset rel {arm['offset_rel_err']:.4f}  "
                  f"gain*eps rel {arm['gain_eps_rel_err']:.4f}")
            if noise == 2.0:
                np.savez(
                    Path(__file__).resolve().parents[1] / "figures" / "experiment_a_fields.npz",
                    eps_true=eps_true, eps_rec=rec["eps"], T=T,
                    y_meas=y, clean=clean, history=np.asarray(rec["history"]),
                )

    out = Path(__file__).resolve().parents[1] / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "grid": [args.nx, args.ny],
        "sensor": [96, 64],
        "iters": args.iters,
        "lambda_tv": LAMBDA_TV,
        "true": {"sigma": SIGMA_TRUE, "gain": GAIN_TRUE, "offset": OFFSET_TRUE},
        "init": {"eps": 0.9, "sigma": 0.8, "gain": 20.0, "offset": 400.0},
        "noise_seed": 42,
        "problem_seed": 0,
        "arms": arms,
        "seconds": time.time() - t0,
    }, indent=2) + "\n")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
