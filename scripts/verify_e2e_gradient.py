# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""THE correctness gate for this repository.

Finite-difference check of the END-TO-END gradient: a scalar loss on rendered
PIXELS, differentiated back through the thermal-camera Tesseract (JAX), the
coupled flow-viscosity-heat equilibrium (Fortran hand adjoint + PyTorch tape +
JAX implicit diff, chained matrix-free through the implicit function theorem),
down to the volumetric heat source q(x, y).

If this disagrees with finite differences, nothing else here means anything.

Run:  python3 scripts/verify_e2e_gradient.py --nx 16 --ny 8
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
from coupler.thermography import ThermographyForward, two_blob_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--n-u", type=int, default=48)
    ap.add_argument("--n-v", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--one-way", action="store_true",
                    help="check the frozen-feedback gradient instead (only "
                    "consistent when the physics itself is one-way, i.e. "
                    "--coupling-scale 0)")
    ap.add_argument("--coupling-scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="figures/e2e_gradient_check.json")
    args = ap.parse_args()

    plate = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil",
                      coupling_scale=args.coupling_scale)
    filt = DensityFilter(plate.shape, plate.filter_radius_cells)
    rng = np.random.default_rng(args.seed)
    gamma = filt.forward(rng.uniform(0.2, 0.8, plate.shape))
    eps = 0.80 + 0.15 * rng.random(plate.shape)

    q_scale = plate.q_peak
    q0 = two_blob_source(plate.shape, q_scale)
    # a synthetic "measured" image from a nearby q, so the loss is not at zero
    d = rng.normal(size=plate.shape)
    d /= np.linalg.norm(d)

    t0 = time.time()
    with coupled_session(plate) as system, camera_session(
        dict(n_u=args.n_u, n_v=args.n_v, t_ambient=295.0)
    ) as cam:
        fwd = ThermographyForward(
            system=system, camera=cam, gamma=gamma, eps=eps,
            psf_sigma=1.2, gain=25.0, offset=500.0, t_init=plate.t_in,
            fp_tol=1e-11, fp_maxiter=200,
        )
        st_meas = fwd.solve(q0 * 1.15)
        y_meas = fwd.render(st_meas)
        fwd._T_warm = None  # measurement must not warm-start the check

        loss0, grad_q, info = fwd.loss_and_grad_q(q0, y_meas, one_way=args.one_way)
        st0 = info["state"]
        print(f"coupled solve: {st0.iterations} iters, converged={st0.converged}")
        print(f"  T range {st0.T.min():.2f} .. {st0.T.max():.2f} K")
        print(f"  adjoint matvecs {info['adjoint_matvecs']}, "
              f"converged={info['adjoint_converged']}")
        print(f"  loss {loss0:.6e}  |dJ/dq| {np.linalg.norm(grad_q):.3e}")

        directional_ad = float(np.sum(grad_q * d))

        rows = []
        # eps in physical q units: q_scale ~ 1e8, so these are relative ~1e-3..1e-6
        for eps_frac in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6):
            h = eps_frac * q_scale
            # loss only: the FD points do not need the adjoint
            fwd._T_warm = st0.T
            stp = fwd.solve(q0 + h * d)
            rp = fwd.render(stp) - y_meas
            jp = 0.5 * float(np.mean(rp**2))
            fwd._T_warm = st0.T
            stm = fwd.solve(q0 - h * d)
            rm = fwd.render(stm) - y_meas
            jm = 0.5 * float(np.mean(rm**2))
            fd = (jp - jm) / (2 * h)
            rel = abs(directional_ad - fd) / max(abs(fd), 1e-300)
            rows.append({"eps_frac": eps_frac, "fd": fd, "ad": directional_ad,
                         "rel_err": rel})
            print(f"  eps={eps_frac:.0e}*q_peak  ad={directional_ad: .10e}  "
                  f"fd={fd: .10e}  rel={rel:.3e}")

    best = min(r["rel_err"] for r in rows)
    verdict = "PASS" if best < 1e-4 else "FAIL"
    print(f"\n>>> BEST RELATIVE ERROR: {best:.3e}")
    print(f">>> {verdict}")

    out = Path(__file__).resolve().parents[1] / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "grid": [args.nx, args.ny],
        "sensor": [args.n_u, args.n_v],
        "seed": args.seed,
        "coupling_scale": args.coupling_scale,
        "one_way": args.one_way,
        "loss": loss0,
        "grad_norm": float(np.linalg.norm(grad_q)),
        "adjoint_matvecs": info["adjoint_matvecs"],
        "fd_table": rows,
        "best_rel_err": best,
        "verdict": verdict,
        "seconds": time.time() - t0,
    }, indent=2) + "\n")
    print(f"wrote {out}")
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
