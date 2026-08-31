# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Calibrate the fine-minus-coarse temperature discrepancy on chip-cell loads."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from coupler import ColdPlate, DensityFilter
from coupler.session import coupled_session
from experiment_b_v2 import Q_SCALE
from experiment_d_generalization import PROBLEM_SEED, block_average, chip_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--truth-scale", type=int, default=2)
    parser.add_argument("--out", default="figures/experiment_d/multifidelity_calibration.npz")
    args = parser.parse_args()
    t0 = time.time()
    scale = args.truth_scale
    plate = ColdPlate(nx=args.nx, ny=args.ny, fluid="oil")
    truth_plate = ColdPlate(nx=args.nx * scale, ny=args.ny * scale, fluid="oil")
    raw = np.random.default_rng(PROBLEM_SEED).uniform(0.2, 0.8, plate.shape)
    gamma = DensityFilter(plate.shape, plate.filter_radius_cells).forward(raw)
    gamma_truth = np.repeat(np.repeat(gamma, scale, axis=0), scale, axis=1)
    support = chip_mask(plate.shape, plate)
    support_flat = np.flatnonzero(support.ravel())

    with ExitStack() as stack:
        coarse = stack.enter_context(coupled_session(plate))
        fine = stack.enter_context(coupled_session(truth_plate))
        coarse.heat.params["q_source"] = np.zeros(plate.shape)
        fine.heat.params["q_source"] = np.zeros(truth_plate.shape)
        st0c = coarse.solve(gamma, t_init=plate.t_in, tol=1e-9, maxiter=180)
        st0f = fine.solve(gamma_truth, t_init=truth_plate.t_in, tol=1e-9, maxiter=180)
        if not st0c.converged or not st0f.converged:
            raise RuntimeError("baseline calibration solve did not converge")
        offset = block_average(st0f.T, scale) - st0c.T
        basis = []
        for k, flat in enumerate(support_flat):
            i, j = np.unravel_index(int(flat), plate.shape)
            qc = np.zeros(plate.shape)
            qf = np.zeros(truth_plate.shape)
            qc[i, j] = Q_SCALE
            qf[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = Q_SCALE
            coarse.heat.params["q_source"] = qc
            fine.heat.params["q_source"] = qf
            stc = coarse.solve(gamma, T0=st0c.T, t_init=plate.t_in,
                               tol=1e-9, maxiter=180)
            stf = fine.solve(gamma_truth, T0=st0f.T, t_init=truth_plate.t_in,
                             tol=1e-9, maxiter=180)
            if not stc.converged or not stf.converged:
                raise RuntimeError(f"basis calibration solve {k} did not converge")
            discrepancy = block_average(stf.T, scale) - stc.T
            basis.append(discrepancy - offset)
            print(f"basis {k + 1:2d}/{support_flat.size} cell=({i},{j}) "
                  f"max|dT|={np.max(np.abs(basis[-1])):.4f} K", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, offset=offset, basis=np.asarray(basis),
             support_flat=support_flat, gamma=gamma)
    payload = {"protocol": "writeup/EXPERIMENT_D_PROTOCOL.md",
               "problem_seed": PROBLEM_SEED, "coarse_grid": list(plate.shape),
               "truth_grid": list(truth_plate.shape), "q_scale": Q_SCALE,
               "support_cells": int(support_flat.size),
               "basis_shape": list(np.asarray(basis).shape),
               "max_abs_offset_K": float(np.max(np.abs(offset))),
               "max_abs_basis_K": float(np.max(np.abs(basis))),
               "seconds": time.time() - t0}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} ({payload['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
