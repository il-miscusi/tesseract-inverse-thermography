# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Train the hybrid viscosity closure and save weights next to this file.

Run:  python train_closure.py [--fluid oil|water]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _model import FLUIDS, ViscosityClosure, reference_viscosity

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fluid", choices=sorted(FLUIDS), default="oil")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--noise", type=float, default=0.02, help="relative noise on measurements")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    spec = FLUIDS[args.fluid]
    ref = reference_viscosity(args.fluid)
    T = torch.linspace(*spec["t_range"], args.n)
    mu_true = ref(T)
    mu_obs = mu_true * (1.0 + args.noise * torch.randn_like(mu_true))

    model = ViscosityClosure(fluid=args.fluid)
    opt = torch.optim.Adam(model.net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # fit in log space: viscosity spans more than an order of magnitude
    target = torch.log(mu_obs)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = torch.mean((torch.log(model(T)) - target) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 500 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:5d}  log-mse {loss.item():.6e}")

    with torch.no_grad():
        pred = model(T)
        rel_hybrid = (torch.abs(pred - mu_true) / mu_true).mean().item()
        rel_arr = (torch.abs(model.arrhenius(T) - mu_true) / mu_true).mean().item()
        mu300, mu340, mu360 = (float(model(torch.tensor([t]))[0]) for t in (300.0, 340.0, 360.0))

    torch.save(model.state_dict(), HERE / f"closure_weights_{args.fluid}.pt")
    meta = {
        "fluid": args.fluid,
        "mean_rel_err_hybrid": rel_hybrid,
        "mean_rel_err_arrhenius_backbone": rel_arr,
        "improvement_factor": rel_arr / max(rel_hybrid, 1e-12),
        "train_range_K": list(spec["t_range"]),
        "arrhenius_A": float(model.arr_a),
        "arrhenius_B": float(model.arr_b),
        "mu_300K": mu300,
        "mu_340K": mu340,
        "mu_360K": mu360,
        "thinning_300_to_340": mu300 / mu340,
        "thinning_300_to_360": mu300 / mu360,
        "noise": args.noise,
        "epochs": args.epochs,
    }
    (HERE / f"closure_metrics_{args.fluid}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
