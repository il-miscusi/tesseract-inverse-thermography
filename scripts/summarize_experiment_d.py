#!/usr/bin/env python3
"""Aggregate the frozen Experiment D bank without changing its protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = tuple(range(101, 113))
ARMS = ("full", "mismatch")
METRICS = {
    "holdout_rms_counts": lambda r: r["holdout_residual"]["rms_counts"],
    "holdout_mean_abs_counts": lambda r: abs(r["holdout_residual"]["mean_counts"]),
    "holdout_max_abs_lag1": lambda r: max(
        abs(r["holdout_residual"]["lag1_horizontal"]),
        abs(r["holdout_residual"]["lag1_vertical"]),
    ),
    "source_relative_l2": lambda r: r["rel_l2"],
    "centroid_error_mm": lambda r: r["centroid_error_mm"],
    "peak_error_mm": lambda r: r["peak_error_mm"],
    "wasserstein_mm": lambda r: r["wasserstein_mm"],
    "total_power_relative_error": lambda r: r["total_power_rel_error"],
}
DIAGNOSTIC_METRICS = (
    "source_relative_l2",
    "centroid_error_mm",
    "peak_error_mm",
    "wasserstein_mm",
    "total_power_relative_error",
)


def distribution(values: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "median": float(np.median(values)),
        "iqr": [float(x) for x in np.quantile(values, [0.25, 0.75])],
        "p90": float(np.quantile(values, 0.90)),
        "worst": float(np.max(values)),
    }


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(10_000, len(values)), replace=True)
    return [float(x) for x in np.quantile(np.median(draws, axis=1), [0.025, 0.975])]


def load_bank(bank_dir: Path) -> list[dict]:
    expected = {bank_dir / f"bank_seed_{seed}.json" for seed in SEEDS}
    present = set(bank_dir.glob("bank_seed_*.json"))
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise SystemExit(f"bank must contain exactly seeds 101-112; missing={missing}, extra={extra}")
    rows = [json.loads(path.read_text()) for path in sorted(expected)]
    if [row["seed"] for row in rows] != list(SEEDS):
        raise SystemExit("bank JSON seed fields do not match filenames")
    if any(row.get("stage") != "bank" for row in rows):
        raise SystemExit("refusing to summarize non-bank artifacts")
    return rows


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(20260831)
    values = {
        arm: {
            metric: np.asarray([extract(row["results"][arm]) for row in rows], dtype=float)
            for metric, extract in METRICS.items()
        }
        for arm in ARMS
    }
    arm_summary = {}
    for arm in ARMS:
        results = [row["results"][arm] for row in rows]
        arm_summary[arm] = {
            "absolute_plausible_fit_count": sum(r["absolute_plausible_fit"] for r in results),
            "operationally_useful_count": sum(r["operationally_useful_diagnosis"] for r in results),
            "optimizer_converged_count": sum(r["optimizer"]["converged"] for r in results),
            "metrics": {metric: distribution(series) for metric, series in values[arm].items()},
        }

    paired = {}
    for metric in DIAGNOSTIC_METRICS:
        # Positive means the mismatch produced more diagnostic error.
        delta = values["mismatch"][metric] - values["full"][metric]
        paired[metric] = {
            "sign_convention": "mismatch_minus_full; positive favors calibrated full arm",
            "values_by_seed": {str(seed): float(x) for seed, x in zip(SEEDS, delta)},
            "median": float(np.median(delta)),
            "mean": float(np.mean(delta)),
            "median_bootstrap_95_ci": bootstrap_median_ci(delta, rng),
            "full_lower_error_count": int(np.count_nonzero(delta > 0)),
            "ties_count": int(np.count_nonzero(delta == 0)),
        }

    plausible_mismatch = np.asarray(
        [row["results"]["mismatch"]["absolute_plausible_fit"] for row in rows], dtype=bool
    )
    power_delta = values["mismatch"]["total_power_relative_error"] - values["full"]["total_power_relative_error"]
    full_plausible = arm_summary["full"]["absolute_plausible_fit_count"]
    harmful = sum(row["materially_harmful_plausible_mismatch"] for row in rows)
    accepted = full_plausible >= 10 and harmful >= 6
    return {
        "protocol": "writeup/EXPERIMENT_D_PROTOCOL.md",
        "bank_seeds": list(SEEDS),
        "scene_count": len(rows),
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20260831,
        "arms": arm_summary,
        "materially_harmful_plausible_mismatch_count": harmful,
        "paired_diagnostic_error": paired,
        "plausible_mismatch_power_error_increase": {
            "scene_count": int(plausible_mismatch.sum()),
            "median_percentage_points": float(100 * np.median(power_delta[plausible_mismatch])),
            "iqr_percentage_points": [
                float(x) for x in 100 * np.quantile(power_delta[plausible_mismatch], [0.25, 0.75])
            ],
        },
        "preregistered_verdict": {
            "full_absolute_fit_requirement": ">=10/12",
            "full_absolute_fit_pass": full_plausible >= 10,
            "harmful_plausible_mismatch_requirement": ">=6/12",
            "harmful_plausible_mismatch_pass": harmful >= 6,
            "calibration_risk_claim_accepted": accepted,
            "fallback_if_rejected": "verified composition claim",
        },
    }


def make_figure(rows: list[dict], summary: dict, output: Path) -> None:
    seeds = np.asarray(SEEDS)
    full_power = 100 * np.asarray([r["results"]["full"]["total_power_rel_error"] for r in rows])
    mismatch_power = 100 * np.asarray([r["results"]["mismatch"]["total_power_rel_error"] for r in rows])
    full_rms = np.asarray([r["results"]["full"]["holdout_rms_noise_sigmas"] for r in rows])
    mismatch_rms = np.asarray([r["results"]["mismatch"]["holdout_rms_noise_sigmas"] for r in rows])
    mismatch_ok = np.asarray([r["results"]["mismatch"]["absolute_plausible_fit"] for r in rows])

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figstyle

    figstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2),
                             gridspec_kw={"wspace": 0.24, "top": 0.745,
                                          "bottom": 0.115, "left": 0.06,
                                          "right": 0.985})
    for ax in axes:
        ax.grid(axis="y", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0]
    for x, a, b in zip(seeds, full_power, mismatch_power):
        ax.plot([x - 0.12, x + 0.12], [a, b], color=figstyle.FAINT,
                alpha=0.7, lw=1.1)
    ax.scatter(seeds - 0.12, full_power, color=figstyle.ACCENT,
               label="calibrated", s=36, zorder=3)
    ax.scatter(seeds + 0.12, mismatch_power, color=figstyle.ACCENT2,
               label="4% emissivity error", s=36, zorder=3)
    ax.axhline(5, color=figstyle.TRUTH, ls="--", lw=1, alpha=0.9,
               label="5 pp harm threshold")
    ax.set_title("Diagnostic power error on unseen scenes", loc="left",
                 fontsize=10.5, pad=9)
    ax.set_xlabel("frozen bank seed")
    ax.set_ylabel("absolute total-power error (%)")
    ax.set_xticks(seeds)
    ax.legend(frameon=False, fontsize=8)
    figstyle.panel_letter(ax, "a", x=-0.065, y=1.02)

    ax = axes[1]
    ax.scatter(seeds - 0.12, full_rms, color=figstyle.ACCENT,
               label="calibrated", s=36)
    colors = np.where(mismatch_ok, figstyle.ACCENT2, figstyle.TRUTH)
    ax.scatter(seeds + 0.12, mismatch_rms, c=colors,
               label="4% emissivity error", s=36)
    ax.axhline(2, color=figstyle.TRUTH, ls="--", lw=1, alpha=0.9,
               label="RMS gate (2σ)")
    ax.set_title("Held-out pixel residual", loc="left", fontsize=10.5, pad=9)
    ax.set_xlabel("frozen bank seed")
    ax.set_ylabel("RMS / sensor-noise σ")
    ax.set_xticks(seeds)
    ax.legend(frameon=False, fontsize=8)
    figstyle.panel_letter(ax, "b", x=-0.09, y=1.02)

    verdict = summary["preregistered_verdict"]
    useful = summary["arms"]["full"]["operationally_useful_count"]
    figstyle.headline(
        fig,
        f"The calibrated pipeline generalizes: {useful}/12 useful diagnoses "
        "on unseen scenes",
        "Experiment D — independently frozen 12-scene bank:  "
        f"full plausible {summary['arms']['full']['absolute_plausible_fit_count']}/12  ·  "
        f"harmful + plausible mismatch {summary['materially_harmful_plausible_mismatch_count']}/12  ·  "
        f"preregistered claim {'ACCEPTED' if verdict['calibration_risk_claim_accepted'] else 'NOT ACCEPTED'}",
        x=0.028, y=0.965, sub_dy=0.062, size=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, default=Path("figures/experiment_d/bank"))
    parser.add_argument("--summary", type=Path, default=Path("figures/experiment_d/experiment_d_summary.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/experiment_d_generalization.png"))
    args = parser.parse_args()
    rows = load_bank(args.bank_dir)
    summary = summarize(rows)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(rows, summary, args.figure)
    print(json.dumps(summary["preregistered_verdict"], indent=2))


if __name__ == "__main__":
    main()
