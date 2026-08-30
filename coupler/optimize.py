# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Projected-gradient design optimisation under a volume constraint."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def project_volume(x: np.ndarray, target: float, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Project onto {lo <= x <= hi, mean(x) == target} by bisection on a shift.

    This is the standard way to hold a material budget exactly, rather than
    nudging it with a penalty term whose weight then has to be tuned.

    It is strict about non-finite input on purpose.  A single inf in ``x`` sends
    the bisection to a bracket end and the function returns every finite cell
    pinned to ``hi`` -- a design that satisfies nothing and looks like a result.
    That is exactly what happened once: a sweep point came back as a solid-metal
    plate at volume fraction 1.000 against a budget of 0.35, scored the best
    temperature of the whole sweep, and was duly written into a Pareto front.
    A constraint routine that fails silently is worse than one that crashes.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise FloatingPointError(
            f"project_volume received {n_bad} non-finite value(s) of {x.size}. "
            "Refusing to project: the result would silently satisfy no constraint."
        )
    if target <= lo:
        return np.full_like(x, lo)
    if target >= hi:
        return np.full_like(x, hi)

    a, b = lo - float(x.max()) - 1.0, hi - float(x.min()) + 1.0
    shift = 0.5 * (a + b)
    for _ in range(200):
        shift = 0.5 * (a + b)
        m = float(np.clip(x + shift, lo, hi).mean())
        if abs(m - target) < 1e-14:
            break
        if m < target:
            a = shift
        else:
            b = shift

    out = np.clip(x + shift, lo, hi)
    achieved = float(out.mean())
    if abs(achieved - target) > 1e-6:
        raise RuntimeError(
            f"project_volume failed to hit the volume target: wanted {target:.6f}, "
            f"got {achieved:.6f}. Bisection did not converge."
        )
    return out


def project_volume_filtered(x: np.ndarray, filt, target: float,
                            lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Shift-and-clip ``x`` so that the *filtered* field ``filt.forward(x)`` has
    mean ``target``.

    The physics sees gamma = W rho, not rho, and the row-normalised filter is
    not exactly volume-preserving at the walls (a design projected to 35.00% in
    rho came out at 34.66% in gamma).  Because every entry of W is non-negative,
    mean(W clip(x + s)) is monotone in the shift s, so bisection on s is exact.
    The result is still a rho in [lo, hi]; the bound is what keeps the
    permeability interpolation positive."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("project_volume_filtered received non-finite input")
    a, b = lo - float(x.max()) - 1.0, hi - float(x.min()) + 1.0
    for _ in range(200):
        shift = 0.5 * (a + b)
        m = float(filt.forward(np.clip(x + shift, lo, hi)).mean())
        if abs(m - target) < 1e-13:
            break
        if m < target:
            a = shift
        else:
            b = shift
    out = np.clip(x + shift, lo, hi)
    achieved = float(filt.forward(out).mean())
    if abs(achieved - target) > 1e-6:
        raise RuntimeError(
            f"project_volume_filtered missed the target: wanted {target:.6f}, got {achieved:.6f}"
        )
    return out


@dataclass
class OptimizeHistory:
    J: list[float] = field(default_factory=list)
    t_peak: list[float] = field(default_factory=list)
    t_peak_true: list[float] = field(default_factory=list)
    flux: list[float] = field(default_factory=list)
    step: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    picard_iters: list[int] = field(default_factory=list)
    gmres_matvecs: list[int] = field(default_factory=list)
    designs: list[np.ndarray] = field(default_factory=list)


def optimize(
    system,
    plate,
    filt,
    objective,
    rho0: np.ndarray,
    *,
    iterations: int = 40,
    step: float = 0.15,
    snapshot_every: int = 1,
    on_iteration=None,
    gradient_kwargs: dict | None = None,
    verbose: bool = True,
    betas: tuple[float, ...] | None = None,
    qramps: tuple[float, ...] | None = None,
    fp_tol: float = 1e-10,
) -> tuple[np.ndarray, OptimizeHistory]:
    """Projected gradient with a simple backtracking line search.

    Each accepted iteration costs one coupled solve plus one coupled adjoint;
    a rejected trial costs one more coupled solve.

    ``filt`` may be a plain ``DensityFilter`` or a ``DesignChain`` (filter +
    Heaviside projection).  With a chain and ``betas`` given, the projection
    sharpness is raised on a schedule -- ``iterations`` is split evenly across
    the betas -- so the design is driven to solid/void only after the material
    has found where it wants to be.  Every beta change re-evaluates the current
    design under the new projection and re-projects the volume, because the
    objective is not comparable across betas.
    """
    hist = OptimizeHistory()
    schedule = list(betas) if (betas and hasattr(filt, "beta")) else None
    # optional continuation on the Brinkman penalty too: a near-threshold cell
    # at RAMP q = 8 still leaks ~10% of an open cell's flow, and a design that
    # has learned to use leaky walls falls apart when thresholded.  Raising q in
    # step with beta removes that loophole before the design is frozen.
    qsched = list(qramps) if (qramps and schedule and len(qramps) == len(schedule)) else None
    q_orig = system.darcy.params.get("qramp") if qsched else None
    if qsched:
        system.darcy.params["qramp"] = qsched[0]
    if schedule:
        filt.beta = schedule[0]
    per_beta = max(1, iterations // len(schedule)) if schedule else iterations
    rho = project_volume_filtered(np.array(rho0, float), filt, plate.volume_fraction)
    T_warm = None

    def evaluate(r, T0):
        gamma = filt.forward(r)
        st = system.solve(gamma, T0=T0, tol=fp_tol, maxiter=200, t_init=plate.t_in)
        j, extra = objective.value_and_cotangents(st)
        return gamma, st, j, extra

    gamma, state, J, extra = evaluate(rho, T_warm)
    T_warm = state.T

    for it in range(iterations):
        if schedule and it > 0 and it % per_beta == 0 and (it // per_beta) < len(schedule):
            filt.beta = schedule[it // per_beta]
            if qsched:
                system.darcy.params["qramp"] = qsched[it // per_beta]
            rho = project_volume_filtered(rho, filt, plate.volume_fraction)
            gamma, state, J, extra = evaluate(rho, T_warm)
            T_warm = state.T
            if verbose:
                print(f"  [{it:3d}] projection beta -> {filt.beta:g}: J={J:.5f}  "
                      f"Tpeak(true)={float(state.T.max() - plate.t_in):.3f}K")
        ad = system.gradient(
            gamma,
            state,
            dJ_dT=extra["dJ_dT"],
            dJ_dflux=extra["dJ_dflux"],
            **(gradient_kwargs or {}),
        )
        g = filt.adjoint(ad.grad, rho)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError(
                f"non-finite coupled gradient at iteration {it}: "
                f"{int(np.sum(~np.isfinite(g)))} of {g.size} entries. "
                "Refusing to step -- see project_volume for why this must be loud."
            )
        gnorm = float(np.linalg.norm(g))
        # normalise so `step` is a design-space distance, not a physical scale
        d = g / max(gnorm, 1e-300) * np.sqrt(g.size)

        accepted = False
        trial_step = step
        # Backtrack generously.  With the oil the response to a design change is
        # sharply nonlinear (the loop is nearly expansive), and a line search that
        # gave up after six halvings stopped a 64x32 run at iteration 5 with the
        # objective still falling 2 K per step.  A trial whose coupled solve did
        # not converge is rejected outright rather than compared.
        for _ in range(10):
            rho_try = project_volume_filtered(rho - trial_step * d, filt, plate.volume_fraction)
            gamma_try, state_try, J_try, extra_try = evaluate(rho_try, T_warm)
            if np.isfinite(J_try) and state_try.converged and J_try < J:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            if verbose:
                print(f"  [{it:3d}] no improving step found down to {trial_step:.2e}; stopping")
            break

        rho, gamma, state, J, extra = rho_try, gamma_try, state_try, J_try, extra_try
        T_warm = state.T
        step = min(trial_step * 1.3, 0.3)

        hist.J.append(J)
        hist.t_peak.append(extra["t_peak"])
        hist.t_peak_true.append(float(state.T.max() - plate.t_in))
        hist.flux.append(state.flux)
        hist.step.append(trial_step)
        hist.grad_norm.append(gnorm)
        hist.picard_iters.append(state.iterations)
        hist.gmres_matvecs.append(ad.gmres_iters)
        if it % snapshot_every == 0:
            hist.designs.append(gamma.copy())

        if verbose:
            print(
                f"  [{it:3d}] J={J:.5f}  Tpeak(smooth)={extra['t_peak']:.3f}K  "
                f"Tpeak(true)={hist.t_peak_true[-1]:.3f}K  flux={state.flux:.4e}  "
                f"step={trial_step:.3f}  picard={state.iterations}  gmres={ad.gmres_iters}"
            )

        # Checkpoint after every accepted step.  A long design run that only
        # writes at the end throws away hours of work on any single failure.
        if on_iteration is not None:
            on_iteration(it, rho, gamma, state, J, extra, hist)

    if schedule and getattr(filt, "beta", None) != schedule[-1]:
        # too few iterations to reach the last beta: finish the continuation anyway,
        # so the returned design is always at the sharpest projection
        filt.beta = schedule[-1]
        rho = project_volume_filtered(rho, filt, plate.volume_fraction)
    if qsched:
        system.darcy.params["qramp"] = q_orig       # scoring uses the plate's own physics
    achieved = float(filt.forward(rho).mean())
    if abs(achieved - plate.volume_fraction) > 1e-3:
        raise RuntimeError(
            f"optimize() is about to return a design at volume fraction "
            f"{achieved:.4f} against a budget of {plate.volume_fraction:.4f}. "
            "A design that violates its own constraint must never be scored."
        )
    return rho, hist
