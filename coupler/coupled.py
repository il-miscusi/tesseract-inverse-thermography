# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Solve a coupled multiphysics equilibrium spanning several Tesseracts, and
differentiate through it.

The coupling
------------
    mu = N(T)              viscosity closure   [PyTorch autograd]
    u  = F(gamma, mu)      Darcy flow          [Fortran hand-written adjoint]
    T  = H(gamma, u)       heat transport      [JAX autodiff]

Eliminating mu and u leaves a fixed point in the temperature field alone:

    T = G(gamma, T) := H(gamma, F(gamma, N(T)))

No single AD framework can trace G: one third of it is compiled Fortran whose
derivative was derived with pen and paper.  So we never ask one to.

Differentiating the equilibrium
-------------------------------
At a converged T*, the implicit function theorem gives

    (I - dG/dT) dT*/dgamma = dG/dgamma

and for an objective J(gamma, T*, flux*),

    dJ/dgamma = dJ/dgamma|_expl + lambda^T dG/dgamma,
    (I - dG/dT)^T lambda = (dJ/dT)|_expl

Both the adjoint system and the final contraction need **only VJPs of G**, and a
VJP of G is a chain of the three components' VJP endpoints:

    w --> VJP_H wrt u --> VJP_F wrt mu --> VJP_N wrt T

The adjoint system is therefore solved matrix-free (GMRES), one container
round-trip per component per matvec.  Nothing is ever assembled, and no
component ever sees another's internals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

logger = logging.getLogger(__name__)


@dataclass
class FixedPointResult:
    T: np.ndarray
    mu: np.ndarray
    p: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    flux: float
    converged: bool
    iterations: int
    stagnated: bool = False
    residuals: list[float] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class AdjointResult:
    grad: np.ndarray
    gmres_iters: int
    converged: bool
    seconds: float
    component_calls: dict = field(default_factory=dict)



def fixed_point(
    G,
    T0: np.ndarray,
    *,
    tol: float = 1e-9,
    maxiter: int = 60,
    damping: float = 1.0,
    adaptive: bool = True,
    anderson_depth: int = 5,
    newton_fallback: bool = True,
    stall_patience: int = 6,
    stagnation_tol: float = 1e-6,
    t_floor: float = -np.inf,
    t_ceiling: float = np.inf,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float], bool, bool]:
    """Solve T = G(T) for a black-box map G, using only evaluations of G.

    Shared by the protocol-level coupler and the tesseract-jax front end, so the
    two routes cannot disagree about what "converged" means.

    Method, in order of escalation:

    1. **Anderson-accelerated Picard** (depth ``anderson_depth``; 0 = plain damped
       Picard with adaptive relaxation).  With an oil coolant the viscosity
       feedback is strong: the measured contraction factor of G is 0.83 at the
       design heat load and creeps past 0.9 on hot plates, so plain Picard needs
       ~120 round-trips through all three containers and stalls outright above
       ~0.93.  Anderson mixing extrapolates from the last m iterates using only
       the residuals and brings that back to ~15.  Extrapolated iterates that
       leave the physical band [t_floor, t_ceiling] are discarded.
    2. **Jacobian-free Newton-Krylov** when the loop has turned *expansive* (an
       eigenvalue of dG/dT at or beyond 1 -- a choked design can do that): Newton
       on F(T) = G(T) - T with GMRES, each matvec one finite-difference directional
       derivative of G.  Triggered when ten evaluations fail to cut the residual
       by 3x while it is still far from tolerance.

    Neither stage needs a derivative endpoint on any component, exactly like the
    adjoint: the whole pipeline runs on ``apply`` and ``vector_jacobian_product``.

    Stagnation -- the residual flooring at the component solvers' own precision
    (~1e-10 here, far worse than their reported 1e-12 for a permeability contrast
    of 1e5) -- is only diagnosed once the residual is already below
    ``stagnation_tol``; an early plateau is a convergence problem, not a floor.
    """
    T = np.array(T0, dtype=float)
    residuals: list[float] = []
    beta = float(damping)
    prev_r, best_r, stall = np.inf, np.inf, 0
    converged = stagnated = False
    hist_T: list[np.ndarray] = []
    hist_F: list[np.ndarray] = []
    evals = 0
    while evals < maxiter:
        T_new = np.asarray(G(T), dtype=float)
        evals += 1
        denom = max(float(np.linalg.norm(T_new)), 1e-300)
        r = float(np.linalg.norm(T_new - T)) / denom
        residuals.append(r)
        if verbose:
            print(f"    fixed-point {evals:3d}  rel_change {r:.3e}  beta {beta:.2f}")
        if not np.isfinite(r):
            raise FloatingPointError("fixed-point iterate became non-finite")
        if r < tol:
            T = T_new
            converged = True
            break
        if r < best_r:
            best_r = r

        # stagnation at the numerical floor (only meaningful once nearly converged)
        if r < stagnation_tol:
            if r < 0.5 * min(residuals[:-1], default=np.inf):
                stall = 0
            else:
                stall += 1
                if stall >= stall_patience:
                    stagnated = True
                    T = T_new
                    break

        # escalate to Newton-Krylov if Anderson is not getting anywhere
        if newton_fallback and evals >= 10 and r > 1e-4 and r > 0.3 * residuals[-10]:
            T, converged, used = _newton_krylov(G, T, tol, maxiter - evals, residuals, verbose)
            break

        if anderson_depth > 0:
            f = (T_new - T).ravel()
            hist_T.append(T.ravel().copy())
            hist_F.append(f)
            if len(hist_T) > anderson_depth + 1:
                hist_T.pop(0)
                hist_F.pop(0)
            m = len(hist_F) - 1
            T_next = None
            if m >= 1:
                dF = np.stack([hist_F[i + 1] - hist_F[i] for i in range(m)], axis=1)
                dT = np.stack([hist_T[i + 1] - hist_T[i] for i in range(m)], axis=1)
                gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
                cand = (T.ravel() + beta * f) - (dT + beta * dF) @ gam
                if np.all(np.isfinite(cand)) and cand.min() > t_floor and cand.max() < t_ceiling:
                    T_next = cand.reshape(T.shape)
                else:
                    hist_T, hist_F = [], []
            T = T_next if T_next is not None else (1.0 - beta) * T + beta * T_new
        else:
            if adaptive and evals > 2 and r > 0.7 * prev_r:
                beta = max(beta * 0.6, 0.15)
            T = (1.0 - beta) * T + beta * T_new
        prev_r = r

    if stagnated and residuals[-1] < stagnation_tol:
        converged = True
    return T, residuals, converged, stagnated


def _newton_krylov(G, T, tol, budget, residuals, verbose=False):
    """Jacobian-free Newton-Krylov on F(T) = G(T) - T with backtracking."""
    from scipy.sparse.linalg import LinearOperator, gmres

    shape, n = T.shape, T.size
    F = np.asarray(G(T)) - T
    nF = float(np.linalg.norm(F))
    used = 0
    while used < max(budget // 8, 3):
        scale = max(float(np.linalg.norm(T)), 1.0)

        def jv(v_flat):
            v = v_flat.reshape(shape)
            nv = max(float(np.linalg.norm(v)), 1e-300)
            h = 1e-6 * scale / nv
            Fp = np.asarray(G(T + h * v)) - (T + h * v)
            return np.array((Fp - F) / h).ravel()

        delta, _ = gmres(LinearOperator((n, n), matvec=jv, dtype=float), -F.ravel(),
                         rtol=1e-3, atol=0.0, restart=20, maxiter=40)
        delta = delta.reshape(shape)
        alpha, accepted = 1.0, False
        for _ in range(6):
            T_try = T + alpha * delta
            F_try = np.asarray(G(T_try)) - T_try
            nF_try = float(np.linalg.norm(F_try))
            if np.isfinite(nF_try) and nF_try < (1.0 - 1e-4 * alpha) * nF:
                T, F, nF, accepted = T_try, F_try, nF_try, True
                break
            alpha *= 0.5
        used += 1
        r = nF / max(float(np.linalg.norm(T)), 1e-300)
        residuals.append(r)
        if verbose:
            print(f"    newton-krylov {used:2d}  rel_residual {r:.3e}  alpha {alpha:.3f}")
        logger.info("newton-krylov step %d: rel residual %.2e (alpha %.3f)", used, r, alpha)
        if r < tol:
            return T, True, used
        if not accepted:
            break
    return T, residuals[-1] < 1e-6, used


class CoupledSystem:
    """The coupled equilibrium and its adjoint."""

    def __init__(self, darcy, heat, closure):
        self.darcy = darcy
        self.heat = heat
        self.closure = closure

    # ------------------------------------------------------------------ forward
    def step(self, gamma: np.ndarray, T: np.ndarray) -> dict:
        """One evaluation of G: T -> mu -> (u, flux) -> T_new."""
        mu = self.closure.apply(T)
        flow = self.darcy.apply(gamma, mu)
        heat = self.heat.apply(gamma, flow["ux"], flow["uy"])
        return {
            "T_new": heat["T"],
            "mu": mu,
            "p": flow["p"],
            "ux": flow["ux"],
            "uy": flow["uy"],
            "flux": flow["flux"],
        }

    def solve(
        self,
        gamma: np.ndarray,
        T0: np.ndarray | None = None,
        *,
        tol: float = 1e-9,
        maxiter: int = 60,
        damping: float = 1.0,
        adaptive: bool = True,
        anderson_depth: int = 5,
        newton_fallback: bool = True,
        stall_patience: int = 6,
        stagnation_tol: float = 1e-6,
        t_init: float = 300.0,
        verbose: bool = False,
    ) -> FixedPointResult:
        """Solve T = G(gamma, T).  See :func:`fixed_point` for the method."""
        t0 = time.time()
        T_start = np.full(gamma.shape, t_init) if T0 is None else np.array(T0, dtype=float)
        cache: dict = {}

        def G(T):
            st = self.step(gamma, T)
            cache["state"] = st
            return st["T_new"]

        T, residuals, converged, stagnated = fixed_point(
            G, T_start, tol=tol, maxiter=maxiter, damping=damping, adaptive=adaptive,
            anderson_depth=anderson_depth, newton_fallback=newton_fallback,
            stall_patience=stall_patience, stagnation_tol=stagnation_tol,
            t_floor=t_init - 50.0, t_ceiling=t_init + 1500.0, verbose=verbose,
        )
        if not converged:
            logger.warning(
                "coupled fixed point did not reach %.1e in %d evaluations (last %.2e, stagnated=%s)",
                tol, len(residuals), residuals[-1] if residuals else float("nan"), stagnated,
            )
        # one last evaluation so the reported fields are exactly consistent with T
        state = self.step(gamma, T)
        return FixedPointResult(
            T=T, mu=state["mu"], p=state["p"], ux=state["ux"], uy=state["uy"],
            flux=state["flux"], converged=converged, iterations=len(residuals),
            stagnated=stagnated, residuals=residuals, seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------ adjoint
    def _dG_dT_transpose(self, gamma, state: FixedPointResult, w: np.ndarray) -> np.ndarray:
        """(dG/dT)^T w, as a chain of three container VJPs."""
        h = self.heat.vjp(gamma, state.ux, state.uy, w, wrt=("ux", "uy"))
        d = self.darcy.vjp(
            gamma, state.mu, {"ux": h["ux"], "uy": h["uy"]}, wrt=("mu",)
        )
        return self.closure.vjp(state.T, d["mu"])

    def _dflux_dT_transpose(self, gamma, state: FixedPointResult, c: float) -> np.ndarray:
        """(dflux/dT)^T c -- flux depends on T only through the viscosity closure."""
        if c == 0.0:
            return np.zeros_like(state.T)
        d = self.darcy.vjp(gamma, state.mu, {"flux": c}, wrt=("mu",))
        return self.closure.vjp(state.T, d["mu"])

    def gradient(
        self,
        gamma: np.ndarray,
        state: FixedPointResult,
        *,
        dJ_dT: np.ndarray,
        dJ_dflux: float = 0.0,
        dJ_dgamma_direct: np.ndarray | None = None,
        gmres_tol: float = 1e-8,
        gmres_maxiter: int = 200,
        gmres_restart: int = 30,
        neglect_feedback: bool = False,
        verbose: bool = False,
    ) -> AdjointResult:
        """dJ/dgamma through the coupled equilibrium.

        ``neglect_feedback=True`` computes the ONE-WAY gradient instead: it keeps
        the true coupled equilibrium for the forward state but pretends, when
        differentiating, that viscosity does not depend on temperature.  That is
        exactly what you get by dropping the ``(I - dG/dT)^-1`` factor, i.e. by
        setting lambda = dJ/dT and chaining T <- u <- gamma once.

        It exists so the value of the coupled adjoint is *measured* rather than
        asserted.  If the one-way gradient pointed the same way, this whole
        construction would be decoration."""
        t0 = time.time()
        for c in (self.darcy.counter, self.heat.counter, self.closure.counter):
            c.reset()

        shape = gamma.shape
        n = gamma.size

        # ---- explicit cotangents, including the flux path ------------------
        g_expl = (
            np.zeros(shape) if dJ_dgamma_direct is None else np.array(dJ_dgamma_direct, float)
        )
        T_expl = np.array(dJ_dT, dtype=float)
        if dJ_dflux != 0.0:
            d = self.darcy.vjp(gamma, state.mu, {"flux": dJ_dflux}, wrt=("gamma", "mu"))
            g_expl = g_expl + d["gamma"]
            T_expl = T_expl + self.closure.vjp(state.T, d["mu"])

        # ---- adjoint system  (I - dG/dT)^T lambda = T_expl -----------------
        n_matvec = 0

        def matvec(w_flat):
            nonlocal n_matvec
            n_matvec += 1
            w = w_flat.reshape(shape)
            return (w - self._dG_dT_transpose(gamma, state, w)).ravel()

        if neglect_feedback:
            lam = T_expl
            info = 0
        else:
            op = LinearOperator((n, n), matvec=matvec, dtype=float)
            lam_flat, info = gmres(
                op,
                T_expl.ravel(),
                rtol=gmres_tol,
                atol=0.0,
                restart=gmres_restart,
                maxiter=gmres_maxiter,
            )
            lam = lam_flat.reshape(shape)
        if verbose:
            print(f"    gmres info={info}  matvecs={n_matvec}")

        # ---- contract:  dJ/dgamma = g_expl + (dG/dgamma)^T lambda ----------
        h = self.heat.vjp(gamma, state.ux, state.uy, lam, wrt=("gamma", "ux", "uy"))
        d = self.darcy.vjp(
            gamma, state.mu, {"ux": h["ux"], "uy": h["uy"]}, wrt=("gamma",)
        )
        grad = g_expl + h["gamma"] + d["gamma"]

        return AdjointResult(
            grad=grad,
            gmres_iters=n_matvec,
            converged=(info == 0),
            seconds=time.time() - t0,
            component_calls={
                "darcy": (self.darcy.counter.apply, self.darcy.counter.vjp),
                "heat": (self.heat.counter.apply, self.heat.counter.vjp),
                "closure": (self.closure.counter.apply, self.closure.counter.vjp),
            },
        )
