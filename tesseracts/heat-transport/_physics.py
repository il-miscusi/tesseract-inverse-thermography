# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Steady advection-diffusion of temperature on a cell-centred FV grid, in JAX.

    rho*c (u . grad T) = div( k(gamma) grad T ) + Q

Written as a flux balance per cell:

    sum_faces [ advective flux out + diffusive flux out ]  =  Q * dx * dy

Diffusion uses harmonic-mean face conductivities (correct for fluxes in series);
advection uses first-order upwinding on face velocities obtained by averaging the
neighbouring cell velocities.

The velocity field arrives from a *different solver in a different container* and
is therefore not discretely divergence-free on this grid -- exactly the situation
in industrial co-simulation.  Upwinding keeps that stable.

Boundaries: coolant enters on the left at T_in (advective inflow), pure outflow
on the right, adiabatic impermeable walls on top and bottom.

The inlet face also conducts, and WHAT it conducts through turned out to
matter more than whether it conducts at all.  Originally the half-cell link used
the local cell conductivity k(gamma).  Where the design put metal against the
inlet, that made the boundary a ~200 W/m/K path into an infinite isothermal
reservoir -- and the optimiser duly found it: it grew a metal bridge from the hot
spot to the inlet wall, let the coolant flow collapse to ~1e-8 of baseline, and
still reported the best temperature in the sweep.  The heat was leaving through
the boundary condition, not in the coolant.

``inlet_conducts_solid`` selects between the two:
  * 0.0 (default) -- the link uses k_fluid, i.e. the coolant at the inlet plane
    is at T_in and conducts weakly.  Physical, and a metal bridge buys nothing.
  * 1.0 -- the original k(gamma) link, kept so the artefact is reproducible.
``inlet_conduction`` scales the whole term; setting it to 0 removes the sink
entirely, which makes a zero-flow design singular (no way out for the heat) --
correct, but numerically hostile, so it is not the default.

Everything here is pure JAX, so `jax.vjp` differentiates it -- a completely
different mechanism from the hand-written Fortran adjoint next door.
"""

from __future__ import annotations

import functools
from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["conductivity", "solve_temperature"]


def conductivity(gamma, k_fluid: float, k_solid: float, k_penal: float = 1.0):
    """SIMP-penalised interpolation between coolant and metal conductivity.

        k = k_fluid + (k_solid - k_fluid) * gamma**k_penal

    ``k_penal = 1`` is the linear rule this component shipped with, and it was a
    mistake: the flow solver penalises the permeability of intermediate density
    convexly (RAMP, q = 8) while this gave a half-density cell *half* of
    aluminium's conductivity.  Grey material therefore conducted almost like
    metal and flowed almost like coolant, and the optimiser -- correctly, given
    the model -- never committed to a 0/1 design: the 64x32 "optimum" was 80%
    intermediate cells with gamma_max = 0.905, and thresholded to solid/void it ran
    +606 K against the fins' +53 K.  With ``k_penal = 3`` grey loses its
    conductivity as fast as it loses its permeability, which is the standard
    SIMP discipline and what makes a projected 0/1 design the true optimum."""
    return k_fluid + (k_solid - k_fluid) * gamma ** k_penal


def _harmonic(a, b):
    return 2.0 * a * b / jnp.maximum(a + b, 1e-300)


def _operator(T, kx, ky, kbl, uxf, uyf, ux_in, ux_out, dx, dy, rhoc):
    """T-dependent part of the flux balance: returns net outflow per cell."""
    out = jnp.zeros_like(T)

    # ---- diffusion ---------------------------------------------------------
    fx = kx * (T[:-1, :] - T[1:, :])          # flux i -> i+1
    out = out.at[:-1, :].add(fx)
    out = out.at[1:, :].add(-fx)

    fy = ky * (T[:, :-1] - T[:, 1:])
    out = out.at[:, :-1].add(fy)
    out = out.at[:, 1:].add(-fy)

    # inlet half-cell diffusive link (T-dependent half)
    out = out.at[0, :].add(kbl * T[0, :])

    # ---- advection (first-order upwind, advective form) --------------------
    # The velocity comes from a different solver in a different container, so it
    # is NOT discretely divergence-free on this grid.  A pure conservative flux
    # sum would then contain a spurious  T * div(u)  source that can push the
    # temperature outside the physical range (we measured T dipping below the
    # inlet value before this was fixed).  So we assemble the conservative flux
    # sum and subtract  rhoc * div(u) * T , which recovers the advective form
    # u . grad T exactly and is bounded whatever div(u) happens to be.
    Tx_up = jnp.where(uxf > 0, T[:-1, :], T[1:, :])
    ax = rhoc * uxf * dy * Tx_up
    out = out.at[:-1, :].add(ax)
    out = out.at[1:, :].add(-ax)

    Ty_up = jnp.where(uyf > 0, T[:, :-1], T[:, 1:])
    ay = rhoc * uyf * dx * Ty_up
    out = out.at[:, :-1].add(ay)
    out = out.at[:, 1:].add(-ay)

    out = out.at[0, :].add(-rhoc * jnp.minimum(ux_in, 0.0) * dy * T[0, :])
    out = out.at[-1, :].add(rhoc * jnp.maximum(ux_out, 0.0) * dy * T[-1, :])

    out = out - rhoc * _net_outflow(uxf, uyf, ux_in, ux_out, dx, dy, T.shape) * T

    return out


def _net_outflow(uxf, uyf, ux_in, ux_out, dx, dy, shape):
    """Discrete  integral of div(u)  over each cell, from the same face fluxes."""
    m = jnp.zeros(shape)
    fx = uxf * dy
    m = m.at[:-1, :].add(fx)
    m = m.at[1:, :].add(-fx)
    fy = uyf * dx
    m = m.at[:, :-1].add(fy)
    m = m.at[:, 1:].add(-fy)
    m = m.at[0, :].add(-ux_in * dy)
    m = m.at[-1, :].add(ux_out * dy)
    return m


def _diagonal(kx, ky, kbl, uxf, uyf, ux_in, ux_out, dx, dy, rhoc, shape):
    """Diagonal of the operator, for Jacobi preconditioning."""
    d = jnp.zeros(shape)
    d = d.at[:-1, :].add(kx)
    d = d.at[1:, :].add(kx)
    d = d.at[:, :-1].add(ky)
    d = d.at[:, 1:].add(ky)
    d = d.at[0, :].add(kbl)

    d = d.at[:-1, :].add(rhoc * dy * jnp.where(uxf > 0, uxf, 0.0))
    d = d.at[1:, :].add(rhoc * dy * jnp.where(uxf > 0, 0.0, -uxf))
    d = d.at[:, :-1].add(rhoc * dx * jnp.where(uyf > 0, uyf, 0.0))
    d = d.at[:, 1:].add(rhoc * dx * jnp.where(uyf > 0, 0.0, -uyf))
    d = d.at[0, :].add(-rhoc * dy * jnp.minimum(ux_in, 0.0))
    d = d.at[-1, :].add(rhoc * dy * jnp.maximum(ux_out, 0.0))
    d = d - rhoc * _net_outflow(uxf, uyf, ux_in, ux_out, dx, dy, shape)
    return jnp.where(jnp.abs(d) < 1e-30, 1.0, d)


def _residual(T, gamma, ux, uy, q_source, prm):
    """R(T, theta) = A(theta) T - b(theta).  Zero at the solution."""
    (dx, dy, k_fluid, k_solid, rhoc, t_in, inlet_conduction, inlet_conducts_solid, k_penal) = prm
    k = conductivity(gamma, k_fluid, k_solid, k_penal)

    kx = _harmonic(k[:-1, :], k[1:, :]) * (dy / dx)
    ky = _harmonic(k[:, :-1], k[:, 1:]) * (dx / dy)
    kbl = inlet_conduction * 2.0 * (dy / dx) * (
        (1.0 - inlet_conducts_solid) * k_fluid + inlet_conducts_solid * k[0, :]
    )

    uxf = 0.5 * (ux[:-1, :] + ux[1:, :])
    uyf = 0.5 * (uy[:, :-1] + uy[:, 1:])
    ux_in, ux_out = ux[0, :], ux[-1, :]

    lhs = _operator(T, kx, ky, kbl, uxf, uyf, ux_in, ux_out, dx, dy, rhoc)

    rhs = q_source * dx * dy
    rhs = rhs.at[0, :].add(kbl * t_in)
    rhs = rhs.at[0, :].add(rhoc * jnp.maximum(ux_in, 0.0) * dy * t_in)
    return lhs - rhs


def _diag_for(gamma, ux, uy, prm):
    (dx, dy, k_fluid, k_solid, rhoc, t_in, inlet_conduction, inlet_conducts_solid, k_penal) = prm
    k = conductivity(gamma, k_fluid, k_solid, k_penal)
    kx = _harmonic(k[:-1, :], k[1:, :]) * (dy / dx)
    ky = _harmonic(k[:, :-1], k[:, 1:]) * (dx / dy)
    kbl = inlet_conduction * 2.0 * (dy / dx) * (
        (1.0 - inlet_conducts_solid) * k_fluid + inlet_conducts_solid * k[0, :]
    )
    uxf = 0.5 * (ux[:-1, :] + ux[1:, :])
    uyf = 0.5 * (uy[:, :-1] + uy[:, 1:])
    return _diagonal(kx, ky, kbl, uxf, uyf, ux[0, :], ux[-1, :], dx, dy, rhoc, gamma.shape)


def _linsolve(matvec, b, diag, tol, maxiter):
    """Restarted GMRES.

    We used BiCGSTAB here first and it *broke down* -- returning all-NaN -- on
    exactly the configurations this problem is about: a blocked fin design where
    the hot spot reaches ~440 K, the learned viscosity drops 7x, and the flow
    field becomes strongly non-uniform.  The linear system itself was healthy
    (condition number 5.4e3, all eigenvalues with positive real part, exact dense
    solve fine); BiCGSTAB simply hit its classic rho/omega breakdown when asked
    for 1e-12.  GMRES has no such failure mode, so it is worth the extra memory.
    """
    restart = int(min(b.size, 100))
    return jax.scipy.sparse.linalg.gmres(
        matvec,
        b,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        M=lambda r: r / diag,
    )[0]


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _solve(gamma, ux, uy, q_source, prm, tol, maxiter):
    """Forward solve.  Differentiated by the implicit function theorem below.

    JAX cannot differentiate an iterative solver through its operator closure,
    so we never ask it to: the forward pass is opaque and the derivative comes
    from the implicit function theorem applied to the residual.
    """
    zero = jnp.zeros_like(q_source)
    b = -_residual(zero, gamma, ux, uy, q_source, prm)

    def matvec(T):
        return _residual(T, gamma, ux, uy, q_source, prm) + b

    diag = _diag_for(gamma, ux, uy, prm)
    return _linsolve(matvec, b, diag, tol, maxiter)


def _solve_fwd(gamma, ux, uy, q_source, prm, tol, maxiter):
    T = _solve(gamma, ux, uy, q_source, prm, tol, maxiter)
    return T, (T, gamma, ux, uy, q_source)


def _solve_bwd(prm, tol, maxiter, res, T_bar):
    T, gamma, ux, uy, q_source = res

    zero = jnp.zeros_like(q_source)
    b = -_residual(zero, gamma, ux, uy, q_source, prm)

    def matvec(x):
        return _residual(x, gamma, ux, uy, q_source, prm) + b

    # adjoint system  A^T w = T_bar
    transpose = jax.linear_transpose(matvec, T)
    diag = _diag_for(gamma, ux, uy, prm)
    w = _linsolve(lambda x: transpose(x)[0], T_bar, diag, tol, maxiter)

    # theta_bar = - w^T dR/dtheta   (T held fixed)
    _, vjp_theta = jax.vjp(
        lambda g, a, c, q: _residual(T, g, a, c, q, prm), gamma, ux, uy, q_source
    )
    g_bar, ux_bar, uy_bar, q_bar = vjp_theta(w)
    return (-g_bar, -ux_bar, -uy_bar, -q_bar)


_solve.defvjp(_solve_fwd, _solve_bwd)


@functools.lru_cache(maxsize=16)
def jitted_solve(prm, tol, maxiter):
    """Compile once per parameter set and reuse.

    Without this the container recompiles the GMRES loop on essentially every
    request: measured 0.715 s unjitted vs 0.076 s jitted per solve, and 1.23 s
    vs 0.178 s per gradient.  Since the coupled fixed point calls this component
    ~10 times per solve and the design loop runs hundreds of solves, the jit
    cache is the difference between a 50-minute run and a 10-minute one.
    """
    return jax.jit(functools.partial(_solve, prm=prm, tol=tol, maxiter=maxiter))


@functools.lru_cache(maxsize=16)
def jitted_vjp(order, prm, tol, maxiter):
    """Compiled reverse-mode for a given subset of inputs."""
    solve = jitted_solve(prm, tol, maxiter)

    def f(gamma, ux, uy, q_source, cotangent):
        base = {"gamma": gamma, "ux": ux, "uy": uy, "q_source": q_source}

        def g(*args):
            vals = dict(zip(order, args))
            return solve(
                vals.get("gamma", gamma),
                vals.get("ux", ux),
                vals.get("uy", uy),
                vals.get("q_source", q_source),
            )

        primals = tuple(base[n] for n in order)
        _, vjp_fn = jax.vjp(g, *primals)
        return vjp_fn(cotangent)

    return jax.jit(f)


def solve_temperature(
    gamma,
    ux,
    uy,
    q_source,
    *,
    dx: float,
    dy: float,
    k_fluid: float,
    k_solid: float,
    rhoc: float,
    t_in: float,
    inlet_conduction: float = 1.0,
    inlet_conducts_solid: float = 0.0,
    k_penal: float = 1.0,
    tol: float = 1e-12,
    maxiter: int = 4000,
):
    """Solve the steady advection-diffusion system for T. Returns (T, rel_residual)."""
    prm = (float(dx), float(dy), float(k_fluid), float(k_solid), float(rhoc),
           float(t_in), float(inlet_conduction), float(inlet_conducts_solid), float(k_penal))
    T = jitted_solve(prm, float(tol), int(maxiter))(gamma, ux, uy, q_source)
    r = _residual(T, gamma, ux, uy, q_source, prm)
    kb = inlet_conduction * 2.0 * (dy / dx) * (
        (1.0 - inlet_conducts_solid) * k_fluid
        + inlet_conducts_solid * conductivity(gamma, k_fluid, k_solid, k_penal)[0, :]
    )
    scale = (
        jnp.linalg.norm(q_source * dx * dy)
        + jnp.abs(t_in) * jnp.linalg.norm(kb)
        + jnp.abs(t_in) * rhoc * jnp.abs(dy) * jnp.linalg.norm(jnp.maximum(ux[0, :], 0.0))
    )
    return T, jnp.linalg.norm(r) / jnp.maximum(scale, 1e-300)
