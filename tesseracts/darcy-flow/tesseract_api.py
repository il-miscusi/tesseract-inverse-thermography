# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Darcy / Brinkman flow through a designable porous medium.

The numerics live in Fortran (``fortran/darcy.f90``) and the derivative is a
HAND-DERIVED DISCRETE ADJOINT, also written in Fortran.  No automatic
differentiation tool is involved anywhere in this component -- which is exactly
why it cannot be traced by the JAX or PyTorch components it is composed with.

Forward:  div( kappa(gamma)/mu grad p ) = 0,   u = -(kappa/mu) grad p
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, Int64, ShapeDType

from darcy_bridge import run_darcy


class InputSchema(BaseModel):
    gamma: Differentiable[Array[(None, None), Float64]] = Field(
        description="Design field in [0,1]: 0 = open coolant channel, 1 = solid metal fin."
    )
    mu: Differentiable[Array[(None, None), Float64]] = Field(
        description="Dynamic viscosity field (Pa s), same shape as gamma. "
        "Supplied by the closure Tesseract -- this is the coupling inlet."
    )
    dx: float = Field(default=1.0, description="Cell size in x (m).")
    dy: float = Field(default=1.0, description="Cell size in y (m).")
    pin: float = Field(default=1.0, description="Inlet pressure (Pa); outlet is 0.")
    kmin: float = Field(
        default=1e-8, description="Permeability of solid material (Brinkman floor)."
    )
    kmax: float = Field(default=1.0, description="Permeability of open channel.")
    qramp: float = Field(default=8.0, description="RAMP interpolation sharpness.")
    tol: float = Field(default=1e-12, description="CG relative tolerance.")
    maxit: int = Field(default=20000, description="CG iteration cap.")


class OutputSchema(BaseModel):
    p: Differentiable[Array[(None, None), Float64]] = Field(description="Pressure field (Pa).")
    ux: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred x velocity (m/s)."
    )
    uy: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred y velocity (m/s)."
    )
    flux: Differentiable[Float64] = Field(
        description="Total volumetric inlet flux; pumping power is pin * flux."
    )
    # Diagnostics are 0-d Arrays rather than plain int/float so that every output
    # has a shape under abstract evaluation -- tesseract-jax needs that to trace.
    cg_iters: Array[(), Int64] = Field(description="CG iterations used by the forward solve.")
    cg_resid: Array[(), Float64] = Field(description="Final relative residual of the forward solve.")


def _solver_kwargs(inputs: InputSchema) -> dict:
    return dict(
        dx=inputs.dx,
        dy=inputs.dy,
        pin=inputs.pin,
        kmin=inputs.kmin,
        kmax=inputs.kmax,
        qramp=inputs.qramp,
        tol=inputs.tol,
        maxit=inputs.maxit,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    r = run_darcy(inputs.gamma, inputs.mu, **_solver_kwargs(inputs))
    if not (np.all(np.isfinite(r["p"])) and np.isfinite(r["resid"])):
        # A NaN here serialises to JSON `null` and surfaces as a TypeError three
        # layers away in the caller.  The boundary is the place to refuse it.
        raise RuntimeError(
            f"Darcy forward solve produced non-finite output (resid={r['resid']!r}, "
            f"iters={r['iters']}); mu range {float(np.min(inputs.mu)):.3e}.."
            f"{float(np.max(inputs.mu)):.3e}"
        )
    return OutputSchema(
        p=r["p"],
        ux=r["ux"],
        uy=r["uy"],
        flux=r["flux"],
        cg_iters=np.asarray(r["iters"], dtype=np.int64),
        cg_resid=np.asarray(r["resid"], dtype=np.float64),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Hand-derived discrete adjoint (see fortran/main.f90).

    Costs exactly one extra linear solve, independent of the number of design
    variables -- which is the whole reason to write an adjoint by hand.
    """
    shape = np.asarray(inputs.gamma).shape
    zeros = np.zeros(shape)
    cot = {
        "p": np.asarray(cotangent_vector.get("p", zeros), dtype=np.float64),
        "ux": np.asarray(cotangent_vector.get("ux", zeros), dtype=np.float64),
        "uy": np.asarray(cotangent_vector.get("uy", zeros), dtype=np.float64),
        "flux": float(np.asarray(cotangent_vector.get("flux", 0.0))),
    }
    for key in ("p", "ux", "uy"):
        if key not in vjp_outputs:
            cot[key] = zeros
    if "flux" not in vjp_outputs:
        cot["flux"] = 0.0

    r = run_darcy(inputs.gamma, inputs.mu, cotangents=cot, **_solver_kwargs(inputs))

    # The adjoint is a second linear solve, and an unconverged one returns a
    # plausible-looking but wrong gradient.  Refuse to hand that across the
    # boundary: a wrong number that looks fine is worse than an error.
    adj_resid = r.get("adj_resid", float("nan"))
    if not np.isfinite(adj_resid) or adj_resid > max(1e3 * inputs.tol, 1e-8):
        raise RuntimeError(
            f"Darcy adjoint solve did not converge: residual {adj_resid:.3e} "
            f"after {r.get('adj_iters')} CG iterations (tol {inputs.tol:.1e}). "
            "The returned gradient would be silently wrong."
        )

    out = {}
    if "gamma" in vjp_inputs:
        out["gamma"] = r["gamma_bar"]
    if "mu" in vjp_inputs:
        out["mu"] = r["mu_bar"]
    return out


def abstract_eval(abstract_inputs):
    shape = abstract_inputs.gamma.shape
    field = ShapeDType(shape=shape, dtype="float64")
    return {
        "p": field,
        "ux": field,
        "uy": field,
        "flux": ShapeDType(shape=(), dtype="float64"),
        "cg_iters": ShapeDType(shape=(), dtype="int64"),
        "cg_resid": ShapeDType(shape=(), dtype="float64"),
    }
