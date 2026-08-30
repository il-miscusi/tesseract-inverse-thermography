# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Steady advection-diffusion of temperature, in JAX.

Differentiated by JAX -- but *not* naively: an iterative linear solver cannot be
traced through its operator closure, so the derivative comes from the implicit
function theorem applied to the discrete residual (see ``_physics.py``).  That
makes this component's gradient exact to solver tolerance and independent of the
iteration count.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

jax.config.update("jax_enable_x64", True)

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

from _physics import jitted_vjp, solve_temperature


class InputSchema(BaseModel):
    gamma: Differentiable[Array[(None, None), Float64]] = Field(
        description="Design field in [0,1]: 0 = coolant, 1 = solid metal."
    )
    ux: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred x velocity (m/s) from the flow Tesseract."
    )
    uy: Differentiable[Array[(None, None), Float64]] = Field(
        description="Cell-centred y velocity (m/s) from the flow Tesseract."
    )
    q_source: Differentiable[Array[(None, None), Float64]] = Field(
        description="Volumetric heat source (W/m^3) -- the chip footprint. "
        "Differentiable here (an extension over the Track 02 version of this "
        "component): the inverse-thermography problem recovers this field, so "
        "the VJP must reach it."
    )
    dx: float = Field(default=1.0, description="Cell size in x (m).")
    dy: float = Field(default=1.0, description="Cell size in y (m).")
    k_fluid: float = Field(default=0.6, description="Coolant conductivity (W/m/K).")
    k_solid: float = Field(default=200.0, description="Metal conductivity (W/m/K).")
    rhoc: float = Field(default=4.18e6, description="Coolant rho*c_p (J/m^3/K).")
    t_in: float = Field(default=300.0, description="Inlet temperature (K).")
    inlet_conduction: float = Field(
        default=1.0, description="Scales the inlet half-cell conductive link."
    )
    inlet_conducts_solid: float = Field(
        default=0.0,
        description="0.0 (default, physical) conducts through the coolant at the "
        "inlet plane; 1.0 conducts through the local material k(gamma), which turns "
        "the inlet into an infinite metal-backed heat sink the optimiser will "
        "exploit. Kept so the artefact is reproducible -- see _physics.py.",
    )
    k_penal: float = Field(
        default=1.0,
        description="SIMP exponent on the conductivity interpolation "
        "k = k_fluid + (k_solid - k_fluid) * gamma**k_penal. 1.0 is linear (and "
        "lets intermediate density conduct like metal while flowing like coolant); "
        "the design problem uses 3.0.",
    )
    tol: float = Field(default=1e-12, description="Linear solver tolerance.")
    maxiter: int = Field(default=4000, description="Linear solver iteration cap.")


class OutputSchema(BaseModel):
    T: Differentiable[Array[(None, None), Float64]] = Field(
        description="Temperature field (K)."
    )
    # 0-d Array, not float: every output needs a shape under abstract evaluation
    resid: Array[(), Float64] = Field(description="Relative residual of the discrete system.")


def _prm(inputs: InputSchema) -> tuple:
    """The parameter tuple, in ONE place.

    It was built inline at each call site, so adding two boundary-condition
    parameters updated `solve_temperature` and left the VJP path unpacking a
    6-tuple into 8 names (it is 9 now, with ``k_penal``).  The forward call succeeded and only the gradient
    exploded, which is the worst way for a mismatch like this to show up.
    """
    return (
        float(inputs.dx),
        float(inputs.dy),
        float(inputs.k_fluid),
        float(inputs.k_solid),
        float(inputs.rhoc),
        float(inputs.t_in),
        float(inputs.inlet_conduction),
        float(inputs.inlet_conducts_solid),
        float(inputs.k_penal),
    )


def _kw(inputs: InputSchema) -> dict:
    return dict(
        dx=inputs.dx,
        dy=inputs.dy,
        k_fluid=inputs.k_fluid,
        k_solid=inputs.k_solid,
        rhoc=inputs.rhoc,
        t_in=inputs.t_in,
        inlet_conduction=inputs.inlet_conduction,
        inlet_conducts_solid=inputs.inlet_conducts_solid,
        k_penal=inputs.k_penal,
        tol=inputs.tol,
        maxiter=inputs.maxiter,
    )


def _arrays(inputs: InputSchema):
    return (
        jnp.asarray(np.asarray(inputs.gamma, dtype=np.float64)),
        jnp.asarray(np.asarray(inputs.ux, dtype=np.float64)),
        jnp.asarray(np.asarray(inputs.uy, dtype=np.float64)),
        jnp.asarray(np.asarray(inputs.q_source, dtype=np.float64)),
    )


def apply(inputs: InputSchema) -> OutputSchema:
    gamma, ux, uy, q = _arrays(inputs)
    T, resid = solve_temperature(gamma, ux, uy, q, **_kw(inputs))
    T = np.asarray(T)
    resid = float(resid)
    if not np.all(np.isfinite(T)) or not np.isfinite(resid):
        # Fail loudly here rather than shipping NaN across the wire, where it is
        # serialised to JSON `null` and surfaces as a baffling TypeError in the
        # caller three layers away.
        raise RuntimeError(
            f"heat-transport produced a non-finite solution "
            f"({int(np.sum(~np.isfinite(T)))}/{T.size} cells): the linear solve "
            f"did not converge. u range [{float(ux.min()):.3e}, {float(ux.max()):.3e}]"
        )
    return OutputSchema(T=T, resid=np.asarray(resid, dtype=np.float64))


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    if "T" not in vjp_outputs:
        return {}
    gamma, ux, uy, q = _arrays(inputs)
    order = tuple(n for n in ("gamma", "ux", "uy", "q_source") if n in vjp_inputs)
    if not order:
        return {}
    prm = _prm(inputs)
    cot = jnp.asarray(np.asarray(cotangent_vector["T"], dtype=np.float64))
    fn = jitted_vjp(order, prm, float(inputs.tol), int(inputs.maxiter))
    grads = fn(gamma, ux, uy, q, cot)
    return {n: np.asarray(g) for n, g in zip(order, grads)}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    if "T" not in jvp_outputs:
        return {}
    gamma, ux, uy, q = _arrays(inputs)
    kw = _kw(inputs)
    order = [n for n in ("gamma", "ux", "uy", "q_source") if n in jvp_inputs]
    if not order:
        return {}

    def f(*args):
        vals = dict(zip(order, args))
        T, _ = solve_temperature(
            vals.get("gamma", gamma), vals.get("ux", ux), vals.get("uy", uy),
            vals.get("q_source", q), **kw
        )
        return T

    primals = tuple({"gamma": gamma, "ux": ux, "uy": uy, "q_source": q}[n] for n in order)
    tangents = tuple(
        jnp.asarray(np.asarray(tangent_vector[n], dtype=np.float64)) for n in order
    )
    _, out = jax.jvp(f, primals, tangents)
    return {"T": np.asarray(out)}


def abstract_eval(abstract_inputs):
    return {
        "T": ShapeDType(shape=abstract_inputs.gamma.shape, dtype="float64"),
        "resid": ShapeDType(shape=(), dtype="float64"),
    }
