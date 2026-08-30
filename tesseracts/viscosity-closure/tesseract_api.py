# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Learned coolant-viscosity closure  mu = N_theta(T).

A hybrid mechanistic + neural model, differentiated by **PyTorch's tape-based
autograd** -- a third differentiation mechanism, distinct from the Fortran
hand-adjoint and from JAX's tracing.  Its derivative dmu/dT is what closes the
multiphysics loop: temperature changes viscosity, viscosity changes the flow,
the flow changes temperature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

from _model import ViscosityClosure

_HERE = Path(__file__).parent

_models: dict[str, ViscosityClosure] = {}


def _get_model(fluid: str = "oil") -> ViscosityClosure:
    """One trained closure per fluid, loaded once. Weights ship inside the image."""
    m = _models.get(fluid)
    if m is None:
        m = ViscosityClosure(fluid=fluid)
        m.load_state_dict(torch.load(_HERE / f"closure_weights_{fluid}.pt", map_location="cpu", weights_only=True))
        m.eval().double()
        for p in m.parameters():
            p.requires_grad_(False)
        _models[fluid] = m
    return m


class InputSchema(BaseModel):
    T: Differentiable[Array[(None, None), Float64]] = Field(
        description="Temperature field (K). Viscosity is evaluated pointwise."
    )
    theta: Differentiable[Array[(None,), Float64]] = Field(
        default_factory=lambda: np.zeros(0),
        description="Optional flat vector of the closure's trainable parameters "
        "(see `theta_layout`). Empty (default) evaluates the shipped weights. "
        "Supplying it makes the closure *calibratable*: the VJP w.r.t. theta lets a "
        "host fit the closure through whatever the closure is embedded in -- here, "
        "the entire coupled equilibrium.",
    )
    fluid: str = Field(
        default="oil",
        description="Which trained closure to evaluate: 'oil' (ISO VG 22 immersion "
        "oil, thins ~4x over 300-340 K) or 'water' (thins ~1.5x).",
    )
    coupling_scale: float = Field(
        default=1.0,
        description="Flattens the mu(T) curve about T_ref without moving it there: "
        "mu_s = mu(T_ref) * (mu(T)/mu(T_ref))**s. 1 = the real fluid, 0 = isoviscous "
        "(no thermal feedback). The knob for the coupling-strength sweep.",
    )


class OutputSchema(BaseModel):
    mu: Differentiable[Array[(None, None), Float64]] = Field(
        description="Dynamic viscosity field (Pa s)."
    )


def theta_layout(fluid: str = "oil") -> list[tuple[str, tuple[int, ...]]]:
    """(name, shape) of each trainable tensor, in flat-vector order."""
    return [(k, tuple(v.shape)) for k, v in _get_model(fluid).net.named_parameters()]


def _unflatten(theta: torch.Tensor, fluid: str) -> dict[str, torch.Tensor]:
    params, off = {}, 0
    for name, shape in theta_layout(fluid):
        n = int(np.prod(shape))
        params["net." + name] = theta[off : off + n].reshape(shape)
        off += n
    if off != theta.numel():
        raise ValueError(f"theta has {theta.numel()} entries, closure expects {off}")
    return params


def _evaluate(model, T, theta, coupling_scale):
    """mu = model(T) with the shipped weights, or with `theta` substituted."""
    if theta.numel() == 0:
        return model(T, coupling_scale)
    return torch.func.functional_call(model, _unflatten(theta, model.fluid), (T, coupling_scale))


def apply(inputs: InputSchema) -> OutputSchema:
    T = torch.as_tensor(np.asarray(inputs.T, dtype=np.float64))
    theta = torch.as_tensor(np.asarray(inputs.theta, dtype=np.float64))
    with torch.no_grad():
        mu = _evaluate(_get_model(inputs.fluid), T, theta, inputs.coupling_scale)
    return OutputSchema(mu=mu.numpy())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode through the torch graph, w.r.t. T and/or the parameters."""
    if "mu" not in vjp_outputs:
        return {}
    wanted = [k for k in ("T", "theta") if k in vjp_inputs]
    if not wanted:
        return {}
    T = torch.as_tensor(np.asarray(inputs.T, dtype=np.float64))
    theta = torch.as_tensor(np.asarray(inputs.theta, dtype=np.float64))
    if "theta" in wanted and theta.numel() == 0:
        raise ValueError("VJP w.r.t. theta requested but no theta was supplied")
    leaves = {"T": T, "theta": theta}
    for k in wanted:
        leaves[k].requires_grad_(True)
    mu = _evaluate(_get_model(inputs.fluid), leaves["T"], leaves["theta"], inputs.coupling_scale)
    cot = torch.as_tensor(np.asarray(cotangent_vector["mu"], dtype=np.float64))
    grads = torch.autograd.grad(mu, [leaves[k] for k in wanted], grad_outputs=cot,
                                allow_unused=True)
    out = {}
    for k, g in zip(wanted, grads):
        out[k] = (g if g is not None else torch.zeros_like(leaves[k])).detach().numpy()
    return out


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Forward-mode; the map is pointwise so this is cheap."""
    if "mu" not in jvp_outputs or not ({"T", "theta"} & set(jvp_inputs)):
        return {}
    T = torch.as_tensor(np.asarray(inputs.T, dtype=np.float64))
    theta = torch.as_tensor(np.asarray(inputs.theta, dtype=np.float64))
    model = _get_model(inputs.fluid)
    tan_T = torch.as_tensor(np.asarray(tangent_vector.get("T", np.zeros_like(inputs.T)), dtype=np.float64))
    tan_th = torch.as_tensor(np.asarray(tangent_vector.get("theta", np.zeros(theta.numel())), dtype=np.float64))
    _, out = torch.func.jvp(
        lambda x, th: _evaluate(model, x, th, inputs.coupling_scale), (T, theta), (tan_T, tan_th)
    )
    return {"mu": out.detach().numpy()}


def abstract_eval(abstract_inputs):
    return {"mu": ShapeDType(shape=abstract_inputs.T.shape, dtype="float64")}
