# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""The coupled equilibrium as a PyTorch autograd op, for calibration.

`coupler/jax_frontend.py` exposes the equilibrium to ``jax.grad`` for design.
This is the mirror image for the *other* direction round the loop: the closure's
weights ``theta`` are a ``torch.Tensor`` that requires grad, the equilibrium is a
``torch.autograd.Function`` whose backward runs the same implicit-function-
theorem adjoint as the protocol coupler, and the final link -- the closure's
own VJP with respect to ``theta`` -- is taken by **tesseract-torch**:
``tesseract_torch.apply_tesseract`` makes the viscosity-closure container a
differentiable torch op, so ``mu.backward(mu_bar)`` lands the cotangent in
``theta.grad`` and any ``torch.optim`` optimiser can drive the calibration.

Nothing here knows that the flow solver is Fortran or the heat solver is JAX:
they are reached through the protocol coupler inside ``backward``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, gmres

try:
    from tesseract_torch import apply_tesseract as _torch_apply
except ImportError:  # pragma: no cover - optional dependency
    _torch_apply = None


class EquilibriumSensors(torch.autograd.Function):
    """theta -> temperature at the sensor locations, through the fixed point."""

    @staticmethod
    def forward(ctx, theta, system, closure, gamma, sensors, plate, warm):
        closure.theta = theta.detach().cpu().double().numpy()
        st = system.solve(gamma, T0=warm.get("T"), tol=1e-10, maxiter=300, t_init=plate.t_in)
        warm["T"] = st.T
        warm["state"] = st
        ctx.system, ctx.closure, ctx.gamma, ctx.sensors, ctx.state = system, closure, gamma, sensors, st
        return torch.from_numpy(np.ascontiguousarray(st.T[sensors])).to(theta.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        system, closure, gamma, sensors, st = ctx.system, ctx.closure, ctx.gamma, ctx.sensors, ctx.state
        shape = gamma.shape
        dJ_dT = np.zeros(shape)
        np.add.at(dJ_dT, sensors, grad_out.detach().cpu().double().numpy())

        # (I - dG/dT)^T lam = dJ/dT, matrix-free through the three containers
        def matvec(w_flat):
            w = w_flat.reshape(shape)
            return (w - system._dG_dT_transpose(gamma, st, w)).ravel()

        lam, _ = gmres(LinearOperator((gamma.size, gamma.size), matvec=matvec, dtype=float),
                       dJ_dT.ravel(), rtol=1e-8, atol=0.0, restart=30, maxiter=200)
        lam = lam.reshape(shape)
        h = system.heat.vjp(gamma, st.ux, st.uy, lam, wrt=("ux", "uy"))
        d = system.darcy.vjp(gamma, st.mu, {"ux": h["ux"], "uy": h["uy"]}, wrt=("mu",))
        mu_bar = torch.from_numpy(np.ascontiguousarray(d["mu"]))

        # the last link, taken by tesseract-torch: autograd through the closure container
        theta = torch.from_numpy(closure.theta.copy()).requires_grad_(True)
        if _torch_apply is not None:
            # autograd is switched off inside a Function.backward; re-enable it for
            # the nested graph through the closure container
            with torch.enable_grad():
                T_in = torch.from_numpy(np.ascontiguousarray(st.T))
                mu = _torch_apply(closure.tesseract, {"T": T_in, "theta": theta, **closure.params})["mu"]
                g = torch.autograd.grad(mu, theta, grad_outputs=mu_bar.to(mu.dtype))[0]
        else:  # protocol fallback, identical numbers
            g = torch.from_numpy(closure.vjp_theta(st.T, d["mu"]))
        return g.to(grad_out.dtype), None, None, None, None, None, None


def sensor_temperatures(theta, system, closure, gamma, sensors, plate, warm):
    return EquilibriumSensors.apply(theta, system, closure, gamma, sensors, plate, warm)
