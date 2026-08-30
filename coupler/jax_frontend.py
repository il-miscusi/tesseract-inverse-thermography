# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""The coupled equilibrium as ONE JAX function: ``jax.grad(objective)(gamma)``.

``coupler/coupled.py`` talks to the three containers through the bare Tesseract
protocol (``apply`` / ``vector_jacobian_product``) and knows nothing about any
autodiff framework.  That is the load-bearing abstraction and it stays.

This module is the idiomatic front end on top of it, built with
`tesseract-jax <https://github.com/pasteurlabs/tesseract-jax>`_:

* each container becomes a traceable JAX primitive via ``apply_tesseract``;
* the composite map ``G(gamma, T) = H(gamma, F(gamma, N(T)))`` is then an
  ordinary JAX function whose ``jax.vjp`` dispatches to the three containers'
  VJP endpoints (Fortran hand-adjoint, JAX implicit-diff, Torch autograd);
* the fixed point ``T* = G(gamma, T*)`` is wrapped in a ``jax.custom_vjp`` whose
  backward pass solves the implicit-function-theorem system
  ``(I - dG/dT)^T lam = T_bar`` matrix-free with GMRES, using only those VJPs.

So a user writes::

    T, flux = equilibrium(gamma)
    J = objective(T, flux)
    dJ_dgamma = jax.grad(...)(gamma)

and JAX never learns that one third of ``G`` is compiled Fortran.

The forward fixed-point iteration has data-dependent control flow (it stops when
it converges), so ``equilibrium`` is meant to be used under ``jax.grad`` /
``jax.value_and_grad`` eagerly, not under ``jax.jit``.  Every number it produces
is cross-checked against the protocol-level coupler in ``tests/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)


@dataclass
class JaxCoupled:
    """Holds the three served Tesseracts and the plate parameters."""

    darcy: object
    heat: object
    closure: object
    plate: object
    tol: float = 1e-10
    maxiter: int = 200
    gmres_tol: float = 1e-8
    calls: dict | None = None

    # ---------------------------------------------------------- components
    def N(self, T):
        return apply_tesseract(self.closure, {"T": T, **self.plate.closure_params()})["mu"]

    def F(self, gamma, mu):
        out = apply_tesseract(self.darcy, {"gamma": gamma, "mu": mu, **self.plate.darcy_params()})
        return out["ux"], out["uy"], out["flux"]

    def H(self, gamma, ux, uy):
        prm = dict(self.plate.heat_params())
        prm["q_source"] = jnp.asarray(prm["q_source"])
        return apply_tesseract(self.heat, {"gamma": gamma, "ux": ux, "uy": uy, **prm})["T"]

    def G(self, gamma, T):
        """One pass around the loop: the map whose fixed point is the equilibrium."""
        mu = self.N(T)
        ux, uy, _ = self.F(gamma, mu)
        return self.H(gamma, ux, uy)

    def flux_of(self, gamma, T):
        return self.F(gamma, self.N(T))[2]

    # ---------------------------------------------------------- fixed point
    def _solve(self, gamma, T0=None):
        """Forward solve through the SAME Anderson/Newton-Krylov routine as the
        protocol coupler, with G evaluated eagerly through tesseract-jax."""
        from .coupled import fixed_point

        T_start = np.full(gamma.shape, self.plate.t_in) if T0 is None else np.asarray(T0)
        with jax.disable_jit():
            T, residuals, converged, _ = fixed_point(
                lambda T: np.asarray(self.G(jnp.asarray(gamma), jnp.asarray(T))),
                T_start, tol=self.tol, maxiter=self.maxiter,
                t_floor=self.plate.t_in - 50.0, t_ceiling=self.plate.t_in + 1500.0,
            )
        if not converged:
            raise RuntimeError(f"equilibrium did not converge: last residual {residuals[-1]:.2e}")
        self.last_iterations = len(residuals)
        return jnp.asarray(T)

    def equilibrium(self):
        """Return ``equilibrium(gamma) -> (T*, flux*)`` as a jax.custom_vjp function."""
        self_ = self

        @jax.custom_vjp
        def equilibrium(gamma):
            T = self_._solve(gamma)
            return T, self_.flux_of(gamma, T)

        def fwd(gamma):
            T = self_._solve(gamma)
            flux = self_.flux_of(gamma, T)
            return (T, flux), (gamma, T)

        def bwd(res, cotangents):
            gamma, T = res
            T_bar, flux_bar = cotangents
            n = T.size

            # VJPs of G at the equilibrium: these ARE the three containers' VJP
            # endpoints, chained by JAX through tesseract-jax.
            _, vjp_G = jax.vjp(self_.G, gamma, T)
            _, vjp_flux = jax.vjp(self_.flux_of, gamma, T)

            # explicit flux path, T-dependent through the closure only
            g_flux, T_flux = vjp_flux(flux_bar)
            rhs = T_bar + T_flux

            # (I - dG/dT)^T lam = rhs, matrix-free.  SciPy's GMRES, not JAX's:
            # jax.scipy's solver is a custom_linear_solve that wants to *trace and
            # transpose* the operator, and the operator is a container round-trip.
            # The backward pass runs on concrete values, so an eager solve is exact.
            from scipy.sparse.linalg import LinearOperator, gmres

            def matvec(w_flat):
                w = jnp.asarray(w_flat.reshape(T.shape))
                return np.array(w - vjp_G(w)[1], dtype=float).ravel()   # writable copy

            lam_flat, info = gmres(LinearOperator((n, n), matvec=matvec, dtype=float),
                                   np.asarray(rhs).ravel(), rtol=self_.gmres_tol, atol=0.0,
                                   restart=30, maxiter=200)
            if info != 0:
                raise RuntimeError(f"adjoint GMRES did not converge (info={info})")
            lam = jnp.asarray(lam_flat.reshape(T.shape))
            g_gamma, _ = vjp_G(lam)
            return (g_gamma + g_flux,)

        equilibrium.defvjp(fwd, bwd)
        return equilibrium


def open_jax_coupled(plate, images: dict | None = None):
    """Context manager: serve the three Tesseracts and hand back a JaxCoupled."""
    import contextlib

    from tesseract_core import Tesseract

    from .session import IMAGES

    imgs = {**IMAGES, **(images or {})}

    @contextlib.contextmanager
    def _cm():
        with contextlib.ExitStack() as stack:
            d = stack.enter_context(Tesseract.from_image(imgs["darcy"]))
            h = stack.enter_context(Tesseract.from_image(imgs["heat"]))
            c = stack.enter_context(Tesseract.from_image(imgs["closure"]))
            yield JaxCoupled(d, h, c, plate)

    return _cm()
