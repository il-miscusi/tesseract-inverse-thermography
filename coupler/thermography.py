# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Inverse thermography: pull a pixel loss back through the camera AND the
coupled multiphysics equilibrium, to the volumetric heat source q(x, y).

The chain
---------
    counts = C(T*, eps, sigma, g, o)                 thermal-camera  [JAX]
    T*     solves  T = G(gamma, T; q)                the coupled loop:
           G = H(gamma, F(gamma, N(T)); q)           Torch -> Fortran -> JAX

For a loss J on counts, the camera VJP gives dJ/dT*; the implicit function
theorem turns that into a gradient with respect to q:

    (I - dG/dT)^T lambda = dJ/dT*          (matrix-free GMRES, one three-
                                            container VJP chain per matvec)
    dJ/dq = (dH/dq)^T lambda               (one heat-transport VJP)

``one_way=True`` skips the adjoint solve (lambda = dJ/dT*): that is exactly the
gradient of the frozen-feedback model, kept so the value of the coupled term is
measured rather than asserted.

Nothing in here knows what is inside any component.  The whole inverse problem
runs on ``apply`` and ``vector_jacobian_product``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .coupled import CoupledSystem, FixedPointResult


# --------------------------------------------------------------- gradient wrt q
def gradient_wrt_q(
    system: CoupledSystem,
    gamma: np.ndarray,
    state: FixedPointResult,
    dJ_dT: np.ndarray,
    *,
    gmres_tol: float = 1e-8,
    gmres_maxiter: int = 200,
    gmres_restart: int = 30,
    one_way: bool = False,
) -> tuple[np.ndarray, int, bool]:
    """dJ/dq through the coupled equilibrium.  Returns (grad, matvecs, converged)."""
    shape = state.T.shape
    n = state.T.size
    n_matvec = 0

    if one_way:
        lam = np.asarray(dJ_dT, float)
        info = 0
    else:

        def matvec(w_flat):
            nonlocal n_matvec
            n_matvec += 1
            w = w_flat.reshape(shape)
            return (w - system._dG_dT_transpose(gamma, state, w)).ravel()

        op = LinearOperator((n, n), matvec=matvec, dtype=float)
        lam_flat, info = gmres(
            op,
            np.asarray(dJ_dT, float).ravel(),
            rtol=gmres_tol,
            atol=0.0,
            restart=gmres_restart,
            maxiter=gmres_maxiter,
        )
        lam = lam_flat.reshape(shape)

    h = system.heat.vjp(gamma, state.ux, state.uy, lam, wrt=("q_source",))
    return np.asarray(h["q_source"]), n_matvec, info == 0


# ------------------------------------------------------------------ forward map
@dataclass
class ThermographyForward:
    """q -> equilibrium T* -> rendered counts, with warm-started fixed points."""

    system: CoupledSystem
    camera: object                      # CameraComponent
    gamma: np.ndarray
    eps: np.ndarray
    psf_sigma: float
    gain: float
    offset: float
    t_init: float
    fp_tol: float = 1e-9
    fp_maxiter: int = 120
    _T_warm: np.ndarray | None = field(default=None, repr=False)

    def set_q(self, q: np.ndarray) -> None:
        self.system.heat.params["q_source"] = np.asarray(q, float)

    def solve(self, q: np.ndarray) -> FixedPointResult:
        self.set_q(q)
        st = self.system.solve(
            self.gamma,
            T0=self._T_warm,
            tol=self.fp_tol,
            maxiter=self.fp_maxiter,
            t_init=self.t_init,
        )
        if not st.converged:
            raise RuntimeError(
                f"coupled fixed point did not converge (last "
                f"{st.residuals[-1] if st.residuals else float('nan'):.2e})"
            )
        self._T_warm = st.T
        return st

    def render(self, st: FixedPointResult) -> np.ndarray:
        return self.camera.apply(st.T, self.eps, self.psf_sigma, self.gain, self.offset)

    def loss_and_grad_q(
        self, q: np.ndarray, y_meas: np.ndarray, *, one_way: bool = False
    ) -> tuple[float, np.ndarray, dict]:
        """0.5 * mean((counts - y)^2) and its gradient with respect to q."""
        st = self.solve(q)
        counts = self.render(st)
        r = counts - y_meas
        loss = 0.5 * float(np.mean(r**2))
        cot = r / r.size
        dJ_dT = self.camera.vjp(
            st.T, self.eps, self.psf_sigma, self.gain, self.offset, cot, wrt=("T",)
        )["T"]
        grad_q, matvecs, ok = gradient_wrt_q(
            self.system, self.gamma, st, dJ_dT, one_way=one_way
        )
        return loss, grad_q, {
            "state": st,
            "counts": counts,
            "adjoint_matvecs": matvecs,
            "adjoint_converged": ok,
        }


# -------------------------------------------------------------------- utilities
def laplacian(f: np.ndarray) -> np.ndarray:
    """5-point Laplacian with zero-Neumann edges (edge replication)."""
    p = np.pad(f, 1, mode="edge")
    return p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4.0 * f


def smoothness_penalty(q_n: np.ndarray) -> tuple[float, np.ndarray]:
    """0.5 * ||Lap q||^2 (per cell) and its gradient: Lap^T Lap q = Lap(Lap q)."""
    l = laplacian(q_n)
    val = 0.5 * float(np.mean(l**2))
    grad = laplacian(l) / q_n.size
    return val, grad


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def softplus_grad(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def total_variation_penalty(q_n: np.ndarray, eps: float = 1e-6) -> tuple[float, np.ndarray]:
    """Smoothed isotropic TV: mean(sqrt(dx^2 + dy^2 + eps^2)) and its gradient.

    Edge-preserving prior for hotspot recovery: L2-on-Laplacian penalises the
    sharp rim of a real hotspot quadratically, TV only linearly, so blobs stay
    blobs.  Forward differences with replicated last row/column (zero-gradient
    edge); the gradient is the exact transpose of those differences.
    """
    dx = np.diff(q_n, axis=0, append=q_n[-1:, :])
    dy = np.diff(q_n, axis=1, append=q_n[:, -1:])
    mag = np.sqrt(dx**2 + dy**2 + eps**2)
    val = float(np.mean(mag))
    wx = dx / mag
    wy = dy / mag
    grad = np.zeros_like(q_n)
    grad[:-1, :] -= wx[:-1, :]
    grad[1:, :] += wx[:-1, :]
    grad[:, :-1] -= wy[:, :-1]
    grad[:, 1:] += wy[:, :-1]
    return val, grad / q_n.size


@dataclass
class Adam:
    """Plain Adam on a flat numpy array (small, dependency-free)."""

    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m: np.ndarray | None = None
    _v: np.ndarray | None = None
    _t: int = 0

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self._m is None:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * g
        self._v = self.beta2 * self._v + (1 - self.beta2) * g**2
        mh = self._m / (1 - self.beta1**self._t)
        vh = self._v / (1 - self.beta2**self._t)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------- ground-truth field
def two_blob_source(shape: tuple[int, int], q_scale: float,
                    blobs=((0.42, 0.06, 1.5, 0.05), (0.62, 0.06, 0.9, 0.04))) -> np.ndarray:
    """Two Gaussian hot spots on the chip plane.

    Each blob is (x_frac, y_frac, amplitude_in_q_scale, width_frac).  Defaults
    put both inside the Track 02 chip band but at positions the forward model
    has never been told about.
    """
    nx, ny = shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    q = np.zeros(shape)
    for (cx, cy, amp, w) in blobs:
        q += amp * q_scale * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w**2))
    return q


def source_metrics(q_rec: np.ndarray, q_true: np.ndarray, shape=None) -> dict:
    """Recovery metrics: relative L2 error, amplitude ratio, centroid shift (cells)."""
    rel_l2 = float(np.linalg.norm(q_rec - q_true) / np.linalg.norm(q_true))
    amp = float(q_rec.max() / q_true.max())

    def centroid(q):
        nx, ny = q.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        m = max(float(q.sum()), 1e-300)
        return float((q * X).sum() / m), float((q * Y).sum() / m)

    cr, ct = centroid(q_rec), centroid(q_true)
    shift = float(np.hypot(cr[0] - ct[0], cr[1] - ct[1]))
    return {
        "rel_l2": rel_l2,
        "amplitude_ratio": amp,
        "centroid_shift_cells": shift,
        "total_power_ratio": float(q_rec.sum() / q_true.sum()),
    }
