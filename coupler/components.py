# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Thin typed wrappers around the three Tesseracts.

Each wrapper exposes exactly two operations -- ``apply`` and a VJP -- because
that is the entire contract the coupler needs.  Nothing here knows or cares that
one component is Fortran with a hand-written adjoint, one is JAX, and one is
PyTorch: the derivative arrives as a *protocol*, not as a language feature.
That is the property the whole project rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CallCounter:
    """Bookkeeping so the writeup can state the real cost of a gradient.

    Two scopes, deliberately.  ``apply``/``vjp`` are a *window* that
    ``CoupledSystem.gradient`` zeroes so it can report the cost of one gradient
    (the headline "14 container calls" number).  ``total_apply``/``total_vjp``
    are lifetime counts that nothing resets.

    Keeping only the window was a real bug: a 25-iteration design run reported
    40 applies, roughly a tenth of the truth, because the number quoted at the
    end of the run was only what had happened since the last gradient call.
    A cost figure that is silently a window is worse than no cost figure.
    """

    apply: int = 0
    vjp: int = 0
    total_apply: int = 0
    total_vjp: int = 0

    def bump_apply(self) -> None:
        self.apply += 1
        self.total_apply += 1

    def bump_vjp(self) -> None:
        self.vjp += 1
        self.total_vjp += 1

    def reset(self) -> None:
        """Zero the per-gradient window only; lifetime totals survive."""
        self.apply = 0
        self.vjp = 0


@dataclass
class DarcyComponent:
    """Fortran Darcy solver: (gamma, mu) -> (p, ux, uy, flux)."""

    tesseract: object
    params: dict = field(default_factory=dict)
    counter: CallCounter = field(default_factory=CallCounter)

    def apply(self, gamma: np.ndarray, mu: np.ndarray) -> dict:
        self.counter.bump_apply()
        out = self.tesseract.apply(
            {"gamma": gamma, "mu": mu, **self.params}
        )
        return {
            "p": np.asarray(out["p"]),
            "ux": np.asarray(out["ux"]),
            "uy": np.asarray(out["uy"]),
            "flux": float(np.asarray(out["flux"])),
            "cg_iters": int(out.get("cg_iters", -1)),
            "cg_resid": float(out.get("cg_resid", np.nan)),
        }

    def vjp(
        self,
        gamma: np.ndarray,
        mu: np.ndarray,
        cotangents: dict,
        wrt: tuple[str, ...] = ("gamma", "mu"),
    ) -> dict:
        self.counter.bump_vjp()
        zeros = np.zeros_like(gamma)
        cot = {
            "p": np.asarray(cotangents.get("p", zeros)),
            "ux": np.asarray(cotangents.get("ux", zeros)),
            "uy": np.asarray(cotangents.get("uy", zeros)),
            "flux": float(cotangents.get("flux", 0.0)),
        }
        out = self.tesseract.vector_jacobian_product(
            inputs={"gamma": gamma, "mu": mu, **self.params},
            vjp_inputs=list(wrt),
            vjp_outputs=["p", "ux", "uy", "flux"],
            cotangent_vector=cot,
        )
        return {k: np.asarray(v) for k, v in out.items()}


@dataclass
class HeatComponent:
    """JAX advection-diffusion solver: (gamma, ux, uy) -> T."""

    tesseract: object
    params: dict = field(default_factory=dict)
    counter: CallCounter = field(default_factory=CallCounter)

    def apply(self, gamma: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> dict:
        self.counter.bump_apply()
        out = self.tesseract.apply(
            {"gamma": gamma, "ux": ux, "uy": uy, **self.params}
        )
        resid = out.get("resid")
        return {
            "T": np.asarray(out["T"]),
            # JSON has no NaN, so a non-finite residual arrives as None
            "resid": float(resid) if resid is not None else float("nan"),
        }

    def vjp(
        self,
        gamma: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        cotangent_T: np.ndarray,
        wrt: tuple[str, ...] = ("gamma", "ux", "uy"),
    ) -> dict:
        self.counter.bump_vjp()
        out = self.tesseract.vector_jacobian_product(
            inputs={"gamma": gamma, "ux": ux, "uy": uy, **self.params},
            vjp_inputs=list(wrt),
            vjp_outputs=["T"],
            cotangent_vector={"T": np.asarray(cotangent_T)},
        )
        return {k: np.asarray(v) for k, v in out.items()}


@dataclass
class ClosureComponent:
    """PyTorch viscosity closure: T -> mu."""

    tesseract: object
    params: dict = field(default_factory=dict)
    counter: CallCounter = field(default_factory=CallCounter)

    def apply(self, T: np.ndarray) -> np.ndarray:
        self.counter.bump_apply()
        return np.asarray(self.tesseract.apply({"T": T, **self.params})["mu"])

    def vjp(self, T: np.ndarray, cotangent_mu: np.ndarray) -> np.ndarray:
        self.counter.bump_vjp()
        out = self.tesseract.vector_jacobian_product(
            inputs={"T": T, **self.params},
            vjp_inputs=["T"],
            vjp_outputs=["mu"],
            cotangent_vector={"mu": np.asarray(cotangent_mu)},
        )
        return np.asarray(out["T"])
