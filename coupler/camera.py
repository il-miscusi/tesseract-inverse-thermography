# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Typed wrapper around the thermal-camera Tesseract, plus its session helper.

Same design as ``components.py``: the wrapper exposes ``apply`` and a VJP and
knows nothing about what is inside the container.  The camera is the fourth
component and the second JAX one -- but the coupler cannot tell, which is the
point.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .components import CallCounter

ROOT = Path(__file__).resolve().parents[1]
CAMERA_API_DIR = ROOT / "tesseracts" / "thermal-camera"

CAMERA_DIFF = ("T", "eps", "psf_sigma", "gain", "offset")


@dataclass
class CameraComponent:
    """Thermal camera: (T, eps, psf_sigma, gain, offset) -> image."""

    tesseract: object
    params: dict = field(default_factory=dict)
    counter: CallCounter = field(default_factory=CallCounter)

    def _inputs(self, T, eps, psf_sigma, gain, offset) -> dict:
        return {
            "T": np.asarray(T, float),
            "eps": np.asarray(eps, float),
            "psf_sigma": np.asarray(psf_sigma, float),
            "gain": np.asarray(gain, float),
            "offset": np.asarray(offset, float),
            **self.params,
        }

    def apply(self, T, eps, psf_sigma, gain, offset) -> np.ndarray:
        self.counter.bump_apply()
        out = self.tesseract.apply(self._inputs(T, eps, psf_sigma, gain, offset))
        return np.asarray(out["image"])

    def vjp(
        self,
        T,
        eps,
        psf_sigma,
        gain,
        offset,
        cotangent_image: np.ndarray,
        wrt: tuple[str, ...] = CAMERA_DIFF,
    ) -> dict:
        self.counter.bump_vjp()
        out = self.tesseract.vector_jacobian_product(
            inputs=self._inputs(T, eps, psf_sigma, gain, offset),
            vjp_inputs=list(wrt),
            vjp_outputs=["image"],
            cotangent_vector={"image": np.asarray(cotangent_image, float)},
        )
        return {k: np.asarray(v) for k, v in out.items()}


@contextlib.contextmanager
def camera_session(params: dict | None = None, image: str = "thermal-camera",
                   inprocess: bool | None = None):
    """Open the camera Tesseract (served container, or in-process like session.py)."""
    import os

    from tesseract_core import Tesseract

    if inprocess is None:
        flag = os.environ.get("COUPLER_INPROCESS", "").strip()
        inprocess = bool(flag) and flag != "0"
    with contextlib.ExitStack() as stack:
        if inprocess:
            if str(CAMERA_API_DIR) not in sys.path:
                sys.path.insert(0, str(CAMERA_API_DIR))
            t = stack.enter_context(
                Tesseract.from_tesseract_api(CAMERA_API_DIR / "tesseract_api.py")
            )
        else:
            t = stack.enter_context(Tesseract.from_image(image))
        yield CameraComponent(t, dict(params or {}))
