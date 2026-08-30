# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Open all three Tesseracts and hand back a wired-up CoupledSystem."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from tesseract_core import Tesseract

ROOT = Path(__file__).resolve().parents[1]
API_DIRS = {
    "darcy": ROOT / "tesseracts" / "darcy-flow",
    "heat": ROOT / "tesseracts" / "heat-transport",
    "closure": ROOT / "tesseracts" / "viscosity-closure",
}

from .components import ClosureComponent, DarcyComponent, HeatComponent
from .coupled import CoupledSystem

IMAGES = {
    "darcy": "darcy-flow",
    "heat": "heat-transport",
    "closure": "viscosity-closure",
}


def _open(stack, name: str, image: str, inprocess: set[str]):
    """A served container, or the same tesseract_api.py imported in-process.

    In-process is the iteration mode: identical code, no HTTP, no JSON -- an
    order of magnitude faster per call on these small arrays.  It needs the
    component's dependencies on the host (JAX, torch; for the Fortran solver a
    host-compiled `darcy` binary via DARCY_SOLVER_BIN).  Containers remain the
    reproduction mode and what every verification gate runs against.
    """
    if name in inprocess:
        api_dir = API_DIRS[name]
        if str(api_dir) not in sys.path:
            sys.path.insert(0, str(api_dir))
        return stack.enter_context(Tesseract.from_tesseract_api(api_dir / "tesseract_api.py"))
    return stack.enter_context(Tesseract.from_image(image))


def inprocess_components() -> set[str]:
    """COUPLER_INPROCESS=1 -> heat + closure (+ darcy when DARCY_SOLVER_BIN is set);
    COUPLER_INPROCESS=heat,closure -> exactly those."""
    flag = os.environ.get("COUPLER_INPROCESS", "").strip()
    if not flag or flag == "0":
        return set()
    if flag == "1":
        chosen = {"heat", "closure"}
        if os.environ.get("DARCY_SOLVER_BIN"):
            chosen.add("darcy")
        return chosen
    return {v.strip() for v in flag.split(",") if v.strip()}


@contextlib.contextmanager
def coupled_session(plate, images: dict | None = None, inprocess: set[str] | None = None):
    """Serve the three Tesseracts for the duration of the block (or run them in-process)."""
    imgs = {**IMAGES, **(images or {})}
    inproc = inprocess_components() if inprocess is None else set(inprocess)
    with contextlib.ExitStack() as stack:
        darcy_t = _open(stack, "darcy", imgs["darcy"], inproc)
        heat_t = _open(stack, "heat", imgs["heat"], inproc)
        closure_t = _open(stack, "closure", imgs["closure"], inproc)
        system = CoupledSystem(
            darcy=DarcyComponent(darcy_t, plate.darcy_params()),
            heat=HeatComponent(heat_t, plate.heat_params()),
            closure=ClosureComponent(closure_t, plate.closure_params()),
        )
        yield system
