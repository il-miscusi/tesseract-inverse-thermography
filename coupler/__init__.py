# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Differentiating through a coupled equilibrium that spans several Tesseracts."""

from .components import ClosureComponent, DarcyComponent, HeatComponent
from .coupled import CoupledSystem, FixedPointResult
from .problem import ColdPlate, DensityFilter, DesignChain, HeavisideProjection, Objective, binarize, grey_fraction

__all__ = [
    "ClosureComponent",
    "ColdPlate",
    "CoupledSystem",
    "DarcyComponent",
    "DensityFilter",
    "DesignChain",
    "HeavisideProjection",
    "binarize",
    "grey_fraction",
    "FixedPointResult",
    "HeatComponent",
    "Objective",
]
