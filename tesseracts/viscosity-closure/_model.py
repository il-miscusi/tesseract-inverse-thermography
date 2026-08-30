# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Hybrid mechanistic + learned model of coolant viscosity mu(T).

The mechanistic backbone is a two-parameter Arrhenius law, which gets the broad
exponential decay right but is systematically wrong for real liquids over a wide
temperature range.  A small MLP learns the *log-space correction*:

    mu(T) = mu_arrhenius(T) * exp( f_theta(That) )

Multiplying by an exponential guarantees mu > 0 for any network output, so the
flow solver downstream can never be handed a non-physical viscosity.

Two reference fluids are provided, selected at training time and frozen into the
saved weights:

* ``oil``  (default) -- a single-phase dielectric immersion-cooling oil, ISO VG 22,
  described by the ASTM D341 / Walther equation that the lubricant industry uses
  to interpolate kinematic viscosity between its two grade points:

      log10( log10( nu_cSt + 0.7 ) ) = A - B * log10( T )

  pinned at the ISO 3448 grade definition (22 cSt at 40 C) and a viscosity
  index of ~100 (4.3 cSt at 100 C).  Dynamic viscosity is nu * rho with
  rho = 850 kg/m^3.  Between 300 K and 360 K this fluid thins by ~5x, which
  is what makes the thermal-hydraulic feedback in the coupled problem strong.

* ``water`` -- the Vogel-Fulcher-Tammann correlation for liquid water,
  mu = A * 10**(B / (T - C)), A = 2.414e-5 Pa s, B = 247.8 K, C = 140 K.
  Thins by only ~1.5x across a 40 K plate: the weak-coupling reference.

The closure Tesseract also accepts a ``coupling_scale`` s in [0, 1] that
flattens the curve around T_ref without changing its value there:

    mu_s(T) = mu(T_ref) * ( mu(T) / mu(T_ref) )**s

s = 1 is the real fluid, s = 0 is an isoviscous fluid (no feedback at all).
This is the knob the coupling-strength sweep turns.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# ---------------------------------------------------------------- reference fluids
# VFT constants for liquid water
VFT_A = 2.414e-5
VFT_B = 247.8
VFT_C = 140.0

# ISO VG 22 oil: Walther / ASTM D341 through the two grade points
OIL_RHO = 850.0                      # kg/m^3
_OIL_PTS = ((313.15, 22.0), (373.15, 4.3))   # (T [K], nu [cSt])


def _walther_constants():
    (t1, n1), (t2, n2) = _OIL_PTS
    y1 = math.log10(math.log10(n1 + 0.7))
    y2 = math.log10(math.log10(n2 + 0.7))
    b = (y1 - y2) / (math.log10(t2) - math.log10(t1))
    a = y1 + b * math.log10(t1)
    return a, b


OIL_WALTHER_A, OIL_WALTHER_B = _walther_constants()


def water_viscosity(T):
    """VFT correlation for water. NOTE the base-10 exponent: writing it with exp()
    gives viscosities ~10x too low and silently rescales the whole flow problem."""
    return VFT_A * torch.pow(10.0, VFT_B / (T - VFT_C))


def oil_viscosity(T):
    """ASTM D341 (Walther) kinematic viscosity in cSt -> dynamic viscosity in Pa s."""
    zz = OIL_WALTHER_A - OIL_WALTHER_B * torch.log10(T)
    nu_cst = torch.pow(10.0, torch.pow(10.0, zz)) - 0.7
    return nu_cst * 1e-6 * OIL_RHO


FLUIDS = {
    "oil": dict(reference=oil_viscosity, t_range=(273.15, 393.15), t_ref=330.0, t_scale=40.0),
    "water": dict(reference=water_viscosity, t_range=(278.0, 368.0), t_ref=320.0, t_scale=40.0),
}


def fit_arrhenius(reference, t_range, n: int = 400) -> tuple[float, float]:
    """Least-squares fit of log(mu) = log A + B/T to the reference over its range,
    so the learned correction is measured against the *best* the mechanistic
    model can do, not a strawman."""
    T = torch.linspace(*t_range, n, dtype=torch.float64)
    y = torch.log(reference(T))
    X = torch.stack([torch.ones_like(T), 1.0 / T], dim=1)
    sol = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
    return float(torch.exp(sol[0])), float(sol[1])


class ViscosityClosure(nn.Module):
    """mu(T) = arrhenius(T) * exp(MLP(That)), optionally flattened by coupling_scale.

    The fluid's backbone constants and normalisation are *buffers*, so they travel
    inside the saved state dict and a loaded model cannot disagree with the data
    it was trained on.
    """

    def __init__(self, width: int = 32, fluid: str = "oil"):
        super().__init__()
        spec = FLUIDS[fluid]
        arr_a, arr_b = fit_arrhenius(spec["reference"], spec["t_range"])
        self.fluid = fluid
        self.register_buffer("arr_a", torch.tensor(arr_a, dtype=torch.float64))
        self.register_buffer("arr_b", torch.tensor(arr_b, dtype=torch.float64))
        self.register_buffer("t_ref", torch.tensor(spec["t_ref"], dtype=torch.float64))
        self.register_buffer("t_scale", torch.tensor(spec["t_scale"], dtype=torch.float64))
        self.register_buffer("t_lo", torch.tensor(spec["t_range"][0], dtype=torch.float64))
        self.register_buffer("t_hi", torch.tensor(spec["t_range"][1], dtype=torch.float64))
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        # start as a pure mechanistic model
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def arrhenius(self, T):
        return self.arr_a * torch.exp(self.arr_b / T)

    def _mu(self, T):
        # The closure is only trusted inside its training range; outside it the
        # curve is held at the boundary value (an oil past its flash point is not
        # a viscosity question).  The clamp is part of the model, so its
        # derivative is zero there, and a choked design whose coolant runs to
        # 500 K no longer hands the flow solver an extrapolated MLP output.
        T = torch.clamp(T, self.t_lo, self.t_hi)
        shape = T.shape
        that = ((T - self.t_ref) / self.t_scale).reshape(-1, 1)
        corr = self.net(that).reshape(shape)
        return self.arrhenius(T) * torch.exp(corr)

    def forward(self, T, coupling_scale: float = 1.0):
        mu = self._mu(T)
        if coupling_scale == 1.0:
            return mu
        mu_ref = self._mu(self.t_ref.reshape(1))[0]
        return mu_ref * torch.pow(mu / mu_ref, coupling_scale)


def reference_viscosity(fluid: str):
    return FLUIDS[fluid]["reference"]
