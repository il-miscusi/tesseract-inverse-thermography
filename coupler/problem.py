# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""The design problem: a porous-metal cold plate cooling a chip hot spot.

Physical setting (per unit depth, SI units)
-------------------------------------------
A 20 mm x 10 mm cold plate.  Coolant is pushed left-to-right by a fixed pump
pressure.  A power device dissipates heat into a footprint on the bottom wall.
We choose where to put metal (gamma = 1) and where to leave open coolant path
(gamma = 0) so that the hot spot stays cool without strangling the flow.

Metal conducts ~300x better than water but blocks the coolant; open channel
carries heat away but conducts poorly.  That trade-off is the whole design
problem, and it is *only* visible when the two physics are solved together.

Scales are chosen so the system is strongly advective (Pe ~ 500) and so the
viscosity coupling is strong.  Two coolant presets are provided:

* ``oil``   (default) -- a single-phase dielectric immersion-cooling oil (ISO VG 22,
  ASTM D341 curve).  It thins ~4x between 300 K and 340 K, so heating the coolant
  strongly speeds up the flow that cools it: the feedback loop is load-bearing.
* ``water`` -- thins only ~1.5x across the same plate.  Kept as the weak-coupling
  reference; at this contrast the one-way (frozen-viscosity) gradient reaches a
  design as good as the coupled one, which is exactly the comparison we report.

The pump pressure is scaled per fluid so both run at a comparable Peclet number;
the heat load is scaled so the peak temperature rise is in the same band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Per-coolant operating points.  mu(T) itself lives in the closure Tesseract; these
# are the host-side bulk properties and the pump/heat-load scaling that keeps the
# two fluids at a comparable Peclet number and peak temperature rise.
FLUID_PRESETS = {
    "oil": dict(
        k_fluid=0.13,        # W/m/K   hydrocarbon oil
        rhoc=1.7e6,          # J/m^3/K (rho 850 kg/m^3, cp 2000 J/kg/K)
        pin=2.0e4,           # Pa      oil cold plates run tens of kPa
        q_peak=1.5e8,        # W/m^3  -> uniform plate ~+50 K, a good design ~+35 K under SIMP/RAMP
        t_in=285.0,          # K       chilled-loop inlet (12 C).  Cold oil is where mu(T) is steepest:
                             #         measured contraction 0.88-0.97 on sensible designs, plain Picard
                             #         stalls on the uniform plate; at 300 K it is a comfortable 0.55-0.64
    ),
    "water": dict(
        k_fluid=0.6,
        rhoc=4.18e6,
        pin=500.0,
        q_peak=3.0e8,
        t_in=300.0,
    ),
}


@dataclass
class ColdPlate:
    nx: int = 64
    ny: int = 32
    length: float = 0.020          # m
    height: float = 0.010          # m
    fluid: str = "oil"             # key into FLUID_PRESETS

    # flow  (None -> taken from the fluid preset)
    pin: float | None = None       # Pa pump pressure
    kmax: float = 1.0e-10          # m^2, open porous channel
    kmin: float = 1.0e-15          # m^2, solid metal (Brinkman floor)
    qramp: float = 8.0

    # heat  (None -> taken from the fluid preset)
    k_fluid: float | None = None   # W/m/K
    k_solid: float = 200.0         # W/m/K  (aluminium)
    k_penal: float = 3.0           # SIMP exponent on k(gamma); 1.0 = the linear rule that let grey material win
    rhoc: float | None = None      # J/m^3/K
    t_in: float | None = None      # K  (None -> taken from the fluid preset)
    inlet_conduction: float = 1.0       # see heat-transport/_physics.py
    inlet_conducts_solid: float = 0.0   # 1.0 reproduces the boundary-condition exploit

    # heat source: chip footprint on the bottom wall
    q_peak: float | None = None    # W/m^3
    chip_x: tuple[float, float] = (0.35, 0.65)   # fraction of length
    chip_y: tuple[float, float] = (0.0, 0.12)    # fraction of height

    # design
    volume_fraction: float = 0.35
    filter_radius_cells: float = 2.0
    projection_eta: float = 0.5    # Heaviside threshold; beta is scheduled by the optimiser

    solver_tol: float = 1e-12

    def fast(self) -> "ColdPlate":
        """The search-loop profile: component linear solves at 1e-10 instead of
        1e-12.  Pair with ``optimize(..., fp_tol=1e-8)``.  Only the *search* runs
        here -- every reported number is re-scored at the full tolerances, and
        the verification gates never use it.  Measured: the coupled FD check
        still passes at this setting (see figures/gradient_check_fast.json)."""
        from dataclasses import replace
        return replace(self, solver_tol=1e-10)

    # closure knob: 1 = the real fluid, 0 = isoviscous (no feedback). See the closure.
    coupling_scale: float = 1.0

    def __post_init__(self) -> None:
        preset = FLUID_PRESETS[self.fluid]
        for name in ("pin", "k_fluid", "rhoc", "q_peak", "t_in"):
            if getattr(self, name) is None:
                setattr(self, name, preset[name])
        self.dx = self.length / self.nx
        self.dy = self.height / self.ny

    # ------------------------------------------------------------------ fields
    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    def q_source(self) -> np.ndarray:
        q = np.zeros(self.shape)
        i0, i1 = (int(f * self.nx) for f in self.chip_x)
        j0, j1 = (int(f * self.ny) for f in self.chip_y)
        j1 = max(j1, j0 + 1)
        q[i0:i1, j0:j1] = self.q_peak
        return q

    def closure_params(self) -> dict:
        return dict(fluid=self.fluid, coupling_scale=self.coupling_scale)

    def darcy_params(self) -> dict:
        return dict(
            dx=self.dx,
            dy=self.dy,
            pin=self.pin,
            kmin=self.kmin,
            kmax=self.kmax,
            qramp=self.qramp,
            tol=self.solver_tol,
            maxit=200000,
        )

    def heat_params(self) -> dict:
        return dict(
            q_source=self.q_source(),
            dx=self.dx,
            dy=self.dy,
            k_fluid=self.k_fluid,
            k_solid=self.k_solid,
            k_penal=self.k_penal,
            rhoc=self.rhoc,
            t_in=self.t_in,
            inlet_conduction=self.inlet_conduction,
            inlet_conducts_solid=self.inlet_conducts_solid,
            tol=self.solver_tol,
            maxiter=8000,
        )


# --------------------------------------------------------------------- filter
class DensityFilter:
    """Standard topology-optimisation density filter.

    gamma = W rho with conic weights of radius r.  Without it the optimiser
    produces one-cell checkerboards that are a discretisation artefact rather
    than a design.  It is linear, so its adjoint is just W^T.
    """

    def __init__(self, shape: tuple[int, int], radius: float):
        self.shape = shape
        self.radius = float(radius)
        nx, ny = shape
        r = int(np.ceil(self.radius))
        ii, jj = np.meshgrid(
            np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij"
        )
        w = np.maximum(0.0, self.radius - np.sqrt(ii**2 + jj**2))
        self.kernel = w
        self.offsets = (ii, jj)
        # normalisation: sum of weights reaching each cell (edge-aware)
        ones = np.ones(shape)
        self.norm = self._correlate(ones)

    def _correlate(self, x: np.ndarray) -> np.ndarray:
        nx, ny = self.shape
        out = np.zeros(self.shape)
        ii, jj = self.offsets
        for di, dj, w in zip(ii.ravel(), jj.ravel(), self.kernel.ravel()):
            if w <= 0:
                continue
            xs0, xs1 = max(0, -di), min(nx, nx - di)
            ys0, ys1 = max(0, -dj), min(ny, ny - dj)
            out[xs0:xs1, ys0:ys1] += w * x[xs0 + di : xs1 + di, ys0 + dj : ys1 + dj]
        return out

    def forward(self, rho: np.ndarray) -> np.ndarray:
        """gamma = W rho with W row-normalised: every gamma is a convex combination
        of nearby rho, so gamma stays in [0, 1] whenever rho does.

        That bound is load-bearing.  A column-normalised (exactly volume-
        preserving) variant was tried; it pushes gamma above 1 next to a wall
        where rho = 1, the RAMP interpolation then returns a *negative*
        permeability, the Darcy matrix stops being positive definite, and CG
        breaks down in four iterations.  The volume budget is instead enforced
        on the *filtered* field by ``project_volume_filtered``."""
        return self._correlate(rho) / self.norm

    def adjoint(self, gbar: np.ndarray, rho: np.ndarray | None = None) -> np.ndarray:
        """W^T applied to a cotangent (the kernel is symmetric; normalisation is not).

        ``rho`` is accepted and ignored so the filter and the projected
        ``DesignChain`` share one calling convention."""
        return self._correlate(gbar / self.norm)


# --------------------------------------------------------------- projection
class HeavisideProjection:
    """Smoothed Heaviside  H_beta(x) = [tanh(b*eta) + tanh(b*(x-eta))] / [tanh(b*eta) + tanh(b*(1-eta))].

    beta -> 0 is the identity, beta -> inf is a hard threshold at eta.  It maps
    [0,1] onto [0,1] exactly, so the permeability interpolation stays positive.
    The optimiser raises beta on a schedule (continuation) so the design is
    pushed towards solid/void *after* it has found where the material wants to
    be -- the standard discipline in density-based topology optimisation, and
    the one that was missing here (see ``conductivity`` in the heat component
    for what that cost)."""

    def __init__(self, beta: float = 1.0, eta: float = 0.5):
        self.beta = float(beta)
        self.eta = float(eta)

    def _den(self) -> float:
        return np.tanh(self.beta * self.eta) + np.tanh(self.beta * (1.0 - self.eta))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.beta <= 0.0:
            return np.asarray(x, float)
        return (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (x - self.eta))) / self._den()

    def derivative(self, x: np.ndarray) -> np.ndarray:
        if self.beta <= 0.0:
            return np.ones_like(np.asarray(x, float))
        return self.beta * (1.0 - np.tanh(self.beta * (x - self.eta)) ** 2) / self._den()


class DesignChain:
    """rho -> W rho -> H_beta(W rho) = gamma.  Filter first, then project.

    Exposes the same ``forward`` / ``adjoint`` pair as ``DensityFilter`` so every
    script and the optimiser can use either.  ``adjoint`` needs ``rho`` because
    the projection is nonlinear."""

    def __init__(self, filt: DensityFilter, beta: float = 1.0, eta: float = 0.5):
        self.filt = filt
        self.proj = HeavisideProjection(beta, eta)

    @property
    def beta(self) -> float:
        return self.proj.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self.proj.beta = float(value)

    @property
    def shape(self):
        return self.filt.shape

    def forward(self, rho: np.ndarray) -> np.ndarray:
        return self.proj.forward(self.filt.forward(rho))

    def adjoint(self, gbar: np.ndarray, rho: np.ndarray) -> np.ndarray:
        xbar = gbar * self.proj.derivative(self.filt.forward(rho))
        return self.filt.adjoint(xbar)


def grey_fraction(gamma: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    """Fraction of cells that are neither coolant nor metal.  The number that
    should have been in every design summary from the first run."""
    g = np.asarray(gamma, float)
    return float(np.mean((g > lo) & (g < hi)))


def binarize(gamma: np.ndarray, volume_fraction: float) -> np.ndarray:
    """Volume-preserving hard threshold: the design an engineer would actually
    machine.  Its performance under the true physics is the only honest
    headline for a density-based design."""
    g = np.asarray(gamma, float)
    thr = np.quantile(g, 1.0 - volume_fraction)
    return (g > thr).astype(float)


# ------------------------------------------------------------------ objective
def softmax_temperature(T: np.ndarray, t_ref: float, p: float = 10.0):
    """Smooth maximum of (T - t_ref), and its gradient.

    A hard max has a gradient supported on a single cell, which makes the design
    chase one pixel.  The p-norm spreads it over the whole hot region.
    """
    d = np.maximum(T - t_ref, 0.0)
    scale = max(float(d.max()), 1e-12)
    z = d / scale
    s = float(np.mean(z**p))
    val = scale * s ** (1.0 / p)
    n = z.size
    grad = scale * (1.0 / p) * s ** (1.0 / p - 1.0) * (p * z ** (p - 1) / n) / scale
    grad = np.where(T - t_ref > 0.0, grad, 0.0)
    return val, grad


@dataclass
class Objective:
    """J = peak temperature rise + w_pump * pumping power, with a volume constraint
    handled by projection (not by penalty)."""

    plate: ColdPlate
    w_pump: float = 0.0
    p_norm: float = 10.0
    pump_ref: float = 1.0

    def value_and_cotangents(self, state) -> tuple[float, dict]:
        tpeak, dT = softmax_temperature(state.T, self.plate.t_in, self.p_norm)
        pump = self.plate.pin * state.flux
        j = tpeak + self.w_pump * pump / self.pump_ref
        return j, {
            "dJ_dT": dT,
            "dJ_dflux": self.w_pump * self.plate.pin / self.pump_ref,
            "t_peak": tpeak,
            "pump": pump,
        }
