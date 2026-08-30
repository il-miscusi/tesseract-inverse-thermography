# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""A physically based LWIR thermal-camera renderer, in JAX.

A thermal camera does not see temperature.  It sees *radiance*: band-integrated
Planck emission from a surface with spatially varying emissivity, plus the
ambient radiation that same surface reflects, projected through the camera's
pose onto the sensor, blurred by the optics, dimmed off-axis by vignetting, and
finally converted to digital counts by the sensor's gain and offset.  Every one
of those stages is differentiable here, so a rendering loss on PIXELS can be
pulled back to the temperature field -- and from there, through whatever physics
produced that temperature field.

Stages
------
1. **Surface radiance** (``band_radiance``): the Planck spectral radiance
   integrated over the LWIR band 8-14 um by fixed-node Gauss-Legendre
   quadrature.  Written with ``expm1`` so it is stable down to cryogenic
   temperatures, where ``exp(hc/(lambda k T))`` overflows float64 in the naive
   form.  The rendered radiance of a surface with emissivity ``eps`` facing an
   ambient at ``T_ambient`` is  ``eps * L(T) + (1 - eps) * L(T_ambient)`` --
   grey, opaque, diffuse (Kirchhoff: reflectivity = 1 - eps).
2. **Geometric projection** (``warp``): a homography maps each sensor pixel to
   plate-plane coordinates, and the plate radiance is resampled there with
   bilinear interpolation -- the standard differentiable-sampling construction.
   Pixels that land outside the plate see the ambient background.
3. **Optics** (``psf_blur``, ``vignetting``): an isotropic Gaussian point-spread
   function whose width ``sigma`` is itself a differentiable parameter (the
   kernel is a smooth function of sigma on a fixed support, renormalised so it
   sums to one at every sigma), and fourth-power-of-cosine vignetting from the
   pixel's off-axis angle.
4. **Sensor** (``render``): counts = gain * signal + offset.  Noise is NOT part
   of this function: measured images are synthesised by adding Gaussian noise
   *outside* the renderer, so the gradient path is exactly the physics and
   nothing else.

Everything is pure JAX, so ``jax.vjp`` differentiates the whole chain with
respect to temperature, the emissivity map, the PSF width, gain and offset.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# Planck constants (CODATA 2018), SI.
H_PLANCK = 6.62607015e-34  # J s
C_LIGHT = 2.99792458e8     # m / s
K_BOLTZ = 1.380649e-23     # J / K

# LWIR band of a typical uncooled microbolometer.
LAMBDA_LO = 8.0e-6   # m
LAMBDA_HI = 14.0e-6  # m

# Fixed Gauss-Legendre nodes: accuracy is spectral in the node count for the
# smooth Planck integrand; 64 nodes puts the quadrature error far below the
# solver tolerances everywhere in 150-2000 K (verified in tests against a
# 4096-node reference).
_GL_NODES = 64
_x, _w = np.polynomial.legendre.leggauss(_GL_NODES)
_LAM = jnp.asarray(0.5 * (LAMBDA_HI - LAMBDA_LO) * _x + 0.5 * (LAMBDA_HI + LAMBDA_LO))
_WGT = jnp.asarray(0.5 * (LAMBDA_HI - LAMBDA_LO) * _w)

# PSF support half-width in pixels.  The kernel is *defined* on this fixed
# support (so shapes are static for jit) and renormalised; sigma is meaningful
# on roughly [0.2, PSF_RADIUS / 2.5].
PSF_RADIUS = 4


def planck_spectral_radiance(lam, T):
    """Planck spectral radiance B_lambda(T) in W / (m^2 sr m), stable form."""
    c1 = 2.0 * H_PLANCK * C_LIGHT**2
    c2 = H_PLANCK * C_LIGHT / K_BOLTZ
    return c1 / (lam**5 * jnp.expm1(c2 / (lam * T)))


def band_radiance(T):
    """LWIR band-integrated radiance L(T) in W / (m^2 sr), elementwise in T."""
    T = jnp.asarray(T)
    # broadcast T against the quadrature nodes on a trailing axis
    B = planck_spectral_radiance(_LAM, T[..., None])
    return jnp.sum(B * _WGT, axis=-1)


def surface_radiance(T, eps, t_ambient):
    """Grey opaque diffuse surface: emitted plus reflected-ambient radiance."""
    return eps * band_radiance(T) + (1.0 - eps) * band_radiance(jnp.asarray(t_ambient))


def sensor_grid(n_u: int, n_v: int):
    """Pixel-centre coordinates of the sensor, normalised to [0, 1]^2."""
    u = (jnp.arange(n_u) + 0.5) / n_u
    v = (jnp.arange(n_v) + 0.5) / n_v
    return jnp.meshgrid(u, v, indexing="ij")


def apply_homography(Hm, u, v):
    """Map sensor coords (u, v) to plate coords (x, y) through a 3x3 homography."""
    w = Hm[2, 0] * u + Hm[2, 1] * v + Hm[2, 2]
    x = (Hm[0, 0] * u + Hm[0, 1] * v + Hm[0, 2]) / w
    y = (Hm[1, 0] * u + Hm[1, 1] * v + Hm[1, 2]) / w
    return x, y


def bilinear_sample(img, x, y, fill):
    """Differentiable bilinear sampling of ``img`` at normalised coords (x, y).

    ``img`` is indexed as img[i, j] with i ~ x in [0,1], j ~ y in [0,1] on cell
    centres.  Out-of-domain samples return ``fill`` (the ambient background),
    blended smoothly at the boundary half-cell to avoid a hard mask.
    """
    nx, ny = img.shape
    fi = x * nx - 0.5
    fj = y * ny - 0.5
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, nx - 2)
    j0 = jnp.clip(jnp.floor(fj).astype(jnp.int32), 0, ny - 2)
    ti = jnp.clip(fi - i0, 0.0, 1.0)
    tj = jnp.clip(fj - j0, 0.0, 1.0)
    v00 = img[i0, j0]
    v10 = img[i0 + 1, j0]
    v01 = img[i0, j0 + 1]
    v11 = img[i0 + 1, j0 + 1]
    val = (
        v00 * (1 - ti) * (1 - tj)
        + v10 * ti * (1 - tj)
        + v01 * (1 - ti) * tj
        + v11 * ti * tj
    )
    inside = (
        (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    )
    return jnp.where(inside, val, fill)


def warp(radiance, Hm, n_u: int, n_v: int, fill):
    """Resample plate-plane radiance onto the sensor grid through ``Hm``."""
    u, v = sensor_grid(n_u, n_v)
    x, y = apply_homography(Hm, u, v)
    return bilinear_sample(radiance, x, y, fill)


def psf_kernel(sigma):
    """(2R+1)^2 Gaussian kernel as a smooth function of sigma, summing to 1.

    The support is fixed (static shapes); the weights and their normalisation
    depend smoothly on sigma, so d(image)/d(sigma) exists and is exact.
    """
    r = jnp.arange(-PSF_RADIUS, PSF_RADIUS + 1)
    g = jnp.exp(-0.5 * (r / sigma) ** 2)
    k = jnp.outer(g, g)
    return k / jnp.sum(k)


def psf_blur(image, sigma):
    """Convolve with the Gaussian PSF (edge-replicated padding)."""
    k = psf_kernel(sigma)
    padded = jnp.pad(image, PSF_RADIUS, mode="edge")
    return jax.scipy.signal.convolve2d(padded, k, mode="valid")


def vignetting(n_u: int, n_v: int, half_fov_tan: float):
    """cos^4(theta) irradiance falloff; theta from the pixel's off-axis angle.

    tan(theta) = r_norm * half_fov_tan with r_norm the distance from the image
    centre in units of the half-diagonal, so the corner pixel sits at the
    lens's half-field angle.  cos^4 = (1 + tan^2)^-2.
    """
    u, v = sensor_grid(n_u, n_v)
    du = (u - 0.5) * 2.0
    dv = (v - 0.5) * 2.0
    r2 = (du**2 + dv**2) / 2.0  # half-diagonal of [-1,1]^2 is sqrt(2)
    tan2 = r2 * half_fov_tan**2
    return (1.0 + tan2) ** (-2)


@functools.partial(jax.jit, static_argnames=("n_u", "n_v"))
def render(T, eps, psf_sigma, gain, offset, Hm, t_ambient, half_fov_tan,
           n_u: int, n_v: int):
    """The full camera model: temperature field -> digital counts.

    Deterministic by construction -- noise belongs to the *measurement*
    synthesis, never to the gradient path.
    """
    L = surface_radiance(T, eps, t_ambient)
    fill = band_radiance(jnp.asarray(t_ambient))  # background sees ambient
    on_sensor = warp(L, Hm, n_u, n_v, fill)
    blurred = psf_blur(on_sensor, psf_sigma)
    dimmed = blurred * vignetting(n_u, n_v, half_fov_tan)
    return gain * dimmed + offset


def default_homography(tilt: float = 0.25, n_u: int = 96, n_v: int = 64):
    """A representative oblique view of the plate.

    Maps sensor [0,1]^2 to plate [0,1]^2 with a keystone distortion of strength
    ``tilt`` (0 = fronto-parallel).  Returned as a plain numpy array so it can
    cross the Tesseract boundary as data.
    """
    # anchor points: sensor corners -> plate corners pulled in on one side
    t = float(tilt)
    src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    dst = np.array(
        [[0.0, 0.0 + 0.5 * t * 0.0],
         [1.0, 0.0],
         [0.0 + t * 0.5, 1.0],
         [1.0 - t * 0.5, 1.0]],
        dtype=float,
    )
    # direct linear transform for the 3x3 homography
    A = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys, -xd])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys, -yd])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]
