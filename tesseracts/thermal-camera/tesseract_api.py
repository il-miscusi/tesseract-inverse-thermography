# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""The thermal-camera Tesseract: temperature field -> LWIR digital counts.

Fourth component of the system, and the one that makes this Track 05: it turns
the coupled multiphysics equilibrium into something a camera *measures*, and --
because every stage is differentiable -- lets a loss on pixels reach back
through the optics and the radiometry into the physics that produced the
temperature field.

Differentiated by JAX tracing end to end (no implicit solves in here: rendering
is a feed-forward chain, unlike the two PDE components it composes with).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

jax.config.update("jax_enable_x64", True)

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

from _render import default_homography, render

_DIFF = ("T", "eps", "psf_sigma", "gain", "offset")


class InputSchema(BaseModel):
    T: Differentiable[Array[(None, None), Float64]] = Field(
        description="Plate temperature field (K), cell-centred, from the physics."
    )
    eps: Differentiable[Array[(None, None), Float64]] = Field(
        description="Per-cell surface emissivity in (0, 1], same shape as T."
    )
    psf_sigma: Differentiable[Array[(), Float64]] = Field(
        description="Gaussian PSF width (sensor pixels)."
    )
    gain: Differentiable[Array[(), Float64]] = Field(
        description="Sensor gain (counts per W m^-2 sr^-1)."
    )
    offset: Differentiable[Array[(), Float64]] = Field(
        description="Sensor offset (counts)."
    )
    homography: Array[(3, 3), Float64] = Field(
        default_factory=lambda: default_homography(),
        description="Sensor-to-plate homography (camera pose). Fixed geometry.",
    )
    t_ambient: float = Field(
        default=295.0, description="Ambient/background temperature (K)."
    )
    half_fov_tan: float = Field(
        default=0.45, description="tan(half field of view) for cos^4 vignetting."
    )
    n_u: int = Field(default=96, description="Sensor pixels along u.")
    n_v: int = Field(default=64, description="Sensor pixels along v.")


class OutputSchema(BaseModel):
    image: Differentiable[Array[(None, None), Float64]] = Field(
        description="Rendered digital counts, shape (n_u, n_v)."
    )


def _arrays(inputs: InputSchema):
    return dict(
        T=jnp.asarray(np.asarray(inputs.T, dtype=np.float64)),
        eps=jnp.asarray(np.asarray(inputs.eps, dtype=np.float64)),
        psf_sigma=jnp.asarray(np.asarray(inputs.psf_sigma, dtype=np.float64)),
        gain=jnp.asarray(np.asarray(inputs.gain, dtype=np.float64)),
        offset=jnp.asarray(np.asarray(inputs.offset, dtype=np.float64)),
    )


def _fixed(inputs: InputSchema):
    return dict(
        Hm=jnp.asarray(np.asarray(inputs.homography, dtype=np.float64)),
        t_ambient=float(inputs.t_ambient),
        half_fov_tan=float(inputs.half_fov_tan),
        n_u=int(inputs.n_u),
        n_v=int(inputs.n_v),
    )


def _render_fn(fixed):
    def f(T, eps, psf_sigma, gain, offset):
        return render(T, eps, psf_sigma, gain, offset, **fixed)

    return f


def apply(inputs: InputSchema) -> OutputSchema:
    a = _arrays(inputs)
    img = _render_fn(_fixed(inputs))(*(a[k] for k in _DIFF))
    img = np.asarray(img)
    if not np.all(np.isfinite(img)):
        raise RuntimeError(
            f"thermal-camera produced non-finite pixels "
            f"({int(np.sum(~np.isfinite(img)))}/{img.size}); "
            f"T range [{float(a['T'].min()):.1f}, {float(a['T'].max()):.1f}] K"
        )
    return OutputSchema(image=img)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    if "image" not in vjp_outputs:
        return {}
    order = tuple(n for n in _DIFF if n in vjp_inputs)
    if not order:
        return {}
    a = _arrays(inputs)
    base = _render_fn(_fixed(inputs))

    def g(*args):
        vals = dict(zip(order, args))
        return base(*(vals.get(k, a[k]) for k in _DIFF))

    primals = tuple(a[n] for n in order)
    _, vjp_fn = jax.vjp(g, *primals)
    cot = jnp.asarray(np.asarray(cotangent_vector["image"], dtype=np.float64))
    grads = vjp_fn(cot)
    return {n: np.asarray(g_) for n, g_ in zip(order, grads)}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    if "image" not in jvp_outputs:
        return {}
    order = tuple(n for n in _DIFF if n in jvp_inputs)
    if not order:
        return {}
    a = _arrays(inputs)
    base = _render_fn(_fixed(inputs))

    def g(*args):
        vals = dict(zip(order, args))
        return base(*(vals.get(k, a[k]) for k in _DIFF))

    primals = tuple(a[n] for n in order)
    tangents = tuple(
        jnp.asarray(np.asarray(tangent_vector[n], dtype=np.float64)) for n in order
    )
    _, out = jax.jvp(g, primals, tangents)
    return {"image": np.asarray(out)}


def abstract_eval(abstract_inputs):
    return {
        "image": ShapeDType(
            shape=(int(abstract_inputs.n_u), int(abstract_inputs.n_v)),
            dtype="float64",
        )
    }
