# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Fast physics and derivative gates for the thermal-camera Tesseract."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "tesseracts" / "thermal-camera"),
]

from _render import (  # noqa: E402
    apply_homography,
    band_radiance,
    bilinear_sample,
    default_homography,
    psf_blur,
    psf_kernel,
    render,
    sensor_grid,
    surface_radiance,
    vignetting,
)
from coupler.thermography import (  # noqa: E402
    Adam,
    smoothness_penalty,
    softplus,
    softplus_grad,
    source_metrics,
    two_blob_source,
)

jax.config.update("jax_enable_x64", True)


def test_band_radiance_is_positive_and_monotone():
    values = np.asarray(band_radiance(jnp.array([250.0, 300.0, 400.0])))
    assert np.all(values > 0)
    assert np.all(np.diff(values) > 0)


def test_black_surface_equals_planck_band_radiance():
    T = jnp.array([[300.0, 320.0]])
    got = surface_radiance(T, jnp.ones_like(T), 295.0)
    np.testing.assert_allclose(got, band_radiance(T), rtol=1e-13)


def test_zero_emissivity_reflects_ambient():
    T = jnp.array([[310.0, 350.0]])
    got = surface_radiance(T, jnp.zeros_like(T), 295.0)
    np.testing.assert_allclose(got, band_radiance(295.0), rtol=1e-13)


def test_sensor_grid_shape_and_centres():
    u, v = sensor_grid(8, 6)
    assert u.shape == v.shape == (8, 6)
    assert float(u.min()) > 0 and float(u.max()) < 1
    assert float(v.min()) > 0 and float(v.max()) < 1


def test_identity_homography_is_identity():
    u, v = sensor_grid(4, 3)
    x, y = apply_homography(jnp.eye(3), u, v)
    np.testing.assert_allclose(x, u)
    np.testing.assert_allclose(y, v)


def test_bilinear_sample_preserves_constant_field():
    x, y = sensor_grid(9, 7)
    got = bilinear_sample(jnp.full((5, 4), 3.25), x, y, -1.0)
    np.testing.assert_allclose(got, 3.25)


def test_psf_kernel_is_positive_and_normalised():
    k = np.asarray(psf_kernel(1.2))
    assert np.all(k > 0)
    np.testing.assert_allclose(k.sum(), 1.0, rtol=1e-14)


def test_psf_blur_preserves_constant_image():
    got = psf_blur(jnp.full((12, 8), 7.0), 1.1)
    np.testing.assert_allclose(got, 7.0, rtol=1e-13, atol=1e-13)


def test_vignetting_dims_corners_more_than_centre():
    v = np.asarray(vignetting(21, 21, 0.45))
    assert v[10, 10] > v[0, 0]
    assert v.max() <= 1.0


def test_default_homography_is_finite_and_normalised():
    H = default_homography(0.25)
    assert H.shape == (3, 3)
    assert np.all(np.isfinite(H))
    assert H[2, 2] == 1.0


def test_render_has_declared_shape_and_finite_counts():
    T = jnp.full((8, 4), 320.0)
    eps = jnp.full_like(T, 0.9)
    image = render(T, eps, 1.2, 25.0, 500.0, jnp.eye(3), 295.0, 0.45, 16, 12)
    assert image.shape == (16, 12)
    assert np.all(np.isfinite(np.asarray(image)))


def test_rendered_counts_increase_with_temperature():
    eps = jnp.full((8, 4), 0.9)
    args = (eps, 1.2, 25.0, 500.0, jnp.eye(3), 295.0, 0.45, 16, 12)
    cold = render(jnp.full((8, 4), 300.0), *args)
    hot = render(jnp.full((8, 4), 340.0), *args)
    assert float(jnp.mean(hot)) > float(jnp.mean(cold))


def test_temperature_vjp_matches_directional_finite_difference():
    rng = np.random.default_rng(3)
    T = jnp.asarray(300.0 + 30.0 * rng.random((6, 4)))
    eps = jnp.asarray(0.75 + 0.2 * rng.random((6, 4)))
    direction = jnp.asarray(rng.normal(size=(6, 4)))

    def objective(temp):
        image = render(temp, eps, 1.1, 20.0, 400.0, jnp.eye(3), 295.0, 0.3, 10, 8)
        return jnp.mean(image**2)

    ad = float(jnp.vdot(jax.grad(objective)(T), direction))
    h = 1e-4
    fd = float((objective(T + h * direction) - objective(T - h * direction)) / (2 * h))
    assert abs(ad - fd) / max(abs(fd), 1e-12) < 1e-7


def test_smoothness_penalty_zero_for_constant_field():
    value, gradient = smoothness_penalty(np.ones((7, 5)))
    assert value == 0.0
    np.testing.assert_allclose(gradient, 0.0)


def test_softplus_is_positive_and_gradient_matches_fd():
    z = np.array([-2.0, 0.0, 2.0])
    assert np.all(softplus(z) > 0)
    h = 1e-6
    fd = (softplus(z + h) - softplus(z - h)) / (2 * h)
    np.testing.assert_allclose(softplus_grad(z), fd, rtol=1e-9)


def test_source_metrics_are_exact_for_identical_fields():
    q = two_blob_source((20, 10), 1.5e8)
    metrics = source_metrics(q, q)
    assert metrics["rel_l2"] == 0.0
    assert metrics["centroid_shift_cells"] == 0.0
    np.testing.assert_allclose(metrics["amplitude_ratio"], 1.0)
    np.testing.assert_allclose(metrics["total_power_ratio"], 1.0)


def test_two_blob_source_is_nonnegative_and_nonuniform():
    q = two_blob_source((20, 10), 1.5e8)
    assert q.shape == (20, 10)
    assert np.all(q >= 0)
    assert q.max() > q.mean() > q.min()


def test_adam_decreases_a_quadratic():
    x = np.array([3.0, -2.0])
    opt = Adam(lr=0.1)
    first = float(np.sum(x**2))
    for _ in range(50):
        x = opt.step(x, 2.0 * x)
    assert float(np.sum(x**2)) < first * 0.02
