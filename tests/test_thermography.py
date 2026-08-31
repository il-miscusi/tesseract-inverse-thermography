# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Unit gates for the inverse-thermography utilities (priors, optimizer, metrics).

Everything here is container-free numpy so it runs in the judge in milliseconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupler.thermography import (
    Adam,
    smoothness_penalty,
    softplus,
    softplus_grad,
    source_metrics,
    total_variation_penalty,
    two_blob_source,
)


def test_experiment_d_support_and_block_average():
    from scripts.experiment_d_generalization import (block_average, chip_mask,
                                                     interior_pixel_mask)
    from coupler import ColdPlate

    plate = ColdPlate(nx=8, ny=8)
    mask = chip_mask(plate.shape, plate)
    assert mask.any()
    assert not mask[:, -1].any()
    fine = np.arange(64.0).reshape(8, 8)
    coarse = block_average(fine, 2)
    assert coarse.shape == (4, 4)
    assert coarse[0, 0] == np.mean(fine[:2, :2])
    roi = interior_pixel_mask(np.eye(3), (8, 8), (16, 16))
    assert roi.shape == (16, 16)
    assert roi.any() and not roi.all()


def _fd_grad(fn, q, h=1e-7):
    g = np.zeros_like(q)
    for idx in np.ndindex(q.shape):
        qp = q.copy(); qp[idx] += h
        qm = q.copy(); qm[idx] -= h
        g[idx] = (fn(qp)[0] - fn(qm)[0]) / (2 * h)
    return g


def test_tv_gradient_matches_fd():
    q = np.random.default_rng(1).uniform(0.0, 1.0, (7, 5))
    _, g = total_variation_penalty(q)
    gn = _fd_grad(total_variation_penalty, q)
    assert np.abs(g - gn).max() < 1e-7 * max(np.abs(gn).max(), 1.0)


def test_smoothness_gradient_matches_fd():
    q = np.random.default_rng(2).uniform(0.0, 1.0, (6, 4))
    _, g = smoothness_penalty(q)
    gn = _fd_grad(smoothness_penalty, q)
    assert np.abs(g - gn).max() < 1e-6 * max(np.abs(gn).max(), 1.0)


def test_tv_prefers_piecewise_constant_over_oscillation():
    flat = np.ones((8, 8))
    step = np.ones((8, 8)); step[4:, :] = 0.0
    osc = np.indices((8, 8)).sum(axis=0) % 2 * 1.0
    v_flat = total_variation_penalty(flat)[0]
    v_step = total_variation_penalty(step)[0]
    v_osc = total_variation_penalty(osc)[0]
    assert v_flat < v_step < v_osc


def test_softplus_grad_is_sigmoid_and_positive():
    z = np.linspace(-8, 8, 33)
    g = softplus_grad(z)
    h = 1e-6
    gn = (softplus(z + h) - softplus(z - h)) / (2 * h)
    assert np.abs(g - gn).max() < 1e-6
    assert (g > 0).all()


def test_softplus_grad_is_finite_at_extreme_optimizer_probes():
    z = np.array([-1e4, -1e3, 0.0, 1e3, 1e4])
    with np.errstate(over="raise", invalid="raise"):
        g = softplus_grad(z)
    assert np.isfinite(g).all()
    assert np.array_equal(g[[0, 1]], np.zeros(2))
    assert np.array_equal(g[[3, 4]], np.ones(2))


def test_adam_converges_on_quadratic():
    opt = Adam(lr=0.1)
    x = np.array([5.0, -3.0])
    for _ in range(500):
        x = opt.step(x, 2 * (x - np.array([1.0, 2.0])))
    assert np.abs(x - np.array([1.0, 2.0])).max() < 1e-3


def test_source_metrics_identity_and_shift():
    q = two_blob_source((32, 16), 1.5e8)
    m = source_metrics(q, q)
    assert m["rel_l2"] == 0.0
    assert m["amplitude_ratio"] == 1.0
    assert m["centroid_shift_cells"] == 0.0
    assert m["total_power_ratio"] == 1.0
    m2 = source_metrics(np.roll(q, 2, axis=0), q)
    assert 1.5 < m2["centroid_shift_cells"] < 2.5
