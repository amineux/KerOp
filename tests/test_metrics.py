"""Tests for risk measurement and power-law rate estimation."""

from __future__ import annotations

import numpy as np
import pytest

from kerop.metrics import (
    excess_risk,
    fit_power_law,
    monte_carlo_standard_error,
    relative_error,
)


def test_excess_risk_is_the_l2_norm_of_the_difference() -> None:
    predictions = np.array([[1.0, 0.0], [0.0, 2.0]])
    targets = np.array([[0.0, 0.0], [0.0, 0.0]])
    # Per-sample squared norms are 1 and 4; the mean is 2.5.
    assert excess_risk(predictions, targets) == pytest.approx(np.sqrt(2.5))
    assert excess_risk(targets, targets) == 0.0


def test_excess_risk_accepts_one_dimensional_outputs() -> None:
    assert excess_risk(np.array([[1.0], [1.0]]), np.array([[0.0], [0.0]])) == pytest.approx(1.0)
    assert excess_risk(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == pytest.approx(float(np.sqrt(2.0)))


def test_relative_error_normalizes_by_the_target_norm() -> None:
    targets = np.array([[3.0, 4.0], [3.0, 4.0]])
    predictions = np.zeros_like(targets)
    assert relative_error(predictions, targets) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="target operator vanishes"):
        relative_error(predictions, np.zeros_like(targets))


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        excess_risk(np.zeros((2, 3)), np.zeros((2, 4)))


def test_power_law_fit_recovers_a_known_exponent_exactly() -> None:
    x = np.array([10.0, 100.0, 1_000.0, 10_000.0])
    y = 3.7 * x**-0.25
    fit = fit_power_law(x, y)
    assert fit.slope == pytest.approx(-0.25, abs=1e-12)
    assert np.exp(fit.intercept) == pytest.approx(3.7, rel=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.slope_stderr == pytest.approx(0.0, abs=1e-9)
    assert fit.n_points == 4


def test_confidence_interval_covers_the_truth_under_noise() -> None:
    """With multiplicative noise the interval should contain the true exponent."""
    rng = np.random.default_rng(0)
    truth = -0.3333
    covered = 0
    trials = 200
    for _ in range(trials):
        x = np.array([100.0, 300.0, 1_000.0, 3_000.0, 10_000.0])
        y = 2.0 * x**truth * np.exp(0.05 * rng.standard_normal(x.size))
        if fit_power_law(x, y).covers(truth):
            covered += 1
    # A 95% interval, so this is a loose but meaningful calibration check.
    assert covered / trials > 0.85


def test_weights_are_honoured() -> None:
    """Down-weighting a corrupted point must pull the fit back to the truth."""
    x = np.array([10.0, 100.0, 1_000.0, 10_000.0])
    y = 2.0 * x**-0.4
    corrupted = y.copy()
    corrupted[-1] *= 5.0
    unweighted = fit_power_law(x, corrupted)
    weighted = fit_power_law(x, corrupted, weights=np.array([1.0, 1.0, 1.0, 1e-6]))
    assert abs(weighted.slope - (-0.4)) < abs(unweighted.slope - (-0.4))


def test_fit_requires_positive_data_and_enough_points() -> None:
    with pytest.raises(ValueError, match="at least three points"):
        fit_power_law(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        fit_power_law(np.array([1.0, 2.0, 3.0]), np.array([1.0, -2.0, 3.0]))
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_power_law(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_monte_carlo_standard_error_shrinks_with_the_test_set() -> None:
    rng = np.random.default_rng(0)
    errors = []
    for size in (500, 50_000):
        predictions = rng.standard_normal((size, 3))
        errors.append(monte_carlo_standard_error(predictions, np.zeros_like(predictions)))
    assert errors[1] < errors[0] / 5.0
    assert monte_carlo_standard_error(np.zeros((10, 2)), np.zeros((10, 2))) == 0.0