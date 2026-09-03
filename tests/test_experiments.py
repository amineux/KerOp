"""Tests for the experiment drivers, reporting, and the command line."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kerop.data.spectral import SpectralOperatorModel
from kerop.experiments import (
    DEFAULT_RATE_CONFIGS,
    RateConfig,
    _calibrate_lambda_constant,
    _cheapest_reaching,
    _interpolate_threshold,
    _lambda_constant_bounds,
    _ntk_target,
    rate_results_to_dicts,
    run_feature_threshold,
    run_filter_report,
    run_rate_experiment,
    run_walltime_benchmark,
)
from kerop.features import ScalarNTKFeatures
from kerop.report import flatten, provenance, write_csv, write_json

SMALL_CONFIG = RateConfig(
    name="test",
    r=0.5,
    b=0.5,
    filters=(("tikhonov", {}), ("nu_method", {"nu": 2.0})),
    n_grid=(150, 300, 600),
    n_modes=256,
    output_dim=4,
)


def test_default_configurations_satisfy_the_theorem_hypotheses() -> None:
    """Every shipped configuration must be a legal instance of Theorem 3.4."""
    from kerop import theory

    for config in DEFAULT_RATE_CONFIGS:
        theory.check_assumptions(config.r, config.b)
        assert 0.0 < config.b <= 1.0
        assert min(config.n_grid) >= theory.min_sample_size(config.r, config.b)


def test_default_configurations_avoid_the_trace_class_boundary() -> None:
    r"""No configuration uses :math:`b=1`.

    At :math:`b=1` the eigenvalues decay as :math:`i^{-1}`, the operator is not
    trace class, and :math:`\mathcal{N}(\lambda)` picks up a factor
    :math:`\log(1/\lambda)`.  There is then no clean power law in
    :math:`\lambda` for the measured exponent to agree with, so the instance
    cannot be used to test the rate.
    """
    for config in DEFAULT_RATE_CONFIGS:
        assert config.b < 1.0


def test_lambda_constant_bounds_keep_lambda_in_the_usable_window() -> None:
    model = SpectralOperatorModel(r=0.5, b=0.5, n_modes=1024, output_dim=4, seed=0)
    lower, upper = _lambda_constant_bounds(model, (150, 600), 0.5, 0.5)
    assert lower < upper
    window_low, window_high = model.usable_lambda_window()
    scale = model.kappa_squared()
    exponent = 1.0 / (2 * 0.5 + 0.5)
    for constant in (lower, upper):
        assert window_low * 0.999 <= constant * scale * 600.0**-exponent
        assert constant * scale * 150.0**-exponent <= window_high * 1.001


def test_lambda_constant_bounds_reject_an_impossible_range() -> None:
    model = SpectralOperatorModel(r=0.5, b=0.3, n_modes=8, output_dim=2, seed=0)
    with pytest.raises(ValueError, match="usable window|truncation too small|no admissible"):
        _lambda_constant_bounds(model, (10, 10**9), 0.5, 0.3)


def test_lambda_calibration_selects_the_risk_minimizing_constant() -> None:
    """The calibration must return a candidate from the admissible range."""
    model = SpectralOperatorModel(
        r=0.5, b=0.5, n_modes=256, output_dim=4, noise_std=0.05, seed=0
    )
    lower, upper = _lambda_constant_bounds(
        model, SMALL_CONFIG.n_grid, SMALL_CONFIG.r, SMALL_CONFIG.b
    )
    constant, trace = _calibrate_lambda_constant(
        model, SMALL_CONFIG, reference_filter=("tikhonov", {}), seed=0, repeats=1, n_candidates=5
    )
    assert lower <= constant <= upper
    assert len(trace) == 5
    best = min(trace, key=lambda row: row["mean_excess_risk"])
    assert best["lambda_constant"] == pytest.approx(constant)


def test_rate_experiment_produces_a_consistent_record() -> None:
    results = run_rate_experiment((SMALL_CONFIG,), repeats=2, n_test=400, verbose=False)
    assert len(results) == 1
    result = results[0]
    assert result.config["name"] == "test"
    # Two filters times three sample sizes times two repeats.
    assert len(result.rows) == 2 * 3 * 2
    assert len(result.summary) == 2
    for entry in result.summary:
        assert entry["n"] == [150.0, 300.0, 600.0]
        assert len(entry["mean_excess_risk"]) == 3
        assert len(entry["local_slopes"]) == 2
        assert entry["slope"] < 0.0
        assert 0.0 < entry["r_squared"] <= 1.0
    assert 0.0 < result.measured_r
    assert 0.0 < result.measured_b <= 1.0
    assert result.measured_exponent == pytest.approx(
        result.measured_r / (2 * result.measured_r + result.measured_b)
    )


def test_rate_experiment_measures_the_imposed_exponents() -> None:
    """The measured exponents must recover the ones the instance was built with."""
    results = run_rate_experiment((SMALL_CONFIG,), repeats=1, n_test=200, verbose=False)
    result = results[0]
    assert result.measured_r == pytest.approx(SMALL_CONFIG.r, abs=0.08)
    assert result.measured_b == pytest.approx(SMALL_CONFIG.b, abs=0.08)


def test_rate_experiment_flags_a_saturating_filter() -> None:
    r"""A filter with :math:`\nu < r\vee1` must be marked as violating the hypothesis."""
    config = RateConfig(
        name="saturating",
        r=1.5,
        b=0.5,
        filters=(("tikhonov", {}), ("landweber", {})),
        n_grid=(150, 300, 600),
        n_modes=256,
        output_dim=4,
        include_feature_log=False,
    )
    results = run_rate_experiment((config,), repeats=1, n_test=200, verbose=False)
    by_name = {entry["filter"]: entry for entry in results[0].summary}
    assert by_name["tikhonov"]["qualification_ok"] is False
    assert by_name["tikhonov"]["qualification_required"] == 1.5
    assert by_name["landweber"]["qualification_ok"] is True


def test_rate_results_are_json_serializable() -> None:
    results = run_rate_experiment((SMALL_CONFIG,), repeats=1, n_test=200, verbose=False)
    text = json.dumps(rate_results_to_dicts(results))
    assert "measured_exponent" in text


def test_ntk_target_is_normalized_and_shared_across_input_sets() -> None:
    r"""The target must have unit :math:`L^2` norm and one common scaling.

    Without the normalization the target's :math:`L^2` norm is of order
    :math:`(pM)^{-1/2}` and any nominal noise level swamps it, which would make
    the threshold study measure nothing.
    """
    rng = np.random.default_rng(0)
    features = ScalarNTKFeatures(2, 512, rng)
    first = rng.standard_normal((3_000, 2))
    second = rng.standard_normal((3_000, 2))
    values = _ntk_target(features, [first, second], rng)
    assert len(values) == 2
    combined = np.concatenate(values)
    assert np.sqrt((combined**2).mean()) == pytest.approx(1.0, abs=1e-12)
    # Both halves see the same function, so their scales agree closely.
    ratio = np.sqrt((values[0] ** 2).mean()) / np.sqrt((values[1] ** 2).mean())
    assert ratio == pytest.approx(1.0, abs=0.15)


def test_interpolate_threshold_recovers_a_known_crossing() -> None:
    features = np.array([1.0, 10.0, 100.0])
    errors = np.array([4.0, 2.0, 1.0])  # error = 4 * M^{-1/3} in log-log
    # The level 2.0 is attained exactly at M = 10.
    assert _interpolate_threshold(features, errors, 2.0) == pytest.approx(10.0, rel=1e-9)
    # Between grid points.
    crossing = _interpolate_threshold(features, errors, 2.8)
    assert 1.0 < crossing < 10.0
    # Already below the level at the smallest M, and never below it.
    assert _interpolate_threshold(features, errors, 5.0) == 1.0
    assert _interpolate_threshold(features, errors, 0.5) == 100.0


def test_feature_threshold_runs_and_checks_sufficiency() -> None:
    payload = run_feature_threshold(
        settings=((1, 400),),
        feature_multipliers=(0.25, 1.0, 4.0),
        iteration_grid=(8, 32),
        repeats=2,
        n_test=300,
        verbose=False,
    )
    assert payload["verdict"]["cases_tested"] == 2
    assert len(payload["grid"]) == 3 * 2
    for entry in payload["sufficiency"]:
        assert entry["features_at_unit_multiplier"] < entry["features_at_plateau"]
        assert entry["plateau_error"] > 0.0
    for entry in payload["thresholds"]:
        assert entry["threshold_features_interpolated"] > 0.0


def test_feature_threshold_grid_is_expressed_in_sqrt_n_p_units() -> None:
    payload = run_feature_threshold(
        settings=((3, 400),),
        feature_multipliers=(0.5, 1.0),
        iteration_grid=(8,),
        repeats=1,
        n_test=200,
        verbose=False,
    )
    reference = math.sqrt(400) * 5  # p = d + 2 = 5
    multipliers = sorted({row["feature_multiplier"] for row in payload["grid"]})
    assert multipliers[-1] == pytest.approx(1.0, abs=0.02)
    assert max(row["n_features"] for row in payload["grid"]) == pytest.approx(
        reference, rel=0.02
    )


def test_walltime_benchmark_reports_both_comparisons() -> None:
    payload = run_walltime_benchmark(
        task="spectral",
        train_sizes=(100, 200, 400),
        lambda_grid=(1e-2, 1e-3, 1e-4),
        feature_multipliers=(1.0, 4.0),
        iteration_grid=(64,),
        n_targets=3,
        task_kwargs={"n_modes": 256, "output_dim": 4, "n_test": 500},
        verbose=False,
    )
    assert len(payload["matched_sample_size"]) == 3
    for entry in payload["matched_sample_size"]:
        assert entry["exact_operator_dim"] == entry["n_train"] * 4
        # The random feature operator is much smaller than the exact one.
        assert entry["rf_operator_dim"] < entry["exact_operator_dim"]
    assert len(payload["matched_excess_risk"]) == 3
    for entry in payload["matched_excess_risk"]:
        for method in ("exact", "random_features"):
            choice = entry[method]
            if choice is not None:
                assert choice["excess_risk"] <= entry["target_risk"] * (1 + 1e-12)


def test_walltime_benchmark_rejects_an_unknown_task() -> None:
    with pytest.raises(KeyError, match="unknown task"):
        run_walltime_benchmark(task="burgers", verbose=False)


def test_cheapest_reaching_picks_the_fastest_qualifying_row() -> None:
    rows = [
        {"excess_risk": 0.1, "fit_seconds": 0.5},
        {"excess_risk": 0.05, "fit_seconds": 2.0},
        {"excess_risk": 0.04, "fit_seconds": 1.0},
    ]
    assert _cheapest_reaching(rows, 0.05)["fit_seconds"] == 1.0
    assert _cheapest_reaching(rows, 0.2)["fit_seconds"] == 0.5
    assert _cheapest_reaching(rows, 0.01) is None


def test_filter_report_records_saturation() -> None:
    r"""Tikhonov must saturate at :math:`\lambda^1` while Landweber follows
    :math:`\lambda^r`."""
    payload = run_filter_report(
        families=(("tikhonov", {}), ("landweber", {})),
        saturation_source_exponents=(0.5, 1.5),
        verbose=False,
    )
    assert len(payload["families"]) == 2
    rows = {
        (row["filter"], row["source_exponent_r"]): row for row in payload["bias_saturation"]
    }
    # At r = 0.5 both follow the source condition.
    assert rows[("tikhonov", 0.5)]["measured_bias_exponent"] == pytest.approx(0.5, abs=0.08)
    assert rows[("landweber", 0.5)]["measured_bias_exponent"] == pytest.approx(0.5, abs=0.08)
    # At r = 1.5 Tikhonov saturates at its qualification of one.
    assert rows[("tikhonov", 1.5)]["measured_bias_exponent"] == pytest.approx(1.0, abs=0.08)
    assert rows[("tikhonov", 1.5)]["saturated"] is True
    assert rows[("landweber", 1.5)]["measured_bias_exponent"] == pytest.approx(1.5, abs=0.1)
    assert rows[("landweber", 1.5)]["saturated"] is False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_provenance_records_the_environment() -> None:
    record = provenance()
    assert record["paper"] == "arXiv:2603.00971"
    assert record["kerop_version"]
    assert record["numpy"]
    assert record["cpu_count"] is None or record["cpu_count"] > 0


def test_flatten_nests_and_joins() -> None:
    flat = flatten({"a": 1, "b": {"c": 2.5, "d": [1, 2]}, "e": "x"})
    assert flat == {"a": 1, "b.c": 2.5, "b.d": "1;2", "e": "x"}


def test_write_json_handles_numpy_and_non_finite_values(tmp_path) -> None:
    path = write_json(
        tmp_path / "nested" / "out.json",
        {"a": np.float64(1.5), "b": np.arange(3), "c": math.inf, "d": {"e": np.int64(2)}},
    )
    document = json.loads(path.read_text())
    assert document["a"] == 1.5
    assert document["b"] == [0, 1, 2]
    assert document["c"] == "inf"
    assert document["d"]["e"] == 2
    assert "provenance" in document


def test_write_csv_uses_the_union_of_keys(tmp_path) -> None:
    path = write_csv(
        tmp_path / "out.csv", [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
    )
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "a,b,c"
    assert lines[1] == "1,2,"
    assert lines[2] == "3,,4"


def test_write_csv_handles_no_rows(tmp_path) -> None:
    path = write_csv(tmp_path / "empty.csv", [])
    assert path.read_text() == ""