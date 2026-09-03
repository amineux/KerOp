"""Tests for the Theorem 3.4 and Corollary 3.5 prescriptions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kerop.theory import (
    check_assumptions,
    excess_risk_exponent,
    feature_exponent,
    features_required,
    iterations_required,
    min_sample_size,
    neural_operator_width,
    prescribe,
    regularization_exponent,
    regularization_parameter,
)


def test_rate_and_regularization_exponents() -> None:
    """The exponents are :math:`r/(2r+b)` and :math:`1/(2r+b)`."""
    assert excess_risk_exponent(0.5, 1.0) == pytest.approx(0.25)
    assert regularization_exponent(0.5, 1.0) == pytest.approx(0.5)
    assert excess_risk_exponent(1.0, 0.5) == pytest.approx(1.0 / 2.5)
    assert regularization_exponent(1.0, 0.5) == pytest.approx(1.0 / 2.5)


def test_well_specified_case_reproduces_the_paper_discussion() -> None:
    """At :math:`r=1/2`, :math:`b=1` the paper reports :math:`t_n=O(\\sqrt n)` and
    :math:`M_n=O(\\sqrt n\\log n)`.

    Section 3.2: "achieving a squared :math:`L^2`-error bound of order
    :math:`O(1/\\sqrt n)` requires :math:`t_n = 1/\\lambda_n = O(\\sqrt n)`
    iterations and :math:`M_n = O(\\sqrt n\\log n)` random features."
    """
    assert excess_risk_exponent(0.5, 1.0) == pytest.approx(0.25)
    assert feature_exponent(0.5, 1.0) == pytest.approx(0.5)
    n = 10_000
    assert iterations_required(n, 0.5, 1.0) == pytest.approx(math.sqrt(n), rel=1e-9)
    expected = math.ceil(math.sqrt(n) * math.log(n))
    assert features_required(n, 0.5, 1.0) == expected


def test_feature_exponent_branches_and_their_interpretation() -> None:
    """The branch structure of the feature requirement.

    In the misspecified regime the requirement coincides with the iteration
    count :math:`1/\\lambda_n`; in the smooth regime it is
    :math:`(1/\\lambda_n)^{2r}`.
    """
    b = 0.7
    assert feature_exponent(0.3, b) == pytest.approx(regularization_exponent(0.3, b))
    assert feature_exponent(1.5, b) == pytest.approx(2.0 * 1.5 * regularization_exponent(1.5, b))
    # The first two branches agree at r = 1/2, where b(2r-1) vanishes.
    assert feature_exponent(0.5 - 1e-9, b) == pytest.approx(feature_exponent(0.5, b), abs=1e-6)


def test_feature_exponent_jumps_at_r_equals_one() -> None:
    """The requirement of Theorem 3.4 is discontinuous at :math:`r=1`.

    The second branch gives :math:`(1+b(2r-1))/(2r+b)`, which at :math:`r=1` is
    :math:`(1+b)/(2+b)`, while the third gives :math:`2r/(2r+b) = 2/(2+b)`.
    These differ by :math:`(1-b)/(2+b)`, so the bound jumps upward as soon as
    :math:`r` exceeds one, and coincides only in the capacity-independent case
    :math:`b=1`.  The two branches come from different arguments in the proof -
    the :math:`r>1` case needs the novel operator inequalities of Section B.5 -
    so this is a property of the theorem as stated, not of the implementation.
    """
    for b in (0.3, 0.5, 0.7):
        below = feature_exponent(1.0, b)
        above = feature_exponent(1.0 + 1e-9, b)
        assert above > below
        assert above - below == pytest.approx((1.0 - b) / (2.0 + b), abs=1e-6)
    # No jump when b = 1.
    assert feature_exponent(1.0, 1.0) == pytest.approx(feature_exponent(1.0 + 1e-9, 1.0), abs=1e-6)


def test_smoothness_trades_iterations_against_features() -> None:
    """Section 3.2's trade-off: more smoothness means fewer steps, more features."""
    n, b = 5_000, 0.5
    rough = prescribe(n, 0.5, b)
    smooth = prescribe(n, 1.5, b)
    assert smooth.iterations < rough.iterations
    assert smooth.n_features > rough.n_features


def test_feature_requirement_is_linear_in_p() -> None:
    """The factor :math:`p` from the sum in (2.5) enters multiplicatively."""
    base = features_required(1_000, 0.5, 0.5, n_summands=1, include_log=False)
    for p in (2, 5, 16):
        scaled = features_required(1_000, 0.5, 0.5, n_summands=p, include_log=False)
        assert scaled == pytest.approx(p * base, rel=1e-9)


def test_regularization_parameter_decays_at_the_prescribed_rate() -> None:
    r, b = 0.5, 0.5
    sizes = np.array([100, 1_000, 10_000], dtype=float)
    lambdas = np.asarray(regularization_parameter(sizes, r, b))
    slope = np.polyfit(np.log(sizes), np.log(lambdas), 1)[0]
    assert slope == pytest.approx(-regularization_exponent(r, b), abs=1e-9)


def test_confidence_factor_is_applied_only_when_requested() -> None:
    plain = regularization_parameter(1_000, 0.5, 0.5)
    with_delta = regularization_parameter(1_000, 0.5, 0.5, delta=0.05)
    assert with_delta == pytest.approx(plain * math.log(2.0 / 0.05) ** 3)


def test_accelerated_iterations_are_the_square_root() -> None:
    """The :math:`\\nu`-method reaches :math:`\\lambda_n` in :math:`O(\\sqrt{t_n})` steps."""
    for n in (500, 5_000, 50_000):
        plain = iterations_required(n, 0.5, 0.5)
        accelerated = iterations_required(n, 0.5, 0.5, accelerated=True)
        # Both counts are rounded up to integers, so allow one step of slack.
        assert abs(accelerated - math.sqrt(plain)) <= 1.0
        assert accelerated < plain


def test_min_sample_size_diverges_at_the_easy_learning_boundary() -> None:
    """:math:`n_0=\\exp\\bigl(\\frac{2r+b}{2r+b-1}\\bigr)` blows up as
    :math:`2r+b\\downarrow1`."""
    assert min_sample_size(0.5, 1.0) == pytest.approx(math.exp(2.0))
    near = min_sample_size(0.5, 1.0 + 0.0)
    closer = min_sample_size(0.3, 0.45)  # 2r + b = 1.05
    assert closer > near
    assert min_sample_size(1.0, 1.0) < near


def test_assumptions_are_enforced() -> None:
    with pytest.raises(ValueError, match="r must be positive"):
        check_assumptions(0.0, 0.5)
    with pytest.raises(ValueError, match=r"b must lie in \[0, 1\]"):
        check_assumptions(0.5, 1.5)
    with pytest.raises(ValueError, match="easy-learning condition"):
        check_assumptions(0.2, 0.4)  # 2r + b = 0.8
    with pytest.raises(ValueError, match="qualification"):
        check_assumptions(1.5, 0.5, qualification=1.0)
    # Tikhonov's qualification of one is enough for r <= 1 but not beyond.
    check_assumptions(1.0, 0.5, qualification=1.0)


def test_neural_operator_width_follows_corollary_3_5() -> None:
    """Width must grow like :math:`T_n^{2r\\vee1}` and quadratically in
    :math:`\\tilde d`."""
    n, r, b = 5_000, 1.0, 0.5
    base = neural_operator_width(n, r, b, feature_dim=1, drift_bound=1.0)
    quadrupled = neural_operator_width(n, r, b, feature_dim=2, drift_bound=1.0)
    # Exact proportionality up to the rounding of each value to an integer.
    assert quadrupled == pytest.approx(4 * base, rel=1e-4)

    steps = iterations_required(n, r, b)
    expected = math.ceil(max(steps ** (2 * r), steps) * math.log(n) ** 2)
    assert base == expected


def test_prescribe_bundles_consistent_values() -> None:
    prescription = prescribe(4_000, 0.5, 0.5, n_summands=9)
    assert prescription.n == 4_000
    assert prescription.lam == pytest.approx(regularization_parameter(4_000, 0.5, 0.5))
    assert prescription.iterations == pytest.approx(1.0 / prescription.lam, rel=1.0)
    assert prescription.risk_bound_exponent == pytest.approx(-excess_risk_exponent(0.5, 0.5))
    assert prescription.meets_min_sample_size
    assert prescription.n_features == features_required(4_000, 0.5, 0.5, n_summands=9)


def test_prescribe_flags_samples_below_n0() -> None:
    """Near the easy-learning boundary the theorem needs a large sample."""
    assert not prescribe(5, 0.3, 0.45).meets_min_sample_size