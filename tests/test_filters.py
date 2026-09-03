"""Tests for the spectral regularization families of Definition 2.2."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh

from kerop.filters import (
    FILTER_REGISTRY,
    HeavyBall,
    IteratedTikhonov,
    Landweber,
    NuMethod,
    SpectralCutoff,
    Tikhonov,
    _IterativeFilter,
    filter_diagnostics,
    make_filter,
    measure_qualification,
)

# (registry name, extra construction options) for every shipped family.
FAMILIES = [
    ("tikhonov", {}),
    ("iterated_tikhonov", {"order": 2}),
    ("iterated_tikhonov", {"order": 3}),
    ("landweber", {}),
    ("landweber", {"step": 0.5}),
    ("cutoff", {}),
    ("heavy_ball", {"momentum": 0.9}),
    ("heavy_ball", {"momentum": 0.0}),
    ("nu_method", {"nu": 1.0}),
    ("nu_method", {"nu": 2.0}),
]


def _ids(params: list[tuple[str, dict]]) -> list[str]:
    return [f"{name}{sorted(kwargs.items())}" for name, kwargs in params]


@pytest.mark.parametrize(("name", "kwargs"), FAMILIES, ids=_ids(FAMILIES))
def test_definition_2_2_constants_are_finite(name: str, kwargs: dict) -> None:
    """Each family must satisfy (2.7)-(2.9) with finite constants D, E, c_0."""
    diagnostics = filter_diagnostics(name, **kwargs)
    assert math.isfinite(diagnostics.D) and diagnostics.D > 0.0
    assert math.isfinite(diagnostics.E) and diagnostics.E > 0.0
    assert math.isfinite(diagnostics.c0) and diagnostics.c0 > 0.0
    # The residual r_lambda(t) = 1 - t phi(t) is a contraction for all families
    # considered here, and t phi(t) is bounded by a small multiple of one.
    assert diagnostics.c0 <= 1.0 + 1e-9
    assert diagnostics.D <= 4.0


def test_tikhonov_constants_are_exactly_one() -> None:
    """Tikhonov attains D = E = c_0 = 1, the textbook values.

    On the domain :math:`(0,1]` of Definition 2.2 the supremum
    :math:`\\sup_t t/(t+\\lambda) = 1/(1+\\lambda)` is approached but not
    attained, which is why ``D`` is one only up to :math:`O(\\lambda)`.
    """
    diagnostics = filter_diagnostics("tikhonov")
    assert diagnostics.D <= 1.0
    assert diagnostics.D == pytest.approx(1.0, abs=1e-5)
    assert diagnostics.E == pytest.approx(1.0, abs=1e-9)
    assert diagnostics.c0 == pytest.approx(1.0, abs=1e-9)


def test_iterated_tikhonov_E_grows_with_order() -> None:
    """Iterated Tikhonov of order m has E = m, since phi_lambda(0) = m/lambda."""
    for order in (1, 2, 3, 4):
        diagnostics = filter_diagnostics("iterated_tikhonov", order=order)
        assert diagnostics.E == pytest.approx(float(order), rel=1e-6)


@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    [
        ("tikhonov", {}, 1.0),
        ("iterated_tikhonov", {"order": 2}, 2.0),
        ("iterated_tikhonov", {"order": 3}, 3.0),
        ("nu_method", {"nu": 1.0}, 1.0),
        ("nu_method", {"nu": 2.0}, 2.0),
        ("nu_method", {"nu": 3.0}, 3.0),
    ],
)
def test_measured_qualification_matches_theory(name: str, kwargs: dict, expected: float) -> None:
    """The measured qualification reproduces the value known analytically.

    Theorem 3.4 requires qualification at least :math:`r\\vee1`, so these values
    determine which source exponents each family can reach.
    """
    report = measure_qualification(name, q_grid=np.arange(0.5, 4.01, 0.5), **kwargs)
    assert report.nu_estimate == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [("landweber", {}), ("cutoff", {}), ("heavy_ball", {"momentum": 0.9})],
)
def test_unbounded_qualification_families(name: str, kwargs: dict) -> None:
    """Landweber, cut-off and heavy-ball do not saturate on the probed range.

    Their constants :math:`c_q(\\lambda)` stay flat as :math:`\\lambda\\to0` up to
    the largest exponent probed, i.e. no saturation is detectable, matching the
    infinite qualification of the first two.  Heavy-ball with fixed momentum
    behaves asymptotically like gradient descent with step
    :math:`\\alpha/(1-\\beta)`, so it belongs in this group too.
    """
    q_grid = np.arange(0.5, 4.01, 0.5)
    report = measure_qualification(name, q_grid=q_grid, **kwargs)
    assert report.nu_estimate == pytest.approx(float(q_grid[-1]), abs=1e-9)


def test_saturating_family_reveals_its_qualification_in_the_slope() -> None:
    """Beyond qualification, c_q(lambda) ~ lambda^{nu-q}; the slope recovers nu."""
    report = measure_qualification("tikhonov", q_grid=np.arange(0.5, 4.01, 0.5))
    assert report.saturation_estimate() == pytest.approx(1.0, abs=0.1)
    report = measure_qualification("nu_method", q_grid=np.arange(0.5, 5.01, 0.5), nu=2.0)
    assert report.saturation_estimate() == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize(("name", "kwargs"), FAMILIES, ids=_ids(FAMILIES))
def test_exact_residual_matches_generic_definition(name: str, kwargs: dict) -> None:
    """The cancellation-free residual agrees with ``1 - t*phi(t)``.

    Each family overrides ``residual_function`` with a closed form so that the
    qualification diagnostics are not swamped by floating-point cancellation.
    This checks the override against the definition at a moderate lambda, where
    the generic route is still accurate.
    """
    flt = make_filter(name, 1e-3, **kwargs)
    t = np.logspace(-8.0, 0.0, 400)
    reference = 1.0 - t * flt.filter_function(t)
    assert np.allclose(flt.residual_function(t), reference, atol=1e-11)


@pytest.mark.parametrize(
    ("filt", "tolerance"),
    [
        (Landweber.from_iterations(137, step=0.8), 1e-12),
        (Landweber.from_iterations(3, step=1.0), 1e-12),
        (HeavyBall.from_iterations(137, step=0.5, momentum=0.8), 1e-6),
        (HeavyBall.from_iterations(64, step=1.0, momentum=0.0), 1e-9),
        (HeavyBall.from_iterations(50, step=0.4, momentum=0.99), 1e-6),
    ],
)
def test_closed_form_filter_matches_its_recursion(filt: _IterativeFilter, tolerance: float) -> None:
    """Closed-form filters agree with the recursion they replace.

    Landweber and heavy-ball override ``filter_function`` with closed forms so
    the spectral diagnostics do not have to run millions of iterations.  The
    heavy-ball tolerance is looser because recovering ``phi`` from the residual
    divides by ``t``, which loses digits for tiny ``t``; the residual itself,
    which is what the analysis uses, is accurate to about 1e-12.
    """
    t = np.logspace(-8.0, 0.0, 300)
    reference = filt.filter_function_recursive(t)
    assert np.allclose(filt.filter_function(t), reference, rtol=tolerance, atol=1e-12)


@pytest.mark.parametrize(("name", "kwargs"), FAMILIES, ids=_ids(FAMILIES))
def test_apply_on_diagonal_operator_equals_scalar_filter(name: str, kwargs: dict) -> None:
    """``apply`` on a diagonal operator must reproduce the scalar filter.

    This is the bridge between the algorithm and the analysis: the operator
    :math:`\\phi_\\lambda(\\widehat\\Sigma_M)` the estimator computes has to be
    the function the qualification condition (2.10) constrains.
    """
    flt = make_filter(name, 5e-3, **kwargs)
    spectrum = np.linspace(1e-4, 1.0, 40)
    operator = np.diag(spectrum)
    result = flt.apply(operator, np.ones(spectrum.size))
    assert np.allclose(result, flt.filter_function(spectrum), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("name", "kwargs"), FAMILIES, ids=_ids(FAMILIES))
def test_apply_commutes_with_eigendecomposition(name: str, kwargs: dict) -> None:
    """On a general symmetric PSD matrix, ``apply`` equals ``U phi(L) U^T b``."""
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    spectrum = np.sort(rng.uniform(1e-3, 1.0, size=12))
    operator = basis @ np.diag(spectrum) @ basis.T
    rhs = rng.standard_normal(12)

    flt = make_filter(name, 1e-2, **kwargs)
    evals, evecs = eigh(operator)
    expected = evecs @ (flt.filter_function(evals) * (evecs.T @ rhs))
    assert np.allclose(flt.apply(operator, rhs), expected, rtol=1e-7, atol=1e-9)


def test_tikhonov_apply_solves_the_ridge_system() -> None:
    """Tikhonov's ``apply`` is a solve of ``(A + lambda I) x = b``."""
    rng = np.random.default_rng(1)
    factor = rng.standard_normal((20, 8))
    operator = factor.T @ factor / 20.0
    rhs = rng.standard_normal(8)
    flt = Tikhonov(0.07)
    expected = np.linalg.solve(operator + 0.07 * np.eye(8), rhs)
    assert np.allclose(flt.apply(operator, rhs), expected)


def test_iterative_filters_accept_a_matvec_callable() -> None:
    """Iterative filters must run without ever forming the operator."""
    rng = np.random.default_rng(2)
    factor = rng.standard_normal((30, 10))
    operator = factor.T @ factor / 30.0
    operator /= np.linalg.norm(operator, 2)
    rhs = rng.standard_normal(10)
    for flt in (
        Landweber.from_iterations(40),
        HeavyBall.from_iterations(40, momentum=0.5),
        NuMethod.from_iterations(20, nu=2.0),
    ):
        dense = flt.apply(operator, rhs)
        free = flt.apply(lambda x: operator @ x, rhs, dim=10)
        assert np.allclose(dense, free)


def test_direct_filters_reject_a_matvec_callable() -> None:
    """Direct filters need the matrix and must say so rather than fail obscurely."""
    for flt in (Tikhonov(0.1), IteratedTikhonov(0.1, order=2), SpectralCutoff(0.1)):
        with pytest.raises(TypeError, match="dense operator"):
            flt.apply(lambda x: x, np.ones(3))


def test_heavy_ball_with_zero_momentum_is_gradient_descent() -> None:
    """Momentum zero must reduce exactly to the Landweber iteration."""
    t = np.logspace(-8.0, 0.0, 200)
    heavy = HeavyBall.from_iterations(75, step=0.6, momentum=0.0)
    plain = Landweber.from_iterations(75, step=0.6)
    assert np.allclose(heavy.residual_function(t), plain.residual_function(t), atol=1e-14)
    assert heavy.lam == pytest.approx(plain.lam)


def test_landweber_realizes_the_requested_lambda() -> None:
    """``from_lambda`` reports the level the integer iteration count achieves."""
    for target in (1e-1, 1e-2, 3e-3):
        flt = Landweber.from_lambda(target, step=0.5)
        assert flt.lam == pytest.approx(1.0 / (0.5 * flt.iterations))
        assert flt.lam <= target * (1.0 + 1e-12)


def test_nu_method_needs_the_square_root_of_landweber_iterations() -> None:
    """The nu-method reaches a given lambda in O(sqrt(t)) iterations.

    This is the acceleration the paper cites from Pagliana & Rosasco: the same
    implicit regularization level as gradient descent at a quadratically smaller
    iteration count.
    """
    lam = 1e-4
    gradient = Landweber.from_lambda(lam)
    accelerated = NuMethod.from_lambda(lam, nu=2.0)
    assert accelerated.iterations == pytest.approx(math.sqrt(gradient.iterations), rel=0.02)
    assert accelerated.matvec_count < gradient.matvec_count


def test_registry_covers_all_families_and_rejects_unknown_names() -> None:
    assert set(FILTER_REGISTRY) == {
        "tikhonov",
        "iterated_tikhonov",
        "landweber",
        "cutoff",
        "heavy_ball",
        "nu_method",
    }
    with pytest.raises(KeyError, match="unknown filter"):
        make_filter("does_not_exist", 0.1)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="lambda must be positive"):
        Tikhonov(0.0)
    with pytest.raises(ValueError, match="order must be"):
        IteratedTikhonov(0.1, order=0)
    with pytest.raises(ValueError, match="step must lie"):
        Landweber(0.1, 10, step=2.0)
    with pytest.raises(ValueError, match="momentum must lie"):
        HeavyBall(0.1, 10, 1.0, momentum=1.0)
    with pytest.raises(ValueError, match="nu must be positive"):
        NuMethod(0.1, 10, 1.0, nu=0.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        make_filter("tikhonov", 0.1, step=0.5)