"""Tests for the vector-valued random feature maps of Assumption 2.1."""

from __future__ import annotations

import numpy as np
import pytest

from kerop.data.pde import PoissonDataset
from kerop.data.spectral import SpectralOperatorModel
from kerop.features import (
    MercerFeatures,
    OperatorNTKFeatures,
    ScalarNTKFeatures,
    SeparableRFF,
    relu,
    relu_derivative,
)
from kerop.kernels import (
    OperatorNTKKernel,
    ScalarNTKKernel,
    SeparableGaussianKernel,
    arccos_kernel_pair,
)


def test_feature_tensor_and_design_matrix_shapes_agree() -> None:
    model = SpectralOperatorModel(n_modes=64, output_dim=5, seed=0)
    features = model.features(37, np.random.default_rng(0))
    inputs = np.random.default_rng(1).uniform(0.0, 1.0, size=(11, 1))
    tensor = features.feature_tensor(inputs)
    assert tensor.shape == (11, 5, features.coefficient_dim)
    assert features.coefficient_dim == features.n_summands * features.n_features
    assert features.design_matrix(inputs).shape == (11 * 5, features.coefficient_dim)


def test_rf_kernel_equals_the_feature_tensor_product() -> None:
    """:math:`K_M(u,\\tilde u) = \\Psi_M(u)\\Psi_M(\\tilde u)^*` by construction."""
    model = SpectralOperatorModel(n_modes=64, output_dim=4, seed=0)
    features = model.features(23, np.random.default_rng(0))
    inputs = np.random.default_rng(1).uniform(0.0, 1.0, size=(6, 1))
    tensor = features.feature_tensor(inputs)
    expected = np.einsum("iam,jbm->ijab", tensor, tensor)
    assert np.allclose(features.rf_kernel(inputs), expected)


def test_mercer_representation_is_exact_in_expectation() -> None:
    r"""Assumption 2.1 must hold exactly, not just approximately.

    With :math:`\pi(i)=\sigma_i/Z` and :math:`\varphi(u,i)=\sqrt Z\Phi_i(u)`,
    the expectation :math:`\sum_i\pi(i)\varphi(u,i)\otimes\varphi(\tilde u,i)`
    equals :math:`\sum_i\sigma_i\Phi_i(u)\otimes\Phi_i(\tilde u) = K(u,\tilde
    u)`.  Summing over *all* indices with their exact probabilities makes this a
    deterministic identity rather than a Monte Carlo check.
    """
    model = SpectralOperatorModel(n_modes=48, output_dim=4, seed=0)
    inputs = np.random.default_rng(2).uniform(0.0, 1.0, size=(5, 1))

    all_indices = np.arange(model.eigenvalues.size)
    basis = model._indexed_basis(inputs, all_indices)  # (n, d_v, S)
    weights = model.eigenvalues.reshape(-1)
    expectation = np.einsum("iam,jbm,m->ijab", basis, basis, weights, optimize=True)

    exact = model.kernel().blocks(inputs)
    assert np.allclose(expectation, exact, atol=1e-12)


def test_mercer_features_respect_the_kappa_bound() -> None:
    """The empirical feature norm must not exceed the analytic :math:`\\kappa^2`."""
    model = SpectralOperatorModel(n_modes=128, output_dim=6, seed=0)
    features = model.features(400, np.random.default_rng(0))
    inputs = np.random.default_rng(3).uniform(0.0, 1.0, size=(200, 1))
    assert features.empirical_kappa_squared(inputs) <= model.kappa_squared() + 1e-9


def test_scalar_ntk_has_p_equal_to_d_plus_two() -> None:
    """The NTK representation of a two-layer network with bias has p = d + 2.

    This is the value quoted in Appendix A.3 of the paper, where the feature
    threshold is reported as :math:`M = O(\\sqrt n\\,p)` with :math:`p = d+2`:
    one summand for the activation block and one for each of the
    :math:`\\tilde d = d+1` derivative blocks.
    """
    for input_dim in (1, 3, 14):
        features = ScalarNTKFeatures(input_dim, 8, np.random.default_rng(0))
        assert features.n_summands == input_dim + 2
    features = ScalarNTKFeatures(5, 8, np.random.default_rng(0), include_bias=False)
    assert features.n_summands == 6


def _mean_kernel_error(build_features, exact: np.ndarray, inputs, seeds=range(6)) -> float:
    """Average relative Gram error over several feature draws.

    A single realization of :math:`K_M` fluctuates enough that comparing two
    values of :math:`M` can misjudge the :math:`M^{-1/2}` Monte Carlo scaling;
    averaging over draws measures the expected error instead.
    """
    errors = []
    for seed in seeds:
        design = build_features(seed).design_matrix(inputs)
        errors.append(np.linalg.norm(design @ design.T - exact) / np.linalg.norm(exact))
    return float(np.mean(errors))


def test_scalar_ntk_features_converge_to_the_closed_form_kernel() -> None:
    inputs = np.random.default_rng(5).standard_normal((7, 2))
    exact = ScalarNTKKernel(2).block_gram(inputs)
    coarse = _mean_kernel_error(
        lambda s: ScalarNTKFeatures(2, 2_000, np.random.default_rng(s)), exact, inputs
    )
    fine = _mean_kernel_error(
        lambda s: ScalarNTKFeatures(2, 50_000, np.random.default_rng(s)), exact, inputs
    )
    assert coarse < 0.1
    # A 25-fold increase in M should cut the Monte Carlo error by about five.
    assert fine < coarse / 3.0


def test_operator_ntk_features_are_genuinely_vector_valued() -> None:
    """Each :math:`\\psi_m(u)` must be an element of :math:`\\mathcal{V}`.

    The distinguishing feature of the operator-valued case is that
    :math:`J(u)(\\cdot)` depends on :math:`x`, so a single random feature is a
    function rather than a scalar.  If the features were constant in :math:`x`,
    the columns of the feature tensor would be rank one across the output axis.
    """
    dataset = PoissonDataset(n_points=12)
    lifted = dataset.lift(dataset.sample(4, np.random.default_rng(0)).fields)
    features = OperatorNTKFeatures(
        dataset.feature_dim, dataset.n_points, 16, np.random.default_rng(1)
    )
    tensor = features.feature_tensor(lifted)
    assert tensor.shape == (4, 12, 16 * dataset.n_summands)
    variation = tensor.std(axis=1).max()
    assert variation > 1e-3


def test_operator_ntk_features_converge_to_the_closed_form_kernel() -> None:
    dataset = PoissonDataset(n_points=10)
    lifted = dataset.lift(dataset.sample(5, np.random.default_rng(0)).fields)
    kernel = OperatorNTKKernel(
        dataset.feature_dim, dataset.n_points, output_scale=dataset.output_scale()
    )
    exact = kernel.block_gram(lifted)

    def build(n_features: int):
        return lambda seed: OperatorNTKFeatures(
            dataset.feature_dim,
            dataset.n_points,
            n_features,
            np.random.default_rng(seed),
            output_scale=dataset.output_scale(),
        )

    coarse = _mean_kernel_error(build(2_000), exact, lifted)
    fine = _mean_kernel_error(build(50_000), exact, lifted)
    assert coarse < 0.1
    assert fine < coarse / 3.0


def test_arccos_kernels_match_direct_monte_carlo() -> None:
    """The arc-cosine formulas are the Gaussian expectations they claim to be."""
    rng = np.random.default_rng(0)
    points = rng.standard_normal((4, 3))
    weights = rng.standard_normal((400_000, 3))
    pre = points @ weights.T
    activated = relu(pre)
    derivative = relu_derivative(pre)
    k1, k0 = arccos_kernel_pair(points, points)
    assert np.allclose(activated @ activated.T / weights.shape[0], k1, atol=0.02)
    assert np.allclose(derivative @ derivative.T / weights.shape[0], k0, atol=0.01)


def test_arccos_kernels_handle_zero_vectors() -> None:
    """A zero input has no direction; both kernels must vanish against it."""
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    k1, k0 = arccos_kernel_pair(points, points)
    assert np.all(np.isfinite(k1)) and np.all(np.isfinite(k0))
    assert k1[0, 0] == 0.0 and k1[0, 1] == 0.0


def test_separable_rff_converges_to_the_gaussian_kernel() -> None:
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-1.0, 1.0, size=(6, 3))
    output_root = np.linalg.qr(rng.standard_normal((4, 4)))[0] @ np.diag([1.0, 0.7, 0.5, 0.3])
    kernel = SeparableGaussianKernel(0.8, output_root @ output_root.T)
    exact = kernel.block_gram(inputs)
    features = SeparableRFF(3, 200_000, 0.8, np.random.default_rng(1), output_root=output_root)
    design = features.design_matrix(inputs)
    approx = design @ design.T
    assert np.linalg.norm(approx - exact) / np.linalg.norm(exact) < 0.05


def test_feature_maps_validate_their_input_shapes() -> None:
    features = ScalarNTKFeatures(3, 8, np.random.default_rng(0))
    with pytest.raises(ValueError, match="expected inputs with 3 columns"):
        features.feature_tensor(np.zeros((4, 5)))

    operator_features = OperatorNTKFeatures(4, 6, 8, np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"expected lifted inputs"):
        operator_features.feature_tensor(np.zeros((4, 6)))


def test_mercer_features_reject_non_positive_eigenvalues() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        MercerFeatures(
            np.array([1.0, 0.0]),
            lambda u, i: np.zeros((1, 1, len(i))),
            1,
            4,
            np.random.default_rng(0),
        )