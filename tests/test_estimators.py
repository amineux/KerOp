"""Tests for the random feature estimator (2.11) and the exact kernel baseline."""

from __future__ import annotations

import numpy as np
import pytest

from kerop.data.pde import PoissonDataset
from kerop.data.spectral import SpectralOperatorModel
from kerop.estimators import ExactOperatorFilter, VectorValuedRFRegressor
from kerop.features import OperatorNTKFeatures
from kerop.filters import Landweber, NuMethod
from kerop.kernels import OperatorNTKKernel
from kerop.metrics import excess_risk


def _model_and_data(n_samples: int = 200, n_features: int = 60, seed: int = 0):
    model = SpectralOperatorModel(
        r=0.5, b=0.5, n_modes=256, output_dim=5, seed=0, noise_std=0.02
    )
    rng = np.random.default_rng(seed)
    inputs, outputs = model.sample(n_samples, rng)
    features = model.features(n_features, rng)
    return model, features, inputs, outputs


def test_rf_tikhonov_equals_the_explicit_ridge_solution() -> None:
    r"""The estimator must be exactly :math:`(\widehat\Sigma_M+\lambda)^{-1}
    \widehat{\mathcal{S}}^*_M\mathbf v`."""
    model, features, inputs, outputs = _model_and_data()
    lam, scale = 0.05, 1.0
    estimator = VectorValuedRFRegressor(
        features, "tikhonov", lam, spectral_scale=scale, assemble=True
    ).fit(inputs, outputs)

    design = features.design_matrix(inputs)
    n_samples = outputs.shape[0]
    covariance = design.T @ design / n_samples
    rhs = design.T @ outputs.reshape(-1) / n_samples
    expected = np.linalg.solve(
        covariance + lam * np.eye(covariance.shape[0]), rhs
    )
    assert np.allclose(estimator.coefficients, expected, rtol=1e-8, atol=1e-10)


def test_matrix_free_and_assembled_paths_agree() -> None:
    """The O(nMt) and O(nM^2) routes must compute the same estimator."""
    model, features, inputs, outputs = _model_and_data()
    filt = Landweber.from_iterations(40)
    assembled = VectorValuedRFRegressor(
        features, filter_obj=filt, spectral_scale=2.0, assemble=True
    ).fit(inputs, outputs)
    matrix_free = VectorValuedRFRegressor(
        features, filter_obj=filt, spectral_scale=2.0, assemble=False
    ).fit(inputs, outputs)
    assert assembled.report.extras["matrix_free"] == 0.0
    assert matrix_free.report.extras["matrix_free"] == 1.0
    assert np.allclose(assembled.coefficients, matrix_free.coefficients, rtol=1e-8, atol=1e-10)


def test_predictions_use_the_same_feature_map_as_the_fit() -> None:
    _, features, inputs, outputs = _model_and_data()
    estimator = VectorValuedRFRegressor(features, "tikhonov", 0.05).fit(inputs, outputs)
    predictions = estimator.predict(inputs[:7])
    manual = features.feature_tensor(inputs[:7]) @ estimator.coefficients
    assert predictions.shape == (7, features.output_dim)
    assert np.allclose(predictions, manual)


def test_exact_tikhonov_equals_the_block_kernel_ridge_solution() -> None:
    r"""The exact estimator must reduce to :math:`(\mathbf{G}+n\lambda)^{-1}\mathbf v`.

    The filter acts on :math:`\mathbf{G}/n`, so
    :math:`c = (\mathbf{G}/n+\lambda)^{-1}\mathbf v` and the prediction carries
    the extra :math:`1/n`; this is the familiar operator-valued kernel ridge
    regression written so that the same filter objects apply.
    """
    model = SpectralOperatorModel(r=0.5, b=0.5, n_modes=128, output_dim=4, seed=0)
    rng = np.random.default_rng(1)
    inputs, outputs = model.sample(40, rng)
    kernel = model.kernel()
    lam = 0.03

    estimator = ExactOperatorFilter(kernel, "tikhonov", lam, spectral_scale=1.0)
    estimator.fit(inputs, outputs)

    gram = kernel.block_gram(inputs) / 40.0
    expected = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), outputs.reshape(-1))
    assert np.allclose(estimator.dual_coefficients, expected, rtol=1e-7, atol=1e-9)

    test_inputs = inputs[:5]
    cross = kernel.block_gram(test_inputs, inputs)
    manual = (cross @ expected / 40.0).reshape(5, model.output_dim)
    assert np.allclose(estimator.predict(test_inputs), manual, rtol=1e-8, atol=1e-10)


@pytest.mark.slow
def test_random_features_approach_the_exact_kernel_estimator() -> None:
    """As :math:`M` grows the random feature estimator must track the exact one.

    Theorem 3.4 is the statement that this happens fast enough to preserve the
    minimax rate; here we only check the qualitative convergence, which is what
    ties :class:`VectorValuedRFRegressor` to :class:`ExactOperatorFilter`.
    """
    model = SpectralOperatorModel(r=0.5, b=0.5, n_modes=256, output_dim=4, seed=0, noise_std=0.0)
    rng = np.random.default_rng(2)
    inputs, outputs = model.sample(150, rng)
    test_inputs, test_targets = model.test_set(400, rng)
    scale = model.kappa_squared()
    lam = 0.01

    exact = ExactOperatorFilter(model.kernel(), "tikhonov", lam, spectral_scale=scale)
    exact.fit(inputs, outputs)
    reference = exact.predict(test_inputs)

    gaps = []
    for n_features in (100, 1_000, 10_000):
        features = model.features(n_features, np.random.default_rng(7))
        approx = VectorValuedRFRegressor(
            features, "tikhonov", lam, spectral_scale=scale
        ).fit(inputs, outputs)
        gaps.append(excess_risk(approx.predict(test_inputs), reference))
    assert gaps[-1] < gaps[0] / 3.0
    assert gaps[-1] < 0.1 * excess_risk(reference, np.zeros_like(reference))


def test_estimator_fits_a_noiseless_well_specified_target() -> None:
    """With no noise and ample features, the excess risk must be small."""
    model = SpectralOperatorModel(
        r=1.0, b=0.5, n_modes=128, output_dim=4, seed=0, noise_std=0.0
    )
    rng = np.random.default_rng(3)
    inputs, outputs = model.sample(600, rng)
    test_inputs, test_targets = model.test_set(800, rng)
    features = model.features(600, rng)
    estimator = VectorValuedRFRegressor(
        features, "tikhonov", 1e-4, spectral_scale=model.kappa_squared()
    ).fit(inputs, outputs)
    relative = excess_risk(estimator.predict(test_inputs), test_targets) / model.target_norm()
    assert relative < 0.15


def test_operator_ntk_random_features_learn_the_poisson_map() -> None:
    """End-to-end check on a PDE solution operator with NTK features."""
    dataset = PoissonDataset(n_points=17, n_modes=8)
    rng = np.random.default_rng(0)
    train = dataset.sample(300, rng)
    test = dataset.sample(200, rng)
    features = OperatorNTKFeatures(
        dataset.feature_dim,
        dataset.n_points,
        300,
        rng,
        output_scale=dataset.output_scale(),
    )
    estimator = VectorValuedRFRegressor(
        features, filter_obj=NuMethod.from_iterations(60, nu=2.0)
    ).fit(dataset.lift(train.fields), train.outputs)
    predictions = estimator.predict(dataset.lift(test.fields))
    baseline = excess_risk(np.zeros_like(test.targets), test.targets)
    assert excess_risk(predictions, test.targets) < 0.2 * baseline


def test_fit_report_records_sizes_and_timings() -> None:
    model, features, inputs, outputs = _model_and_data()
    estimator = VectorValuedRFRegressor(features, "tikhonov", 0.05).fit(inputs, outputs)
    report = estimator.report
    assert report.operator_dim == features.coefficient_dim
    assert report.fit_seconds > 0.0
    assert report.spectral_scale > 0.0
    assert report.extras["n_samples"] == 200.0
    assert report.peak_operator_bytes > 0


def test_exact_estimator_cost_grows_with_the_output_dimension() -> None:
    """The exact operator dimension is :math:`nd_v`, not :math:`n`.

    This is the scaling the random feature method exists to avoid: for an
    operator-valued problem the block Gram matrix is :math:`d_v` times larger
    per side than in the scalar case.
    """
    dataset = PoissonDataset(n_points=8, n_modes=6)
    rng = np.random.default_rng(0)
    train = dataset.sample(20, rng)
    kernel = OperatorNTKKernel(
        dataset.feature_dim, dataset.n_points, output_scale=dataset.output_scale()
    )
    estimator = ExactOperatorFilter(kernel, "tikhonov", 0.01)
    estimator.fit(dataset.lift(train.fields), train.outputs)
    assert estimator.report.operator_dim == 20 * dataset.n_points


def test_estimators_validate_their_arguments() -> None:
    model, features, inputs, outputs = _model_and_data()
    with pytest.raises(ValueError, match="exactly one of lam or filter_obj"):
        VectorValuedRFRegressor(features, "tikhonov")
    with pytest.raises(ValueError, match="exactly one of lam or filter_obj"):
        VectorValuedRFRegressor(features, "tikhonov", 0.1, filter_obj=Landweber.from_iterations(3))
    with pytest.raises(ValueError, match="output_dim"):
        VectorValuedRFRegressor(features, "tikhonov", 0.1).fit(inputs, outputs[:, :2])
    with pytest.raises(RuntimeError, match="call fit before predict"):
        VectorValuedRFRegressor(features, "tikhonov", 0.1).predict(inputs)
    with pytest.raises(TypeError, match="block_gram"):
        ExactOperatorFilter(object(), "tikhonov", 0.1)