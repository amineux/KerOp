"""Tests for the synthetic spectral instance and the PDE solution operators."""

from __future__ import annotations

import numpy as np
import pytest

from kerop.data import isometric_scale
from kerop.data.pde import DarcyDataset, PoissonDataset, make_dataset
from kerop.data.spectral import SpectralOperatorModel, cosine_multi_indices

# A fine midpoint rule integrates the cosine basis exactly up to aliasing, which
# lets the orthonormality and eigenvalue claims be checked without sampling.
QUADRATURE_POINTS = 4_000


def _midpoint_grid(n_points: int, dim: int) -> np.ndarray:
    axis = (np.arange(n_points) + 0.5) / n_points
    if dim == 1:
        return axis[:, None]
    grids = np.meshgrid(*([axis] * dim), indexing="ij")
    return np.stack([g.reshape(-1) for g in grids], axis=1)


def test_cosine_multi_indices_are_ordered_by_total_degree() -> None:
    indices = cosine_multi_indices(10, 2)
    degrees = indices.sum(axis=1)
    assert np.all(np.diff(degrees) >= 0)
    assert indices.shape == (10, 2)
    assert tuple(indices[0]) == (0, 0)
    with pytest.raises(ValueError, match="n_modes must be positive"):
        cosine_multi_indices(0, 1)


def test_cosine_basis_is_orthonormal() -> None:
    """The basis must be orthonormal in :math:`L^2([0,1]^d)` for the eigenvalue
    claim to hold."""
    model = SpectralOperatorModel(n_modes=25, output_dim=3, input_dim=1, seed=0)
    grid = _midpoint_grid(QUADRATURE_POINTS, 1)
    values = model.basis(grid)
    gram = values.T @ values / QUADRATURE_POINTS
    assert np.allclose(gram, np.eye(model.n_modes), atol=1e-8)


def test_cosine_basis_is_orthonormal_in_two_dimensions() -> None:
    model = SpectralOperatorModel(n_modes=15, output_dim=2, input_dim=2, seed=0)
    grid = _midpoint_grid(120, 2)
    values = model.basis(grid)
    gram = values.T @ values / grid.shape[0]
    assert np.allclose(gram, np.eye(model.n_modes), atol=1e-8)


def test_basis_is_uniformly_bounded_as_assumption_2_1_requires() -> None:
    for input_dim in (1, 2, 3):
        model = SpectralOperatorModel(n_modes=20, output_dim=2, input_dim=input_dim, seed=0)
        grid = np.random.default_rng(0).uniform(0.0, 1.0, size=(500, input_dim))
        assert np.abs(model.basis(grid)).max() <= model.basis_bound() + 1e-9


def test_rotations_are_orthogonal() -> None:
    model = SpectralOperatorModel(n_modes=40, output_dim=6, seed=0)
    products = np.einsum("jab,jcb->jac", model.rotations, model.rotations)
    assert np.allclose(products, np.eye(6)[None, :, :], atol=1e-12)


def test_kernel_integral_operator_has_the_claimed_eigenpairs() -> None:
    r"""Check :math:`\mathcal{L}(e_j g_{j,k}) = \mu_j\nu_k\,e_j g_{j,k}`.

    This is the identity the whole construction rests on: it is what makes the
    spectrum of :math:`\mathcal{L}` the product set :math:`\{\mu_j\nu_k\}` and
    hence makes :math:`r` and :math:`b` known rather than estimated.  The
    integral :math:`(\mathcal{L}F)(u) = \int K(u,\tilde u)F(\tilde u)
    d\rho_\mathcal{U}(\tilde u)` is evaluated by the midpoint rule, which is
    exact for these trigonometric integrands.
    """
    model = SpectralOperatorModel(n_modes=12, output_dim=4, input_dim=1, seed=0)
    kernel = model.kernel()
    quadrature = _midpoint_grid(1_500, 1)
    probes = np.array([[0.17], [0.43], [0.81]])
    blocks = kernel.blocks(probes, quadrature)  # (3, N, d_v, d_v)
    basis_quadrature = model.basis(quadrature)
    basis_probes = model.basis(probes)

    for mode in (0, 3, 7):
        for output in (0, 2):
            direction = model.rotations[mode, :, output]
            field = basis_quadrature[:, mode][:, None] * direction[None, :]
            applied = np.einsum("inab,nb->ia", blocks, field) / quadrature.shape[0]
            expected_value = model.mode_weights[mode] * model.output_weights[output]
            expected = expected_value * basis_probes[:, mode][:, None] * direction[None, :]
            assert np.allclose(applied, expected, atol=1e-8)


def test_kernel_is_not_separable() -> None:
    """Distinct rotations per mode mean :math:`K` is not :math:`k(u,\\tilde u)T`.

    If it were separable, the normalized diagonal blocks would be the same
    matrix at every input, and the problem would decouple into :math:`d_v`
    scalar regressions.
    """
    model = SpectralOperatorModel(n_modes=64, output_dim=5, seed=0)
    blocks = model.kernel().blocks(np.array([[0.2], [0.6], [0.9]]))
    normalized = [blocks[i, i] / np.trace(blocks[i, i]) for i in range(3)]
    spread = max(
        np.linalg.norm(normalized[i] - normalized[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert spread > 1e-2


def test_separable_model_is_recovered_by_flat_output_weights() -> None:
    """With flat :math:`\\nu`, :math:`T_j = d_v^{-1}I` and separability returns.

    Included to document why ``output_decay`` must be non-zero.
    """
    model = SpectralOperatorModel(n_modes=32, output_dim=5, output_decay=0.0, seed=0)
    blocks = model.kernel().blocks(np.array([[0.2], [0.6]]))
    identity = np.eye(5) / 5.0
    for i in range(2):
        scalar = blocks[i, i, 0, 0] / identity[0, 0]
        assert np.allclose(blocks[i, i], scalar * identity, atol=1e-12)


def test_effective_dimension_matches_its_definition() -> None:
    model = SpectralOperatorModel(n_modes=64, output_dim=4, seed=0)
    spectrum = model.eigenvalues.reshape(-1)
    for lam in (1e-1, 1e-3):
        expected = float((spectrum / (spectrum + lam)).sum())
        assert model.effective_dimension(lam) == pytest.approx(expected)
    assert np.asarray(model.effective_dimension(np.array([1e-1, 1e-2]))).shape == (2,)


def test_measured_exponents_match_the_nominal_ones() -> None:
    r"""Assumptions 3.2 and 3.3 must hold with the exponents requested.

    This is the precondition for using the instance to test Theorem 3.4: the
    effective dimension is measured from the exact spectrum and the source
    exponent from the exact bias, both without sampling.  ``b`` is kept at or
    below 0.5 because at :math:`b=1` the operator sits at the trace-class
    boundary, where :math:`\mathcal{N}(\lambda)\asymp\lambda^{-1}\log(1/\lambda)`
    and no clean power law exists.
    """
    for r, b in [(0.4, 0.5), (0.5, 0.5), (1.0, 0.5), (0.5, 0.3)]:
        model = SpectralOperatorModel(r=r, b=b, n_modes=4_096, output_dim=6, seed=0)
        low, high = model.usable_lambda_window()
        lambdas = np.logspace(np.log10(low), np.log10(high), 15)
        measured_b = -model.effective_dimension_fit(lambdas).slope
        measured_r = model.source_exponent_fit(lambdas).slope
        assert measured_b == pytest.approx(b, abs=0.03)
        assert measured_r == pytest.approx(r, abs=0.05)


def test_usable_lambda_window_shrinks_with_the_truncation() -> None:
    small = SpectralOperatorModel(n_modes=64, output_dim=4, b=0.5, seed=0)
    large = SpectralOperatorModel(n_modes=4_096, output_dim=4, b=0.5, seed=0)
    assert large.usable_lambda_window()[0] < small.usable_lambda_window()[0]
    with pytest.raises(ValueError, match="truncation too small"):
        SpectralOperatorModel(n_modes=4, output_dim=2, b=0.2, seed=0).usable_lambda_window(1e6)


def test_target_norm_and_sampling_are_consistent() -> None:
    model = SpectralOperatorModel(n_modes=256, output_dim=5, seed=0, noise_std=0.0)
    rng = np.random.default_rng(0)
    inputs, outputs = model.sample(4_000, rng)
    empirical = np.sqrt((outputs**2).sum(axis=1).mean())
    assert empirical == pytest.approx(model.target_norm(), rel=0.1)
    assert np.allclose(outputs, model.regression_operator(inputs))


def test_noise_enters_training_labels_but_not_the_test_targets() -> None:
    """``sample`` returns noisy labels; ``test_set`` returns :math:`G_\\rho` itself.

    The excess risk of Theorem 3.4 compares against the regression operator, so
    the test targets must be noiseless.
    """
    noise_std = 0.1
    model = SpectralOperatorModel(n_modes=64, output_dim=3, seed=0, noise_std=noise_std)
    train_inputs, train_labels = model.sample(5_000, np.random.default_rng(0))
    residuals = train_labels - model.regression_operator(train_inputs)
    assert residuals.std() == pytest.approx(noise_std, rel=0.05)

    test_inputs, test_targets = model.test_set(500, np.random.default_rng(1))
    assert np.allclose(test_targets, model.regression_operator(test_inputs))


def test_model_rejects_invalid_exponents() -> None:
    with pytest.raises(ValueError, match=r"b must lie in \(0, 1\]"):
        SpectralOperatorModel(b=1.5)
    with pytest.raises(ValueError, match="r must be positive"):
        SpectralOperatorModel(r=0.0)
    with pytest.raises(ValueError, match="easy-learning"):
        SpectralOperatorModel(r=0.2, b=0.4)


# --------------------------------------------------------------------------- #
# PDE datasets
# --------------------------------------------------------------------------- #


def test_isometric_scale() -> None:
    assert isometric_scale(4) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="must be positive"):
        isometric_scale(0)


def test_poisson_solution_satisfies_the_equation() -> None:
    """Check :math:`-u''=f` by finite differences on a fine grid."""
    dataset = PoissonDataset(n_points=257, n_modes=10)
    samples = dataset.sample(4, np.random.default_rng(0))
    solution = dataset.solve(samples.fields)
    spacing = dataset.grid[1] - dataset.grid[0]
    laplacian = (
        solution[:, 2:] - 2.0 * solution[:, 1:-1] + solution[:, :-2]
    ) / spacing**2
    relative = np.abs(-laplacian - samples.fields[:, 1:-1]).max() / np.abs(samples.fields).max()
    assert relative < 1e-3
    assert np.allclose(solution[:, 0], 0.0, atol=1e-12)
    assert np.allclose(solution[:, -1], 0.0, atol=1e-12)


def test_poisson_operator_is_linear() -> None:
    dataset = PoissonDataset(n_points=65, n_modes=8)
    fields = dataset.random_fields(3, np.random.default_rng(0))
    combination = 0.3 * fields[0] + 1.7 * fields[1]
    direct = dataset.solve(combination[None, :])[0]
    superposed = 0.3 * dataset.solve(fields[0][None, :])[0] + 1.7 * dataset.solve(
        fields[1][None, :]
    )[0]
    assert np.allclose(direct, superposed, atol=1e-10)


def test_darcy_solution_satisfies_the_equation() -> None:
    """Check :math:`-(au')'=f` by finite differences on a fine grid."""
    dataset = DarcyDataset(n_points=513, n_modes=6)
    samples = dataset.sample(4, np.random.default_rng(0))
    coefficient = dataset.coefficient(samples.fields)
    solution = dataset.solve(samples.fields)
    spacing = dataset.grid[1] - dataset.grid[0]
    midpoints = 0.5 * (coefficient[:, 1:] + coefficient[:, :-1])
    flux = midpoints * (solution[:, 1:] - solution[:, :-1]) / spacing
    residual = -(flux[:, 1:] - flux[:, :-1]) / spacing
    relative = np.abs(residual - dataset.source[None, 1:-1]).max() / np.abs(dataset.source).max()
    assert relative < 1e-3
    assert np.allclose(solution[:, 0], 0.0, atol=1e-14)
    assert np.allclose(solution[:, -1], 0.0, atol=1e-12)


def test_darcy_operator_is_nonlinear() -> None:
    dataset = DarcyDataset(n_points=65, n_modes=6)
    fields = dataset.random_fields(2, np.random.default_rng(0))
    direct = dataset.solve((fields[0] + fields[1])[None, :])[0]
    superposed = dataset.solve(fields[0][None, :])[0] + dataset.solve(fields[1][None, :])[0]
    assert not np.allclose(direct, superposed, atol=1e-6)


def test_darcy_coefficient_stays_positive_and_bounded() -> None:
    dataset = DarcyDataset(n_points=65, amplitude=0.6, n_modes=8)
    coefficient = dataset.coefficient(dataset.random_fields(500, np.random.default_rng(0)))
    assert coefficient.min() > 1.0 - dataset.amplitude
    assert coefficient.max() < 1.0 + dataset.amplitude


def test_lifting_operator_shape_and_channel_order() -> None:
    r"""The lifted features must be :math:`(A(u)(x), u(x), c(x))`."""
    dataset = PoissonDataset(n_points=16, n_lift=3)
    fields = dataset.random_fields(5, np.random.default_rng(0))
    lifted = dataset.lift(fields)
    assert lifted.shape == (5, 16, dataset.feature_dim)
    assert dataset.feature_dim == 3 + 1 + dataset.encoding_dim
    assert dataset.n_summands == 1 + dataset.feature_dim
    # The channel after the smoothing bank is the pointwise input trace.
    assert np.allclose(lifted[:, :, dataset.n_lift], fields)
    # The final encoding channel is the constant one.
    assert np.allclose(lifted[:, :, -1], 1.0)


def test_lifting_operator_is_nonlocal() -> None:
    """:math:`A(u)(x)` must depend on the whole input function.

    This is what makes the induced kernel operator-valued rather than a
    collection of pointwise scalar problems: perturbing the input away from
    :math:`x` still changes the feature at :math:`x`.
    """
    dataset = PoissonDataset(n_points=32, n_lift=3)
    field = dataset.random_fields(1, np.random.default_rng(0))
    perturbed = field.copy()
    perturbed[0, 25] += 1.0
    baseline = dataset.lift(field)
    changed = dataset.lift(perturbed)
    assert np.abs(changed[0, 5, : dataset.n_lift] - baseline[0, 5, : dataset.n_lift]).max() > 1e-6


def test_outputs_are_in_isometric_coordinates() -> None:
    dataset = PoissonDataset(n_points=33)
    samples = dataset.sample(6, np.random.default_rng(0))
    physical = dataset.solve(samples.fields)
    assert np.allclose(samples.targets, physical * dataset.output_scale())
    # Euclidean norm of the scaled output equals the empirical L2 norm.
    assert np.allclose(
        np.linalg.norm(samples.targets, axis=1),
        np.sqrt((physical**2).mean(axis=1)),
    )


def test_dataset_registry() -> None:
    assert isinstance(make_dataset("poisson", n_points=9), PoissonDataset)
    assert isinstance(make_dataset("darcy", n_points=9), DarcyDataset)
    with pytest.raises(KeyError, match="unknown dataset"):
        make_dataset("burgers")


def test_dataset_validates_arguments() -> None:
    with pytest.raises(ValueError, match="at least three grid points"):
        PoissonDataset(n_points=2)
    with pytest.raises(ValueError, match=r"amplitude must lie in \(0, 1\)"):
        DarcyDataset(amplitude=1.5)
    with pytest.raises(ValueError, match="expected fields with"):
        PoissonDataset(n_points=8).lift(np.zeros((2, 5)))