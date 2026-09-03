r"""Exact operator-valued kernels.

Each kernel here is the :math:`M\to\infty` limit of one of the random feature
maps in :mod:`kerop.features`, so that the random feature estimator can be
compared against the exact kernel method it approximates.  The interface is a
single method returning the *block Gram matrix*

.. math::

    \mathbf{G} \in \mathbb{R}^{n_1 d_v\times n_2 d_v},
    \qquad
    \mathbf{G}[i d_v:(i+1)d_v,\; j d_v:(j+1)d_v] = K(u_i, \tilde u_j),

whose row blocks are ordered to match ``outputs.reshape(-1)`` for outputs of
shape ``(n, d_v)``.

The neural tangent kernels are available in closed form.  For a two-layer
network with ReLU activation and :math:`b^{(0)}\sim\mathcal{N}(0,s^2I)`, the
limiting kernel (2.2) is, with :math:`a = J(u)(x)` and
:math:`a'=J(\tilde u)(x')`,

.. math::

    K(u,\tilde u)(x,x') =
      \underbrace{\frac{s^2\|a\|\|a'\|}{2\pi}
        \bigl(\sin\theta + (\pi-\theta)\cos\theta\bigr)}
        _{\mathbb{E}[\psi(u)\psi(\tilde u)]}
      \;+\;
      \underbrace{\langle a,a'\rangle\,\frac{\pi-\theta}{2\pi}}
        _{\mathbb{E}[\sum_j \psi'_j(u)\psi'_j(\tilde u)]},

where :math:`\theta = \angle(a,a')`.  The two terms are the arc-cosine kernels
of degree one and zero (Cho & Saul, 2009): ReLU is positively homogeneous, so
the activation block carries the factor :math:`s^2` while the derivative block,
built from the scale-invariant Heaviside function, does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "OperatorValuedKernel",
    "MercerOperatorKernel",
    "ScalarNTKKernel",
    "OperatorNTKKernel",
    "SeparableGaussianKernel",
    "arccos_kernel_pair",
]


def _relu_ntk_gram(
    left: Array, right: Array, weight_scale: float, chunk_size: int
) -> Array:
    """Assemble the ReLU NTK Gram matrix in row blocks.

    The exact operator-valued baseline needs an :math:`nd_v\\times nd_v` matrix,
    and :func:`arccos_kernel_pair` allocates several intermediates of that size.
    At :math:`nd_v\\approx10^4` those temporaries alone run to several gigabytes,
    so the rows are processed in blocks and only the result is held in full.
    """
    n_left, n_right = left.shape[0], right.shape[0]
    result = np.empty((n_left, n_right), dtype=float)
    for start in range(0, n_left, chunk_size):
        stop = min(start + chunk_size, n_left)
        block = left[start:stop]
        k1, k0 = arccos_kernel_pair(block, right)
        inner = block @ right.T
        result[start:stop] = weight_scale**2 * k1 + inner * k0
    return result


def arccos_kernel_pair(left: Array, right: Array) -> tuple[Array, Array]:
    r"""Return the degree-1 and degree-0 arc-cosine kernels between point sets.

    Given ``left`` of shape ``(N1, d)`` and ``right`` of shape ``(N2, d)``,
    returns ``(k1, k0)`` with

    .. math::
        k_1(a,a') = \frac{\|a\|\|a'\|}{2\pi}
                    \bigl(\sin\theta+(\pi-\theta)\cos\theta\bigr)
        = \mathbb{E}\bigl[\sigma(\langle g,a\rangle)\sigma(\langle g,a'\rangle)\bigr],
        \qquad
        k_0(a,a') = \frac{\pi-\theta}{2\pi}
        = \mathbb{E}\bigl[\sigma'(\langle g,a\rangle)\sigma'(\langle g,a'\rangle)\bigr]

    for :math:`g\sim\mathcal{N}(0,I)` and ReLU :math:`\sigma`.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    norms_left = np.linalg.norm(left, axis=1)
    norms_right = np.linalg.norm(right, axis=1)
    inner = left @ right.T
    denom = np.outer(norms_left, norms_right)
    safe = denom > 0.0
    cosine = np.zeros_like(inner)
    np.divide(inner, denom, out=cosine, where=safe)
    np.clip(cosine, -1.0, 1.0, out=cosine)
    theta = np.arccos(cosine)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
    k1 = denom * (sin_theta + (np.pi - theta) * cosine) / (2.0 * np.pi)
    k0 = (np.pi - theta) / (2.0 * np.pi)
    # A zero vector has no direction; both kernels vanish against it.
    k1 = np.where(safe, k1, 0.0)
    return k1, k0


class OperatorValuedKernel(ABC):
    """A kernel :math:`K:\\mathcal{U}\\times\\mathcal{U}\\to\\mathcal{L}(\\mathcal{V})`."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension :math:`d_v` of :math:`\\mathcal{V}`."""

    @abstractmethod
    def block_gram(self, inputs: Array, other: Array | None = None) -> Array:
        """Return the block Gram matrix of shape ``(n1*d_v, n2*d_v)``."""

    def blocks(self, inputs: Array, other: Array | None = None) -> Array:
        """Return the Gram matrix reshaped to ``(n1, n2, d_v, d_v)``."""
        gram = self.block_gram(inputs, other)
        d_v = self.output_dim
        n1 = gram.shape[0] // d_v
        n2 = gram.shape[1] // d_v
        return gram.reshape(n1, d_v, n2, d_v).transpose(0, 2, 1, 3)


class MercerOperatorKernel(OperatorValuedKernel):
    r"""Non-separable operator-valued kernel given by a Mercer expansion.

    The kernel is

    .. math::
        K(u,\tilde u) = \sum_{j=1}^{J} \mu_j\, e_j(u)e_j(\tilde u)\; T_j,
        \qquad T_j = R_j\,\mathrm{diag}(\nu)\,R_j^\top,

    with :math:`\{e_j\}` orthonormal in :math:`L^2(\mathcal{U},\rho_\mathcal{U})`
    and each :math:`R_j` an orthogonal matrix on :math:`\mathcal{V}`.  Because
    the :math:`R_j` differ across :math:`j`, the kernel is *not* of the
    separable form :math:`k(u,\tilde u)T`: the output covariance rotates with the
    input mode, so the problem does not decouple into :math:`d_v` scalar
    regressions.

    The eigen-decomposition of the induced integral operator :math:`\mathcal{L}`
    on :math:`L^2(\mathcal{U},\rho_\mathcal{U};\mathcal{V})` is nonetheless
    explicit: taking :math:`F = e_j\,g_{j,k}` with :math:`g_{j,k}` the
    :math:`k`-th column of :math:`R_j` gives :math:`\mathcal{L}F = \mu_j\nu_k F`.
    So the spectrum is exactly the product set
    :math:`\{\mu_j\nu_k\}`, which is what allows Assumptions 3.2 and 3.3 to be
    imposed with known exponents.

    The Gram matrix is assembled through the exact finite-rank factorization
    :math:`K(u,\tilde u) = \Xi(u)\Xi(\tilde u)^\top` with
    :math:`\Xi(u)[a,(j,c)] = \sqrt{\mu_j}\,e_j(u)\,L_j[a,c]` and
    :math:`L_j = T_j^{1/2}`, which costs :math:`O(n^2 d_v^2 J)` through a single
    matrix product rather than a loop over modes.

    Parameters
    ----------
    mode_weights:
        The input-mode eigenvalues :math:`\mu_j`, shape ``(J,)``.
    output_weights:
        The output eigenvalues :math:`\nu_k`, shape ``(d_v,)``.
    rotations:
        The orthogonal matrices :math:`R_j`, shape ``(J, d_v, d_v)``.
    basis:
        Callable ``inputs -> (n, J)`` evaluating :math:`e_j` on the inputs.
    """

    def __init__(
        self,
        mode_weights: Array,
        output_weights: Array,
        rotations: Array,
        basis: object,
    ) -> None:
        self.mode_weights = np.asarray(mode_weights, dtype=float)
        self.output_weights = np.asarray(output_weights, dtype=float)
        self.rotations = np.asarray(rotations, dtype=float)
        if not callable(basis):
            raise TypeError("basis must be callable as inputs -> (n, J)")
        self.basis = basis
        n_modes = self.mode_weights.size
        d_v = self.output_weights.size
        if self.rotations.shape != (n_modes, d_v, d_v):
            raise ValueError(
                f"rotations must have shape {(n_modes, d_v, d_v)}, got {self.rotations.shape}"
            )
        # L_j = R_j diag(sqrt(nu)) R_j^T is the symmetric square root of T_j.
        root = self.rotations * np.sqrt(self.output_weights)[None, None, :]
        self._roots = root @ self.rotations.transpose(0, 2, 1)

    @property
    def output_dim(self) -> int:
        return int(self.output_weights.size)

    def factor(self, inputs: Array) -> Array:
        """Return the exact factorization :math:`\\Xi`, shape ``(n*d_v, J*d_v)``."""
        values = np.asarray(self.basis(inputs), dtype=float)
        scaled = values * np.sqrt(self.mode_weights)[None, :]
        tensor = np.einsum("ij,jac->iajc", scaled, self._roots, optimize=True)
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)

    def block_gram(self, inputs: Array, other: Array | None = None) -> Array:
        left = self.factor(inputs)
        right = left if other is None else self.factor(other)
        return left @ right.T


class ScalarNTKKernel(OperatorValuedKernel):
    r"""Closed-form NTK of a two-layer network with ReLU activation.

    This is the :math:`M\to\infty` limit of :class:`kerop.features.ScalarNTKFeatures`.
    """

    def __init__(self, input_dim: int, weight_scale: float = 1.0, include_bias: bool = True):
        self.input_dim = int(input_dim)
        self.weight_scale = float(weight_scale)
        self.include_bias = bool(include_bias)

    @property
    def output_dim(self) -> int:
        return 1

    def lift(self, inputs: Array) -> Array:
        """Return :math:`J(u)`, i.e. the inputs with a bias coordinate."""
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        if not self.include_bias:
            return inputs
        return np.hstack([inputs, np.ones((inputs.shape[0], 1))])

    def block_gram(self, inputs: Array, other: Array | None = None) -> Array:
        left = self.lift(inputs)
        right = left if other is None else self.lift(other)
        k1, k0 = arccos_kernel_pair(left, right)
        return self.weight_scale**2 * k1 + (left @ right.T) * k0


class OperatorNTKKernel(OperatorValuedKernel):
    r"""Closed-form operator-valued NTK of a shallow neural operator.

    The :math:`M\to\infty` limit of :class:`kerop.features.OperatorNTKFeatures`.
    Inputs are the lifted arrays :math:`J(u_i)(x_k)` of shape
    ``(n, n_x, d_tilde)``; flattening the sample and collocation axes turns the
    block Gram matrix into an arc-cosine kernel evaluated on the resulting point
    cloud, with the block layout falling out automatically.

    Parameters
    ----------
    feature_dim:
        The per-point dimension :math:`\tilde d`.
    n_points:
        Number of collocation points, i.e. :math:`d_v`.
    weight_scale:
        Standard deviation of the initialization.
    output_scale:
        Scaling applied to the features; the kernel picks up its square.  Used
        to absorb the isometric :math:`1/\sqrt{n_\mathcal{X}}` rescaling of the
        :math:`\mathcal{V}` inner product.
    """

    def __init__(
        self,
        feature_dim: int,
        n_points: int,
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
        chunk_size: int = 2048,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.n_points = int(n_points)
        self.weight_scale = float(weight_scale)
        self.output_scale = float(output_scale)
        self.chunk_size = int(chunk_size)

    @property
    def output_dim(self) -> int:
        return self.n_points

    def _flatten(self, lifted: Array) -> Array:
        lifted = np.asarray(lifted, dtype=float)
        if lifted.ndim != 3 or lifted.shape[2] != self.feature_dim:
            raise ValueError(
                f"expected lifted inputs of shape (n, n_x, {self.feature_dim}), "
                f"got {lifted.shape}"
            )
        if lifted.shape[1] != self.n_points:
            raise ValueError(
                f"expected {self.n_points} collocation points, got {lifted.shape[1]}"
            )
        return lifted.reshape(-1, self.feature_dim)

    def block_gram(self, inputs: Array, other: Array | None = None) -> Array:
        left = self._flatten(inputs)
        right = left if other is None else self._flatten(other)
        gram = _relu_ntk_gram(left, right, self.weight_scale, self.chunk_size)
        gram *= self.output_scale**2
        return gram


class SeparableGaussianKernel(OperatorValuedKernel):
    r"""Separable kernel :math:`K(u,\tilde u) = k(u,\tilde u)\,T` with Gaussian
    :math:`k`.

    The :math:`M\to\infty` limit of :class:`kerop.features.SeparableRFF`.
    """

    def __init__(self, bandwidth: float, output_covariance: Array) -> None:
        self.bandwidth = float(bandwidth)
        self.output_covariance = np.asarray(output_covariance, dtype=float)

    @property
    def output_dim(self) -> int:
        return int(self.output_covariance.shape[0])

    def scalar_gram(self, inputs: Array, other: Array | None = None) -> Array:
        """Return the scalar Gaussian Gram matrix :math:`k(u_i,\\tilde u_j)`."""
        left = np.atleast_2d(np.asarray(inputs, dtype=float))
        right = left if other is None else np.atleast_2d(np.asarray(other, dtype=float))
        sq = (
            (left**2).sum(axis=1)[:, None]
            + (right**2).sum(axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        return np.exp(-np.maximum(sq, 0.0) / (2.0 * self.bandwidth**2))

    def block_gram(self, inputs: Array, other: Array | None = None) -> Array:
        return np.kron(self.scalar_gram(inputs, other), self.output_covariance)