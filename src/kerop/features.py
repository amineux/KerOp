r"""Vector-valued random feature maps.

Assumption 2.1 of arXiv:2603.00971 asks that the operator-valued kernel
:math:`K:\mathcal{U}\times\mathcal{U}\to\mathcal{L}(\mathcal{V})` admit the
integral representation

.. math::

    K(u,\tilde u) = \sum_{i=1}^p \int_\Omega
        \varphi_i(u,\omega)\otimes\varphi_i(\tilde u,\omega)\,d\pi(\omega),
    \qquad
    \sum_{i=1}^p \|\varphi_i(u,\omega)\|_{\mathcal{V}}^2 \le \kappa^2,

with :math:`\mathcal{V}`-valued feature maps :math:`\varphi_i`.  Drawing
:math:`\omega_1,\dots,\omega_M\sim\pi` gives the random feature kernel

.. math::

    K_M(u,\tilde u) = \sum_{i=1}^p \frac{1}{M}\sum_{m=1}^M
        \varphi_i(u,\omega_m)\otimes\varphi_i(\tilde u,\omega_m).

The sum over :math:`i` is what lets the framework cover operator-valued neural
tangent kernels, whose feature expansion carries one block for the activation
and one per input coordinate of the derivative term; this is the source of the
factor :math:`p` in the feature requirement of Theorem 3.4.

Everything in this module is expressed through a single object, the *feature
tensor*

.. math::
    \Psi_M(u)c = \frac{1}{\sqrt M}\sum_{i,m} c_{i,m}\,\varphi_i(u,\omega_m),
    \qquad \Psi_M(u)\in\mathcal{L}(\mathbb{R}^{pM},\mathcal{V}),

which satisfies :math:`\Psi_M(u)\Psi_M(\tilde u)^* = K_M(u,\tilde u)` and turns
the random feature hypothesis space :math:`\mathcal{H}_M` into
:math:`\mathbb{R}^{pM}`.  The empirical operators of the paper are then

.. math::
    \widehat\Sigma_M = \frac1n\sum_j \Psi_M(u_j)^*\Psi_M(u_j), \qquad
    \widehat{\mathcal{S}}_M^*\mathbf v = \frac1n\sum_j \Psi_M(u_j)^* v_j.

Conventions
-----------
Outputs live in :math:`\mathcal{V}=\mathbb{R}^{d_v}` with the *standard*
Euclidean inner product.  Datasets that discretize a function space are
responsible for supplying outputs in isometric coordinates, i.e. rescaled so
that the Euclidean norm equals the intended :math:`L^2(\mathcal{X},\rho_x)`
norm; see :func:`kerop.data.isometric_scale`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "RandomFeatureMap",
    "MercerFeatures",
    "ScalarNTKFeatures",
    "OperatorNTKFeatures",
    "SeparableRFF",
    "relu",
    "relu_derivative",
    "erf_activation",
    "erf_derivative",
    "ACTIVATIONS",
]


def relu(z: Array) -> Array:
    """ReLU activation :math:`\\sigma(z)=\\max(z,0)`."""
    return np.maximum(z, 0.0)


def relu_derivative(z: Array) -> Array:
    """Derivative of ReLU, the Heaviside step."""
    return (z > 0.0).astype(float)


def erf_activation(z: Array) -> Array:
    """Smooth activation :math:`\\sigma(z)=\\mathrm{erf}(z/\\sqrt2)`."""
    from scipy.special import erf

    return np.asarray(erf(z / np.sqrt(2.0)), dtype=float)


def erf_derivative(z: Array) -> Array:
    """Derivative of :func:`erf_activation`."""
    return np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z**2)


#: Activations usable by the NTK feature maps, as ``(sigma, sigma')`` pairs.
ACTIVATIONS: dict[str, tuple[object, object]] = {
    "relu": (relu, relu_derivative),
    "erf": (erf_activation, erf_derivative),
}


class RandomFeatureMap(ABC):
    """A finite sample :math:`\\{\\omega_m\\}_{m=1}^M` from a representation (2.5).

    Subclasses hold the drawn randomness, so a given instance defines one fixed
    realization of :math:`K_M` and hence one fixed hypothesis space
    :math:`\\mathcal{H}_M\\cong\\mathbb{R}^{pM}`.
    """

    @property
    @abstractmethod
    def n_summands(self) -> int:
        """The number :math:`p` of summands in the representation (2.5)."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """The number :math:`M` of random features drawn."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """The dimension :math:`d_v` of :math:`\\mathcal{V}`."""

    @property
    def coefficient_dim(self) -> int:
        """Dimension :math:`pM` of the random feature coefficient space."""
        return self.n_summands * self.n_features

    @abstractmethod
    def feature_tensor(self, inputs: Array) -> Array:
        """Return the feature tensor of shape ``(n, d_v, p*M)``.

        Entry ``[j, :, (i, m)]`` holds :math:`\\varphi_i(u_j,\\omega_m)/\\sqrt M`,
        so that slice ``[j]`` is the matrix of :math:`\\Psi_M(u_j)`.
        """

    def design_matrix(self, inputs: Array) -> Array:
        """Return the feature tensor flattened to ``(n*d_v, p*M)``.

        This is the design matrix of the equivalent linear least-squares
        problem: with it, :math:`\\widehat\\Sigma_M = Z^\\top Z/n` and
        :math:`\\widehat{\\mathcal{S}}_M^*\\mathbf v = Z^\\top\\mathrm{vec}
        (\\mathbf v)/n`.  The vector-valued structure is retained through the
        rows: every sample contributes :math:`d_v` of them, all sharing the same
        :math:`pM` coefficients.
        """
        tensor = self.feature_tensor(inputs)
        return tensor.reshape(-1, tensor.shape[-1])

    def rf_kernel(self, inputs: Array, other: Array | None = None) -> Array:
        """Evaluate :math:`K_M` on pairs, returning shape ``(n1, n2, d_v, d_v)``."""
        left = self.feature_tensor(inputs)
        right = left if other is None else self.feature_tensor(other)
        return np.einsum("iam,jbm->ijab", left, right, optimize=True)

    def empirical_kappa_squared(self, inputs: Array) -> float:
        """Largest observed value of :math:`\\sum_i\\|\\varphi_i(u,\\omega)\\|^2`.

        Assumption 2.1 requires this to be bounded :math:`\\pi`-almost surely.
        For feature maps built from a uniformly bounded orthonormal basis the
        bound is available in closed form; for the NTK maps, whose weights are
        Gaussian, boundedness holds only with high probability, so this
        empirical maximum is the honest diagnostic.
        """
        tensor = self.feature_tensor(inputs)
        # Undo the 1/sqrt(M) normalization and sum the p blocks per feature.
        per_feature = self.n_features * (tensor**2).sum(axis=1)
        blocks = per_feature.reshape(per_feature.shape[0], self.n_summands, self.n_features)
        return float(blocks.sum(axis=1).max())


class MercerFeatures(RandomFeatureMap):
    r"""Importance-sampled Mercer features for a diagonalizable kernel.

    Suppose the kernel is given through a Mercer-type expansion

    .. math::
        K(u,\tilde u) = \sum_{i=1}^{S} \sigma_i\,
            \Phi_i(u)\otimes\Phi_i(\tilde u),
        \qquad \Phi_i:\mathcal{U}\to\mathcal{V},

    with :math:`\sigma_i>0` summable and the :math:`\Phi_i` orthonormal in
    :math:`L^2(\mathcal{U},\rho_{\mathcal{U}};\mathcal{V})`.  Taking
    :math:`\Omega=\{1,\dots,S\}` with :math:`\pi(i)=\sigma_i/Z`,
    :math:`Z=\sum_i\sigma_i`, and

    .. math::
        \varphi(u,i) := \sqrt{Z}\,\Phi_i(u)

    reproduces the representation (2.5) exactly with :math:`p=1`, since
    :math:`\sum_i \pi(i)\,Z\,\Phi_i(u)\otimes\Phi_i(\tilde u) = K(u,\tilde u)`.
    The bound (2.6) holds with :math:`\kappa^2 = Z\sup_{u,i}\|\Phi_i(u)\|^2`.

    Because :math:`\{\Phi_i\}` diagonalizes the kernel integral operator
    :math:`\mathcal{L}` with eigenvalues :math:`\sigma_i`, this construction is
    the one instance where the source condition (Assumption 3.2) and the
    effective dimension (Assumption 3.3) are known *exactly* rather than
    estimated, which is what makes it usable to test the rate of Theorem 3.4.

    Parameters
    ----------
    eigenvalues:
        The weights :math:`\sigma_i`, shape ``(S,)``.
    basis:
        Callable ``(inputs, indices) -> (n, d_v, len(indices))`` evaluating the
        selected :math:`\Phi_i` on the inputs.  Taking indices rather than
        returning all :math:`S` functions keeps the cost proportional to
        :math:`M`, not to the truncation level.
    output_dim:
        Dimension of :math:`\mathcal{V}`.
    n_features:
        Number :math:`M` of features to draw.
    rng:
        Source of randomness for the importance sampling.
    """

    def __init__(
        self,
        eigenvalues: Array,
        basis: object,
        output_dim: int,
        n_features: int,
        rng: np.random.Generator,
    ) -> None:
        self._eigenvalues = np.asarray(eigenvalues, dtype=float)
        if np.any(self._eigenvalues <= 0.0):
            raise ValueError("Mercer eigenvalues must be strictly positive")
        if not callable(basis):
            raise TypeError("basis must be callable as (inputs, indices) -> tensor")
        self._basis = basis
        self._output_dim = int(output_dim)
        self._n_features = int(n_features)
        self.trace = float(self._eigenvalues.sum())
        probabilities = self._eigenvalues / self.trace
        self.indices = rng.choice(
            self._eigenvalues.size, size=self._n_features, replace=True, p=probabilities
        )

    @property
    def n_summands(self) -> int:
        return 1

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def feature_tensor(self, inputs: Array) -> Array:
        basis_values = self._basis(inputs, self.indices)
        scale = np.sqrt(self.trace / self._n_features)
        return scale * np.asarray(basis_values, dtype=float)


class ScalarNTKFeatures(RandomFeatureMap):
    r"""Random features of the neural tangent kernel of a two-layer network.

    This is the real-valued (:math:`d_v=1`) specialization used for the paper's
    numerical illustration in Appendix A.3.  With
    :math:`J(u) = (u, 1)\in\mathbb{R}^{\tilde d}`, :math:`\tilde d = d+1`, and
    weights :math:`b^{(0)}_m\sim\pi_0`, the NTK feature blocks are

    .. math::
        \psi_m(u) = \sigma\bigl(\langle b^{(0)}_m, J(u)\rangle\bigr),
        \qquad
        \psi'_{m,j}(u) = \sigma'\bigl(\langle b^{(0)}_m,
            J(u)\rangle\bigr)\,J(u)^{(j)},

    giving :math:`p = 1 + \tilde d = d + 2` summands: one activation block plus
    one derivative block per feature coordinate.  This is exactly the
    :math:`p = d+2` quoted in Appendix A.3, where the paper reports that
    :math:`M = O(\sqrt{n}\,p)` features suffice.

    Parameters
    ----------
    input_dim:
        Dimension :math:`d` of the inputs.
    n_features:
        Number :math:`M` of neurons / random features.
    rng:
        Source of randomness for the initialization :math:`b^{(0)}`.
    activation:
        Key into :data:`ACTIVATIONS`.
    weight_scale:
        Standard deviation of the Gaussian initialization :math:`\pi_0`.
    include_bias:
        Whether to append the constant coordinate to :math:`J(u)`, which is what
        makes :math:`\tilde d = d+1` and hence :math:`p = d+2`.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        rng: np.random.Generator,
        activation: str = "relu",
        weight_scale: float = 1.0,
        include_bias: bool = True,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise KeyError(f"unknown activation {activation!r}; have {sorted(ACTIVATIONS)}")
        self.input_dim = int(input_dim)
        self.include_bias = bool(include_bias)
        self.feature_dim = self.input_dim + (1 if self.include_bias else 0)
        self._n_features = int(n_features)
        self.activation = activation
        self.weight_scale = float(weight_scale)
        self.weights = weight_scale * rng.standard_normal((self._n_features, self.feature_dim))

    @property
    def n_summands(self) -> int:
        return 1 + self.feature_dim

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def output_dim(self) -> int:
        return 1

    def lift(self, inputs: Array) -> Array:
        """Return :math:`J(u)`, the inputs with the bias coordinate appended."""
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"expected inputs with {self.input_dim} columns, got {inputs.shape}")
        if not self.include_bias:
            return inputs
        return np.hstack([inputs, np.ones((inputs.shape[0], 1))])

    def feature_tensor(self, inputs: Array) -> Array:
        lifted = self.lift(inputs)
        sigma, sigma_prime = ACTIVATIONS[self.activation]
        pre = lifted @ self.weights.T  # (n, M)
        activated = sigma(pre)
        derivative = sigma_prime(pre)
        # Block 0 is psi_m; blocks 1..d_tilde are psi'_{m,j}, stacked along the
        # feature axis so that the coefficient vector is indexed by (i, m).
        blocks = [activated] + [derivative * lifted[:, [j]] for j in range(self.feature_dim)]
        stacked = np.concatenate(blocks, axis=1)  # (n, p*M)
        return stacked[:, None, :] / np.sqrt(self._n_features)


class OperatorNTKFeatures(RandomFeatureMap):
    r"""Operator-valued NTK random features of a shallow neural operator.

    For the shallow neural operators of Section 2.1, the NTK feature map is
    :math:`\Phi^M_u(v) = \nabla_\theta\langle G_{\theta_0}(u),
    v\rangle_{L^2(\rho_x)}`, and the induced vector-valued kernel expands as

    .. math::
        K_M(u,\tilde u) = \frac1M\sum_m \psi_m(u)\otimes\psi_m(\tilde u)
            + \frac1M\sum_m\sum_{j=1}^{\tilde d}
              \psi'_{m,j}(u)\otimes\psi'_{m,j}(\tilde u),

    where, with :math:`J(u)(x) = (A(u)(x),\,u(x),\,c(x))^\top\in
    \mathbb{R}^{\tilde d}`,

    .. math::
        \psi_m(u) = \sigma\bigl(\langle b^{(0)}_m, J(u)\rangle\bigr),
        \qquad
        \psi'_{m,j}(u) = \sigma'\bigl(\langle b^{(0)}_m,
            J(u)\rangle\bigr)\,J(u)^{(j)}.

    The crucial difference from :class:`ScalarNTKFeatures` is that
    :math:`J(u)(\cdot)` is a *function of* :math:`x`, so each
    :math:`\psi_m(u)` is an element of :math:`\mathcal{V}` rather than a scalar:
    these are genuinely vector-valued random features, with :math:`p = 1 +
    \tilde d` summands and :math:`\tilde d = d_k + d_y + d_b`.

    Inputs are supplied already lifted, as an array of shape
    ``(n, n_x, d_tilde)`` holding :math:`J(u_i)(x_k)`; see
    :meth:`kerop.data.pde.OperatorDataset.lift` for how the lifting operator
    :math:`A`, the input trace :math:`u(x)` and the positional encoding
    :math:`c(x)` are assembled.

    Parameters
    ----------
    feature_dim:
        The per-point feature dimension :math:`\tilde d`.
    n_points:
        Number of collocation points :math:`n_{\mathcal{X}}`, i.e. :math:`d_v`.
    n_features:
        Network width / number of random features :math:`M`.
    rng:
        Source of randomness for :math:`b^{(0)}`.
    activation, weight_scale:
        As in :class:`ScalarNTKFeatures`.
    output_scale:
        Multiplies the features, letting a dataset absorb the isometric
        rescaling :math:`1/\sqrt{n_{\mathcal{X}}}` of the :math:`\mathcal{V}`
        inner product into the feature map.
    """

    def __init__(
        self,
        feature_dim: int,
        n_points: int,
        n_features: int,
        rng: np.random.Generator,
        activation: str = "relu",
        weight_scale: float = 1.0,
        output_scale: float = 1.0,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise KeyError(f"unknown activation {activation!r}; have {sorted(ACTIVATIONS)}")
        self.feature_dim = int(feature_dim)
        self.n_points = int(n_points)
        self._n_features = int(n_features)
        self.activation = activation
        self.weight_scale = float(weight_scale)
        self.output_scale = float(output_scale)
        self.weights = weight_scale * rng.standard_normal((self._n_features, self.feature_dim))

    @property
    def n_summands(self) -> int:
        return 1 + self.feature_dim

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def output_dim(self) -> int:
        return self.n_points

    def feature_tensor(self, inputs: Array) -> Array:
        lifted = np.asarray(inputs, dtype=float)
        if lifted.ndim != 3 or lifted.shape[2] != self.feature_dim:
            raise ValueError(
                f"expected lifted inputs of shape (n, n_x, {self.feature_dim}), "
                f"got {lifted.shape}"
            )
        sigma, sigma_prime = ACTIVATIONS[self.activation]
        pre = lifted @ self.weights.T  # (n, n_x, M)
        activated = sigma(pre)
        derivative = sigma_prime(pre)
        blocks = [activated] + [
            derivative * lifted[:, :, [j]] for j in range(self.feature_dim)
        ]
        stacked = np.concatenate(blocks, axis=2)  # (n, n_x, p*M)
        return stacked * (self.output_scale / np.sqrt(self._n_features))


class SeparableRFF(RandomFeatureMap):
    r"""Random Fourier features for a separable operator-valued kernel.

    The kernel is :math:`K(u,\tilde u) = k(u,\tilde u)\,T` with :math:`k` a
    shift-invariant scalar kernel and :math:`T\in\mathcal{L}(\mathcal{V})`
    positive semi-definite.  Bochner's theorem gives
    :math:`k(u,\tilde u) = \int \cos(\langle\omega,u-\tilde u\rangle)d\pi(\omega)`,
    and with :math:`\omega=(w,\tau)`, :math:`\tau\sim\mathrm{Unif}[0,2\pi]`, the
    :math:`\mathcal{V}`-valued feature

    .. math::
        \varphi(u,\omega) = \sqrt{2}\,
            \cos\bigl(\langle w,u\rangle + \tau\bigr)\, T^{1/2}e_{k(\omega)}
        \cdot \sqrt{d_v}

    with :math:`k(\omega)` drawn uniformly from :math:`\{1,\dots,d_v\}`
    reproduces (2.5) with :math:`p=1`.  Included as the classical
    vector-valued random feature baseline (Brault et al., 2016; Minh, 2016)
    against which the spectral filters can be compared on PDE tasks.

    Parameters
    ----------
    input_dim:
        Dimension of the (flattened) inputs.
    n_features:
        Number of features :math:`M`.
    bandwidth:
        Length scale :math:`\ell` of the Gaussian kernel
        :math:`k(u,\tilde u)=\exp(-\|u-\tilde u\|^2/(2\ell^2))`.
    output_root:
        A square root :math:`T^{1/2}` of the output covariance, shape
        ``(d_v, d_v)``.  Defaults to the identity.
    rng:
        Source of randomness.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        bandwidth: float,
        rng: np.random.Generator,
        output_root: Array | None = None,
        output_dim: int | None = None,
    ) -> None:
        if output_root is None:
            if output_dim is None:
                raise ValueError("provide either output_root or output_dim")
            self.output_root = np.eye(int(output_dim))
        else:
            self.output_root = np.asarray(output_root, dtype=float)
        self.input_dim = int(input_dim)
        self._n_features = int(n_features)
        self.bandwidth = float(bandwidth)
        self.frequencies = rng.standard_normal((self._n_features, self.input_dim)) / self.bandwidth
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self._n_features)
        self.output_indices = rng.integers(0, self.output_root.shape[1], size=self._n_features)

    @property
    def n_summands(self) -> int:
        return 1

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def output_dim(self) -> int:
        return int(self.output_root.shape[0])

    def feature_tensor(self, inputs: Array) -> Array:
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        n_out = self.output_root.shape[1]
        cosines = np.sqrt(2.0 * n_out) * np.cos(inputs @ self.frequencies.T + self.phases)
        columns = self.output_root[:, self.output_indices]  # (d_v, M)
        tensor = cosines[:, None, :] * columns[None, :, :]
        return tensor / np.sqrt(self._n_features)