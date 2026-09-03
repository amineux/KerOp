r"""A synthetic instance with prescribed source and capacity exponents.

Theorem 3.4 predicts the excess risk :math:`n^{-r/(2r+b)}` where :math:`r` comes
from the source condition (Assumption 3.2) and :math:`b` from the effective
dimension (Assumption 3.3).  Testing that prediction requires an instance where
:math:`r` and :math:`b` are *known*, which in turn requires control over the
spectrum of the kernel integral operator :math:`\mathcal{L}`.  This module
builds such an instance by specifying that spectrum directly.

Construction
------------
Let :math:`\mathcal{U}=[0,1]^d` with :math:`\rho_\mathcal{U}` uniform, and let
:math:`\{e_j\}_{j=1}^J` be the tensorized cosine basis

.. math::
    e_\alpha(x) = \prod_{l=1}^{d} c_{\alpha_l}\cos(\pi\alpha_l x_l),
    \qquad c_0 = 1,\quad c_k = \sqrt2\ (k\ge1),

which is orthonormal in :math:`L^2([0,1]^d)` and uniformly bounded by
:math:`2^{d/2}`.  Fix output weights :math:`\nu_1,\dots,\nu_{d_v}>0` summing to
one and Haar-random orthogonal matrices :math:`R_j` on :math:`\mathcal{V}`, and
set

.. math::
    K(u,\tilde u) = \sum_{j=1}^J \mu_j\,e_j(u)e_j(\tilde u)\,T_j,
    \qquad T_j = R_j\,\mathrm{diag}(\nu)\,R_j^\top,
    \qquad \mu_j = j^{-1/b}.

Writing :math:`g_{j,k}` for the :math:`k`-th column of :math:`R_j`, one checks
directly that :math:`\mathcal{L}(e_j g_{j,k}) = \mu_j\nu_k\,e_j g_{j,k}`, so the
spectrum of :math:`\mathcal{L}` is the product set :math:`\{\mu_j\nu_k\}` and

* **Assumption 3.3 holds with exponent** :math:`b`.  The counting function is
  :math:`\#\{(j,k):\mu_j\nu_k>\epsilon\} = \epsilon^{-b}\sum_k\nu_k^{b}`, so the
  sorted eigenvalues decay as :math:`i^{-1/b}` and
  :math:`\mathcal{N}(\lambda)\asymp\lambda^{-b}`, with the constant depending on
  :math:`\nu` but the exponent not.
* **Assumption 3.2 holds with exponent** :math:`r`, by defining the target
  through :math:`G_\rho = \mathcal{L}^r H` for an explicit
  :math:`H=\sum_i h_i\Phi_i\in L^2`.

Because the :math:`R_j` differ across modes, :math:`T_j\ne T_{j'}` and the
kernel is genuinely non-separable: it is *not* of the form :math:`k(u,\tilde
u)T`, so the problem does not reduce to :math:`d_v` independent scalar
regressions.  (Flat output weights would make :math:`T_j=d_v^{-1}I` and destroy
this, which is why :math:`\nu` is chosen non-constant.)

Two honest caveats, both checkable with the diagnostics below:

* The expansion is truncated at :math:`J` modes, so :math:`\mathcal{L}` has
  finite rank :math:`Jd_v` and :math:`\mathcal{N}(\lambda)` flattens once
  :math:`\lambda` drops below the smallest eigenvalue.  :math:`J` must be large
  enough that the whole range of :math:`\lambda_n` used stays above it;
  :meth:`SpectralOperatorModel.effective_dimension_fit` measures the realized
  exponent so this can be verified rather than assumed.
* The coefficients :math:`h_i` sit exactly at the boundary of
  square-summability, :math:`h_i\propto i^{-1/2}`.  This matters more than it
  looks.  With :math:`\sigma_i\asymp i^{-1/b}` and :math:`h_i\propto
  i^{-(1/2+\epsilon)}`, the exact bias of the spectral cut-off behaves as

  .. math::
      \|r_\lambda(\mathcal{L})G_\rho\|^2
        \;\approx \sum_{i>i_\lambda} \sigma_i^{2r}h_i^2
        \;\asymp\; \lambda^{2r+2\epsilon b},
      \qquad i_\lambda \asymp \lambda^{-b},

  so any extra decay :math:`\epsilon>0` raises the realized source exponent to
  :math:`r+\epsilon b` and would make the measured rate disagree with the
  nominal one.  Choosing :math:`\epsilon=0` gives a clean :math:`\lambda^{r}`
  bias; :math:`H` is still a legitimate :math:`L^2` element because the
  expansion is truncated, with :math:`\|H\|^2\asymp\log(Jd_v)`.  Damping
  :math:`h_i` by a logarithm instead - the textbook way to sit just inside
  :math:`\ell^2` in infinite dimensions - would introduce a
  :math:`1/\log(1/\lambda)` factor and inflate a power-law fit of the bias by
  roughly :math:`+0.2` over three decades of :math:`\lambda`.
  :meth:`SpectralOperatorModel.source_exponent_fit` measures the realized
  exponent, so this is checked rather than assumed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from kerop.features import MercerFeatures
from kerop.filters import SpectralFilter, make_filter
from kerop.kernels import MercerOperatorKernel
from kerop.metrics import RateFit, fit_power_law

Array = NDArray[np.float64]

__all__ = ["SpectralOperatorModel", "cosine_multi_indices"]


def cosine_multi_indices(n_modes: int, input_dim: int) -> Array:
    """Return the first ``n_modes`` multi-indices ordered by total degree.

    Ties within a degree are broken lexicographically.  Ordering by total degree
    is what makes the assignment :math:`\\mu_j = j^{-1/b}` a statement about the
    *rank* of a mode, so the resulting eigenvalue decay exponent is exactly
    :math:`1/b` regardless of the input dimension.
    """
    if n_modes <= 0:
        raise ValueError(f"n_modes must be positive, got {n_modes}")
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    collected: list[tuple[int, ...]] = []
    degree = 0
    while len(collected) < n_modes:
        for candidate in itertools.product(range(degree + 1), repeat=input_dim):
            if sum(candidate) == degree:
                collected.append(candidate)
        degree += 1
    return np.array(sorted(collected[:n_modes], key=lambda a: (sum(a), a)), dtype=int)


@dataclass
class SpectralOperatorModel:
    r"""Synthetic operator-valued instance with known :math:`(r,b)`.

    Parameters
    ----------
    r:
        Source-condition exponent of Assumption 3.2.  :math:`r=1/2` is the
        well-specified case :math:`G_\rho\in\mathcal{H}`, :math:`r<1/2` the
        misspecified one.
    b:
        Effective-dimension exponent of Assumption 3.3, in :math:`(0,1]`.
        Eigenvalues decay as :math:`i^{-1/b}`; smaller :math:`b` means faster
        decay and faster rates.
    n_modes:
        Truncation level :math:`J` of the input-mode expansion.
    output_dim:
        Dimension :math:`d_v` of :math:`\mathcal{V}`.
    input_dim:
        Dimension :math:`d` of the latent input space :math:`[0,1]^d`.  The
        rates of Theorem 3.4 are dimension-free in :math:`\mathcal{U}`, which
        this parameter exists to test.
    output_decay:
        Exponent :math:`\gamma` in :math:`\nu_k\propto k^{-\gamma}`.  Must be
        non-zero for the kernel to be non-separable.
    source_radius:
        The radius :math:`R` bounding :math:`\|H\|_{L^2(\rho_\mathcal{U})}`.
    source_tail:
        Extra decay :math:`\epsilon` in :math:`h_i\propto i^{-(1/2+\epsilon)}`,
        beyond the :math:`i^{-1/2}` that makes :math:`H` sit exactly at the
        boundary of square-summability.  The default :math:`\epsilon=0` is the
        right choice: it is what makes the bias a clean :math:`\lambda^r` power
        law.  Positive values add smoothness the nominal exponent does not
        account for, shifting the realized exponent to :math:`r+\epsilon b`.
    noise_std:
        Standard deviation of the additive label noise, per output coordinate.
    seed:
        Seed for the model's own randomness (rotations, target coefficients).
        Sampling uses separately supplied generators, so the model is fixed
        while the data varies.
    """

    r: float = 0.5
    b: float = 1.0
    n_modes: int = 512
    output_dim: int = 8
    input_dim: int = 1
    output_decay: float = 1.0
    source_radius: float = 1.0
    source_tail: float = 0.0
    noise_std: float = 0.05
    seed: int = 0

    multi_indices: Array = field(init=False, repr=False)
    mode_weights: Array = field(init=False, repr=False)
    output_weights: Array = field(init=False, repr=False)
    rotations: Array = field(init=False, repr=False)
    eigenvalues: Array = field(init=False, repr=False)
    source_coefficients: Array = field(init=False, repr=False)
    target_coefficients: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.b <= 1.0):
            raise ValueError(f"b must lie in (0, 1], got {self.b}")
        if self.r <= 0.0:
            raise ValueError(f"r must be positive, got {self.r}")
        if 2.0 * self.r + self.b <= 1.0:
            raise ValueError(
                f"the easy-learning condition 2r + b > 1 fails for r={self.r}, b={self.b}"
            )
        rng = np.random.default_rng(self.seed)

        self.multi_indices = cosine_multi_indices(self.n_modes, self.input_dim)
        ranks = np.arange(1, self.n_modes + 1, dtype=float)
        self.mode_weights = ranks ** (-1.0 / self.b)

        output_ranks = np.arange(1, self.output_dim + 1, dtype=float)
        weights = output_ranks ** (-self.output_decay)
        self.output_weights = weights / weights.sum()

        self.rotations = self._haar_orthogonal(rng, self.output_dim, self.n_modes)

        # Spectrum of L, as the product set {mu_j nu_k}, laid out (J, d_v).
        self.eigenvalues = np.outer(self.mode_weights, self.output_weights)

        # Source coefficients h_i, indexed by decreasing eigenvalue so that the
        # decay is a statement about spectral rank.
        flat = self.eigenvalues.reshape(-1)
        order = np.argsort(flat)[::-1]
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(1, flat.size + 1)
        rank_grid = rank_of.reshape(self.eigenvalues.shape).astype(float)
        magnitudes = rank_grid ** -(0.5 + self.source_tail)
        signs = rng.choice([-1.0, 1.0], size=self.eigenvalues.shape)
        raw = signs * magnitudes
        self.source_coefficients = self.source_radius * raw / np.linalg.norm(raw)

        # G_rho = L^r H, i.e. coefficient sigma_i^r h_i in the eigenbasis.
        self.target_coefficients = self.eigenvalues**self.r * self.source_coefficients

    @staticmethod
    def _haar_orthogonal(rng: np.random.Generator, dim: int, count: int) -> Array:
        """Draw ``count`` Haar-distributed orthogonal matrices of size ``dim``.

        A QR decomposition of a Gaussian matrix gives an orthogonal factor whose
        distribution is Haar only after the sign ambiguity is fixed, which is
        what the multiplication by the signs of ``diag(R)`` does.  The
        decomposition is batched over modes; looping instead costs a few minutes
        at the truncation levels used here.
        """
        gaussian = rng.standard_normal((count, dim, dim))
        q, upper = np.linalg.qr(gaussian)
        diagonals = np.einsum("...ii->...i", upper)
        return q * np.sign(diagonals)[:, None, :]

    # ------------------------------------------------------------------ #
    # Basis, kernel, features
    # ------------------------------------------------------------------ #

    def basis(self, inputs: Array) -> Array:
        """Evaluate the cosine basis, returning shape ``(n, J)``."""
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"expected inputs with {self.input_dim} columns, got {inputs.shape}"
            )
        # prod_l c_{alpha_l} cos(pi alpha_l x_l), computed as a product over
        # coordinates of an (n, J) factor each.
        values = np.ones((inputs.shape[0], self.n_modes), dtype=float)
        for axis in range(self.input_dim):
            orders = self.multi_indices[:, axis].astype(float)
            normalizers = np.where(orders > 0, np.sqrt(2.0), 1.0)
            values *= normalizers[None, :] * np.cos(
                np.pi * np.outer(inputs[:, axis], orders)
            )
        return values

    def basis_bound(self) -> float:
        """Return :math:`\\sup_{u,j}|e_j(u)|`, needed for the constant in (2.6)."""
        support = (self.multi_indices > 0).sum(axis=1).max()
        return float(np.sqrt(2.0) ** support)

    def kappa_squared(self) -> float:
        """Return the constant :math:`\\kappa^2` of Assumption 2.1 for these features.

        With :math:`\\varphi(u,i)=\\sqrt Z\\Phi_i(u)` and
        :math:`\\Phi_i(u)=e_{j}(u)g_{j,k}`, one has
        :math:`\\|\\varphi(u,i)\\|^2 = Z\\,e_j(u)^2 \\le Z\\sup_{u,j}e_j(u)^2`.
        """
        return float(self.eigenvalues.sum()) * self.basis_bound() ** 2

    def kernel(self) -> MercerOperatorKernel:
        """Return the exact operator-valued kernel of this model."""
        return MercerOperatorKernel(
            mode_weights=self.mode_weights,
            output_weights=self.output_weights,
            rotations=self.rotations,
            basis=self.basis,
        )

    def features(self, n_features: int, rng: np.random.Generator) -> MercerFeatures:
        """Draw ``n_features`` importance-sampled Mercer random features.

        Feature :math:`i=(j,k)` is drawn with probability
        :math:`\\mu_j\\nu_k/Z` and equals :math:`\\sqrt{Z}\\,e_j(u)g_{j,k}`, so
        the resulting :math:`K_M` is an unbiased Monte Carlo estimate of
        :math:`K` in the sense of (2.5) with :math:`p=1`.
        """
        return MercerFeatures(
            eigenvalues=self.eigenvalues.reshape(-1),
            basis=self._indexed_basis,
            output_dim=self.output_dim,
            n_features=n_features,
            rng=rng,
        )

    def _indexed_basis(self, inputs: Array, indices: Array) -> Array:
        """Evaluate :math:`\\Phi_i` for flattened indices ``i``, shape ``(n, d_v, m)``."""
        indices = np.asarray(indices, dtype=int)
        mode_index, output_index = np.unravel_index(indices, self.eigenvalues.shape)
        basis_values = self.basis(inputs)[:, mode_index]  # (n, m)
        directions = self.rotations[mode_index, :, output_index]  # (m, d_v)
        return np.einsum("im,ma->iam", basis_values, directions, optimize=True)

    # ------------------------------------------------------------------ #
    # Target operator and sampling
    # ------------------------------------------------------------------ #

    def regression_operator(self, inputs: Array) -> Array:
        r"""Evaluate :math:`G_\rho`, returning shape ``(n, d_v)``.

        Since :math:`G_\rho = \sum_{j,k}\sigma_{jk}^r h_{jk}\,e_j\,g_{j,k}`, the
        sum over :math:`k` can be absorbed into per-mode vectors
        :math:`w_j = R_j c_j`, leaving a single matrix product.
        """
        weights = np.einsum("jak,jk->ja", self.rotations, self.target_coefficients)
        return self.basis(inputs) @ weights

    def target_norm(self) -> float:
        """Return :math:`\\|G_\\rho\\|_{L^2(\\rho_\\mathcal{U})}`."""
        return float(np.linalg.norm(self.target_coefficients))

    def sample(self, n_samples: int, rng: np.random.Generator) -> tuple[Array, Array]:
        """Draw ``n_samples`` pairs :math:`(u,v)` with :math:`v = G_\\rho(u)+\\varepsilon`.

        The noise is Gaussian, which satisfies the Bernstein-type moment
        condition of Assumption 3.1.
        """
        inputs = rng.uniform(0.0, 1.0, size=(n_samples, self.input_dim))
        outputs = self.regression_operator(inputs)
        if self.noise_std > 0.0:
            outputs = outputs + self.noise_std * rng.standard_normal(outputs.shape)
        return inputs, outputs

    def test_set(self, n_samples: int, rng: np.random.Generator) -> tuple[Array, Array]:
        """Draw a noiseless test set, i.e. inputs paired with :math:`G_\\rho(u)`."""
        inputs = rng.uniform(0.0, 1.0, size=(n_samples, self.input_dim))
        return inputs, self.regression_operator(inputs)

    # ------------------------------------------------------------------ #
    # Exact diagnostics for Assumptions 3.2 and 3.3
    # ------------------------------------------------------------------ #

    def effective_dimension(self, lam: float | Array) -> float | Array:
        r"""Return :math:`\mathcal{N}(\lambda)=\mathrm{tr}(\mathcal{L}
        (\mathcal{L}+\lambda)^{-1})` exactly.

        Because the spectrum is known, this is a finite sum rather than an
        estimate.
        """
        spectrum = self.eigenvalues.reshape(-1)
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
        values = (spectrum[None, :] / (spectrum[None, :] + lam_arr[:, None])).sum(axis=1)
        return float(values[0]) if np.ndim(lam) == 0 else values

    def usable_lambda_window(self, margin: float = 30.0) -> tuple[float, float]:
        r"""Return the range of :math:`\lambda` over which the power laws hold.

        The expansion is truncated at :math:`J` modes, so :math:`\mathcal{L}`
        has finite rank and :math:`\mathcal{N}(\lambda)` flattens out once
        :math:`\lambda` falls below the smallest eigenvalue.  Conversely, for
        :math:`\lambda` above the largest eigenvalue there is nothing left to
        resolve.  This returns
        :math:`[\,\mathrm{margin}\cdot\sigma_{\min},\;\sigma_{\max}/\mathrm{margin}\,]`,
        the interval in which both :math:`\mathcal{N}(\lambda)\asymp\lambda^{-b}`
        and :math:`\mathrm{bias}\asymp\lambda^{r}` can be expected to be
        accurate, and which any experiment should keep :math:`\lambda_n` inside.
        """
        spectrum = self.eigenvalues.reshape(-1)
        low = margin * float(spectrum.min())
        high = float(spectrum.max()) / margin
        if low >= high:
            raise ValueError(
                f"truncation too small for margin={margin}: the spectrum spans "
                f"[{spectrum.min():.3g}, {spectrum.max():.3g}]; increase n_modes"
            )
        return low, high

    def effective_dimension_fit(self, lambdas: Array) -> RateFit:
        r"""Measure the realized exponent :math:`b` from :math:`\mathcal{N}(\lambda)`.

        Assumption 3.3 posits :math:`\mathcal{N}(\lambda)\le c_b\lambda^{-b}`;
        fitting :math:`\log\mathcal{N}` against :math:`\log\lambda` returns a
        slope of :math:`-b`.  Comparing it with the nominal :math:`b` verifies
        that the truncation level :math:`J` is large enough for the range of
        :math:`\lambda` supplied.
        """
        lambdas = np.asarray(lambdas, dtype=float)
        values = np.atleast_1d(self.effective_dimension(lambdas))
        return fit_power_law(lambdas, values)

    def exact_bias(self, filt: SpectralFilter, spectral_scale: float | None = None) -> float:
        r"""Return the exact bias :math:`\|r_\lambda(\mathcal{L})G_\rho\|_{L^2}`.

        This is the approximation error of the *population* estimator, which
        Theorem 3.4's proof controls by :math:`\lambda^r` through the
        qualification of the filter.  Diagonality gives it in closed form,

        .. math::
            \|r_\lambda(\mathcal{L})G_\rho\|^2
              = \sum_i r_\lambda(\sigma_i)^2\,\sigma_i^{2r}\,h_i^2 .
        """
        spectrum = self.eigenvalues.reshape(-1)
        scale = float(np.max(spectrum)) if spectral_scale is None else float(spectral_scale)
        residual = filt.residual_function(spectrum / scale)
        coefficients = self.target_coefficients.reshape(-1)
        return float(np.sqrt(np.sum((residual * coefficients) ** 2)))

    def source_exponent_fit(
        self, lambdas: Array, filter_name: str = "cutoff", **filter_kwargs: object
    ) -> RateFit:
        r"""Measure the realized exponent :math:`r` from the decay of the bias.

        Under Assumption 3.2 with exponent :math:`r`, and for a filter of
        qualification at least :math:`r`, the bias satisfies
        :math:`\|r_\lambda(\mathcal{L})G_\rho\|\asymp\lambda^{r}`.  The slope of
        the fit is therefore an estimate of :math:`r` that uses no sampling at
        all.  The spectral cut-off is the default probe because its
        qualification is infinite and its constants are exactly one, so the
        measurement is not contaminated by the filter's own saturation.
        """
        lambdas = np.asarray(lambdas, dtype=float)
        scale = float(np.max(self.eigenvalues))
        biases = np.array(
            [
                self.exact_bias(make_filter(filter_name, float(lam), **filter_kwargs), scale)
                for lam in lambdas
            ]
        )
        return fit_power_law(lambdas, biases)