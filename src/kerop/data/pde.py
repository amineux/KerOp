r"""PDE solution operators: the Poisson and Darcy maps.

These are the operator-learning tasks: inputs are discretized source or
coefficient fields, outputs are discretized solutions, and the object being
learned is the solution operator :math:`G_\rho` of a boundary value problem.
Unlike :mod:`kerop.data.spectral`, the exponents :math:`(r,b)` are not known
here, so these datasets are used for the wall-clock comparison against exact
operator-valued kernel regression and for the command-line demos.

Both problems are posed on :math:`\mathcal{X}=[0,1]` with homogeneous Dirichlet
conditions, and both solution operators are available to machine precision, so
the excess risk is measured against the true operator rather than against noisy
labels.

Poisson
    :math:`-u'' = f`, :math:`u(0)=u(1)=0`.  Writing the source in the Dirichlet
    sine basis :math:`f = \sum_k a_k\sqrt2\sin(k\pi x)` gives
    :math:`u = \sum_k a_k(k\pi)^{-2}\sqrt2\sin(k\pi x)`, so the operator is
    *linear* and exact in closed form.

Darcy
    :math:`-(a u')' = f`, :math:`u(0)=u(1)=0`, learning the *nonlinear* map
    :math:`a\mapsto u` at fixed :math:`f`.  In one dimension this integrates
    twice:

    .. math::
        u'(x) = \frac{C - F(x)}{a(x)},\quad F(x)=\int_0^x f,
        \qquad
        C = \frac{\int_0^1 F/a}{\int_0^1 1/a},

    the constant being fixed by :math:`u(1)=0`.  Evaluating the two integrals by
    the trapezoidal rule on the grid gives the solution without a linear solve.

The lifting operator
--------------------
The shallow neural operators of Section 2.1 act on

.. math::
    J(u)(x) = \bigl(A(u)(x),\ u(x),\ c(x)\bigr)^\top \in \mathbb{R}^{\tilde d},
    \qquad \tilde d = d_k + d_y + d_b,

where :math:`A:\mathcal{U}\to\mathcal{F}(\mathcal{X},\mathbb{R}^{d_k})` is a
continuous operator, :math:`u(x)` is the pointwise input trace and :math:`c(x)`
a positional encoding.  Here :math:`A` is a bank of :math:`d_k` nonlocal
smoothing operators,

.. math::
    A(u)(x)_i = \int_0^1 \frac{1}{Z_i}
        \exp\!\Bigl(-\frac{(x-y)^2}{2\ell_i^2}\Bigr)u(y)\,dy,

with length scales :math:`\ell_i` spread over the domain.  This is what makes
the resulting kernel genuinely operator-valued: the value of a feature at
:math:`x` depends on the whole input function, not just on :math:`u(x)`.  Since
:math:`p = 1+\tilde d`, the choice of :math:`d_k` and :math:`d_b` directly sets
the feature requirement of Theorem 3.4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kerop.data import isometric_scale

Array = NDArray[np.float64]

__all__ = [
    "OperatorSamples",
    "OperatorDataset",
    "PoissonDataset",
    "DarcyDataset",
    "DATASETS",
    "make_dataset",
]


@dataclass(frozen=True)
class OperatorSamples:
    """One batch of input/output function pairs.

    Attributes
    ----------
    fields:
        The input functions sampled on the grid, shape ``(n, n_x)``, in physical
        units.
    outputs:
        The solutions, shape ``(n, n_x)``, in *isometric* coordinates: already
        multiplied by :math:`1/\\sqrt{n_\\mathcal{X}}` so that Euclidean norms
        equal empirical :math:`L^2(\\mathcal{X})` norms.
    targets:
        The noiseless solutions in the same isometric coordinates, i.e.
        :math:`G_\\rho(u)`.  Equal to ``outputs`` when the noise level is zero.
    """

    fields: Array
    outputs: Array
    targets: Array


class OperatorDataset(ABC):
    """A solution operator sampled on a fixed grid.

    Parameters
    ----------
    n_points:
        Number of collocation points :math:`n_\\mathcal{X}`, which is also the
        output dimension :math:`d_v`.
    n_modes:
        Number of Karhunen-Loeve modes in the input random field.
    field_decay:
        Decay exponent of the mode amplitudes; larger means smoother inputs.
    n_lift:
        Number :math:`d_k` of nonlocal smoothing channels in :math:`A`.
    noise_std:
        Standard deviation of the additive observation noise on the solution, in
        physical units.
    """

    def __init__(
        self,
        n_points: int = 33,
        n_modes: int = 12,
        field_decay: float = 2.0,
        n_lift: int = 3,
        noise_std: float = 0.0,
    ) -> None:
        if n_points < 3:
            raise ValueError(f"need at least three grid points, got {n_points}")
        self.n_points = int(n_points)
        self.n_modes = int(n_modes)
        self.field_decay = float(field_decay)
        self.n_lift = int(n_lift)
        self.noise_std = float(noise_std)
        self.grid = np.linspace(0.0, 1.0, self.n_points)
        self.mode_amplitudes = np.arange(1, self.n_modes + 1, dtype=float) ** (-field_decay)
        self._smoothers = self._build_smoothers()
        self._encoding = self._build_encoding()

    # ------------------------------------------------------------------ #
    # Lifting operator
    # ------------------------------------------------------------------ #

    def _build_smoothers(self) -> Array:
        """Return the :math:`d_k` nonlocal smoothing matrices, shape ``(d_k, n_x, n_x)``."""
        if self.n_lift == 0:
            return np.zeros((0, self.n_points, self.n_points))
        scales = np.logspace(-1.5, -0.5, self.n_lift)
        separation = self.grid[:, None] - self.grid[None, :]
        kernels = np.exp(-(separation[None, :, :] ** 2) / (2.0 * scales[:, None, None] ** 2))
        # Row-normalize so each channel is an averaging operator, keeping the
        # lifted features on the same scale as the input itself.
        return kernels / kernels.sum(axis=2, keepdims=True)

    def _build_encoding(self) -> Array:
        """Return the positional encoding :math:`c(x)`, shape ``(n_x, d_b)``."""
        return np.stack(
            [
                self.grid,
                np.sin(np.pi * self.grid),
                np.cos(np.pi * self.grid),
                np.ones_like(self.grid),
            ],
            axis=1,
        )

    @property
    def encoding_dim(self) -> int:
        """The dimension :math:`d_b` of the positional encoding."""
        return int(self._encoding.shape[1])

    @property
    def feature_dim(self) -> int:
        """The per-point feature dimension :math:`\\tilde d = d_k + d_y + d_b`."""
        return self.n_lift + 1 + self.encoding_dim

    @property
    def n_summands(self) -> int:
        """The number :math:`p = 1 + \\tilde d` of summands in the NTK representation."""
        return 1 + self.feature_dim

    def lift(self, fields: Array) -> Array:
        """Return :math:`J(u)(x)`, shape ``(n, n_x, d_tilde)``.

        The channels are ordered as :math:`(A(u)(x),\\,u(x),\\,c(x))` to match
        the definition in Section 2.1.
        """
        fields = np.atleast_2d(np.asarray(fields, dtype=float))
        if fields.shape[1] != self.n_points:
            raise ValueError(
                f"expected fields with {self.n_points} columns, got {fields.shape}"
            )
        n = fields.shape[0]
        channels = [np.einsum("kxy,iy->ixk", self._smoothers, fields, optimize=True)]
        channels.append(fields[:, :, None])
        channels.append(np.broadcast_to(self._encoding[None, :, :], (n, self.n_points, self.encoding_dim)))
        return np.concatenate(channels, axis=2)

    def output_scale(self) -> float:
        """Return the isometric scale :math:`1/\\sqrt{n_\\mathcal{X}}`."""
        return isometric_scale(self.n_points)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def random_fields(self, n_samples: int, rng: np.random.Generator) -> Array:
        r"""Draw input fields from a truncated Karhunen-Loeve expansion.

        The coefficients are uniform on :math:`[-\sqrt3,\sqrt3]`, so they have
        unit variance and are *bounded*, which keeps the label distribution
        inside the Bernstein moment condition of Assumption 3.1 and the feature
        bound of Assumption 2.1.
        """
        coefficients = rng.uniform(
            -np.sqrt(3.0), np.sqrt(3.0), size=(n_samples, self.n_modes)
        )
        basis = np.sqrt(2.0) * np.sin(
            np.pi * np.outer(self.grid, np.arange(1, self.n_modes + 1))
        )
        return (coefficients * self.mode_amplitudes) @ basis.T

    @abstractmethod
    def solve(self, fields: Array) -> Array:
        """Apply the solution operator, returning shape ``(n, n_x)`` in physical units."""

    def sample(self, n_samples: int, rng: np.random.Generator) -> OperatorSamples:
        """Draw ``n_samples`` input/solution pairs."""
        fields = self.random_fields(n_samples, rng)
        solutions = self.solve(fields)
        scale = self.output_scale()
        targets = solutions * scale
        if self.noise_std > 0.0:
            noisy = solutions + self.noise_std * rng.standard_normal(solutions.shape)
        else:
            noisy = solutions
        return OperatorSamples(fields=fields, outputs=noisy * scale, targets=targets)


class PoissonDataset(OperatorDataset):
    r"""The Poisson solution operator :math:`f\mapsto u` with :math:`-u''=f`.

    Solved exactly in the Dirichlet sine basis: the operator is diagonal with
    eigenvalues :math:`(k\pi)^{-2}`, so it is linear and smoothing of order two.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        wavenumbers = np.arange(1, self.n_modes + 1, dtype=float)
        self._sine = np.sqrt(2.0) * np.sin(np.pi * np.outer(self.grid, wavenumbers))
        self._solution_eigenvalues = (wavenumbers * np.pi) ** -2.0

    def solve(self, fields: Array) -> Array:
        """Apply :math:`(-\\Delta)^{-1}` by projecting onto the sine basis.

        The projection is exact rather than quadrature-based: the fields are
        generated from the same finite sine expansion, so their coefficients are
        recovered by least squares without discretization error.
        """
        fields = np.atleast_2d(np.asarray(fields, dtype=float))
        coefficients, *_ = np.linalg.lstsq(self._sine, fields.T, rcond=None)
        return (self._sine @ (self._solution_eigenvalues[:, None] * coefficients)).T


class DarcyDataset(OperatorDataset):
    r"""The Darcy solution operator :math:`a\mapsto u` with :math:`-(au')'=f`.

    The coefficient field is :math:`a(x) = 1 + \text{amplitude}\cdot\tanh(w(x))`
    for a Karhunen-Loeve field :math:`w`, which keeps :math:`a` bounded away from
    zero and infinity, so the operator is Lipschitz on the sampled set.  The map
    is nonlinear in :math:`a`.

    Parameters
    ----------
    amplitude:
        Contrast of the coefficient field; must lie in :math:`(0,1)` so that
        :math:`a\in(1-\text{amplitude},\,1+\text{amplitude})`.
    source:
        The fixed right-hand side :math:`f`, evaluated on the grid.  Defaults to
        :math:`f\equiv1`.
    """

    def __init__(self, amplitude: float = 0.6, source: Array | None = None, **kwargs: object):
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if not (0.0 < amplitude < 1.0):
            raise ValueError(f"amplitude must lie in (0, 1), got {amplitude}")
        self.amplitude = float(amplitude)
        self.source = (
            np.ones_like(self.grid) if source is None else np.asarray(source, dtype=float)
        )
        if self.source.shape != self.grid.shape:
            raise ValueError(f"source must have shape {self.grid.shape}")
        # F(x) = int_0^x f, by cumulative trapezoid on the grid.
        spacing = self.grid[1] - self.grid[0]
        increments = 0.5 * spacing * (self.source[1:] + self.source[:-1])
        self._source_integral = np.concatenate([[0.0], np.cumsum(increments)])

    def coefficient(self, fields: Array) -> Array:
        """Map a Karhunen-Loeve field to the positive coefficient :math:`a(x)`."""
        return 1.0 + self.amplitude * np.tanh(np.asarray(fields, dtype=float))

    def solve(self, fields: Array) -> Array:
        """Integrate the one-dimensional Darcy problem twice."""
        coefficient = self.coefficient(np.atleast_2d(np.asarray(fields, dtype=float)))
        spacing = self.grid[1] - self.grid[0]
        reciprocal = 1.0 / coefficient  # (n, n_x)
        integral = self._source_integral[None, :]

        def trapezoid(values: Array) -> Array:
            return 0.5 * spacing * (values[:, 1:] + values[:, :-1]).sum(axis=1)

        constant = trapezoid(integral * reciprocal) / trapezoid(reciprocal)
        derivative = (constant[:, None] - integral) * reciprocal
        increments = 0.5 * spacing * (derivative[:, 1:] + derivative[:, :-1])
        return np.concatenate(
            [np.zeros((coefficient.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1
        )


#: The PDE datasets available from the command line.
DATASETS: dict[str, type[OperatorDataset]] = {
    "poisson": PoissonDataset,
    "darcy": DarcyDataset,
}


def make_dataset(name: str, **kwargs: object) -> OperatorDataset:
    """Construct the dataset registered under ``name``."""
    try:
        cls = DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}") from None
    return cls(**kwargs)  # type: ignore[arg-type]