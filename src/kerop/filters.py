r"""Spectral regularization filters.

This module implements families of *regularization functions* in the sense of
Definition 2.2 of Nguyen & Mucke, "Random Features for Operator-Valued Kernels"
(arXiv:2603.00971).  A family :math:`\{\phi_\lambda\}_{\lambda\in(0,1]}` of
functions :math:`\phi_\lambda:[0,1]\to\mathbb{R}` is a family of regularization
functions if there are constants :math:`D, E, c_0 > 0` with

.. math::

    \sup_{0<t\le1}|t\phi_\lambda(t)| \le D, \qquad
    \sup_{0<t\le1}|\phi_\lambda(t)| \le E/\lambda, \qquad
    \sup_{0<t\le1}|r_\lambda(t)| \le c_0,

where :math:`r_\lambda(t) := 1 - t\phi_\lambda(t)` is the *residual*.  The
*qualification* of the family is the largest :math:`\nu>0` such that

.. math::

    \sup_{0<t\le1} |r_\lambda(t)|\, t^q \le c_q \lambda^q
    \qquad \text{for all } q\in[0,\nu],\ \lambda\in(0,1].

Theorem 3.4 of the paper requires qualification :math:`\nu \ge r\vee 1`, so the
choice of filter constrains which source-condition exponents :math:`r` are
reachable.  Tikhonov regularization has qualification exactly :math:`1` and
therefore saturates for :math:`r>1`; gradient descent (Landweber) has infinite
qualification; Brakhage's :math:`\nu`-method has qualification :math:`\nu` while
requiring only :math:`O(\sqrt{t})` iterations, which is the acceleration
mechanism referenced in the paper's introduction (Pagliana & Rosasco, 2019).

All filters here act on operators whose spectrum has been normalized to
:math:`[0,1]`; see :mod:`kerop.estimators` for the rescaling.  Every filter
exposes both

* :meth:`SpectralFilter.apply`, the algorithm actually used to compute
  :math:`\phi_\lambda(A)b`, and
* :meth:`SpectralFilter.filter_function`, the scalar function
  :math:`t\mapsto\phi_\lambda(t)` used for the spectral analysis,

and the test suite checks that the two agree on diagonal operators.  For the
iterative filters both are produced by the *same* recursion, so consistency is
structural rather than a coincidence of two derivations.

References
----------
Gerfo, Rosasco, Odone, De Vito & Verri (2008), "Spectral algorithms for
supervised learning"; Engl, Hanke & Neubauer (1996), "Regularization of inverse
problems", chapter 6; Bauer, Pereverzev & Rosasco (2007).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, eigh

Array = NDArray[np.float64]
MatVec = Callable[[Array], Array]

__all__ = [
    "FilterDiagnostics",
    "QualificationReport",
    "SpectralFilter",
    "Tikhonov",
    "IteratedTikhonov",
    "Landweber",
    "SpectralCutoff",
    "HeavyBall",
    "NuMethod",
    "FILTER_REGISTRY",
    "make_filter",
    "filter_diagnostics",
    "measure_qualification",
]


@dataclass(frozen=True)
class FilterDiagnostics:
    """Numerically measured constants of Definition 2.2 for a filter family.

    Attributes
    ----------
    D:
        Measured :math:`\\sup_{\\lambda}\\sup_t |t\\phi_\\lambda(t)|`, cf. (2.7).
    E:
        Measured :math:`\\sup_{\\lambda}\\sup_t \\lambda|\\phi_\\lambda(t)|`, cf. (2.8).
    c0:
        Measured :math:`\\sup_{\\lambda}\\sup_t |r_\\lambda(t)|`, cf. (2.9).
    """

    D: float
    E: float
    c0: float


class SpectralFilter(ABC):
    """A single member :math:`\\phi_\\lambda` of a regularization family.

    Concrete subclasses are constructed either directly (with their natural
    hyper-parameter, e.g. a number of iterations) or through
    :meth:`from_lambda`, which converts a target regularization level
    :math:`\\lambda` into that natural hyper-parameter.

    Parameters
    ----------
    lam:
        Regularization level :math:`\\lambda\\in(0,1]` in *normalized* spectral
        units, i.e. assuming the operator spectrum lies in :math:`[0,1]`.
    """

    name: str = "filter"
    #: Theoretical qualification :math:`\nu` from the literature; ``inf`` when
    #: the filter has arbitrary qualification.
    qualification: float = math.inf

    def __init__(self, lam: float) -> None:
        if not (lam > 0.0):
            raise ValueError(f"lambda must be positive, got {lam!r}")
        self._lam = float(lam)

    @property
    def lam(self) -> float:
        """Regularization level :math:`\\lambda` in normalized spectral units."""
        return self._lam

    @classmethod
    @abstractmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> SpectralFilter:
        """Build the family member realizing regularization level ``lam``."""

    @abstractmethod
    def apply(self, operator: Array | MatVec, rhs: Array, *, dim: int | None = None) -> Array:
        """Return :math:`\\phi_\\lambda(A)\\,b`.

        Parameters
        ----------
        operator:
            Either a dense symmetric positive semi-definite matrix ``A``, or a
            callable implementing ``x -> A @ x``.  Direct filters
            (:class:`Tikhonov`, :class:`IteratedTikhonov`,
            :class:`SpectralCutoff`) require the dense form.
        rhs:
            Right-hand side ``b``, of shape ``(d,)`` or ``(d, k)``.
        dim:
            Dimension of the operator; only needed when ``operator`` is a
            callable and cannot be inferred from ``rhs``.
        """

    @abstractmethod
    def filter_function(self, t: Array) -> Array:
        """Evaluate :math:`\\phi_\\lambda` pointwise on ``t``."""

    def residual_function(self, t: Array) -> Array:
        """Evaluate the residual :math:`r_\\lambda(t) = 1 - t\\phi_\\lambda(t)`.

        This generic implementation subtracts two nearly equal quantities
        wherever the residual is small, which is precisely the regime the
        qualification condition (2.10) probes: dividing a rounding error of size
        ``eps`` by :math:`\\lambda^q` with :math:`\\lambda=10^{-7}` and
        :math:`q=3` manufactures a spurious constant of order :math:`10^5`.
        Every filter shipped here therefore overrides this method with a
        cancellation-free closed form, and the test suite checks the two against
        each other at moderate :math:`\\lambda`.
        """
        t = np.asarray(t, dtype=float)
        return 1.0 - t * self.filter_function(t)

    @property
    def matvec_count(self) -> int:
        """Number of operator applications used by :meth:`apply`.

        Used for cost accounting; direct solves report ``0`` since their cost is
        a factorization rather than a sequence of matrix-vector products.
        """
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lam={self.lam:.6g})"


def _as_matvec(operator: Array | MatVec) -> MatVec:
    if callable(operator):
        return operator
    arr = np.asarray(operator, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-d operator, got shape {arr.shape}")
    return lambda x: arr @ x


def _require_dense(operator: Array | MatVec, who: str) -> Array:
    if callable(operator):
        raise TypeError(f"{who} requires a dense operator matrix, not a matvec callable")
    arr = np.asarray(operator, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{who} requires a square matrix, got shape {arr.shape}")
    return arr


# --------------------------------------------------------------------------- #
# Direct (explicitly regularized) filters
# --------------------------------------------------------------------------- #


class Tikhonov(SpectralFilter):
    r"""Tikhonov regularization / kernel ridge regression.

    :math:`\phi_\lambda(t) = (t+\lambda)^{-1}`, residual
    :math:`r_\lambda(t) = \lambda/(t+\lambda)`.  Constants of Definition 2.2 are
    :math:`D = E = c_0 = 1`; the qualification is exactly :math:`1` with
    :math:`c_1 = 1`, so Theorem 3.4 applies only up to smoothness :math:`r=1`.
    """

    name = "tikhonov"
    qualification = 1.0

    @classmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> Tikhonov:
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        return cls(lam)

    def apply(self, operator: Array | MatVec, rhs: Array, *, dim: int | None = None) -> Array:
        A = _require_dense(operator, "Tikhonov")
        reg = A + self.lam * np.eye(A.shape[0])
        factor = cho_factor(reg, lower=True, check_finite=False)
        return np.asarray(cho_solve(factor, rhs, check_finite=False), dtype=float)

    def filter_function(self, t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        return 1.0 / (t + self.lam)

    def residual_function(self, t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        return self.lam / (t + self.lam)


class IteratedTikhonov(SpectralFilter):
    r"""Iterated Tikhonov regularization of order ``order``.

    The residual is :math:`r_\lambda(t) = (\lambda/(t+\lambda))^{m}` for
    :math:`m=` ``order``, hence

    .. math::
        \phi_\lambda(t) = \frac{(t+\lambda)^m - \lambda^m}{t\,(t+\lambda)^m},

    extended continuously by :math:`\phi_\lambda(0) = m/\lambda`.  The
    qualification is exactly :math:`m`, which lets one reach source-condition
    exponents :math:`r\le m` with an explicitly regularized method.
    """

    name = "iterated_tikhonov"

    def __init__(self, lam: float, order: int = 2) -> None:
        super().__init__(lam)
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.order = int(order)
        self.qualification = float(self.order)

    @classmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> IteratedTikhonov:
        return cls(lam, order=int(kwargs.pop("order", 2)))

    def apply(self, operator: Array | MatVec, rhs: Array, *, dim: int | None = None) -> Array:
        A = _require_dense(operator, "IteratedTikhonov")
        reg = A + self.lam * np.eye(A.shape[0])
        factor = cho_factor(reg, lower=True, check_finite=False)
        x = np.zeros_like(np.asarray(rhs, dtype=float))
        for _ in range(self.order):
            x = np.asarray(cho_solve(factor, rhs + self.lam * x, check_finite=False), dtype=float)
        return x

    def filter_function(self, t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        lam, m = self.lam, self.order
        # 1 - (1 + t/lam)^-m via expm1/log1p, accurate when t << lam/m and the
        # difference would otherwise cancel to the last few bits.
        numerator = -np.expm1(-m * np.log1p(t / lam))
        out = np.empty_like(t)
        positive = t > 0.0
        np.divide(numerator, t, out=out, where=positive)
        # Removable singularity at t = 0 resolved by l'Hopital.
        out[~positive] = m / lam
        return out

    def residual_function(self, t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.order * np.log1p(t / self.lam))

    def __repr__(self) -> str:
        return f"IteratedTikhonov(lam={self.lam:.6g}, order={self.order})"


class SpectralCutoff(SpectralFilter):
    r"""Spectral cut-off (truncated SVD).

    :math:`\phi_\lambda(t) = t^{-1}\mathbf{1}\{t\ge\lambda\}` with residual
    :math:`r_\lambda(t) = \mathbf{1}\{t<\lambda\}`.  Constants are
    :math:`D=E=c_0=1` and the qualification is infinite with :math:`c_q=1`,
    since :math:`|r_\lambda(t)|t^q \le \lambda^q`.
    """

    name = "cutoff"
    qualification = math.inf

    @classmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> SpectralCutoff:
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        return cls(lam)

    def apply(self, operator: Array | MatVec, rhs: Array, *, dim: int | None = None) -> Array:
        A = _require_dense(operator, "SpectralCutoff")
        evals, evecs = eigh(A)
        inv = np.where(evals >= self.lam, 1.0 / np.where(evals > 0, evals, 1.0), 0.0)
        b = np.asarray(rhs, dtype=float)
        if b.ndim == 1:
            return evecs @ (inv * (evecs.T @ b))
        return evecs @ (inv[:, None] * (evecs.T @ b))

    def filter_function(self, t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        return np.where(t >= self.lam, 1.0 / np.where(t > 0, t, 1.0), 0.0)

    def residual_function(self, t: Array) -> Array:
        t = np.asarray(t, dtype=float)
        return (t < self.lam).astype(float)


# --------------------------------------------------------------------------- #
# Iterative (implicitly regularized) filters
# --------------------------------------------------------------------------- #


class _IterativeFilter(SpectralFilter):
    """Base class for filters defined by a fixed number of linear recursions.

    Subclasses implement :meth:`_run`, a recursion expressed purely in terms of
    an operator application.  Both :meth:`apply` and :meth:`filter_function`
    dispatch to it, the latter with the diagonal operator ``x -> t * x``, so the
    scalar filter used in the analysis is by construction the one the algorithm
    realizes.
    """

    def __init__(self, lam: float, iterations: int, step: float) -> None:
        super().__init__(lam)
        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations}")
        if not (0.0 < step <= 1.0):
            raise ValueError(f"step must lie in (0, 1] for spectra in [0, 1], got {step}")
        self.iterations = int(iterations)
        self.step = float(step)

    @abstractmethod
    def _run(self, matvec: MatVec, rhs: Array) -> Array:
        """Run the recursion, returning :math:`\\phi_\\lambda(A)b`."""

    def apply(self, operator: Array | MatVec, rhs: Array, *, dim: int | None = None) -> Array:
        return self._run(_as_matvec(operator), np.asarray(rhs, dtype=float))

    def filter_function_recursive(self, t: Array) -> Array:
        """Evaluate :math:`\\phi_\\lambda` by running the recursion itself.

        This is the definitional route: it applies :meth:`_run` to the diagonal
        operator ``x -> t * x``, so it returns exactly the filter the algorithm
        realizes.  Its cost is proportional to the iteration count, which makes
        it impractical for the very small :math:`\\lambda` used in the spectral
        diagnostics; subclasses with a closed form override
        :meth:`filter_function` and are checked against this method in the
        tests.
        """
        t = np.asarray(t, dtype=float)
        return self._run(lambda x: t * x, np.ones_like(t))

    def filter_function(self, t: Array) -> Array:
        return self.filter_function_recursive(t)

    @property
    def matvec_count(self) -> int:
        return self.iterations

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(lam={self.lam:.6g}, "
            f"iterations={self.iterations}, step={self.step:.3g})"
        )


class Landweber(_IterativeFilter):
    r"""Landweber iteration, i.e. gradient descent with early stopping.

    The iteration :math:`x_{k+1} = x_k - \alpha(Ax_k - b)` started at
    :math:`x_0=0` yields after :math:`T` steps

    .. math::
        x_T = \alpha\sum_{k=0}^{T-1}(I-\alpha A)^k b = \phi_\lambda(A) b,
        \qquad r_\lambda(t) = (1-\alpha t)^T,

    and the implicit regularization level is :math:`\lambda = 1/(\alpha T)`.
    The qualification is infinite: :math:`\sup_t (1-\alpha t)^T t^q \le
    (q/e)^q\lambda^q`.  This is the filter for which Theorem 3.4 specializes to
    the neural-operator gradient-descent dynamics of Corollary 3.5.
    """

    name = "landweber"
    qualification = math.inf

    @classmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> Landweber:
        step = float(kwargs.pop("step", 1.0))
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        iterations = max(1, int(math.ceil(1.0 / (step * lam))))
        # Report the lambda actually realized by the integer iteration count.
        return cls(1.0 / (step * iterations), iterations, step)

    @classmethod
    def from_iterations(cls, iterations: int, step: float = 1.0) -> Landweber:
        """Build the filter from a number of gradient steps."""
        iterations = max(1, int(iterations))
        return cls(1.0 / (step * iterations), iterations, step)

    def _run(self, matvec: MatVec, rhs: Array) -> Array:
        x = np.zeros_like(rhs)
        for _ in range(self.iterations):
            x = x - self.step * (matvec(x) - rhs)
        return x

    def filter_function(self, t: Array) -> Array:
        r"""Closed form :math:`\phi_\lambda(t) = (1-(1-\alpha t)^T)/t`.

        Evaluated through ``log1p``/``expm1`` so that the :math:`t\to0` regime,
        where the numerator cancels to :math:`\alpha T t`, keeps full relative
        accuracy.
        """
        t = np.asarray(t, dtype=float)
        alpha, iters = self.step, self.iterations
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = -np.expm1(iters * np.log1p(-alpha * t))
        out = np.empty_like(t)
        positive = t > 0.0
        np.divide(numerator, t, out=out, where=positive)
        out[~positive] = alpha * iters
        return out

    def residual_function(self, t: Array) -> Array:
        r"""Closed form :math:`r_\lambda(t) = (1-\alpha t)^T`."""
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore"):
            return np.exp(self.iterations * np.log1p(-self.step * t))


class HeavyBall(_IterativeFilter):
    r"""Heavy-ball (Polyak) momentum with constant momentum parameter.

    The iteration is

    .. math::
        x_{k+1} = x_k - \alpha(Ax_k - b) + \beta\,(x_k - x_{k-1}),

    whose residual obeys :math:`r_{k+1}(t) = (1-\alpha t+\beta)r_k(t) - \beta
    r_{k-1}(t)`.  With :math:`\beta` held fixed the small-eigenvalue behaviour is
    :math:`r_T(t)\approx(1-\alpha t/(1-\beta))^T`, so the effective
    regularization level is     :math:`\lambda = (1-\beta)/(\alpha T)`: constant
    momentum rescales the step rather than producing the :math:`T^{-2}`
    acceleration, for which the momentum must approach one (see
    :class:`NuMethod`).

    Consistently with that reading, :func:`measure_qualification` finds the
    constants :math:`c_q(\lambda)` flattening onto the Landweber values as
    :math:`\lambda\to0`, so the qualification is unbounded.  The constants are
    however strongly inflated in the large-:math:`\lambda` (few-iteration)
    regime, where the momentum transient has not yet decayed.
    """

    name = "heavy_ball"
    qualification = math.inf

    def __init__(self, lam: float, iterations: int, step: float, momentum: float) -> None:
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must lie in [0, 1), got {momentum}")
        self.momentum = float(momentum)
        super().__init__(lam, iterations, step)

    @classmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> HeavyBall:
        step = float(kwargs.pop("step", 1.0))
        momentum = float(kwargs.pop("momentum", 0.9))
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        iterations = max(1, int(math.ceil((1.0 - momentum) / (step * lam))))
        return cls((1.0 - momentum) / (step * iterations), iterations, step, momentum)

    @classmethod
    def from_iterations(
        cls, iterations: int, step: float = 1.0, momentum: float = 0.9
    ) -> HeavyBall:
        """Build the filter from a number of momentum steps."""
        iterations = max(1, int(iterations))
        return cls((1.0 - momentum) / (step * iterations), iterations, step, momentum)

    def _run(self, matvec: MatVec, rhs: Array) -> Array:
        x = np.zeros_like(rhs)
        x_prev = np.zeros_like(rhs)
        for _ in range(self.iterations):
            nxt = x - self.step * (matvec(x) - rhs) + self.momentum * (x - x_prev)
            x_prev, x = x, nxt
        return x

    def residual_function(self, t: Array) -> Array:
        r"""Closed form for :math:`r_T` via Chebyshev polynomials.

        The residuals obey :math:`r_{k+1} = (1+\beta-\alpha t)r_k - \beta
        r_{k-1}` with :math:`r_0=1`, :math:`r_1=1-\alpha t`.  Substituting
        :math:`r_k = \beta^{k/2}s_k` turns this into the Chebyshev recursion
        :math:`s_{k+1}=2us_k-s_{k-1}` with

        .. math::
            u = \frac{1+\beta-\alpha t}{2\sqrt\beta}, \qquad
            s_1 = \frac{1-\alpha t}{\sqrt\beta},

        so that :math:`r_T = \beta^{T/2}\bigl(T_T(u) + (s_1-u)\,U_{T-1}(u)\bigr)`
        with :math:`T_T`, :math:`U_{T-1}` the Chebyshev polynomials of the first
        and second kind.

        Solving the recursion instead through the roots :math:`z_\pm` of the
        companion polynomial is unstable: the coefficients carry a factor
        :math:`1/(z_+-z_-)`, which blows up near the critically damped point
        :math:`u=1` and cost about eight digits in testing.  The Chebyshev form
        has no such factor.  For :math:`|u|>1` the two kinds are evaluated as
        :math:`\cosh`/:math:`\sinh` and combined with :math:`\beta^{T/2}` in the
        exponent, keeping every intermediate bounded by one.
        """
        t = np.asarray(t, dtype=float)
        alpha, beta, iters = self.step, self.momentum, self.iterations
        if beta == 0.0:
            with np.errstate(divide="ignore"):
                return np.exp(iters * np.log1p(-alpha * t))

        sqrt_beta = math.sqrt(beta)
        log_sqrt_beta = math.log(sqrt_beta)
        u = (1.0 + beta - alpha * t) / (2.0 * sqrt_beta)
        # s_1 - u simplifies to (1 - beta - alpha t) / (2 sqrt(beta)).
        coef = (1.0 - beta - alpha * t) / (2.0 * sqrt_beta)

        out = np.empty_like(t)
        trig = np.abs(u) <= 1.0

        # Deep in the stopped regime the residual is genuinely below the
        # smallest normal float; flushing it to zero is the intended answer.
        with np.errstate(under="ignore"):
            self._residual_branches(t, u, coef, trig, out, log_sqrt_beta)
        return out

    def _residual_branches(
        self,
        t: Array,
        u: Array,
        coef: Array,
        trig: Array,
        out: Array,
        log_sqrt_beta: float,
    ) -> None:
        iters = self.iterations
        if np.any(trig):
            ut = u[trig]
            theta = np.arccos(ut)
            sin_theta = np.sin(theta)
            cheb_t = np.cos(iters * theta)
            cheb_u = np.where(
                np.abs(sin_theta) > 1e-150,
                np.sin(iters * theta) / np.where(np.abs(sin_theta) > 1e-150, sin_theta, 1.0),
                iters * ut ** (iters - 1),
            )
            out[trig] = math.exp(iters * log_sqrt_beta) * (cheb_t + coef[trig] * cheb_u)

        hyp = ~trig
        if np.any(hyp):
            uh = u[hyp]
            sign = np.sign(uh)
            phi = np.arccosh(np.abs(uh))
            # exp(T*(log sqrt(beta) +- phi)) are both <= 1 for a stable
            # iteration, since sqrt(beta)*(|u| + sqrt(u^2-1)) is the modulus of
            # the dominant root.
            hi = np.exp(iters * (log_sqrt_beta + phi))
            lo = np.exp(iters * (log_sqrt_beta - phi))
            sign_t = sign**iters
            cheb_t = 0.5 * sign_t * (hi + lo)
            cheb_u = sign_t * sign * (hi - lo) / (2.0 * np.sinh(phi))
            out[hyp] = cheb_t + coef[hyp] * cheb_u

    def filter_function(self, t: Array) -> Array:
        r"""Return :math:`\phi_\lambda(t) = (1-r_T(t))/t`.

        The value at :math:`t=0` follows from solving :math:`\phi_{k+1} =
        (1+\beta)\phi_k - \beta\phi_{k-1} + \alpha` with :math:`\phi_0=0`,
        :math:`\phi_1=\alpha`, giving :math:`\phi_T(0) = \alpha T/(1-\beta) +
        \alpha\beta(\beta^T-1)/(1-\beta)^2`.
        """
        t = np.asarray(t, dtype=float)
        alpha, beta, iters = self.step, self.momentum, self.iterations
        numerator = 1.0 - self.residual_function(t)
        out = np.empty_like(t)
        positive = t > 0.0
        np.divide(numerator, t, out=out, where=positive)
        out[~positive] = alpha * iters / (1.0 - beta) + alpha * beta * (beta**iters - 1.0) / (
            1.0 - beta
        ) ** 2
        return out

    def __repr__(self) -> str:
        return (
            f"HeavyBall(lam={self.lam:.6g}, iterations={self.iterations}, "
            f"step={self.step:.3g}, momentum={self.momentum:.3g})"
        )


class NuMethod(_IterativeFilter):
    r"""Brakhage's :math:`\nu`-method (Chebyshev semi-iterative acceleration).

    The recursion is

    .. math::
        x_k = x_{k-1} + \mu_k (x_{k-1}-x_{k-2})
              + \omega_k\,\alpha\,(b - A x_{k-1}),

    with the classical coefficients (Engl, Hanke & Neubauer, 1996, §6.3)

    .. math::
        \mu_k = \frac{(k-1)(2k-3)(2k+2\nu-1)}
                      {(k+2\nu-1)(2k+4\nu-1)(2k+2\nu-3)}, \qquad
        \omega_k = 4\,\frac{(2k+2\nu-1)(k+\nu-1)}{(k+2\nu-1)(2k+4\nu-1)}.

    Its residual is a Jacobi polynomial satisfying :math:`\sup_t|r_T(t)|t^q
    \lesssim T^{-2q}` for :math:`q\le\nu`, so the effective regularization level
    after :math:`T` iterations is :math:`\lambda = 1/(\alpha T^2)` and the
    qualification equals :math:`\nu`.  This is the acceleration referred to in
    the paper: the generalization error of :math:`T` gradient steps is matched in
    :math:`O(\sqrt{T})` iterations.
    """

    name = "nu_method"

    def __init__(self, lam: float, iterations: int, step: float, nu: float = 1.0) -> None:
        if nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.nu = float(nu)
        self.qualification = float(nu)
        super().__init__(lam, iterations, step)

    @classmethod
    def from_lambda(cls, lam: float, **kwargs: Any) -> NuMethod:
        step = float(kwargs.pop("step", 1.0))
        nu = float(kwargs.pop("nu", 1.0))
        if kwargs:
            raise TypeError(f"unexpected keyword arguments {sorted(kwargs)}")
        iterations = max(1, int(math.ceil(math.sqrt(1.0 / (step * lam)))))
        return cls(1.0 / (step * iterations**2), iterations, step, nu)

    @classmethod
    def from_iterations(cls, iterations: int, step: float = 1.0, nu: float = 1.0) -> NuMethod:
        """Build the filter from a number of accelerated steps."""
        iterations = max(1, int(iterations))
        return cls(1.0 / (step * iterations**2), iterations, step, nu)

    def _coefficients(self, k: int) -> tuple[float, float]:
        nu = self.nu
        mu = ((k - 1) * (2 * k - 3) * (2 * k + 2 * nu - 1)) / (
            (k + 2 * nu - 1) * (2 * k + 4 * nu - 1) * (2 * k + 2 * nu - 3)
        )
        omega = 4.0 * ((2 * k + 2 * nu - 1) * (k + nu - 1)) / ((k + 2 * nu - 1) * (2 * k + 4 * nu - 1))
        return mu, omega

    def _run(self, matvec: MatVec, rhs: Array) -> Array:
        x = np.zeros_like(rhs)
        x_prev = np.zeros_like(rhs)
        for k in range(1, self.iterations + 1):
            mu, omega = self._coefficients(k)
            nxt = x + mu * (x - x_prev) + omega * self.step * (rhs - matvec(x))
            x_prev, x = x, nxt
        return x

    def residual_function(self, t: Array) -> Array:
        r"""Residuals via their own recursion, avoiding cancellation.

        Substituting :math:`x_k=\phi_k(A)b` into the :math:`\nu`-method
        recursion and using :math:`r_k = 1-t\phi_k(t)` gives

        .. math::
            r_k = (1+\mu_k-\omega_k\alpha t)\,r_{k-1} - \mu_k r_{k-2},
            \qquad r_0 = r_{-1} = 1,

        which propagates the residual directly and so stays accurate where
        :math:`r_k` is many orders of magnitude below one.  The iteration count
        is :math:`O(\lambda^{-1/2})` rather than :math:`O(\lambda^{-1})`, which
        keeps this affordable for the spectral diagnostics.
        """
        t = np.asarray(t, dtype=float)
        r_curr = np.ones_like(t)
        r_prev = np.ones_like(t)
        with np.errstate(under="ignore"):
            for k in range(1, self.iterations + 1):
                mu, omega = self._coefficients(k)
                nxt = (1.0 + mu - omega * self.step * t) * r_curr - mu * r_prev
                r_prev, r_curr = r_curr, nxt
        return r_curr

    def __repr__(self) -> str:
        return (
            f"NuMethod(lam={self.lam:.6g}, iterations={self.iterations}, "
            f"step={self.step:.3g}, nu={self.nu:.3g})"
        )


# --------------------------------------------------------------------------- #
# Registry and diagnostics
# --------------------------------------------------------------------------- #

FILTER_REGISTRY: dict[str, type[SpectralFilter]] = {
    Tikhonov.name: Tikhonov,
    IteratedTikhonov.name: IteratedTikhonov,
    Landweber.name: Landweber,
    SpectralCutoff.name: SpectralCutoff,
    HeavyBall.name: HeavyBall,
    NuMethod.name: NuMethod,
}


def make_filter(name: str, lam: float, **kwargs: Any) -> SpectralFilter:
    """Construct the filter registered under ``name`` at regularization ``lam``.

    Parameters
    ----------
    name:
        One of the keys of :data:`FILTER_REGISTRY`.
    lam:
        Target regularization level in normalized spectral units.
    **kwargs:
        Filter-specific options, e.g. ``step`` for the iterative filters,
        ``order`` for :class:`IteratedTikhonov`, ``nu`` for :class:`NuMethod`.
    """
    try:
        cls = FILTER_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown filter {name!r}; available: {sorted(FILTER_REGISTRY)}") from None
    return cls.from_lambda(lam, **kwargs)


def _spectral_grid(n_points: int = 6001) -> Array:
    """A grid on (0, 1] refined near zero, where the residuals vary fastest."""
    log_part = np.logspace(-12.0, 0.0, n_points // 2, dtype=float)
    lin_part = np.linspace(1e-12, 1.0, n_points - n_points // 2, dtype=float)
    return np.unique(np.concatenate([log_part, lin_part]))


def filter_diagnostics(
    name: str,
    lambdas: Array | None = None,
    *,
    grid: Array | None = None,
    **kwargs: Any,
) -> FilterDiagnostics:
    """Measure the Definition 2.2 constants ``D``, ``E``, ``c0`` for a family.

    The suprema of (2.7)-(2.9) are evaluated numerically on a grid over both
    :math:`t\\in(0,1]` and :math:`\\lambda`.  Because the bounds are required to
    hold uniformly in :math:`\\lambda`, the returned constants are maxima over
    the ``lambdas`` grid.
    """
    t = _spectral_grid() if grid is None else np.asarray(grid, dtype=float)
    lam_grid = np.logspace(-6.0, 0.0, 25) if lambdas is None else np.asarray(lambdas, dtype=float)
    D = E = c0 = 0.0
    for lam in lam_grid:
        flt = make_filter(name, float(lam), **kwargs)
        phi = flt.filter_function(t)
        residual = 1.0 - t * phi
        D = max(D, float(np.max(np.abs(t * phi))))
        # Use the lambda the filter actually realizes: integer iteration counts
        # cannot hit an arbitrary target level exactly.
        E = max(E, float(np.max(np.abs(phi)) * flt.lam))
        c0 = max(c0, float(np.max(np.abs(residual))))
    return FilterDiagnostics(D=D, E=E, c0=c0)


@dataclass(frozen=True)
class QualificationReport:
    """Result of numerically probing the qualification (2.10) of a family.

    Attributes
    ----------
    nu_estimate:
        Largest probed exponent whose constant stays flat in :math:`\\lambda`.
    growth:
        Per-exponent slope of :math:`\\log c_q(\\lambda)` against
        :math:`\\log\\lambda`.  A slope near zero means (2.10) holds with a
        genuine constant; a slope near :math:`\\nu-q<0` means the family has
        saturated at qualification :math:`\\nu`.
    constants:
        Per-exponent value of :math:`c_q(\\lambda)` at the smallest
        :math:`\\lambda` probed.
    lambdas:
        The realized regularization levels probed.
    """

    nu_estimate: float
    growth: dict[float, float]
    constants: dict[float, float]
    lambdas: Array

    def saturation_estimate(self) -> float:
        """Estimate :math:`\\nu` from the observed saturation slopes.

        Beyond the qualification one has :math:`\\sup_t |r_\\lambda(t)|t^q
        \\asymp \\lambda^{\\nu}`, hence :math:`c_q(\\lambda)\\asymp
        \\lambda^{\\nu-q}` and the measured slope is :math:`\\nu-q`.  Averaging
        ``q + slope`` over the saturated exponents recovers :math:`\\nu`.
        """
        saturated = [q + s for q, s in self.growth.items() if s < -0.1]
        if not saturated:
            return math.inf
        return float(np.median(saturated))


def measure_qualification(
    name: str,
    q_grid: Array | None = None,
    lambdas: Array | None = None,
    *,
    slope_tol: float = 0.1,
    grid: Array | None = None,
    **kwargs: Any,
) -> QualificationReport:
    """Numerically estimate the qualification of a filter family.

    For each candidate exponent ``q`` we evaluate

    .. math::
        c_q(\\lambda) := \\sup_{0<t\\le1} |r_\\lambda(t)|\\,t^q\\,\\lambda^{-q}

    on a grid of :math:`t` and :math:`\\lambda`.  Condition (2.10) asks for
    :math:`c_q(\\lambda)` to be bounded by a constant uniformly in
    :math:`\\lambda`, so the discriminating signal is not the *size* of
    :math:`c_q` but whether it diverges as :math:`\\lambda\\to0`.  We therefore
    regress :math:`\\log c_q(\\lambda)` on :math:`\\log\\lambda` and call ``q``
    admissible when the slope is flat to within ``slope_tol``.

    This distinction matters in practice: momentum methods run for few
    iterations have large but *bounded* constants, which a threshold on
    :math:`c_q` alone would misreport as saturation.
    """
    t = _spectral_grid() if grid is None else np.asarray(grid, dtype=float)
    q_values = np.arange(0.25, 3.01, 0.25) if q_grid is None else np.asarray(q_grid, dtype=float)
    lam_grid = np.logspace(-3.0, -7.0, 9) if lambdas is None else np.asarray(lambdas, dtype=float)

    filters = [make_filter(name, float(lam), **kwargs) for lam in lam_grid]
    realized = np.array([flt.lam for flt in filters], dtype=float)
    residuals = [np.abs(flt.residual_function(t)) for flt in filters]

    growth: dict[float, float] = {}
    constants: dict[float, float] = {}
    log_lam = np.log(realized)
    order = np.argsort(realized)
    for q in q_values:
        weighted = t**q
        c_q = np.array(
            [float(np.max(res * weighted)) / lam**q for res, lam in zip(residuals, realized)]
        )
        slope = float(np.polyfit(log_lam, np.log(c_q), 1)[0])
        growth[float(q)] = slope
        constants[float(q)] = float(c_q[order[0]])

    # Only divergence disqualifies an exponent: a positive slope means c_q
    # shrinks as lambda decreases, which satisfies (2.10) comfortably.
    nu_estimate = 0.0
    for q in sorted(growth):
        if growth[q] >= -slope_tol:
            nu_estimate = q
        else:
            break
    return QualificationReport(
        nu_estimate=nu_estimate, growth=growth, constants=constants, lambdas=realized
    )
