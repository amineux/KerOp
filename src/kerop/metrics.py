r"""Risk measurement and rate estimation.

The quantity Theorem 3.4 bounds is the :math:`L^2(\rho_\mathcal{U})` distance
between the regression operator and the fitted estimator,

.. math::

    \|G_\rho - \mathcal{S}_M F^M_\lambda\|_{L^2(\rho_\mathcal{U})}^2
      = \int_\mathcal{U} \bigl\|G_\rho(u) - \widehat G(u)\bigr\|_\mathcal{V}^2
        \,d\rho_\mathcal{U}(u),

which is a *noiseless* functional of the estimator: it compares against
:math:`G_\rho` rather than against noisy labels.  On the synthetic instances
:math:`G_\rho` is known in closed form, so this is estimated by plain Monte
Carlo on a large held-out sample and its own sampling error is negligible
relative to the effect being measured.  :func:`excess_risk` returns the norm,
not its square, to match the statement of the theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "RateFit",
    "excess_risk",
    "relative_error",
    "fit_power_law",
    "monte_carlo_standard_error",
]


def _as_samples(values: Array) -> Array:
    """View an array as ``(n, d_v)``, reading a 1-d input as ``n`` scalar outputs.

    This matches the convention of :meth:`kerop.estimators.VectorValuedRFRegressor.fit`,
    where a one-dimensional ``outputs`` argument means :math:`d_v=1` rather than
    a single sample.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected a 1-d or 2-d array, got shape {array.shape}")
    return array


def excess_risk(predictions: Array, targets: Array) -> float:
    """Return the Monte Carlo estimate of the :math:`L^2(\\rho_\\mathcal{U})` error.

    Both arguments have shape ``(n, d_v)`` and are expected in isometric
    coordinates, so the Euclidean norm of a row equals its
    :math:`\\|\\cdot\\|_\\mathcal{V}` norm.
    """
    predictions = np.atleast_2d(np.asarray(predictions, dtype=float))
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    if predictions.shape != targets.shape:
        raise ValueError(f"shape mismatch: {predictions.shape} vs {targets.shape}")
    per_sample = ((predictions - targets) ** 2).sum(axis=1)
    return float(np.sqrt(per_sample.mean()))


def relative_error(predictions: Array, targets: Array) -> float:
    """Return the excess risk divided by :math:`\\|G_\\rho\\|_{L^2(\\rho_\\mathcal{U})}`."""
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    denominator = float(np.sqrt((targets**2).sum(axis=1).mean()))
    if denominator == 0.0:
        raise ValueError("the target operator vanishes; relative error is undefined")
    return excess_risk(predictions, targets) / denominator


def monte_carlo_standard_error(predictions: Array, targets: Array) -> float:
    """Standard error of :func:`excess_risk` from the test-sample fluctuation.

    The squared risk is a sample mean of :math:`\\|G_\\rho(u)-\\widehat
    G(u)\\|^2`; propagating its standard error through the square root gives
    :math:`\\mathrm{se}(\\sqrt{\\bar S}) \\approx \\mathrm{se}(\\bar S)/
    (2\\sqrt{\\bar S})`.
    """
    predictions = np.atleast_2d(np.asarray(predictions, dtype=float))
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    per_sample = ((predictions - targets) ** 2).sum(axis=1)
    mean = per_sample.mean()
    if mean <= 0.0:
        return 0.0
    se_mean = per_sample.std(ddof=1) / np.sqrt(per_sample.size)
    return float(se_mean / (2.0 * np.sqrt(mean)))


@dataclass(frozen=True)
class RateFit:
    """A power-law fit :math:`y \\approx C x^{s}` in log-log coordinates.

    Attributes
    ----------
    slope:
        The estimated exponent :math:`s`.
    intercept:
        :math:`\\log C`.
    slope_stderr:
        Standard error of the slope from the least-squares residuals.
    r_squared:
        Coefficient of determination of the log-log fit; a value close to one
        indicates the measurements really do follow a power law over the range
        probed, which is a precondition for reading the slope as a rate.
    n_points:
        Number of points used.
    """

    slope: float
    intercept: float
    slope_stderr: float
    r_squared: float
    n_points: int

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Return a normal-approximation interval for the slope."""
        return (self.slope - z * self.slope_stderr, self.slope + z * self.slope_stderr)

    def covers(self, target: float, z: float = 1.96) -> bool:
        """Whether the interval for the slope contains ``target``."""
        low, high = self.confidence_interval(z)
        return bool(low <= target <= high)


def fit_power_law(x: Array, y: Array, weights: Array | None = None) -> RateFit:
    """Least-squares fit of :math:`\\log y` against :math:`\\log x`.

    Parameters
    ----------
    x, y:
        Positive arrays of equal length.
    weights:
        Optional weights, e.g. inverse variances of the log-risk across repeats.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size < 3:
        raise ValueError("need at least three points to estimate a rate and its error")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("power-law fits require strictly positive data")

    log_x = np.log(x)
    log_y = np.log(y)
    design = np.vstack([log_x, np.ones_like(log_x)]).T
    if weights is None:
        sqrt_w = np.ones_like(log_x)
    else:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=float))
    weighted_design = design * sqrt_w[:, None]
    weighted_target = log_y * sqrt_w

    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    slope, intercept = float(coefficients[0]), float(coefficients[1])

    residuals = weighted_target - weighted_design @ coefficients
    dof = log_x.size - 2
    residual_variance = float(residuals @ residuals) / dof
    gram_inverse = np.linalg.inv(weighted_design.T @ weighted_design)
    slope_stderr = float(np.sqrt(max(residual_variance * gram_inverse[0, 0], 0.0)))

    total = float(((weighted_target - weighted_target.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0.0 else 1.0

    return RateFit(
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_stderr,
        r_squared=r_squared,
        n_points=int(log_x.size),
    )