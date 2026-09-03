r"""The prescriptions of Theorem 3.4, as executable code.

Theorem 3.4 of arXiv:2603.00971 states: under Assumptions 3.1-3.3, for a family
of regularization functions with qualification :math:`\nu \ge r\vee1`, choosing

.. math::

    \lambda_n = C\,n^{-\frac{1}{2r+b}}\log^3\!\bigl(\tfrac{2}{\delta}\bigr)

gives, with probability at least :math:`1-\delta`,

.. math::

    \bigl\|G_\rho - \mathcal{S}_{M_n}F^{M_n}_{\lambda_n}\bigr\|_{L^2(\rho_\mathcal{U})}
      \;\le\; \bar{C}\,n^{-\frac{r}{2r+b}}\log^{3r+1}\!\bigl(\tfrac1\delta\bigr),

provided :math:`n\ge n_0 := \exp\bigl(\frac{2r+b}{2r+b-1}\bigr)` and the number
of random features satisfies

.. math::

    M_n \;\ge\; p\cdot\tilde C\cdot\log(n)\cdot
    \begin{cases}
      n^{\frac{1}{2r+b}}, & r\in(0,\tfrac12),\\
      n^{\frac{1+b(2r-1)}{2r+b}}, & r\in[\tfrac12,1],\\
      n^{\frac{2r}{2r+b}}, & r\in(1,\infty).
    \end{cases}

Note the shape of the feature requirement.  In the misspecified regime
:math:`r<\frac12` it coincides with :math:`1/\lambda_n`, i.e. with the iteration
count of gradient descent.  In the well-specified regime :math:`r=\frac12` it
becomes :math:`n^{1/(1+b)}`, which is :math:`\sqrt n` at :math:`b=1` and
recovers the :math:`M=O(\sqrt n\log n)` of Rudi & Rosasco.  For smooth targets
:math:`r>1` it grows like :math:`(1/\lambda_n)^{2r}`, so extra smoothness buys
fewer iterations but costs more features - the trade-off discussed in
Section 3.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "TheoryPrescription",
    "excess_risk_exponent",
    "regularization_exponent",
    "feature_exponent",
    "regularization_parameter",
    "features_required",
    "iterations_required",
    "min_sample_size",
    "check_assumptions",
    "neural_operator_width",
    "prescribe",
]


def check_assumptions(r: float, b: float, qualification: float | None = None) -> None:
    """Validate the standing assumptions of Theorem 3.4.

    Raises
    ------
    ValueError
        If :math:`r\\le0`, :math:`b\\notin[0,1]`, the easy-learning condition
        :math:`2r+b>1` of Assumption 3.3 fails, or the filter qualification is
        below the required :math:`r\\vee1`.
    """
    if r <= 0.0:
        raise ValueError(f"the source exponent r must be positive, got {r}")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"the effective-dimension exponent b must lie in [0, 1], got {b}")
    if 2.0 * r + b <= 1.0:
        raise ValueError(
            f"Theorem 3.4 requires the easy-learning condition 2r + b > 1, "
            f"got 2*{r} + {b} = {2 * r + b}"
        )
    if qualification is not None and qualification < max(r, 1.0):
        raise ValueError(
            f"Theorem 3.4 requires qualification nu >= r or 1, whichever is larger "
            f"(here {max(r, 1.0)}), but the filter has nu = {qualification}"
        )


def excess_risk_exponent(r: float, b: float) -> float:
    """Return :math:`r/(2r+b)`, the exponent of the :math:`L^2` excess risk."""
    check_assumptions(r, b)
    return r / (2.0 * r + b)


def regularization_exponent(r: float, b: float) -> float:
    """Return :math:`1/(2r+b)`, the exponent of :math:`\\lambda_n`."""
    check_assumptions(r, b)
    return 1.0 / (2.0 * r + b)


def feature_exponent(r: float, b: float) -> float:
    """Return the exponent of :math:`n` in the feature requirement of Theorem 3.4.

    The three branches are implemented exactly as stated.  Note that they are
    continuous at :math:`r=1/2`, where :math:`b(2r-1)` vanishes, but *not* at
    :math:`r=1`: the second branch gives :math:`(1+b)/(2+b)` there while the
    third gives :math:`2/(2+b)`, a jump of :math:`(1-b)/(2+b)` that closes only
    in the capacity-independent case :math:`b=1`.  The branches come from
    different arguments in the proof - the :math:`r>1` case relies on the
    operator inequalities developed in Appendix B.5 - so the bound for
    :math:`r` just above one is not tight against the bound at :math:`r=1`.
    """
    check_assumptions(r, b)
    denom = 2.0 * r + b
    if r < 0.5:
        return 1.0 / denom
    if r <= 1.0:
        return (1.0 + b * (2.0 * r - 1.0)) / denom
    return 2.0 * r / denom


def min_sample_size(r: float, b: float) -> float:
    """Return :math:`n_0 = \\exp\\bigl(\\frac{2r+b}{2r+b-1}\\bigr)`.

    Below this sample size the logarithmic factors in the proof are not yet
    controlled and the theorem asserts nothing.
    """
    check_assumptions(r, b)
    return math.exp((2.0 * r + b) / (2.0 * r + b - 1.0))


def regularization_parameter(
    n: int | Array, r: float, b: float, *, constant: float = 1.0, delta: float | None = None
) -> float | Array:
    """Return :math:`\\lambda_n = C n^{-1/(2r+b)}\\log^3(2/\\delta)`.

    The confidence factor is applied only when ``delta`` is given; in the
    experiments the constant is calibrated once and held fixed across
    :math:`n`, so that the measured slope reflects the exponent alone.
    """
    exponent = regularization_exponent(r, b)
    value = constant * np.asarray(n, dtype=float) ** (-exponent)
    if delta is not None:
        value = value * math.log(2.0 / delta) ** 3
    return value if np.ndim(value) else float(value)


def features_required(
    n: int | Array,
    r: float,
    b: float,
    *,
    n_summands: int = 1,
    constant: float = 1.0,
    include_log: bool = True,
) -> int | Array:
    """Return the smallest :math:`M_n` allowed by Theorem 3.4.

    Parameters
    ----------
    n:
        Sample size.
    r, b:
        Source and effective-dimension exponents.
    n_summands:
        The number :math:`p` of summands in the kernel representation (2.5); the
        requirement is proportional to it.
    constant:
        The unspecified constant :math:`\\tilde C`.
    include_log:
        Whether to include the :math:`\\log n` factor.
    """
    exponent = feature_exponent(r, b)
    n_arr = np.asarray(n, dtype=float)
    value = n_summands * constant * n_arr**exponent
    if include_log:
        value = value * np.log(np.maximum(n_arr, math.e))
    result = np.ceil(value).astype(int)
    return result if np.ndim(result) else int(result)


def iterations_required(
    n: int | Array,
    r: float,
    b: float,
    *,
    constant: float = 1.0,
    accelerated: bool = False,
) -> int | Array:
    """Return the iteration count matching :math:`\\lambda_n`.

    Gradient descent realizes :math:`\\lambda = 1/(\\alpha t)`, so
    :math:`t_n = 1/\\lambda_n \\asymp n^{1/(2r+b)}`.  Accelerated schemes such
    as Brakhage's :math:`\\nu`-method realize :math:`\\lambda = 1/(\\alpha t^2)`
    and need only :math:`\\sqrt{t_n}` iterations for the same regularization
    level, which is the :math:`\\sqrt{t}` speed-up quoted in the paper.
    """
    lam = np.asarray(regularization_parameter(n, r, b, constant=1.0), dtype=float)
    steps = 1.0 / (constant * lam)
    if accelerated:
        steps = np.sqrt(steps)
    result = np.maximum(1, np.ceil(steps).astype(int))
    return result if np.ndim(result) else int(result)


def neural_operator_width(
    n: int | Array,
    r: float,
    b: float,
    *,
    feature_dim: int = 1,
    constant: float = 1.0,
    drift_bound: float | None = None,
) -> int | Array:
    """Return the network width required by Corollary 3.5.

    Corollary 3.5 asks for
    :math:`M_n \\ge \\tilde d^2\\tilde C B_{T_n}^6 (T_n^{2r}\\vee T_n)\\log^2 n`
    with :math:`T_n = 1/\\lambda_n`.  Nguyen & Mucke (2024) show the parameter
    drift satisfies :math:`B_{T}=O(\\log T)`, which is the default used when
    ``drift_bound`` is not supplied.
    """
    check_assumptions(r, b)
    steps = np.asarray(iterations_required(n, r, b), dtype=float)
    drift = math.log(math.e + float(np.max(steps))) if drift_bound is None else drift_bound
    n_arr = np.asarray(n, dtype=float)
    value = (
        feature_dim**2
        * constant
        * drift**6
        * np.maximum(steps ** (2.0 * r), steps)
        * np.log(np.maximum(n_arr, math.e)) ** 2
    )
    result = np.ceil(value).astype(int)
    return result if np.ndim(result) else int(result)


@dataclass(frozen=True)
class TheoryPrescription:
    """Everything Theorem 3.4 prescribes for one sample size.

    Attributes
    ----------
    n:
        Sample size.
    r, b:
        Source and effective-dimension exponents.
    lam:
        The prescribed :math:`\\lambda_n`.
    n_features:
        The prescribed lower bound on :math:`M_n`.
    iterations:
        Gradient-descent iteration count :math:`t_n=1/\\lambda_n`.
    accelerated_iterations:
        Iteration count for a :math:`\\nu`-method realizing the same
        :math:`\\lambda_n`.
    risk_bound_exponent:
        The exponent :math:`-r/(2r+b)` of the predicted :math:`L^2` error.
    meets_min_sample_size:
        Whether :math:`n\\ge n_0`, i.e. whether the theorem applies at all.
    """

    n: int
    r: float
    b: float
    lam: float
    n_features: int
    iterations: int
    accelerated_iterations: int
    risk_bound_exponent: float
    meets_min_sample_size: bool


def prescribe(
    n: int,
    r: float,
    b: float,
    *,
    n_summands: int = 1,
    lambda_constant: float = 1.0,
    feature_constant: float = 1.0,
    include_log: bool = True,
) -> TheoryPrescription:
    """Bundle the Theorem 3.4 prescriptions for a single sample size."""
    check_assumptions(r, b)
    return TheoryPrescription(
        n=int(n),
        r=float(r),
        b=float(b),
        lam=float(regularization_parameter(n, r, b, constant=lambda_constant)),
        n_features=int(
            features_required(
                n,
                r,
                b,
                n_summands=n_summands,
                constant=feature_constant,
                include_log=include_log,
            )
        ),
        iterations=int(iterations_required(n, r, b, constant=lambda_constant)),
        accelerated_iterations=int(
            iterations_required(n, r, b, constant=lambda_constant, accelerated=True)
        ),
        risk_bound_exponent=-excess_risk_exponent(r, b),
        meets_min_sample_size=bool(n >= min_sample_size(r, b)),
    )