r"""Datasets and synthetic operator models.

Two families live here, serving different purposes.

:mod:`kerop.data.spectral`
    A synthetic vector-valued instance in which the source condition
    (Assumption 3.2) and the effective dimension (Assumption 3.3) hold *by
    construction* with prescribed exponents :math:`r` and :math:`b`, because the
    kernel is specified through the eigen-decomposition of its integral
    operator.  This is the only setting in which the exponent predicted by
    Theorem 3.4 is known rather than estimated, so it is the one used to test
    the rate.  Note that the theorem's rates are dimension-free in
    :math:`\mathcal{U}`, and the paper's own numerical illustration
    (Appendix A.3) likewise uses finite-dimensional inputs.

:mod:`kerop.data.pde`
    Genuine PDE solution-operator learning - the Poisson and Darcy maps - where
    inputs are discretized source or coefficient fields and outputs are
    discretized solutions.  Here :math:`(r,b)` are unknown and must be
    estimated, so these tasks are used for the wall-clock comparison against
    exact operator-valued kernel regression and for the CLI demos.

Isometric coordinates
---------------------
Throughout the package, outputs in :math:`\mathcal{V}` are represented as plain
vectors in :math:`\mathbb{R}^{d_v}` under the *standard* Euclidean inner
product.  A discretized function space carries instead the empirical inner
product of the paper,

.. math::
    \langle f,g\rangle_{n_\mathcal{X}}
      = \frac{1}{n_\mathcal{X}}\sum_{k=1}^{n_\mathcal{X}} f(x_k)g(x_k),

so function values must be rescaled by :math:`1/\sqrt{n_\mathcal{X}}` before
entering the estimators.  :func:`isometric_scale` returns that factor.  Keeping
the rescaling in the datasets, rather than carrying a weight through the linear
algebra, means every risk this package reports is already an
:math:`L^2(\mathcal{X},\rho_x)` quantity.
"""

from __future__ import annotations

import math

__all__ = ["isometric_scale"]


def isometric_scale(n_points: int) -> float:
    """Return :math:`1/\\sqrt{n_\\mathcal{X}}`, the isometry to Euclidean coordinates.

    Multiplying discretized function values by this factor makes their Euclidean
    norm equal to the empirical :math:`L^2(\\mathcal{X},\\rho_x)` norm.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    return 1.0 / math.sqrt(n_points)