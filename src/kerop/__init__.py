r"""KerOp: random features for operator-valued kernels with spectral filtering.

A reference implementation of

    Mike Nguyen and Nicole Mucke, *Random Features for Operator-Valued Kernels:
    Bridging Kernel Methods and Neural Operators*, AISTATS 2026 (PMLR 300:1495-1503),
    arXiv:2603.00971.

The package implements the estimator of equation (2.11),

.. math::
    F^M_\lambda = \phi_\lambda(\widehat\Sigma_M)\,
                  \widehat{\mathcal{S}}^*_M\mathbf{v},

for vector-valued (operator-valued) random features and a family of spectral
filters :math:`\{\phi_\lambda\}`, together with the exact operator-valued kernel
method it approximates, the Theorem 3.4 parameter prescriptions, and synthetic
and PDE benchmarks on which those prescriptions can be checked.

This is an implementation of theory developed by the authors above.  No claim of
originality is made for any of the theorems; the contribution here is a
transparent, tested implementation and a set of reproducible numerical checks.

Module map
----------
:mod:`kerop.filters`
    Spectral regularization families (Definition 2.2): Tikhonov, iterated
    Tikhonov, Landweber/gradient descent, spectral cut-off, heavy-ball, and
    Brakhage's :math:`\nu`-method, with exact residuals and numerically
    measurable qualification.
:mod:`kerop.features`
    Vector-valued random feature maps satisfying Assumption 2.1, including the
    operator-valued NTK features of a shallow neural operator.
:mod:`kerop.kernels`
    The exact operator-valued kernels those feature maps approximate.
:mod:`kerop.estimators`
    The random feature estimator (2.11) and the exact kernel baseline.
:mod:`kerop.theory`
    Theorem 3.4 and Corollary 3.5 as executable prescriptions.
:mod:`kerop.data`
    A synthetic instance with prescribed :math:`(r,b)`, plus Poisson and Darcy
    solution-operator datasets.
:mod:`kerop.metrics`
    Excess risk and power-law rate estimation.
"""

from __future__ import annotations

from kerop import data, estimators, features, filters, kernels, metrics, theory
from kerop.estimators import ExactOperatorFilter, VectorValuedRFRegressor
from kerop.features import (
    MercerFeatures,
    OperatorNTKFeatures,
    RandomFeatureMap,
    ScalarNTKFeatures,
    SeparableRFF,
)
from kerop.filters import (
    FILTER_REGISTRY,
    HeavyBall,
    IteratedTikhonov,
    Landweber,
    NuMethod,
    SpectralCutoff,
    SpectralFilter,
    Tikhonov,
    make_filter,
    measure_qualification,
)
from kerop.kernels import (
    MercerOperatorKernel,
    OperatorNTKKernel,
    OperatorValuedKernel,
    ScalarNTKKernel,
    SeparableGaussianKernel,
)
from kerop.metrics import RateFit, excess_risk, fit_power_law, relative_error
from kerop.theory import TheoryPrescription, prescribe

__version__ = "0.1.0"

#: Citation for the paper this package implements.
PAPER_CITATION = (
    "Mike Nguyen and Nicole Mucke. Random Features for Operator-Valued Kernels: "
    "Bridging Kernel Methods and Neural Operators. In Proceedings of the 29th "
    "International Conference on Artificial Intelligence and Statistics (AISTATS), "
    "PMLR 300:1495-1503, 2026. arXiv:2603.00971."
)

# Grouped by module rather than sorted alphabetically, so that the ordering
# follows the structure of the paper: filters, then features and the kernels
# they approximate, then the estimators built from both.
__all__ = [  # noqa: RUF022
    "__version__",
    "PAPER_CITATION",
    # submodules
    "data",
    "estimators",
    "features",
    "filters",
    "kernels",
    "metrics",
    "theory",
    # filters
    "SpectralFilter",
    "Tikhonov",
    "IteratedTikhonov",
    "Landweber",
    "SpectralCutoff",
    "HeavyBall",
    "NuMethod",
    "FILTER_REGISTRY",
    "make_filter",
    "measure_qualification",
    # features
    "RandomFeatureMap",
    "MercerFeatures",
    "ScalarNTKFeatures",
    "OperatorNTKFeatures",
    "SeparableRFF",
    # kernels
    "OperatorValuedKernel",
    "MercerOperatorKernel",
    "ScalarNTKKernel",
    "OperatorNTKKernel",
    "SeparableGaussianKernel",
    # estimators
    "VectorValuedRFRegressor",
    "ExactOperatorFilter",
    # theory and metrics
    "TheoryPrescription",
    "prescribe",
    "RateFit",
    "excess_risk",
    "relative_error",
    "fit_power_law",
]