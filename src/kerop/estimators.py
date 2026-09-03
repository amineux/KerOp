r"""Estimators: random feature spectral filtering, and the exact kernel baseline.

The central object is equation (2.11) of arXiv:2603.00971,

.. math::

    F^M_\lambda = \phi_\lambda(\widehat\Sigma_M)\,
                  \widehat{\mathcal{S}}^*_M\mathbf{v} \;\in\;\mathcal{H}_M,

built from the empirical operators

.. math::

    \widehat\Sigma_M = \frac1n\sum_{j=1}^n K_{M,u_j}K^*_{M,u_j},
    \qquad
    \widehat{\mathcal{S}}^*_M\mathbf v = \frac1n\sum_{j=1}^n K_{M,u_j}v_j .

Writing the random feature map as :math:`\Psi_M(u)` (see :mod:`kerop.features`)
identifies :math:`\mathcal{H}_M` with :math:`\mathbb{R}^{pM}` and turns these
into :math:`\widehat\Sigma_M = Z^\top Z/n` and
:math:`\widehat{\mathcal{S}}^*_M\mathbf v = Z^\top\mathrm{vec}(\mathbf v)/n` for
the design matrix :math:`Z\in\mathbb{R}^{nd_v\times pM}`.

For comparison, :class:`ExactOperatorFilter` runs *the same* filters on the
exact operator-valued kernel.  The operator identity
:math:`\phi_\lambda(\mathcal{S}^*\mathcal{S})\mathcal{S}^* =
\mathcal{S}^*\phi_\lambda(\mathcal{S}\mathcal{S}^*)` moves the filter onto the
block Gram matrix :math:`\mathbf{G}/n \in \mathbb{R}^{nd_v\times nd_v}`, so the
two estimators differ only in *which* operator the filter is applied to.  That
is the comparison Theorem 3.4 is about, and the reason the two classes below
share the filter implementations verbatim.

Cost model, matching the accounting in Section 1 and Section 2.2 of the paper:

===================================  ==========================  ==============
Estimator                            Time                        Memory
===================================  ==========================  ==============
Random features, explicit filter     :math:`O(nd_v(pM)^2+(pM)^3)` :math:`O(nd_vpM)`
Random features, iterative filter    :math:`O(nd_v\,pM\,t)`       :math:`O(nd_vpM)`
Exact operator-valued kernel         :math:`O((nd_v)^3)`          :math:`O((nd_v)^2)`
===================================  ==========================  ==============
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh

from kerop.features import RandomFeatureMap
from kerop.filters import SpectralFilter, _IterativeFilter, make_filter

Array = NDArray[np.float64]

__all__ = [
    "FitReport",
    "VectorValuedRFRegressor",
    "ExactOperatorFilter",
    "spectral_norm_estimate",
]


@dataclass
class FitReport:
    """Timing and size information recorded by a fit.

    Attributes
    ----------
    fit_seconds:
        Wall-clock time of :meth:`fit`, excluding data generation.
    assemble_seconds:
        Time spent forming the operator (the normal equations or the Gram
        matrix) as opposed to applying the filter.
    solve_seconds:
        Time spent applying the filter.
    operator_dim:
        Side length of the operator the filter was applied to: :math:`pM` for
        random features, :math:`nd_v` for the exact estimator.
    spectral_scale:
        The factor by which the operator was divided to bring its spectrum into
        :math:`[0,1]`, as Definition 2.2 requires.
    peak_operator_bytes:
        Size of the largest dense array held during the fit.
    """

    fit_seconds: float = 0.0
    assemble_seconds: float = 0.0
    solve_seconds: float = 0.0
    operator_dim: int = 0
    spectral_scale: float = 1.0
    peak_operator_bytes: int = 0
    extras: dict[str, float] = field(default_factory=dict)



#: How much less efficiently the matrix-free route uses the machine than the
#: assembled one, per floating-point operation.
_MATVEC_INEFFICIENCY = 4.0


def spectral_norm_estimate(
    matvec: LinearOperator | Array, dim: int, *, tol: float = 1e-4
) -> float:
    """Estimate the largest eigenvalue of a symmetric PSD operator.

    Definition 2.2 constrains the filters on :math:`t\\in(0,1]`, so the operator
    must be rescaled by an upper bound on its spectral norm before a filter is
    applied.  A Lanczos estimate is used because the trivial bound
    :math:`\\mathrm{tr}` overestimates by up to a factor of the dimension, which
    would silently inflate the iteration counts of the iterative filters by the
    same factor.
    """
    if dim == 1:
        value = float(np.asarray(matvec @ np.ones(1)).ravel()[0])
        return max(value, 0.0)
    try:
        top = eigsh(matvec, k=1, which="LM", return_eigenvectors=False, tol=tol)
        return float(np.abs(top[0]))
    except Exception:
        # Lanczos can fail to converge on nearly rank-deficient operators; the
        # trace is a safe fallback upper bound.
        if isinstance(matvec, np.ndarray):
            return float(np.trace(matvec))
        probe = np.eye(dim)
        return float(sum((matvec @ probe[:, i])[i] for i in range(dim)))


class VectorValuedRFRegressor:
    r"""Spectral filtering with vector-valued random features, i.e. (2.11).

    Parameters
    ----------
    features:
        The random feature map, one realization of the representation (2.5).
    filter_name:
        Key into :data:`kerop.filters.FILTER_REGISTRY`.
    lam:
        Regularization level :math:`\lambda`, in units of the spectral norm of
        :math:`\widehat\Sigma_M` (i.e. after the normalization that puts the
        spectrum in :math:`[0,1]`).  Either ``lam`` or ``filter_obj`` must be
        given.
    filter_obj:
        A pre-built filter, e.g. ``Landweber.from_iterations(50)`` when the
        natural hyper-parameter is the iteration count rather than
        :math:`\lambda`.
    filter_kwargs:
        Extra options forwarded to the filter, such as ``step`` or ``nu``.
    spectral_scale:
        ``"power"`` estimates the spectral norm by Lanczos (default),
        ``"trace"`` uses the trace bound, and a float fixes the scale.  Fixing
        it is preferable in a rate study, where :math:`\lambda_n` should follow
        the theoretical prescription in absolute units rather than drift with a
        sample-dependent normalization.
    assemble:
        Whether to form :math:`\widehat\Sigma_M` explicitly. ``"auto"`` keeps it
        implicit for iterative filters whose iteration count is below
        :math:`pM`, which is the regime where the paper's :math:`O(nMt)`
        accounting beats the :math:`O(nM^2)` one.

    Attributes
    ----------
    coefficients:
        The fitted :math:`F^M_\lambda\in\mathbb{R}^{pM}`.
    report:
        A :class:`FitReport` with timings and operator sizes.
    """

    def __init__(
        self,
        features: RandomFeatureMap,
        filter_name: str = "tikhonov",
        lam: float | None = None,
        *,
        filter_obj: SpectralFilter | None = None,
        filter_kwargs: dict[str, object] | None = None,
        spectral_scale: float | Literal["power", "trace"] = "power",
        assemble: bool | Literal["auto"] = "auto",
    ) -> None:
        if (lam is None) == (filter_obj is None):
            raise ValueError("provide exactly one of lam or filter_obj")
        self.features = features
        self.filter_name = filter_name
        self.lam = lam
        self._filter_obj = filter_obj
        self.filter_kwargs = dict(filter_kwargs or {})
        self.spectral_scale_spec = spectral_scale
        self.assemble = assemble
        self.coefficients: Array | None = None
        self.report = FitReport()
        self.filter: SpectralFilter | None = filter_obj

    def _build_filter(self) -> SpectralFilter:
        if self._filter_obj is not None:
            return self._filter_obj
        assert self.lam is not None
        return make_filter(self.filter_name, self.lam, **self.filter_kwargs)

    def fit(self, inputs: Array, outputs: Array) -> VectorValuedRFRegressor:
        """Fit the estimator on ``n`` input/output pairs.

        ``outputs`` has shape ``(n, d_v)`` and is expected in isometric
        coordinates, so that Euclidean norms coincide with
        :math:`\\|\\cdot\\|_{\\mathcal{V}}`.
        """
        start = time.perf_counter()
        outputs = np.asarray(outputs, dtype=float)
        if outputs.ndim == 1:
            outputs = outputs[:, None]
        n_samples = outputs.shape[0]
        if outputs.shape[1] != self.features.output_dim:
            raise ValueError(
                f"outputs have {outputs.shape[1]} columns but the feature map has "
                f"output_dim={self.features.output_dim}"
            )

        t0 = time.perf_counter()
        design = self.features.design_matrix(inputs)  # (n*d_v, p*M)
        if design.shape[0] != n_samples * self.features.output_dim:
            raise ValueError("inputs and outputs disagree on the number of samples")
        rhs = design.T @ outputs.reshape(-1) / n_samples
        dim = design.shape[1]

        flt = self._build_filter()
        use_matvec = self._use_matvec(flt, dim, n_samples)
        gram: Array | None = None
        if not use_matvec:
            gram = design.T @ design / n_samples
        assemble_seconds = time.perf_counter() - t0

        scale = self._resolve_scale(gram, design, n_samples, dim)
        if scale <= 0.0:
            raise ValueError("the empirical covariance operator vanishes")

        t0 = time.perf_counter()
        if use_matvec:
            def matvec(x: Array) -> Array:
                return design.T @ (design @ x) / (n_samples * scale)

            self.coefficients = flt.apply(matvec, rhs / scale, dim=dim)
        else:
            assert gram is not None
            self.coefficients = flt.apply(gram / scale, rhs / scale)
        solve_seconds = time.perf_counter() - t0

        self.filter = flt
        self.report = FitReport(
            fit_seconds=time.perf_counter() - start,
            assemble_seconds=assemble_seconds,
            solve_seconds=solve_seconds,
            operator_dim=dim,
            spectral_scale=scale,
            peak_operator_bytes=int(
                design.nbytes + (0 if gram is None else gram.nbytes)
            ),
            extras={
                "n_samples": float(n_samples),
                "n_features": float(self.features.n_features),
                "n_summands": float(self.features.n_summands),
                "lambda": float(flt.lam),
                "matvecs": float(flt.matvec_count),
                "matrix_free": float(use_matvec),
            },
        )
        return self

    def _use_matvec(self, flt: SpectralFilter, dim: int, n_samples: int) -> bool:
        if self.assemble is True:
            return False
        if self.assemble is False:
            return True
        if not isinstance(flt, _IterativeFilter):
            return False
        n_rows = n_samples * self.features.output_dim
        matvec_cost = 2.0 * flt.iterations * n_rows * dim * _MATVEC_INEFFICIENCY
        assemble_cost = float(n_rows) * dim**2 + flt.iterations * dim**2
        return matvec_cost < assemble_cost

    def _resolve_scale(
        self, gram: Array | None, design: Array, n_samples: int, dim: int
    ) -> float:
        spec = self.spectral_scale_spec
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return float(spec)
        if spec == "trace":
            if gram is not None:
                return float(np.trace(gram))
            return float((design**2).sum() / n_samples)
        if spec != "power":
            raise ValueError(f"unknown spectral_scale {spec!r}")
        if gram is not None:
            return spectral_norm_estimate(gram, dim)
        operator = LinearOperator(
            shape=(dim, dim),
            matvec=lambda x: design.T @ (design @ x) / n_samples,
            dtype=float,
        )
        return spectral_norm_estimate(operator, dim)

    def predict(self, inputs: Array) -> Array:
        """Return :math:`\\mathcal{S}_M F^M_\\lambda` evaluated at ``inputs``."""
        if self.coefficients is None:
            raise RuntimeError("call fit before predict")
        tensor = self.features.feature_tensor(inputs)
        return np.einsum("iam,m->ia", tensor, self.coefficients, optimize=True)

    def rkhs_norm(self) -> float:
        """Return :math:`\\|F^M_\\lambda\\|_{\\mathcal{H}_M}`."""
        if self.coefficients is None:
            raise RuntimeError("call fit before rkhs_norm")
        return float(np.linalg.norm(self.coefficients))


class ExactOperatorFilter:
    r"""Spectral filtering with the exact operator-valued kernel.

    This is the :math:`M\to\infty` reference: the same filter families applied to
    the exact kernel rather than to a random feature approximation of it.  Using
    :math:`\phi_\lambda(\mathcal{S}^*\mathcal{S})\mathcal{S}^* =
    \mathcal{S}^*\phi_\lambda(\mathcal{S}\mathcal{S}^*)`, the estimator is

    .. math::
        \widehat G(u) = \frac1n\sum_{j=1}^n K(u,u_j)\,c_j,
        \qquad c = \phi_\lambda\!\bigl(\mathbf{G}/n\bigr)\,\mathrm{vec}(\mathbf v),

    where :math:`\mathbf{G}\in\mathbb{R}^{nd_v\times nd_v}` is the block Gram
    matrix with blocks :math:`K(u_i,u_j)\in\mathbb{R}^{d_v\times d_v}`.  For
    Tikhonov this reduces to the familiar :math:`(\mathbf{G}+n\lambda I)^{-1}`
    operator-valued kernel ridge regression.

    The point of the class is the cost: the Gram matrix grows as
    :math:`(nd_v)^2` in memory and its factorization as :math:`(nd_v)^3` in
    time, which is what the random feature estimator is meant to avoid.

    Parameters
    ----------
    kernel:
        An object exposing ``output_dim`` and ``block_gram(U, Utilde)`` as in
        :mod:`kerop.kernels`.
    filter_name, lam, filter_obj, filter_kwargs, spectral_scale:
        As in :class:`VectorValuedRFRegressor`.
    """

    def __init__(
        self,
        kernel: object,
        filter_name: str = "tikhonov",
        lam: float | None = None,
        *,
        filter_obj: SpectralFilter | None = None,
        filter_kwargs: dict[str, object] | None = None,
        spectral_scale: float | Literal["power", "trace"] = "power",
    ) -> None:
        if (lam is None) == (filter_obj is None):
            raise ValueError("provide exactly one of lam or filter_obj")
        if not hasattr(kernel, "block_gram") or not hasattr(kernel, "output_dim"):
            raise TypeError("kernel must expose block_gram and output_dim")
        self.kernel = kernel
        self.filter_name = filter_name
        self.lam = lam
        self._filter_obj = filter_obj
        self.filter_kwargs = dict(filter_kwargs or {})
        self.spectral_scale_spec = spectral_scale
        self.dual_coefficients: Array | None = None
        self.train_inputs: Array | None = None
        self.report = FitReport()
        self.filter: SpectralFilter | None = filter_obj

    def fit(self, inputs: Array, outputs: Array) -> ExactOperatorFilter:
        """Form the block Gram matrix and apply the filter to it."""
        start = time.perf_counter()
        outputs = np.asarray(outputs, dtype=float)
        if outputs.ndim == 1:
            outputs = outputs[:, None]
        n_samples = outputs.shape[0]
        output_dim = int(self.kernel.output_dim)  # type: ignore[attr-defined]
        if outputs.shape[1] != output_dim:
            raise ValueError(
                f"outputs have {outputs.shape[1]} columns but the kernel has "
                f"output_dim={output_dim}"
            )

        t0 = time.perf_counter()
        gram = self.kernel.block_gram(inputs)  # type: ignore[attr-defined]
        assemble_seconds = time.perf_counter() - t0
        dim = gram.shape[0]
        if dim != n_samples * output_dim:
            raise ValueError(f"block Gram matrix has shape {gram.shape}, expected {dim}")

        gram = gram / n_samples
        scale = self._resolve_scale(gram, dim)
        flt = self._filter_obj if self._filter_obj is not None else make_filter(
            self.filter_name, float(self.lam), **self.filter_kwargs  # type: ignore[arg-type]
        )

        t0 = time.perf_counter()
        self.dual_coefficients = flt.apply(gram / scale, outputs.reshape(-1) / scale)
        solve_seconds = time.perf_counter() - t0

        self.train_inputs = np.asarray(inputs)
        self.filter = flt
        self._n_train = n_samples
        self.report = FitReport(
            fit_seconds=time.perf_counter() - start,
            assemble_seconds=assemble_seconds,
            solve_seconds=solve_seconds,
            operator_dim=dim,
            spectral_scale=scale,
            peak_operator_bytes=int(gram.nbytes),
            extras={
                "n_samples": float(n_samples),
                "lambda": float(flt.lam),
                "matvecs": float(flt.matvec_count),
            },
        )
        return self

    def _resolve_scale(self, gram: Array, dim: int) -> float:
        spec = self.spectral_scale_spec
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return float(spec)
        if spec == "trace":
            return float(np.trace(gram))
        if spec != "power":
            raise ValueError(f"unknown spectral_scale {spec!r}")
        return spectral_norm_estimate(gram, dim)

    def predict(self, inputs: Array, chunk_size: int = 256) -> Array:
        """Evaluate the fitted operator at ``inputs``.

        The cross-Gram matrix is formed in chunks of ``chunk_size`` test points,
        since its full size is :math:`n_{\\text{test}}d_v\\times nd_v`.

        Kernels that expose an exact factorization :math:`K(u,\\tilde u) =
        \\Xi(u)\\Xi(\\tilde u)^\\top` take a fast path: the training-side factor is
        contracted with the dual coefficients once, after which prediction is a
        single matrix product.  This affects evaluation only - the fit still
        forms and factorizes the full :math:`nd_v\\times nd_v` Gram matrix, which
        is the cost being measured.
        """
        if self.dual_coefficients is None or self.train_inputs is None:
            raise RuntimeError("call fit before predict")
        output_dim = int(self.kernel.output_dim)  # type: ignore[attr-defined]
        inputs = np.asarray(inputs)
        n_test = inputs.shape[0]

        factor = getattr(self.kernel, "factor", None)
        if callable(factor):
            weights = factor(self.train_inputs).T @ self.dual_coefficients
            predictions = factor(inputs) @ weights / self._n_train
            return predictions.reshape(n_test, output_dim)

        result = np.empty((n_test, output_dim), dtype=float)
        for start in range(0, n_test, chunk_size):
            stop = min(start + chunk_size, n_test)
            cross = self.kernel.block_gram(  # type: ignore[attr-defined]
                inputs[start:stop], self.train_inputs
            )
            block = (cross @ self.dual_coefficients) / self._n_train
            result[start:stop] = block.reshape(stop - start, output_dim)
        return result