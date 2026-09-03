r"""Reproducible numerical experiments.

Three experiments, each answering a specific quantitative question about
arXiv:2603.00971.

:func:`run_rate_experiment`
    Does the excess risk decay at the exponent :math:`r/(2r+b)` of Theorem 3.4?
    Run on :class:`kerop.data.spectral.SpectralOperatorModel`, where the source
    and capacity exponents are known by construction, and where they are
    additionally *measured* from the exact spectrum before the rate is fitted,
    so a disagreement can be attributed either to the instance or to the
    estimator.

:func:`run_feature_threshold`
    Is :math:`M \gtrsim \sqrt n\,p` enough?  This recreates the setup of
    Appendix A.3: kernel gradient descent on the real-valued NTK, with the test
    error mapped over the number of random features :math:`M` and the number of
    iterations :math:`T`, and the plateau threshold in :math:`M` extracted and
    checked against :math:`\sqrt{n}\,p` with :math:`p=d+2`.

:func:`run_walltime_benchmark`
    Is the random feature estimator faster than exact operator-valued kernel
    regression at *matched* excess risk?  Run on a PDE solution operator with
    the operator-valued NTK, comparing against the exact kernel in closed form.

Every function returns plain dictionaries and lists so that results serialize
to JSON without special handling.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kerop import theory
from kerop.data.pde import OperatorDataset, make_dataset
from kerop.data.spectral import SpectralOperatorModel
from kerop.estimators import ExactOperatorFilter, VectorValuedRFRegressor
from kerop.features import OperatorNTKFeatures, ScalarNTKFeatures
from kerop.filters import Landweber, make_filter
from kerop.kernels import OperatorNTKKernel
from kerop.metrics import excess_risk, fit_power_law

Array = NDArray[np.float64]

__all__ = [
    "RateConfig",
    "RateResult",
    "run_rate_experiment",
    "DEFAULT_RATE_CONFIGS",
    "run_feature_threshold",
    "run_walltime_benchmark",
    "run_filter_report",
]


# --------------------------------------------------------------------------- #
# Experiment 1: the rate of Theorem 3.4
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RateConfig:
    """One :math:`(r,b)` setting for the rate experiment.

    Attributes
    ----------
    name:
        Short label used in the output files.
    r, b:
        Nominal source and capacity exponents imposed on the instance.
    filters:
        ``(registry name, options)`` pairs to run.  A filter whose
        qualification is below :math:`r\\vee1` violates the hypotheses of
        Theorem 3.4 and is expected to underperform; such entries are kept as
        negative controls and flagged in the output.
    n_grid:
        Sample sizes.
    n_modes, output_dim, input_dim:
        Geometry of the synthetic instance.
    feature_constant:
        The constant :math:`\\tilde C` multiplying the feature requirement.
    include_feature_log:
        Whether to include the :math:`\\log n` factor of Theorem 3.4 in
        :math:`M_n`.
    noise_std:
        Label noise level.
    """

    name: str
    r: float
    b: float
    filters: tuple[tuple[str, dict[str, Any]], ...]
    n_grid: tuple[int, ...] = (250, 500, 1000, 2000, 4000)
    n_modes: int = 4096
    output_dim: int = 6
    input_dim: int = 1
    feature_constant: float = 1.0
    include_feature_log: bool = True
    noise_std: float = 0.05


#: The configurations reported in ``results/``.  Values of ``b`` are kept at or
#: below 0.7 because at :math:`b=1` the integral operator sits at the
#: trace-class boundary, where the effective dimension carries an extra
#: logarithm and no clean power law in :math:`\lambda` exists to compare against.
DEFAULT_RATE_CONFIGS: tuple[RateConfig, ...] = (
    RateConfig(
        name="well-specified",
        r=0.5,
        b=0.5,
        filters=(
            ("tikhonov", {}),
            ("landweber", {}),
            ("nu_method", {"nu": 2.0}),
            ("heavy_ball", {"momentum": 0.9}),
        ),
    ),
    RateConfig(
        name="misspecified",
        r=0.3,
        b=0.7,
        filters=(("tikhonov", {}), ("landweber", {}), ("nu_method", {"nu": 2.0})),
        # This configuration sits closest to the easy-learning boundary
        # 2r + b > 1, where the minimum sample size n_0 = exp((2r+b)/(2r+b-1))
        # is largest and convergence to the asymptotic exponent slowest.  Two
        # choices follow from that, both checked in docs/reproducing.md: the
        # smallest sample size is dropped, because it carries the most
        # curvature, and the unspecified constant in M_n is left at one rather
        # than reduced, because the feature exponent 1/(2r+b) = 0.77 is steep
        # enough that a smaller constant starves the small-n end of features and
        # steepens the measured slope.
        n_grid=(500, 1000, 2000, 4000),
    ),
    RateConfig(
        name="smooth",
        r=1.0,
        b=0.5,
        filters=(("tikhonov", {}), ("landweber", {}), ("nu_method", {"nu": 2.0})),
    ),
    RateConfig(
        name="beyond-tikhonov-qualification",
        r=1.5,
        b=0.5,
        filters=(
            # Tikhonov has qualification 1 < r, so Theorem 3.4 does not cover
            # it here; it is kept as a negative control.
            ("tikhonov", {}),
            ("landweber", {}),
            ("nu_method", {"nu": 2.0}),
            ("iterated_tikhonov", {"order": 2}),
        ),
        # The r > 1 branch of the feature requirement grows like n^{2r/(2r+b)}.
        feature_constant=0.25,
    ),
)


@dataclass
class RateResult:
    """Outcome of the rate experiment for one configuration."""

    config: dict[str, Any]
    nominal_exponent: float
    measured_r: float
    measured_b: float
    measured_exponent: float
    lambda_constant: float
    lambda_constant_bounds: tuple[float, float]
    lambda_calibration_filter: str
    lambda_calibration_trace: list[dict[str, float]]
    lambda_window: tuple[float, float]
    lambda_range: tuple[float, float]
    assumption_fit_quality: dict[str, float]
    rows: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)


#: Largest iteration count an iterative filter may be asked for in the rate
#: experiment.  An iterative filter realizes :math:`\lambda` through
#: :math:`t=1/(\alpha\lambda)`, so nothing bounds the work from below unless the
#: regularization constant is bounded away from zero: at :math:`r=1.5`,
#: :math:`b=0.5` the smallest constant compatible with the instance's spectral
#: window implies over :math:`10^7` gradient steps, which is neither affordable
#: nor a method anyone would run.  The constant is therefore additionally
#: constrained to keep gradient descent within this many iterations at the
#: largest sample size.
MAX_FILTER_ITERATIONS = 20_000


def _lambda_constant_bounds(
    model: SpectralOperatorModel,
    n_grid: tuple[int, ...],
    r: float,
    b: float,
    max_iterations: int = MAX_FILTER_ITERATIONS,
) -> tuple[float, float]:
    r"""Range of :math:`C` admissible for :math:`\lambda_n = Cn^{-1/(2r+b)}`.

    Two constraints apply, both in the *normalized* spectral units the
    estimators use.

    The truncated instance has a finite spectrum, so its power laws in
    :math:`\lambda` hold only inside
    :meth:`SpectralOperatorModel.usable_lambda_window`; this bounds :math:`C`
    from both sides.

    Separately, an iterative filter reaches :math:`\lambda` only by iterating
    :math:`t=1/(\alpha\lambda)` times, so a small :math:`C` makes gradient
    descent arbitrarily expensive.  Requiring :math:`t\le` ``max_iterations`` at
    the largest sample size raises the lower bound, which for the smooth
    configurations is the binding one.
    """
    low, high = model.usable_lambda_window()
    scale = model.kappa_squared()
    exponent = theory.regularization_exponent(r, b)
    largest_factor = min(n_grid) ** (-exponent)
    smallest_factor = max(n_grid) ** (-exponent)
    window_lower = low / smallest_factor / scale
    iteration_lower = max(n_grid) ** exponent / max_iterations
    lower = max(window_lower, iteration_lower)
    upper = high / largest_factor / scale
    if lower >= upper:
        raise ValueError(
            f"no admissible regularization constant for n in "
            f"{min(n_grid)}..{max(n_grid)}: the spectral window requires C >= "
            f"{window_lower:.3g}, an iteration budget of {max_iterations} requires C >= "
            f"{iteration_lower:.3g}, and the window caps C at {upper:.3g}; "
            f"increase n_modes or the iteration budget"
        )
    return lower, upper


def _calibrate_lambda_constant(
    model: SpectralOperatorModel,
    config: RateConfig,
    *,
    reference_filter: tuple[str, dict[str, Any]],
    seed: int,
    repeats: int = 2,
    n_candidates: int = 9,
    n_test: int = 1000,
) -> tuple[float, list[dict[str, float]]]:
    r"""Choose the constant :math:`C` in :math:`\lambda_n = Cn^{-1/(2r+b)}`.

    Theorem 3.4 leaves :math:`C` free and requires only that it not depend on
    :math:`n`.  That freedom matters in a finite-sample experiment: an arbitrary
    :math:`C` leaves the estimator uniformly over- or under-regularized, which
    biases the measured slope even though the exponent is unaffected
    asymptotically.

    The constant is therefore selected once, by minimizing the excess risk at a
    single reference sample size - the largest one, where the asymptotics are
    closest - and then held fixed across every :math:`n`, filter, and repeat.
    This is legitimate precisely because the theorem asserts that the *optimal*
    :math:`\lambda` scales as :math:`n^{-1/(2r+b)}`, so the best constant is
    independent of :math:`n`; calibrating at one sample size and extrapolating
    with the prescribed exponent is a test of that claim rather than an
    exploitation of it.  Candidates are restricted to keep every
    :math:`\lambda_n` inside the instance's power-law window.

    Returns
    -------
    (constant, trace):
        The selected constant and the risk at each candidate, for the record.
    """
    lower, upper = _lambda_constant_bounds(model, config.n_grid, config.r, config.b)
    candidates = np.logspace(np.log10(lower), np.log10(upper), n_candidates)
    exponent = theory.regularization_exponent(config.r, config.b)
    scale = model.kappa_squared()
    reference_n = max(config.n_grid)
    n_features = int(
        theory.features_required(
            reference_n,
            config.r,
            config.b,
            n_summands=1,
            constant=config.feature_constant,
            include_log=config.include_feature_log,
        )
    )
    filter_name, filter_kwargs = reference_filter

    calibration_rng = np.random.default_rng([seed, 777])
    test_inputs, test_targets = model.test_set(n_test, calibration_rng)

    trace: list[dict[str, float]] = []
    for candidate in candidates:
        risks = []
        for repeat in range(repeats):
            stream = np.random.default_rng([seed, 888, repeat])
            inputs, outputs = model.sample(reference_n, stream)
            features = model.features(n_features, stream)
            estimator = VectorValuedRFRegressor(
                features,
                filter_name,
                float(candidate * reference_n**-exponent),
                filter_kwargs=filter_kwargs,
                spectral_scale=scale,
            ).fit(inputs, outputs)
            risks.append(excess_risk(estimator.predict(test_inputs), test_targets))
        trace.append(
            {
                "lambda_constant": float(candidate),
                "lambda_absolute_at_reference": float(
                    candidate * scale * reference_n**-exponent
                ),
                "mean_excess_risk": float(np.mean(risks)),
            }
        )
    best = min(trace, key=lambda row: row["mean_excess_risk"])
    return float(best["lambda_constant"]), trace


def run_rate_experiment(
    configs: tuple[RateConfig, ...] = DEFAULT_RATE_CONFIGS,
    *,
    repeats: int = 12,
    n_test: int = 4000,
    seed: int = 20260301,
    verbose: bool = True,
) -> list[RateResult]:
    r"""Measure the excess-risk exponent and compare it with Theorem 3.4.

    For each configuration the protocol is:

    1. Build the synthetic instance with nominal :math:`(r,b)`.
    2. Calibrate the single constant :math:`C` so that
       :math:`\{\lambda_n\}` lies inside the instance's power-law window.
    3. *Measure* :math:`r` and :math:`b` on exactly that range of
       :math:`\lambda`, from the exact spectrum and the exact bias.  No sampling
       is involved, so these are properties of the instance.
    4. For each filter, sample size and repeat, fit
       :math:`F^{M_n}_{\lambda_n}` with :math:`M_n` from Theorem 3.4 and
       evaluate the excess risk against :math:`G_\rho` on a fixed test set.
    5. Fit :math:`\log(\text{risk})` against :math:`\log n` and compare the
       slope with :math:`-r/(2r+b)`, using both the nominal and the measured
       exponents.

    Parameters
    ----------
    configs:
        Settings to run.
    repeats:
        Independent draws of data and features per point.  The reported
        uncertainty is the standard error of the mean log risk.
    n_test:
        Size of the noiseless test set used for the Monte Carlo risk.
    seed:
        Base seed; every ``(config, filter, n, repeat)`` gets a distinct stream.
    verbose:
        Print progress as configurations complete.
    """
    results: list[RateResult] = []

    for config in configs:
        model = SpectralOperatorModel(
            r=config.r,
            b=config.b,
            n_modes=config.n_modes,
            output_dim=config.output_dim,
            input_dim=config.input_dim,
            noise_std=config.noise_std,
            seed=seed,
        )
        scale = model.kappa_squared()
        # Calibrate against a filter whose qualification satisfies the theorem's
        # hypothesis, so the constant is not tuned around a saturating method.
        reference_filter = next(
            (
                entry
                for entry in config.filters
                if make_filter(entry[0], 0.1, **entry[1]).qualification >= max(config.r, 1.0)
            ),
            config.filters[0],
        )
        lambda_constant, calibration_trace = _calibrate_lambda_constant(
            model, config, reference_filter=reference_filter, seed=seed
        )
        lambdas = np.array(
            [
                lambda_constant * n ** (-theory.regularization_exponent(config.r, config.b))
                for n in config.n_grid
            ]
        )
        absolute_lambdas = np.sort(lambdas * scale)

        # Measure the assumptions on exactly the lambda range that will be used.
        probe = np.logspace(
            np.log10(absolute_lambdas[0]), np.log10(absolute_lambdas[-1]), 15
        )
        capacity_fit = model.effective_dimension_fit(probe)
        source_fit = model.source_exponent_fit(probe)
        measured_b = float(min(max(-capacity_fit.slope, 1e-6), 1.0))
        measured_r = float(max(source_fit.slope, 1e-6))
        measured_exponent = measured_r / (2.0 * measured_r + measured_b)

        test_rng = np.random.default_rng(seed + 1)
        test_inputs, test_targets = model.test_set(n_test, test_rng)

        rows: list[dict[str, Any]] = []
        for filter_index, (filter_name, filter_kwargs) in enumerate(config.filters):
            for n_index, n_samples in enumerate(config.n_grid):
                lam = float(lambdas[n_index])
                n_features = int(
                    theory.features_required(
                        n_samples,
                        config.r,
                        config.b,
                        n_summands=1,
                        constant=config.feature_constant,
                        include_log=config.include_feature_log,
                    )
                )
                for repeat in range(repeats):
                    stream = np.random.default_rng(
                        [seed, filter_index, n_index, repeat]
                    )
                    inputs, outputs = model.sample(n_samples, stream)
                    features = model.features(n_features, stream)
                    estimator = VectorValuedRFRegressor(
                        features,
                        filter_name,
                        lam,
                        filter_kwargs=filter_kwargs,
                        spectral_scale=scale,
                    ).fit(inputs, outputs)
                    risk = excess_risk(estimator.predict(test_inputs), test_targets)
                    rows.append(
                        {
                            "filter": filter_name,
                            "filter_options": dict(filter_kwargs),
                            "n": n_samples,
                            "n_features": n_features,
                            "lambda_normalized": lam,
                            "lambda_absolute": lam * scale,
                            "iterations": int(estimator.report.extras["matvecs"]),
                            "repeat": repeat,
                            "excess_risk": risk,
                            "fit_seconds": estimator.report.fit_seconds,
                        }
                    )

        summary = _summarize_rate_rows(
            rows, config, model, nominal=theory.excess_risk_exponent(config.r, config.b),
            measured=measured_exponent,
        )
        result = RateResult(
            config={
                "name": config.name,
                "r": config.r,
                "b": config.b,
                "n_grid": list(config.n_grid),
                "n_modes": config.n_modes,
                "output_dim": config.output_dim,
                "input_dim": config.input_dim,
                "feature_constant": config.feature_constant,
                "include_feature_log": config.include_feature_log,
                "noise_std": config.noise_std,
                "repeats": repeats,
                "n_test": n_test,
            },
            nominal_exponent=theory.excess_risk_exponent(config.r, config.b),
            measured_r=measured_r,
            measured_b=measured_b,
            measured_exponent=measured_exponent,
            lambda_constant=lambda_constant,
            lambda_constant_bounds=_lambda_constant_bounds(
                model, config.n_grid, config.r, config.b
            ),
            lambda_calibration_filter=reference_filter[0],
            lambda_calibration_trace=calibration_trace,
            lambda_window=model.usable_lambda_window(),
            lambda_range=(float(absolute_lambdas[0]), float(absolute_lambdas[-1])),
            assumption_fit_quality={
                "capacity_r_squared": capacity_fit.r_squared,
                "source_r_squared": source_fit.r_squared,
            },
            rows=rows,
            summary=summary,
        )
        results.append(result)
        if verbose:
            print(f"[rate] {config.name}: r={config.r} b={config.b}")
            print(
                f"        measured r={measured_r:.4f} b={measured_b:.4f} -> "
                f"predicted exponent {measured_exponent:.4f} "
                f"(nominal {result.nominal_exponent:.4f})"
            )
            for entry in summary:
                verdict = "ok " if entry["within_tolerance"] else "MISS"
                note = "" if entry["qualification_ok"] else "  [qualification violated]"
                print(
                    f"        {entry['filter']:<18s} slope={entry['slope']:+.4f}"
                    f" (tail {entry['tail_slope']:+.4f})"
                    f"  rel.err={100 * entry['relative_error_vs_measured']:5.1f}%"
                    f"  R2={entry['r_squared']:.4f}  {verdict}{note}"
                )
    return results


#: Relative tolerance on the measured exponent.  Theorem 3.4 is an asymptotic
#: upper bound with unspecified constants, so a finite-sample experiment can
#: confirm the *order* of the rate, not the exponent to several digits: over a
#: bounded range of `n` the unknown constants leave residual curvature in the
#: log-log plot.  The agreement claimed here is therefore relative, with the
#: local slopes reported alongside so the trend can be inspected directly.
RATE_TOLERANCE = 0.15


def _summarize_rate_rows(
    rows: list[dict[str, Any]],
    config: RateConfig,
    model: SpectralOperatorModel,
    *,
    nominal: float,
    measured: float,
    tolerance: float = RATE_TOLERANCE,
) -> list[dict[str, Any]]:
    """Aggregate repeats and fit the rate for each filter."""
    summary: list[dict[str, Any]] = []
    for filter_name, filter_kwargs in config.filters:
        selected = [row for row in rows if row["filter"] == filter_name]
        sizes, means, stderrs = [], [], []
        for n_samples in config.n_grid:
            risks = np.array(
                [row["excess_risk"] for row in selected if row["n"] == n_samples]
            )
            # Average in log space: the fit regresses log risk, and the spread
            # across repeats is closer to multiplicative than additive.
            logs = np.log(risks)
            sizes.append(float(n_samples))
            means.append(float(np.exp(logs.mean())))
            stderrs.append(
                float(logs.std(ddof=1) / math.sqrt(logs.size)) if logs.size > 1 else 0.0
            )

        size_array = np.array(sizes)
        mean_array = np.array(means)
        fit = fit_power_law(size_array, mean_array)
        # Slopes between consecutive sample sizes: if the measurement is
        # approaching the asymptotic exponent, these drift toward it.
        local_slopes = list(
            np.diff(np.log(mean_array)) / np.diff(np.log(size_array))
        )
        tail_slope = (
            float(
                np.polyfit(
                    np.log(size_array[len(sizes) // 2 :]),
                    np.log(mean_array[len(sizes) // 2 :]),
                    1,
                )[0]
            )
            if len(sizes) >= 4
            else fit.slope
        )

        required = max(config.r, 1.0)
        qualification = make_filter(filter_name, 0.1, **filter_kwargs).qualification
        relative = abs(fit.slope + measured) / measured
        summary.append(
            {
                "filter": filter_name,
                "filter_options": dict(filter_kwargs),
                "qualification": qualification if math.isfinite(qualification) else "inf",
                "qualification_required": required,
                "qualification_ok": bool(qualification >= required),
                "n": sizes,
                "mean_excess_risk": means,
                "log_risk_stderr": stderrs,
                "slope": fit.slope,
                "slope_stderr": fit.slope_stderr,
                "slope_ci95": list(fit.confidence_interval()),
                "local_slopes": [float(value) for value in local_slopes],
                "tail_slope": tail_slope,
                "r_squared": fit.r_squared,
                "nominal_exponent": nominal,
                "measured_exponent": measured,
                "covers_nominal": fit.covers(-nominal),
                "covers_measured": fit.covers(-measured),
                "relative_error_vs_measured": relative,
                "relative_error_vs_nominal": abs(fit.slope + nominal) / nominal,
                "relative_error_tail_vs_measured": abs(tail_slope + measured) / measured,
                "tolerance": tolerance,
                "within_tolerance": bool(relative <= tolerance),
            }
        )
    return summary


# --------------------------------------------------------------------------- #
# Experiment 2: how many random features are needed
# --------------------------------------------------------------------------- #


def _ntk_target(
    features: ScalarNTKFeatures,
    input_sets: list[Array],
    rng: np.random.Generator,
) -> list[Array]:
    r"""A target function drawn from the NTK RKHS, normalized in :math:`L^2`.

    Drawing :math:`f^* = \sum_m c_m\varphi_m` with :math:`\|c\|=1` puts the
    target in :math:`\mathcal{H}` with unit RKHS norm, realizing the
    well-specified case :math:`r=1/2` of Assumption 3.2 - the regime the feature
    threshold :math:`M=O(\sqrt n\,p)` refers to.

    The values are then rescaled to unit empirical :math:`L^2` norm.  Without
    that step the experiment is meaningless: a unit-RKHS-norm coefficient vector
    spread over :math:`pM` directions has
    :math:`\|f^*\|_{L^2}^2 = c^\top\Sigma c \approx \mathrm{tr}(\Sigma)/(pM)`,
    which for :math:`pM\sim10^4` and :math:`\mathrm{tr}(\Sigma)=O(1)` is of order
    :math:`10^{-2}` - so a nominally modest noise level would drown the signal
    entirely and the measured error would be flat in :math:`M` for the wrong
    reason.  Normalizing makes ``noise_std`` a genuine noise-to-signal ratio.

    All input sets are evaluated against the same coefficients and divided by
    the same constant, so train and test see one common target function.
    """
    coefficients = rng.standard_normal(features.coefficient_dim)
    coefficients /= np.linalg.norm(coefficients)
    raw = [features.feature_tensor(inputs)[:, 0, :] @ coefficients for inputs in input_sets]
    scale = float(np.sqrt(np.mean(np.concatenate(raw) ** 2)))
    if scale == 0.0:
        raise ValueError("the drawn target vanishes; increase the number of features")
    return [values / scale for values in raw]


def _gradient_descent_path(
    design: Array,
    targets: Array,
    test_design: Array,
    test_targets: Array,
    checkpoints: list[int],
    step: float,
    scale: float,
) -> list[float]:
    r"""Run Landweber iteration once, recording the test error at checkpoints.

    Sweeping the number of iterations :math:`T` is naturally done along a single
    gradient descent trajectory: the estimator at iteration :math:`T` is
    :math:`\phi_{1/(\alpha T)}(\widehat\Sigma_M)\widehat{\mathcal{S}}^*_M
    \mathbf v`, so all values of :math:`T` are available from one run at the cost
    of the largest.  Each iteration costs :math:`O(nMp)`, matching the paper's
    :math:`O(nMt)` accounting for gradient descent.
    """
    n_samples = targets.shape[0]
    if scale <= 0.0:
        # A ReLU feature map can degenerate for very small M, when every drawn
        # neuron is inactive on the whole sample; the estimator is then the zero
        # predictor.
        zero = np.zeros_like(test_targets)
        return [excess_risk(zero, test_targets) for _ in checkpoints]
    rhs = design.T @ targets / (n_samples * scale)
    coefficients = np.zeros(design.shape[1])
    errors: list[float] = []
    next_checkpoint = 0
    for iteration in range(1, max(checkpoints) + 1):
        gradient = design.T @ (design @ coefficients) / (n_samples * scale) - rhs
        coefficients -= step * gradient
        while next_checkpoint < len(checkpoints) and checkpoints[next_checkpoint] == iteration:
            predictions = test_design @ coefficients
            errors.append(excess_risk(predictions, test_targets))
            next_checkpoint += 1
    return errors


#: Feature-count grid for the threshold study, in multiples of ``sqrt(n) * p``.
#: The largest entry serves as the plateau reference, so it must be well above
#: one while remaining affordable: at :math:`d=14` a multiplier of four already
#: means a design matrix with :math:`pM\approx4.6\times10^4` columns.
DEFAULT_FEATURE_MULTIPLIERS: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def run_feature_threshold(
    *,
    settings: tuple[tuple[int, int], ...] = ((1, 1250), (1, 2500), (1, 5000), (14, 500), (14, 1000), (14, 2000)),
    feature_multipliers: tuple[float, ...] = DEFAULT_FEATURE_MULTIPLIERS,
    iteration_grid: tuple[int, ...] = (8, 16, 32, 64),
    repeats: int = 4,
    n_test: int = 1500,
    noise_std: float = 0.2,
    step: float = 0.5,
    tolerance: float = 0.05,
    seed: int = 20260301,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""Recreate the feature-threshold study of Appendix A.3.

    The paper reports that for kernel gradient descent on the real-valued NTK,
    "once :math:`M` exceeds a threshold of order :math:`O(\sqrt n\,p)` and the
    number of GD iterations :math:`T` is fixed, further increasing :math:`M`
    does not lead to any improvement in the test error", with :math:`p = d+2`.
    That is a *sufficiency* claim, and it is what is tested here: the test error
    is mapped over :math:`(M,T)` with :math:`M` expressed in multiples of
    :math:`\sqrt n\,p`, and the error at multiplier one is compared with the
    plateau, taken at the largest multiplier on the grid.  The claim holds if
    the former is within ``tolerance`` of the latter.

    The location of the threshold is reported too - the smallest multiplier
    within tolerance of the plateau - together with how it scales in :math:`n`
    and :math:`d`.  That is a finer question than the paper's claim and the
    answer is grid-limited, so it is offered as a measurement rather than a
    test.

    The target is drawn from the NTK RKHS and rescaled to unit :math:`L^2` norm,
    so the problem is well-specified (:math:`r=1/2`, the regime the
    :math:`\sqrt n` threshold refers to) and ``noise_std`` is a true
    noise-to-signal ratio.  Sweeping :math:`T` costs no more than its largest
    value, since every iterate along one gradient descent trajectory is an
    estimator with :math:`\lambda = 1/(\alpha T)`.

    Deviations from the paper
        Appendix A.3 uses :math:`n=5000` with a Gaussian design at :math:`d=1`
        and a subset of SUSY at :math:`d=14`, averaged over 50 runs.  SUSY is
        not retrievable in this environment, so the :math:`d=14` design is
        Gaussian as well; the claim under test concerns the scaling in
        :math:`d` through :math:`p=d+2`, which is preserved.  Sample sizes at
        :math:`d=14` are capped at 2000 because the plateau reference needs
        :math:`M=4\sqrt n\,p`, i.e. a design matrix with :math:`pM` columns, and
        :math:`n=5000` would need one of several gigabytes.

    Parameters
    ----------
    settings:
        ``(input_dim, n)`` pairs to run.
    feature_multipliers:
        Grid of :math:`M/(\sqrt n\,p)`; the largest is the plateau reference.
    tolerance:
        Relative margin within which an error counts as "at the plateau".
    """
    rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    sufficiency: list[dict[str, Any]] = []
    multipliers = tuple(sorted(feature_multipliers))
    checkpoints = sorted(set(iteration_grid))

    for input_dim, n_samples in settings:
        n_summands = input_dim + 2
        reference = math.sqrt(n_samples) * n_summands
        feature_grid = sorted(
            {max(1, int(round(multiplier * reference))) for multiplier in multipliers}
        )
        accumulated = np.zeros((len(feature_grid), len(checkpoints)))
        baseline = 0.0

        for repeat in range(repeats):
            stream = np.random.default_rng([seed, input_dim, n_samples, repeat])
            train_inputs = stream.standard_normal((n_samples, input_dim))
            test_inputs = stream.standard_normal((n_test, input_dim))
            # A wide independent feature map defines the target, so the target
            # does not lie in the span of the features being fitted.
            target_features = ScalarNTKFeatures(input_dim, 4096, stream)
            train_clean, test_targets = _ntk_target(
                target_features, [train_inputs, test_inputs], stream
            )
            train_targets = train_clean + noise_std * stream.standard_normal(n_samples)
            baseline += excess_risk(np.zeros_like(test_targets), test_targets) / repeats

            for m_index, n_features in enumerate(feature_grid):
                features = ScalarNTKFeatures(input_dim, n_features, stream)
                design = features.design_matrix(train_inputs)
                test_design = features.design_matrix(test_inputs)
                # Normalizing by the trace keeps one step size stable for every
                # M and d on the grid.
                scale = float((design**2).sum() / n_samples)
                errors = _gradient_descent_path(
                    design,
                    train_targets,
                    test_design,
                    test_targets,
                    checkpoints,
                    step,
                    scale,
                )
                accumulated[m_index] += np.array(errors) / repeats

        for m_index, n_features in enumerate(feature_grid):
            for t_index, iterations in enumerate(checkpoints):
                rows.append(
                    {
                        "input_dim": input_dim,
                        "n_summands": n_summands,
                        "n": n_samples,
                        "n_features": n_features,
                        "feature_multiplier": n_features / reference,
                        "iterations": iterations,
                        "mean_test_error": float(accumulated[m_index, t_index]),
                    }
                )

        unit_index = int(np.argmin(np.abs(np.array(multipliers) - 1.0)))
        grid_array = np.array(feature_grid, dtype=float)
        for t_index, iterations in enumerate(checkpoints):
            column = accumulated[:, t_index]
            plateau = float(column[-1])
            at_unit = float(column[unit_index])
            within = np.nonzero(column <= plateau * (1.0 + tolerance))[0]
            index = int(within[0]) if within.size else len(feature_grid) - 1
            interpolated = _interpolate_threshold(
                grid_array, column, plateau * (1.0 + tolerance)
            )
            sufficiency.append(
                {
                    "input_dim": input_dim,
                    "n_summands": n_summands,
                    "n": n_samples,
                    "iterations": iterations,
                    "features_at_unit_multiplier": feature_grid[unit_index],
                    "features_at_plateau": feature_grid[-1],
                    "error_at_unit_multiplier": at_unit,
                    "plateau_error": plateau,
                    "excess_over_plateau": at_unit / plateau - 1.0,
                    "tolerance": tolerance,
                    "sqrt_n_p_suffices": bool(at_unit <= plateau * (1.0 + tolerance)),
                }
            )
            thresholds.append(
                {
                    "input_dim": input_dim,
                    "n_summands": n_summands,
                    "n": n_samples,
                    "iterations": iterations,
                    "baseline_error": baseline,
                    "plateau_error": plateau,
                    "threshold_features": feature_grid[index],
                    "threshold_multiplier": feature_grid[index] / reference,
                    "threshold_over_sqrt_n": feature_grid[index] / math.sqrt(n_samples),
                    "threshold_features_interpolated": interpolated,
                    "threshold_multiplier_interpolated": interpolated / reference,
                }
            )

        if verbose:
            relevant = [
                entry
                for entry in sufficiency
                if entry["input_dim"] == input_dim and entry["n"] == n_samples
            ]
            excesses = [entry["excess_over_plateau"] for entry in relevant]
            passed = sum(entry["sqrt_n_p_suffices"] for entry in relevant)
            located = [
                entry["threshold_multiplier"]
                for entry in thresholds
                if entry["input_dim"] == input_dim and entry["n"] == n_samples
            ]
            print(
                f"[features] d={input_dim:<3d} p={n_summands:<3d} n={n_samples:<6d} "
                f"sqrt(n)p={reference:7.0f}  error at M=sqrt(n)p exceeds plateau by "
                f"{100 * max(excesses):+5.1f}% (worst over T); "
                f"{passed}/{len(relevant)} within {100 * tolerance:.0f}%; "
                f"threshold at {np.median(located):.3f} x sqrt(n)p"
            )

    return {
        "settings": {
            "cases": [list(pair) for pair in settings],
            "feature_multipliers": list(multipliers),
            "iteration_grid": list(checkpoints),
            "repeats": repeats,
            "n_test": n_test,
            "noise_std": noise_std,
            "step": step,
            "tolerance": tolerance,
            "target": "drawn from the NTK RKHS, unit L2 norm (well-specified, r = 1/2)",
            "d14_design": "Gaussian; SUSY not retrievable in this environment",
        },
        "grid": rows,
        "sufficiency": sufficiency,
        "thresholds": thresholds,
        "scaling": _threshold_scaling(thresholds),
        "verdict": {
            "cases_tested": len(sufficiency),
            "cases_where_sqrt_n_p_suffices": sum(
                entry["sqrt_n_p_suffices"] for entry in sufficiency
            ),
            "worst_excess_over_plateau": max(
                entry["excess_over_plateau"] for entry in sufficiency
            ),
        },
    }


def _interpolate_threshold(features: Array, errors: Array, level: float) -> float:
    r"""Locate where the error curve crosses ``level``, interpolating in log-log.

    Reading the threshold off the grid directly quantizes it to the grid points,
    which makes any subsequent fit of its scaling in :math:`n` a fit to a
    staircase.  Interpolating between the last grid point above ``level`` and
    the first below it gives a continuous estimate.  If the curve is already
    below ``level`` at the smallest :math:`M`, the threshold is reported as that
    smallest value, i.e. as an upper bound.
    """
    below = errors <= level
    if below[0]:
        return float(features[0])
    if not below.any():
        return float(features[-1])
    first = int(np.argmax(below))
    x_lo, x_hi = math.log(features[first - 1]), math.log(features[first])
    y_lo, y_hi = math.log(errors[first - 1]), math.log(errors[first])
    if y_hi == y_lo:
        return float(features[first])
    weight = (math.log(level) - y_lo) / (y_hi - y_lo)
    return float(math.exp(x_lo + weight * (x_hi - x_lo)))


def _threshold_scaling(thresholds: list[dict[str, Any]]) -> dict[str, Any]:
    r"""Measure how the located threshold scales with :math:`n` and with :math:`p`.

    The dependence on :math:`n` is a power law and is fitted; the dependence on
    :math:`p` is a proportionality and is reported as the ratio of thresholds
    between input dimensions at matched :math:`n`.  Both are grid-limited: the
    located threshold can only take the values on the feature grid, so the
    fitted slopes are coarse.
    """
    scaling: dict[str, Any] = {"per_dimension": [], "dimension_ratio": []}
    dims = sorted({entry["input_dim"] for entry in thresholds})

    def median_for(input_dim: int, n_samples: int) -> float:
        values = [
            entry["threshold_features"]
            for entry in thresholds
            if entry["input_dim"] == input_dim and entry["n"] == n_samples
        ]
        return float(np.median(values)) if values else float("nan")

    for input_dim in dims:
        sizes = sorted(
            {entry["n"] for entry in thresholds if entry["input_dim"] == input_dim}
        )
        medians = [median_for(input_dim, n_samples) for n_samples in sizes]
        entry: dict[str, Any] = {
            "input_dim": input_dim,
            "n": sizes,
            "median_threshold": medians,
        }
        if len(sizes) >= 3:
            fit = fit_power_law(np.array(sizes, dtype=float), np.array(medians))
            entry.update(
                {
                    "slope_in_n": fit.slope,
                    "slope_stderr": fit.slope_stderr,
                    "slope_ci95": list(fit.confidence_interval()),
                    "covers_one_half": fit.covers(0.5),
                    "r_squared": fit.r_squared,
                }
            )
        scaling["per_dimension"].append(entry)

    if len(dims) >= 2:
        low, high = dims[0], dims[-1]
        shared = sorted(
            {entry["n"] for entry in thresholds if entry["input_dim"] == low}
            & {entry["n"] for entry in thresholds if entry["input_dim"] == high}
        )
        for n_samples in shared:
            scaling["dimension_ratio"].append(
                {
                    "n": n_samples,
                    "input_dims": [low, high],
                    "threshold_ratio": median_for(high, n_samples)
                    / median_for(low, n_samples),
                    "p_ratio": (high + 2) / (low + 2),
                }
            )
    return scaling


# --------------------------------------------------------------------------- #
# Experiment 3: wall-clock against the exact operator-valued kernel
# --------------------------------------------------------------------------- #


def _pde_setup(
    dataset_name: str, n_points: int, n_train: int, n_test: int, noise_std: float, seed: int
) -> tuple[OperatorDataset, Array, Array, Array, Array]:
    dataset = make_dataset(dataset_name, n_points=n_points, noise_std=noise_std)
    rng = np.random.default_rng(seed)
    train = dataset.sample(n_train, rng)
    test = dataset.sample(n_test, rng)
    return (
        dataset,
        dataset.lift(train.fields),
        train.outputs,
        dataset.lift(test.fields),
        test.targets,
    )


class _BenchmarkTask:
    """A task presented identically to the exact and random feature estimators.

    Bundles the data source, the exact operator-valued kernel, and the matching
    random feature map, so the two estimators differ only in which operator the
    filter is applied to.
    """

    name: str
    n_summands: int
    output_dim: int
    spectral_scale: float

    def train(self, n_samples: int, seed: int) -> tuple[Array, Array]:
        raise NotImplementedError

    def test(self) -> tuple[Array, Array]:
        raise NotImplementedError

    def exact_kernel(self) -> Any:
        raise NotImplementedError

    def features(self, n_features: int, seed: int) -> Any:
        raise NotImplementedError


class _SpectralTask(_BenchmarkTask):
    r"""The synthetic instance of :mod:`kerop.data.spectral`.

    Here :math:`p=1`, so the random feature coefficient space has dimension
    :math:`M` rather than :math:`pM`, and the exponents :math:`(r,b)` are known,
    which makes it the cleanest setting in which to compare costs.
    """

    def __init__(
        self,
        r: float = 0.5,
        b: float = 0.5,
        n_modes: int = 2048,
        output_dim: int = 6,
        noise_std: float = 0.05,
        n_test: int = 3000,
        seed: int = 20260301,
    ) -> None:
        self.name = f"spectral(r={r},b={b})"
        self.model = SpectralOperatorModel(
            r=r,
            b=b,
            n_modes=n_modes,
            output_dim=output_dim,
            noise_std=noise_std,
            seed=seed,
        )
        self.n_summands = 1
        self.output_dim = output_dim
        self.spectral_scale = self.model.kappa_squared()
        self._test = self.model.test_set(n_test, np.random.default_rng([seed, 31337]))

    def train(self, n_samples: int, seed: int) -> tuple[Array, Array]:
        return self.model.sample(n_samples, np.random.default_rng([seed, n_samples]))

    def test(self) -> tuple[Array, Array]:
        return self._test

    def exact_kernel(self) -> Any:
        return self.model.kernel()

    def features(self, n_features: int, seed: int) -> Any:
        return self.model.features(n_features, np.random.default_rng([seed, n_features]))


class _PDETask(_BenchmarkTask):
    r"""A PDE solution operator learned with the operator-valued NTK.

    Here :math:`p = 1+\tilde d`, so the coefficient space has dimension
    :math:`pM`; this is the realistic operator-learning setting and the one
    where the :math:`\tilde d^2` dependence discussed in Section 3.2 bites.
    """

    def __init__(
        self,
        dataset_name: str = "darcy",
        n_points: int = 12,
        noise_std: float = 0.02,
        n_test: int = 500,
        seed: int = 20260301,
    ) -> None:
        self.name = f"{dataset_name}(n_x={n_points})"
        self.dataset = make_dataset(dataset_name, n_points=n_points, noise_std=noise_std)
        self.n_summands = self.dataset.n_summands
        self.output_dim = self.dataset.n_points
        self.spectral_scale = "power"  # type: ignore[assignment]
        self._seed = seed
        test = self.dataset.sample(n_test, np.random.default_rng([seed, 31337]))
        self._test = (self.dataset.lift(test.fields), test.targets)

    def train(self, n_samples: int, seed: int) -> tuple[Array, Array]:
        batch = self.dataset.sample(n_samples, np.random.default_rng([seed, n_samples]))
        return self.dataset.lift(batch.fields), batch.outputs

    def test(self) -> tuple[Array, Array]:
        return self._test

    def exact_kernel(self) -> Any:
        return OperatorNTKKernel(
            self.dataset.feature_dim,
            self.dataset.n_points,
            output_scale=self.dataset.output_scale(),
        )

    def features(self, n_features: int, seed: int) -> Any:
        return OperatorNTKFeatures(
            self.dataset.feature_dim,
            self.dataset.n_points,
            n_features,
            np.random.default_rng([seed, n_features]),
            output_scale=self.dataset.output_scale(),
        )


def _sweep_exact(
    task: _BenchmarkTask,
    train_sizes: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    seed: int,
) -> list[dict[str, Any]]:
    r"""Fit the exact estimator at every sample size and :math:`\lambda`.

    The Gram matrix is re-formed for each :math:`\lambda`, but the *reported*
    per-fit time is what a single fit costs, so the baseline is not charged for
    the sweep.  A practitioner tuning :math:`\lambda` would reuse the
    factorization; the frontier below therefore uses single-fit times, which is
    the conservative choice for the claim being made.
    """
    test_inputs, test_targets = task.test()
    kernel = task.exact_kernel()
    rows: list[dict[str, Any]] = []
    for n_train in train_sizes:
        train_inputs, train_outputs = task.train(n_train, seed)
        for lam in lambda_grid:
            estimator = ExactOperatorFilter(
                kernel, "tikhonov", lam, spectral_scale=task.spectral_scale
            )
            estimator.fit(train_inputs, train_outputs)
            rows.append(
                {
                    "method": "exact",
                    "label": f"exact tikhonov(lam={lam:.0e})",
                    "n_train": n_train,
                    "lambda": lam,
                    "excess_risk": excess_risk(estimator.predict(test_inputs), test_targets),
                    "fit_seconds": estimator.report.fit_seconds,
                    "assemble_seconds": estimator.report.assemble_seconds,
                    "solve_seconds": estimator.report.solve_seconds,
                    "operator_dim": estimator.report.operator_dim,
                    "operator_megabytes": estimator.report.peak_operator_bytes / 1e6,
                }
            )
    return rows


def _sweep_random_features(
    task: _BenchmarkTask,
    train_sizes: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    feature_multipliers: tuple[float, ...],
    iteration_grid: tuple[int, ...],
    seed: int,
) -> list[dict[str, Any]]:
    r"""Fit the random feature estimator over sample sizes, :math:`M`, and filters.

    :math:`M` is taken as multiples of :math:`\sqrt n\,p`, the well-specified
    prescription of Theorem 3.4, so the multiplier stands in for the theorem's
    unspecified constant :math:`\tilde C`.
    """
    test_inputs, test_targets = task.test()
    rows: list[dict[str, Any]] = []
    for n_train in train_sizes:
        train_inputs, train_outputs = task.train(n_train, seed)
        reference = math.sqrt(n_train) * task.n_summands
        feature_grid = sorted(
            {max(8, int(round(multiplier * reference))) for multiplier in feature_multipliers}
        )
        for n_features in feature_grid:
            features = task.features(n_features, seed)
            variants: list[tuple[str, Any, str]] = [
                ("tikhonov", lam, f"rf tikhonov(lam={lam:.0e})") for lam in lambda_grid
            ]
            variants += [
                ("landweber", steps, f"rf landweber(T={steps})") for steps in iteration_grid
            ]
            for filter_name, parameter, label in variants:
                if filter_name == "tikhonov":
                    estimator = VectorValuedRFRegressor(
                        features,
                        "tikhonov",
                        float(parameter),
                        spectral_scale=task.spectral_scale,
                        assemble=True,
                    )
                else:
                    estimator = VectorValuedRFRegressor(
                        features,
                        filter_obj=Landweber.from_iterations(int(parameter)),
                        spectral_scale=task.spectral_scale,
                    )
                estimator.fit(train_inputs, train_outputs)
                rows.append(
                    {
                        "method": "random_features",
                        "label": f"{label}, M={n_features}",
                        "n_train": n_train,
                        "n_features": n_features,
                        "feature_multiplier": n_features / reference,
                        "filter": filter_name,
                        "excess_risk": excess_risk(
                            estimator.predict(test_inputs), test_targets
                        ),
                        "fit_seconds": estimator.report.fit_seconds,
                        "assemble_seconds": estimator.report.assemble_seconds,
                        "solve_seconds": estimator.report.solve_seconds,
                        "operator_dim": estimator.report.operator_dim,
                        "operator_megabytes": estimator.report.peak_operator_bytes / 1e6,
                        "matrix_free": bool(estimator.report.extras["matrix_free"]),
                    }
                )
    return rows


def _cheapest_reaching(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    """The fastest configuration in ``rows`` whose excess risk is at most ``target``."""
    candidates = [row for row in rows if row["excess_risk"] <= target]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["fit_seconds"])


def run_walltime_benchmark(
    *,
    task: str = "spectral",
    train_sizes: tuple[int, ...] = (250, 500, 1000, 2000),
    lambda_grid: tuple[float, ...] = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5),
    feature_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    iteration_grid: tuple[int, ...] = (64, 256),
    n_targets: int = 5,
    seed: int = 20260301,
    task_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    r"""Compare random features against exact operator-valued kernel regression.

    Both estimators use the *same* operator-valued kernel - the exact one in
    closed form for the baseline, its random feature approximation for the other
    - and the same filter implementations, so the comparison isolates the effect
    of the approximation.  The exact estimator forms and factorizes the block
    Gram matrix of size :math:`nd_v\times nd_v`, at :math:`O((nd_v)^3)` time and
    :math:`O((nd_v)^2)` memory; the random feature estimator works with a
    :math:`pM`-dimensional operator at :math:`O(nd_v(pM)^2)` time and
    :math:`O(nd_vpM)` memory.

    Two comparisons are reported, because they answer different questions and
    give different answers.

    *Matched sample size.*  At a fixed :math:`n`, which method reaches the lower
    excess risk?  The exact method does, and by a margin that closes only slowly
    in :math:`M`: the random feature approximation contributes an error of order
    :math:`M^{-1/2}` that no amount of regularization removes.  Theorem 3.4
    preserves the *rate*, not the constant, so this is the expected outcome and
    is reported plainly.

    *Matched excess risk.*  Which method reaches a given error level in less
    wall-clock time, each free to choose its own sample size and
    hyper-parameters?  This is the question that matters in practice and the one
    the paper's scalability claim addresses, since the random feature method can
    afford far more data at equal cost.  Both methods are swept over the same
    grid of sample sizes and their own hyper-parameters, a cost/accuracy
    frontier is built for each, and the speed-up is read off at several target
    risk levels drawn from the range both methods can reach.

    Parameters
    ----------
    task:
        ``"spectral"`` for the synthetic instance with known :math:`(r,b)` and
        :math:`p=1`, or ``"poisson"``/``"darcy"`` for a PDE solution operator
        learned with the operator-valued NTK, where :math:`p=1+\tilde d`.
    train_sizes:
        Sample sizes offered to *both* methods.
    n_targets:
        Number of target risk levels at which to report the speed-up.
    task_kwargs:
        Forwarded to the task constructor.
    """
    options = dict(task_kwargs or {})
    if task == "spectral":
        benchmark: _BenchmarkTask = _SpectralTask(seed=seed, **options)
    elif task in {"poisson", "darcy"}:
        benchmark = _PDETask(dataset_name=task, seed=seed, **options)
    else:
        raise KeyError(f"unknown task {task!r}; expected 'spectral', 'poisson' or 'darcy'")

    test_inputs, test_targets = benchmark.test()
    baseline_risk = excess_risk(np.zeros_like(test_targets), test_targets)

    exact_rows = _sweep_exact(benchmark, train_sizes, lambda_grid, seed)
    rf_rows = _sweep_random_features(
        benchmark, train_sizes, lambda_grid, feature_multipliers, iteration_grid, seed
    )

    # Matched sample size: best risk each method can reach at each n.
    matched_size: list[dict[str, Any]] = []
    for n_train in train_sizes:
        best_exact = min(
            (row for row in exact_rows if row["n_train"] == n_train),
            key=lambda row: row["excess_risk"],
        )
        best_rf = min(
            (row for row in rf_rows if row["n_train"] == n_train),
            key=lambda row: row["excess_risk"],
        )
        matched_size.append(
            {
                "n_train": n_train,
                "exact_best_risk": best_exact["excess_risk"],
                "exact_seconds": best_exact["fit_seconds"],
                "exact_operator_dim": best_exact["operator_dim"],
                "exact_megabytes": best_exact["operator_megabytes"],
                "rf_best_risk": best_rf["excess_risk"],
                "rf_seconds": best_rf["fit_seconds"],
                "rf_label": best_rf["label"],
                "rf_operator_dim": best_rf["operator_dim"],
                "rf_megabytes": best_rf["operator_megabytes"],
                "risk_ratio_rf_over_exact": best_rf["excess_risk"] / best_exact["excess_risk"],
                "speedup_at_this_n": best_exact["fit_seconds"] / best_rf["fit_seconds"],
            }
        )

    # Matched excess risk: the frontier both methods can reach.
    reachable_exact = min(row["excess_risk"] for row in exact_rows)
    reachable_rf = min(row["excess_risk"] for row in rf_rows)
    worst_useful = min(
        max(row["excess_risk"] for row in exact_rows if row["n_train"] == min(train_sizes)),
        baseline_risk,
    )
    lowest = max(reachable_exact, reachable_rf)
    highest = max(lowest * 2.0, min(worst_useful, lowest * 4.0))
    targets = np.geomspace(lowest, highest, n_targets)

    frontier: list[dict[str, Any]] = []
    for target in targets:
        exact_choice = _cheapest_reaching(exact_rows, float(target))
        rf_choice = _cheapest_reaching(rf_rows, float(target))
        frontier.append(
            {
                "target_risk": float(target),
                "target_relative_error": float(target) / baseline_risk,
                "exact": exact_choice,
                "random_features": rf_choice,
                "speedup": (
                    exact_choice["fit_seconds"] / rf_choice["fit_seconds"]
                    if exact_choice and rf_choice
                    else None
                ),
                # The most conservative reading available: the exact method is
                # charged only for its factorization, as though its Gram matrix
                # were free, while the random feature method pays for
                # everything.  Reported so the claim can be seen not to rest on
                # the cost of assembling the Gram matrix.
                "speedup_exact_solve_only": (
                    exact_choice["solve_seconds"] / rf_choice["fit_seconds"]
                    if exact_choice and rf_choice
                    else None
                ),
                "memory_ratio": (
                    exact_choice["operator_megabytes"] / rf_choice["operator_megabytes"]
                    if exact_choice and rf_choice
                    else None
                ),
            }
        )

    speedups = [entry["speedup"] for entry in frontier if entry["speedup"] is not None]
    conservative = [
        entry["speedup_exact_solve_only"]
        for entry in frontier
        if entry["speedup_exact_solve_only"] is not None
    ]
    verdict = {
        "baseline_risk_zero_predictor": baseline_risk,
        "n_targets_with_both_methods": len(speedups),
        "median_speedup_at_matched_risk": float(np.median(speedups)) if speedups else None,
        "min_speedup_at_matched_risk": float(np.min(speedups)) if speedups else None,
        "max_speedup_at_matched_risk": float(np.max(speedups)) if speedups else None,
        "median_speedup_exact_solve_only": (
            float(np.median(conservative)) if conservative else None
        ),
        "min_speedup_exact_solve_only": float(np.min(conservative)) if conservative else None,
        "random_features_faster_at_every_target": bool(
            speedups and all(value > 1.0 for value in speedups)
        ),
        "random_features_faster_at_every_target_solve_only": bool(
            conservative and all(value > 1.0 for value in conservative)
        ),
    }

    if verbose:
        print(f"[walltime] task={benchmark.name}  p={benchmark.n_summands}  d_v={benchmark.output_dim}")
        print("           matched sample size (exact wins on risk, as expected):")
        for entry in matched_size:
            print(
                f"             n={entry['n_train']:<5d} "
                f"exact risk={entry['exact_best_risk']:.5f} in {entry['exact_seconds']:6.2f}s "
                f"(dim {entry['exact_operator_dim']:>6d}) | "
                f"RF risk={entry['rf_best_risk']:.5f} in {entry['rf_seconds']:6.3f}s "
                f"(dim {entry['rf_operator_dim']:>6d})"
            )
        print("           matched excess risk (each method picks its own n):")
        for entry in frontier:
            if entry["speedup"] is None:
                continue
            print(
                f"             target={entry['target_risk']:.5f}  "
                f"exact: n={entry['exact']['n_train']:<5d} {entry['exact']['fit_seconds']:6.2f}s  |  "
                f"RF: n={entry['random_features']['n_train']:<5d} "
                f"M={entry['random_features']['n_features']:<5d} "
                f"{entry['random_features']['fit_seconds']:6.3f}s  ->  "
                f"{entry['speedup']:5.1f}x faster, {entry['memory_ratio']:5.1f}x less memory"
            )
        if verdict["median_speedup_at_matched_risk"] is not None:
            print(
                f"           median speed-up at matched risk: "
                f"{verdict['median_speedup_at_matched_risk']:.1f}x"
            )

    return {
        "settings": {
            "task": task,
            "task_name": benchmark.name,
            "task_kwargs": options,
            "n_summands": benchmark.n_summands,
            "output_dim": benchmark.output_dim,
            "train_sizes": list(train_sizes),
            "lambda_grid": list(lambda_grid),
            "feature_multipliers": list(feature_multipliers),
            "iteration_grid": list(iteration_grid),
            "seed": seed,
        },
        "exact": exact_rows,
        "random_features": rf_rows,
        "matched_sample_size": matched_size,
        "matched_excess_risk": frontier,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Filter diagnostics
# --------------------------------------------------------------------------- #


def run_filter_report(
    *,
    families: tuple[tuple[str, dict[str, Any]], ...] = (
        ("tikhonov", {}),
        ("iterated_tikhonov", {"order": 2}),
        ("iterated_tikhonov", {"order": 3}),
        ("landweber", {}),
        ("cutoff", {}),
        ("heavy_ball", {"momentum": 0.9}),
        ("nu_method", {"nu": 1.0}),
        ("nu_method", {"nu": 2.0}),
        ("nu_method", {"nu": 3.0}),
    ),
    saturation_source_exponents: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    verbose: bool = True,
) -> dict[str, Any]:
    r"""Measure the Definition 2.2 constants, qualification, and saturation.

    Two things are reported.  First, for each family, the constants
    :math:`D, E, c_0` of (2.7)-(2.9) and the qualification :math:`\nu` of (2.10),
    measured numerically and compared with the value known analytically.

    Second, the *consequence* of qualification, on the exact bias
    :math:`\|r_\lambda(\mathcal{L})G_\rho\|` of the synthetic instance.  This is
    where saturation is visible cleanly: a family of qualification :math:`\nu`
    cannot do better than :math:`\lambda^{\nu}` no matter how smooth the target
    is, so for :math:`r>\nu` the measured bias exponent sticks at :math:`\nu`
    instead of following :math:`r`.  Since the bias is computed from the exact
    spectrum, this involves no sampling and isolates the filter's contribution.
    """
    from kerop.filters import filter_diagnostics, measure_qualification

    rows: list[dict[str, Any]] = []
    for name, options in families:
        diagnostics = filter_diagnostics(name, **options)
        report = measure_qualification(name, q_grid=np.arange(0.5, 4.01, 0.5), **options)
        analytic = make_filter(name, 0.1, **options).qualification
        rows.append(
            {
                "filter": name,
                "options": dict(options),
                "D": diagnostics.D,
                "E": diagnostics.E,
                "c0": diagnostics.c0,
                "qualification_measured": report.nu_estimate,
                "qualification_analytic": analytic if math.isfinite(analytic) else "inf",
                "qualification_probe_max": 4.0,
                "saturation_slope_estimate": (
                    report.saturation_estimate()
                    if math.isfinite(report.saturation_estimate())
                    else "inf"
                ),
                "c_q": {str(q): value for q, value in report.constants.items()},
                "growth_in_lambda": {str(q): value for q, value in report.growth.items()},
            }
        )
        if verbose:
            print(
                f"[filters] {name + str(options if options else ''):<30s} "
                f"D={diagnostics.D:.3f} E={diagnostics.E:.3f} c0={diagnostics.c0:.3f} "
                f"nu_measured={report.nu_estimate:.2f} nu_analytic={analytic}"
            )

    saturation: list[dict[str, Any]] = []
    for source in saturation_source_exponents:
        model = SpectralOperatorModel(r=source, b=0.5, n_modes=4096, output_dim=6, seed=0)
        low, high = model.usable_lambda_window()
        lambdas = np.logspace(np.log10(low), np.log10(high), 15)
        for name, options in families:
            fit = model.source_exponent_fit(lambdas, filter_name=name, **options)
            qualification = make_filter(name, 0.1, **options).qualification
            # An iterative filter realizes lambda through its iteration count, so
            # at the top of the probe window it may be running only a handful of
            # steps.  Its residual has then not reached the asymptotic form the
            # qualification describes, and the fitted bias exponent is
            # contaminated; heavy-ball, whose momentum transient decays slowly,
            # is the case where this bites.
            iteration_counts = [
                getattr(make_filter(name, float(lam), **options), "iterations", None)
                for lam in lambdas
            ]
            realized = [count for count in iteration_counts if count is not None]
            min_iterations = min(realized) if realized else None
            transient_limited = min_iterations is not None and min_iterations < 20
            saturation.append(
                {
                    "source_exponent_r": source,
                    "filter": name,
                    "options": dict(options),
                    "qualification": qualification if math.isfinite(qualification) else "inf",
                    "measured_bias_exponent": fit.slope,
                    "r_squared": fit.r_squared,
                    "expected_exponent": min(source, qualification),
                    "saturated": bool(source > qualification),
                    "min_iterations_over_probe": min_iterations,
                    "transient_limited": bool(transient_limited),
                }
            )
        if verbose:
            entries = [row for row in saturation if row["source_exponent_r"] == source]
            summary = ", ".join(
                f"{row['filter']}={row['measured_bias_exponent']:.2f}"
                for row in entries
                if row["filter"] in {"tikhonov", "landweber", "nu_method"}
            )
            print(f"[filters] bias exponent at r={source}: {summary}")

    return {"families": rows, "bias_saturation": saturation}


def rate_results_to_dicts(results: list[RateResult]) -> list[dict[str, Any]]:
    """Convert :class:`RateResult` objects to JSON-serializable dictionaries."""
    return [asdict(result) for result in results]