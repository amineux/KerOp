r"""On-disk filter contract for a downstream inverse-problem consumer.

KerOp's wall-time bar compares random-feature spectral filtering against exact
operator-valued kernel ridge regression.  SpecInv
(https://github.com/amineux/SpecInv) applies spectral filters to a *forward*
spectrum.  This module writes the quantities that handshake needs — eigenvalues
or the empirical random-feature spectrum, plus the seed, sample size, feature
count, and the settings that reproduce the existing bar — as a versioned
``.npz`` plus a JSON sidecar.

The files are self-describing.  A consumer loads them with NumPy and the
standard library; it does not import this package.  The on-disk layout is
documented in ``docs/FILTER_CONTRACT.md``.

This export does not change the scientific claim of the bar: the committed
quick-run medians remain ~28x (spectral) and ~3.6x (Poisson) RF vs exact KRR
at matched excess risk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kerop import __version__
from kerop.data.pde import PoissonDataset
from kerop.data.spectral import SpectralOperatorModel
from kerop.features import OperatorNTKFeatures

Array = NDArray[np.float64]

CONTRACT_VERSION = 1
SCHEMA_ID = "kerop.filter_contract/v1"

#: Seed used by ``scripts/run_walltime_benchmark.py`` and the committed
#: ``results/walltime_*.json`` records.
BAR_SEED = 20260301

#: Sample size used for the per-operator RF spectrum in the artifact.  It is
#: the smallest ``n`` on the committed quick wall-time grid; the full grid
#: lives in :data:`BAR_REPRODUCE`.
BAR_N = 150

#: Feature multiplier relative to :math:`\sqrt{n}\,p` (Theorem 3.4, well-
#: specified case).  The committed bar also sweeps 0.5 and 2.0; those values
#: are recorded in metadata, not re-run here.
BAR_FEATURE_MULTIPLIER = 1.0

#: Recorded medians from the committed quick wall-time JSONs.  Copied so a
#: consumer can check it is looking at the same bar; not a new measurement.
RECORDED_BAR: dict[str, Any] = {
    "protocol": (
        "quick wall-time vs exact operator-valued KRR at matched excess risk "
        "(scripts/run_walltime_benchmark.py --quick)"
    ),
    "spectral": {
        "results_file": "results/walltime_spectral.json",
        "median_speedup_at_matched_risk": 27.71727018410563,
        "min_speedup_at_matched_risk": 24.668054198508425,
        "max_speedup_at_matched_risk": 126.04174549573541,
    },
    "poisson": {
        "results_file": "results/walltime_poisson.json",
        "median_speedup_at_matched_risk": 3.55604718390518,
        "min_speedup_at_matched_risk": 1.594794749678986,
        "max_speedup_at_matched_risk": 44.17186286847536,
    },
}

BAR_REPRODUCE: dict[str, Any] = {
    "script": "scripts/run_walltime_benchmark.py",
    "cli": "python scripts/run_walltime_benchmark.py --quick --task {task}",
    "seed": BAR_SEED,
    "train_sizes": [150, 300, 600],
    "feature_multipliers": [0.5, 1.0, 2.0],
    "iteration_grid": [64, 256],
    "spectral": {
        "task_kwargs": {"n_modes": 512, "n_test": 1000},
        "lambda_grid": [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 3e-05],
        "output_dim": 6,
        "n_summands": 1,
        "r": 0.5,
        "b": 0.5,
    },
    "poisson": {
        "task_kwargs": {"n_points": 12},
        "lambda_grid": [0.01, 0.0001, 1e-05, 1e-06, 1e-07],
        "output_dim": 12,
        "n_summands": 9,
        "noise_std": 0.02,
    },
}

__all__ = [
    "BAR_SEED",
    "CONTRACT_VERSION",
    "SCHEMA_ID",
    "FilterContractBundle",
    "OperatorSpectrum",
    "bar_operator_ids",
    "build_bar_bundle",
    "build_operator_spectrum",
    "load_filter_contract",
    "write_filter_contract",
]


def _descending(values: Array) -> Array:
    return np.sort(np.asarray(values, dtype=float).reshape(-1))[::-1].copy()


def _feature_count(n: int, n_summands: int, multiplier: float = BAR_FEATURE_MULTIPLIER) -> int:
    """``M`` as in the wall-time sweep: a multiple of ``sqrt(n) * p``, at least 8."""
    return max(8, round(multiplier * math.sqrt(n) * n_summands))


def _rf_spectrum(design: Array, n_samples: int) -> Array:
    """Eigenvalues of ``Z.T @ Z / n``, descending.

    This is the spectrum of :math:`\\widehat\\Sigma_M` that the random-feature
    estimator actually filters.  Computed from the design matrix so the export
    does not depend on estimator internals.
    """
    gram = design.T @ design / float(n_samples)
    # eigvalsh is Hermitian; clip tiny negatives from roundoff.
    evals = np.linalg.eigvalsh(gram)
    return _descending(np.maximum(evals, 0.0))


@dataclass(frozen=True)
class OperatorSpectrum:
    """Forward spectrum of one operator that appears in the KerOp bar.

    Attributes
    ----------
    operator_id:
        Stable string identifier, e.g. ``kerop.spectral``.
    seed, n, n_features:
        Random seed, training sample size, and random-feature count used for
        ``rf_spectrum``.
    eigenvalues:
        Population / closed-form forward spectrum, descending.
    rf_spectrum:
        Eigenvalues of the empirical RF covariance :math:`\\widehat\\Sigma_M`
        at ``(n, n_features, seed)``, descending.
    metadata:
        Extra fields written into the JSON sidecar (geometry, bar settings).
    """

    operator_id: str
    seed: int
    n: int
    n_features: int
    eigenvalues: Array
    rf_spectrum: Array
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FilterContractBundle:
    """Versioned collection of operator spectra written as one npz + JSON pair."""

    version: int
    operators: tuple[OperatorSpectrum, ...]

    def by_id(self, operator_id: str) -> OperatorSpectrum:
        for item in self.operators:
            if item.operator_id == operator_id:
                return item
        raise KeyError(
            f"unknown operator_id {operator_id!r}; "
            f"available: {[item.operator_id for item in self.operators]}"
        )


def bar_operator_ids() -> tuple[str, ...]:
    """Identifiers of the two operators that constitute the committed bar."""
    return ("kerop.spectral", "kerop.poisson")


def build_operator_spectrum(
    operator: str,
    *,
    n: int = BAR_N,
    seed: int = BAR_SEED,
    feature_multiplier: float = BAR_FEATURE_MULTIPLIER,
    include_rf_spectrum: bool = True,
) -> OperatorSpectrum:
    """Build the forward spectrum for ``spectral`` or ``poisson``.

    Construction and RNG streams match :mod:`kerop.experiments` so that
    ``(seed, n, M)`` reproduces the random-feature draw used by the wall-time
    bar.  The population eigenvalues do not depend on the training sample.
    """
    name = operator.lower().removeprefix("kerop.")
    if name == "spectral":
        return _spectral_spectrum(
            n=n,
            seed=seed,
            feature_multiplier=feature_multiplier,
            include_rf_spectrum=include_rf_spectrum,
        )
    if name == "poisson":
        return _poisson_spectrum(
            n=n,
            seed=seed,
            feature_multiplier=feature_multiplier,
            include_rf_spectrum=include_rf_spectrum,
        )
    raise KeyError(
        f"unknown operator {operator!r}; expected 'spectral' or 'poisson' "
        f"(or the ids {list(bar_operator_ids())})"
    )


def _spectral_spectrum(
    *, n: int, seed: int, feature_multiplier: float, include_rf_spectrum: bool
) -> OperatorSpectrum:
    spec = BAR_REPRODUCE["spectral"]
    model = SpectralOperatorModel(
        r=spec["r"],
        b=spec["b"],
        n_modes=int(spec["task_kwargs"]["n_modes"]),
        output_dim=int(spec["output_dim"]),
        seed=seed,
    )
    n_features = _feature_count(n, spec["n_summands"], feature_multiplier)
    if include_rf_spectrum:
        inputs, _ = model.sample(n, np.random.default_rng([seed, n]))
        features = model.features(n_features, np.random.default_rng([seed, n_features]))
        rf = _rf_spectrum(features.design_matrix(inputs), n)
    else:
        rf = np.zeros(0, dtype=float)
    return OperatorSpectrum(
        operator_id="kerop.spectral",
        seed=seed,
        n=n,
        n_features=n_features,
        eigenvalues=_descending(model.eigenvalues),
        rf_spectrum=rf,
        metadata={
            "kind": "kernel_integral_operator",
            "description": (
                "Eigenvalues of the synthetic kernel integral operator L "
                "(product set mu_j * nu_k) used by the spectral wall-time bar."
            ),
            "r": spec["r"],
            "b": spec["b"],
            "n_modes": spec["task_kwargs"]["n_modes"],
            "output_dim": spec["output_dim"],
            "n_summands": spec["n_summands"],
            "input_dim": 1,
            "feature_multiplier": feature_multiplier,
            "reproduce": {
                **BAR_REPRODUCE["spectral"],
                "seed": seed,
                "n": n,
                "n_features": n_features,
                "cli": BAR_REPRODUCE["cli"].format(task="spectral"),
                "script": BAR_REPRODUCE["script"],
                "train_sizes": BAR_REPRODUCE["train_sizes"],
                "feature_multipliers": BAR_REPRODUCE["feature_multipliers"],
                "iteration_grid": BAR_REPRODUCE["iteration_grid"],
            },
            "recorded_bar": RECORDED_BAR["spectral"],
        },
    )


def _poisson_spectrum(
    *, n: int, seed: int, feature_multiplier: float, include_rf_spectrum: bool
) -> OperatorSpectrum:
    spec = BAR_REPRODUCE["poisson"]
    n_points = int(spec["task_kwargs"]["n_points"])
    dataset = PoissonDataset(n_points=n_points, noise_std=float(spec["noise_std"]))
    n_summands = dataset.n_summands
    n_features = _feature_count(n, n_summands, feature_multiplier)
    wavenumbers = np.arange(1, dataset.n_modes + 1, dtype=float)
    eigenvalues = (wavenumbers * np.pi) ** -2.0
    if include_rf_spectrum:
        batch = dataset.sample(n, np.random.default_rng([seed, n]))
        features = OperatorNTKFeatures(
            dataset.feature_dim,
            dataset.n_points,
            n_features,
            np.random.default_rng([seed, n_features]),
            output_scale=dataset.output_scale(),
        )
        rf = _rf_spectrum(features.design_matrix(dataset.lift(batch.fields)), n)
    else:
        rf = np.zeros(0, dtype=float)
    return OperatorSpectrum(
        operator_id="kerop.poisson",
        seed=seed,
        n=n,
        n_features=n_features,
        eigenvalues=_descending(eigenvalues),
        rf_spectrum=rf,
        metadata={
            "kind": "pde_solution_operator",
            "description": (
                "Eigenvalues (k*pi)^{-2} of the 1D Dirichlet Poisson solution "
                "operator used by the Poisson wall-time bar."
            ),
            "n_points": n_points,
            "n_modes": dataset.n_modes,
            "output_dim": dataset.n_points,
            "n_summands": n_summands,
            "feature_dim": dataset.feature_dim,
            "noise_std": spec["noise_std"],
            "feature_multiplier": feature_multiplier,
            "reproduce": {
                **BAR_REPRODUCE["poisson"],
                "seed": seed,
                "n": n,
                "n_features": n_features,
                "n_summands": n_summands,
                "cli": BAR_REPRODUCE["cli"].format(task="poisson"),
                "script": BAR_REPRODUCE["script"],
                "train_sizes": BAR_REPRODUCE["train_sizes"],
                "feature_multipliers": BAR_REPRODUCE["feature_multipliers"],
                "iteration_grid": BAR_REPRODUCE["iteration_grid"],
            },
            "recorded_bar": RECORDED_BAR["poisson"],
        },
    )


def build_bar_bundle(
    operators: tuple[str, ...] | None = None,
    *,
    n: int = BAR_N,
    seed: int = BAR_SEED,
    include_rf_spectrum: bool = True,
) -> FilterContractBundle:
    """Build the contract for the operators that make up the committed bar."""
    names = operators if operators is not None else ("spectral", "poisson")
    return FilterContractBundle(
        version=CONTRACT_VERSION,
        operators=tuple(
            build_operator_spectrum(name, n=n, seed=seed, include_rf_spectrum=include_rf_spectrum)
            for name in names
        ),
    )


def _array_key(operator_id: str, name: str) -> str:
    slug = operator_id.removeprefix("kerop.")
    return f"{slug}/{name}"


def write_filter_contract(
    path: Path | str,
    bundle: FilterContractBundle | None = None,
) -> tuple[Path, Path]:
    """Write ``<stem>.npz`` and ``<stem>.json``.  Returns ``(npz, json)``.

    ``path`` may be a directory (files are named ``filter_contract_v1``) or a
    file stem.  A trailing ``.npz`` / ``.json`` is stripped.
    """
    if bundle is None:
        bundle = build_bar_bundle()
    path = Path(path)
    if path.exists() and path.is_dir():
        stem = path / f"filter_contract_v{bundle.version}"
    else:
        stem = path
        if stem.suffix in {".npz", ".json"}:
            stem = stem.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Array] = {
        "contract_version": np.asarray(bundle.version, dtype=np.int64),
    }
    operators_json: list[dict[str, Any]] = []
    for item in bundle.operators:
        evals_key = _array_key(item.operator_id, "eigenvalues")
        rf_key = _array_key(item.operator_id, "rf_spectrum")
        arrays[evals_key] = np.asarray(item.eigenvalues, dtype=np.float64)
        arrays[rf_key] = np.asarray(item.rf_spectrum, dtype=np.float64)
        operators_json.append(
            {
                "operator_id": item.operator_id,
                "seed": int(item.seed),
                "n": int(item.n),
                "n_features": int(item.n_features),
                "feature_count": int(item.n_features),
                "arrays": {
                    "eigenvalues": evals_key,
                    "rf_spectrum": rf_key,
                },
                "eigenvalues_shape": list(item.eigenvalues.shape),
                "rf_spectrum_shape": list(item.rf_spectrum.shape),
                "ordering": "descending",
                "spectrum_units": "absolute",
                **item.metadata,
            }
        )

    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    np.savez(npz_path, allow_pickle=False, **arrays)  # type: ignore[arg-type]
    document = {
        "schema": SCHEMA_ID,
        "contract_version": bundle.version,
        "producer": "kerop",
        "producer_version": __version__,
        "consumer": "specinv",
        "description": (
            "Forward spectral quantities for the KerOp wall-time bar. "
            "Load the sibling .npz for the arrays named under operators[].arrays. "
            "See docs/FILTER_CONTRACT.md."
        ),
        "recorded_bar": {
            "protocol": RECORDED_BAR["protocol"],
            "spectral_median_speedup_at_matched_risk": (
                RECORDED_BAR["spectral"]["median_speedup_at_matched_risk"]
            ),
            "poisson_median_speedup_at_matched_risk": (
                RECORDED_BAR["poisson"]["median_speedup_at_matched_risk"]
            ),
        },
        "operators": operators_json,
    }
    json_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    return npz_path, json_path


def load_filter_contract(path: Path | str) -> FilterContractBundle:
    """Load a bundle written by :func:`write_filter_contract`.

    Accepts either the ``.json`` sidecar, the ``.npz``, or a bare stem.
    Intended for KerOp tests; SpecInv should follow ``docs/FILTER_CONTRACT.md``
    and use only ``json`` + ``numpy.load``.
    """
    path = Path(path)
    if path.suffix in {".npz", ".json"}:
        stem = path.with_suffix("")
    else:
        stem = path
    meta = json.loads(stem.with_suffix(".json").read_text())
    if int(meta["contract_version"]) != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported contract_version {meta['contract_version']}; "
            f"this loader understands {CONTRACT_VERSION}"
        )
    with np.load(stem.with_suffix(".npz")) as data:
        operators = []
        for record in meta["operators"]:
            arrays = record["arrays"]
            operators.append(
                OperatorSpectrum(
                    operator_id=record["operator_id"],
                    seed=int(record["seed"]),
                    n=int(record["n"]),
                    n_features=int(record["n_features"]),
                    eigenvalues=np.asarray(data[arrays["eigenvalues"]], dtype=float),
                    rf_spectrum=np.asarray(data[arrays["rf_spectrum"]], dtype=float),
                    metadata={
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "operator_id",
                            "seed",
                            "n",
                            "n_features",
                            "feature_count",
                            "arrays",
                            "eigenvalues_shape",
                            "rf_spectrum_shape",
                            "ordering",
                            "spectrum_units",
                        }
                    },
                )
            )
    return FilterContractBundle(version=int(meta["contract_version"]), operators=tuple(operators))
