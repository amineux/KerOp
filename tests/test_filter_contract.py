"""Tests for the SpecInv filter-contract export."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kerop.cli import main
from kerop.data.spectral import SpectralOperatorModel
from kerop.filter_contract import (
    BAR_N,
    BAR_SEED,
    CONTRACT_VERSION,
    SCHEMA_ID,
    build_bar_bundle,
    build_operator_spectrum,
    load_filter_contract,
    write_filter_contract,
)


def test_spectral_eigenvalues_match_the_bar_model() -> None:
    item = build_operator_spectrum("spectral", include_rf_spectrum=False)
    model = SpectralOperatorModel(r=0.5, b=0.5, n_modes=512, output_dim=6, seed=BAR_SEED)
    expected = np.sort(model.eigenvalues.reshape(-1))[::-1]
    assert item.operator_id == "kerop.spectral"
    assert item.seed == BAR_SEED
    assert item.n == BAR_N
    assert item.n_features == max(8, round(math.sqrt(BAR_N)))
    assert np.allclose(item.eigenvalues, expected)
    assert item.eigenvalues[0] >= item.eigenvalues[-1]


def test_poisson_eigenvalues_are_dirichlet_poisson() -> None:
    item = build_operator_spectrum("poisson", include_rf_spectrum=False)
    wavenumbers = np.arange(1, 13, dtype=float)
    expected = np.sort((wavenumbers * np.pi) ** -2.0)[::-1]
    assert item.operator_id == "kerop.poisson"
    assert item.n_features == max(8, round(math.sqrt(BAR_N) * 9))
    assert np.allclose(item.eigenvalues, expected)


def test_rf_spectrum_is_a_valid_covariance_spectrum() -> None:
    item = build_operator_spectrum("spectral")
    assert item.rf_spectrum.size == item.n_features
    assert np.all(item.rf_spectrum >= -1e-12)
    assert np.all(np.diff(item.rf_spectrum) <= 1e-12)


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown operator"):
        build_operator_spectrum("darcy")


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    bundle = build_bar_bundle(include_rf_spectrum=True)
    npz_path, json_path = write_filter_contract(tmp_path / "contract", bundle)
    assert npz_path.exists() and json_path.exists()

    loaded = load_filter_contract(json_path)
    assert loaded.version == CONTRACT_VERSION
    assert [item.operator_id for item in loaded.operators] == [
        "kerop.spectral",
        "kerop.poisson",
    ]
    original = bundle.by_id("kerop.spectral")
    again = loaded.by_id("kerop.spectral")
    assert again.n == original.n
    assert again.n_features == original.n_features
    assert again.seed == original.seed
    assert np.allclose(again.eigenvalues, original.eigenvalues)
    assert np.allclose(again.rf_spectrum, original.rf_spectrum)


def test_artifact_is_loadable_with_only_numpy_and_json(tmp_path: Path) -> None:
    """SpecInv's documented load path: json + numpy.load, no KerOp imports."""
    npz_path, json_path = write_filter_contract(tmp_path / "filter_contract_v1")
    meta = json.loads(json_path.read_text())
    assert meta["schema"] == SCHEMA_ID
    assert meta["contract_version"] == CONTRACT_VERSION
    assert meta["producer"] == "kerop"
    assert "spectral_median_speedup_at_matched_risk" in meta["recorded_bar"]
    assert meta["recorded_bar"]["spectral_median_speedup_at_matched_risk"] == pytest.approx(
        27.717, rel=1e-3
    )
    assert meta["recorded_bar"]["poisson_median_speedup_at_matched_risk"] == pytest.approx(
        3.556, rel=1e-3
    )

    arrays = np.load(npz_path)
    assert int(arrays["contract_version"]) == CONTRACT_VERSION
    ids = {record["operator_id"] for record in meta["operators"]}
    assert ids == {"kerop.spectral", "kerop.poisson"}
    for record in meta["operators"]:
        for required in (
            "seed",
            "n",
            "n_features",
            "feature_count",
            "coefficient_dim",
            "arrays",
            "reproduce",
        ):
            assert required in record
        assert record["n_features"] == record["feature_count"]
        evals = arrays[record["arrays"]["eigenvalues"]]
        rf = arrays[record["arrays"]["rf_spectrum"]]
        assert evals.ndim == 1 and evals.size > 0
        # hat Sigma_M is pM x pM, so the RF spectrum has length p*M, not M.
        assert rf.ndim == 1 and rf.size == record["coefficient_dim"]
        assert record["coefficient_dim"] == record["n_summands"] * record["n_features"]
        assert "cli" in record["reproduce"]
        assert record["reproduce"]["seed"] == record["seed"]


def test_cli_writes_the_artifact(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "export-filter-contract",
                "--output-dir",
                str(tmp_path),
                "--operator",
                "spectral",
                "--skip-rf-spectrum",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    json_path = tmp_path / "filter_contract_v1.json"
    npz_path = tmp_path / "filter_contract_v1.npz"
    assert json_path.exists() and npz_path.exists()
    assert str(json_path) in output or "wrote" in output
    meta = json.loads(json_path.read_text())
    assert meta["operators"][0]["operator_id"] == "kerop.spectral"
    assert meta["operators"][0]["rf_spectrum_shape"] == [0]
