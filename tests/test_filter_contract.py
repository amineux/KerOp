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
    OPERATOR_DIRICHLET1D,
    OPERATOR_SPECTRAL,
    SCHEMA_ID,
    build_bar_bundle,
    build_operator_spectrum,
    load_filter_contract,
    resolve_operator_id,
    write_filter_contract,
)

REPO_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "filter_contract_v1"


def test_spectral_eigenvalues_match_the_bar_model() -> None:
    item = build_operator_spectrum("spectral", include_rf_spectrum=False)
    model = SpectralOperatorModel(r=0.5, b=0.5, n_modes=512, output_dim=6, seed=BAR_SEED)
    expected = np.sort(model.eigenvalues.reshape(-1))[::-1]
    assert item.operator_id == OPERATOR_SPECTRAL
    assert item.seed == BAR_SEED
    assert item.n == BAR_N
    assert item.n_features == max(8, round(math.sqrt(BAR_N)))
    assert np.allclose(item.eigenvalues, expected)
    assert item.eigenvalues[0] >= item.eigenvalues[-1]


def test_dirichlet1d_eigenvalues_are_closed_form_sine() -> None:
    item = build_operator_spectrum("dirichlet1d", include_rf_spectrum=False)
    wavenumbers = np.arange(1, 13, dtype=float)
    expected = np.sort((wavenumbers * np.pi) ** -2.0)[::-1]
    assert item.operator_id == OPERATOR_DIRICHLET1D
    assert item.n_features == max(8, round(math.sqrt(BAR_N) * 9))
    assert np.allclose(item.eigenvalues, expected)
    assert item.metadata["is_fem"] is False


def test_kerop_poisson_is_not_an_operator_id() -> None:
    with pytest.raises(KeyError, match="FEM"):
        resolve_operator_id("kerop.poisson")
    with pytest.raises(KeyError, match="FEM"):
        build_operator_spectrum("poisson")


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
        OPERATOR_SPECTRAL,
        OPERATOR_DIRICHLET1D,
    ]
    original = bundle.by_id(OPERATOR_SPECTRAL)
    again = loaded.by_id(OPERATOR_SPECTRAL)
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
    assert meta["schema"] == "kerop.filter_contract/v1"
    assert meta["contract_version"] == CONTRACT_VERSION
    assert meta["producer"] == "kerop"
    assert "spectral_median_speedup_at_matched_risk" in meta["recorded_bar"]
    assert meta["recorded_bar"]["spectral_median_speedup_at_matched_risk"] == pytest.approx(
        27.717, rel=1e-3
    )
    assert meta["recorded_bar"]["dirichlet1d_median_speedup_at_matched_risk"] == pytest.approx(
        3.556, rel=1e-3
    )

    arrays = np.load(npz_path)
    assert int(arrays["contract_version"]) == CONTRACT_VERSION
    ids = {record["operator_id"] for record in meta["operators"]}
    assert ids == {OPERATOR_SPECTRAL, OPERATOR_DIRICHLET1D}
    assert "kerop.poisson" not in ids
    for record in meta["operators"]:
        for required in (
            "operator_id",
            "seed",
            "n",
            "n_features",
            "feature_count",
            "n_summands",
            "coefficient_dim",
            "arrays",
            "eigenvalues_shape",
            "rf_spectrum_shape",
            "ordering",
            "spectrum_units",
        ):
            assert required in record
        assert record["n_features"] == record["feature_count"]
        assert record["ordering"] == "descending"
        evals = arrays[record["arrays"]["eigenvalues"]]
        rf = arrays[record["arrays"]["rf_spectrum"]]
        assert evals.ndim == 1 and evals.size > 0
        assert list(evals.shape) == record["eigenvalues_shape"]
        assert list(rf.shape) == record["rf_spectrum_shape"]
        # hat Sigma_M is pM x pM, so the RF spectrum has length p*M, not M.
        assert rf.ndim == 1 and rf.size == record["coefficient_dim"]
        assert record["coefficient_dim"] == record["n_summands"] * record["n_features"]


def test_committed_fixture_matches_the_schema() -> None:
    """The checked-in fixture is what SpecInv should load: json + numpy.load."""
    json_path = REPO_FIXTURE.with_suffix(".json")
    npz_path = REPO_FIXTURE.with_suffix(".npz")
    assert json_path.is_file() and npz_path.is_file()
    meta = json.loads(json_path.read_text())
    assert meta["schema"] == "kerop.filter_contract/v1"
    assert int(meta["contract_version"]) == 1
    ids = [record["operator_id"] for record in meta["operators"]]
    assert ids == [OPERATOR_SPECTRAL, OPERATOR_DIRICHLET1D]
    assert "kerop.poisson" not in ids
    arrays = np.load(npz_path)
    for record in meta["operators"]:
        evals = arrays[record["arrays"]["eigenvalues"]]
        rf = arrays[record["arrays"]["rf_spectrum"]]
        assert evals.ndim == 1 and evals.size == record["eigenvalues_shape"][0]
        assert rf.ndim == 1 and rf.size == record["rf_spectrum_shape"][0]
        if record["operator_id"] == OPERATOR_DIRICHLET1D:
            assert record.get("is_fem") is False


def test_cli_writes_the_artifact(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "export-filter-contract",
                "--output-dir",
                str(tmp_path),
                "--operator",
                "dirichlet1d",
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
    assert meta["schema"] == "kerop.filter_contract/v1"
    assert meta["operators"][0]["operator_id"] == OPERATOR_DIRICHLET1D
    assert meta["operators"][0]["rf_spectrum_shape"] == [0]
    assert "kerop.poisson" not in json_path.read_text()
