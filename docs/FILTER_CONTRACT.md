# Filter contract — canonical schema `kerop.filter_contract/v1`

This is the **only** KerOp → SpecInv handshake schema. SpecInv loads these
files with `json` and `numpy.load`. It must not mint a second schema
(`kerop.specinv.spectrum/v1` or any other id).

This is a file format, not a new estimator and not a new theorem. The
committed wall-time bar is unchanged: median RF vs exact operator-valued KRR
at matched excess risk is **~27.7×** on `kerop.spectral` and **~3.6×** on
`kerop.dirichlet1d` (quick run; see `results/walltime_*.json`).

## Files

One stem `S`. Committed fixture: `fixtures/filter_contract_v1`.
CLI default: `results/filter_contract/filter_contract_v1`.

| File | Role |
|---|---|
| `S.json` | Schema id, operator records, seed, `n`, feature count, bar settings |
| `S.npz` | Numeric arrays named by `operators[].arrays` (no pickle) |

```bash
kerop export-filter-contract --output-dir results/filter_contract
```

or `kerop.filter_contract.write_filter_contract(path)`.

## Load (json + numpy only)

```python
import json
from pathlib import Path
import numpy as np

stem = Path("fixtures/filter_contract_v1")
meta = json.loads(stem.with_suffix(".json").read_text())
assert meta["schema"] == "kerop.filter_contract/v1"
assert int(meta["contract_version"]) == 1

arrays = np.load(stem.with_suffix(".npz"))
for op in meta["operators"]:
    eigenvalues = arrays[op["arrays"]["eigenvalues"]]   # 1-D, descending
    rf_spectrum = arrays[op["arrays"]["rf_spectrum"]]   # 1-D, descending; may be empty
    operator_id = op["operator_id"]   # "kerop.spectral" or "kerop.dirichlet1d"
    seed = int(op["seed"])
    n = int(op["n"])
    n_features = int(op["n_features"])
    assert operator_id != "kerop.poisson"
```

Refuse any other `schema`. Ignore unknown keys.

## Required top-level JSON keys

| Key | Type | Rule |
|---|---|---|
| `schema` | string | Exactly `"kerop.filter_contract/v1"` |
| `contract_version` | int | Exactly `1` |
| `producer` | string | `"kerop"` |
| `operators` | list | One object per exported operator; length ≥ 1 |

Optional top-level: `producer_version`, `consumer` (`"specinv"`), `description`,
`recorded_bar` (copied medians from the committed wall-time JSONs).

## Required `operators[]` keys

| Key | Type | Rule |
|---|---|---|
| `operator_id` | string | Canonical id, see below |
| `seed` | int | RNG seed for the RF draw |
| `n` | int | Training sample size used for `rf_spectrum` |
| `n_features` | int | Random-feature count \(M\) |
| `feature_count` | int | Same value as `n_features` |
| `n_summands` | int | Factor \(p\); coefficient space is \(pM\) |
| `coefficient_dim` | int | Length of `rf_spectrum` when present, else \(pM\) |
| `arrays` | object | Maps logical names → `.npz` keys |
| `arrays.eigenvalues` | string | `.npz` key of the population / closed-form spectrum |
| `arrays.rf_spectrum` | string | `.npz` key of \(\widehat\Sigma_M = Z^\top Z/n\) |
| `eigenvalues_shape` | `[int]` | Shape of that array, e.g. `[3072]` |
| `rf_spectrum_shape` | `[int]` | Shape of that array, e.g. `[12]` or `[0]` |
| `ordering` | string | `"descending"` |
| `spectrum_units` | string | `"absolute"` (not rescaled to \([0,1]\)) |

Optional per operator: `kind`, `is_fem`, `reproduce`, `recorded_bar`,
geometry fields (`r`, `b`, `n_modes`, `output_dim`, …).

## Array shapes

Both arrays are `float64`, rank 1, **descending**, no pickle.

| Array | Shape | Meaning |
|---|---|---|
| `eigenvalues` | `(K,)` \(K\ge 1\) | Closed-form / population forward spectrum |
| `rf_spectrum` | `(pM,)` or `(0,)` | Eigenvalues of \(\widehat\Sigma_M\). Empty if `--skip-rf-spectrum` |

`n` is the **training sample size**, not `K`. Do not assume `n == eigenvalues.size`.

On the committed bar fixture:

| `operator_id` | `eigenvalues` | `rf_spectrum` | `n` | `M` | `p` |
|---|---|---|---|---|---|
| `kerop.spectral` | `(3072,)` product set \(\{\mu_j\nu_k\}\) | `(12,)` | 150 | 12 | 1 |
| `kerop.dirichlet1d` | `(12,)` values \((k\pi)^{-2}\) | `(990,)` | 150 | 110 | 9 |

## Operator id rules

Pattern: `kerop.<slug>` with `slug` matching `[a-z][a-z0-9_]*`.

An id names an operator **this package actually ships**. It must not imply a
killed attachment (GPU RF, imaging, FEM, NeuroFEM).

| Id | What it is | What it is not |
|---|---|---|
| `kerop.spectral` | Synthetic kernel integral operator of the spectral wall-time bar (`r=0.5`, `b=0.5`, 512 modes, \(d_v=6\)) | A new instance or a 100× claim |
| `kerop.dirichlet1d` | Existing 1D Dirichlet map \(-u''=f\) on \([0,1]\), closed form in the sine basis; wall-time `--task poisson` | **Not FEM.** Not a Poisson stiffness matrix. Not a 2D/3D attachment |

`kerop.poisson` is **not** a valid id and must not appear in an artifact.

## Compatibility

- v1 is additive: new optional JSON keys are allowed; required keys keep their
  meaning and types.
- A future v2 must bump both `contract_version` and `schema`.
- SpecInv should ignore unknown keys and refuse an unknown `schema`.
