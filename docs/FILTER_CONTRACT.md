# Filter contract (v1)

KerOp writes a **forward spectrum** so [SpecInv](https://github.com/amineux/SpecInv)
can consume it without importing KerOp. This is a file format, not a new
estimator and not a new theorem.

The committed wall-time bar is unchanged: median RF vs exact operator-valued
KRR speed-up at matched excess risk is **~27.7×** on the synthetic spectral
instance and **~3.6×** on Poisson (quick run; see `results/walltime_*.json`).

## Files

One stem `S` (default `results/filter_contract/filter_contract_v1`):

| File | Role |
|---|---|
| `S.json` | Schema, operator ids, seed, `n`, feature count, bar settings |
| `S.npz` | Numeric arrays named by `operators[].arrays` |

Write them with:

```bash
kerop export-filter-contract --output-dir results/filter_contract
```

or, in Python, `kerop.filter_contract.write_filter_contract(path)`.

## Load without KerOp

```python
import json
from pathlib import Path
import numpy as np

stem = Path("results/filter_contract/filter_contract_v1")
meta = json.loads(stem.with_suffix(".json").read_text())
assert meta["schema"] == "kerop.filter_contract/v1"
assert int(meta["contract_version"]) == 1

arrays = np.load(stem.with_suffix(".npz"))
for op in meta["operators"]:
    eigenvalues = arrays[op["arrays"]["eigenvalues"]]   # descending
    rf_spectrum = arrays[op["arrays"]["rf_spectrum"]]   # descending; may be empty
    operator_id = op["operator_id"]   # "kerop.spectral" or "kerop.poisson"
    seed = op["seed"]
    n = op["n"]
    n_features = op["n_features"]
```

Do not unpickle the `.npz`. Keys are plain arrays.

## JSON fields

Top level:

| Field | Meaning |
|---|---|
| `schema` | `"kerop.filter_contract/v1"` |
| `contract_version` | Integer `1` |
| `producer` / `producer_version` | `"kerop"` and the package version |
| `consumer` | `"specinv"` (intended reader) |
| `recorded_bar` | Copied medians from the committed wall-time JSONs |
| `operators` | List of operator records |

Each `operators[]` entry:

| Field | Meaning |
|---|---|
| `operator_id` | Stable id: `kerop.spectral` or `kerop.poisson` |
| `seed` | Integer seed used for the RF draw (`20260301` on the bar) |
| `n` | Training sample size used for `rf_spectrum` |
| `n_features` / `feature_count` | Random-feature count \(M\) (same value twice) |
| `n_summands` | The factor \(p\) in the feature expansion; coefficient space is \(pM\) |
| `coefficient_dim` | Length of `rf_spectrum`, equal to \(pM\) |
| `arrays.eigenvalues` | Key in the `.npz` for the population / closed-form spectrum |
| `arrays.rf_spectrum` | Key in the `.npz` for eigenvalues of \(\widehat\Sigma_M = Z^\top Z/n\) (length \(pM\)) |
| `ordering` | Always `"descending"` |
| `spectrum_units` | `"absolute"` (not rescaled to \([0,1]\)) |
| `reproduce` | CLI, seed, train-size grid, \(\lambda\) grid, feature multipliers |
| `recorded_bar` | Per-operator medians from `results/walltime_<task>.json` |

`kerop.spectral` eigenvalues are the product set \(\{\mu_j\nu_k\}\) of the
synthetic kernel integral operator (`r=0.5`, `b=0.5`, 512 modes, \(d_v=6\)).
`kerop.poisson` eigenvalues are \((k\pi)^{-2}\) for the 1D Dirichlet Poisson
map on 12 collocation points.

`n` and `M` on the artifact are the smallest bar grid point
(\(n=150\), \(M \approx \sqrt{n}\,p\)). The RF spectrum has length \(pM\),
not \(M\). The full sweep that produced the medians is in `reproduce`.

## Compatibility

- v1 is additive: new optional JSON keys are allowed; existing keys keep their
  meaning.
- A future v2 must bump `contract_version` and `schema`.
- SpecInv should ignore unknown keys and refuse an unknown `contract_version`.
