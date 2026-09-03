#!/usr/bin/env python3
"""Measure the excess-risk rate and compare it with Theorem 3.4.

Writes ``results/rate_experiment.json`` with the full record, plus two CSVs:
``rate_experiment_summary.csv`` with the fitted exponent per configuration and
filter, and ``rate_experiment_raw.csv`` with every individual fit.

Usage::

    python scripts/run_rate_experiment.py                 # full run
    python scripts/run_rate_experiment.py --quick          # small, for CI
    python scripts/run_rate_experiment.py --repeats 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kerop.experiments import (
    DEFAULT_RATE_CONFIGS,
    RateConfig,
    rate_results_to_dicts,
    run_rate_experiment,
)
from kerop.report import write_csv, write_json

QUICK_CONFIGS = (
    RateConfig(
        name="well-specified",
        r=0.5,
        b=0.5,
        filters=(("tikhonov", {}), ("nu_method", {"nu": 2.0})),
        n_grid=(200, 400, 800, 1600),
        n_modes=1024,
        output_dim=5,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one small configuration, for smoke tests and CI",
    )
    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else DEFAULT_RATE_CONFIGS
    repeats = 3 if args.quick else args.repeats
    n_test = 1000 if args.quick else args.n_test

    results = run_rate_experiment(
        configs, repeats=repeats, n_test=n_test, seed=args.seed, verbose=True
    )
    payload = {
        "experiment": "rate_of_theorem_3_4",
        "description": (
            "Excess risk against sample size on a synthetic instance where the source "
            "and capacity exponents are known by construction and measured from the "
            "exact spectrum, compared with the exponent r/(2r+b) of Theorem 3.4."
        ),
        "configurations": rate_results_to_dicts(results),
    }
    json_path = write_json(args.output_dir / "rate_experiment.json", payload)

    summary_rows = []
    raw_rows = []
    for result in results:
        for entry in result.summary:
            summary_rows.append(
                {
                    "configuration": result.config["name"],
                    "r_nominal": result.config["r"],
                    "b_nominal": result.config["b"],
                    "r_measured": result.measured_r,
                    "b_measured": result.measured_b,
                    "exponent_nominal": result.nominal_exponent,
                    "exponent_measured": result.measured_exponent,
                    "filter": entry["filter"],
                    "filter_options": entry["filter_options"],
                    "qualification": entry["qualification"],
                    "qualification_required": entry["qualification_required"],
                    "qualification_ok": entry["qualification_ok"],
                    "fitted_slope": entry["slope"],
                    "fitted_slope_stderr": entry["slope_stderr"],
                    "tail_slope": entry["tail_slope"],
                    "r_squared": entry["r_squared"],
                    "relative_error_vs_measured": entry["relative_error_vs_measured"],
                    "relative_error_vs_nominal": entry["relative_error_vs_nominal"],
                    "within_tolerance": entry["within_tolerance"],
                }
            )
        for row in result.rows:
            raw_rows.append({"configuration": result.config["name"], **row})

    summary_path = write_csv(args.output_dir / "rate_experiment_summary.csv", summary_rows)
    raw_path = write_csv(args.output_dir / "rate_experiment_raw.csv", raw_rows)

    passed = sum(row["within_tolerance"] for row in summary_rows if row["qualification_ok"])
    eligible = sum(1 for row in summary_rows if row["qualification_ok"])
    print()
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {raw_path}")
    print(
        f"{passed}/{eligible} filter/configuration pairs satisfying the qualification "
        f"hypothesis matched the predicted exponent within tolerance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())