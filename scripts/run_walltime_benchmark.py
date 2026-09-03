#!/usr/bin/env python3
"""Compare random features against exact operator-valued kernel regression.

Sweeps both estimators over the same grid of sample sizes and their own
hyper-parameters, then reports two things: which method reaches the lower risk
at matched sample size (the exact one), and which reaches a given risk level in
less wall-clock time (the random feature one).

Writes ``results/walltime_<task>.json`` plus CSVs for the raw fits, the
matched-sample-size comparison, and the matched-risk frontier.

Usage::

    python scripts/run_walltime_benchmark.py                     # synthetic, known (r, b)
    python scripts/run_walltime_benchmark.py --task darcy         # PDE operator with NTK
    python scripts/run_walltime_benchmark.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kerop.experiments import run_walltime_benchmark
from kerop.report import write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="spectral",
        choices=["spectral", "poisson", "darcy"],
        help="spectral has p=1 and known (r, b); poisson/darcy use the operator-valued NTK",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=None,
        help="sample sizes offered to both methods",
    )
    parser.add_argument("--n-targets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.train_sizes is not None:
        train_sizes = tuple(args.train_sizes)
    elif args.quick:
        train_sizes = (150, 300, 600)
    elif args.task == "spectral":
        train_sizes = (250, 500, 1000, 2000)
    else:
        train_sizes = (150, 300, 600, 1000)

    task_kwargs: dict[str, object] = {}
    if args.task != "spectral":
        task_kwargs["n_points"] = 12
    if args.quick and args.task == "spectral":
        task_kwargs.update({"n_modes": 512, "n_test": 1000})

    # The synthetic instance is regularization-limited and its best lambda sits
    # around 1e-3; the PDE tasks are nearly noiseless and theirs is several
    # decades lower, so each task gets a grid spanning its own useful range.
    if args.task == "spectral":
        lambda_grid = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
        feature_multipliers = (0.5, 1.0, 2.0) if args.quick else (0.5, 1.0, 2.0, 4.0, 8.0)
    else:
        lambda_grid = (1e-2, 1e-4, 1e-5, 1e-6, 1e-7)
        feature_multipliers = (0.5, 1.0, 2.0) if args.quick else (0.5, 1.0, 2.0, 4.0)

    payload = run_walltime_benchmark(
        task=args.task,
        train_sizes=train_sizes,
        lambda_grid=lambda_grid,
        feature_multipliers=feature_multipliers,
        n_targets=args.n_targets,
        seed=args.seed,
        task_kwargs=task_kwargs,
        verbose=True,
    )
    payload = {
        "experiment": "walltime_vs_exact_operator_valued_kernel",
        "description": (
            "Wall-clock and memory of the random feature spectral filtering estimator "
            "against exact operator-valued kernel regression with the same kernel, "
            "compared both at matched sample size and at matched excess risk."
        ),
        **payload,
    }

    suffix = args.task
    json_path = write_json(args.output_dir / f"walltime_{suffix}.json", payload)
    fits_path = write_csv(
        args.output_dir / f"walltime_{suffix}_fits.csv",
        list(payload["exact"]) + list(payload["random_features"]),
    )
    matched_size_path = write_csv(
        args.output_dir / f"walltime_{suffix}_matched_sample_size.csv",
        payload["matched_sample_size"],
    )
    frontier_rows = []
    for entry in payload["matched_excess_risk"]:
        row: dict[str, object] = {
            "target_risk": entry["target_risk"],
            "target_relative_error": entry["target_relative_error"],
            "speedup": entry["speedup"],
            "memory_ratio": entry["memory_ratio"],
        }
        for method in ("exact", "random_features"):
            choice = entry[method]
            prefix = "exact_" if method == "exact" else "rf_"
            if choice is None:
                row[prefix + "reached"] = False
                continue
            row[prefix + "reached"] = True
            row[prefix + "n_train"] = choice["n_train"]
            row[prefix + "label"] = choice["label"]
            row[prefix + "risk"] = choice["excess_risk"]
            row[prefix + "seconds"] = choice["fit_seconds"]
            row[prefix + "operator_dim"] = choice["operator_dim"]
            row[prefix + "megabytes"] = choice["operator_megabytes"]
            if method == "random_features":
                row["rf_n_features"] = choice["n_features"]
        frontier_rows.append(row)
    frontier_path = write_csv(
        args.output_dir / f"walltime_{suffix}_matched_risk.csv", frontier_rows
    )

    print()
    for path in (json_path, fits_path, matched_size_path, frontier_path):
        print(f"wrote {path}")
    verdict = payload["verdict"]
    if verdict["median_speedup_at_matched_risk"] is not None:
        print(
            f"median speed-up at matched excess risk: "
            f"{verdict['median_speedup_at_matched_risk']:.1f}x "
            f"(range {verdict['min_speedup_at_matched_risk']:.1f}x-"
            f"{verdict['max_speedup_at_matched_risk']:.1f}x over "
            f"{verdict['n_targets_with_both_methods']} target levels)"
        )
    else:
        print("no target risk level was reachable by both methods on this grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())