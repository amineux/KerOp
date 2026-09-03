#!/usr/bin/env python3
"""Recreate the feature-threshold study of Appendix A.3 (the Figure 1 setup).

Maps the test error of kernel gradient descent on the real-valued NTK over the
number of random features M and the number of iterations T, and checks the
paper's claim that M of order sqrt(n) * p suffices, with p = d + 2.

Writes ``results/feature_threshold.json`` plus three CSVs: the (M, T) grid that
the heat plot is drawn from, the per-case sufficiency check, and the located
thresholds.

Usage::

    python scripts/run_feature_threshold.py
    python scripts/run_feature_threshold.py --quick
    python scripts/run_feature_threshold.py --plot   # requires matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kerop.experiments import run_feature_threshold
from kerop.report import write_csv, write_json

DEFAULT_CASES = ((1, 1250), (1, 2500), (1, 5000), (14, 1250), (14, 2500))
QUICK_CASES = ((1, 500), (1, 1000))


def _plot(payload: dict, output_dir: Path) -> Path | None:
    """Draw the (M, T) heat map, the visual analogue of the paper's Figure 1."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the plot")
        return None

    import numpy as np

    cases = sorted({(row["input_dim"], row["n"]) for row in payload["grid"]})
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.6), squeeze=False)
    for axis, (input_dim, n_samples) in zip(axes[0], cases):
        rows = [
            row
            for row in payload["grid"]
            if row["input_dim"] == input_dim and row["n"] == n_samples
        ]
        features = sorted({row["n_features"] for row in rows})
        iterations = sorted({row["iterations"] for row in rows})
        grid = np.array(
            [
                [
                    next(
                        row["mean_test_error"]
                        for row in rows
                        if row["n_features"] == m and row["iterations"] == t
                    )
                    for m in features
                ]
                for t in iterations
            ]
        )
        image = axis.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        axis.set_xticks(range(len(features)))
        axis.set_xticklabels([str(m) for m in features], rotation=45, fontsize=7)
        axis.set_yticks(range(len(iterations)))
        axis.set_yticklabels([str(t) for t in iterations], fontsize=7)
        axis.set_xlabel("random features $M$")
        axis.set_ylabel("iterations $T$")
        reference = np.sqrt(n_samples) * (input_dim + 2)
        closest = int(np.argmin([abs(m - reference) for m in features]))
        axis.axvline(closest, color="red", linewidth=1.4, linestyle="--")
        axis.set_title(f"$d={input_dim}$, $n={n_samples}$\nred: $M=\\sqrt{{n}}\\,p$", fontsize=9)
        fig.colorbar(image, ax=axis, label="test error")
    fig.suptitle(
        "Test error over random features and iterations (Appendix A.3 setup)", fontsize=10
    )
    fig.tight_layout()
    path = output_dir / "feature_threshold_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--n-test", type=int, default=1500)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot", action="store_true", help="also write a heat map PNG")
    args = parser.parse_args()

    cases = QUICK_CASES if args.quick else DEFAULT_CASES
    payload = run_feature_threshold(
        settings=cases,
        repeats=2 if args.quick else args.repeats,
        n_test=500 if args.quick else args.n_test,
        noise_std=args.noise_std,
        tolerance=args.tolerance,
        seed=args.seed,
        verbose=True,
    )
    payload = {
        "experiment": "feature_threshold_appendix_a3",
        "description": (
            "Test error of kernel gradient descent on the real-valued NTK over the "
            "number of random features and iterations, checking that M of order "
            "sqrt(n) * p suffices to reach the large-M plateau."
        ),
        **payload,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(args.output_dir / "feature_threshold.json", payload)
    grid_path = write_csv(args.output_dir / "feature_threshold_grid.csv", payload["grid"])
    suff_path = write_csv(
        args.output_dir / "feature_threshold_sufficiency.csv", payload["sufficiency"]
    )
    thresh_path = write_csv(
        args.output_dir / "feature_threshold_locations.csv", payload["thresholds"]
    )

    print()
    for path in (json_path, grid_path, suff_path, thresh_path):
        print(f"wrote {path}")
    if args.plot:
        plot_path = _plot(payload, args.output_dir)
        if plot_path:
            print(f"wrote {plot_path}")

    verdict = payload["verdict"]
    print(
        f"M = sqrt(n) p reached the plateau in "
        f"{verdict['cases_where_sqrt_n_p_suffices']}/{verdict['cases_tested']} cases; "
        f"worst excess over plateau {100 * verdict['worst_excess_over_plateau']:+.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())