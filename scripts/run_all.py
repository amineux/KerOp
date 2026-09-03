#!/usr/bin/env python3
"""Run every experiment and write a single summary of the falsifiable claims.

This is the entry point referenced by the README.  It runs the filter
diagnostics, the rate experiment, the feature-threshold study and both
wall-clock benchmarks, then writes ``results/summary.json`` collecting the
claims each one is meant to support along with whether the measurement bears
them out.

Usage::

    python scripts/run_all.py            # full run, a few minutes
    python scripts/run_all.py --quick    # smoke test, well under a minute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("filter diagnostics", "run_filter_diagnostics.py", []),
    ("rate of Theorem 3.4", "run_rate_experiment.py", []),
    ("feature threshold (Appendix A.3)", "run_feature_threshold.py", ["--plot"]),
    ("wall-clock, synthetic instance", "run_walltime_benchmark.py", ["--task", "spectral"]),
    ("wall-clock, Darcy operator", "run_walltime_benchmark.py", ["--task", "darcy"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="run only the scripts whose file name contains one of these strings",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    timings: dict[str, float] = {}
    failures: list[str] = []

    for label, script, extra in SCRIPTS:
        if args.only and not any(token in script for token in args.only):
            continue
        command = [sys.executable, str(here / script), "--output-dir", str(args.output_dir)]
        command += extra
        if args.quick:
            command.append("--quick")
        print("=" * 78)
        print(f"== {label}")
        print("=" * 78)
        start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        timings[label] = time.perf_counter() - start
        if completed.returncode != 0:
            failures.append(label)
        print()

    print("=" * 78)
    print("== timings")
    for label, seconds in timings.items():
        print(f"   {label:<38s} {seconds:8.1f}s")
    print(f"   {'total':<38s} {sum(timings.values()):8.1f}s")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1

    summary_script = here / "collect_summary.py"
    if summary_script.exists():
        subprocess.run(
            [sys.executable, str(summary_script), "--output-dir", str(args.output_dir)],
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())