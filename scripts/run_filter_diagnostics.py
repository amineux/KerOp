#!/usr/bin/env python3
"""Measure the Definition 2.2 constants, qualification, and bias saturation.

For each spectral filter family this reports the constants D, E, c_0 of
(2.7)-(2.9) and the qualification of (2.10), measured numerically and compared
with the value known analytically.  It then measures the consequence of
qualification on the exact bias of the synthetic instance: a family with
qualification nu cannot beat lambda^nu no matter how smooth the target is, so
for r > nu the measured bias exponent sticks at nu.  This is why Theorem 3.4
requires nu >= r or 1.

Writes ``results/filter_diagnostics.json`` and two CSVs.

Usage::

    python scripts/run_filter_diagnostics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kerop.experiments import run_filter_report
from kerop.report import summarize_table, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    families = (
        (("tikhonov", {}), ("landweber", {}), ("nu_method", {"nu": 2.0}))
        if args.quick
        else (
            ("tikhonov", {}),
            ("iterated_tikhonov", {"order": 2}),
            ("iterated_tikhonov", {"order": 3}),
            ("landweber", {}),
            ("cutoff", {}),
            ("heavy_ball", {"momentum": 0.9}),
            ("nu_method", {"nu": 1.0}),
            ("nu_method", {"nu": 2.0}),
            ("nu_method", {"nu": 3.0}),
        )
    )
    exponents = (0.5, 1.5) if args.quick else (0.5, 1.0, 1.5, 2.0)

    payload = run_filter_report(
        families=families, saturation_source_exponents=exponents, verbose=True
    )
    payload = {
        "experiment": "filter_diagnostics",
        "description": (
            "Numerically measured constants and qualification of each spectral "
            "regularization family, and the effect of qualification on the exact bias."
        ),
        **payload,
    }

    json_path = write_json(args.output_dir / "filter_diagnostics.json", payload)
    families_path = write_csv(
        args.output_dir / "filter_families.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"c_q", "growth_in_lambda"}
            }
            for row in payload["families"]
        ],
    )
    saturation_path = write_csv(
        args.output_dir / "filter_bias_saturation.csv", payload["bias_saturation"]
    )

    print()
    print("Bias exponent of the exact bias ||r_lambda(L) G_rho||, by source exponent r:")
    print(
        summarize_table(
            payload["bias_saturation"],
            ["source_exponent_r", "filter", "qualification", "measured_bias_exponent",
             "expected_exponent", "saturated"],
        )
    )
    print()
    for path in (json_path, families_path, saturation_path):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())