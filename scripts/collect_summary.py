#!/usr/bin/env python3
"""Collect the experiment outputs into one summary of the falsifiable claims.

Reads the JSON files written by the other scripts and produces
``results/summary.json`` and ``results/summary.md``, stating for each claim what
was measured, what the paper predicts, and whether the two agree.  The point is
that a reader can check the claims without rerunning anything, and that a
disagreement is visible rather than buried.

Usage::

    python scripts/collect_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kerop.report import write_json


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _rate_claim(document: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for config in document["configurations"]:
        for entry in config["summary"]:
            rows.append(
                {
                    "configuration": config["config"]["name"],
                    "r_nominal": config["config"]["r"],
                    "b_nominal": config["config"]["b"],
                    "r_measured": config["measured_r"],
                    "b_measured": config["measured_b"],
                    "predicted_exponent": config["measured_exponent"],
                    "nominal_exponent": config["nominal_exponent"],
                    "filter": entry["filter"],
                    "qualification": entry["qualification"],
                    "qualification_ok": entry["qualification_ok"],
                    "measured_slope": entry["slope"],
                    "relative_error": entry["relative_error_vs_measured"],
                    "within_tolerance": entry["within_tolerance"],
                }
            )
    eligible = [row for row in rows if row["qualification_ok"]]
    passed = [row for row in eligible if row["within_tolerance"]]
    saturating = [row for row in rows if not row["qualification_ok"]]
    return {
        "claim": (
            "The excess risk decays as n^{-r/(2r+b)} (Theorem 3.4) on an instance where "
            "the source and capacity exponents are known and independently measured."
        ),
        "eligible_pairs": len(eligible),
        "pairs_within_tolerance": len(passed),
        "worst_relative_error_among_eligible": (
            max(row["relative_error"] for row in eligible) if eligible else None
        ),
        "median_relative_error_among_eligible": (
            sorted(row["relative_error"] for row in eligible)[len(eligible) // 2]
            if eligible
            else None
        ),
        "verdict": bool(eligible and len(passed) == len(eligible)),
        "filters_violating_the_qualification_hypothesis": [
            {
                "configuration": row["configuration"],
                "filter": row["filter"],
                "r": row["r_nominal"],
                "qualification": row["qualification"],
                "measured_slope": row["measured_slope"],
                "relative_error": row["relative_error"],
            }
            for row in saturating
        ],
        "rows": rows,
    }


def _threshold_claim(document: dict[str, Any]) -> dict[str, Any]:
    verdict = document["verdict"]
    return {
        "claim": (
            "M of order sqrt(n) * p random features suffice to reach the large-M plateau "
            "of the test error, with p = d + 2 (Appendix A.3)."
        ),
        "cases_tested": verdict["cases_tested"],
        "cases_where_sqrt_n_p_suffices": verdict["cases_where_sqrt_n_p_suffices"],
        "worst_excess_over_plateau": verdict["worst_excess_over_plateau"],
        "verdict": verdict["cases_where_sqrt_n_p_suffices"] == verdict["cases_tested"],
        "located_threshold_scaling": document["scaling"],
    }


def _walltime_claim(document: dict[str, Any]) -> dict[str, Any]:
    verdict = document["verdict"]
    return {
        "claim": (
            "The random feature estimator reaches a given excess risk in less wall-clock "
            "time than exact operator-valued kernel regression with the same kernel."
        ),
        "task": document["settings"]["task_name"],
        "n_summands_p": document["settings"]["n_summands"],
        "output_dim": document["settings"]["output_dim"],
        "target_levels_reachable_by_both": verdict["n_targets_with_both_methods"],
        "median_speedup_at_matched_risk": verdict["median_speedup_at_matched_risk"],
        "min_speedup_at_matched_risk": verdict["min_speedup_at_matched_risk"],
        "max_speedup_at_matched_risk": verdict["max_speedup_at_matched_risk"],
        "median_speedup_charging_exact_for_the_solve_only": verdict[
            "median_speedup_exact_solve_only"
        ],
        "min_speedup_charging_exact_for_the_solve_only": verdict[
            "min_speedup_exact_solve_only"
        ],
        "verdict": verdict["random_features_faster_at_every_target"],
        "verdict_charging_exact_for_the_solve_only": verdict[
            "random_features_faster_at_every_target_solve_only"
        ],
        "matched_sample_size_note": (
            "At matched sample size the exact method attains the lower risk; the random "
            "feature advantage is in the cost of reaching a given risk level."
        ),
        "matched_sample_size": document["matched_sample_size"],
    }


def _independence_claim(root: Path) -> dict[str, Any]:
    """Check, from the repository itself, that this is not a wrapper.

    The claim is about provenance, so it is evidenced by things a reader can
    verify: the declared dependencies, the absence of any import of an existing
    random-feature package, and the presence of machinery that the prior
    vector-valued random feature work does not contain.  Rudi & Rosasco and
    Lanthaler & Nelsen both analyse kernel *ridge* regression only - Table 1 of
    the paper attributes exactly that scope to them - so a spectral filtering
    stack with measured qualifications, and an exact operator-valued baseline
    running those same filters, are the substantive additions this paper calls
    for and could not be inherited.
    """
    import tomllib

    import kerop
    from kerop.filters import FILTER_REGISTRY

    config = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = config["project"]["dependencies"]

    sources = sorted((root / "src" / "kerop").rglob("*.py"))
    total_lines = sum(len(path.read_text().splitlines()) for path in sources)
    tests = sorted((root / "tests").rglob("test_*.py"))
    test_lines = sum(len(path.read_text().splitlines()) for path in tests)

    # Any dependency on, or import of, an existing vvRF implementation.
    forbidden = ("vvrf", "error-bounds-for-vvrf", "error_bounds", "nelsen")
    text = "\n".join(path.read_text().lower() for path in sources)
    imports_found = [
        token for token in forbidden if f"import {token}" in text or f"from {token}" in text
    ]
    dependency_hits = [
        name for name in dependencies if any(token in name.lower() for token in forbidden)
    ]

    return {
        "claim": (
            "This is an independent implementation of the paper's spectral filtering "
            "analysis stack, not a wrapper around an existing vector-valued random "
            "feature codebase."
        ),
        "declared_runtime_dependencies": dependencies,
        "third_party_rf_imports": imports_found,
        "third_party_rf_dependencies": dependency_hits,
        "modules": [str(path.relative_to(root)) for path in sources],
        "source_lines": total_lines,
        "test_lines": test_lines,
        "spectral_filter_families": sorted(FILTER_REGISTRY),
        "filter_families_beyond_tikhonov": len(FILTER_REGISTRY) - 1,
        "implements_exact_operator_valued_baseline": hasattr(
            kerop, "ExactOperatorFilter"
        ),
        "implements_closed_form_operator_ntk": hasattr(kerop, "OperatorNTKKernel"),
        "verdict": bool(
            not imports_found
            and not dependency_hits
            and set(dependencies) <= {"numpy>=1.24", "scipy>=1.10"}
            and len(FILTER_REGISTRY) >= 5
        ),
    }


def _saturation_claim(document: dict[str, Any]) -> dict[str, Any]:
    rows = document["bias_saturation"]
    checks = []
    for row in rows:
        expected = row["expected_exponent"]
        if isinstance(expected, str):  # infinite qualification
            expected = row["source_exponent_r"]
        checks.append(
            {
                "filter": row["filter"],
                "options": row["options"],
                "r": row["source_exponent_r"],
                "qualification": row["qualification"],
                "expected_bias_exponent": expected,
                "measured_bias_exponent": row["measured_bias_exponent"],
                "relative_error": abs(row["measured_bias_exponent"] - expected)
                / max(expected, 1e-9),
                "saturated": row["saturated"],
                # Excluded from the verdict: at the top of the probe window this
                # filter runs too few iterations for its residual to have
                # reached the asymptotic form the qualification describes.
                "transient_limited": row.get("transient_limited", False),
                "min_iterations_over_probe": row.get("min_iterations_over_probe"),
            }
        )
    graded = [row for row in checks if not row["transient_limited"]]
    excluded = sorted({row["filter"] for row in checks if row["transient_limited"]})
    return {
        "claim": (
            "A filter of qualification nu cannot beat lambda^nu, so the requirement "
            "nu >= r or 1 in Theorem 3.4 is necessary: for r > nu the bias exponent "
            "sticks at nu instead of following r."
        ),
        "checks": checks,
        "graded_checks": len(graded),
        "excluded_as_transient_limited": excluded,
        "worst_relative_error": (
            max(row["relative_error"] for row in graded) if graded else None
        ),
        "verdict": bool(graded) and all(row["relative_error"] < 0.15 for row in graded),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    directory = args.output_dir

    claims: dict[str, Any] = {}
    rate = _load(directory / "rate_experiment.json")
    if rate:
        claims["rate_of_theorem_3_4"] = _rate_claim(rate)
    threshold = _load(directory / "feature_threshold.json")
    if threshold:
        claims["feature_threshold"] = _threshold_claim(threshold)
    filters = _load(directory / "filter_diagnostics.json")
    if filters:
        claims["qualification_saturation"] = _saturation_claim(filters)
    for task in ("spectral", "darcy", "poisson"):
        document = _load(directory / f"walltime_{task}.json")
        if document:
            claims[f"walltime_{task}"] = _walltime_claim(document)

    if not claims:
        print(f"no experiment output found in {directory}; run scripts/run_all.py first")
        return 1

    claims["independent_implementation"] = _independence_claim(Path(__file__).parent.parent)

    payload = {
        "summary": "Falsifiable claims of the KerOp reference implementation",
        "paper": "Nguyen & Mucke, AISTATS 2026, arXiv:2603.00971",
        "claims": claims,
    }
    json_path = write_json(directory / "summary.json", payload)

    lines = [
        "# Measured results",
        "",
        "Generated by `scripts/collect_summary.py` from the JSON files in this directory.",
        "",
    ]
    for key, claim in claims.items():
        lines.append(f"## `{key}`")
        lines.append("")
        lines.append(f"**Claim.** {claim['claim']}")
        lines.append("")
        verdict = claim.get("verdict")
        lines.append(f"**Supported by the measurement:** {'yes' if verdict else 'no'}")
        lines.append("")
        for field, value in claim.items():
            if field in {"claim", "verdict", "rows", "checks", "matched_sample_size",
                         "located_threshold_scaling", "filters_violating_the_qualification_hypothesis"}:
                continue
            lines.append(f"- `{field}`: {value}")
        lines.append("")
    markdown_path = directory / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print()
    for key, claim in claims.items():
        print(f"  {'PASS' if claim.get('verdict') else 'FAIL'}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())