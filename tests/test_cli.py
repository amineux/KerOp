"""Tests for the ``kerop`` command line."""

from __future__ import annotations

import pytest

from kerop.cli import main


def test_info_prints_the_citation(capsys) -> None:
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "arXiv:2603.00971" in output
    assert "no claim of" in output


def test_theory_prints_the_prescriptions(capsys) -> None:
    assert main(["theory", "--r", "0.5", "--b", "1.0", "--n", "10000"]) == 0
    output = capsys.readouterr().out
    # The well-specified case: sqrt(n) iterations and sqrt(n) log n features.
    assert "gradient descent iterations      100" in output
    assert "risk exponent                    -0.250000" in output


def test_theory_rejects_an_illegal_instance(capsys) -> None:
    assert main(["theory", "--r", "0.2", "--b", "0.4"]) == 2
    assert "easy-learning" in capsys.readouterr().err


def test_theory_lists_only_eligible_filters(capsys) -> None:
    """At :math:`r>1` Tikhonov no longer meets the qualification requirement."""
    main(["theory", "--r", "1.5", "--b", "0.5", "--n", "5000"])
    output = capsys.readouterr().out
    eligible = output.split("eligible filters")[1]
    assert "tikhonov" not in eligible.replace("iterated_tikhonov", "")
    assert "landweber" in eligible


def test_filters_table_reports_measured_and_analytic_qualification(capsys) -> None:
    assert main(["filters", "--filter", "tikhonov"]) == 0
    output = capsys.readouterr().out
    assert "tikhonov" in output
    assert "qualification" in output


@pytest.mark.parametrize("dataset", ["poisson", "darcy"])
def test_demo_learns_the_operator(dataset: str, capsys) -> None:
    assert (
        main(
            [
                "demo",
                dataset,
                "--n-train",
                "200",
                "--n-test",
                "150",
                "--n-points",
                "13",
                "--filter",
                "nu_method",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "excess risk" in output
    relative_line = next(
        line for line in output.splitlines() if "relative to the zero predictor" in line
    )
    percentage = float(relative_line.split()[-1].rstrip("%"))
    assert percentage < 25.0


def test_demo_reports_the_memory_the_exact_kernel_would_need(capsys) -> None:
    main(["demo", "poisson", "--n-train", "200", "--n-points", "13", "--n-features", "40"])
    output = capsys.readouterr().out
    operator = float(
        next(line for line in output.splitlines() if "operator memory" in line).split()[-2]
    )
    exact = float(
        next(line for line in output.splitlines() if "exact kernel would need" in line).split()[-2]
    )
    assert exact > operator


def test_demo_json_output_is_valid(capsys) -> None:
    import json

    main(["demo", "poisson", "--n-train", "150", "--n-points", "9", "--json"])
    output = capsys.readouterr().out
    document = json.loads(output[output.index("{") :])
    assert document["dataset"] == "poisson"
    assert document["excess_risk"] > 0.0