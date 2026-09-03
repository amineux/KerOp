r"""Command-line interface.

Subcommands
-----------
``kerop demo``
    Learn a PDE solution operator - Poisson or Darcy - with operator-valued NTK
    random features and a chosen spectral filter, printing the excess risk, the
    Theorem 3.4 prescriptions, and the cost.

``kerop filters``
    Print the measured constants and qualification of the spectral filter
    families.

``kerop theory``
    Print what Theorem 3.4 prescribes for a given :math:`(r,b,n)`.

``kerop info``
    Print the version and the citation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import numpy as np

from kerop import __version__, theory
from kerop.data.pde import DATASETS, make_dataset
from kerop.estimators import VectorValuedRFRegressor
from kerop.features import OperatorNTKFeatures
from kerop.filters import FILTER_REGISTRY, filter_diagnostics, make_filter, measure_qualification
from kerop.metrics import excess_risk, relative_error

__all__ = ["main"]


def _demo(args: argparse.Namespace) -> int:
    dataset = make_dataset(
        args.dataset,
        n_points=args.n_points,
        n_modes=args.n_modes,
        noise_std=args.noise_std,
    )
    rng = np.random.default_rng(args.seed)
    train = dataset.sample(args.n_train, rng)
    test = dataset.sample(args.n_test, rng)
    train_inputs = dataset.lift(train.fields)
    test_inputs = dataset.lift(test.fields)

    if args.n_features is None:
        # Well-specified prescription of Theorem 3.4: M of order sqrt(n) * p.
        n_features = int(math.ceil(math.sqrt(args.n_train) * dataset.n_summands))
    else:
        n_features = args.n_features

    features = OperatorNTKFeatures(
        dataset.feature_dim,
        dataset.n_points,
        n_features,
        rng,
        output_scale=dataset.output_scale(),
    )

    filter_kwargs: dict[str, Any] = {}
    if args.filter == "nu_method":
        filter_kwargs["nu"] = args.nu
    if args.filter == "iterated_tikhonov":
        filter_kwargs["order"] = args.order
    if args.filter == "heavy_ball":
        filter_kwargs["momentum"] = args.momentum

    estimator = VectorValuedRFRegressor(
        features, args.filter, args.lam, filter_kwargs=filter_kwargs
    ).fit(train_inputs, train.outputs)
    predictions = estimator.predict(test_inputs)

    risk = excess_risk(predictions, test.targets)
    relative = relative_error(predictions, test.targets)
    baseline = excess_risk(np.zeros_like(test.targets), test.targets)

    print(f"KerOp demo: the {args.dataset} solution operator")
    print("-" * 62)
    print(f"  grid points n_x (= d_v)        {dataset.n_points}")
    print(f"  feature dimension d_tilde      {dataset.feature_dim}")
    print(f"  summands p = 1 + d_tilde       {dataset.n_summands}")
    print(f"  training pairs n               {args.n_train}")
    print(f"  random features M              {n_features}")
    print(f"  coefficient dimension pM       {features.coefficient_dim}")
    print(f"  filter                         {estimator.filter!r}")
    print()
    print(f"  excess risk ||G_rho - S_M F||  {risk:.6g}")
    print(f"  relative to ||G_rho||          {100 * relative:.3f}%")
    print(f"  relative to the zero predictor {100 * risk / baseline:.3f}%")
    print()
    print(f"  fit wall-clock                 {estimator.report.fit_seconds:.3f}s")
    print(f"    assembling the operator      {estimator.report.assemble_seconds:.3f}s")
    print(f"    applying the filter          {estimator.report.solve_seconds:.3f}s")
    print(f"  operator memory                {estimator.report.peak_operator_bytes / 1e6:.1f} MB")
    print(f"  exact kernel would need        {(args.n_train * dataset.n_points) ** 2 * 8 / 1e6:.1f} MB")
    print()

    prescription = theory.prescribe(
        args.n_train, args.r, args.b, n_summands=dataset.n_summands
    )
    print(f"  Theorem 3.4 at r={args.r}, b={args.b} (assumed, not measured, for this task):")
    print(f"    lambda_n proportional to     n^-{theory.regularization_exponent(args.r, args.b):.4f}")
    print(f"    M_n at least                 {prescription.n_features}")
    print(f"    gradient descent iterations  {prescription.iterations}")
    print(f"    nu-method iterations         {prescription.accelerated_iterations}")
    print(f"    predicted risk exponent      {prescription.risk_bound_exponent:.4f}")
    if args.json:
        print()
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "n_train": args.n_train,
                    "n_features": n_features,
                    "coefficient_dim": features.coefficient_dim,
                    "excess_risk": risk,
                    "relative_error": relative,
                    "fit_seconds": estimator.report.fit_seconds,
                },
                indent=2,
            )
        )
    return 0


def _filters(args: argparse.Namespace) -> int:
    families: list[tuple[str, dict[str, Any]]] = [
        ("tikhonov", {}),
        ("iterated_tikhonov", {"order": 2}),
        ("iterated_tikhonov", {"order": 3}),
        ("landweber", {}),
        ("cutoff", {}),
        ("heavy_ball", {"momentum": 0.9}),
        ("nu_method", {"nu": 1.0}),
        ("nu_method", {"nu": 2.0}),
        ("nu_method", {"nu": 3.0}),
    ]
    if args.filter:
        families = [(name, options) for name, options in families if name == args.filter]
        if not families:
            families = [(args.filter, {})]

    print("Spectral regularization families (Definition 2.2 and qualification (2.10))")
    print(
        f"{'family':<30s} {'D':>7s} {'E':>7s} {'c_0':>7s} "
        f"{'nu measured':>12s} {'nu analytic':>12s} {'T at lam=1e-3':>14s}"
    )
    print("-" * 96)
    for name, options in families:
        diagnostics = filter_diagnostics(name, **options)
        report = measure_qualification(name, q_grid=np.arange(0.5, 4.01, 0.5), **options)
        instance = make_filter(name, 1e-3, **options)
        analytic = instance.qualification
        analytic_text = "inf" if math.isinf(analytic) else f"{analytic:.1f}"
        measured = report.nu_estimate
        measured_text = f">= {measured:.1f}" if measured >= 4.0 else f"{measured:.2f}"
        iterations = getattr(instance, "iterations", None)
        label = name + (f"{sorted(options.items())}" if options else "")
        print(
            f"{label:<30s} {diagnostics.D:7.3f} {diagnostics.E:7.3f} {diagnostics.c0:7.3f} "
            f"{measured_text:>12s} {analytic_text:>12s} "
            f"{'-' if iterations is None else iterations:>14}"
        )
    print()
    print("Theorem 3.4 needs qualification nu >= max(r, 1); families with nu = 1 such as")
    print("Tikhonov therefore saturate for source exponents r > 1.  The measured column is")
    print("capped by the probe range q <= 4.")
    return 0


def _theory(args: argparse.Namespace) -> int:
    theory.check_assumptions(args.r, args.b)
    prescription = theory.prescribe(
        args.n, args.r, args.b, n_summands=args.n_summands
    )
    print(f"Theorem 3.4 at r={args.r}, b={args.b}, n={args.n}, p={args.n_summands}")
    print("-" * 62)
    print(f"  easy-learning condition 2r+b>1   {2 * args.r + args.b:.3f}")
    print(f"  minimum sample size n_0          {theory.min_sample_size(args.r, args.b):.1f}")
    print(f"  n >= n_0                         {prescription.meets_min_sample_size}")
    print(f"  lambda exponent                  -{theory.regularization_exponent(args.r, args.b):.6f}")
    print(f"  lambda_n (C = 1)                 {prescription.lam:.6g}")
    print(f"  risk exponent                    {prescription.risk_bound_exponent:.6f}")
    print(f"  feature exponent                 {theory.feature_exponent(args.r, args.b):.6f}")
    print(f"  M_n at least                     {prescription.n_features}")
    print(f"  gradient descent iterations      {prescription.iterations}")
    print(f"  nu-method iterations             {prescription.accelerated_iterations}")
    print(f"  required qualification           {max(args.r, 1.0):.3f}")
    print(
        f"  eligible filters                 "
        f"{', '.join(sorted(name for name, cls in FILTER_REGISTRY.items() if _eligible(name, args.r)))}"
    )
    return 0


def _eligible(name: str, r: float) -> bool:
    """Whether a family's default qualification meets the theorem's requirement."""
    try:
        instance = make_filter(name, 0.1)
    except TypeError:
        return False
    return bool(instance.qualification >= max(r, 1.0))


def _info(args: argparse.Namespace) -> int:
    import kerop

    print(f"kerop {__version__}")
    print()
    print("Reference implementation of:")
    print(f"  {kerop.PAPER_CITATION}")
    print()
    print("This package implements theory developed by those authors; no claim of")
    print("originality is made for any of the theorems.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``kerop`` console script."""
    parser = argparse.ArgumentParser(
        prog="kerop",
        description=(
            "Random features for operator-valued kernels with spectral filtering "
            "(Nguyen & Mucke, AISTATS 2026, arXiv:2603.00971)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"kerop {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="learn a PDE solution operator with operator-valued NTK features"
    )
    demo.add_argument("dataset", choices=sorted(DATASETS), nargs="?", default="poisson")
    demo.add_argument("--n-train", type=int, default=800)
    demo.add_argument("--n-test", type=int, default=500)
    demo.add_argument("--n-points", type=int, default=33, help="collocation points n_x")
    demo.add_argument("--n-modes", type=int, default=12, help="modes in the input field")
    demo.add_argument("--n-features", type=int, default=None, help="default: sqrt(n) * p")
    demo.add_argument("--noise-std", type=float, default=0.0)
    demo.add_argument("--filter", choices=sorted(FILTER_REGISTRY), default="nu_method")
    demo.add_argument("--lam", type=float, default=1e-5)
    demo.add_argument("--nu", type=float, default=2.0)
    demo.add_argument("--order", type=int, default=2)
    demo.add_argument("--momentum", type=float, default=0.9)
    demo.add_argument("--r", type=float, default=0.5, help="assumed source exponent")
    demo.add_argument("--b", type=float, default=1.0, help="assumed capacity exponent")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--json", action="store_true", help="also print a JSON record")
    demo.set_defaults(handler=_demo)

    filters = subparsers.add_parser(
        "filters", help="measure the constants and qualification of the spectral filters"
    )
    filters.add_argument("--filter", choices=sorted(FILTER_REGISTRY), default=None)
    filters.set_defaults(handler=_filters)

    prescriptions = subparsers.add_parser(
        "theory", help="print the Theorem 3.4 prescriptions for given (r, b, n)"
    )
    prescriptions.add_argument("--r", type=float, default=0.5)
    prescriptions.add_argument("--b", type=float, default=1.0)
    prescriptions.add_argument("--n", type=int, default=10_000)
    prescriptions.add_argument("--n-summands", type=int, default=1, help="the factor p")
    prescriptions.set_defaults(handler=_theory)

    info = subparsers.add_parser("info", help="print the version and citation")
    info.set_defaults(handler=_info)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())