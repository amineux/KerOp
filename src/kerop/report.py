"""Serialization helpers for experiment output.

Every experiment writes both a JSON file, holding the complete nested result
including settings and provenance, and one or more flat CSV files holding the
tabular rows a plotting script or spreadsheet would want.  The JSON is the
record of what was run; the CSVs are the numbers.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["provenance", "write_json", "write_csv", "flatten"]


def provenance() -> dict[str, Any]:
    """Record enough about the environment to interpret timing numbers later.

    Wall-clock comparisons are only meaningful alongside the machine they were
    measured on, so the CPU count and BLAS configuration are captured with the
    results.
    """
    import kerop

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""

    blas: dict[str, Any] = {}
    try:
        config = np.__config__.show(mode="dicts")  # type: ignore[call-arg]
        blas = {
            "name": config.get("Build Dependencies", {}).get("blas", {}).get("name", ""),
            "version": config.get("Build Dependencies", {}).get("blas", {}).get("version", ""),
        }
    except Exception:
        blas = {}

    return {
        "kerop_version": kerop.__version__,
        "paper": "arXiv:2603.00971",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "blas": blas,
    }


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars and arrays to JSON-representable Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as indented JSON, adding provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"provenance": provenance(), **_to_builtin(dict(payload))}
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    return path


def flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten one level of nesting into ``parent.child`` column names.

    Experiment rows carry small nested dictionaries such as the chosen
    configuration of each method; flattening keeps the CSV readable without
    discarding them.
    """
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{name}."))
        elif isinstance(value, (list, tuple)) and not isinstance(value, str):
            flat[name] = ";".join(str(_to_builtin(item)) for item in value)
        else:
            flat[name] = _to_builtin(value)
    return flat


def write_csv(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as CSV, using the union of all keys as the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten(row) for row in rows]
    if not flattened:
        path.write_text("")
        return path
    columns: list[str] = []
    for record in flattened:
        for key in record:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in flattened:
            writer.writerow({column: record.get(column, "") for column in columns})
    return path


def summarize_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Render a small fixed-width table for printing to a terminal."""
    if not rows:
        return "(no rows)"
    widths = {column: max(len(column), 8) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(_format_cell(row.get(column))))
    header = "  ".join(column.rjust(widths[column]) for column in columns)
    lines = [header, "  ".join("-" * widths[column] for column in columns)]
    for row in rows:
        lines.append(
            "  ".join(_format_cell(row.get(column)).rjust(widths[column]) for column in columns)
        )
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.3e}"
        return f"{value:.5g}"
    return str(value)