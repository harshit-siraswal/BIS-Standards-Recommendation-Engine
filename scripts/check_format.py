"""Validate result JSON shape before submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_KEYS = {"id", "query", "expected_standards", "retrieved_standards", "latency_seconds"}


def check_format(path: str | Path) -> None:
    """Raise ValueError when a results file does not match the expected schema."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("results must be a JSON list")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"row {index} must be an object")
        missing = REQUIRED_KEYS - set(item)
        if missing:
            raise ValueError(f"row {index} missing keys: {sorted(missing)}")
        if not isinstance(item["retrieved_standards"], list) or len(item["retrieved_standards"]) != 5:
            raise ValueError(f"row {index} must contain exactly 5 retrieved_standards")
        if not all(isinstance(code, str) and code for code in item["retrieved_standards"]):
            raise ValueError(f"row {index} contains an invalid retrieved standard")
        if not isinstance(item["latency_seconds"], (int, float)):
            raise ValueError(f"row {index} latency_seconds must be numeric")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check BIS result JSON schema.")
    parser.add_argument("results", help="Path to results JSON.")
    args = parser.parse_args(argv)

    try:
        check_format(args.results)
    except ValueError as exc:
        print(f"format check failed: {exc}", file=sys.stderr)
        return 1
    print("format check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
