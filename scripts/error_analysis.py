"""Analyze public-test misses and print query-processing diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import BISPipeline  # noqa: E402
from src.query_processor import QueryProcessor  # noqa: E402
from src.retriever import normalize_standard_code  # noqa: E402

MAX_EXPANDED_CHARS = 240
MAX_TOKENS = 20
MAX_EXPLICIT_CODES = 5


def analyze(test_set_path: str = "public_test_set.json") -> list[dict[str, object]]:
    """Run error analysis and return problematic query records."""
    try:
        with Path(test_set_path).open(encoding="utf-8") as file:
            test_set: list[dict[str, Any]] = json.load(file)
    except FileNotFoundError:
        print(f"Error: test set file not found: {test_set_path}", file=sys.stderr)
        return []
    except PermissionError as exc:
        print(f"Error: cannot read test set file {test_set_path}: {exc}", file=sys.stderr)
        return []
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {test_set_path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(test_set, list):
        print(f"Error: test set must be a JSON list: {test_set_path}", file=sys.stderr)
        return []

    pipeline = BISPipeline()
    qp = QueryProcessor()
    misses: list[dict[str, object]] = []

    for item in test_set:
        expected_list = [
            normalize_standard_code(code)
            for code in item.get("expected_standards", [])
        ]
        if len(expected_list) != len(set(expected_list)):
            print(f"Warning: {item.get('id', '<unknown>')} has duplicate expected standards", file=sys.stderr)
        expected = set(expected_list)
        try:
            out = pipeline.run_query(str(item.get("query", "")))
        except Exception as exc:
            print(f"Error processing {item.get('id', '<unknown>')}: {exc}", file=sys.stderr)
            continue
        retrieved = [normalize_standard_code(code) for code in out["retrieved_standards"]]
        ranks = [
            retrieved.index(code) + 1 if code in retrieved else None
            for code in expected
        ]

        if any(rank is None or rank > 3 for rank in ranks):
            processed = qp.process(str(item.get("query", "")))
            misses.append(
                {
                    "id": item.get("id"),
                    "query": item.get("query", ""),
                    "expected": sorted(expected),
                    "retrieved": retrieved[:5],
                    "ranks": ranks,
                    "expanded": processed.expanded[:MAX_EXPANDED_CHARS],
                    "tokens": processed.tokens[:MAX_TOKENS],
                    "explicit_codes": processed.explicit_codes[:MAX_EXPLICIT_CODES],
                }
            )

    return misses


def print_report(misses: list[dict[str, object]]) -> None:
    """Print a human-readable report for analyze() output."""
    print(f"{len(misses)} problematic queries")
    for miss in misses:
        print(f"\n{miss['id']}: {miss['query']}")
        print(f"  expected: {miss['expected']}")
        print(f"  top-5: {miss['retrieved']}")
        print(f"  ranks: {miss['ranks']}")
        print(f"  expanded: {miss['expanded']}")
        print(f"  tokens: {miss['tokens']}")
        print(f"  codes: {miss['explicit_codes']}")


if __name__ == "__main__":
    print_report(analyze())
