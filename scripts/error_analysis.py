"""Analyze public-test misses and print query-processing diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import BISPipeline  # noqa: E402
from src.query_processor import QueryProcessor  # noqa: E402
from src.retriever import normalize_standard_code  # noqa: E402


def analyze(test_set_path: str = "public_test_set.json") -> list[dict[str, object]]:
    """Run error analysis and return problematic query records."""
    with Path(test_set_path).open(encoding="utf-8") as file:
        test_set = json.load(file)

    pipeline = BISPipeline()
    qp = QueryProcessor()
    misses: list[dict[str, object]] = []

    for item in test_set:
        expected = {normalize_standard_code(code) for code in item["expected_standards"]}
        out = pipeline.run_query(item["query"])
        retrieved = [normalize_standard_code(code) for code in out["retrieved_standards"]]
        ranks = [
            retrieved.index(code) + 1 if code in retrieved else None
            for code in expected
        ]

        if any(rank is None or rank > 3 for rank in ranks):
            processed = qp.process(item["query"])
            misses.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "expected": sorted(expected),
                    "retrieved": retrieved[:5],
                    "ranks": ranks,
                    "expanded": processed.expanded[:240],
                    "tokens": processed.tokens[:20],
                    "explicit_codes": processed.explicit_codes,
                }
            )

    print(f"{len(misses)} problematic queries")
    for miss in misses:
        print(f"\n{miss['id']}: {miss['query']}")
        print(f"  expected: {miss['expected']}")
        print(f"  top-5: {miss['retrieved']}")
        print(f"  ranks: {miss['ranks']}")
        print(f"  expanded: {miss['expanded']}")
        print(f"  tokens: {miss['tokens']}")
        print(f"  codes: {miss['explicit_codes']}")
    return misses


if __name__ == "__main__":
    analyze()
