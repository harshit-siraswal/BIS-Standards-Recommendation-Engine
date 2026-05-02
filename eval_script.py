"""Evaluate BIS recommendation output against expected standards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def normalize_standard(code: str) -> str:
    """Normalize standard codes using the public matching convention."""
    return str(code or "").replace(" ", "").lower()


def evaluate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Hit@3, MRR@5, average latency, and per-query ranks."""
    if not results:
        return {"hit_rate_at_3": 0.0, "mrr_at_5": 0.0, "avg_latency_seconds": 0.0, "details": []}

    hits_at_3 = 0
    reciprocal_rank_sum = 0.0
    details: list[dict[str, Any]] = []

    for item in results:
        expected = {normalize_standard(code) for code in item.get("expected_standards", [])}
        retrieved = [normalize_standard(code) for code in item.get("retrieved_standards", [])]
        rank = next((index for index, code in enumerate(retrieved[:5], start=1) if code in expected), None)
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
        details.append(
            {
                "id": item.get("id"),
                "rank": rank,
                "hit_at_3": rank is not None and rank <= 3,
                "latency_seconds": float(item.get("latency_seconds", 0.0)),
            }
        )

    count = len(results)
    return {
        "hit_rate_at_3": hits_at_3 / count,
        "mrr_at_5": reciprocal_rank_sum / count,
        "avg_latency_seconds": mean(float(item.get("latency_seconds", 0.0)) for item in results),
        "details": details,
    }


def load_results(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Results JSON must contain a list")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BIS recommendation results.")
    parser.add_argument("--results", required=True, help="Path to results JSON.")
    args = parser.parse_args(argv)

    metrics = evaluate_results(load_results(args.results))
    print(f"Hit Rate @3 : {metrics['hit_rate_at_3']:.2%}")
    print(f"MRR @5      : {metrics['mrr_at_5']:.4f}")
    print(f"Avg Latency : {metrics['avg_latency_seconds']:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
