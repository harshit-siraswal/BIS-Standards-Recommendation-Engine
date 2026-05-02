"""Convenience wrapper around eval_script.py for local evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_script import evaluate_results, load_results  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BIS results file.")
    parser.add_argument("--results", default="results.json", help="Path to results JSON.")
    args = parser.parse_args(argv)

    metrics = evaluate_results(load_results(args.results))
    print(f"Hit Rate @3 : {metrics['hit_rate_at_3']:.2%}")
    print(f"MRR @5      : {metrics['mrr_at_5']:.4f}")
    print(f"Avg Latency : {metrics['avg_latency_seconds']:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
