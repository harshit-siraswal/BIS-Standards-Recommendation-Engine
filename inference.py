"""Judge-facing inference entrypoint for BIS standards recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run import ensure_artifacts, load_input
from src.pipeline import BISPipeline


def run_inference(
    input_path: str | Path,
    output_path: str | Path,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Run the submitted RAG pipeline and write the strict judge output schema."""
    ensure_artifacts(rebuild=rebuild)
    pipeline = BISPipeline(use_reranker=False)

    results: list[dict[str, Any]] = []
    for item in load_input(input_path):
        output = pipeline.run_query(str(item.get("query", "")))
        result = {
            "id": item.get("id"),
            "retrieved_standards": output["retrieved_standards"],
            "latency_seconds": output["latency_seconds"],
        }
        if "query" in item:
            result["query"] = item["query"]
        if "expected_standards" in item:
            result["expected_standards"] = item["expected_standards"]
        results.append(result)

    output_file = Path(output_path)
    if output_file.parent != Path("."):
        output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BIS standard inference for hackathon judging.")
    parser.add_argument("--input", required=True, help="Path to hidden/private query JSON.")
    parser.add_argument("--output", required=True, help="Path to write judge-format result JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild parser and index artifacts before inference.")
    args = parser.parse_args(argv)

    run_inference(args.input, args.output, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
