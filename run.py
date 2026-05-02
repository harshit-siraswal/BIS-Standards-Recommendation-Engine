"""CLI entry point for generating BIS recommendation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.indexer import build_all_indexes
from src.parser import parse_pdf_to_catalog
from src.pipeline import BISPipeline


INDEX_ARTIFACTS = (
    Path("data/standards_indexed.json"),
    Path("data/faiss.index"),
    Path("data/bm25.pkl"),
    Path("data/codes.json"),
)


def load_input(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluator-style query rows."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of query objects")
    return data


def run_file(
    input_path: str | Path,
    output_path: str | Path,
    rebuild: bool = False,
    use_reranker: bool = False,
) -> list[dict[str, Any]]:
    """Run the pipeline over an input test set and write result JSON."""
    ensure_artifacts(rebuild=rebuild)

    pipeline = BISPipeline(use_reranker=use_reranker)
    results: list[dict[str, Any]] = []
    for item in load_input(input_path):
        out = pipeline.run_query(str(item.get("query", "")))
        results.append(
            {
                "id": item.get("id"),
                "query": item.get("query", ""),
                "expected_standards": item.get("expected_standards", []),
                "retrieved_standards": out["retrieved_standards"],
                "latency_seconds": out["latency_seconds"],
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True) if output.parent != Path(".") else None
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def ensure_artifacts(rebuild: bool = False) -> None:
    """Build parser and index artifacts when requested or missing."""
    catalog = Path("data/standards_catalog.json")
    if rebuild or not catalog.exists():
        parse_pdf_to_catalog("dataset.pdf", "data/standards_catalog.json")
    if rebuild or any(not path.exists() for path in INDEX_ARTIFACTS):
        build_all_indexes(catalog_path=catalog, output_dir="data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BIS standards recommendation engine.")
    parser.add_argument("--input", required=True, help="Path to evaluator input JSON.")
    parser.add_argument("--output", required=True, help="Path to write results JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild data/standards_catalog.json from dataset.pdf.")
    parser.add_argument("--use-reranker", action="store_true", help="Enable optional cross-encoder reranking.")
    args = parser.parse_args(argv)

    run_file(args.input, args.output, rebuild=args.rebuild, use_reranker=args.use_reranker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
