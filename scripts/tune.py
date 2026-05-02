"""Grid search lightweight retrieval weights on the public test set."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.retriever as retriever_module  # noqa: E402
from src.pipeline import BISPipeline  # noqa: E402
from src.retriever import normalize_standard_code  # noqa: E402


def eval_set(pipeline: BISPipeline, test_set: list[dict[str, object]]) -> tuple[float, float]:
    """Return Hit@3 and MRR@5 for one configured pipeline."""
    hits_at_3 = 0
    reciprocal_rank_sum = 0.0
    for item in test_set:
        expected = {
            normalize_standard_code(code)
            for code in item.get("expected_standards", [])
        }
        out = pipeline.run_query(str(item.get("query", "")))
        retrieved = [normalize_standard_code(code) for code in out["retrieved_standards"]]
        rank = next((index for index, code in enumerate(retrieved[:5], start=1) if code in expected), None)
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

    count = max(len(test_set), 1)
    return hits_at_3 / count, reciprocal_rank_sum / count


def tune(test_set_path: str = "public_test_set.json") -> tuple[float, float, tuple[float, float, int]]:
    """Search public-set retrieval weights and print the best configuration."""
    with Path(test_set_path).open(encoding="utf-8") as file:
        test_set = json.load(file)

    grid = {
        "BOOST_EXPLICIT_CODE": [0.20, 0.30, 0.40],
        "BOOST_TITLE_TOKEN": [0.03, 0.05, 0.08],
        "RRF_K": [40, 60, 90],
    }

    pipeline = BISPipeline()
    best = (0.0, 0.0, (retriever_module.BOOST_EXPLICIT_CODE, retriever_module.BOOST_TITLE_TOKEN, retriever_module.RRF_K))
    for explicit_boost, title_boost, rrf_k in itertools.product(*grid.values()):
        retriever_module.BOOST_EXPLICIT_CODE = explicit_boost
        retriever_module.BOOST_TITLE_TOKEN = title_boost
        retriever_module.RRF_K = rrf_k
        pipeline.retriever.use_dense = False
        hit3, mrr5 = eval_set(pipeline, test_set)
        print(
            f"BEC={explicit_boost:.2f} BTT={title_boost:.2f} RRF={rrf_k} "
            f"-> H@3={hit3:.2%} MRR@5={mrr5:.4f}"
        )
        if hit3 + mrr5 > best[0] + best[1]:
            best = (hit3, mrr5, (explicit_boost, title_boost, rrf_k))

    print(f"\nBest: H@3={best[0]:.2%}, MRR@5={best[1]:.4f}, params={best[2]}")
    return best


if __name__ == "__main__":
    tune()
