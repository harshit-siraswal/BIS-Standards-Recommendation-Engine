"""End-to-end BIS recommendation pipeline."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from src.query_processor import QueryProcessor
from src.reranker import FINAL_TOP_K, Reranker
from src.retriever import CANDIDATE_TOP_K, HybridRetriever, RetrievalResult


MAX_QUERY_CHARS = 500
SEED = 42


def set_random_seeds(seed: int = SEED) -> None:
    """Set available random seeds for deterministic local execution."""
    random.seed(seed)
    if "numpy" in sys.modules:
        import numpy as np

        np.random.seed(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)


class BISPipeline:
    """Process a raw query and return the top BIS standards."""

    def __init__(
        self,
        catalog_path: str | Path = "data/standards_catalog.json",
        use_reranker: bool = False,
    ) -> None:
        set_random_seeds()
        self.query_processor = QueryProcessor()
        self.retriever = HybridRetriever(catalog_path)
        self.reranker = Reranker() if use_reranker else None

    def run_query(self, query: str, top_k: int = FINAL_TOP_K) -> dict[str, Any]:
        """Run one query and return retrieved standards plus latency."""
        start = time.perf_counter()
        clean_query = self._prepare_query(query)
        processed = self.query_processor.process(clean_query)
        candidates = self.retriever.retrieve(processed, top_k=max(CANDIDATE_TOP_K, top_k))
        ranked = self._rank(clean_query, candidates, top_k)
        latency = time.perf_counter() - start
        return {
            "query": str(query or ""),
            "retrieved_standards": [self._code(result.standard) for result in ranked],
            "latency_seconds": round(latency, 4),
        }

    def _rank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if self.reranker is None:
            return sorted(candidates, key=lambda result: -result.score)[:top_k]
        return self.reranker.rerank_safe(query, candidates, top_k=top_k)

    @staticmethod
    def _prepare_query(query: str) -> str:
        return str(query or "")[:MAX_QUERY_CHARS]

    @staticmethod
    def _code(standard: Mapping[str, Any]) -> str:
        return str(standard.get("is_code", ""))
