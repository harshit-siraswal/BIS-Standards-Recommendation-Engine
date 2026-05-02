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
_NUMPY_RNG: Any | None = None


def set_random_seeds(seed: int = SEED) -> None:
    """Set available random seeds for deterministic local execution."""
    global _NUMPY_RNG
    random.seed(seed)
    if "numpy" in sys.modules:
        import numpy as np

        _NUMPY_RNG = np.random.default_rng(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)


class BISPipeline:
    """Process a raw query and return the top BIS standards."""

    def __init__(
        self,
        artifacts_dir: str | Path = "data",
        use_reranker: bool = False,
        catalog_path: str | Path | None = None,
        boost_explicit_code: float | None = None,
        boost_title_token: float | None = None,
        rrf_k: int | None = None,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            artifacts_dir: Directory containing retriever artifacts built by the
                indexer (for example ``standards_indexed.json`` and FAISS files).
            use_reranker: Whether to enable the reranker.
            catalog_path: Backward-compatible alias for older callers that passed
                a catalog file path such as ``data/standards_catalog.json``. When
                provided, its parent directory is used as the artifacts directory.
            boost_explicit_code: Optional retriever explicit-code boost override.
            boost_title_token: Optional retriever title-token boost override.
            rrf_k: Optional retriever RRF fusion constant override.
        """
        set_random_seeds()
        self.query_processor = QueryProcessor()
        resolved_artifacts_dir = self._resolve_artifacts_dir(
            artifacts_dir=artifacts_dir,
            catalog_path=catalog_path,
        )
        retriever_kwargs: dict[str, Any] = {}
        if boost_explicit_code is not None:
            retriever_kwargs["boost_explicit_code"] = boost_explicit_code
        if boost_title_token is not None:
            retriever_kwargs["boost_title_token"] = boost_title_token
        if rrf_k is not None:
            retriever_kwargs["rrf_k"] = rrf_k
        self.retriever = HybridRetriever(resolved_artifacts_dir, **retriever_kwargs)
        self.reranker = Reranker() if use_reranker else None

    @staticmethod
    def _resolve_artifacts_dir(
        artifacts_dir: str | Path,
        catalog_path: str | Path | None = None,
    ) -> Path:
        if catalog_path is None:
            return Path(artifacts_dir)

        legacy_path = Path(catalog_path)
        return legacy_path.parent if legacy_path.suffix else legacy_path

    def run_query(self, query: str, top_k: int = FINAL_TOP_K) -> dict[str, Any]:
        """Run one query and return retrieved standards plus latency."""
        start = time.perf_counter()
        clean_query = self._prepare_query(query)
        processed = self.query_processor.process(clean_query)
        if processed.out_of_scope:
            latency = time.perf_counter() - start
            return {
                "query": clean_query,
                "retrieved_standards": [],
                "latency_seconds": round(latency, 4),
                "out_of_scope": True,
            }
        candidates = self.retriever.retrieve(processed, top_k=max(CANDIDATE_TOP_K, top_k))
        ranked = self._rank(clean_query, candidates, top_k)
        latency = time.perf_counter() - start
        return {
            "query": clean_query,
            "retrieved_standards": [self._code(result.standard) for result in ranked],
            "latency_seconds": round(latency, 4),
            "out_of_scope": False,
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
