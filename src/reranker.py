"""Cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, List, Mapping, Optional, Sequence, TYPE_CHECKING

try:
    from src.retriever import RetrievalResult
except ImportError:  # pragma: no cover - used when Plan 04 is not present yet
    @dataclass
    class RetrievalResult:
        """Fallback retrieval result shape used by the reranker tests."""

        standard: Mapping[str, Any]
        score: float
        sources: dict[str, Any] = field(default_factory=dict)

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from sentence_transformers import CrossEncoder


RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_BATCH_SIZE = 32
FINAL_TOP_K = 5
RERANK_INPUT_K = 20
RERANK_WEIGHT = 0.7
MAX_DOCUMENT_CHARS = 512


class Reranker:
    """Rerank top retrieval candidates with a cross-encoder model."""

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        model: Optional["CrossEncoder"] = None,
        batch_size: int = RERANK_BATCH_SIZE,
        input_k: int = RERANK_INPUT_K,
        rerank_weight: float = RERANK_WEIGHT,
    ) -> None:
        if not 0 <= rerank_weight <= 1:
            raise ValueError("rerank_weight must be between 0 and 1")

        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._input_k = input_k
        self._rerank_weight = rerank_weight

    @property
    def model(self) -> "CrossEncoder":
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc

            print(f"Loading reranker: {self._model_name}")
            self._model = CrossEncoder(self._model_name, max_length=MAX_DOCUMENT_CHARS)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = FINAL_TOP_K,
    ) -> List[RetrievalResult]:
        """Re-score candidates with the cross-encoder and return the best results."""
        if top_k <= 0 or not candidates:
            return []

        rerank_candidates = list(candidates[: self._input_k])
        pairs = [[query, self._candidate_document_text(candidate)] for candidate in rerank_candidates]

        scores = self.model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        retrieval_weight = 1.0 - self._rerank_weight
        reranked: list[RetrievalResult] = []
        for index, candidate in enumerate(rerank_candidates):
            new_score = self._rerank_weight * float(scores[index]) + retrieval_weight * float(candidate.score)
            reranked.append(replace(candidate, score=new_score))

        return sorted(reranked, key=lambda result: -result.score)[:top_k]

    def rerank_safe(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = FINAL_TOP_K,
    ) -> List[RetrievalResult]:
        """Rerank with retrieval-order fallback if the cross-encoder is unavailable."""
        try:
            return self.rerank(query, candidates, top_k)
        except Exception as exc:  # pragma: no cover - exact failures depend on runtime
            print(f"Reranker failed: {exc}. Falling back to retrieval ranking.")
            return sorted(candidates, key=lambda result: -result.score)[:top_k]

    @classmethod
    def _candidate_document_text(cls, candidate: RetrievalResult) -> str:
        standard = candidate.standard
        is_code = cls._standard_value(standard, "is_code")
        title = cls._standard_value(standard, "title")
        scope = cls._standard_value(standard, "scope")
        body = cls._standard_value(standard, "body")
        keywords = cls._standard_value(standard, "keywords", default=[])

        if isinstance(keywords, Sequence) and not isinstance(keywords, (str, bytes, bytearray)):
            keyword_text = " ".join(str(keyword) for keyword in keywords[:8])
        else:
            keyword_text = ""
        text = f"{is_code} {title}. {scope or body} {keyword_text}"
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_DOCUMENT_CHARS]

    @staticmethod
    def _standard_value(standard: Any, key: str, default: Any = "") -> Any:
        if isinstance(standard, dict):
            return standard.get(key, default)
        return getattr(standard, key, default)
