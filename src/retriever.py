"""Hybrid dense/sparse retriever for BIS standards."""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.indexer import EMBED_MODEL, normalize_is_code, tokenize
from src.query_processor import ProcessedQuery


DATA_DIR = Path("data")
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
RRF_K = 60
BOOST_EXPLICIT_CODE = 0.30
BOOST_TITLE_TOKEN = 0.05
BOOST_TITLE_CAP = 0.20
BOOST_EXACT_TITLE = 0.25
BOOST_KEYWORD = 0.03
BOOST_KEYWORD_CAP = 0.15
BOOST_NUMBER_TOKEN = 0.05
BOOST_NUMBER_CAP = 0.15
RETRIEVE_TOP_K = 20


@dataclass
class RetrievalResult:
    """A scored standard returned by the retrieval stage."""

    standard: Mapping[str, Any]
    score: float
    sources: dict[str, Any] = field(default_factory=dict)
    source: str = "hybrid"


def normalize_standard_code(code: str) -> str:
    """Normalize an IS code for evaluator-compatible comparisons."""
    return normalize_is_code(code)


class HybridRetriever:
    """Retrieve standards with FAISS, BM25, RRF fusion, and metadata boosts."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        model: Any | None = None,
        embed_model: str = EMBED_MODEL,
    ):
        self.data_dir = Path(data_dir)
        self.standards = self._load_standards(self.data_dir)
        self.faiss_index = self._load_faiss_index(self.data_dir / "faiss.index")
        self.bm25 = self._load_bm25(self.data_dir / "bm25.pkl")
        self.code_lookup = self._load_code_lookup(self.data_dir / "codes.json")
        self._model = model
        self._embed_model = embed_model

    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Dense retrieval requires the optional dependency "
                "`sentence-transformers`. Install it to enable embedding-based "
                "search, or construct `HybridRetriever` with a preloaded `model=`."
            ) from exc

        try:
            return SentenceTransformer(self._embed_model)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the sentence-transformers model "
                f"{self._embed_model!r}. Ensure the model name/path is correct and "
                "that the model is available locally or downloadable in this "
                "environment. You can also pass a preloaded `model=` to "
                "`HybridRetriever`."
            ) from exc

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_sentence_transformer()
        return self._model

    @staticmethod
    def _missing_artifacts_message(paths: list[Path]) -> str:
        missing = ", ".join(str(path) for path in paths if not path.exists())
        return (
            "Retrieval artifacts are missing: "
            f"{missing}. Build them with `python -m src.indexer --catalog "
            "data/standards_catalog.json --output-dir data`."
        )

    def _load_standards(self, data_dir: Path) -> list[dict[str, Any]]:
        standards_path = data_dir / "standards_indexed.json"
        if not standards_path.exists():
            raise FileNotFoundError(self._missing_artifacts_message([standards_path]))

        with standards_path.open(encoding="utf-8") as file:
            standards = json.load(file)
        if not isinstance(standards, list):
            raise ValueError(f"Standards index must be a JSON list: {standards_path}")
        return standards

    def _load_faiss_index(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(self._missing_artifacts_message([path]))

        import faiss

        index = faiss.read_index(str(path))
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = max(getattr(index.hnsw, "efSearch", 0), 64)
        return index

    def _load_bm25(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(self._missing_artifacts_message([path]))

        with path.open("rb") as file:
            payload = pickle.load(file)
        if isinstance(payload, dict) and "bm25" in payload:
            return payload["bm25"]
        return payload

    def _load_code_lookup(self, path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(self._missing_artifacts_message([path]))

        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return {normalize_is_code(code): int(index) for code, index in payload.items()}

    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[int]:
        """Return standard indices ranked by dense semantic similarity."""
        if not query.strip():
            return []

        search_k = min(max(top_k, 0), int(getattr(self.faiss_index, "ntotal", top_k)))
        if search_k <= 0:
            return []

        prefixed = "Represent this sentence for searching relevant passages: " + query
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        _distances, indices = self.faiss_index.search(embedding, search_k)
        return self._valid_unique_indices(indices[0].tolist())

    def _sparse_search(self, tokens: list[str], top_k: int = SPARSE_TOP_K) -> list[int]:
        """Return standard indices ranked by BM25 score."""
        if not tokens or top_k <= 0:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        if scores.size == 0:
            return []

        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []

        top_count = min(top_k, positive.size)
        ranked_local = np.argsort(scores[positive])[::-1][:top_count]
        return self._valid_unique_indices(positive[ranked_local].tolist())

    def _valid_unique_indices(self, indices: list[int]) -> list[int]:
        seen: set[int] = set()
        valid: list[int] = []
        for index in indices:
            idx = int(index)
            if idx < 0 or idx >= len(self.standards) or idx in seen:
                continue
            seen.add(idx)
            valid.append(idx)
        return valid

    def _rrf_fuse(
        self,
        dense_ranking: list[int],
        sparse_ranking: list[int],
        k: int = RRF_K,
    ) -> dict[int, float]:
        """Fuse dense and sparse rankings with reciprocal rank fusion."""
        scores: defaultdict[int, float] = defaultdict(float)
        for ranking in (dense_ranking, sparse_ranking):
            seen: set[int] = set()
            for rank, idx in enumerate(ranking):
                if idx in seen:
                    continue
                seen.add(idx)
                scores[idx] += 1.0 / (k + rank + 1)
        return dict(scores)

    def _explicit_code_indices(self, explicit_codes: Iterable[str]) -> set[int]:
        matches: set[int] = set()
        for code in explicit_codes:
            normalized = normalize_is_code(code)
            if normalized in self.code_lookup:
                matches.add(self.code_lookup[normalized])
                continue

            if ":" in normalized:
                continue
            for known_code, index in self.code_lookup.items():
                if known_code.startswith(normalized + ":") or known_code.startswith(normalized + "("):
                    matches.add(index)
        return matches

    def _apply_boosts(
        self,
        rrf_scores: dict[int, float],
        pq: ProcessedQuery,
    ) -> dict[int, float]:
        """Apply deterministic metadata boosts on top of RRF scores."""
        boosted = dict(rrf_scores)
        query_tokens = set(pq.tokens)
        query_low = pq.expanded.lower()

        for idx in self._explicit_code_indices(pq.explicit_codes):
            boosted[idx] = boosted.get(idx, 0.0) + BOOST_EXPLICIT_CODE

        query_numbers = set(re.findall(r"\b(\d{2,4})\b", query_low))
        for idx, score in list(boosted.items()):
            standard = self.standards[idx]

            title_tokens = set(tokenize(str(standard.get("title") or "")))
            title_overlap = len(title_tokens & query_tokens)
            title_boost = min(title_overlap * BOOST_TITLE_TOKEN, BOOST_TITLE_CAP)
            title_text = " ".join(tokenize(str(standard.get("title") or "")))
            if title_text and len(title_text) >= 5 and title_text in query_low:
                title_boost += BOOST_EXACT_TITLE

            keyword_tokens: set[str] = set()
            for keyword in standard.get("keywords") or []:
                keyword_tokens.update(tokenize(str(keyword)))
            keyword_overlap = len(keyword_tokens & query_tokens)
            keyword_boost = min(keyword_overlap * BOOST_KEYWORD, BOOST_KEYWORD_CAP)

            standard_text = " ".join(
                str(standard.get(field_name) or "")
                for field_name in ("is_code", "title", "scope", "chunk_text")
            ).lower()
            num_boost = sum(
                BOOST_NUMBER_TOKEN
                for number in query_numbers
                if len(number) >= 2 and number in standard_text
            )
            num_boost = min(num_boost, BOOST_NUMBER_CAP)

            boosted[idx] = score + title_boost + keyword_boost + num_boost

        return boosted

    def retrieve(self, pq: ProcessedQuery, top_k: int = RETRIEVE_TOP_K) -> list[RetrievalResult]:
        """Return top retrieval candidates for a processed query."""
        if top_k <= 0:
            return []

        dense = self._dense_search(pq.expanded, DENSE_TOP_K)
        sparse = self._sparse_search(pq.tokens, SPARSE_TOP_K)
        boosted = self._apply_boosts(self._rrf_fuse(dense, sparse), pq)
        ranked = sorted(boosted.items(), key=lambda item: (-item[1], item[0]))[:top_k]

        dense_pos = {idx: rank for rank, idx in enumerate(dense)}
        sparse_pos = {idx: rank for rank, idx in enumerate(sparse)}
        explicit_matches = self._explicit_code_indices(pq.explicit_codes)

        results: list[RetrievalResult] = []
        for idx, score in ranked:
            dense_rank = dense_pos.get(idx, -1)
            sparse_rank = sparse_pos.get(idx, -1)
            code_match = idx in explicit_matches
            results.append(
                RetrievalResult(
                    standard=self.standards[idx],
                    score=float(score),
                    sources={
                        "dense": dense_rank,
                        "sparse": sparse_rank,
                        "code": code_match,
                        "dense_rank": dense_rank,
                        "sparse_rank": sparse_rank,
                        "code_match": code_match,
                    },
                )
            )
        return results
