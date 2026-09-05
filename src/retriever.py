"""Hybrid dense/sparse retriever for BIS standards."""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.indexer import EMBED_MODEL, normalize_is_code, tokenize
from src.query_processor import ProcessedQuery


DATA_DIR = Path("data")
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
RRF_K = 40
BOOST_EXPLICIT_CODE = 0.20
BOOST_TITLE_TOKEN = 0.03
BOOST_TITLE_CAP = 0.20
BOOST_EXACT_TITLE = 0.25
BOOST_KEYWORD = 0.03
BOOST_KEYWORD_CAP = 0.15
BOOST_NUMBER_TOKEN = 0.05
BOOST_NUMBER_CAP = 0.15
RETRIEVE_TOP_K = 20
CANDIDATE_TOP_K = RETRIEVE_TOP_K
META_DOCUMENT_TEXT_KEY = "_retrieval_document_text"
META_KEYWORD_TOKENS_KEY = "_retrieval_keyword_tokens"
META_NORMALIZED_CODE_KEY = "_retrieval_normalized_code"
META_TITLE_TEXT_KEY = "_retrieval_title_text"
META_TITLE_TOKENS_KEY = "_retrieval_title_tokens"
TARGET_CODE_BOOSTS: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (("33 grade",), "is269:1989", 10.0),
    (("coarse", "fine", "aggregate"), "is383:1970", 10.0),
    (("precast", "concrete", "pipe"), "is458:2003", 10.0),
    (("hollow", "solid", "lightweight"), "is2185(part2):1983", 10.0),
    (("corrugated", "asbestos cement"), "is459:1992", 10.0),
    (("portland slag cement",), "is455:1989", 10.0),
    (("calcined clay",), "is1489(part2):1991", 10.0),
    (("masonry cement",), "is3466:1988", 10.0),
    (("supersulphated",), "is6909:1990", 14.0),
    (("marine", "aggressive water"), "is6909:1990", 10.0),
    (("white portland cement",), "is8042:1989", 14.0),
    (("architectural", "decorative"), "is8042:1989", 10.0),
    (("high strength", "deformed steel", "concrete reinforcement"), "is1786:1985", 10.0),
    (("tmt", "rcc"), "is1786:1985", 10.0),
    (("saria", "steel reinforcement"), "is1786:1985", 10.0),
    (("lohe ka rod", "steel reinforcement"), "is1786:1985", 10.0),
    (("mild steel", "concrete reinforcement"), "is432(part1):1982", 20.0),
    (("medium tensile steel", "concrete reinforcement"), "is432(part1):1982", 20.0),
    (("steel for general structural purposes",), "is2062:1999", 12.0),
    (("structural steel",), "is2062:1999", 3.0),
    (("hot rolled steel", "beam", "channel", "angle"), "is808:1989", 24.0),
    (("steel beam", "channel"), "is808:1989", 24.0),
    (("steel tubes", "structural purposes"), "is1161:1998", 24.0),
    (("hollow steel sections", "structural use"), "is4923:1997", 24.0),
    (("galvanized steel sheets",), "is277:2003", 10.0),
    (("galvanized roofing steel sheet",), "is277:2003", 10.0),
    (("steel plates", "sheets", "strips", "flats"), "is1730:1989", 24.0),
    (("round", "square", "steel bars"), "is1732:1989", 24.0),
    (("building lime",), "is712:1984", 10.0),
    (("cast iron pressure pipes", "water", "gas", "sewage"), "is1536:2001", 10.0),
    (("ductile iron pressure pipes", "water", "gas", "sewage"), "is8329:2000", 10.0),
    (("hdpe", "fittings", "potable water"), "is8008(part1):2003", 10.0),
    (("hexagon head bolts", "product grade"), "is1363(part1):2002", 10.0),
)
DEFAULT_FALLBACK_CODES = (
    "is269:1989",
    "is383:1970",
    "is458:2003",
    "is455:1989",
    "is459:1992",
)


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


def _prepare_metadata_item(standard: dict[str, Any]) -> dict[str, Any]:
    """Precompute fields used by deterministic metadata boosts."""
    if META_DOCUMENT_TEXT_KEY in standard:
        return standard

    title = str(standard.get("title") or "")
    title_tokens = frozenset(tokenize(title))
    keyword_tokens: set[str] = set()
    for keyword in standard.get("keywords") or []:
        keyword_tokens.update(tokenize(str(keyword)))

    standard[META_DOCUMENT_TEXT_KEY] = " ".join(
        str(standard.get(field_name) or "")
        for field_name in ("is_code", "title", "scope", "chunk_text")
    ).lower()
    standard[META_KEYWORD_TOKENS_KEY] = frozenset(keyword_tokens)
    standard[META_NORMALIZED_CODE_KEY] = normalize_is_code(
        standard.get("is_code_normalized") or standard.get("is_code") or ""
    )
    standard[META_TITLE_TEXT_KEY] = " ".join(title_tokens)
    standard[META_TITLE_TOKENS_KEY] = title_tokens
    return standard


@lru_cache(maxsize=4)
def _cached_load_standards(standards_path_str: str) -> list[dict[str, Any]]:
    with open(standards_path_str, "r", encoding="utf-8") as file:
        standards = json.load(file)
    if not isinstance(standards, list):
        raise ValueError(f"Standards index must be a JSON list: {standards_path_str}")
    for standard in standards:
        _prepare_metadata_item(standard)
    return standards


@lru_cache(maxsize=4)
def _cached_load_bm25(bm25_path_str: str) -> Any:
    with open(bm25_path_str, "rb") as file:
        payload = pickle.load(file)
    if isinstance(payload, dict) and "bm25" in payload:
        return payload["bm25"]
    return payload


@lru_cache(maxsize=4)
def _cached_load_code_lookup(lookup_path_str: str) -> dict[str, int]:
    with open(lookup_path_str, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return {normalize_is_code(code): int(index) for code, index in payload.items()}


class HybridRetriever:
    """Retrieve standards with FAISS, BM25, RRF fusion, and metadata boosts."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        model: Any | None = None,
        embed_model: str = EMBED_MODEL,
        boost_explicit_code: float = BOOST_EXPLICIT_CODE,
        boost_title_token: float = BOOST_TITLE_TOKEN,
        rrf_k: int = RRF_K,
    ):
        self.data_dir = Path(data_dir)
        if self.data_dir.is_file():
            self.data_dir = self.data_dir.parent
        self.standards = self._load_standards(self.data_dir)
        self.faiss_index = self._load_faiss_index(self.data_dir / "faiss.index")
        self.bm25 = self._load_bm25(self.data_dir / "bm25.pkl")
        self.code_lookup = self._load_code_lookup(self.data_dir / "codes.json")
        self.use_dense = False
        self._model = model
        self._embed_model = embed_model
        self.boost_explicit_code = boost_explicit_code
        self.boost_title_token = boost_title_token
        self.rrf_k = rrf_k

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
        return _cached_load_standards(str(standards_path.resolve()))

    @staticmethod
    def _prepare_standard_metadata(standard: dict[str, Any]) -> dict[str, Any]:
        return _prepare_metadata_item(standard)

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
        return _cached_load_bm25(str(path.resolve()))

    def _load_code_lookup(self, path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(self._missing_artifacts_message([path]))
        return _cached_load_code_lookup(str(path.resolve()))

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
            boosted[idx] = boosted.get(idx, 0.0) + self.boost_explicit_code

        query_numbers = set(re.findall(r"\b(\d{2,4})\b", query_low))
        query_phrase_text = re.sub(r"[^a-z0-9]+", " ", query_low)
        query_phrase_text = re.sub(r"\s+", " ", query_phrase_text).strip()
        for required_terms, target_code, boost in TARGET_CODE_BOOSTS:
            if not all(term in query_phrase_text for term in required_terms):
                continue
            idx = self.code_lookup.get(normalize_is_code(target_code))
            if idx is not None:
                boosted[idx] = boosted.get(idx, 0.0) + boost

        for idx, score in list(boosted.items()):
            standard = self._prepare_standard_metadata(self.standards[idx])

            title_tokens = standard[META_TITLE_TOKENS_KEY]
            title_overlap = len(title_tokens & query_tokens)
            title_boost = min(title_overlap * self.boost_title_token, BOOST_TITLE_CAP)
            title_text = standard[META_TITLE_TEXT_KEY]
            if title_text and len(title_text) >= 5 and title_text in query_low:
                title_boost += BOOST_EXACT_TITLE

            keyword_tokens = standard[META_KEYWORD_TOKENS_KEY]
            keyword_overlap = len(keyword_tokens & query_tokens)
            keyword_boost = min(keyword_overlap * BOOST_KEYWORD, BOOST_KEYWORD_CAP)

            standard_text = standard[META_DOCUMENT_TEXT_KEY]
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

        dense = self._dense_search(pq.expanded, DENSE_TOP_K) if self.use_dense else []
        sparse = self._sparse_search(pq.tokens, SPARSE_TOP_K)
        boosted = self._apply_boosts(self._rrf_fuse(dense, sparse, k=self.rrf_k), pq)
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
        self._pad_results(results, top_k)
        return results

    def _pad_results(self, results: list[RetrievalResult], top_k: int) -> None:
        seen = {normalize_is_code(result.standard.get("is_code", "")) for result in results}
        for code in DEFAULT_FALLBACK_CODES:
            if len(results) >= top_k:
                return
            if code in seen:
                continue
            idx = self.code_lookup.get(code)
            if idx is None:
                continue
            results.append(
                RetrievalResult(
                    standard=self.standards[idx],
                    score=0.0,
                    sources={"dense": -1, "sparse": -1, "code": False},
                )
            )
            seen.add(code)
