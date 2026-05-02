import json
from pathlib import Path

import numpy as np
import pytest

from src.query_processor import ProcessedQuery, QueryProcessor
from src.retriever import BOOST_EXPLICIT_CODE, BOOST_TITLE_TOKEN, RRF_K, HybridRetriever, normalize_standard_code


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REQUIRED_ARTIFACTS = [
    DATA_DIR / "standards_indexed.json",
    DATA_DIR / "faiss.index",
    DATA_DIR / "bm25.pkl",
    DATA_DIR / "codes.json",
]


class FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeFaissIndex:
    ntotal = 3

    def search(self, embedding, top_k):
        return np.array([[0.9, 0.8, 0.7]], dtype=np.float32), np.array([[1, 0, 2]])


class FakeBM25:
    def get_scores(self, tokens):
        return np.array([2.0, 0.2, 1.0], dtype=np.float32)


def make_fake_retriever():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.standards = [
        {
            "is_code": "IS 269: 1989",
            "is_code_normalized": "is269:1989",
            "title": "Specification for 33 grade ordinary Portland cement",
            "scope": "Covers chemical and physical requirements for cement.",
            "keywords": ["opc", "cement", "33 grade"],
            "chunk_text": "33 grade ordinary portland cement",
        },
        {
            "is_code": "IS 458: 2003",
            "is_code_normalized": "is458:2003",
            "title": "Precast concrete pipes with and without reinforcement",
            "scope": "Covers concrete pipes for water mains.",
            "keywords": ["concrete pipes", "water mains"],
            "chunk_text": "precast concrete pipes water mains",
        },
        {
            "is_code": "IS 383: 1970",
            "is_code_normalized": "is383:1970",
            "title": "Coarse and fine aggregates from natural sources",
            "scope": "For aggregates used in structural concrete.",
            "keywords": ["aggregate", "structural concrete"],
            "chunk_text": "coarse fine aggregates structural concrete",
        },
    ]
    retriever.faiss_index = FakeFaissIndex()
    retriever.bm25 = FakeBM25()
    retriever.code_lookup = {
        "is269:1989": 0,
        "is458:2003": 1,
        "is383:1970": 2,
    }
    retriever._model = FakeModel()
    retriever.use_dense = False
    retriever.boost_explicit_code = BOOST_EXPLICIT_CODE
    retriever.boost_title_token = BOOST_TITLE_TOKEN
    retriever.rrf_k = RRF_K
    return retriever


def artifacts_available():
    return all(path.exists() for path in REQUIRED_ARTIFACTS)


def test_standard_code_normalization_matches_evaluator():
    assert normalize_standard_code("IS 1489 (Part 2): 1991") == "is1489(part2):1991"


def test_rrf_fuse_combines_dense_and_sparse_rankings():
    retriever = make_fake_retriever()

    scores = retriever._rrf_fuse([1, 0], [0, 2], k=60)

    assert scores[0] == pytest.approx((1 / 62) + (1 / 61))
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62)


def test_explicit_code_boost_lifts_code_match_to_top():
    retriever = make_fake_retriever()
    pq = ProcessedQuery(
        raw="Tell me about IS 458: 2003",
        normalized="tell me about is 458: 2003",
        expanded="tell me about is 458: 2003",
        explicit_codes=["is458:2003"],
        tokens=["tell", "about", "iscode458p0y2003"],
    )

    results = retriever.retrieve(pq, top_k=3)

    assert results[0].standard["is_code_normalized"] == "is458:2003"
    assert results[0].sources["code"] is True


def test_query_processor_detects_and_expands_codes_and_domain_terms():
    processor = QueryProcessor()

    pq = processor.process("Need IS 269: 1989 for OPC")

    assert "is269:1989" in pq.explicit_codes
    assert "ordinary portland cement" in pq.expanded
    assert "opc" in pq.tokens


@pytest.fixture(scope="session")
def real_retriever():
    if not artifacts_available():
        pytest.skip("Plan 02 artifacts are not built; run `python -m src.indexer` first")
    return HybridRetriever(DATA_DIR)


def test_retriever_loads(real_retriever):
    assert len(real_retriever.standards) > 50


def test_top_result_for_opc(real_retriever):
    processor = QueryProcessor()

    pq = processor.process("33 Grade Ordinary Portland Cement chemical requirements")
    results = real_retriever.retrieve(pq, top_k=5)
    top_codes = [result.standard["is_code_normalized"] for result in results]

    assert "is269:1989" in top_codes[:3], f"Expected IS 269 in top 3, got {top_codes}"


def test_explicit_code_lifts_to_top(real_retriever):
    processor = QueryProcessor()

    pq = processor.process("Tell me about IS 458: 2003")
    results = real_retriever.retrieve(pq, top_k=5)

    assert results[0].standard["is_code_normalized"] == "is458:2003"


def test_all_public_queries_recall(real_retriever):
    processor = QueryProcessor()
    test_set_path = ROOT / "public_test_set.json"
    if not test_set_path.exists():
        pytest.skip("public_test_set.json is not available")
    test_set = json.loads(test_set_path.read_text(encoding="utf-8"))

    misses = []
    for item in test_set:
        pq = processor.process(item["query"])
        results = real_retriever.retrieve(pq, top_k=5)
        top_codes = {result.standard["is_code_normalized"] for result in results}
        expected = {code.replace(" ", "").lower() for code in item["expected_standards"]}
        if not top_codes & expected:
            misses.append((item["id"], item["query"], sorted(top_codes)))

    assert not misses, f"Misses on public set BEFORE rerank: {misses}"
