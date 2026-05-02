import pytest

from src.reranker import MAX_DOCUMENT_CHARS, RERANK_INPUT_K, Reranker, RetrievalResult


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return self.scores[: len(pairs)]


class BrokenCrossEncoder:
    def predict(self, *args, **kwargs):
        raise RuntimeError("model unavailable")


def make_result(code, title, scope, score):
    return RetrievalResult(
        standard={
            "is_code": code,
            "title": title,
            "scope": scope,
            "body": "fallback body",
            "keywords": ["cement", "physical requirements"],
        },
        score=score,
    )


def test_rerank_scores_pairs_and_combines_with_retrieval_score():
    candidates = [
        make_result("IS 269: 1989", "33 Grade Ordinary Portland Cement", "Chemical requirements", 0.10),
        make_result("IS 455: 1989", "Portland Slag Cement", "Slag cement requirements", 0.05),
        make_result("IS 8042: 1989", "White Portland Cement", "Decorative cement", 0.20),
    ]
    model = FakeCrossEncoder([0.20, 0.10, 0.80])
    reranker = Reranker(model=model)

    results = reranker.rerank("white portland cement", candidates, top_k=2)

    assert [result.standard["is_code"] for result in results] == ["IS 8042: 1989", "IS 269: 1989"]
    assert results[0].score == pytest.approx(0.7 * 0.80 + 0.3 * 0.20)
    assert results[1].score == pytest.approx(0.7 * 0.20 + 0.3 * 0.10)
    assert candidates[2].score == pytest.approx(0.20)
    assert candidates[0].score == pytest.approx(0.10)
    assert model.calls[0]["batch_size"] == 32
    assert model.calls[0]["show_progress_bar"] is False
    assert model.calls[0]["convert_to_numpy"] is True
    assert model.calls[0]["pairs"][0][0] == "white portland cement"
    assert "IS 269: 1989 33 Grade Ordinary Portland Cement" in model.calls[0]["pairs"][0][1]


def test_rerank_limits_cross_encoder_input_window():
    candidates = [
        make_result(f"IS {index}: 2000", f"Standard {index}", "scope", float(index))
        for index in range(RERANK_INPUT_K + 5)
    ]
    scores = [100.0 - index for index in range(RERANK_INPUT_K)]
    model = FakeCrossEncoder(scores)
    reranker = Reranker(model=model)

    results = reranker.rerank("query", candidates, top_k=5)

    assert len(model.calls[0]["pairs"]) == RERANK_INPUT_K
    assert len(results) == 5
    input_codes = {candidate.standard["is_code"] for candidate in candidates[:RERANK_INPUT_K]}
    assert all(result.standard["is_code"] in input_codes for result in results)


def test_rerank_safe_falls_back_to_original_scores():
    candidates = [
        make_result("IS 1: 2000", "Low", "scope", 0.1),
        make_result("IS 2: 2000", "High", "scope", 0.9),
        make_result("IS 3: 2000", "Middle", "scope", 0.4),
    ]
    reranker = Reranker(model=BrokenCrossEncoder())

    results = reranker.rerank_safe("query", candidates, top_k=2)

    assert [result.standard["is_code"] for result in results] == ["IS 2: 2000", "IS 3: 2000"]


def test_document_text_is_truncated_and_uses_body_fallback():
    candidate = RetrievalResult(
        standard={
            "is_code": "IS 999: 2000",
            "title": "Long Standard",
            "scope": "",
            "body": "x" * 1000,
            "keywords": [],
        },
        score=0.5,
    )

    text = Reranker._candidate_document_text(candidate)

    assert text.startswith("IS 999: 2000 Long Standard")
    assert len(text) == MAX_DOCUMENT_CHARS


def test_empty_and_zero_top_k_return_empty_without_model_call():
    model = FakeCrossEncoder([1.0])
    reranker = Reranker(model=model)

    assert reranker.rerank("query", [], top_k=5) == []
    assert reranker.rerank("query", [make_result("IS 1: 2000", "Title", "scope", 0.1)], top_k=0) == []
    assert model.calls == []
