from src.pipeline import BISPipeline
from src.retriever import normalize_standard_code


def test_paraphrase_robustness():
    pairs = [
        ("OPC 33 grade specs", "is269:1989"),
        ("Specification for ordinary portland cement 33", "is269:1989"),
        ("BIS standard for 33-grade Portland cement", "is269:1989"),
        ("manufacture of 33 grade cement", "is269:1989"),
    ]
    pipeline = BISPipeline()
    for query, expected in pairs:
        out = pipeline.run_query(query)
        retrieved = [normalize_standard_code(code) for code in out["retrieved_standards"]]
        assert expected in retrieved[:3], f"Failed for: {query} -> {retrieved}"


def test_noisy_query_robustness():
    queries = [
        "33 grad ordnary portland cemnt",
        "OPC",
        "WHAT IS THE STANDARD FOR 33 GRADE OPC",
    ]
    pipeline = BISPipeline()
    for query in queries:
        out = pipeline.run_query(query)
        retrieved = out["retrieved_standards"]
        assert len(retrieved) == 5, f"Query {query!r} returned {len(retrieved)} results: {retrieved}"


def test_long_and_unicode_queries_do_not_crash():
    pipeline = BISPipeline()
    out = pipeline.run_query(("standards " * 80) + "33 grade OPC")
    unicode_out = pipeline.run_query("Which standard covers cafe\u0301 style white Portland cement?")

    assert len(out["retrieved_standards"]) == 5, (
        f"Long query returned {len(out['retrieved_standards'])} results: {out['retrieved_standards']}"
    )
    assert len(unicode_out["retrieved_standards"]) == 5, (
        "Unicode query returned "
        f"{len(unicode_out['retrieved_standards'])} results: {unicode_out['retrieved_standards']}"
    )
