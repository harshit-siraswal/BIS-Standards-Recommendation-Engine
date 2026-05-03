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


def test_steel_hidden_set_style_queries():
    pairs = [
        ("TMT bars for RCC construction", "is1786:1985"),
        ("steel reinforcement bars for concrete", "is1786:1985"),
        ("saria for RCC", "is1786:1985"),
        ("lohe ka rod concrete reinforcement", "is1786:1985"),
        ("RCC ke liye steel bar", "is1786:1985"),
        ("mild steel bars for concrete reinforcement", "is432(part1):1982"),
        ("structural steel beams and channels", "is808:1989"),
        ("hollow steel sections for construction", "is4923:1997"),
        ("galvanized roofing steel sheet", "is277:2003"),
        ("steel tubes structural purposes", "is1161:1998"),
        ("steel plates strips flats structural engineering", "is1730:1989"),
        ("round square steel bars structural engineering", "is1732:1989"),
    ]
    pipeline = BISPipeline()
    for query, expected in pairs:
        out = pipeline.run_query(query)
        retrieved = [normalize_standard_code(code) for code in out["retrieved_standards"]]
        assert expected in retrieved[:3], f"Failed for: {query} -> {retrieved}"


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
