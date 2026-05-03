import json

from src.query_processor import QueryProcessor, SynonymExpander, detect_is_codes


def test_normalization():
    qp = QueryProcessor()
    pq = qp.process("  33 GRADE\u00a0OPC  ")

    assert pq.normalized == "33 grade opc"


def test_expansion_opc():
    qp = QueryProcessor()
    pq = qp.process("OPC requirements")

    assert "ordinary portland cement" in pq.expanded
    assert "specification" in pq.expanded
    assert "ordinary" in pq.tokens
    assert "portland" in pq.tokens


def test_grade_expansion_does_not_cross_pollute_other_grades():
    qp = QueryProcessor()
    pq = qp.process("43 grade OPC")

    assert "ordinary portland cement" in pq.expanded
    assert "43 grade cement" in pq.expanded


def test_expansion_ppc_calcined():
    qp = QueryProcessor()
    pq = qp.process("calcined clay based pozzolana cement")

    assert "ppc" in pq.expanded
    assert "portland pozzolana cement" in pq.expanded
    assert "is 1489 part 2" in pq.expanded


def test_marine_works_expands_to_supersulphated():
    qp = QueryProcessor()
    pq = qp.process("cement for marine works and aggressive water")

    assert "supersulphated cement" in pq.expanded
    assert "ssc" in pq.expanded


def test_steel_reinforcement_aliases_expand_to_catalog_terms():
    qp = QueryProcessor()
    pq = qp.process("saria for RCC construction")

    assert pq.out_of_scope is False
    assert "steel reinforcement bars" in pq.expanded
    assert "high strength deformed steel bars" in pq.expanded
    assert "concrete reinforcement" in pq.expanded


def test_structural_steel_aliases_expand_to_sections_and_tubes():
    qp = QueryProcessor()
    pq = qp.process("structural steel beams and hollow steel sections")

    assert pq.out_of_scope is False
    assert "hot rolled steel sections" in pq.expanded
    assert "hollow steel sections structural use" in pq.expanded
    assert "steel for general structural purposes" in pq.expanded


def test_multilingual_product_phrase_expands_to_catalog_terms():
    qp = QueryProcessor()
    pq = qp.process("\u0938\u092b\u0947\u0926 \u092a\u094b\u0930\u094d\u091f\u0932\u0948\u0902\u0921 \u0938\u0940\u092e\u0947\u0902\u091f")

    assert "white portland cement" in pq.expanded
    assert "architectural decorative chemical physical requirements" in pq.expanded


def test_explicit_code_detection():
    qp = QueryProcessor()
    pq = qp.process("Does IS 269: 1989 apply to my product?")

    assert "is269:1989" in pq.explicit_codes
    assert "iscode269p0y1989" in pq.tokens


def test_explicit_part_code_detection():
    assert detect_is_codes("Check IS 1489 (Part 2): 1991 and IS No. 458-2003") == [
        "is1489(part2):1991",
        "is458:2003",
    ]


def test_no_false_codes():
    qp = QueryProcessor()
    pq = qp.process("a 33 grade cement and IS 269 without a year")

    assert pq.explicit_codes == []


def test_asbestos_terms_emit_compliance_warning():
    qp = QueryProcessor()
    pq = qp.process("corrugated asbestos cement sheet")

    assert any("asbestos" in warning.lower() for warning in pq.compliance_warnings)


def test_pencil_query_is_marked_out_of_scope():
    qp = QueryProcessor()
    queries = [
        "we are making graphite lead pencils",
        "हम ग्रेफाइट लेड पेंसिल बनाते हैं",
        "আমরা গ্রাফাইট লেড পেন্সিল তৈরি করি",
        "நாங்கள் கிராஃபைட் லீட் பென்சில்கள் தயாரிக்கிறோம்",
        "మేము గ్రాఫైట్ లీడ్ పెన్సిల్స్ తయారు చేస్తున్నాము",
        "અમે ગ્રાફાઇટ લીડ પેન્સિલ બનાવીએ છીએ",
        "ہم گریفائٹ لیڈ پنسل بناتے ہیں",
    ]

    for query in queries:
        pq = qp.process(query)
        assert pq.out_of_scope is True, query
        assert any("pencil" in warning.lower() for warning in pq.compliance_warnings)


def test_edible_oil_query_is_marked_out_of_scope():
    qp = QueryProcessor()
    queries = [
        "we make edible oil",
        "\u0939\u092e \u0916\u093e\u0926\u094d\u092f \u0924\u0947\u0932 \u092c\u0928\u093e\u0924\u0947 \u0939\u0948\u0902",
        "hum khane ka tel banate hain",
        "\u0ba8\u0bbe\u0b99\u0bcd\u0b95\u0bb3\u0bcd \u0b89\u0ba3\u0bb5\u0bc1 \u0b8e\u0ba3\u0bcd\u0ba3\u0bc6\u0baf\u0bcd \u0ba4\u0baf\u0bbe\u0bb0\u0bbf\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0bcb\u0bae\u0bcd",
    ]

    for query in queries:
        pq = qp.process(query)

        assert pq.out_of_scope is True, query
        assert any("edible oil" in warning.lower() for warning in pq.compliance_warnings)


def test_expander_supports_reverse_mapping(tmp_path):
    synonyms_path = tmp_path / "synonyms.json"
    synonyms_path.write_text(
        json.dumps({"cement": {"opc": ["ordinary portland cement", "33 grade cement"]}}),
        encoding="utf-8",
    )
    expander = SynonymExpander(synonyms_path)

    assert "opc" in expander.expand("ordinary portland cement")
