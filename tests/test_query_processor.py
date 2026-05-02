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


def test_expander_supports_reverse_mapping(tmp_path):
    synonyms_path = tmp_path / "synonyms.json"
    synonyms_path.write_text(
        json.dumps({"cement": {"opc": ["ordinary portland cement", "33 grade cement"]}}),
        encoding="utf-8",
    )
    expander = SynonymExpander(synonyms_path)

    assert "opc" in expander.expand("ordinary portland cement")
