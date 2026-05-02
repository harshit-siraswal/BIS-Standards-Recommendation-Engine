import json
from pathlib import Path

import pytest

from src.parser import (
    IS_CODE_PATTERN,
    Standard,
    format_is_code,
    normalize_is_code,
    parse_pdf_to_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset.pdf"
PUBLIC_TEST_SET = ROOT / "public_test_set.json"


@pytest.fixture(scope="session")
def parsed_standards(tmp_path_factory):
    if not DATASET.exists():
        pytest.skip("dataset.pdf is not available")
    output_path = tmp_path_factory.mktemp("catalog") / "standards_catalog.json"
    return parse_pdf_to_catalog(str(DATASET), str(output_path))


def test_code_regex_accepts_real_formats():
    examples = {
        "IS 269: 1989": "IS 269: 1989",
        "IS 269 : 1989": "IS 269: 1989",
        "IS  269:1989": "IS 269: 1989",
        "IS 1489 (Part 1): 1991": "IS 1489 (Part 1): 1991",
        "IS 1489 (Part 1) : 1991": "IS 1489 (Part 1): 1991",
        "IS 2185 (Part 2): 1983": "IS 2185 (Part 2): 1983",
        "IS:269-1989": "IS 269: 1989",
        "IS  2185 (PART 2) : 1983": "IS 2185 (Part 2): 1983",
    }

    for raw, expected in examples.items():
        match = IS_CODE_PATTERN.search(raw)
        assert match, raw
        assert format_is_code(match.group(1), match.group(2), match.group(3)) == expected


def test_canonical_format():
    standard = Standard(
        is_code="IS 269: 1989",
        is_code_normalized="is269:1989",
        number="269",
        year="1989",
        part=None,
        title="Ordinary Portland Cement, 33 Grade",
        scope="Covers chemical and physical requirements.",
        body="Body",
        keywords=["cement", "ordinary portland cement", "33 grade"],
    )
    assert standard.is_code_normalized == normalize_is_code(standard.is_code)


def test_extracts_minimum_standards(parsed_standards):
    assert len(parsed_standards) >= 50, f"Only got {len(parsed_standards)} standards"


def test_known_public_standards_present(parsed_standards):
    codes = {standard.is_code_normalized for standard in parsed_standards}
    if PUBLIC_TEST_SET.exists():
        expected = {
            normalize_is_code(code)
            for row in json.loads(PUBLIC_TEST_SET.read_text(encoding="utf-8"))
            for code in row["expected_standards"]
        }
    else:
        expected = {
            "is269:1989",
            "is383:1970",
            "is458:2003",
            "is2185(part2):1983",
            "is459:1992",
            "is455:1989",
            "is1489(part2):1991",
            "is3466:1988",
            "is6909:1990",
            "is8042:1989",
        }

    missing = sorted(expected - codes)
    assert not missing, f"Missing critical standards: {missing}"


def test_catalog_schema_and_quality(parsed_standards):
    seen = set()
    required_keys = {
        "is_code",
        "is_code_normalized",
        "number",
        "year",
        "part",
        "title",
        "scope",
        "body",
        "keywords",
    }

    for standard in parsed_standards:
        data = standard.to_dict()
        assert set(data) == required_keys
        assert standard.is_code_normalized == normalize_is_code(standard.is_code)
        assert standard.is_code_normalized not in seen
        seen.add(standard.is_code_normalized)
        assert standard.title and standard.title != "Untitled standard"
        assert standard.scope
        assert standard.body
        assert len(standard.keywords) >= 3
