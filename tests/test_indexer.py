import json
import pickle

from src import indexer


def test_tokenize_preserves_standard_is_code_as_atomic_token():
    tokens = indexer.tokenize("IS 269: 1989 33 grade ordinary Portland cement")

    assert "iscode269p0y1989" in tokens
    assert "269" not in tokens
    assert "1989" not in tokens


def test_tokenize_preserves_part_code_as_atomic_token():
    tokens = indexer.tokenize("IS 1489 (Part 2): 1991 Portland pozzolana cement")

    assert "iscode1489p2y1991" in tokens


def test_build_chunk_text_repeats_title_and_limits_long_fields():
    standard = {
        "is_code": "IS 269: 1989",
        "title": "Specification for 33 grade ordinary Portland cement",
        "keywords": ["opc", "cement"],
        "scope": "s" * 1100,
        "body": "b" * 1600,
    }

    chunk = indexer.build_chunk_text(standard)

    assert chunk.count("Specification for 33 grade ordinary Portland cement") == 2
    assert "IS 269: 1989" in chunk
    assert "opc cement" in chunk
    assert "s" * 1000 in chunk
    assert "s" * 1001 not in chunk
    assert "b" * 1500 in chunk
    assert "b" * 1501 not in chunk


def test_build_code_lookup_uses_existing_or_fallback_normalization():
    standards = [
        {"is_code": "IS 269: 1989", "is_code_normalized": " IS 269 : 1989 "},
        {"is_code": "IS 1489 (Part 2): 1991"},
    ]

    codes = indexer.build_code_lookup(standards)

    assert codes == {
        "is269:1989": 0,
        "is1489(part2):1991": 1,
    }


def test_build_all_indexes_writes_plan_02_artifacts(tmp_path, monkeypatch):
    catalog_path = tmp_path / "standards_catalog.json"
    output_dir = tmp_path / "data"
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "is_code": "IS 269: 1989",
                    "is_code_normalized": "is269:1989",
                    "title": "Specification for 33 grade ordinary Portland cement",
                    "keywords": ["opc", "cement"],
                    "scope": "Covers ordinary Portland cement.",
                    "body": "Physical and chemical requirements for cement.",
                },
                {
                    "is_code": "IS 1489 (Part 2): 1991",
                    "is_code_normalized": "is1489(part2):1991",
                    "title": "Portland-pozzolana cement specification",
                    "keywords": ["ppc", "pozzolana"],
                    "scope": "Covers Portland-pozzolana cement.",
                    "body": "Fly ash based cement requirements.",
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_build_dense_index(standards):
        assert all(standard["chunk_text"] for standard in standards)
        return {"vectors": len(standards)}

    def fake_write_dense_index(dense_index, path):
        path.write_bytes(f"vectors={dense_index['vectors']}".encode("ascii"))

    def fake_build_sparse_index(standards):
        tokenized = [indexer.tokenize(standard["chunk_text"]) for standard in standards]
        return {"documents": len(tokenized)}, tokenized

    monkeypatch.setattr(indexer, "build_dense_index", fake_build_dense_index)
    monkeypatch.setattr(indexer, "write_dense_index", fake_write_dense_index)
    monkeypatch.setattr(indexer, "build_sparse_index", fake_build_sparse_index)

    result = indexer.build_all_indexes(catalog_path=catalog_path, output_dir=output_dir)

    assert result["count"] == 2
    assert (output_dir / "faiss.index").read_bytes() == b"vectors=2"
    assert (output_dir / "bm25.pkl").exists()
    assert (output_dir / "codes.json").exists()
    assert (output_dir / "standards_indexed.json").exists()

    indexed = json.loads((output_dir / "standards_indexed.json").read_text(encoding="utf-8"))
    assert indexed[0]["chunk_text"]

    codes = json.loads((output_dir / "codes.json").read_text(encoding="utf-8"))
    assert codes["is269:1989"] == 0
    assert codes["is1489(part2):1991"] == 1

    with (output_dir / "bm25.pkl").open("rb") as file:
        bm25_payload = pickle.load(file)
    assert bm25_payload["bm25"] == {"documents": 2}
    assert "iscode269p0y1989" in bm25_payload["tokenized"][0]
