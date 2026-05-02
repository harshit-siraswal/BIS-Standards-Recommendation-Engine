"""Build dense, sparse, and exact-code indexes for BIS standards."""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

DATA_DIR = Path("data")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
IS_CODE_PATTERN = re.compile(
    r"\bIS\s*(?:No\.?\s*)?(\d{2,5})(?:\s*\(?\s*Part\s*[-:]?\s*(\d+)\s*\)?)?\s*[:\-]\s*(\d{4})\b",
    flags=re.IGNORECASE,
)
STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "be",
    }
)


def _optional_import(module_name: str, package_name: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        install_name = package_name or module_name
        raise RuntimeError(
            f"Missing dependency '{install_name}'. Install project dependencies with "
            "`pip install -r requirements.txt` before building indexes."
        ) from exc


def normalize_is_code(code: str) -> str:
    """Normalize an IS code the same way the evaluator does."""
    return re.sub(r"\s+", "", str(code or "")).lower()


def _keywords_text(keywords: Any) -> str:
    if not keywords:
        return ""
    if isinstance(keywords, str):
        return keywords
    return " ".join(str(keyword) for keyword in keywords if keyword)


def build_chunk_text(standard: dict[str, Any]) -> str:
    """Build the text representation used by dense and sparse retrieval."""
    title = str(standard.get("title") or "")
    scope = str(standard.get("scope") or "")
    body = str(standard.get("body") or standard.get("full_text") or "")

    parts = [
        str(standard.get("is_code") or ""),
        title,
        title,
        _keywords_text(standard.get("keywords")),
        scope[:1000],
        body[:1500],
    ]
    return " ".join(part.strip() for part in parts if str(part).strip()).strip()


def _code_token(match: re.Match[str]) -> str:
    number = str(int(match.group(1)))
    part = str(int(match.group(2))) if match.group(2) else "0"
    year = match.group(3)
    return f" iscode{number}p{part}y{year} "


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric BM25 tokenization with atomic IS-code tokens."""
    with_code_tokens = IS_CODE_PATTERN.sub(_code_token, text or "")
    tokens = TOKEN_PATTERN.findall(with_code_tokens.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def build_dense_index(standards: list[dict[str, Any]]) -> Any:
    """Build a FAISS HNSW index over normalized BGE embeddings."""
    faiss = _optional_import("faiss", "faiss-cpu")
    sentence_transformers = _optional_import("sentence_transformers")
    model = sentence_transformers.SentenceTransformer(EMBED_MODEL)

    embeddings = model.encode(
        [standard["chunk_text"] for standard in standards],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Expected {EMBED_DIM}-dim embeddings from {EMBED_MODEL}, got {embeddings.shape}."
        )

    index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)
    return index


def write_dense_index(index: Any, path: Path) -> None:
    faiss = _optional_import("faiss", "faiss-cpu")
    faiss.write_index(index, str(path))


def build_sparse_index(standards: list[dict[str, Any]]) -> tuple[Any, list[list[str]]]:
    rank_bm25 = _optional_import("rank_bm25", "rank-bm25")
    tokenized = [tokenize(standard["chunk_text"]) for standard in standards]
    return rank_bm25.BM25Okapi(tokenized, k1=1.5, b=0.75), tokenized


def build_code_lookup(standards: list[dict[str, Any]]) -> dict[str, int]:
    """Map normalized IS codes to their position in the indexed catalog."""
    codes: dict[str, int] = {}
    for index, standard in enumerate(standards):
        raw_code = standard.get("is_code_normalized") or standard.get("is_code", "")
        normalized = normalize_is_code(raw_code)
        if normalized:
            codes[normalized] = index
    return codes


def _load_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    with catalog_path.open(encoding="utf-8") as file:
        standards = json.load(file)
    if not isinstance(standards, list):
        raise ValueError(f"Catalog must contain a JSON list: {catalog_path}")
    return standards


def build_all_indexes(
    catalog_path: str | Path = DATA_DIR / "standards_catalog.json",
    output_dir: str | Path = DATA_DIR,
) -> dict[str, Path | int]:
    """Build all retrieval artifacts from the standards catalog."""
    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standards = _load_catalog(catalog_path)
    for standard in standards:
        standard["chunk_text"] = build_chunk_text(standard)

    standards_path = output_dir / "standards_indexed.json"
    with standards_path.open("w", encoding="utf-8") as file:
        json.dump(standards, file, ensure_ascii=False, indent=2)

    dense_index = build_dense_index(standards)
    faiss_path = output_dir / "faiss.index"
    write_dense_index(dense_index, faiss_path)

    bm25, tokenized = build_sparse_index(standards)
    bm25_path = output_dir / "bm25.pkl"
    with bm25_path.open("wb") as file:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, file)

    codes_path = output_dir / "codes.json"
    with codes_path.open("w", encoding="utf-8") as file:
        json.dump(build_code_lookup(standards), file, indent=2)

    return {
        "count": len(standards),
        "standards": standards_path,
        "faiss": faiss_path,
        "bm25": bm25_path,
        "codes": codes_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BIS hybrid retrieval indexes.")
    parser.add_argument(
        "--catalog",
        default=str(DATA_DIR / "standards_catalog.json"),
        help="Path to standards_catalog.json produced by the parser.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Directory where retrieval index artifacts are written.",
    )
    args = parser.parse_args(argv)

    result = build_all_indexes(catalog_path=args.catalog, output_dir=args.output_dir)
    print(f"Built indexes for {result['count']} standards")
    print(f"FAISS: {result['faiss']}")
    print(f"BM25: {result['bm25']}")
    print(f"Codes: {result['codes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
