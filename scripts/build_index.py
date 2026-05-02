"""Rebuild the parsed BIS catalog and retrieval indexes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexer import build_all_indexes  # noqa: E402
from src.parser import parse_pdf_to_catalog  # noqa: E402


def main() -> int:
    parse_pdf_to_catalog("dataset.pdf", "data/standards_catalog.json")
    build_all_indexes("data/standards_catalog.json", "data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
