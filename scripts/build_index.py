"""Rebuild the parsed BIS catalog and retrieval indexes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexer import build_all_indexes  # noqa: E402
from src.parser import parse_pdf_to_catalog  # noqa: E402


def main() -> int:
    catalog_path = ROOT / "data" / "standards_catalog.json"
    data_dir = ROOT / "data"
    try:
        parse_pdf_to_catalog(str(ROOT / "dataset.pdf"), str(catalog_path))
        build_all_indexes(str(catalog_path), str(data_dir))
    except Exception as exc:
        print(f"Failed to rebuild BIS index artifacts: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
