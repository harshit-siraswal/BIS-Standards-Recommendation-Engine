"""Normalize and expand raw BIS recommendation queries."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_SYNONYMS_PATH = Path(__file__).resolve().parents[1] / "data" / "synonyms.json"
WORD_JOIN_PATTERN = re.compile(r"[^a-z0-9]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
IS_CODE_QUERY_PATTERN = re.compile(
    r"\bIS\s*(?:No\.?\s*)?(\d{2,5})\s*"
    r"(?:\(?\s*Part\s*[-:]?\s*(\d+)\s*\)?)?\s*"
    r"(?::|\-)?\s*(\d{4})?",
    re.IGNORECASE,
)
STOP_WORDS = frozenset({"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "be"})


@dataclass(frozen=True)
class ProcessedQuery:
    raw: str
    normalized: str
    expanded: str
    explicit_codes: list[str]
    tokens: list[str]


def normalize_query_text(query: str) -> str:
    """Apply query-time Unicode, whitespace, and case normalization."""
    normalized = unicodedata.normalize("NFKC", str(query or "")).strip()
    return WHITESPACE_PATTERN.sub(" ", normalized.lower())


def normalize_match_text(text: str) -> str:
    """Normalize text for phrase matching across punctuation variants."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.replace("&", " and ")
    normalized = WORD_JOIN_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _normalize_code_number(number: str) -> str:
    return str(int(number)) if str(number).isdigit() else str(number).strip()


def _code_token(match: re.Match[str]) -> str:
    number = _normalize_code_number(match.group(1))
    part = _normalize_code_number(match.group(2)) if match.group(2) else "0"
    year = match.group(3)
    return f" iscode{number}p{part}y{year} " if year else match.group(0)


def fallback_tokenize(text: str) -> list[str]:
    """BM25-style tokenizer used when the Plan 02 indexer is not available."""
    with_code_tokens = IS_CODE_QUERY_PATTERN.sub(_code_token, text or "")
    tokens = TOKEN_PATTERN.findall(with_code_tokens.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def _load_tokenizer() -> Callable[[str], list[str]]:
    try:
        from src.indexer import tokenize
    except ImportError:
        return fallback_tokenize
    return tokenize


def detect_is_codes(query: str) -> list[str]:
    """Return normalized, year-qualified IS codes mentioned in a query."""
    codes: list[str] = []
    seen: set[str] = set()
    for match in IS_CODE_QUERY_PATTERN.finditer(str(query or "")):
        number, part, year = match.group(1), match.group(2), match.group(3)
        if not year:
            continue

        code = f"is{_normalize_code_number(number)}"
        if part:
            code += f"(part{_normalize_code_number(part)})"
        code += f":{year}"
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _ordered_unique(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_query_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


class SynonymExpander:
    """Bidirectional synonym expander backed by ``data/synonyms.json``."""

    def __init__(self, synonyms_path: str | Path = DEFAULT_SYNONYMS_PATH):
        self.synonyms_path = Path(synonyms_path)
        try:
            with self.synonyms_path.open(encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Synonyms file not found: {self.synonyms_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Synonyms file is not valid JSON: {self.synonyms_path}") from exc
        self.flat = self._flatten(data)

    @staticmethod
    def _flatten(data: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
        flat: dict[str, list[str]] = {}
        for mapping in data.values():
            for term, synonyms in mapping.items():
                related = _ordered_unique([term, *synonyms])
                for alias in related:
                    key = normalize_match_text(alias)
                    expansions = [item for item in related if normalize_match_text(item) != key]
                    existing = flat.setdefault(key, [])
                    for expansion in expansions:
                        if expansion not in existing:
                            existing.append(expansion)
        return flat

    def expand(self, query: str) -> str:
        """Append matching synonyms to enrich sparse retrieval scoring."""
        normalized_query = normalize_query_text(query)
        searchable_query = f" {normalize_match_text(normalized_query)} "
        appended: list[str] = []
        appended_match_keys: set[str] = set()

        for term_key in sorted(self.flat, key=lambda key: (-len(key), key)):
            if f" {term_key} " not in searchable_query:
                continue

            for synonym in self.flat[term_key]:
                synonym_key = normalize_match_text(synonym)
                if (
                    not synonym_key
                    or f" {synonym_key} " in searchable_query
                    or synonym_key in appended_match_keys
                ):
                    continue
                appended_match_keys.add(synonym_key)
                appended.append(synonym)

        if not appended:
            return normalized_query
        return f"{normalized_query} {' '.join(appended)}"


class QueryProcessor:
    def __init__(self, synonyms_path: str | Path = DEFAULT_SYNONYMS_PATH):
        self.expander = SynonymExpander(synonyms_path)
        self.tokenize = _load_tokenizer()

    def process(self, query: str) -> ProcessedQuery:
        normalized = normalize_query_text(query)
        expanded = self.expander.expand(normalized)
        explicit_codes = detect_is_codes(query)

        return ProcessedQuery(
            raw=str(query or ""),
            normalized=normalized,
            expanded=expanded,
            explicit_codes=explicit_codes,
            tokens=self.tokenize(expanded),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize and expand a BIS query.")
    parser.add_argument("query", help="Raw natural-language query to process.")
    parser.add_argument(
        "--synonyms",
        default=str(DEFAULT_SYNONYMS_PATH),
        help="Path to the curated synonyms JSON file.",
    )
    args = parser.parse_args(argv)

    processed = QueryProcessor(args.synonyms).process(args.query)
    print(json.dumps(asdict(processed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
