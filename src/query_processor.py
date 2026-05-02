"""Normalize and expand raw BIS recommendation queries."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_SYNONYMS_PATH = Path(__file__).resolve().parents[1] / "data" / "synonyms.json"
DEFAULT_COMPLIANCE_FLAGS_PATH = Path(__file__).resolve().parents[1] / "data" / "compliance_flags.json"
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
PENCIL_TERMS = (
    "pencil",
    "pencils",
    "pensil",
    "pencilon",
    "pencilan",
    "\u092a\u0947\u0902\u0938\u093f\u0932",
    "\u092a\u0947\u0928\u094d\u0938\u093f\u0932",
    "\u0baa\u0bc6\u0ba9\u0bcd\u0b9a\u0bbf\u0bb2\u0bcd",
    "\u0baa\u0bc6\u0ba9\u0bcd\u0b9a\u0bbf\u0bb2\u0bcd\u0b95\u0bb3\u0bcd",
    "\u0c2a\u0c46\u0c28\u0c4d\u0c38\u0c3f\u0c32\u0c4d",
    "\u0c2a\u0c46\u0c28\u0c4d\u0c38\u0c3f\u0c32\u0c4d\u0c38\u0c4d",
    "\u0aaa\u0ac7\u0aa8\u0acd\u0ab8\u0abf\u0ab2",
    "\u0caa\u0cc6\u0ca8\u0ccd\u0cb8\u0cbf\u0cb2\u0ccd",
    "\u0d2a\u0d46\u0d7b\u0d38\u0d3f\u0d7d",
    "\u0d2a\u0d46\u0d7b\u0d38\u0d3f\u0d32\u0d41\u0d15\u0d7e",
    "\u09aa\u09c7\u09a8\u09cd\u09b8\u09bf\u09b2",
    "\u0a2a\u0a48\u0a02\u0a38\u0a3f\u0a32",
    "\u067e\u0646\u0633\u0644",
)
EDIBLE_OIL_TERMS = (
    "edible oil",
    "edible oils",
    "cooking oil",
    "cooking oils",
    "vegetable oil",
    "vegetable oils",
    "khadya tel",
    "khane ka tel",
    "khaane ka tel",
    "\u0916\u093e\u0926\u094d\u092f \u0924\u0947\u0932",
    "\u0916\u093e\u0928\u0947 \u0915\u093e \u0924\u0947\u0932",
    "\u0b89\u0ba3\u0bb5\u0bc1 \u0b8e\u0ba3\u0bcd\u0ba3\u0bc6\u0baf\u0bcd",
    "\u0c24\u0c3f\u0c28\u0c47 \u0c28\u0c42\u0c28\u0c46",
    "\u0a96\u0abe\u0aa6\u0acd\u0aaf \u0aa4\u0ac7\u0ab2",
    "\u0c86\u0cb9\u0cbe\u0cb0 \u0c8e\u0ca3\u0ccd\u0ca3\u0cc6",
    "\u0d2d\u0d15\u0d4d\u0d37\u0d4d\u0d2f \u0d0e\u0d23\u0d4d\u0d23",
    "\u09ad\u09cb\u099c\u09cd\u09af \u09a4\u09c7\u09b2",
    "\u0a16\u0a3e\u0a26 \u0a24\u0a47\u0a32",
    "\u062e\u0648\u0631\u062f\u0646\u06cc \u062a\u06cc\u0644",
)
OUT_OF_SCOPE_PRODUCT_RULES = (
    (
        PENCIL_TERMS,
        "Graphite or black lead pencils are outside the bundled BIS SP 21 building-materials catalog. "
        "Verify pencil-specific BIS standards separately, such as IS 1375:2021 and IS 2079:2022.",
    ),
    (
        EDIBLE_OIL_TERMS,
        "Edible oil is outside the bundled BIS SP 21 building-materials catalog. "
        "Use oil and fats standards such as IS 548 and the relevant oil-type specification.",
    ),
)


@dataclass(frozen=True)
class ProcessedQuery:
    raw: str
    normalized: str
    expanded: str
    explicit_codes: list[str]
    tokens: list[str]
    compliance_warnings: list[str] = field(default_factory=list)
    out_of_scope: bool = False


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


def normalize_script_text(text: str) -> str:
    """Normalize text without stripping Indic and right-to-left scripts."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.replace("\u200c", "").replace("\u200d", "")
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def contains_any_term(text: str, terms: Iterable[str]) -> bool:
    normalized_text = normalize_script_text(text)
    compact_text = normalized_text.replace(" ", "")
    latin_searchable = f" {normalize_match_text(text)} "

    for term in terms:
        normalized_term = normalize_script_text(term)
        if not normalized_term:
            continue
        if TOKEN_PATTERN.search(normalized_term):
            term_key = normalize_match_text(normalized_term)
            if term_key and f" {term_key} " in latin_searchable:
                return True
            continue
        compact_term = normalized_term.replace(" ", "")
        if compact_term and compact_term in compact_text:
            return True
    return False


def is_out_of_scope_product(query: str) -> bool:
    return any(contains_any_term(query, terms) for terms, _warning in OUT_OF_SCOPE_PRODUCT_RULES)


def is_pencil_query(query: str) -> bool:
    return contains_any_term(query, PENCIL_TERMS)


def is_edible_oil_query(query: str) -> bool:
    return contains_any_term(query, EDIBLE_OIL_TERMS)


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
        self._sorted_keys = sorted(self.flat, key=lambda key: (-len(key), key))

    @staticmethod
    def _flatten(data: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
        flat: dict[str, list[str]] = {}
        for mapping in data.values():
            for term, synonyms in mapping.items():
                related = _ordered_unique([term, *synonyms])
                normalized_related: list[tuple[str, str]] = []
                for item in related:
                    item_key = normalize_match_text(item)
                    if item_key:
                        normalized_related.append((item_key, item))
                expansions_by_key: dict[str, list[str]] = {}
                for key, _alias in normalized_related:
                    if key not in expansions_by_key:
                        expansions_by_key[key] = [
                            item
                            for item_key, item in normalized_related
                            if item_key != key
                        ]
                for key, expansions in expansions_by_key.items():
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

        for term_key in self._sorted_keys:
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
    def __init__(
        self,
        synonyms_path: str | Path = DEFAULT_SYNONYMS_PATH,
        compliance_flags_path: str | Path = DEFAULT_COMPLIANCE_FLAGS_PATH,
    ):
        self.expander = SynonymExpander(synonyms_path)
        self.compliance_flags = self._load_compliance_flags(compliance_flags_path)
        self.tokenize = _load_tokenizer()

    @staticmethod
    def _load_compliance_flags(flags_path: str | Path) -> dict[str, str]:
        path = Path(flags_path)
        if not path.exists():
            return {}
        try:
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Compliance flags file is not valid JSON: {path}") from exc

        flags: dict[str, str] = {}
        for term, metadata in (payload.get("regulated_terms") or {}).items():
            key = normalize_match_text(term)
            if not key:
                continue
            if isinstance(metadata, dict):
                warning = str(metadata.get("warning") or "").strip()
            else:
                warning = str(metadata).strip()
            if warning:
                flags[key] = warning
        return flags

    def _compliance_warnings(self, normalized: str, expanded: str) -> list[str]:
        searchable = f" {normalize_match_text(normalized)} {normalize_match_text(expanded)} "
        warnings: list[str] = []
        for term_key, warning in self.compliance_flags.items():
            if f" {term_key} " in searchable and warning not in warnings:
                warnings.append(warning)
        for terms, warning in OUT_OF_SCOPE_PRODUCT_RULES:
            if contains_any_term(normalized, terms) or contains_any_term(expanded, terms):
                warnings.append(warning)
        return warnings

    @staticmethod
    def _is_out_of_scope(normalized: str) -> bool:
        return is_out_of_scope_product(normalized)

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
            compliance_warnings=self._compliance_warnings(normalized, expanded),
            out_of_scope=self._is_out_of_scope(normalized),
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
