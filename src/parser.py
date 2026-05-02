"""PDF parser and standards extractor for the BIS recommendation catalog."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import fitz

try:
    import pdfplumber
except ImportError:  # pragma: no cover - exercised only in minimal environments
    pdfplumber = None


LOGGER = logging.getLogger(__name__)

IS_CODE_PATTERN = re.compile(
    r"\b[IiLl]S[\s:]*(\d{2,5})\s*"
    r"(?:\(\s*Part\s*(\d+)\s*\))?\s*[:\-]\s*(\d{4})",
    re.IGNORECASE,
)

SUMMARY_PATTERN = re.compile(r"\bSUMMARY\s+OF\b", re.IGNORECASE)
SCOPE_PATTERN = re.compile(r"\b(?:\d+(?:\.\d+)?\.?\s*)?Scope\b\s*[\u2013\u2014:\-]?", re.IGNORECASE)
SECTION_BREAK_PATTERN = re.compile(
    r"\n\s*\d+(?:\.\d+)?\.?\s*"
    r"(?:Application|Classification|Chemical|Delivery|Dimensions|General|"
    r"Material|Materials|Physical|Quality|Raw|Requirements|Tests?|Workmanship)\b",
    re.IGNORECASE,
)

BODY_LIMIT = 3000
BEFORE_CONTEXT = 2600
AFTER_CONTEXT = 3200

DOMAIN_TERMS = {
    "cement": [
        "cement",
        "opc",
        "ppc",
        "portland cement",
        "ordinary portland cement",
        "portland pozzolana cement",
        "portland slag cement",
        "white portland cement",
        "masonry cement",
        "supersulphated cement",
        "hydrophobic cement",
        "rapid hardening cement",
    ],
    "aggregate": [
        "aggregate",
        "aggregates",
        "sand",
        "gravel",
        "coarse aggregate",
        "fine aggregate",
        "structural concrete",
    ],
    "concrete": [
        "concrete",
        "precast concrete",
        "reinforced concrete",
        "masonry",
        "block",
        "blocks",
        "pipes",
    ],
    "pipe": ["pipe", "pipes", "conduit", "conduits", "water mains", "sewer"],
    "block": ["block", "blocks", "masonry", "brick", "lightweight concrete"],
    "sheet": ["sheet", "sheets", "roofing", "cladding", "asbestos cement"],
    "steel": ["steel", "reinforcement", "structural steel", "bars", "wire"],
    "timber": ["timber", "wood", "plywood", "veneer", "board"],
    "lime": ["lime", "hydrated lime", "building lime"],
    "bitumen": ["bitumen", "tar", "waterproofing", "damp proofing"],
    "sanitary": ["sanitary", "water fitting", "cistern", "tap", "valve"],
}

STOP_TITLE_PREFIXES = (
    "indian standard",
    "specification for",
    "summary of",
)


@dataclass
class Standard:
    is_code: str
    is_code_normalized: str
    number: str
    year: str
    part: Optional[str]
    title: str
    scope: str
    body: str
    keywords: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Normalize PDF text while preserving useful line breaks."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    return normalized


def clean_inline_text(text: str) -> str:
    """Collapse line breaks and repeated spaces for catalog fields."""
    text = normalize_text(text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;,.-\n\t")


def format_is_code(number: str, part: Optional[str], year: str) -> str:
    number = str(int(number)) if number.isdigit() else number.strip()
    year = year.strip()
    if part:
        return f"IS {number} (Part {int(part)}): {year}"
    return f"IS {number}: {year}"


def normalize_is_code(is_code: str) -> str:
    return is_code.replace(" ", "").lower()


def _extract_text_with_fitz(pdf_path: Path) -> str:
    document = fitz.open(pdf_path)
    page_texts: list[str] = []
    for index in range(document.page_count):
        try:
            text = document.load_page(index).get_text() or ""
        except Exception as exc:  # pragma: no cover - depends on malformed PDFs
            LOGGER.warning("PyMuPDF failed on page %s: %s", index + 1, exc)
            text = ""
        page_texts.append(f"\n[PAGE {index + 1}]\n{text}")
    document.close()
    return normalize_text("".join(page_texts))


def _extract_text_with_pdfplumber(pdf_path: Path) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed")

    page_texts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - depends on malformed PDFs
                LOGGER.warning("pdfplumber failed on page %s: %s", index + 1, exc)
                text = ""
            page_texts.append(f"\n[PAGE {index + 1}]\n{text}")
    return normalize_text("".join(page_texts))


def extract_full_text(pdf_path: str, prefer_pdfplumber: bool = False) -> str:
    """Extract full PDF text with page separators.

    The bundled BIS PDF opens very slowly with pdfplumber, so the default path
    uses PyMuPDF for the full pass. pdfplumber remains available through
    prefer_pdfplumber=True for PDFs where its layout extraction is required.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if prefer_pdfplumber:
        try:
            text = _extract_text_with_pdfplumber(path)
            if text.strip():
                return text
        except Exception as exc:
            LOGGER.warning("pdfplumber extraction failed, falling back to PyMuPDF: %s", exc)

    text = _extract_text_with_fitz(path)
    if text.strip():
        return text

    LOGGER.warning("PyMuPDF returned empty text, trying pdfplumber fallback")
    return _extract_text_with_pdfplumber(path)


def _line_is_noise(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return True
    lowered = compact.lower()
    if lowered.startswith("[page "):
        return True
    if lowered.startswith("sp 21"):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", compact):
        return True
    if re.fullmatch(r"\(?[a-z ]*revision\)?", lowered):
        return True
    return False


def _clean_title(title: str) -> str:
    title = clean_inline_text(title)
    title = re.sub(r"\((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+revis(?:ion|on)\)", "", title, flags=re.I)
    title = clean_inline_text(title)
    for prefix in STOP_TITLE_PREFIXES:
        if title.lower().startswith(prefix):
            title = clean_inline_text(title[len(prefix) :])
    return title


def extract_title(window: str, code_relative_end: int) -> str:
    """Extract a title from the line containing the IS code and following lines."""
    after = window[code_relative_end : code_relative_end + 700]
    lines = after.splitlines()
    title_lines: list[str] = []

    for raw_line in lines[:8]:
        line = clean_inline_text(raw_line)
        if re.fullmatch(r"\(?[a-z ]*revision\)?", line.lower()) and title_lines:
            break
        if _line_is_noise(line):
            continue
        if title_lines and re.match(r"^[a-z]", line):
            break
        lowered = line.lower()
        nested_code = IS_CODE_PATTERN.search(line)
        if nested_code:
            line = clean_inline_text(line[: nested_code.start()])
            if line:
                title_lines.append(line)
            break
        if (
            lowered.startswith("note")
            or lowered.startswith("for detailed")
            or lowered.startswith("table")
            or lowered.startswith("scope")
            or re.match(r"^\d+(?:\.\d+)?\.?\s+(scope|application|chemical|classification|delivery|dimensions|physical|requirements)\b", lowered)
        ):
            break
        line = re.sub(r"\((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+revis(?:ion|on)\).*", "", line, flags=re.I)
        if line:
            title_lines.append(line)
        if len(title_lines) >= 4:
            break

    title = _clean_title(" ".join(title_lines))
    return title or "Untitled standard"


def _scope_text_from_match(segment: str, match: re.Match[str]) -> str:
    scope_text = segment[match.end() : match.end() + 1600]
    break_match = SECTION_BREAK_PATTERN.search(scope_text)
    if break_match and break_match.start() >= 40:
        scope_text = scope_text[: break_match.start()]
    scope = clean_inline_text(scope_text)
    scope = re.sub(r"^\d+(?:\.\d+)?\.?\s*", "", scope)
    return scope[:700]


def _extract_scope_from_segment(segment: str, prefer_last: bool = False) -> str:
    matches = list(SCOPE_PATTERN.finditer(segment))
    if not matches:
        return ""

    match = matches[-1] if prefer_last else matches[0]
    return _scope_text_from_match(segment, match)


def extract_scope(
    window: str,
    title: str = "",
    code_relative_start: Optional[int] = None,
    code_relative_end: Optional[int] = None,
) -> str:
    candidates: list[tuple[int, str]] = []

    if code_relative_start is not None:
        before_segment = window[:code_relative_start]
        for match in SCOPE_PATTERN.finditer(before_segment):
            text = _scope_text_from_match(before_segment, match)
            if text:
                distance = code_relative_start - match.start()
                candidates.append((distance, text))

    if code_relative_end is not None:
        after_segment = window[code_relative_end:]
        for match in SCOPE_PATTERN.finditer(after_segment):
            text = _scope_text_from_match(after_segment, match)
            if text:
                crosses_page = "[PAGE " in after_segment[: match.start()]
                penalty = 2000 if crosses_page and candidates else 0
                candidates.append((match.start() + penalty, text))

    if candidates:
        return sorted(candidates, key=lambda item: item[0])[0][1]

    scope = _extract_scope_from_segment(window, prefer_last=False)
    if scope:
        return scope

    match = SCOPE_PATTERN.search(window)
    if match:
        scope_text = window[match.end() : match.end() + 1600]
        break_match = SECTION_BREAK_PATTERN.search(scope_text)
        if break_match and break_match.start() >= 40:
            scope_text = scope_text[: break_match.start()]
        scope = clean_inline_text(scope_text)
        if scope:
            return scope[:700]

    detailed_match = re.search(
        r"For detailed information,\s*refer to\s+IS\s+\d{2,5}.*?(?:Specification for\s+)?(.{20,220})",
        window,
        re.IGNORECASE | re.DOTALL,
    )
    if detailed_match:
        fallback = clean_inline_text(detailed_match.group(1))
        if fallback:
            return fallback[:700]

    for paragraph in re.split(r"\n\s*\n", window):
        paragraph = clean_inline_text(paragraph)
        if len(paragraph) > 40 and not paragraph.lower().startswith(("summary of", "sp 21")):
            return paragraph[:700]

    return title or "Scope not available in extracted text"


def _candidate_quality(window: str, title: str, scope: str, keywords: list[str], match_start: int) -> float:
    prefix = window[max(0, match_start - 80) : match_start]
    quality = 0.0
    if SUMMARY_PATTERN.search(prefix):
        quality += 100.0
    if title and title != "Untitled standard":
        quality += 35.0
    if scope and "not available" not in scope.lower():
        quality += 35.0
    quality += min(len(keywords), 12)
    quality += min(len(window), BODY_LIMIT) / 300.0
    return quality


def _title_phrases(text: str) -> Iterable[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)
    for size in (4, 3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[index : index + size]).lower()
            if len(phrase) > 6:
                yield phrase


def extract_keywords(title: str, scope: str) -> List[str]:
    keywords: set[str] = set()
    combined = clean_inline_text(f"{title} {scope}")
    text_low = combined.lower()

    if title and title != "Untitled standard":
        keywords.add(title.lower())
        for phrase in _title_phrases(title):
            keywords.add(phrase)

    for canonical, variants in DOMAIN_TERMS.items():
        if any(variant in text_low for variant in variants):
            keywords.add(canonical)
            keywords.update(variants)

    for grade in re.findall(r"\b(\d{2})\s*grade\b", text_low):
        keywords.add(f"{grade} grade")

    for number in re.findall(r"\b\d{2,5}\b", text_low):
        if len(keywords) >= 12:
            break
        keywords.add(number)

    for token in re.findall(r"\b[a-z][a-z]{3,}\b", text_low):
        if len(keywords) >= 18:
            break
        if token not in {"shall", "with", "from", "this", "that", "standard", "requirements"}:
            keywords.add(token)

    if len(keywords) < 3 and title:
        for token in title.lower().split():
            if len(token) > 2:
                keywords.add(token.strip("(),.;:"))
            if len(keywords) >= 3:
                break

    return sorted(keyword for keyword in keywords if keyword)[:30]


def _candidate_from_match(full_text: str, match: re.Match[str]) -> tuple[Standard, float]:
    number, part, year = match.group(1), match.group(2), match.group(3)
    canonical = format_is_code(number, part, year)
    start = max(0, match.start() - BEFORE_CONTEXT)
    end = min(len(full_text), match.end() + AFTER_CONTEXT)
    window = full_text[start:end]
    relative_start = match.start() - start
    relative_end = match.end() - start

    title = extract_title(window, relative_end)
    scope = extract_scope(
        window,
        title=title,
        code_relative_start=relative_start,
        code_relative_end=relative_end,
    )
    body = clean_inline_text(window)[:BODY_LIMIT]
    keywords = extract_keywords(title, scope)

    code_keyword = canonical.lower()
    normalized_keyword = normalize_is_code(canonical)
    keyword_set = set(keywords)
    keyword_set.update({code_keyword, normalized_keyword, number, year})
    while len(keyword_set) < 3:
        keyword_set.add(canonical.lower())
    keywords = sorted(keyword_set)[:30]

    standard = Standard(
        is_code=canonical,
        is_code_normalized=normalize_is_code(canonical),
        number=number,
        year=year,
        part=str(int(part)) if part else None,
        title=title,
        scope=scope,
        body=body,
        keywords=keywords,
    )
    quality = _candidate_quality(window, title, scope, keywords, relative_start)
    return standard, quality


def parse_pdf_to_catalog(pdf_path: str, output_path: str) -> List[Standard]:
    full_text = extract_full_text(pdf_path)
    candidates: dict[str, tuple[Standard, float]] = {}

    for match in IS_CODE_PATTERN.finditer(full_text):
        standard, quality = _candidate_from_match(full_text, match)
        key = standard.is_code_normalized
        previous = candidates.get(key)
        if previous is None or quality > previous[1]:
            candidates[key] = (standard, quality)

    standards = sorted(
        (standard for standard, _quality in candidates.values()),
        key=lambda item: (int(item.number) if item.number.isdigit() else 0, item.part or "", item.year),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump([standard.to_dict() for standard in standards], file, indent=2, ensure_ascii=False)

    print(f"Extracted {len(standards)} standards -> {output}")
    return standards


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parse_pdf_to_catalog("dataset.pdf", "data/standards_catalog.json")


if __name__ == "__main__":
    main()
