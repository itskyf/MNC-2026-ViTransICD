"""Silver-stage ontology normalizer for official ICD-10 PDF.

Reads bronze page-level records from ON-1 and emits canonical 3-character
ICD entries as ``OntologyCode`` records with bilingual titles and aliases.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from mnc.datasets._io import iter_jsonl, now_utc, write_jsonl
from mnc.schemas.document import DocumentRecord
from mnc.schemas.ontology import OntologyCode

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)


class _PageScan(TypedDict):
    """Accumulated bilingual page scan results."""

    en_titles: dict[str, str]
    vi_titles: dict[str, str]
    en_sub: dict[str, list[str]]
    vi_sub: dict[str, list[str]]
    chapter_map: dict[str, str | None]


SOURCE = "icd10_official_pdf"

CODE_PAGE_START = 20

CODE_3CHAR_RE = re.compile(r"^([A-Z]\d{2})$")

STANDALONE_3CHAR_RE = re.compile(r"^([A-Z]\d{2})\s*$")

CODE_4CHAR_LINE_RE = re.compile(r"^([A-Z]\d{2})\.\d+[†*]?\s+(.+)$")

CHAPTER_EN_RE = re.compile(
    r"^Chapter\s+(X{0,3}IX|X{0,3}IV|X{0,3}V?I{0,3})\b",
    re.IGNORECASE,
)
CHAPTER_VI_RE = re.compile(
    r"^Chương\s+(X{0,3}IX|X{0,3}IV|X{0,3}V?I{0,3})\b",
    re.IGNORECASE,
)

BLOCK_RANGE_RE = re.compile(r"\(([A-Z]\d{2}-[A-Z]\d{2})\)")

# Structural line prefixes indicating end of a title block.
_STRUCT_PREFIXES = (
    "incl",
    "excl",
    "use additional",
    "bao gồm",
    "loại trừ",
    "sử dụng mã",
)

_VI_CHARS = set("ăắằẳẵặâấầẩẫậđêếềểễệôốồổỗộơớờởỡợưứừửữự")


def _chapter_id_from_roman(roman: str) -> str | None:
    """Convert Roman numeral chapter to canonical chapter_id like 'Ch-I'."""
    roman_map: dict[str, int] = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
        "XI": 11,
        "XII": 12,
        "XIII": 13,
        "XIV": 14,
        "XV": 15,
        "XVI": 16,
        "XVII": 17,
        "XVIII": 18,
        "XIX": 19,
        "XX": 20,
        "XXI": 21,
        "XXII": 22,
    }
    num = roman_map.get(roman.upper())
    if num is None:
        return None
    return f"Ch-{num}"


def _is_vi_line(line: str) -> bool:
    """Check if a line contains Vietnamese-specific characters."""
    return bool(set(line) & _VI_CHARS)


def _read_bronze_pages(input_path: Path) -> dict[int, DocumentRecord]:
    """Read ON-1 bronze page records and return page_no → record map."""
    pages: dict[int, DocumentRecord] = {}
    for _, rec in iter_jsonl(input_path, DocumentRecord):
        page_no = rec.payload.get("page_no") if rec.payload else None
        if isinstance(page_no, int):
            pages[page_no] = rec
    return pages


def _find_section_split(lines: list[str]) -> int:
    """Find where the Vietnamese section starts on a bilingual page.

    Strategy 1: find where a standalone 3-char code repeats (EN section
    lists each code once; VI section repeats them).
    Strategy 2: fall back to detecting first Vietnamese non-code line.
    """
    seen_codes: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = STANDALONE_3CHAR_RE.match(stripped)
        if m:
            code = m.group(1)
            if code in seen_codes:
                return i
            seen_codes[code] = i

    # Fallback: first Vietnamese non-code line after EN codes found.
    found_en_codes = bool(seen_codes)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if STANDALONE_3CHAR_RE.match(stripped):
            continue
        if (
            found_en_codes
            and stripped
            and _is_vi_line(stripped)
            and not CODE_4CHAR_LINE_RE.match(stripped)
        ):
            return i

    return len(lines)


def _normalize_title(text: str) -> str:
    """Normalize a title: collapse whitespace, strip."""
    t = text.strip()
    while "  " in t:
        t = t.replace("  ", " ")
    t = t.replace("\n", " ")
    return t.strip()


def _normalize_alias(text: str) -> str:
    """Normalize alias: lowercase, trim, collapse spaces."""
    t = text.strip().lower()
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def _deduplicate_aliases(aliases: list[str]) -> list[str]:
    """Remove duplicates and empty strings from aliases."""
    seen: set[str] = set()
    result: list[str] = []
    for a in aliases:
        norm = _normalize_alias(a)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(a.strip())
    return result


def _save_pending_title(
    code: str | None,
    title_lines: list[str],
    titles: dict[str, str],
) -> None:
    """Save pending 3-char title if valid and not already stored."""
    if code and title_lines:
        full = _normalize_title(" ".join(title_lines))
        if full and code not in titles:
            titles[code] = full


def _is_structural_line(stripped: str) -> bool:
    """Check if a line is structural (inclusion/exclusion/heading)."""
    lower = stripped.lower()
    return (
        any(lower.startswith(p) for p in _STRUCT_PREFIXES)
        or bool(BLOCK_RANGE_RE.search(stripped))
        or bool(CHAPTER_EN_RE.match(stripped))
        or bool(CHAPTER_VI_RE.match(stripped))
    )


def _parse_code_section(
    lines: list[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse a section of lines for 3-char codes and their titles/aliases.

    Returns (titles, sub_entries) where:
      titles: code_3char -> main title
      sub_entries: code_3char -> list of 4-char sub-entry titles (aliases)
    """
    titles: dict[str, str] = {}
    sub_entries: dict[str, list[str]] = {}
    current_3char: str | None = None
    title_lines: list[str] = []
    collecting_title = False

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Skip blank lines and page-number lines.
        if not stripped or (
            stripped.startswith("\u2013") and stripped.endswith("\u2013")
        ):
            i += 1
            continue

        # 4-char code line.
        m4 = CODE_4CHAR_LINE_RE.match(stripped)
        if m4:
            code = m4.group(1)
            title = m4.group(2).strip()
            _save_pending_title(current_3char, title_lines, titles)
            sub_entries.setdefault(code, []).append(title)
            current_3char = None
            collecting_title = False
            title_lines = []
            i += 1
            continue

        # Standalone 3-char code.
        m3 = STANDALONE_3CHAR_RE.match(stripped)
        if m3:
            _save_pending_title(current_3char, title_lines, titles)
            current_3char = m3.group(1)
            collecting_title = True
            title_lines = []
            i += 1
            continue

        # Collecting title lines for current 3-char code.
        if collecting_title and current_3char:
            if _is_structural_line(stripped):
                _save_pending_title(current_3char, title_lines, titles)
                collecting_title = False
                title_lines = []
                current_3char = None
                i += 1
                continue

            # Stop if we hit a new code.
            if STANDALONE_3CHAR_RE.match(stripped) or CODE_4CHAR_LINE_RE.match(
                stripped,
            ):
                _save_pending_title(current_3char, title_lines, titles)
                collecting_title = False
                title_lines = []
                continue

            title_lines.append(stripped)
            i += 1
            continue

        i += 1

    # Save last pending title.
    _save_pending_title(current_3char, title_lines, titles)

    return titles, sub_entries


def _build_search_text(
    code: str,
    title_vi: str,
    title_en: str | None,
    aliases: list[str],
) -> str:
    """Build compact lexical search_text string."""
    parts = [code, title_vi]
    if title_en:
        parts.append(title_en)
    parts.extend(aliases)
    return " | ".join(parts)


def _validate_output(records: list[OntologyCode]) -> None:
    """Validate output records against spec requirements."""
    if not records:
        msg = "No ontology records produced"
        raise ValueError(msg)

    codes: set[str] = set()
    for rec in records:
        if rec.code_3char in codes:
            msg = f"Duplicate code_3char: {rec.code_3char}"
            raise ValueError(msg)
        codes.add(rec.code_3char)

        if not CODE_3CHAR_RE.match(rec.code_3char):
            msg = f"Invalid 3-char code: {rec.code_3char!r}"
            raise ValueError(msg)
        if not rec.title_vi:
            msg = f"Empty title_vi for {rec.code_3char}"
            raise ValueError(msg)
        if not rec.search_text:
            msg = f"Empty search_text for {rec.code_3char}"
            raise ValueError(msg)

        alias_lower = [_normalize_alias(a) for a in rec.aliases]
        if len(alias_lower) != len(set(alias_lower)):
            msg = f"Duplicate aliases for {rec.code_3char}: {rec.aliases}"
            raise ValueError(msg)
        if any(a == "" for a in rec.aliases):
            msg = f"Empty alias string for {rec.code_3char}"
            raise ValueError(msg)


def _merge_page_results(
    page_titles: dict[str, str],
    all_titles: dict[str, str],
    chapter_map: dict[str, str | None],
    current_chapter: str | None,
) -> None:
    """Merge page-level titles into accumulators (first occurrence wins)."""
    for code, title in page_titles.items():
        if code not in all_titles:
            all_titles[code] = title
        if code not in chapter_map:
            chapter_map[code] = current_chapter


BRONZE_JSONL = "data/bronze/icd10_official_pdf/primary/document_records.jsonl"
SILVER_ONTOLOGY_DIR = "data/silver/icd10_official_pdf/normalized_ontology"


def _scan_pages(pages: dict[int, DocumentRecord]) -> _PageScan:
    """Scan code pages for bilingual titles, sub-entries, and chapter map."""
    all_en_titles: dict[str, str] = {}
    all_vi_titles: dict[str, str] = {}
    all_en_sub: dict[str, list[str]] = {}
    all_vi_sub: dict[str, list[str]] = {}
    chapter_map: dict[str, str | None] = {}
    current_chapter: str | None = None

    code_pages = sorted(pn for pn in pages if pn >= CODE_PAGE_START)

    for page_no in code_pages:
        rec = pages[page_no]
        lines = rec.raw_text.split("\n")

        for line in lines:
            stripped = line.strip()
            m = CHAPTER_EN_RE.match(stripped) or CHAPTER_VI_RE.match(stripped)
            if m:
                cid = _chapter_id_from_roman(m.group(1))
                if cid is not None:
                    current_chapter = cid

        split_idx = _find_section_split(lines)
        en_titles, en_sub = _parse_code_section(lines[:split_idx])
        vi_titles, vi_sub = _parse_code_section(lines[split_idx:])

        _merge_page_results(en_titles, all_en_titles, chapter_map, current_chapter)
        _merge_page_results(vi_titles, all_vi_titles, chapter_map, current_chapter)

        for code, subs in en_sub.items():
            all_en_sub.setdefault(code, []).extend(subs)
        for code, subs in vi_sub.items():
            all_vi_sub.setdefault(code, []).extend(subs)

    return _PageScan(
        en_titles=all_en_titles,
        vi_titles=all_vi_titles,
        en_sub=all_en_sub,
        vi_sub=all_vi_sub,
        chapter_map=chapter_map,
    )


def _build_records(
    scan: _PageScan,
    created_at: datetime,
) -> list[OntologyCode]:
    """Build ``OntologyCode`` records from accumulated page data."""
    all_codes = sorted(set(scan["en_titles"]) | set(scan["vi_titles"]))
    if not all_codes:
        msg = "No 3-character codes extracted"
        raise ValueError(msg)

    records: list[OntologyCode] = []
    title_en_count = 0
    alias_count = 0

    for code in all_codes:
        title_vi = scan["vi_titles"].get(code, "")
        title_en = scan["en_titles"].get(code)
        if not title_vi and title_en:
            title_vi = title_en

        aliases = _collect_aliases(
            code,
            title_vi,
            title_en,
            scan["en_sub"],
            scan["vi_sub"],
        )

        if title_en:
            title_en_count += 1
        if aliases:
            alias_count += 1

        search_text = _build_search_text(code, title_vi, title_en, aliases)
        records.append(
            OntologyCode(
                code_3char=code,
                chapter_id=scan["chapter_map"].get(code),
                title_vi=title_vi,
                title_en=title_en or None,
                aliases=aliases,
                search_text=search_text,
                created_at=created_at,
            ),
        )

    logger.info("3-char entries extracted: %d", len(records))
    logger.info("Entries with title_en: %d", title_en_count)
    logger.info("Entries with aliases: %d", alias_count)

    return records


def _collect_aliases(
    code: str,
    title_vi: str,
    title_en: str | None,
    all_en_sub: dict[str, list[str]],
    all_vi_sub: dict[str, list[str]],
) -> list[str]:
    """Collect and deduplicate aliases for a single code."""
    aliases: list[str] = []
    for a in all_en_sub.get(code, []):
        norm = _normalize_alias(a)
        if norm and norm != _normalize_alias(title_en or ""):
            aliases.append(a)
    for a in all_vi_sub.get(code, []):
        norm = _normalize_alias(a)
        if norm and norm != _normalize_alias(title_vi):
            aliases.append(a)
    return _deduplicate_aliases(aliases)


def normalize_icd10_ontology(
    input_path: Path = Path(BRONZE_JSONL),
    output_dir: Path = Path(SILVER_ONTOLOGY_DIR),
) -> list[OntologyCode]:
    """Normalize official 3-character ICD ontology from ON-1 page records.

    Args:
        input_path: Path to ON-1 bronze JSONL.
        output_dir: Output directory for silver artifacts.

    Returns:
        List of ``OntologyCode`` records, one per extracted 3-char code.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If input is empty or no 3-char codes extracted.
    """
    logger.info("Input path: %s", input_path)
    logger.info("Output dir: %s", output_dir)

    if not input_path.exists():
        msg = f"Input not found: {input_path}"
        raise FileNotFoundError(msg)

    pages = _read_bronze_pages(input_path)
    logger.info("Bronze pages read: %d", len(pages))

    if not pages:
        msg = "No bronze page records found"
        raise ValueError(msg)

    created_at = now_utc()
    scan = _scan_pages(pages)
    records = _build_records(scan, created_at)
    _validate_output(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "ontology_codes.jsonl"
    count = write_jsonl(records, jsonl_path)

    logger.info("Records written: %d", count)
    logger.info("Output: %s", jsonl_path)

    return records
