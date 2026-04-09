"""Silver-stage extractor for intro guidance from the official ICD-10 PDF.

Reads bronze page-level records from ON-1 and emits structured
``DocumentRecord`` artifacts for four required intro-guidance topics:
principal diagnosis, symptom fallback, mortality coding, and
official 3-character national default policy.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mnc.datasets._io import iter_jsonl, now_utc, write_jsonl
from mnc.schemas.document import DocumentRecord

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)

SOURCE = "icd10_official_pdf"
EXTRACTOR = "rule_based_intro_guidance"
MATCH_METHOD = "keyword_window"
EXPECTED_TOPIC_COUNT = 4

REQUIRED_TOPICS = (
    "principal_diagnosis",
    "symptom_fallback",
    "mortality_coding",
    "official_3char_policy",
)

# Intro pages in the PDF (roman numeral pages i-vii, mapped to file pages 13-19).
INTRO_PAGE_RANGE = range(13, 20)

# Topic detection: each tuple is (keyword_trigger, page_range).
# Keywords are matched case-insensitively against the page text.
_TOPIC_RULES: dict[str, tuple[tuple[str, ...], range]] = {
    "principal_diagnosis": (
        ("bệnh chính", "chẩn đoán cuối cùng", "chẩn đoán xác định", "bệnh phụ"),
        range(15, 17),
    ),
    "symptom_fallback": (
        (
            "không thể xác định chẩn đoán",
            "dấu hiệu và tình trang",
            "dấu hiệu đặc trưng",
        ),
        range(16, 17),
    ),
    "mortality_coding": (
        ("nguyên nhân tử vong", "tử vong", "nguyên nhân tử vong chính"),
        range(16, 19),
    ),
    "official_3char_policy": (
        ("tạm thời sử dụng bộ mã 3 kí tự", "bộ mã 3 kí tự", "mã 3 kí tự"),
        range(14, 16),
    ),
}


def _read_bronze_pages(input_path: Path) -> dict[int, DocumentRecord]:
    """Read ON-1 bronze page records and return a page_no → record map."""
    pages: dict[int, DocumentRecord] = {}
    for _, rec in iter_jsonl(input_path, DocumentRecord):
        page_no = rec.payload.get("page_no") if rec.payload else None
        if isinstance(page_no, int):
            pages[page_no] = rec
    return pages


def _normalize_text(raw: str) -> str:
    """Normalize whitespace while preserving Vietnamese diacritics."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _split_sentences(text: str) -> list[str]:
    """Sentence split on Vietnamese text using punctuation boundaries."""
    if not text:
        return []
    # Split on sentence-ending punctuation before whitespace+uppercase.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ])",
        text,
    )
    return [s.strip() for s in parts if s.strip()]


def _find_topic_excerpt(
    topic: str,
    pages: dict[int, DocumentRecord],
) -> tuple[str, int, int] | None:
    """Find and extract the supporting excerpt for *topic* from intro pages.

    Returns (concatenated_raw_text, page_start, page_end) or None.
    """
    keywords, page_range = _TOPIC_RULES[topic]
    excerpts: list[str] = []
    first_page: int | None = None
    last_page: int = 0

    for page_no in page_range:
        rec = pages.get(page_no)
        if rec is None:
            continue
        text = rec.raw_text
        text_lower = text.lower()
        if not any(kw in text_lower for kw in keywords):
            continue
        excerpts.append(text)
        if first_page is None:
            first_page = page_no
        last_page = page_no

    if not excerpts or first_page is None:
        return None

    return "\n\n".join(excerpts), first_page, last_page


def _build_record(
    *,
    topic: str,
    raw_text: str,
    page_start: int,
    page_end: int,
    created_at: datetime,
) -> DocumentRecord:
    """Build a ``DocumentRecord`` for one guidance topic."""
    normalized = _normalize_text(raw_text)
    sentences = _split_sentences(normalized)

    page_span = (
        str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    )

    return DocumentRecord(
        doc_id=f"{SOURCE}:intro_guidance:{topic}",
        source=SOURCE,
        language="vi",
        raw_text=raw_text,
        source_record_id=topic,
        split=None,
        payload={
            "topic": topic,
            "page_start": page_start,
            "page_end": page_end,
            "page_span": page_span,
            "extractor": EXTRACTOR,
            "match_method": MATCH_METHOD,
        },
        normalized_text=normalized,
        retrieval_text=normalized,
        sentences=sentences,
        created_at=created_at,
    )


def _validate_record(rec: DocumentRecord) -> str | None:
    """Validate a single record; returns topic string or raises."""
    if rec.source != SOURCE:
        msg = f"Unexpected source: {rec.source!r}"
        raise ValueError(msg)
    if rec.language != "vi":
        msg = f"Unexpected language: {rec.language!r}"
        raise ValueError(msg)
    if rec.raw_text == "":
        msg = f"Empty raw_text for {rec.doc_id}"
        raise ValueError(msg)
    if rec.normalized_text == "":
        msg = f"Empty normalized_text for {rec.doc_id}"
        raise ValueError(msg)

    topic = rec.payload.get("topic") if rec.payload else None
    if not isinstance(topic, str) or topic not in REQUIRED_TOPICS:
        msg = f"Invalid or missing topic in payload for {rec.doc_id}: {topic!r}"
        raise ValueError(msg)

    page_start = rec.payload.get("page_start") if rec.payload else None
    page_end = rec.payload.get("page_end") if rec.payload else None
    if not isinstance(page_start, int) or not isinstance(page_end, int):
        msg = f"Missing page_start/page_end in {rec.doc_id}"
        raise TypeError(msg)
    if page_start > page_end:
        msg = f"page_start > page_end in {rec.doc_id}: {page_start} > {page_end}"
        raise ValueError(msg)

    return topic


def _validate_output(records: list[DocumentRecord]) -> None:
    """Validate output records against spec requirements."""
    if len(records) != EXPECTED_TOPIC_COUNT:
        msg = f"Expected {EXPECTED_TOPIC_COUNT} records, got {len(records)}"
        raise ValueError(msg)

    doc_ids: set[str] = set()
    source_ids: set[str] = set()
    topics: set[str] = set()

    for rec in records:
        if rec.doc_id in doc_ids:
            msg = f"Duplicate doc_id: {rec.doc_id}"
            raise ValueError(msg)
        doc_ids.add(rec.doc_id)

        if rec.source_record_id in source_ids:
            msg = f"Duplicate source_record_id: {rec.source_record_id}"
            raise ValueError(msg)
        source_ids.add(rec.source_record_id)

        topic = _validate_record(rec)
        topics.add(topic)

    missing = set(REQUIRED_TOPICS) - topics
    if missing:
        msg = f"Missing required topics: {missing}"
        raise ValueError(msg)


def extract_intro_guidance(
    input_path: Path = Path(
        "data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    ),
    output_dir: Path = Path("data/silver/icd10_official_pdf/intro_guidance"),
) -> list[DocumentRecord]:
    """Extract required intro guidance topics from ON-1 page records.

    Args:
        input_path: Path to ON-1 bronze JSONL.
        output_dir: Output directory for silver artifacts.

    Returns:
        List of 4 ``DocumentRecord``, one per required topic.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If input is empty or required topics are missing.
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
    records: list[DocumentRecord] = []

    for topic in REQUIRED_TOPICS:
        result = _find_topic_excerpt(topic, pages)
        if result is None:
            msg = f"Cannot find evidence for required topic: {topic}"
            raise ValueError(msg)
        raw_text, page_start, page_end = result
        logger.info(
            "Topic %s: matched pages %s-%s",
            topic,
            page_start,
            page_end,
        )
        rec = _build_record(
            topic=topic,
            raw_text=raw_text,
            page_start=page_start,
            page_end=page_end,
            created_at=created_at,
        )
        records.append(rec)

    _validate_output(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "document_records.jsonl"
    count = write_jsonl(records, jsonl_path)

    logger.info("Records written: %d", count)
    logger.info("Output: %s", jsonl_path)

    return records
