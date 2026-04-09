"""Bronze-stage parser for the official ICD-10 PDF.

Reads the official Vietnamese ICD-10 PDF and emits page-level
``DocumentRecord`` artifacts for downstream ontology parsing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pymupdf

from mnc.schemas.document import DocumentRecord

logger = logging.getLogger(__name__)

DEFAULT_PDF_URL = (
    "https://soyte.laichau.gov.vn/upload/20762/20191121/ICD-10--tap-1_b55f6.pdf"
)
SOURCE = "icd10_official_pdf"
EXTRACTOR = "pymupdf"


@dataclass(frozen=True, slots=True)
class _PageContext:
    """Shared PDF metadata attached to every page record."""

    pdf_url: str
    pdf_sha256: str
    total_pages: int
    created_at: datetime


def _download_pdf(url: str, *, timeout: float = 120.0) -> bytes:
    """Download a PDF from *url* and return raw bytes."""
    logger.info("Downloading PDF from %s", url)
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_text(raw: str) -> str:
    """Minimal whitespace cleanup while preserving Vietnamese diacritics."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text from each page, returning ``(page_no, text)`` pairs.

    *page_no* is 1-indexed.
    """
    records: list[tuple[int, str]] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = doc.page_count
        if total == 0:
            msg = "PDF contains zero pages"
            raise ValueError(msg)
        for i in range(total):
            page = doc[i]
            raw = page.get_text()
            cleaned = _clean_text(raw)
            records.append((i + 1, cleaned))
    return records


def _build_record(
    *,
    page_no: int,
    text: str,
    ctx: _PageContext,
) -> DocumentRecord:
    """Build a single ``DocumentRecord`` for one PDF page."""
    is_empty = text == ""
    return DocumentRecord(
        doc_id=f"{SOURCE}:page:{page_no}",
        source=SOURCE,
        language="vi",
        raw_text=text,
        source_record_id=str(page_no),
        split=None,
        payload={
            "page_no": page_no,
            "pdf_url": ctx.pdf_url,
            "extractor": EXTRACTOR,
            "is_empty_page": is_empty,
            "pdf_sha256": ctx.pdf_sha256,
            "total_pages": ctx.total_pages,
        },
        normalized_text="",
        retrieval_text="",
        sentences=[],
        created_at=ctx.created_at,
    )


def parse_icd10_official_pdf(
    pdf_source: str,
    output_dir: str = "data/bronze/icd10_official_pdf/primary",
) -> list[DocumentRecord]:
    """Parse the official ICD-10 PDF into page-level ``DocumentRecord`` artifacts.

    Args:
        pdf_source: URL or local path to the PDF file.
        output_dir: Directory for output artifacts.

    Returns:
        List of ``DocumentRecord``, one per PDF page.

    Raises:
        FileNotFoundError: If a local PDF path does not exist.
        ValueError: If the PDF has zero pages.
    """
    logger.info("Source: %s", pdf_source)
    logger.info("Output dir: %s", output_dir)

    source_path = Path(pdf_source)
    if source_path.exists():
        pdf_bytes = source_path.read_bytes()
        pdf_url = str(source_path.resolve())
    elif pdf_source.startswith(("http://", "https://")):
        pdf_bytes = _download_pdf(pdf_source)
        pdf_url = pdf_source
    else:
        msg = f"PDF not found: {pdf_source}"
        raise FileNotFoundError(msg)

    pdf_hash = _sha256(pdf_bytes)
    logger.info("PDF SHA-256: %s", pdf_hash)

    pages = _extract_pages(pdf_bytes)
    total_pages = len(pages)
    logger.info("Total pages discovered: %d", total_pages)

    ctx = _PageContext(
        pdf_url=pdf_url,
        pdf_sha256=pdf_hash,
        total_pages=total_pages,
        created_at=datetime.now(UTC),
    )

    records: list[DocumentRecord] = []
    empty_count = 0
    for page_no, text in pages:
        rec = _build_record(page_no=page_no, text=text, ctx=ctx)
        records.append(rec)
        if rec.payload and rec.payload.get("is_empty_page"):
            empty_count += 1

    _validate_records(records, total_pages)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "document_records.jsonl"
    _write_jsonl(records, jsonl_path)

    logger.info("Empty pages: %d", empty_count)
    logger.info("Total records written: %d", len(records))
    logger.info("Output: %s", jsonl_path)

    return records


def _validate_records(records: list[DocumentRecord], expected_total: int) -> None:
    """Run structural validation checks on *records*."""
    if not records:
        msg = "No records produced"
        raise ValueError(msg)

    doc_ids: set[str] = set()
    source_ids: set[str] = set()

    for rec in records:
        if rec.doc_id in doc_ids:
            msg = f"Duplicate doc_id: {rec.doc_id}"
            raise ValueError(msg)
        doc_ids.add(rec.doc_id)

        if rec.source_record_id in source_ids:
            msg = f"Duplicate source_record_id: {rec.source_record_id}"
            raise ValueError(msg)
        source_ids.add(rec.source_record_id)

        if rec.source != SOURCE:
            msg = f"Unexpected source: {rec.source!r}"
            raise ValueError(msg)
        if rec.language != "vi":
            msg = f"Unexpected language: {rec.language!r}"
            raise ValueError(msg)

    page_numbers = sorted(int(rec.source_record_id) for rec in records)
    expected = list(range(1, expected_total + 1))
    if page_numbers != expected:
        got_preview = page_numbers[:5]
        exp_preview = expected[:5]
        msg = f"Non-contiguous pages: got {got_preview}... expected {exp_preview}..."
        raise ValueError(msg)

    if len(records) != expected_total:
        msg = f"Record count {len(records)} != expected {expected_total}"
        raise ValueError(msg)


def _write_jsonl(records: list[DocumentRecord], path: Path) -> None:
    """Write records as JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
