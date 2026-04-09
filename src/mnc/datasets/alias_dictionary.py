"""Build alias dictionary from official ICD-10 PDF."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from mnc.schemas.alias import AliasRecord
from mnc.schemas.document import DocumentRecord
from mnc.schemas.ontology import OntologyCode

logger = logging.getLogger(__name__)

AliasType = Literal[
    "title_vi",
    "title_en",
    "parenthetical",
    "inclusion",
    "nos_form",
    "bilingual_variant",
]
Language = Literal["vi", "en", "mixed", "unknown"]


@dataclass
class AliasMeta:
    """Metadata for alias extraction."""

    type_val: AliasType
    lang: Language
    page: int | None = None
    line: str | None = None


def normalize_alias(text: str) -> str:
    """Normalize alias text deterministically."""
    norm = text.lower()
    norm = re.sub(r"\s+", " ", norm).strip()
    return re.sub(r"[.,:;]+$", "", norm).strip()


def extract_nos_form(text: str) -> str | None:
    """Extract NOS or KXDK form if present."""
    match = re.search(r"\b(nos|kxđk)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _read_ontology(path: Path) -> list[OntologyCode]:
    codes = []
    with path.open(encoding="utf-8") as f:
        for text_line in f:
            line_str = text_line.strip()
            if line_str:
                codes.append(OntologyCode.model_validate_json(line_str))
    return codes


def _read_bronze(path: Path) -> list[DocumentRecord]:
    pages = []
    with path.open(encoding="utf-8") as f:
        for text_line in f:
            line_str = text_line.strip()
            if line_str:
                pages.append(DocumentRecord.model_validate_json(line_str))
    return pages


class AliasBuilder:
    """Helper to build and deduplicate aliases."""

    def __init__(self) -> None:
        """Initialize the alias builder."""
        self.records: list[AliasRecord] = []
        self.seen: set[str] = set()

    def add(self, code_3char: str, alias: str, meta: AliasMeta) -> None:
        """Add an alias if not already seen."""
        alias_norm = normalize_alias(alias)
        if not alias_norm or alias_norm == code_3char.lower():
            return

        alias_id = f"{code_3char}:{alias_norm}"
        if alias_id in self.seen:
            return

        self.seen.add(alias_id)

        record = AliasRecord(
            alias_id=alias_id,
            code_3char=code_3char,
            alias=alias,
            alias_norm=alias_norm,
            alias_type=meta.type_val,
            language=meta.lang,
            source_page=meta.page,
            source_line=meta.line,
            created_at=datetime.now(tz=UTC),
        )
        self.records.append(record)


def _extract_from_ontology(codes: list[OntologyCode], builder: AliasBuilder) -> None:
    """Extract aliases directly from ontology titles."""
    for code in codes:
        builder.add(code.code_3char, code.title_vi, AliasMeta("title_vi", "vi"))
        if code.title_en:
            builder.add(code.code_3char, code.title_en, AliasMeta("title_en", "en"))

        for p in re.findall(r"\(([^)]+)\)", code.title_vi):
            builder.add(code.code_3char, p, AliasMeta("parenthetical", "vi"))

        if code.title_en:
            for p in re.findall(r"\(([^)]+)\)", code.title_en):
                builder.add(code.code_3char, p, AliasMeta("parenthetical", "en"))


def _extract_from_bronze(
    pages: list[DocumentRecord],
    code_map: dict[str, OntologyCode],
    builder: AliasBuilder,
) -> None:
    """Extract aliases heuristically from bronze pages."""
    pages.sort(key=lambda p: int(p.payload.get("page_no", 0)) if p.payload else 0)

    for page in pages:
        page_no = int(page.payload.get("page_no", 0)) if page.payload else None
        lines = page.raw_text.splitlines()
        current_code = None

        for i, raw_line in enumerate(lines):
            line_str = raw_line.strip()
            if not line_str:
                continue

            code_match = re.match(r"^([A-Z]\d{2})\b", line_str)
            if code_match and code_match.group(1) in code_map:
                current_code = code_match.group(1)

            if not current_code:
                continue

            lookahead = lines[i + 1 : i + 4]
            _process_bronze_line(
                line_str,
                current_code,
                page_no,
                lookahead,
                builder,
            )


def _process_bronze_line(
    line_str: str,
    current_code: str,
    page_no: int | None,
    lookahead: list[str],
    builder: AliasBuilder,
) -> None:
    """Process a single bronze line for aliases."""
    if re.match(r"^(bao gồm|includes):?", line_str, flags=re.IGNORECASE):
        content = re.sub(r"^(bao gồm|includes):?\s*", "", line_str, flags=re.IGNORECASE)
        content = content.strip()
        if content:
            builder.add(
                current_code,
                content,
                AliasMeta("inclusion", "mixed", page_no, line_str),
            )

        for next_line_raw in lookahead:
            next_line = next_line_raw.strip()
            if re.match(r"^[A-Z]\d{2}\b", next_line) or re.match(
                r"^(loại trừ|excludes|sử dụng mã|use additional code):",
                next_line,
                flags=re.IGNORECASE,
            ):
                break
            if next_line:
                builder.add(
                    current_code,
                    next_line,
                    AliasMeta("inclusion", "mixed", page_no, next_line),
                )

    if extract_nos_form(line_str):
        builder.add(
            current_code,
            line_str,
            AliasMeta("nos_form", "mixed", page_no, line_str),
        )

    for p in re.findall(r"\(([^)]+)\)", line_str):
        if not re.search(r"[A-Z]\d{2}", p):
            builder.add(
                current_code,
                p,
                AliasMeta("parenthetical", "mixed", page_no, line_str),
            )


def build_icd10_alias_dictionary(
    ontology_path: str
    | Path = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    bronze_path: str
    | Path = "data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    output_dir: str | Path = "data/silver/icd10_official_pdf/alias_dictionary",
) -> list[AliasRecord]:
    """Build a deterministic alias dictionary from official PDF-derived text."""
    ontology_path = Path(ontology_path)
    bronze_path = Path(bronze_path)
    output_dir = Path(output_dir)

    if not ontology_path.exists():
        msg = f"Ontology file missing: {ontology_path}"
        raise FileNotFoundError(msg)
    if not bronze_path.exists():
        msg = f"Bronze file missing: {bronze_path}"
        raise FileNotFoundError(msg)

    ontology_codes = _read_ontology(ontology_path)
    logger.info("Read %d ontology codes", len(ontology_codes))

    bronze_pages = _read_bronze(bronze_path)
    logger.info("Read %d bronze pages", len(bronze_pages))

    builder = AliasBuilder()
    code_map = {oc.code_3char: oc for oc in ontology_codes}

    _extract_from_ontology(ontology_codes, builder)
    _extract_from_bronze(bronze_pages, code_map, builder)

    builder.records.sort(key=lambda r: (r.code_3char, r.alias_norm))

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "alias_records.jsonl"
    csv_path = output_dir / "alias_records.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.writelines(record.model_dump_json() + "\n" for record in builder.records)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(AliasRecord.model_fields)
        for record in builder.records:
            row = [
                getattr(record, k).isoformat()
                if k == "created_at"
                else getattr(record, k)
                for k in AliasRecord.model_fields
            ]
            writer.writerow(row)

    logger.info("Wrote %d alias records to %s", len(builder.records), output_dir)
    return builder.records


def main() -> None:
    """CLI entrypoint for alias dictionary builder."""
    parser = argparse.ArgumentParser(description="Build ICD-10 alias dictionary.")
    parser.add_argument(
        "--ontology",
        default="data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    )
    parser.add_argument(
        "--bronze",
        default="data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/silver/icd10_official_pdf/alias_dictionary",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_icd10_alias_dictionary(args.ontology, args.bronze, args.output_dir)


if __name__ == "__main__":
    main()
