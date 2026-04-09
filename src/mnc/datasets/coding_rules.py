"""Extract coding rules from official ICD-10 PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from mnc.schemas.document import DocumentRecord
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.rule import RuleRecord

logger = logging.getLogger(__name__)

RuleTopic = Literal[
    "principal_diagnosis",
    "symptom_fallback",
    "mortality_coding",
    "official_3char_policy",
    "include_note",
    "exclude_note",
    "use_additional_code",
    "code_first",
    "general_note",
]

RuleAction = Literal[
    "prefer",
    "fallback",
    "restrict",
    "allow",
    "require_additional_code",
    "code_first",
    "note",
]


def hash_text(text: str) -> str:
    """Generate an 8-character hash for stable ID generation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def normalize_rule_text(text: str) -> str:
    """Normalize rule text."""
    return re.sub(r"\s+", " ", text).strip()


def _read_docs(path: Path) -> list[DocumentRecord]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for text_line in f:
            line_str = text_line.strip()
            if line_str:
                docs.append(DocumentRecord.model_validate_json(line_str))
    return docs


def _process_global_rules(
    intro_records: list[DocumentRecord],
) -> tuple[list[RuleRecord], set[str]]:
    rules = []
    found_topics = set()

    for rec in intro_records:
        topic = rec.doc_id.split(":")[-1] if ":" in rec.doc_id else rec.doc_id

        topic_val: RuleTopic | None = None
        action_val: RuleAction | None = None
        priority = 0

        if topic == "principal_diagnosis":
            topic_val = "principal_diagnosis"
            action_val = "prefer"
            priority = 90
        elif topic == "symptom_fallback":
            topic_val = "symptom_fallback"
            action_val = "fallback"
            priority = 70
        elif topic == "mortality_coding":
            topic_val = "mortality_coding"
            action_val = "prefer"
            priority = 80
        elif topic == "official_3char_policy":
            topic_val = "official_3char_policy"
            action_val = "restrict"
            priority = 100

        if topic_val and action_val:
            rule_id = f"global:{topic_val}"
            norm_text = normalize_rule_text(rec.raw_text)
            page_no = (
                int(rec.payload.get("page_no", 0))
                if rec.payload and "page_no" in rec.payload
                else None
            )

            rule = RuleRecord(
                rule_id=rule_id,
                scope="global",
                topic=topic_val,
                action=action_val,
                priority=priority,
                source_page_start=page_no,
                source_page_end=None,
                evidence_text=rec.raw_text,
                normalized_text=norm_text,
                created_at=datetime.now(tz=UTC),
            )
            rules.append(rule)
            found_topics.add(topic_val)

    return rules, found_topics


def _get_code_rule_attrs(
    lower_line: str,
) -> tuple[RuleTopic | None, RuleAction | None, int | None]:
    if re.match(r"^(bao gồm|includes):?", lower_line):
        return "include_note", "allow", 30
    if re.match(r"^(loại trừ|excludes):?", lower_line):
        return "exclude_note", "restrict", 40
    if re.match(r"^(sử dụng mã|mã thêm|use additional code):?", lower_line):
        return "use_additional_code", "require_additional_code", 50
    if re.match(r"^(code first):?", lower_line):
        return "code_first", "code_first", 60
    if re.match(r"^(ghi chú|note):?", lower_line):
        return "general_note", "note", 10
    return None, None, None


def _capture_block_text(start_line: str, i: int, lines: list[str]) -> str:
    block_text = [start_line]
    stop_pattern = (
        r"^(bao gồm|includes|loại trừ|excludes|sử dụng mã|mã thêm|"
        r"use additional code|code first|ghi chú|note):?"
    )
    for j in range(1, 4):
        if i + j >= len(lines):
            break
        next_line = lines[i + j].strip()
        if re.match(r"^[A-Z]\d{2}\b", next_line) or re.match(
            stop_pattern,
            next_line,
            flags=re.IGNORECASE,
        ):
            break
        if next_line:
            block_text.append(next_line)
    return "\n".join(block_text)


def _process_code_rules(
    bronze_pages: list[DocumentRecord],
    code_map: set[str],
) -> list[RuleRecord]:
    rules = []
    bronze_pages.sort(
        key=lambda p: int(p.payload.get("page_no", 0)) if p.payload else 0,
    )

    for page in bronze_pages:
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

            topic, action, priority = _get_code_rule_attrs(line_str.lower())
            if topic and action and priority:
                evidence = _capture_block_text(line_str, i, lines)
                norm_text = normalize_rule_text(evidence)
                rule_id = f"{current_code}:{topic}:{hash_text(norm_text)}"

                rules.append(
                    RuleRecord(
                        rule_id=rule_id,
                        scope="code",
                        code_3char=current_code,
                        topic=topic,
                        action=action,
                        priority=priority,
                        source_page_start=page_no,
                        source_page_end=page_no,
                        evidence_text=evidence,
                        normalized_text=norm_text,
                        created_at=datetime.now(tz=UTC),
                    ),
                )
    return rules


def extract_icd10_coding_rules(
    intro_path: str
    | Path = "data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl",
    ontology_path: str
    | Path = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    bronze_path: str
    | Path = "data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    output_dir: str | Path = "data/silver/icd10_official_pdf/coding_rules",
) -> list[RuleRecord]:
    """Extract global and per-code rules from PDF-derived artifacts."""
    intro_path, ontology_path, bronze_path, output_dir = (
        Path(intro_path),
        Path(ontology_path),
        Path(bronze_path),
        Path(output_dir),
    )

    for p in (intro_path, ontology_path, bronze_path):
        if not p.exists():
            msg = f"Input file missing: {p}"
            raise FileNotFoundError(msg)

    intro_records = _read_docs(intro_path)
    logger.info("Read %d intro records", len(intro_records))

    global_rules, found_global_topics = _process_global_rules(intro_records)

    code_map = set()
    with ontology_path.open(encoding="utf-8") as f:
        for text_line in f:
            line_str = text_line.strip()
            if line_str:
                code_map.add(OntologyCode.model_validate_json(line_str).code_3char)
    logger.info("Read %d ontology codes", len(code_map))

    bronze_pages = _read_docs(bronze_path)
    logger.info("Read %d bronze pages", len(bronze_pages))

    code_rules = _process_code_rules(bronze_pages, code_map)
    rules = sorted(
        global_rules + code_rules,
        key=lambda r: (r.scope, r.code_3char or "", r.topic, r.rule_id),
    )

    seen = set()
    unique_rules = []
    for r in rules:
        if r.rule_id not in seen:
            seen.add(r.rule_id)
            unique_rules.append(r)

    if not unique_rules:
        msg = "No rule records extracted."
        raise ValueError(msg)

    for req_topic in [
        "principal_diagnosis",
        "symptom_fallback",
        "mortality_coding",
        "official_3char_policy",
    ]:
        if req_topic not in found_global_topics:
            msg = f"Missing required global topic: {req_topic}"
            raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "rule_records.jsonl"
    template_path = output_dir / "heuristic_templates.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        f.writelines(r.model_dump_json() + "\n" for r in unique_rules)

    templates = [
        {
            "rule_id": r.rule_id,
            "scope": r.scope,
            "code_3char": r.code_3char,
            "topic": r.topic,
            "action": r.action,
            "priority": r.priority,
            "normalized_text": r.normalized_text,
        }
        for r in unique_rules
    ]

    with template_path.open("w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

    logger.info(
        "Wrote %d global rules and %d code rules",
        len(found_global_topics),
        len(unique_rules) - len(found_global_topics),
    )

    return unique_rules


def main() -> None:
    """CLI entrypoint for coding rules extraction."""
    parser = argparse.ArgumentParser(description="Extract ICD-10 coding rules.")
    parser.add_argument(
        "--intro",
        default="data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl",
    )
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
        default="data/silver/icd10_official_pdf/coding_rules",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    extract_icd10_coding_rules(args.intro, args.ontology, args.bronze, args.output_dir)


if __name__ == "__main__":
    main()
