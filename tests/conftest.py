"""Shared test fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from mnc.ontology._types import (
    ErrorEntry,
    JsonValue,
    ManifestEntry,
    RawEnvelope,
)

if TYPE_CHECKING:
    from pathlib import Path

# -- Ontology fixtures (matching real KCB API format) -----------------------

SAMPLE_ROOT_VI = {
    "status": "success",
    "data": [
        {
            "model": "chapter",
            "id": "1",
            "is_leaf": False,
            "data": {
                "code": "I",
                "id": "1",
                "name": "Certain infectious diseases (A00-B99)",
            },
        },
        {
            "model": "chapter",
            "id": "2",
            "is_leaf": False,
            "data": {"code": "II", "id": "2", "name": "Neoplasms (C00-D49)"},
        },
    ],
    "html": "",
}

SAMPLE_ROOT_DUAL = {
    "status": "success",
    "data": [
        {
            "model": "chapter",
            "id": "1",
            "is_leaf": False,
            "data": {
                "code": "I",
                "id": "1",
                "name": "Một số bệnh truyền nhiễm (A00-B99)",
                "name_en": "Certain infectious diseases (A00-B99)",
            },
        },
        {
            "model": "chapter",
            "id": "2",
            "is_leaf": False,
            "data": {
                "code": "II",
                "id": "2",
                "name": "U (C00-D49)",
                "name_en": "Neoplasms (C00-D49)",
            },
        },
    ],
    "html": "",
}

SAMPLE_CHAPTER = {
    "status": "success",
    "data": [
        {
            "model": "section",
            "id": "A00-A09",
            "is_leaf": False,
            "data": {"code": "A00-A09", "name": "Intestinal infectious diseases"},
        },
        {
            "model": "section",
            "id": "A15-A19",
            "is_leaf": False,
            "data": {"code": "A15-A19", "name": "Tuberculosis"},
        },
    ],
}

SAMPLE_SECTION = {
    "status": "success",
    "data": [
        {
            "model": "type",
            "id": "A00",
            "is_leaf": False,
            "data": {"code": "A00", "name": "Cholera"},
        },
    ],
}

SAMPLE_TYPE = {
    "status": "success",
    "data": [
        {
            "model": "disease",
            "id": "A00.0",
            "is_leaf": True,
            "data": {
                "code": "A00.0",
                "name": "Cholera due to Vibrio cholerae 01, biovar cholerae",
            },
        },
    ],
}

SAMPLE_DISEASE = {
    "status": "success",
    "data": {
        "code": "A00.0",
        "name": "Cholera due to Vibrio cholerae 01, biovar cholerae",
        "id": "A00.0",
    },
}


# -- Ontology helpers -------------------------------------------------------


def make_raw_envelope(
    *,
    endpoint_kind: str = "data",
    node_id: str = "1",
    lang: str = "vi",
    data: JsonValue = None,
) -> RawEnvelope:
    """Build a raw envelope like the crawler saves to disk."""
    return RawEnvelope(
        request_url=f"https://ccs.whiteneuron.com/api/ICD10/{endpoint_kind}/x?id={node_id}&lang={lang}",
        endpoint_kind=endpoint_kind,
        id=node_id,
        lang=lang,
        http_status=200,
        headers={"content-type": "application/json"},
        retrieved_at="2025-01-01T00:00:00+00:00",
        data=data if data is not None else {},
    )


def write_manifest(tmp: Path, entries: list[ManifestEntry]) -> None:
    """Write a crawl manifest JSON file."""
    (tmp / "crawl_manifest.json").write_text(json.dumps(entries, ensure_ascii=False))


def write_errors(tmp: Path, errors: list[ErrorEntry]) -> None:
    """Write a crawl errors JSON file."""
    (tmp / "crawl_errors.json").write_text(json.dumps(errors, ensure_ascii=False))


def write_discovery(tmp: Path, endpoint_map: dict[str, str]) -> None:
    """Write a discovery summary JSON file."""
    discovery = {
        "endpoint_map": endpoint_map,
        "discovered_kinds": sorted(endpoint_map),
        "generated_at": "2025-01-01T00:00:00+00:00",
    }
    (tmp / "discovery_summary.json").write_text(
        json.dumps(discovery, ensure_ascii=False),
    )


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    d = tmp_path / "kcb_data"
    d.mkdir()
    return d


# -- Dataset ingestion fixtures ---------------------------------------------


FIXTURE_TS = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.fixture
def vietmed_dir(tmp_path: Path) -> Path:
    """Create a tiny VietMed-Sum fixture directory with 3 Parquet files."""
    data_dir = tmp_path / "vietmed_raw"
    data_dir.mkdir()

    valid = pl.DataFrame(
        {
            "transcript": ["bệnh nhân sốt cao ba ngày", "ho khan kéo dài"],
            "summary": [
                "bệnh nhân bị sốt cao ba ngày liên tục",
                "ho kéo dài cần khám",
            ],
        },
    )
    valid.write_parquet(data_dir / "train_whole.parquet")

    dev = pl.DataFrame(
        {
            "transcript": ["đau đầu nhẹ", ""],
            "summary": ["khám đau đầu", "bỏ qua"],
        },
    )
    dev.write_parquet(data_dir / "dev_whole.parquet")

    test = pl.DataFrame(
        {
            "transcript": ["mệt mỏi toàn thân"],
            "summary": ["cần nghỉ ngơi"],
        },
    )
    test.write_parquet(data_dir / "test_whole.parquet")

    return data_dir


@pytest.fixture
def vihealthqa_dir(tmp_path: Path) -> Path:
    """Create a tiny ViHealthQA fixture directory with 3 CSV files."""
    data_dir = tmp_path / "vihealthqa_raw"
    data_dir.mkdir()

    train = pl.DataFrame(
        {
            "id": [1, 2],
            "question": [
                "Tiêm vaccine có an toàn không?",
                "Ăn kiêng thế nào?",
            ],
            "answer": [
                "Có, vaccine đã qua kiểm định.",
                "Nên ăn nhiều rau xanh.",
            ],
            "link": [
                "https://example.com/1",
                "https://example.com/2",
            ],
        },
    )
    train.write_csv(data_dir / "train.csv")

    val = pl.DataFrame(
        {
            "id": [1],
            "question": ["Uống thuốc khi đói được không?"],
            "answer": ["Nên ăn no trước khi uống."],
            "link": ["https://example.com/3"],
        },
    )
    val.write_csv(data_dir / "val.csv")

    test = pl.DataFrame(
        {
            "id": [1, 2],
            "question": ["Cơn đau ngực là gì?", "Câu hỏi lạc đề"],
            "answer": ["Đau vùng ngực cần cấp cứu.", ""],
            "link": [
                "https://example.com/4",
                "https://example.com/5",
            ],
        },
    )
    test.write_csv(data_dir / "test.csv")

    return data_dir
