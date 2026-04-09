"""Tests for the SEA-LION ModernBERT encoder backbone."""

import typing
from datetime import UTC, datetime

import pytest
import torch

from mnc.models.sea_lion_encoder import (
    SeaLionEncoder,
    SeaLionEncoderOutput,
    _extract_text,
)
from mnc.schemas.document import DocumentRecord


def _make_doc(
    doc_id: str = "doc-001",
    raw_text: str = "Bệnh nhân sốt cao ba ngày liên tục.",
    normalized_text: str = "",
    retrieval_text: str = "",
) -> DocumentRecord:
    """Build a minimal DocumentRecord for testing."""
    return DocumentRecord(
        doc_id=doc_id,
        source="test",
        language="vi",
        raw_text=raw_text,
        normalized_text=normalized_text,
        retrieval_text=retrieval_text,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture(scope="session")
def encoder() -> SeaLionEncoder:
    """Session-scoped encoder to avoid reloading the 300M model per test."""
    return SeaLionEncoder()


# 1. Model/tokenizer load smoke test
def test_encoder_loads(encoder: SeaLionEncoder) -> None:
    assert encoder.tokenizer is not None
    assert encoder.model is not None


# 2. encode_documents returns all required fields
def test_encode_returns_all_fields(encoder: SeaLionEncoder) -> None:
    output = encoder.encode_documents([_make_doc()])

    assert isinstance(output, SeaLionEncoderOutput)
    assert isinstance(output.doc_ids, list)
    assert isinstance(output.input_ids, torch.Tensor)
    assert isinstance(output.attention_mask, torch.Tensor)
    assert isinstance(output.token_embeddings, torch.Tensor)
    assert isinstance(output.pooled_embeddings, torch.Tensor)


# 3. Batch size equals number of input documents
def test_batch_size_matches(encoder: SeaLionEncoder) -> None:
    docs = [_make_doc(doc_id=f"doc-{i:03d}") for i in range(3)]
    output = encoder.encode_documents(docs)

    assert output.input_ids.shape[0] == 3
    assert output.attention_mask.shape[0] == 3
    assert output.token_embeddings.shape[0] == 3
    assert output.pooled_embeddings.shape[0] == 3


# 4. doc_ids preserve input order
def test_doc_ids_preserve_order(encoder: SeaLionEncoder) -> None:
    ids = ["alpha", "beta", "gamma"]
    docs = [_make_doc(doc_id=did) for did in ids]
    output = encoder.encode_documents(docs)

    assert output.doc_ids == ids


# 5. token_embeddings.ndim == 3
def test_token_embeddings_ndim(encoder: SeaLionEncoder) -> None:
    output = encoder.encode_documents([_make_doc()])
    assert output.token_embeddings.ndim == 3


# 6. pooled_embeddings.ndim == 2
def test_pooled_embeddings_ndim(encoder: SeaLionEncoder) -> None:
    output = encoder.encode_documents([_make_doc()])
    assert output.pooled_embeddings.ndim == 2


# 7. Pooled embedding batch dimension matches input size
def test_pooled_batch_matches_input(encoder: SeaLionEncoder) -> None:
    docs = [_make_doc(doc_id=f"doc-{i}") for i in range(4)]
    output = encoder.encode_documents(docs)
    assert output.pooled_embeddings.shape[0] == 4


# 8. Fallback text selection works as specified
def test_text_fallback() -> None:
    # normalized_text takes priority
    doc = _make_doc(
        normalized_text="normalized",
        retrieval_text="retrieval",
        raw_text="raw",
    )
    assert _extract_text(doc) == "normalized"

    # Falls back to retrieval_text
    doc = _make_doc(
        normalized_text="",
        retrieval_text="retrieval",
        raw_text="raw",
    )
    assert _extract_text(doc) == "retrieval"

    # Falls back to raw_text
    doc = _make_doc(
        normalized_text="",
        retrieval_text="",
        raw_text="raw",
    )
    assert _extract_text(doc) == "raw"


def test_text_fallback_all_empty_raises() -> None:
    doc = _make_doc(normalized_text="", retrieval_text="", raw_text="")
    with pytest.raises(ValueError, match="All text fields are empty"):
        _extract_text(doc)


# 9. Empty input raises ValueError
def test_empty_input_raises(encoder: SeaLionEncoder) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        encoder.encode_documents([])


# 10. Invalid item type raises TypeError
def test_invalid_type_raises(encoder: SeaLionEncoder) -> None:
    bad = typing.cast("list[DocumentRecord]", ["not a document"])
    with pytest.raises(TypeError, match="expected DocumentRecord"):
        encoder.encode_documents(bad)
