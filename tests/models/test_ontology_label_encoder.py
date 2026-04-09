"""Unit tests for OntologyLabelEncoder."""

import typing
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
import torch

from mnc.models.ontology_label_encoder import (
    OntologyLabelEncoder,
    OntologyLabelEncoderOutput,
    _build_label_text,
)
from mnc.models.sea_lion_encoder import SeaLionEncoderOutput
from mnc.schemas.ontology import OntologyCode

_HIDDEN_SIZE = 32
_SEQ_LEN = 16


def _make_code(
    code_3char: str = "A00",
    title_vi: str = "Tả",
    title_en: str | None = "Cholera",
    search_text: str = "",
    aliases: list[str] | None = None,
) -> OntologyCode:
    """Build a minimal OntologyCode for testing."""
    return OntologyCode(
        code_3char=code_3char,
        title_vi=title_vi,
        title_en=title_en,
        aliases=aliases or [],
        search_text=search_text,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_mock_encoder(
    hidden_size: int = _HIDDEN_SIZE,
    seq_len: int = _SEQ_LEN,
) -> MagicMock:
    """Create a mock SeaLionEncoder that returns synthetic tensors."""
    mock = MagicMock()

    def mock_tokenize(
        texts: list[str],
        **_kwargs: str | float | bool | None,
    ) -> dict[str, torch.Tensor]:
        n = len(texts)
        return {
            "input_ids": torch.randint(1, 1000, (n, seq_len)),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }

    mock.tokenizer.side_effect = mock_tokenize
    mock.model.parameters.return_value = iter([torch.randn(1)])

    def mock_forward(
        tokenized_batch: dict[str, torch.Tensor],
    ) -> SeaLionEncoderOutput:
        n = tokenized_batch["input_ids"].shape[0]
        s = tokenized_batch["input_ids"].shape[1]
        return SeaLionEncoderOutput(
            doc_ids=[],
            input_ids=tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["attention_mask"],
            token_embeddings=torch.randn(n, s, hidden_size),
            pooled_embeddings=torch.randn(n, hidden_size),
        )

    mock.forward.side_effect = mock_forward
    return mock


@pytest.fixture
def label_encoder() -> OntologyLabelEncoder:
    """OntologyLabelEncoder with a mocked backbone."""
    return OntologyLabelEncoder(encoder=_make_mock_encoder())


# 1. encode_labels returns all required fields
def test_encode_returns_all_fields(label_encoder: OntologyLabelEncoder) -> None:
    result = label_encoder.encode_labels([_make_code()])

    assert isinstance(result, OntologyLabelEncoderOutput)
    assert isinstance(result.label_codes, list)
    assert isinstance(result.label_texts, list)
    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.attention_mask, torch.Tensor)
    assert isinstance(result.token_embeddings, torch.Tensor)
    assert isinstance(result.label_embeddings, torch.Tensor)


# 2. Output batch size equals number of input labels
def test_batch_size_matches_input(label_encoder: OntologyLabelEncoder) -> None:
    labels = [_make_code(code_3char=f"A{i:02d}") for i in range(3)]
    result = label_encoder.encode_labels(labels)

    assert result.label_embeddings.shape[0] == 3
    assert result.input_ids.shape[0] == 3
    assert result.attention_mask.shape[0] == 3
    assert result.token_embeddings.shape[0] == 3


# 3. label_codes preserve input order
def test_label_codes_preserve_order(label_encoder: OntologyLabelEncoder) -> None:
    codes = ["C34", "A00", "B20"]
    labels = [_make_code(code_3char=c) for c in codes]
    result = label_encoder.encode_labels(labels)

    assert result.label_codes == codes


# 4. label_embeddings.ndim == 2
def test_label_embeddings_ndim(label_encoder: OntologyLabelEncoder) -> None:
    result = label_encoder.encode_labels([_make_code()])

    assert result.label_embeddings.ndim == 2


# 5. token_embeddings.ndim == 3
def test_token_embeddings_ndim(label_encoder: OntologyLabelEncoder) -> None:
    result = label_encoder.encode_labels([_make_code()])

    assert result.token_embeddings.ndim == 3


# 6. Prompt formatting includes code and bilingual titles when present
def test_prompt_includes_code_and_bilingual_titles() -> None:
    code = _make_code(code_3char="A00", title_vi="Tả", title_en="Cholera")
    text = _build_label_text(code, max_descriptor_chars=256)

    assert "ICD-10 code: A00" in text
    assert "Vietnamese title: Tả" in text
    assert "English title: Cholera" in text


# 7. Descriptor text is included from search_text when present
def test_descriptor_included_from_search_text() -> None:
    code = _make_code(search_text="Bệnh nhiễm trùng ruột do vi khuẩn")
    text = _build_label_text(code, max_descriptor_chars=256)

    assert "Descriptor: Bệnh nhiễm trùng ruột do vi khuẩn" in text


# 8. Empty input raises ValueError
def test_empty_input_raises(label_encoder: OntologyLabelEncoder) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        label_encoder.encode_labels([])


# 9. Invalid item type raises TypeError
def test_invalid_type_raises(label_encoder: OntologyLabelEncoder) -> None:
    bad = typing.cast("list[OntologyCode]", ["not an ontology code"])
    with pytest.raises(TypeError, match="expected OntologyCode"):
        label_encoder.encode_labels(bad)


# 10. Duplicate codes raise ValueError
def test_duplicate_codes_raises(label_encoder: OntologyLabelEncoder) -> None:
    labels = [_make_code(code_3char="A00"), _make_code(code_3char="A00")]
    with pytest.raises(ValueError, match="Duplicate"):
        label_encoder.encode_labels(labels)


# --- Additional prompt-construction edge cases ---


def test_prompt_skips_empty_title_en() -> None:
    code = _make_code(title_vi="Tả", title_en=None)
    text = _build_label_text(code, max_descriptor_chars=256)

    assert "Vietnamese title: Tả" in text
    assert "English title" not in text


def test_descriptor_deduplicated_against_title() -> None:
    code = _make_code(
        title_vi="Tả",
        search_text="Tả",
    )
    text = _build_label_text(code, max_descriptor_chars=256)

    assert "Descriptor" not in text


def test_descriptor_truncated_to_max_chars() -> None:
    long_text = "Bệnh nhiễm trùng " * 50
    code = _make_code(search_text=long_text)
    text = _build_label_text(code, max_descriptor_chars=32)

    descriptor_line = next(
        line for line in text.split("\n") if line.startswith("Descriptor:")
    )
    descriptor_value = descriptor_line[len("Descriptor: ") :]
    assert len(descriptor_value) <= 32


def test_prompt_deterministic() -> None:
    code = _make_code(
        title_vi="Tả",
        title_en="Cholera",
        search_text="Bệnh tả",
    )
    first = _build_label_text(code, max_descriptor_chars=256)
    second = _build_label_text(code, max_descriptor_chars=256)

    assert first == second
