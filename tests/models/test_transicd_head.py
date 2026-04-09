"""Unit tests for TransICDHead."""

from __future__ import annotations

import pytest
import torch

from mnc.models.sea_lion_encoder import SeaLionEncoderOutput
from mnc.models.transicd_head import TransICDHead, TransICDHeadOutput

LABEL_CODES = ["A00", "B20", "C34"]
HIDDEN_SIZE = 32
BATCH_SIZE = 2
SEQ_LEN = 10


def _make_encoder_output(
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    hidden_size: int = HIDDEN_SIZE,
    doc_ids: list[str] | None = None,
    *,
    with_padding: bool = True,
) -> SeaLionEncoderOutput:
    """Create a synthetic SeaLionEncoderOutput for testing."""
    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(batch_size)]

    token_embeddings = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    if with_padding and batch_size > 1:
        # Second document is shorter — last 3 tokens are padding
        attention_mask[1, -3:] = 0

    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    pooled_embeddings = torch.randn(batch_size, hidden_size)

    return SeaLionEncoderOutput(
        doc_ids=doc_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_embeddings=token_embeddings,
        pooled_embeddings=pooled_embeddings,
    )


class TestInitialization:
    """Test 1: initialization smoke test with dummy labels."""

    def test_creates_module(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        assert isinstance(head, torch.nn.Module)

    def test_stores_label_codes(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        assert head.label_codes == LABEL_CODES

    def test_stores_hidden_size(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        assert head.hidden_size == HIDDEN_SIZE


class TestForwardOutput:
    """Tests 2-3: forward returns all fields with correct shapes."""

    def test_returns_transicd_head_output(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)
        assert isinstance(result, TransICDHeadOutput)

    def test_output_has_all_fields(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)
        assert hasattr(result, "doc_ids")
        assert hasattr(result, "label_codes")
        assert hasattr(result, "logits")
        assert hasattr(result, "attention_weights")
        assert hasattr(result, "code_representations")

    def test_logits_shape(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)
        assert result.logits.shape == (BATCH_SIZE, len(LABEL_CODES))

    def test_attention_weights_shape(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)
        assert result.attention_weights.shape == (
            BATCH_SIZE,
            len(LABEL_CODES),
            SEQ_LEN,
        )

    def test_code_representations_shape(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)
        assert result.code_representations.shape == (
            BATCH_SIZE,
            len(LABEL_CODES),
            HIDDEN_SIZE,
        )


class TestPreservedFields:
    """Tests 4-5: doc_ids and label_codes are preserved."""

    def test_doc_ids_preserved(self) -> None:
        doc_ids = ["doc_alpha", "doc_beta"]
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output(doc_ids=doc_ids)
        result = head(encoder_out)
        assert result.doc_ids == doc_ids

    def test_label_codes_preserved(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)
        assert result.label_codes == LABEL_CODES


class TestAttentionMasking:
    """Tests 6-7: masking correctness."""

    def test_masked_positions_near_zero(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)

        # Second document has padding in last 3 positions
        padded_attn = result.attention_weights[1, :, -3:]
        assert (padded_attn == 0.0).all()

    def test_attention_sums_to_one(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        result = head(encoder_out)

        # Sum over seq_len dimension should be ~1 for all labels
        attn_sums = result.attention_weights.sum(dim=2)
        assert torch.allclose(attn_sums, torch.ones_like(attn_sums))


class TestExternalLabelEmbeddings:
    """Test 8: external label embeddings path."""

    def test_external_embeddings_same_dim(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        embeddings = torch.randn(len(LABEL_CODES), HIDDEN_SIZE)
        result = head(encoder_out, label_embeddings=embeddings)
        assert result.logits.shape == (BATCH_SIZE, len(LABEL_CODES))

    def test_external_embeddings_with_projection(self) -> None:
        label_dim = 16
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
            label_embedding_dim=label_dim,
        )
        encoder_out = _make_encoder_output()
        embeddings = torch.randn(len(LABEL_CODES), label_dim)
        result = head(encoder_out, label_embeddings=embeddings)
        assert result.logits.shape == (BATCH_SIZE, len(LABEL_CODES))

    def test_external_embeddings_wrong_count_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        bad_embeddings = torch.randn(5, HIDDEN_SIZE)
        with pytest.raises(ValueError, match="Expected 3 label embeddings"):
            head(encoder_out, label_embeddings=bad_embeddings)

    def test_external_embeddings_wrong_dim_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        encoder_out = _make_encoder_output()
        bad_embeddings = torch.randn(len(LABEL_CODES), 64)
        with pytest.raises(ValueError, match="Label embedding dim"):
            head(encoder_out, label_embeddings=bad_embeddings)


class TestValidationErrors:
    """Tests 9-11: constructor and forward validation."""

    def test_duplicate_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            TransICDHead(
                label_codes=["A00", "A00", "B20"],
                hidden_size=HIDDEN_SIZE,
            )

    def test_empty_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            TransICDHead(label_codes=[], hidden_size=HIDDEN_SIZE)

    def test_zero_hidden_size_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TransICDHead(label_codes=LABEL_CODES, hidden_size=0)

    def test_negative_hidden_size_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TransICDHead(label_codes=LABEL_CODES, hidden_size=-1)

    def test_wrong_token_ndim_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        bad_output = SeaLionEncoderOutput(
            doc_ids=["doc_0"],
            input_ids=torch.randint(1, 100, (1, 5)),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            token_embeddings=torch.randn(1, 5),  # 2D instead of 3D
            pooled_embeddings=torch.randn(1, HIDDEN_SIZE),
        )
        with pytest.raises(ValueError, match="3D"):
            head(bad_output)

    def test_wrong_mask_ndim_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        bad_output = SeaLionEncoderOutput(
            doc_ids=["doc_0"],
            input_ids=torch.randint(1, 100, (1, 5)),
            attention_mask=torch.ones(1, dtype=torch.long),  # 1D
            token_embeddings=torch.randn(1, 5, HIDDEN_SIZE),
            pooled_embeddings=torch.randn(1, HIDDEN_SIZE),
        )
        with pytest.raises(ValueError, match="2D"):
            head(bad_output)

    def test_batch_size_mismatch_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        bad_output = SeaLionEncoderOutput(
            doc_ids=["doc_0"],
            input_ids=torch.randint(1, 100, (1, 5)),
            attention_mask=torch.ones(2, 5, dtype=torch.long),  # batch=2
            token_embeddings=torch.randn(1, 5, HIDDEN_SIZE),  # batch=1
            pooled_embeddings=torch.randn(1, HIDDEN_SIZE),
        )
        with pytest.raises(ValueError, match="Batch size mismatch"):
            head(bad_output)

    def test_seq_len_mismatch_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        bad_output = SeaLionEncoderOutput(
            doc_ids=["doc_0"],
            input_ids=torch.randint(1, 100, (1, 5)),
            attention_mask=torch.ones(1, 10, dtype=torch.long),  # seq=10
            token_embeddings=torch.randn(1, 5, HIDDEN_SIZE),  # seq=5
            pooled_embeddings=torch.randn(1, HIDDEN_SIZE),
        )
        with pytest.raises(ValueError, match="Sequence length mismatch"):
            head(bad_output)

    def test_zero_valid_tokens_raises(self) -> None:
        head = TransICDHead(
            label_codes=LABEL_CODES,
            hidden_size=HIDDEN_SIZE,
        )
        bad_output = SeaLionEncoderOutput(
            doc_ids=["doc_0"],
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            attention_mask=torch.zeros(1, 5, dtype=torch.long),  # all masked
            token_embeddings=torch.randn(1, 5, HIDDEN_SIZE),
            pooled_embeddings=torch.randn(1, HIDDEN_SIZE),
        )
        with pytest.raises(ValueError, match="zero valid tokens"):
            head(bad_output)


class TestNoModelDownload:
    """Test 12: no real model download required."""

    def test_runs_with_synthetic_tensors(self) -> None:
        head = TransICDHead(
            label_codes=["X", "Y"],
            hidden_size=8,
        )
        encoder_out = SeaLionEncoderOutput(
            doc_ids=["synth"],
            input_ids=torch.randint(1, 100, (1, 4)),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            token_embeddings=torch.randn(1, 4, 8),
            pooled_embeddings=torch.randn(1, 8),
        )
        result = head(encoder_out)
        assert result.logits.shape == (1, 2)
