"""TransICD classification head for multi-label ICD coding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from mnc.models.sea_lion_encoder import SeaLionEncoderOutput

_MASK_FILL_VALUE = float("-inf")
_TOKENS_NDIM = 3
_MASK_NDIM = 2


@dataclass
class TransICDHeadOutput:
    """Output container for TransICDHead.

    Attributes:
        doc_ids: Document identifiers passed through from encoder output.
        label_codes: ICD code strings corresponding to each label dimension.
        logits: Raw prediction scores. Shape ``[batch_size, num_labels]``.
        attention_weights: Per-label attention over tokens. Shape
            ``[batch_size, num_labels, seq_len]``.
        code_representations: Code-specific document representations. Shape
            ``[batch_size, num_labels, hidden_size]``.
    """

    doc_ids: list[str]
    label_codes: list[str]
    logits: torch.Tensor
    attention_weights: torch.Tensor
    code_representations: torch.Tensor


class TransICDHead(nn.Module):
    """TransICD-style classification head with code-wise attention.

    Consumes token-level encoder outputs and produces per-label logits,
    attention maps, and code-specific document representations.

    Supports two label modes:

    * **Internal queries**: trainable ``nn.Embedding`` table, used when
      ``label_embeddings`` is not supplied to :meth:`forward`.
    * **External embeddings**: caller-supplied tensor, projected to
      ``hidden_size`` when ``label_embedding_dim`` differs.
    """

    def __init__(
        self,
        label_codes: list[str],
        hidden_size: int,
        label_embedding_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize TransICDHead.

        Args:
            label_codes: ICD code strings defining the label vocabulary.
            hidden_size: Encoder hidden dimension (from MA-1).
            label_embedding_dim: Dimension of external label embeddings.
                When provided and different from ``hidden_size``, a
                projection layer maps to ``hidden_size``.  When ``None``,
                internal trainable label queries are used.
            dropout: Dropout probability applied after attention aggregation.

        Raises:
            ValueError: If *label_codes* is empty, contains duplicates,
                or if *hidden_size* is not positive.
        """
        super().__init__()

        if not label_codes:
            msg = "label_codes must not be empty."
            raise ValueError(msg)

        if len(set(label_codes)) != len(label_codes):
            msg = "label_codes must not contain duplicates."
            raise ValueError(msg)

        if hidden_size <= 0:
            msg = "hidden_size must be positive."
            raise ValueError(msg)

        self.label_codes = list(label_codes)
        self.hidden_size = hidden_size
        self.num_labels = len(label_codes)

        # Internal label queries for attention computation
        self.label_queries = nn.Embedding(self.num_labels, hidden_size)

        # Optional projection for external label embeddings
        self.label_projection: nn.Linear | None = None
        if label_embedding_dim is not None and label_embedding_dim != hidden_size:
            self.label_projection = nn.Linear(
                label_embedding_dim,
                hidden_size,
                bias=False,
            )

        self.dropout = nn.Dropout(dropout)

        # Per-label classifier parameters
        self.classifier_weight = nn.Parameter(
            torch.empty(self.num_labels, hidden_size),
        )
        self.classifier_bias = nn.Parameter(torch.zeros(self.num_labels))
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize classifier weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.classifier_weight)

    def _resolve_queries(
        self,
        label_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resolve label queries to shape ``[num_labels, hidden_size]``.

        Args:
            label_embeddings: External embeddings ``[num_labels, dim]``
                or ``None`` to use internal queries.

        Returns:
            Label query tensor on the model's device.

        Raises:
            ValueError: If embedding count or dimension is incompatible.
        """
        if label_embeddings is None:
            indices = torch.arange(
                self.num_labels,
                device=self.label_queries.weight.device,
            )
            return self.label_queries(indices)

        if label_embeddings.shape[0] != self.num_labels:
            msg = (
                f"Expected {self.num_labels} label embeddings, "
                f"got {label_embeddings.shape[0]}."
            )
            raise ValueError(msg)

        if self.label_projection is not None:
            return self.label_projection(label_embeddings)

        if label_embeddings.shape[1] != self.hidden_size:
            msg = (
                f"Label embedding dim {label_embeddings.shape[1]} "
                f"!= hidden_size {self.hidden_size} "
                f"and no projection configured."
            )
            raise ValueError(msg)

        return label_embeddings

    def forward(
        self,
        encoder_output: SeaLionEncoderOutput,
        label_embeddings: torch.Tensor | None = None,
    ) -> TransICDHeadOutput:
        """Run code-wise attention and per-label classification.

        Args:
            encoder_output: Encoder output from MA-1.
            label_embeddings: Optional external label embeddings replacing
                internal queries for attention computation.

        Returns:
            Classification output with logits, attention weights, and
            code-specific representations.

        Raises:
            ValueError: If tensor shapes are invalid or dimensions mismatch.
        """
        tokens = encoder_output.token_embeddings
        mask = encoder_output.attention_mask

        _validate_encoder_tensors(tokens, mask)

        # Resolve label queries: [num_labels, hidden_size]
        queries = self._resolve_queries(label_embeddings)

        # Attention scores: [batch_size, num_labels, seq_len]
        scores = torch.einsum("lh,bsh->bls", queries, tokens)

        # Mask padded positions
        mask_3d = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        scores = scores.masked_fill(mask_3d == 0, _MASK_FILL_VALUE)

        # Softmax over sequence dimension
        attn_weights = torch.softmax(scores, dim=2)

        # Safety: zero masked positions for exact zeros
        attn_weights = attn_weights.masked_fill(mask_3d == 0, 0.0)

        # Weighted sum: [batch_size, num_labels, hidden_size]
        code_reps = torch.einsum("bls,bsh->blh", attn_weights, tokens)
        code_reps = self.dropout(code_reps)

        # Per-label scoring: [batch_size, num_labels]
        logits = torch.einsum("blh,lh->bl", code_reps, self.classifier_weight)
        logits = logits + self.classifier_bias

        return TransICDHeadOutput(
            doc_ids=encoder_output.doc_ids,
            label_codes=self.label_codes,
            logits=logits,
            attention_weights=attn_weights,
            code_representations=code_reps,
        )


def _validate_encoder_tensors(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate shape and dimensionality of encoder output tensors.

    Args:
        tokens: Token embeddings tensor.
        mask: Attention mask tensor.

    Raises:
        ValueError: If ndim, batch size, or seq_len mismatch.
    """
    if tokens.ndim != _TOKENS_NDIM:
        msg = f"token_embeddings must be {_TOKENS_NDIM}D, got {tokens.ndim}D."
        raise ValueError(msg)

    if mask.ndim != _MASK_NDIM:
        msg = f"attention_mask must be {_MASK_NDIM}D, got {mask.ndim}D."
        raise ValueError(msg)

    if tokens.shape[0] != mask.shape[0]:
        msg = (
            f"Batch size mismatch: token_embeddings has "
            f"{tokens.shape[0]}, attention_mask has {mask.shape[0]}."
        )
        raise ValueError(msg)

    if tokens.shape[1] != mask.shape[1]:
        msg = (
            f"Sequence length mismatch: token_embeddings has "
            f"{tokens.shape[1]}, attention_mask has {mask.shape[1]}."
        )
        raise ValueError(msg)

    if (mask.sum(dim=1) == 0).any():
        msg = "One or more sequences have zero valid tokens."
        raise ValueError(msg)
