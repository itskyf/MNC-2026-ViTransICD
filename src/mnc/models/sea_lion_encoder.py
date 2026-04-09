"""SEA-LION ModernBERT encoder backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from mnc.schemas.document import DocumentRecord

_MODEL_ID = "aisingapore/SEA-LION-ModernBERT-300M"


def _extract_text(doc: DocumentRecord) -> str:
    """Extract text from a DocumentRecord using the priority fallback chain.

    Priority: ``normalized_text`` > ``retrieval_text`` > ``raw_text``.
    """
    if doc.normalized_text:
        return doc.normalized_text
    if doc.retrieval_text:
        return doc.retrieval_text
    if doc.raw_text:
        return doc.raw_text
    msg = f"All text fields are empty for document {doc.doc_id!r}"
    raise ValueError(msg)


def _masked_mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked mean pooling over token embeddings.

    Args:
        token_embeddings: Shape ``[batch_size, seq_len, hidden_size]``.
        attention_mask: Shape ``[batch_size, seq_len]``.

    Returns:
        Pooled embeddings of shape ``[batch_size, hidden_size]``.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class SeaLionEncoderOutput:
    """Runtime output container for the SEA-LION encoder backbone."""

    doc_ids: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_embeddings: torch.Tensor
    pooled_embeddings: torch.Tensor


class SeaLionEncoder(nn.Module):
    """Encoder backbone wrapping SEA-LION ModernBERT for document encoding.

    Loads the tokenizer and base model from Hugging Face and exposes
    token-level embeddings plus a pooled document representation.
    """

    def __init__(
        self,
        model_name: str = _MODEL_ID,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        pooling: str = "mean",
    ) -> None:
        """Initialize encoder with tokenizer and backbone model."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.pooling = pooling

    def forward(self, tokenized_batch: dict[str, torch.Tensor]) -> SeaLionEncoderOutput:
        """Run the backbone on a pre-tokenized batch.

        Args:
            tokenized_batch: Dict with ``input_ids`` and ``attention_mask`` tensors.

        Returns:
            Encoder output with token and pooled embeddings. ``doc_ids`` is empty
            when called directly; use ``encode_documents`` for the full pipeline.
        """
        outputs = self.model(**tokenized_batch, return_dict=True)
        token_embeddings = outputs.last_hidden_state
        attention_mask = tokenized_batch["attention_mask"]

        if self.pooling == "mean":
            pooled = _masked_mean_pooling(token_embeddings, attention_mask)
        elif self.pooling == "cls":
            pooled = token_embeddings[:, 0, :]
        else:
            msg = f"Unsupported pooling strategy: {self.pooling!r}"
            raise ValueError(msg)

        return SeaLionEncoderOutput(
            doc_ids=[],
            input_ids=tokenized_batch["input_ids"],
            attention_mask=attention_mask,
            token_embeddings=token_embeddings,
            pooled_embeddings=pooled,
        )

    def encode_documents(
        self,
        documents: list[DocumentRecord],
        max_length: int | None = None,
    ) -> SeaLionEncoderOutput:
        """Encode a batch of documents into embeddings.

        Args:
            documents: List of ``DocumentRecord`` instances to encode.
            max_length: Maximum token length. Uses model default if ``None``.

        Returns:
            Encoder output preserving ``doc_id`` order.

        Raises:
            ValueError: If the input list is empty or any document has no text.
            TypeError: If any item is not a ``DocumentRecord``.
        """
        if not documents:
            msg = "Input document list must not be empty."
            raise ValueError(msg)

        for i, doc in enumerate(documents):
            if not isinstance(doc, DocumentRecord):
                msg = (
                    f"Item at index {i} is"
                    f" {type(doc).__name__}, expected DocumentRecord."
                )
                raise TypeError(msg)

        texts = [_extract_text(doc) for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        output = self.forward(tokenized)
        output.doc_ids = doc_ids
        return output
