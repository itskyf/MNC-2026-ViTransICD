"""Label description encoder using the SEA-LION backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mnc.models.sea_lion_encoder import SeaLionEncoder
from mnc.schemas.ontology import OntologyCode

_MAX_DESCRIPTOR_CHARS = 256


def _compact_descriptor(search_text: str, max_chars: int) -> str:
    """Collapse whitespace and truncate descriptor text.

    Args:
        search_text: Raw search text from ontology.
        max_chars: Maximum character length for the descriptor.

    Returns:
        Normalized and truncated descriptor string.
    """
    text = " ".join(search_text.split())
    return text[:max_chars]


def _build_label_text(code: OntologyCode, max_descriptor_chars: int) -> str:
    """Build a bilingual label prompt from OntologyCode schema fields.

    Args:
        code: Ontology code entry.
        max_descriptor_chars: Maximum chars for the descriptor section.

    Returns:
        Deterministic label text with code, titles, and optional descriptor.
    """
    sections: list[str] = [f"ICD-10 code: {code.code_3char}"]

    if code.title_vi:
        sections.append(f"Vietnamese title: {code.title_vi}")

    if code.title_en:
        sections.append(f"English title: {code.title_en}")

    if code.search_text:
        descriptor = _compact_descriptor(code.search_text, max_descriptor_chars)
        normalized_descriptor = descriptor.lower().strip()
        titles: set[str] = {code.title_vi.lower().strip()}
        if code.title_en:
            titles.add(code.title_en.lower().strip())
        if normalized_descriptor not in titles:
            sections.append(f"Descriptor: {descriptor}")

    return "\n".join(sections)


@dataclass
class OntologyLabelEncoderOutput:
    """Runtime output container for ontology label encoding.

    Attributes:
        label_codes: ICD-10 3-character codes in input order.
        label_texts: Constructed bilingual label prompts.
        input_ids: Tokenized input IDs. Shape ``[num_labels, seq_len]``.
        attention_mask: Attention mask. Shape ``[num_labels, seq_len]``.
        token_embeddings: Token-level embeddings.
            Shape ``[num_labels, seq_len, hidden_size]``.
        label_embeddings: Pooled label embeddings.
            Shape ``[num_labels, hidden_size]``.
    """

    label_codes: list[str]
    label_texts: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_embeddings: torch.Tensor
    label_embeddings: torch.Tensor


class OntologyLabelEncoder(nn.Module):
    """Encodes ICD ontology entries into dense label embeddings.

    Reuses the SEA-LION backbone from MA-1 to encode bilingual label
    descriptions built from OntologyCode schema fields.
    """

    def __init__(
        self,
        encoder: SeaLionEncoder | None = None,
        max_descriptor_chars: int = _MAX_DESCRIPTOR_CHARS,
    ) -> None:
        """Initialize label encoder.

        Args:
            encoder: Existing SEA-LION encoder to reuse. If ``None``,
                instantiates a new one with MA-1 defaults.
            max_descriptor_chars: Maximum character length for descriptor text.
        """
        super().__init__()
        self.encoder = encoder if encoder is not None else SeaLionEncoder()
        self.max_descriptor_chars = max_descriptor_chars

    def encode_labels(
        self,
        labels: list[OntologyCode],
        max_length: int | None = None,
    ) -> OntologyLabelEncoderOutput:
        """Encode a batch of ontology labels into dense embeddings.

        Args:
            labels: List of ``OntologyCode`` instances to encode.
            max_length: Maximum token length. Uses model default if ``None``.

        Returns:
            Label encoder output preserving input order.

        Raises:
            ValueError: If input list is empty, contains duplicate codes,
                or any label produces empty text.
            TypeError: If any item is not an ``OntologyCode``.
        """
        if not labels:
            msg = "Input labels list must not be empty."
            raise ValueError(msg)

        for i, label in enumerate(labels):
            if not isinstance(label, OntologyCode):
                msg = (
                    f"Item at index {i} is"
                    f" {type(label).__name__}, expected OntologyCode."
                )
                raise TypeError(msg)

        seen_codes: set[str] = set()
        for label in labels:
            if label.code_3char in seen_codes:
                msg = f"Duplicate code_3char value: {label.code_3char!r}."
                raise ValueError(msg)
            seen_codes.add(label.code_3char)

        label_codes = [label.code_3char for label in labels]
        label_texts = [
            _build_label_text(label, self.max_descriptor_chars) for label in labels
        ]

        for i, text in enumerate(label_texts):
            if not text.strip():
                msg = (
                    f"Constructed label text is empty for code"
                    f" {labels[i].code_3char!r}."
                )
                raise ValueError(msg)

        tokenized = self.encoder.tokenizer(
            label_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.encoder.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        output = self.encoder.forward(tokenized)

        return OntologyLabelEncoderOutput(
            label_codes=label_codes,
            label_texts=label_texts,
            input_ids=output.input_ids,
            attention_mask=output.attention_mask,
            token_embeddings=output.token_embeddings,
            label_embeddings=output.pooled_embeddings,
        )
