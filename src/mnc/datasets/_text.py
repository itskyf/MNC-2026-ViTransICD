"""Pure text normalization helpers for DC-1 and mention extraction.

All functions are deterministic and side-effect-free.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for performance)
# ---------------------------------------------------------------------------

_RE_CRLF = re.compile(r"\r\n?")
_RE_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\ufeff\u00ad\u200e\u200f\u202a-\u202e]",
)
_RE_WHITESPACE_COLLAPSE = re.compile(r"[^\S\n]+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_RE_SPACE_AFTER_PUNCT = re.compile(r"([.,;:!?])([^\s\d])")
_RE_RETRIEVAL_PUNCT = re.compile(r"[^\w\s]")
_RE_RETRIEVAL_WS = re.compile(r"\s+")

# Patterns that should NOT get a space inserted after punctuation
# e.g. "BS.", "TS.", "3.5", "pH."
_PROTECTED_PUNCT_PREFIX = re.compile(r"(?:\d+[.,]\d+|\b[A-Za-z]{1,4}[.])")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalize_document_text(text: str) -> str:
    r"""Return normalized text for NLP.

    Applies in order:
    1. Unicode NFKC normalization
    2. Line ending normalization to ``\n``
    3. Zero-width / invisible character removal
    4. Whitespace collapse (preserve newlines)
    5. Punctuation spacing normalization
    6. Trim

    Preserves Vietnamese diacritics, digits, and medically meaningful
    punctuation inside abbreviations or lab values.
    """
    if not text:
        return ""

    # 1. NFKC
    out = unicodedata.normalize("NFKC", text)

    # 2. Line endings
    out = _RE_CRLF.sub("\n", out)

    # 3. Zero-width / invisible
    out = _RE_ZERO_WIDTH.sub("", out)

    # 4. Collapse whitespace (keep newlines)
    out = _RE_WHITESPACE_COLLAPSE.sub(" ", out)

    # 5a. Remove space before punctuation
    out = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", out)

    # 5b. Add space after punctuation (skip protected patterns)
    out = _RE_SPACE_AFTER_PUNCT.sub(
        lambda m: (
            m.group(0)
            if _PROTECTED_PUNCT_PREFIX.search(m.group(0))
            else f"{m.group(1)} {m.group(2)}"
        ),
        out,
    )

    # 6. Trim
    return out.strip()


def build_retrieval_text(normalized_text: str) -> str:
    """Return retrieval-oriented text from *normalized_text*.

    Lowercases, replaces most punctuation with spaces, and collapses
    whitespace.  Preserves Vietnamese diacritics.  Does not stem or
    remove stopwords.
    """
    if not normalized_text:
        return ""

    out = normalized_text.lower()
    out = _RE_RETRIEVAL_PUNCT.sub(" ", out)
    out = _RE_RETRIEVAL_WS.sub(" ", out)
    return out.strip()


def normalize_mention_text(surface: str) -> str:
    """Apply mention-level normalization to a surface form.

    Same rules as :func:`normalize_document_text` scoped to a short
    span.  Does NOT expand abbreviations.
    """
    if not surface:
        return ""
    out = unicodedata.normalize("NFKC", surface)
    out = _RE_ZERO_WIDTH.sub("", out)
    out = _RE_WHITESPACE_COLLAPSE.sub(" ", out)
    return out.strip()
