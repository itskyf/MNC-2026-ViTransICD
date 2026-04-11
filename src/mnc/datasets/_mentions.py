"""Rule-based mention extraction for Vietnamese medical text (DC-1).

Extracts disease, symptom, diagnosis, procedure, and abbreviation
mention candidates from ``DocumentRecord.raw_text`` with character
offsets.  Deterministic and side-effect-free.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnc.datasets._text import normalize_mention_text
from mnc.schemas.mention import MentionRecord, MentionType

if TYPE_CHECKING:
    from datetime import datetime

    from mnc.schemas.document import DocumentRecord

# ---------------------------------------------------------------------------
# Clinical cue patterns (compiled once)
# ---------------------------------------------------------------------------

# Vietnamese trailing function words to trim from captured spans
_TRIM_TRAILING = re.compile(
    r"\s+(?:là|của|và|nhưng|được|có|không|hoặc|với|cho|từ|bị"
    r"|ở|đã|sẽ|này|khi|rất|một|những|các|đó|ra|lại|cũng|vẫn"
    r"|nên|để|về|nhiều|ít|nữa|thì|mà|nếu|bởi|do|vì|ngoài"
    r"|trong|trên|dưới|sau|trước|giữa|tại|theo|hơn|đến|đang"
    r"|vừa|chưa|nó|nhé|ạ|đấy|đó|thế|kìa|vậy)$",
)

_RE_TRAILING_SINGLE = re.compile(r"\s+[a-zA-ZÀ-ỹğğı]$")

_RE_DUP_WORDS = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

# --- Disease cues ---
_RE_DISEASE = re.compile(
    r"(?i)\b(bệnh|viêm|ung thư|đột quỵ|suy|xoắn|đái"
    r"|hội chứng|nhiễm|bạch cầu|đái tháo đường|sa sút)\b"
    r"[\w\sÀ-ỹ]{1,25}",
)

# --- Symptom cues ---
_SYMPTOM_HEADS = (
    "đau|mệt|sốt|ho|khó thở|chóng mặt|buồn nôn|nôn|mụn|ngứa"
    "|tiêu chảy|táo bón|đổ mồ hôi|mất ngủ|rụng tóc|sưng|chảy máu"
    "|ra máu|khó chịu|yếu|liệt|co giật|run|mất ổn định"
    "|đông cứng|co cứng|giật"
)
_RE_SYMPTOM = re.compile(
    rf"(?i)\b(?:{_SYMPTOM_HEADS})\b[\w\sÀ-ỹ]{{0,20}}",
)

# --- Diagnosis cues ---
_RE_DIAGNOSIS = re.compile(
    r"(?i)\b(chẩn đoán|xét nghiệm|chỉ số|kết quả"
    r"|siêu âm|x-quang|chụp|sinh thiết)\b"
    r"[\w\sÀ-ỹ]{0,20}",
)

# --- Procedure cues ---
_RE_PROCEDURE = re.compile(
    r"(?i)\b(điều trị|phẫu thuật|mổ|truyền|ghép|tiêm"
    r"|thở oxy|xạ trị|hóa trị|vật lý trị liệu)\b"
    r"[\w\sÀ-ỹ]{0,20}",
)

# --- Abbreviation cues ---
# 2-6 uppercase ASCII letters (exclude common Vietnamese words in caps)
_ABBR_STOPLIST = frozenset(
    {
        "TỔNG",
        "CÁC",
        "VÀ",
        "NẾU",
        "NHƯNG",
        "ĐƯỢC",
        "KHÔNG",
        "NÀY",
        "THÌ",
        "MÀ",
        "CÓ",
        "LÀ",
        "ĐÃ",
        "SẼ",
        "ĐANG",
        "VỚI",
        "CHO",
        "CỦA",
        "TỪ",
        "KHI",
        "NHIỀU",
        "NHỮNG",
        "VÀO",
        "RA",
        "LÊN",
        "XUỐNG",
    },
)

_RE_ABBR = re.compile(r"\b([A-Z]{2,6})\b")

# Pattern-to-type mapping
_PATTERNS: list[tuple[re.Pattern[str], MentionType]] = [
    (_RE_DISEASE, "disease"),
    (_RE_SYMPTOM, "symptom"),
    (_RE_DIAGNOSIS, "diagnosis"),
    (_RE_PROCEDURE, "procedure"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _match_cue_spans(raw: str) -> list[tuple[int, int, str, str]]:
    """Run all cue-pattern regexes on *raw* text and return spans."""
    spans: list[tuple[int, int, str, str]] = []
    for pattern, mtype in _PATTERNS:
        for m in pattern.finditer(raw):
            text = _trim_span(raw[m.start() : m.end()])
            if not text:
                continue
            end = m.start() + len(text)
            spans.append((m.start(), end, text, mtype))
    return spans


def _match_abbr_spans(raw: str) -> list[tuple[int, int, str, str]]:
    """Detect uppercase abbreviation spans in *raw* text."""
    spans: list[tuple[int, int, str, str]] = []
    for m in _RE_ABBR.finditer(raw):
        abbr = m.group(1)
        if abbr in _ABBR_STOPLIST:
            continue
        spans.append((m.start(), m.end(), m.group(0), "abbreviation"))
    return spans


def _dedup_spans(
    spans: list[tuple[int, int, str, str]],
) -> list[tuple[int, int, str, str]]:
    """Same-start: keep longest; identical spans: merge."""
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    deduped: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for start, end, text, mtype in spans:
        if start == end:
            continue
        if deduped and start == deduped[-1][0]:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start, end, text, mtype))
    return deduped


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def extract_mentions(
    doc: DocumentRecord,
    created_at: datetime,
) -> list[MentionRecord]:
    """Extract mention candidates with raw-text offsets.

    Uses rule-based clinical cue patterns.  Returns deduplicated,
    non-overlapping spans sorted by ``char_start``.
    """
    raw = doc.raw_text
    if not raw:
        return []

    spans = _match_cue_spans(raw) + _match_abbr_spans(raw)
    deduped = _dedup_spans(spans)

    return [
        MentionRecord(
            mention_id=f"{doc.doc_id}:m:{start}:{end}",
            doc_id=doc.doc_id,
            text=text,
            normalized_text=normalize_mention_text(text),
            mention_type=mtype,
            char_start=start,
            char_end=end,
            confidence=None,
            created_at=created_at,
        )
        for start, end, text, mtype in deduped
    ]


def _trim_span(text: str) -> str:
    """Trim trailing function words, single-char noise, and duplicate words."""
    out = _RE_DUP_WORDS.sub(r"\1", text.rstrip())
    for _ in range(3):
        new = _TRIM_TRAILING.sub("", out).rstrip()
        new = _RE_TRAILING_SINGLE.sub("", new).rstrip()
        if new == out:
            break
        out = new
    return out
