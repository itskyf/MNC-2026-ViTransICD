"""Pipeline data schemas for ViTransICD.

See ``docs/data/v1/de-01-schemas.md`` for the canonical design document.
"""

from mnc.schemas.alias import AliasRecord
from mnc.schemas.candidate import CandidateLink
from mnc.schemas.document import DocumentRecord, JsonValue
from mnc.schemas.explanation import EvidenceSpan, ExplanationRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord, MentionType
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.prediction import PredictionRecord
from mnc.schemas.rule import RuleRecord
from mnc.schemas.sentence import SentenceSpanRecord
from mnc.schemas.silver import SilverRecord
from mnc.schemas.snapshot import SnapshotRecord
from mnc.schemas.weak_label import WeakEvidenceSpan, WeakLabelRecord

__all__ = [
    "AliasRecord",
    "BronzeManifest",
    "CandidateLink",
    "DocumentRecord",
    "EvidenceSpan",
    "ExplanationRecord",
    "JsonValue",
    "MentionRecord",
    "MentionType",
    "OntologyCode",
    "PredictionRecord",
    "RuleRecord",
    "SentenceSpanRecord",
    "SilverRecord",
    "SnapshotRecord",
    "WeakEvidenceSpan",
    "WeakLabelRecord",
]
