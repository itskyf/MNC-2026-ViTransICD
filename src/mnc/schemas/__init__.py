"""Pipeline data schemas for ViTransICD.

See ``docs/data/v1/de-01-schemas.md`` for the canonical design document.
"""

from mnc.schemas.candidate import CandidateLink
from mnc.schemas.document import DocumentRecord, JsonValue
from mnc.schemas.explanation import EvidenceSpan, ExplanationRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.prediction import PredictionRecord
from mnc.schemas.silver import SilverRecord
from mnc.schemas.snapshot import SnapshotRecord

__all__ = [
    "BronzeManifest",
    "CandidateLink",
    "DocumentRecord",
    "EvidenceSpan",
    "ExplanationRecord",
    "JsonValue",
    "MentionRecord",
    "OntologyCode",
    "PredictionRecord",
    "SilverRecord",
    "SnapshotRecord",
]
