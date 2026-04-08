# Minimal Unified Schema for Agent Collaboration

## Purpose

This document defines the smallest set of schemas needed for agents to work together without blocking each other.

The pipeline target is document-level ICD-10 3-character coding/linking on public Vietnamese medical text.

The schema is intentionally minimal:

* Keep only fields that are required by downstream agents
* Preserve provenance and offsets for auditability
* Separate intermediate artifacts clearly by pipeline step
* Use Pydantic v2-friendly types

## Pipeline mapping

* `DocumentRecord` serves ingestion, normalization, retrieval, training input, and explainability alignment
* `OntologyCode` serves ontology parsing, retrieval, candidate generation, and label semantics
* `MentionRecord` serves mention extraction and evidence tracking
* `CandidateLink` serves candidate generation and weak supervision
* `SilverRecord` serves weak-label aggregation, dataset split, and model training
* `PredictionRecord` serves inference and evaluation
* `ExplanationRecord` serves explainability and demo output

## Pydantic v2 conventions

* Use `datetime` for timestamps
* Use `str | None` instead of `Optional[str]`
* Use `list[str]` and `dict[str, float]`
* Use `Literal[...]` where the value space is small and stable
* Keep `extra="forbid"` to catch schema drift early
* Keep one model per artifact type, not one giant nested model

## 1. DocumentRecord

Step served:

* dataset ingestion
* text normalization
* retrieval input
* explainability alignment

Required fields:

* `doc_id`: stable document key
* `source`: source dataset name
* `raw_text`: original text for audit and highlighting
* `normalized_text`: normalized text for NLP
* `retrieval_text`: normalized text for lexical retrieval
* `sentences`: sentence-level segmentation
* `created_at`: artifact timestamp

```python
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    raw_text: str
    normalized_text: str
    retrieval_text: str
    sentences: list[str]
    created_at: datetime
````

## 2. OntologyCode

Step served:

* ontology construction
* candidate generation
* label lookup
* label-description support

Required fields:

* `code_3char`: canonical prediction label
* `chapter_id`: fallback label group
* `title_vi`: primary Vietnamese label text
* `title_en`: optional bilingual support
* `aliases`: normalized variants for matching
* `search_text`: compact searchable text
* `created_at`: artifact timestamp

```python
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class OntologyCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code_3char: str
    chapter_id: str | None = None
    title_vi: str
    title_en: str | None = None
    aliases: list[str] = []
    search_text: str
    created_at: datetime
```

## 3. MentionRecord

Step served:

* mention extraction
* evidence tracking
* candidate linking

Required fields:

* `mention_id`: stable mention key
* `doc_id`: owning document
* `text`: original surface form
* `normalized_text`: canonicalized mention form
* `mention_type`: disease, symptom, diagnosis, or other
* `char_start`: start offset in raw text
* `char_end`: end offset in raw text
* `confidence`: extractor confidence

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict

class MentionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_id: str
    doc_id: str
    text: str
    normalized_text: str
    mention_type: Literal["disease", "symptom", "diagnosis", "procedure", "abbreviation", "other"]
    char_start: int
    char_end: int
    confidence: float | None = None
    created_at: datetime
```

## 4. CandidateLink

Step served:

* candidate generation
* weak supervision input
* audit of match channels

Required fields:

* `doc_id`: owning document
* `mention_id`: source mention, or `None` for document-level retrieval
* `code_3char`: linked candidate
* `method`: exact, normalized, fuzzy, tfidf, bm25, dense
* `score`: merged candidate score
* `char_start`: optional evidence offset
* `char_end`: optional evidence offset

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict

class CandidateLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    mention_id: str | None = None
    code_3char: str
    method: Literal["exact", "normalized", "fuzzy", "tfidf", "bm25", "dense"]
    score: float
    char_start: int | None = None
    char_end: int | None = None
    created_at: datetime
```

## 5. SilverRecord

Step served:

* weak supervision aggregation
* train/val/test split
* model training target

Required fields:

* `doc_id`: owning document
* `label_granularity`: `code_3char` or `chapter`
* `silver_labels`: final weak labels
* `candidate_codes`: top candidate list for reranking or inspection
* `confidence`: document-level weak-label confidence
* `split`: train, val, or test

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict

class SilverRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    label_granularity: Literal["code_3char", "chapter"] = "code_3char"
    silver_labels: list[str]
    candidate_codes: list[str]
    confidence: float | None = None
    split: Literal["train", "val", "test"]
    created_at: datetime
```

## 6. PredictionRecord

Step served:

* model inference
* evaluation
* comparison across baselines

Required fields:

* `doc_id`: owning document
* `model_name`: model identifier
* `label_granularity`: prediction target level
* `predicted_codes`: final predicted labels
* `scores`: per-label score map

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict

class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    model_name: str
    label_granularity: Literal["code_3char", "chapter"] = "code_3char"
    predicted_codes: list[str]
    scores: dict[str, float]
    created_at: datetime
```

## 7. ExplanationRecord

Step served:

* explainability
* demo formatting
* qualitative error analysis

Required fields:

* `doc_id`: owning document
* `code_3char`: label being explained
* `spans`: top evidence spans
* `matched_label_text`: ontology phrase used for support
* `summary`: compact human-readable explanation

```python
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    char_start: int
    char_end: int
    text: str
    score: float | None = None

class ExplanationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    code_3char: str
    spans: list[EvidenceSpan]
    matched_label_text: str | None = None
    summary: str | None = None
    created_at: datetime
```

## Join keys

* `doc_id` is the main join key across all stages
* `mention_id` links mention extraction to candidate generation
* `code_3char` links ontology, weak labels, predictions, and explanations

## Minimal lineage

* `DocumentRecord` is created first
* `OntologyCode` is built in parallel from official resources
* `MentionRecord` is extracted from `DocumentRecord`
* `CandidateLink` links mentions or documents to `OntologyCode`
* `SilverRecord` aggregates `CandidateLink` into training targets
* `PredictionRecord` is produced by baselines or neural models
* `ExplanationRecord` is produced from predictions plus text offsets

## Recommended storage

* `DocumentRecord`: JSONL or Parquet
* `OntologyCode`: JSONL or Parquet
* `MentionRecord`: JSONL or Parquet
* `CandidateLink`: JSONL or Parquet
* `SilverRecord`: JSONL or Parquet
* `PredictionRecord`: JSONL
* `ExplanationRecord`: JSONL

## What is intentionally omitted

* Source-specific metadata that is not used downstream
* Deep ICD descendant fields if the current task is only 3-character coding
* Full rule-engine structures for instruction logic
* Token-level tensors or model internals
* Duplicated fields that can be joined by key

## One practical rule

If a downstream agent cannot proceed without a field, keep it.
If a field is only nice to have for reporting, omit it for now.
