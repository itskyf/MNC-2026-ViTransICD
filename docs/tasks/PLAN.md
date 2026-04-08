# Project Plan: Vietnamese ICD-10 Coding Adaptation

Objective: A 12-hour feasibility sprint to unblock a low-resource Vietnamese ICD-10 coding/linking pipeline using public Vietnamese medical text and weak supervision.

## 1. Project Overview

This sprint does not attempt to finish the full pipeline.
Its main objective is to complete the minimum reliable public-text ingestion layer after DE-1.

Primary target of the sprint:

- produce validated `DocumentRecord` artifacts from a small number of public datasets
- keep outputs compatible with downstream ontology linking, weak supervision, and TransICD-style modeling

## 2. Existing Specifications

The coding agent should refer to the following documents for implementation details:

- Data Schema: `docs/data/v1/de-01-schemas.md`
- ICD Ontology: `docs/data/v1/on-01-icd-ontology.md`
- Frozen schema code: `src/mnc/schemas/`

## 3. Updated Scope for 12 Hours

Required datasets:

- VietMed-Sum
- ViHealthQA

Optional dataset:

- VietMed-NER

Deferred datasets:

- ViMQ
- VietMed
- PET/CT
- ViX-Ray
- other auxiliary corpora

Primary output of this sprint:

- validated JSONL artifacts of `DocumentRecord`
- one ingestion manifest per dataset
- pytest unit tests for adapters, normalization, validation, and writing

## 4. Task Streams and Status

### Stream A: Data Engineering

Task DE-1: Unified schema definition.
Status:

- Completed

Task DE-2: Public dataset ingestion.
Status:

- Highest priority
- In scope for 12-hour sprint

Task DE-3: Text normalization and sentence preparation.
Status:

- Reduced scope
- Only light deterministic normalization in this sprint

### Stream B: Ontology and Baselines

Task ON-1 to ON-3: Ontology construction and rule extraction.
Status:

- Already available as external dependency for later stages
- Not critical path for DE-2

Task BM-1: Lexical baseline.
Status:

- Deferred

Task BM-2: Retrieval baseline.
Status:

- Deferred

### Stream C: Dataset Creation (Weak Supervision)

Task DC-1: Medical mention extraction.
Status:

- Deferred

Task DC-2: Abbreviation normalization.
Status:

- Deferred

Task DC-3: Candidate ICD generation.
Status:

- Deferred

Task DC-4: Weak supervision aggregation.
Status:

- Deferred

Task DC-5: Train/Val/Test data splitting.
Status:

- Deferred
- Existing source splits should be preserved

### Stream D: Model Architecture

Task MA-1: ModernBERT encoder wrapper.
Status:

- Deferred

Task MA-3: TransICD code-wise attention head.
Status:

- Deferred

Task TE-1: Training on silver-labeled data.
Status:

- Deferred

### Stream E: Evaluation and Explainability

Task TE-2: Evaluation suite.
Status:

- Deferred

Task EX-1: Token-to-text alignment.
Status:

- Deferred

Task EX-2: Code-wise attention span extraction.
Status:

- Deferred

Task EX-5: Final report formatting.
Status:

- Deferred

## 5. Priority Labels

P0:

- DE-2.1: Implement shared light text normalization
- DE-2.2: Implement shared JSONL writer for `DocumentRecord`
- DE-2.3: Implement VietMed-Sum adapter
- DE-2.4: Implement ViHealthQA adapter
- DE-2.5: Implement manifest writer
- DE-2.6: Add pytest unit tests
- DE-2.7: Run end-to-end ingestion for required datasets

P1:

- DE-2.8: Implement VietMed-NER adapter with text-only mapping
- DE-2.9: Add CLI or minimal entrypoint wrapper
- DE-3.1: Better sentence splitting if it is trivial and low risk

P2:

- All ontology-linking, weak supervision, baseline, model, training, evaluation, and explainability tasks

## 6. Detailed Task Breakdown

### DE-2.1 Shared light normalization

Goal:

- normalize public text safely without changing semantics

Expected behavior:

- unicode normalization
- trim whitespace
- collapse repeated whitespace
- replace tabs/newlines with spaces
- preserve casing and Vietnamese diacritics

Output:

- normalized string
- retrieval string identical to normalized string
- `sentences = [normalized_text]`

### DE-2.2 Shared writer

Goal:

- write validated `DocumentRecord` rows to JSONL

Requirements:

- each row must pass schema validation
- deterministic write order
- no empty-text rows in final artifact

### DE-2.3 VietMed-Sum adapter

Goal:

- ingest only Vietnamese summary text for the minimum viable pipeline

Mapping:

- `raw_text = summary`
- `source = "vietmed_sum:<split>"`
- `doc_id = "vietmed_sum:<split>:<row_idx>"`

Notes:

- do not ingest transcript variants in this sprint
- preserve source split

### DE-2.4 ViHealthQA adapter

Goal:

- ingest QA text as a single document

Mapping:

- `raw_text = question + "\n" + answer`
- `source = "vihealthqa:<split>"`
- `doc_id = "vihealthqa:<split>:<id>"`

Notes:

- ignore `link` except optional provenance in manifest or raw landing area

### DE-2.5 Manifest writer

Goal:

- produce one manifest per dataset for auditability

Fields:

- dataset name
- source location
- split names
- written row counts
- skipped row counts
- text field mapping
- ingestion timestamp

### DE-2.6 Pytest suite

Goal:

- ensure adapters and writers are correct without network access

Required unit tests:

- map VietMed-Sum row to `DocumentRecord`
- map ViHealthQA row to `DocumentRecord`
- normalization collapses whitespace
- empty text row is skipped
- missing required field raises typed error
- doc ids are deterministic
- output rows validate against schema
- manifest counts are correct
- `sentences` defaults to one full-text item

### DE-2.7 End-to-end run

Goal:

- generate final bronze artifacts for required datasets

Deliverables:

- `data/bronze/documents/vietmed_sum/*.jsonl`
- `data/bronze/documents/vihealthqa/*.jsonl`
- `data/bronze/manifests/*.json`

### DE-2.8 Optional VietMed-NER adapter

Goal:

- ingest only plain `text` field if P0 is complete

Mapping:

- `raw_text = text`
- `source = "vietmed_ner:<split>"`
- `doc_id = "vietmed_ner:<split>:<row_idx>"`

Notes:

- ignore audio and sequence labels

## 7. Key Dependencies and Parallelism

Foundation:

- DE-1 is already complete and unblocks all downstream DE-2 work

Critical path:

- shared normalization
- required dataset adapters
- writer
- tests
- end-to-end run

Possible parallelism:

- one agent writes shared normalization and writer
- one agent writes VietMed-Sum adapter
- one agent writes ViHealthQA adapter
- tests can start as soon as adapter contracts are fixed

Not on critical path:

- ontology linkage
- model code
- training code
- explainability code

## 8. Scope Constraints

- Target of this sprint is text ingestion only
- Keep target task framing as public Vietnamese medical text -> future ICD chapter / 3-char linking
- Do not claim discharge-summary ICD coding
- Do not add new datasets unless P0 is done
- Do not change frozen schema
- Use pytest for unit tests
- Prefer deterministic, low-risk implementation over feature completeness

## 9. Definition of Done

The sprint is successful if:

- VietMed-Sum train/dev/test are ingested into valid `DocumentRecord` JSONL files
- ViHealthQA train/val/test are ingested into valid `DocumentRecord` JSONL files
- all rows pass schema validation
- manifests are written
- pytest passes for required adapters and shared utilities

The sprint is strong if additionally:

- VietMed-NER is ingested in text-only mode

## 10. Execution Order for the 12-Hour Window

Hour 0-1:

- finalize DE-2 contract and output paths

Hour 1-3:

- implement shared normalizer and writer

Hour 3-5:

- implement VietMed-Sum adapter
- implement ViHealthQA adapter

Hour 5-7:

- write pytest unit tests
- fix validation and edge cases

Hour 7-9:

- run end-to-end ingestion for required datasets
- write manifests
- sanity-check outputs

Hour 9-11:

- optional VietMed-NER adapter

Hour 11-12:

- cleanup
- final verification
- prepare handoff note for downstream tasks

## 11. Handoff Note for Later Phases

After DE-2 completes, the next recommended order is:

- DE-3 reduced normalization improvements
- mention extraction
- candidate generation
- weak supervision aggregation
- baseline retrieval / lexical matching
- TransICD-style modeling
