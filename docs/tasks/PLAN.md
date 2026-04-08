# Project Plan: Vietnamese ICD-10 Coding Adaptation

Objective: A 48-hour feasibility study to adapt TransICD-style modeling for automated ICD-10 coding using public Vietnamese medical text and weak supervision.

## 1. Project Overview

This project implements a low-resource pipeline for 3-character ICD-10 code prediction. It utilizes SEA-LION-ModernBERT-300M as the backbone and generates silver labels through a multi-source weak supervision framework.

## 2. Existing Specifications

The coding agent should refer to the following documents for implementation details:

- Data Schema: `docs/data/v1/de-01-schemas.md` (Defines data structures for text, ontology, and labels)
- ICD Ontology: `docs/data/v1/on-01-icd-ontology.md` (Official MoH ICD-10 structure and searchable fields)

## 3. Task Streams and Status

### Stream A: Data Engineering

Task DE-1: Unified schema definition (Completed).
Task DE-2: Public dataset ingestion (VietMed-Sum, PET/CT, etc.).
Task DE-3: Text normalization and sentence segmentation.

### Stream B: Ontology and Baselines

Task ON-1 to ON-3: Ontology construction and rule extraction (Completed).
Task BM-1: Lexical baseline (Exact/Fuzzy matching).
Task BM-2: Retrieval baseline (TF-IDF/BM25).

### Stream C: Dataset Creation (Weak Supervision)

Task DC-1: Medical mention extraction.
Task DC-2: Abbreviation normalization.
Task DC-3: Candidate ICD generation (Depends on Stream B).
Task DC-4: Weak supervision aggregation (Produces silver labels).
Task DC-5: Train/Val/Test data splitting.

### Stream D: Model Architecture

Task MA-1: ModernBERT encoder wrapper.
Task MA-3: TransICD code-wise attention head.
Task TE-1: Training on silver-labeled data.

### Stream E: Evaluation and Explainability

Task TE-2: Evaluation suite (Micro-F1, Macro-AUC, P@k).
Task EX-1: Token-to-text alignment.
Task EX-2: Code-wise attention span extraction.
Task EX-5: Final report formatting.

## 4. Key Dependencies and Parallelism

- Foundation: DE-1 and ON-1/2 are prerequisites for all logic.
- Parallel Execution: Model Architecture (Stream D) can be developed using dummy data based on DE-1 schemas while Dataset Creation (Stream C) is processing.
- Critical Path: Data Processing -> Weak Supervision -> Model Training -> Explainability.

## 5. Scope Constraints

- Target: 3-character ICD-10 codes.
- Language: Vietnamese text with bilingual ontology support.
- Non-Overclaim: Focus on public data feasibility, not clinical EMR deployment.
