# Project Plan: Vietnamese ICD-10 Coding Adaptation

Objective: A 48-hour feasibility study to adapt TransICD-style modeling for automated ICD-10 coding using public Vietnamese medical text and weak supervision.

## 1. Project Overview

This project implements a low-resource pipeline for **3-character ICD-10 code prediction/linking** on **public Vietnamese medical text**. It uses **SEA-LION-ModernBERT-300M** as the backbone and generates **silver labels** through ontology-driven weak supervision.

The plan has been updated to leverage an **official Vietnamese ICD-10 PDF resource** as the **primary ontology source**, with the MoH/KCB website used only as a fallback when needed.

## 2. Existing Specifications

The coding agent should refer to the documents inside `docs/data/v1/` for implementation detail.

## 3. Updated Implementation Principles

### 3.1 Ontology Source Priority

- **Primary source:** official ICD-10 PDF
- **Fallback source:** MoH/KCB ICD website only for missing sections or parse failures

### 3.2 Target Granularity

- **Primary target:** ICD-10 **3-character** codes
- **Fallback benchmark:** **chapter prediction**
- **Optional only if time remains:** limited 4-character pilot

### 3.3 Why This Scope Is Locked

- The official ICD-10 material already provides:
  - chapter -> block -> 3-character -> 4-character hierarchy,
  - bilingual Vietnamese-English terminology,
  - inclusion/exclusion notes,
  - principal diagnosis and fallback coding guidance.
- This reduces ontology construction effort and supports a more stable 48-hour implementation.

## 4. Task Streams and Status (P0 and P1 only)

### Stream A: Data Engineering

- **DE-1**: Unified schema definition (**Completed**, P0)
- **DE-2**: Public dataset ingestion (**Completed**, P0)
- **DE-3**: Parse source formats to bronze (**Completed**, P0)

### Stream B: Ontology and Rules

- **ON-1**: Parse official ICD-10 PDF (primary) + MoH web fallback (**Completed**, P0)
- **ON-1b**: Extract intro guidance and official 3-char policy from PDF (**Completed**, P0)
- **ON-2**: Normalize ontology from PDF and build bilingual index (**Completed**, P0)
- **ON-2b**: Build alias dictionary from PDF synonyms and inclusion text (**Completed**, P0)
- **ON-3**: Extract coding rules and notes from PDF intro + per-code notes (**Completed**, P0)
- **ON-3b**: Parse dagger/asterisk and cross-reference links (**TODO**, P1)

### Stream C: Dataset Creation (Weak Supervision)

- **DC-1**: Normalize text, segment, and extract mentions (**Completed**, P0)
- **DC-2**: Abbreviation normalization (**Completed**, P0)
- **DC-2b**: Ontology-driven mention normalization using PDF alias dictionary (**Completed**, P0)
- **DC-3**: Candidate ICD generation using PDF-first ontology (**Completed**, P0)
- **DC-4**: Weak supervision aggregation (**TODO**, P0)
- **DC-5**: Build packaged train/val/test split from silver outputs (**TODO**, P0)
- **QA-1**: Manual QA on high-confidence silver labels using PDF rules (**TODO**, P1)

### Stream D: Model Architecture and Training

- **MA-1**: SEA-LION Backbone wrapper (**Completed**, P1)
- **MA-2**: Linear head (**TODO**, P1)
- **MA-3**: TransICD code-wise attention head (**Completed**, P0)
- **MA-4**: Label encoding from bilingual PDF ontology (**Completed**, P0)
- **TE-1**: Training loop on packaged silver-derived splits (**TODO**, P0)
- **TE-1b**: Train chapter-first model before full 3-char model (**TODO**, P1)
- **TE-2**: Evaluation metrics setup (**Completed**, P0)

### Stream E: Baselines, Evaluation, and Explainability

- **BM-1**: Rule-based baseline using official PDF notes (**Completed**, P0)
- **BM-1b**: Chapter-first baseline with official 3-char backoff (**TODO**, P0)
- **BM-2**: TF-IDF/BM25 baseline over bilingual PDF ontology (**Completed**, P1)
- **EX-1**: Token-to-text alignment (**In Progress**, P1)
- **EX-2**: Attention-based explanation (**TODO**, P0)
- **EX-4**: Label alignment explanation from PDF descriptors (**TODO**, P1)
- **EX-5**: Final report formatting (**TODO**, P0)

## 5. Key Dependencies and Parallelism

### 5.1 Foundation Dependencies

- **DE-1** remains the schema foundation for all downstream tasks.
- **ON-1 / ON-2 / ON-3** now form the primary ontology-and-rules foundation.
- **ON-2b** is required before strong ontology-driven normalization and exact/alias matching.

### 5.2 Parallel Execution Opportunities

- **Model Architecture** can continue in parallel using dummy data because **MA-1** and **MA-3** already exist.
- **Ontology parsing** and **public data ingestion** can proceed simultaneously.
- **Baselines** can start as soon as **ON-2** and **DC-1/DC-2b** are available, without waiting for full model training.

### 5.3 Critical Path

PDF Ontology Parsing -> Ontology Normalization + Rules -> Mention Normalization -> Candidate Generation -> Weak Supervision -> Train Split Packaging -> Model Training -> Explainability

## 6. Scope Constraints

- **Target:** ICD-10 **3-character** codes
- **Fallback:** ICD chapter prediction
- **Language:** Vietnamese text with bilingual ontology support
- **Data setting:** public medical text only
- **Non-overclaim:** feasibility study for public-data ICD coding/linking, not hospital EMR deployment

## 7. Updated Implementation Priorities

### P0 Must-Haves

1. Parse the official ICD-10 PDF and normalize the ontology
2. Extract official coding rules from the introduction and note fields
3. Build alias dictionary and ontology-driven candidate generation
4. Produce silver labels and packaged train/val/test splits
5. Run rule-based and chapter-first baselines
6. Train TransICD-style model on 3-character labels
7. Deliver basic explainability outputs

### P1 Important Support Tasks

1. Parse cross-references (dagger/asterisk) if feasible
2. Implement linear baseline head and chapter-first model fallback
3. Run TF-IDF/BM25 retrieval baseline
4. Add token alignment and label-alignment explanations
5. Perform lightweight manual QA on high-confidence silver labels
