#import "@preview/elspub:1.0.0": *
// #import "../src/elspub.typ": *

#show: elspub.with(
  journal: mssp,
  paper-type: none,
  title: [Low-Resource Vietnamese ICD-10 Coding via Weakly Supervised TransICD],
  keywords: (
    "Elsevier",
    "Typst",
    "Template",
  ),
  authors: (
    (
      name: [Ky Anh Pham],
      affiliations: ("a",),
      corresponding: true,
      orcid: "0000-0001-2345-6789",
      email: "s.pythagoras@croton.edu",
    ),
    (name: [M. Thales], affiliations: ("b",)),
  ),
  affiliations: (
    "a": [School of Pythagoreans, Croton, Magna Graecia],
    "b": [Milesian School of Natural Philosophy, Miletus, Ionia],
  ),
  abstract: lorem(100),
  paper-info: (
    year: [510 BCE],
    paper-id: [123456],
    volume: [1],
    issn: [1234-5678],
    received: [01 June 510 BCE],
    revised: [01 July 510 BCE],
    accepted: [01 August 510 BCE],
    online: [01 September 510 BCE],
    doi: "https://doi.org/10.1016/j.aam.510bce.101010",
    open: cc-by,
    extra-info: [Communicated by C. Eratosthenes],
  ),
)

= Introduction
Automatic assignment of International Classification of Diseases (ICD) codes is an important problem in clinical natural language processing because it sits at the intersection of documentation quality, downstream analytics, and administrative workflow support. In high-resource settings, this task has been studied using discharge summaries and other hospital records, often under multi-label classification setups with large and highly imbalanced code spaces. Recent work has shown that label-aware neural architectures, especially those that incorporate code-specific attention, can improve both predictive performance and interpretability. [TODO(cite) dr_caml] [TODO(cite) transicd] [TODO(cite) plm_icd]

However, the Vietnamese setting is fundamentally different. To the best of our knowledge, there is no widely used public benchmark consisting of Vietnamese discharge summaries or electronic medical records paired with ICD-10 labels. As a result, a paper that simply claims ``automatic ICD-10 coding for Vietnamese discharge summaries'' would overstate what the available data can support. This mismatch between the target problem and the available resources is not merely a practical inconvenience; it also affects the scientific validity of model evaluation, because training and testing distributions would not align with the claimed application scenario. [TODO(cite) vietnamese_medical_datasets_survey]

This paper addresses that gap with a more careful and realistic framing. Rather than assuming unavailable private hospital notes, we study whether a TransICD-style model can be adapted to Vietnamese under a low-resource regime using only public resources. Concretely, we combine public Vietnamese medical text corpora with an ontology built from official ICD-10 resources released through the Vietnamese Ministry of Health coding system. This allows us to formulate the task as Vietnamese ICD-10 coding/linking on public medical text, with two practical label granularities: chapter prediction and 3-character ICD-10 code prediction. [TODO(cite) moh_icd10_resource] [TODO(cite) transicd]

Our central methodological contribution is a new dataset construction pipeline designed for this resource-constrained setting. Because gold note-level ICD labels are unavailable, we build a silver-labeled dataset from scratch. The construction process integrates ontology scraping and normalization, mention extraction, abbreviation handling, exact and fuzzy string matching, retrieval over ICD descriptions, and weak supervision aggregation. The output is a document-level dataset with candidate codes, silver labels, evidence spans, and confidence scores. This dataset is not a replacement for expert-coded hospital data, but it is sufficient to support a structured feasibility study and to make the research question empirically testable. [TODO(cite) weak_supervision] [TODO(cite) ontology_linking]

On the modeling side, we keep the spirit of the original TransICD proposal as intact as possible. We adopt `aisingapore/SEA-LION-ModernBERT-300M` as the Vietnamese-capable encoder backbone and place a TransICD-style code-wise attention head on top of it. This design preserves the key idea of label-aware representation learning while adapting the encoder to the target language and the public-data setting. To make the study scientifically grounded, we evaluate against retrieval-based and simpler neural baselines, and we include an ablation study that isolates the impact of ontology semantics, bilingual descriptions, abbreviation normalization, candidate generation, and explainability-related components. [TODO(cite) transicd] [TODO(cite) sea_lion_modernbert]

In summary, this paper makes three contributions. First, it reframes Vietnamese ICD coding in a way that is consistent with the actual data landscape, avoiding unsupported claims about hospital discharge summaries. Second, it proposes a practical ontology-driven data construction pipeline that creates a new silver-labeled benchmark from public Vietnamese medical text. Third, it studies a low-resource adaptation of TransICD with a modern Vietnamese-capable encoder and evaluates the extent to which label-aware modeling remains useful in this setting. [TODO(cite) transicd] [TODO(cite) vietnamese_medical_datasets_survey]
= Related Work

== Automatic ICD Coding

Automatic ICD coding has a long history in clinical NLP, with earlier systems relying on feature engineering, linear models, and pipeline-based approaches before the field shifted toward neural multi-label architectures. A major line of work models ICD assignment as extreme or large-scale multi-label text classification, where the challenge is not only to represent long clinical documents but also to distinguish among many infrequent labels. CNN-based and recurrent architectures established strong early baselines, while later models introduced explicit label attention to better align document evidence with individual codes. [TODO(cite) caml] [TODO(cite) dr_caml] [TODO(cite) early_icd_coding]

Among label-aware approaches, TransICD is particularly relevant to our work. It replaces purely document-level scoring with a Transformer-based encoder followed by code-wise attention, allowing the model to construct code-specific document representations and providing a natural path toward interpretability. TransICD also addresses severe label imbalance through LDAM-style loss design, which is highly relevant for ICD coding because frequent and rare codes often follow a long-tail distribution. Our work does not claim to reproduce the original hospital-note setting of TransICD in Vietnamese; instead, it adapts its central modeling idea to a different supervision regime built from public data. [TODO(cite) transicd] [TODO(cite) ldam]

A second relevant direction is the use of pretrained language models for clinical coding and clinical text classification. Domain-specific encoders, label-description-aware models, and retrieval-enhanced methods have all shown that ICD coding benefits from incorporating semantic information from code definitions rather than treating labels as opaque class identifiers. This insight is especially important in low-resource settings, where label descriptions can compensate partially for limited annotated examples. Our methodology follows this line by explicitly encoding Vietnamese and bilingual ICD descriptions as part of both candidate generation and model scoring. [TODO(cite) clinicalbert] [TODO(cite) plm_icd] [TODO(cite) label_description_models]

== Ontology-Guided and Weakly Supervised Medical NLP

When gold clinical labels are scarce or inaccessible, ontology-guided and weakly supervised approaches provide a useful alternative. In biomedical NLP, controlled vocabularies and taxonomies are frequently used for entity linking, terminology normalization, and distant supervision. The key idea is that expert-defined symbolic resources can provide structure, candidate spaces, and partial supervision, even when document-level gold annotations are missing. This is particularly attractive for ICD coding, where the ontology itself contains hierarchical organization, textual descriptions, synonyms, and coding notes. [TODO(cite) ontology_linking] [TODO(cite) weak_supervision] [TODO(cite) biomedical_entity_linking]

Weak supervision is not a complete substitute for human annotation, and silver labels can be noisy. Still, a properly designed weak supervision pipeline can be scientifically useful if its limitations are made explicit and if it is paired with careful baselines, confidence-aware training, and qualitative analysis. In our case, weak supervision serves two purposes: it creates a workable benchmark from public data, and it preserves evidence spans that later support explainability. This dual role is important because one of the original motivations of TransICD is not just prediction but also the ability to surface code-specific evidence in text. [TODO(cite) snorkel] [TODO(cite) weak_supervision_medical]

== Vietnamese Medical NLP Resources

Vietnamese medical NLP is still comparatively under-resourced. Publicly available datasets exist for related tasks such as medical ASR, spoken medical NER, question answering, conversation summarization, and disease-related text understanding. These resources are valuable for domain adaptation, mention extraction, terminology discovery, and clinical-style language modeling. However, they do not constitute a standard benchmark for note-level ICD coding. In particular, they generally lack expert-assigned document-level ICD labels, and many of them are not discharge summaries in the strict clinical documentation sense. [TODO(cite) vietmed] [TODO(cite) vietmed_sum] [TODO(cite) vimedner] [TODO(cite) vihealthqa]

This gap motivates the core design choice of our work: rather than treating these datasets as if they were direct ICD coding benchmarks, we use them as auxiliary public text sources from which disease and symptom mentions can be extracted and linked to an official ICD-10 ontology. This allows us to remain faithful to the broader goal of Vietnamese ICD coding research while maintaining responsible claims about what the data actually supports. [TODO(cite) vietnamese_medical_datasets_survey]
= Methodology

== Problem Formulation and Scope

We study Vietnamese ICD-10 coding/linking under a low-resource, public-data setting. Let a document $d$ be a public Vietnamese medical text instance, such as a clinical-style report, a doctor--patient summary, or another medically relevant text unit. The goal is to predict either (i) one or more ICD chapters, or (ii) one or more ICD-10 3-character codes associated with the content of the document. The task is therefore formulated as multi-label classification or ranking over a medically structured label space. [TODO(cite) transicd]

Importantly, this formulation is narrower and more defensible than claiming full automatic coding for hospital discharge summaries. Our study should be understood as a feasibility study of low-resource adaptation, not as a deployment-ready hospital coding system. Throughout the paper, we distinguish carefully between silver labels generated by our pipeline and gold labels assigned by professional coders, which we do not have access to. This framing is central to the methodological integrity of the work. [TODO(cite) weak_supervision] [TODO(cite) limitations_in_clinical_nlp]

== Overview of the Pipeline

The complete pipeline consists of five stages:

1. official ICD-10 ontology construction from Vietnamese Ministry of Health resources,
2. ingestion and normalization of public Vietnamese medical text,
3. mention extraction and terminology normalization,
4. ontology-based candidate generation and weak supervision aggregation,
5. model training and evaluation on the resulting silver-labeled dataset.

Conceptually, the data flow is:

`raw public medical text → normalized mentions → ontology candidates → filtered silver labels → model training and evaluation`

This section focuses especially on Stage 1 through Stage 4, since dataset creation is the main methodological novelty of the paper.

== Stage 1: ICD-10 Ontology Construction from Official Resources

Because no ready-made Vietnamese ICD coding benchmark is publicly available, we first construct the label space itself from official Vietnamese ICD-10 resources. We collect data from the Ministry of Health clinical coding system and related official ICD-10 pages, including the Vietnamese ICD-10 hierarchy, the bilingual ICD-10 view, and the coding instruction page. [TODO(cite) moh_icd10_resource]

From these sources, we parse and normalize the following ontology elements:

- chapter identifiers and titles,
- block or group identifiers,
- 3-character ICD codes,
- deeper descendants such as 4-character codes when available,
- Vietnamese descriptions,
- English descriptions from bilingual pages,
- parent--child relations,
- coding notes and instruction snippets where available.

We then flatten this information into a searchable ontology table in which each code entry contains its hierarchical context, canonical names, and textual fields for matching and retrieval. For example, an ontology row may contain a 3-character code, its Vietnamese title, English title, associated block, parent chapter, aliases, and concatenated search text. The bilingual view is retained because medical text in Vietnam often contains mixed Vietnamese and English terminology, especially for disease names, abbreviations, and specialist reporting language. [TODO(cite) moh_icd10_resource]

This ontology construction step is more than a preprocessing convenience. It defines the target label space, provides textual semantics for code-aware models, and supplies structured knowledge for weak supervision. Without a reliable ontology, any claim about ICD prediction would be unstable, because the labels themselves would not be grounded in an official coding standard.

== Stage 2: Public Vietnamese Medical Text Collection

Since Vietnamese lacks a standard public discharge-summary-to-ICD dataset, we build the document collection from multiple public medical NLP resources. We prioritize corpora that are relevant to one or more of the following needs:

- clinical-style or report-like language,
- disease and symptom mention extraction,
- terminology coverage,
- abbreviation normalization,
- public accessibility and reproducibility.

The collected sources include public Vietnamese medical corpora such as medical ASR transcripts, medical conversation summaries, medical NER datasets, healthcare question-answer data, disease-related symptom corpora, and related auxiliary resources. [TODO(cite) vietmed] [TODO(cite) vietmed_sum] [TODO(cite) vimedner] [TODO(cite) vihealthqa] [TODO(cite) vimq] [TODO(cite) vimedical_disease]

These datasets are heterogeneous by design. Some contain relatively clinical language, such as medical conversation summaries or report-like text, while others are more consumer-facing, such as health questions. We do not claim that they are equivalent to discharge summaries. Instead, we treat them as public medical text from which medically meaningful mentions can be extracted and mapped to the ICD ontology. This design choice is essential: it allows the study to remain reproducible and privacy-safe while still engaging with medically grounded language. [TODO(cite) vietnamese_medical_datasets_survey]

Each source is converted into a common document schema containing at least the following fields:

- `doc_id`,
- `source`,
- `raw_text`,
- `normalized_text`,
- optional metadata,
- extracted mention list,
- candidate ICD list,
- silver labels,
- evidence spans,
- confidence score.

The unified schema allows all downstream steps to operate at the document level, even though the original sources may come from different tasks.

== Stage 3: Mention Extraction and Text Normalization

The next step is to convert raw public text into medically meaningful units that can be linked to ICD codes. We first normalize the text by standardizing punctuation, white space, and Unicode forms while preserving the original text for later explanation alignment. We also create a retrieval-oriented normalization view for matching and indexing. [TODO(cite) vietnamese_tokenization] [TODO(cite) medical_ner_vi]

We then extract disease-, symptom-, and diagnosis-related mentions using a hybrid strategy. When NER annotations are available from public datasets, we use them directly or as supervision signals for mention patterns. When explicit annotations are not available, we back off to lexicon matching, phrase heuristics, and terminology mining from public medical corpora. The result is a set of mention spans with normalized text, mention type, and confidence. [TODO(cite) vimedner] [TODO(cite) vietmed_ner] [TODO(cite) medical_ner_vi]

A key difficulty in Vietnamese medical text is abbreviation and variant handling. Clinical phrases may appear as shortened forms, mixed Vietnamese--English expressions, or spelling variants. To reduce fragmentation, we apply abbreviation normalization and lexicon expansion using a combination of public abbreviation resources, mined variants, and bilingual ontology entries. This step is especially important because weak supervision quality depends heavily on whether text mentions can be canonically linked to ontology descriptions. [TODO(cite) abbreviation_normalization] [TODO(cite) moh_icd10_resource]

== Stage 4: Ontology-Based Candidate Generation

After mention extraction, each mention is linked to a set of candidate ICD codes using several complementary matching channels. Our goal is not to rely on a single heuristic, but to build a robust candidate pool that can later be filtered by weak supervision aggregation.

The candidate generation channels include:

- exact string matching against Vietnamese ontology titles and aliases,
- normalized matching after lowercasing and text normalization,
- fuzzy string matching for spelling variation,
- sparse retrieval such as TF-IDF or BM25 over ontology descriptions,
- optional bilingual retrieval using both Vietnamese and English fields,
- hierarchy-aware backoff from fine-grained descendants to 3-character codes or chapters.

This multi-channel design reflects the realities of the task. Exact matching is precise but brittle, while fuzzy and retrieval-based methods increase recall at the cost of more noise. By keeping the per-channel scores, we can later combine precision-oriented and recall-oriented evidence rather than forcing one matching rule to solve everything. [TODO(cite) bm25] [TODO(cite) ontology_linking]

== Stage 5: Weak Supervision and Silver Label Construction

The central dataset construction step is weak supervision aggregation. For each document, we combine mention-level evidence into document-level silver labels. This procedure produces not only predicted codes but also the rationale that supports them, which is valuable for later explainability.

For a given document, the weak supervision module collects:

- extracted mention spans,
- candidate ICD codes for each mention,
- matching and retrieval scores from multiple channels,
- hierarchy information from the ontology,
- rule-based hints from ICD notes and instructions, when available.

These signals are aggregated into a document-level score for each candidate code. In practice, the score can incorporate factors such as the strength of exact or fuzzy matching, retrieval relevance, repetition of medically consistent mentions within the same document, and section-like cues such as diagnostic or conclusion phrases when detectable. Candidate codes that pass a configurable threshold become silver labels; others remain in the candidate list but are not committed as positive labels. [TODO(cite) weak_supervision] [TODO(cite) ontology_linking]

The output of this stage is a silver-labeled dataset in which each document contains:

- a document identifier and source,
- the text itself,
- a ranked candidate ICD list,
- one or more silver ICD labels,
- evidence spans supporting each label,
- an overall confidence score or confidence tier.

This design is important for two reasons. First, it provides a trainable target for downstream models. Second, it preserves local evidence, allowing us to later ask not only which label was predicted, but also which text span most strongly supported it. Compared with plain pseudo-labeling, this dataset is therefore more structured and more suitable for explainable modeling. [TODO(cite) weak_supervision_medical]

== Data Splits and Label Granularity

We consider two label granularities:

- chapter-level prediction,
- 3-character ICD-10 code prediction/linking.

The chapter setting is coarser and typically more stable under weak supervision; the 3-character setting is closer to the original spirit of ICD coding and is therefore our main target. We split the silver dataset at the document level into training, validation, and test partitions, using source-aware and duplicate-aware splitting where possible. [TODO: specify exact split ratios.]

If the full 3-character label space proves too sparse, a practical fallback is to restrict experiments to a sufficiently represented subset or to report chapter-level results as a complementary benchmark. This fallback does not weaken the methodology; instead, it makes the evaluation more honest under low-resource constraints.

== Model Adaptation: SEA-LION-ModernBERT with TransICD-Style Code-Wise Attention

To stay as close as possible to the original proposal while adapting it to Vietnamese, we replace the original English-oriented encoder with `aisingapore/SEA-LION-ModernBERT-300M`, an encoder-only model that supports Vietnamese and provides a more modern backbone than earlier BERT-style architectures. On top of this encoder, we implement a TransICD-style label-aware prediction head. [TODO(cite) sea_lion_modernbert] [TODO(cite) transicd]

Let the encoder produce contextual token representations for a document. Instead of compressing the whole document into a single shared representation, the code-wise attention module computes a code-specific weighting over the token sequence for each candidate label or label embedding. This yields label-specific document representations that are then scored for multi-label prediction. The core intuition is that different ICD codes should attend to different parts of the text, especially in a medical setting where multiple diagnoses or symptom clusters may appear in the same document. [TODO(cite) transicd]

We further enrich the model with label descriptions from the ontology. Each ICD code is associated with textual semantics, starting with the Vietnamese title and optionally augmented with English descriptions and notes. These label descriptions can be used either to initialize label representations or to provide additional semantic features during scoring. This is particularly useful in low-resource settings because it helps the model exploit the meaning of the label rather than relying solely on observed positive examples. [TODO(cite) label_description_models]

== Explainability Design

Explainability is a first-class goal of the study, not an afterthought. We therefore design the pipeline so that the model can map predictions back to clinically meaningful evidence in Vietnamese text. Our explainability mechanism has three components.

First, code-wise attention produces token- or span-level relevance per predicted label. Second, gradient-based attribution methods such as saliency or integrated gradients can be used to validate whether the same regions remain influential beyond attention weights alone. Third, label-description alignment allows us to connect document spans to the ontology phrases that most strongly support a prediction. [TODO(cite) transicd] [TODO(cite) integrated_gradients]

At inference time, the system can therefore report:

- predicted ICD code and confidence,
- top evidence spans in the original Vietnamese text,
- top contributing tokens or phrases,
- the corresponding ICD title or description that aligned with the text.

This design is especially appropriate for a methodology paper because it makes the decision process inspectable, which is important when training labels themselves are only silver.

== Responsible Framing and Limitations of the Methodology

The methodology is designed to be scientifically sound under resource constraints, but it has clear limitations. Most importantly, the resulting dataset is silver-labeled rather than expert-coded, so model performance must be interpreted as performance relative to the constructed benchmark, not as definitive evidence of hospital-grade ICD coding accuracy. Second, the public text sources are heterogeneous and only partially clinical in style. Third, ontology-based weak supervision can introduce biases toward lexical overlap and may miss implicit diagnoses not explicitly verbalized in text. [TODO(cite) weak_supervision] [TODO(cite) limitations_in_clinical_nlp]

For these reasons, we frame the work as a low-resource feasibility study. Its value lies in showing that a careful combination of public Vietnamese medical text, official ICD ontology construction, and label-aware modeling can create a reproducible foundation for future research, rather than in claiming to have solved clinical coding in the strict hospital sense.
= Experiments

== Research Questions

The experiments are designed to answer the following questions:

1. Can an ontology-built, silver-labeled public dataset support meaningful Vietnamese ICD prediction experiments?
2. Does a TransICD-style code-wise attention model outperform simpler retrieval or neural baselines in this low-resource setting?
3. How much do ontology semantics, candidate generation, and preprocessing matter?
4. Can the model produce useful explanations that align predictions with evidence spans and label descriptions?

These questions reflect both the original proposal and the constraints of the available data.

== Experimental Tasks

We evaluate two tasks.

#enum(
  [#strong[Chapter prediction.] Each document is assigned one or more ICD chapters. This task is coarse-grained and serves as a more stable benchmark under noisy supervision.],
  [#strong[3-character ICD coding/linking.] Each document is assigned one or more 3-character ICD-10 codes. This is the main task because it is closer to the original coding objective while remaining more realistic than full fine-grained ICD coding.],
)

If needed, we additionally report a restricted-frequency setting for 3-character codes to reduce sparsity. [TODO]

== Baselines

We compare against three categories of baselines.

#strong[1. Rule-based and retrieval baselines.]
These methods do not require end-to-end supervised training and therefore provide a strong sanity check for the dataset construction pipeline.

- exact match over ontology titles and aliases,
- normalized and fuzzy string matching,
- TF-IDF or BM25 nearest-label retrieval over ontology descriptions,
- optional retrieval + reranking using encoder similarity.

#strong[2. Simple neural baseline.]
A document encoder with a pooled representation and a linear multi-label prediction head built on top of `SEA-LION-ModernBERT-300M`. This baseline tests whether code-wise attention truly adds value beyond a standard document classifier.

#strong[3. Proposed model.]
A TransICD-style model with the same encoder backbone but with code-wise attention and ontology-aware label representations.

Together, these baselines allow us to compare symbolic matching, generic neural classification, and label-aware neural modeling under the same label space. [TODO(cite) transicd] [TODO(cite) bm25] [TODO(cite) sea_lion_modernbert]

== Training Setup

All neural models use `SEA-LION-ModernBERT-300M` as the encoder backbone. [TODO: specify tokenizer, maximum sequence length, truncation or chunking strategy, optimizer, learning rate, number of epochs, batch size, and hardware.] We treat the task as multi-label prediction and use a practical imbalance-aware loss, such as weighted binary cross-entropy, as the default training objective. If time and stability permit, LDAM-style loss is also evaluated to better align with the original proposal. [TODO(cite) ldam]

Because the training labels are silver, we optionally weight training instances by their confidence scores. This helps reduce the effect of noisy positives and keeps the training objective better matched to the data construction process. Early stopping is selected based on validation performance. [TODO]

== Evaluation Metrics

For 3-character multi-label coding, we report:

- Micro-F1,
- Macro-F1,
- Macro-AUC,
- Precision@k,
- Recall@k.

For chapter-level prediction, we report:

- Accuracy,
- Macro-F1,
- Macro-AUC.

Since the benchmark is silver-labeled, quantitative results are supplemented by manual qualitative inspection on a subset of predictions. This is necessary to separate genuine model errors from noisy or ambiguous silver labels. [TODO: define qualitative evaluation protocol.]

== Main Results

[TODO: Insert a main quantitative results table.]

The main quantitative comparison should answer whether the proposed TransICD-style model consistently improves over retrieval and simple neural baselines. In particular, we expect retrieval baselines to perform reasonably well when lexical overlap is strong, but to struggle with paraphrases and implicit diagnostic phrasing. We further expect the linear-head neural baseline to provide stronger generalization than pure retrieval, while the code-wise attention model should improve label discrimination and evidence localization, especially for multi-diagnosis documents. [TODO(cite) transicd]

A concise way to present the findings is:

- retrieval methods provide a strong lower-cost baseline,
- neural classification improves robustness to wording variation,
- code-wise attention yields the best trade-off between prediction and explainability.

[TODO: replace this paragraph with actual findings once experiments are completed.]

== Qualitative Analysis and Explainability

We complement the numerical metrics with case studies. For each selected document, we show:

- the input text,
- the predicted ICD chapter or 3-character code,
- the score or confidence,
- the top evidence spans from code-wise attention,
- the top contributing tokens or phrases from attribution analysis,
- the matching ICD description from the ontology.

[TODO: insert 2--3 example figures or formatted examples.]

This qualitative component is particularly important in a weakly supervised setup. A prediction that appears incorrect relative to a silver label may still be clinically plausible, and conversely, a superficially correct label may depend on brittle lexical matching. By showing evidence spans and label-description alignment, we make these distinctions visible and provide a more nuanced interpretation of model behavior. [TODO(cite) integrated_gradients] [TODO(cite) transicd]

== Discussion of Experimental Validity

The experiments are designed to be reproducible and appropriately scoped, but they must be interpreted with care. Performance on silver labels is not the same as performance against human coders, and heterogeneous public text is not equivalent to real discharge summaries. Accordingly, the results should be read as evidence about the feasibility of ontology-driven Vietnamese ICD coding research under public-data constraints, not as a definitive benchmark for hospital deployment. This distinction is not a weakness of the paper; rather, it is part of the paper's scientific honesty.
