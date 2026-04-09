#import "@preview/elspub:1.0.0": *

#show: elspub.with(
 journal: mssp,
 paper-type: none,

 title: [Adapting the TransICD Model for Automatic Disease Coding #linebreak() on Public Vietnamese Medical Text],

 keywords: (
  "ICD-10 Coding",
  "Vietnamese Medical NLP",
  "Weak Supervision",
  "Label-Aware Attention",
  "TransICD",
 ),

 authors: (
  (name: [Ky Anh Pham], affiliations: ("a",), corresponding: true, email: "25c150abc@student.hcmus.edu.vn"),
  (name: [The Viet Le], affiliations: ("a",), email: "25c150abc@student.hcmus.edu.vn"),
  (name: [Thiet Thuat Nguyen], affiliations: ("a",), email: "25c15025@student.hcmus.edu.vn"),
  (name: [Khac Anh Duc Nguyen], affiliations: ("a",), email: "25c150abc@student.hcmus.edu.vn"),
 ),

 affiliations: (
  "a": [University of Science, Viet Nam National University Ho Chi Minh City],
 ),

 abstract: "Automatic assignment of International Classification of Diseases (ICD) codes is an important task in clinical natural language processing, but it remains underexplored in Vietnamese because publicly available note-level datasets with expert ICD annotations are not available. This paper presents a low-resource framework for Vietnamese ICD-10 coding using only public resources. We construct a silver-labeled benchmark by combining public Vietnamese medical text with an ontology derived from official ICD-10 guidelines. The pipeline integrates mention extraction, terminology normalization, candidate generation, and weak supervision to produce document-level labels and supporting evidence. We then adapt a TransICD-style model with a Vietnamese-capable pretrained encoder and code-wise attention for label-aware prediction. The study is designed around two tasks: chapter-level prediction and 3-character ICD-10 prediction. Rather than claiming hospital-ready performance, the paper aims to provide a reproducible and realistic foundation for Vietnamese ICD coding research under public-data constraints.",

  paper-info: (
    year: [2026],
    paper-id: [123456],
    volume: [1],
    issn: [1234-5678],
    received: [15 January 2026],
    revised: [20 March 2026],
    accepted: [10 April 2026],
    online: [25 April 2026],
    doi: "https://doi.org/10.1016/j.patcog.2026.04.108731",
    open: cc-by,
    extra-info: none,
  ),
),
#set text(font: "Libertinus Serif")
#show math.equation: set text(font: "New Computer Modern Math")
#show raw: set text(font: "DejaVu Sans Mono")

#show heading.where(level: 2): set text(
 style: "normal",
)


#columns(2)[

 = Introduction

 Automatic assignment of International Classification of Diseases (ICD) codes is an important structured prediction task in clinical natural language processing. In healthcare systems, ICD codes support disease surveillance, hospital reporting, reimbursement, epidemiological analysis, and many downstream data applications @ref1 @ref2 @ref3. Because manual coding is time-consuming, costly, and often inconsistent, automatic ICD coding has long been studied as a way to reduce coder workload and improve standardization @ref3 @ref4 .

 From a machine learning perspective, ICD coding is difficult for several reasons. First, the task is naturally multi-label because a single document may correspond to multiple diagnoses. Second, the label space is large and strongly long-tailed, so a small number of frequent codes dominate training while many important labels remain rare @ref3 @ref10 @ref11 @ref12 @ref13. Third, clinical notes are often long, noisy, and heterogeneous, with abbreviations, informal phrasing, repeated evidence, and irrelevant spans @ref4 @ref10 @ref17 . Fourth, coding systems used in practice require some level of interpretability, since a useful system should indicate why a code was suggested rather than only output a label set @ref4 @ref8 @ref9 @ref11 .

 Over the last several years, ICD coding research has moved from feature-engineered methods to neural architectures that learn label-specific evidence from text. CAML showed that label-wise attention can improve both prediction quality and interpretability @ref4 . MultiResCNN, LAAT, HyperCore, HLAN, and HiLAT further developed label-aware and hierarchical approaches for long documents and large label spaces @ref5 @ref6 @ref7 @ref8 @ref9. More recently, pretrained language model approaches have shown strong potential, but they must still address long input sequences, large label spaces, and domain mismatch between general pretraining corpora and clinical text @ref10 . In this context, TransICD is especially relevant because it combines a Transformer encoder with code-wise attention and an imbalance-aware objective @ref11 @ref13 .

 The benchmark landscape has also evolved. Earlier work relied mainly on MIMIC-III with ICD-9 labels @ref1 while more recent work has shifted toward MIMIC-IV and ICD-10, which better reflect current coding practice but also introduce a larger and sparser label space @ref2 @ref12 . Recent studies increasingly incorporate label descriptions, code synonyms, calibration, and stronger explanation mechanisms @ref18 @ref19 @ref20. This trend suggests that future ICD systems should combine strong document modeling with richer label semantics and better evidence localization.

 The Vietnamese setting differs substantially from the English high-resource setting in which most ICD coding models have been developed. To our knowledge, there is no widely used public Vietnamese dataset of discharge summaries or electronic medical records paired with gold ICD-10 labels at a scale suitable for fully supervised ICD coding. Existing Vietnamese medical NLP resources are valuable, but they are distributed across tasks such as named entity recognition and medical question answering rather than note-level ICD coding @ref25 @ref26 .

 This gap has two implications. First, it prevents a straightforward transfer of the standard English evaluation setup to Vietnamese. Second, it raises a methodological question about what can be claimed responsibly when note-level gold labels are not available. In this situation, a realistic study should not imitate the reporting style of a fully supervised benchmark paper. Instead, it should make the data constraints explicit, define the scope conservatively, and design the evaluation so that the reader can distinguish what has been directly observed from what remains provisional.

 This paper therefore studies Vietnamese ICD-10 coding under a low-resource public-data setting. Instead of assuming access to private hospital data, we combine public Vietnamese medical text with an ICD-10 ontology derived from official Vietnamese coding resources @ref27 . Because expert note-level ICD labels are not publicly available, we construct a silver-labeled dataset through weak supervision and define the task at two practical granularities: chapter prediction and 3-character ICD-10 code prediction. The goal is not to claim hospital-ready performance, but to establish a reproducible and credible foundation for future work in Vietnamese ICD coding.

 The paper makes three contributions. First, it defines a transparent data construction pipeline for creating a silver-labeled Vietnamese ICD benchmark from public resources. Second, it adapts a TransICD-style label-aware architecture to a Vietnamese setting in which both the text data and the supervision signal are substantially weaker than in English benchmark studies. Third, it proposes an evaluation framework that combines standard quantitative metrics with qualitative evidence inspection so that model behavior can be interpreted in light of weak supervision noise rather than treated as a black-box outcome.

 The remainder of the paper is organized as follows. Section 2 reviews automatic ICD coding, ontology-guided weak supervision, and Vietnamese medical NLP resources. Section 3 presents the proposed data construction and model adaptation pipeline. Section 4 describes the experimental protocol, baselines, and evaluation strategy. Section 5 presents the reporting structure for results and discusses validity. Section 6 concludes the paper.

 = Related Work

 == Automatic ICD Coding

 Automatic ICD coding has a long history in clinical NLP. Early systems relied on rules, dictionaries, sparse lexical features, and traditional classifiers, but these approaches struggled with long documents and large label spaces @ref3 . The shift toward deep learning enabled stronger document encoders and more flexible multi-label prediction, especially on MIMIC-style datasets @ref4 @ref5 .

 A major turning point came with label-aware models. CAML introduced a convolutional architecture with per-label attention, showing that a separate document representation for each code improves both prediction quality and interpretability @ref4 . MultiResCNN further improved text representation with residual multi-filter convolution @ref5 . LAAT proposed a label-attention mechanism that better handles variable evidence span lengths and infrequent codes @ref6 while HyperCore explicitly modeled code hierarchy and co-occurrence @ref7 . HLAN and HiLAT emphasized hierarchical attention and explainability, showing that it is useful to localize label-specific evidence rather than relying only on pooled document embeddings @ref8 @ref9 .

 The introduction of pretrained language models created another major shift. PLM-ICD showed that fine-tuning pretrained encoders is not enough unless the model also handles long sequences, large label spaces, and domain mismatch @ref10 . TransICD addressed these issues through a Transformer backbone followed by code-wise attention and label-distribution-aware optimization, making it especially relevant to imbalanced and explanation-sensitive settings @ref11 @ref13 . More recent work also highlights the importance of label semantics, synonyms, calibration, and entity-centered compression @ref18 @ref19 @ref20.

 A recurring pattern across this literature is that strong ICD coding systems rarely depend on document encoding alone. As the field has matured, the best-performing methods increasingly combine text modeling with some form of label structure, whether through hierarchy, label descriptions, synonym expansion, or evidence localization. This is particularly relevant to the present work because the Vietnamese setting is not only low-resource but also weakly supervised. Under those conditions, architectures that can expose the relationship between a predicted code and the supporting text are more useful than models that only optimize aggregate performance.

 Another point that emerges from prior work is that interpretability in ICD coding is not merely an auxiliary feature. In many studies, explanation quality matters because the predicted codes correspond to clinical concepts that affect documentation, reporting, and downstream analysis @ref4 @ref8 @ref9 @ref11 . For this reason, code-wise attention remains attractive even when newer encoder backbones are available. It provides a practical compromise between expressive document modeling and evidence localization, which is especially valuable in settings where supervision is noisy and full trust in the predicted label set would be inappropriate.

 == Ontology-Guided and Weakly Supervised Medical NLP

 Weak supervision is increasingly used when gold medical labels are expensive or unavailable. Snorkel formalized the idea of generating training labels programmatically from multiple imperfect supervision sources rather than relying only on manual annotation @ref21 . In biomedical and clinical NLP, ontologies are particularly useful because they already encode normalized concepts, synonyms, hierarchies, and textual definitions @ref22 . Ontology-driven supervision can therefore support mention normalization, entity linking, candidate generation, and document-level pseudo-label construction.

 This line of work is especially relevant for ICD coding in low-resource languages. Unlike ordinary topic classification, ICD coding is tied to a structured coding system whose hierarchy and descriptions can be used directly. Fries et al. showed that ontology-driven weak supervision can support clinical entity classification in electronic health records when multiple label sources are combined carefully @ref22 . In our setting, weak supervision is useful not only for creating labels but also for retaining evidence spans and confidence scores, which later support explanation analysis and confidence-aware training.

 In practice, ontology-guided weak supervision is attractive because it provides more than a shortcut for labeling. It also gives the researcher a structured way to reason about why a label was proposed. If a candidate code is supported by an exact lexical match, a hierarchy-consistent parent code, and a relevant ontology description, the resulting label is easier to audit than a pseudo-label produced by an opaque heuristic. This property is important for low-resource medical NLP, where the credibility of the supervision pipeline is often as important as the raw size of the constructed dataset.

 At the same time, ontology-based supervision has clear limits. It favors labels that can be recovered from lexical or semantic overlap with the ontology and may underrepresent diagnoses that require broader clinical reasoning. The present study therefore uses ontology supervision as a pragmatic starting point rather than as a substitute for expert annotation. The central idea is that a carefully designed silver benchmark can still support useful model comparison, provided that the study remains explicit about the difference between silver labels and coder-assigned ground truth.

 == Vietnamese Medical NLP Resources

 Vietnamese medical NLP remains under-resourced compared with English. Still, recent work shows encouraging progress. VietMed-NER introduced a spoken named entity recognition dataset for the Vietnamese medical domain @ref25 . ViMedAQA provided a Vietnamese medical abstractive question-answering dataset and further expanded the public resource landscape @ref26 . Although these datasets are not ICD coding benchmarks, they contain disease, symptom, and treatment language that can support mention extraction, terminology discovery, and domain adaptation.

 The main gap is that public Vietnamese medical NLP datasets generally do not provide document-level ICD-10 labels aligned with discharge-summary-style notes. This gap motivates the design choice in this paper. Rather than assuming such a benchmark already exists, we construct a reproducible silver-labeled benchmark from public Vietnamese medical text and official ICD-10 resources. This produces a more conservative and more credible research setting that better matches the actual availability of data.

 This resource situation also affects model selection. In a high-resource setting, the main question might be which architecture extracts the best performance from a fixed benchmark. In the Vietnamese setting, a more basic question comes first: how can the task itself be formulated so that the dataset, the supervision mechanism, and the evaluation protocol remain consistent with what public resources actually support? The present work addresses that earlier stage. It treats benchmark construction as part of the contribution rather than as a preliminary step hidden behind the experimental section.

 = Methodology

 == Problem Formulation and Scope

 Each document is treated as a Vietnamese medical text instance, and the model predicts one or more ICD labels for that document. We study the task at two levels of granularity, namely chapter prediction and 3-character ICD-10 prediction. Chapter prediction provides a coarser and more stable labeling target, while 3-character prediction is closer to practical coding behavior and therefore serves as the main task of interest.

 This formulation is narrower than full hospital discharge-summary coding. The study should therefore be interpreted as a low-resource feasibility study based on silver labels, not as a deployment-ready hospital coding system. This distinction is important because the paper aims to build a credible benchmarking setup under public-data constraints, not to imply that the resulting system has already reached the reliability required for routine clinical use.

 == Overview of the Pipeline

 The proposed pipeline contains five stages: ICD-10 ontology construction from official Vietnamese resources; public Vietnamese medical text collection and normalization; mention extraction and terminology normalization; ontology-based candidate generation and weak supervision aggregation; and TransICD-style model adaptation and evaluation on the resulting silver-labeled dataset.

 Conceptually, the workflow moves from raw public Vietnamese medical text to normalized mentions, then to ontology-linked candidates, then to silver labels, and finally to label-aware neural training. This ordering is important because each stage reduces ambiguity before the next one begins. The text collection stage defines the documents, the mention extraction stage identifies medically relevant spans, the candidate generation stage converts those spans into label hypotheses, and the weak supervision stage turns those hypotheses into a document-level training signal.

 The main methodological novelty lies in the first four stages, where the paper constructs the benchmark itself rather than assuming that an annotated benchmark already exists. The model adaptation stage remains important, but it is meaningful only because the preceding stages establish a coherent and reproducible task definition.

 == Stage 1: ICD-10 Ontology Construction from Official Resources

 We begin by constructing a Vietnamese ICD ontology from official resources issued under the Vietnamese Ministry of Health ICD-10 framework @ref27 . For each code entry, we extract and normalize the chapter identifier and title, the block or group identifier, the 3-character ICD code, optional 4-character descendants, the Vietnamese title or description, the English title or description when available, hierarchy links to parent and child codes, and any coding notes or instruction text.

 The ontology is then flattened into a searchable table. Each row contains the code, its canonical Vietnamese name, an optional English name, chapter and block metadata, aliases, and a unified search text field. Retaining bilingual descriptions is important because Vietnamese medical writing often mixes Vietnamese and English terminology, especially for disease names, acronyms, and specialist expressions.

 This ontology serves three functions. First, it defines the target label space. Second, it provides semantic text that can be used in retrieval and label representation learning. Third, it supplies structured prior knowledge for weak supervision. In other words, the ontology is not only a reference dictionary. It is the central object that aligns the label inventory, the retrieval layer, and the explanation layer within one consistent representation.

 A further advantage of this design is that it keeps the label space auditable. When a code appears in the dataset or in the model output, it can be traced back to a normalized entry in the ontology rather than to an ad hoc label string extracted from the source text. This reduces ambiguity and makes later error analysis more interpretable, especially when different sources use slightly different disease expressions for the same underlying condition.

 == Stage 2: Public Vietnamese Medical Text Collection

 Because a public Vietnamese discharge-summary-to-ICD dataset is not available, we build the document collection from heterogeneous public Vietnamese medical resources. These resources are selected because they contain medically meaningful language, diagnosis-related expressions, entity annotations that support normalization, or other properties that make them useful under public-data constraints. Examples include spoken medical NER corpora, medical QA corpora, public health text collections, and medical domain transcripts or summaries @ref25 @ref26 .

 Each source is converted into a unified document schema with at least the fields `doc_id`, `source`, `raw_text`, `normalized_text`, `mention_list`, `candidate_codes`, `silver_labels`, `evidence_spans`, and `confidence_score`. The purpose of this schema is not only organizational convenience. It ensures that data from heterogeneous resources can be processed through the same downstream pipeline without losing traceability about source origin or supervision confidence.

 The goal of this stage is not to claim equivalence with hospital discharge summaries. Rather, it is to build a reproducible pool of medically meaningful Vietnamese text from which candidate ICD evidence can be extracted. This matters because the absence of public discharge summaries should not force the research problem to disappear. A weaker but transparent benchmark is more useful than an implicit assumption that realistic data will somehow become available later.

 The collection stage also creates an opportunity to study generalization across source types. Because the documents are not all drawn from one uniform corpus, the dataset naturally includes differences in style, density, and terminology. While this heterogeneity introduces noise, it also helps test whether the pipeline remains useful when the linguistic surface form changes. That property is desirable in an early benchmark, provided that the paper remains explicit about the gap between public medical text and real hospital documentation.

 == Stage 3: Mention Extraction and Text Normalization

 The third stage transforms raw Vietnamese medical text into structured mention candidates that can be linked to ICD labels. We first standardize Unicode, punctuation, whitespace, and common formatting artifacts. We then apply a hybrid mention extraction procedure that combines direct use of disease or symptom annotations when available, lexicon matching from ontology-derived terminology, heuristic phrase extraction for diagnosis-like spans, and normalization of abbreviations and orthographic variants.

 This stage is especially important in Vietnamese because medical phrases may appear in abbreviated form, mixed Vietnamese-English form, or spelling variants. To reduce fragmentation, we map extracted mention strings to canonical forms before ontology linking. The output of this stage is a set of extracted mention spans for each document, and each span is associated with a normalized form, a mention type, and an extraction confidence score.

 The extraction stage is intentionally hybrid rather than purely learned or purely rule-based. A learned extractor would be difficult to justify without sufficient labeled data, while a purely lexical pipeline would be too brittle in the presence of abbreviation and paraphrase. By combining multiple mention sources, the pipeline aims to recover a broader range of candidate expressions without pretending that a single extractor is already robust enough for all Vietnamese clinical language.

 This normalization layer also plays an important role in later explainability. If an extracted mention can be mapped to a stable canonical form before ontology linking, then the downstream candidate generation stage becomes easier to inspect. It becomes possible to ask whether a wrong label came from a poor mention, a weak candidate ranking, or an overly permissive supervision threshold, rather than collapsing all errors into one opaque failure mode.

 == Stage 4: Ontology-Based Candidate Generation

 Each extracted mention is linked to a set of candidate ICD codes through multiple channels, including exact string matching against Vietnamese ontology titles and aliases, normalized matching after preprocessing, fuzzy string matching for spelling variation, BM25 retrieval over Vietnamese and English code descriptions @ref23 hierarchy-aware backoff from 4-character codes to 3-character codes and chapters, and optional encoder-based reranking using label-description embeddings.

 Using multiple channels is important because exact matching is precise but brittle, while retrieval and fuzzy matching improve recall at the cost of more noise. Rather than selecting only one strategy, we retain per-channel scores and combine them later in a weak supervision layer. This design treats candidate generation as a recall-oriented stage and postpones the final decision until more evidence can be aggregated.

 The hierarchy-aware backoff mechanism is especially useful for low-resource settings. Even when a fine-grained code cannot be linked with high confidence, the pipeline may still recover a plausible 3-character code or chapter-level category. This allows the benchmark to support evaluation at multiple granularities and reduces the chance that partially useful evidence is discarded too early in the pipeline.

 == Stage 5: Weak Supervision and Silver Label Construction

 Because expert note-level ICD labels are not available, we construct document-level silver labels through weak supervision. For each document and candidate ICD code, we combine five signals: exact lexical matching, fuzzy matching, retrieval relevance, hierarchy consistency within the ICD ontology, and document-level mention support. A candidate code is promoted to a silver positive label when its aggregated support exceeds a predefined threshold. This threshold may vary by label granularity and can be calibrated on a manually reviewed development subset when available.

 For each accepted silver label, we retain the supporting mention spans, the channel-wise support signals, the overall confidence score, and the matched ICD title or description. This design allows the resulting dataset to support both supervised learning and later explanation analysis. It also makes it possible to inspect the supervision process after model training, which is important because errors may originate in the silver-label pipeline rather than in the model itself.

 A key principle of this stage is to preserve provenance rather than only final labels. In a fully supervised benchmark, one can usually assume that the observed labels are the ground truth. In the present setting, that assumption would be inappropriate. By keeping the support signals attached to each silver label, the dataset remains usable for confidence-aware training, qualitative review, and later manual auditing. This makes the benchmark more informative even when its labels are imperfect.

 == Model Adaptation: TransICD for Vietnamese

 To preserve the main idea of TransICD while adapting it to Vietnamese, we use a Vietnamese-capable pretrained encoder such as SEA-LION-ModernBERT-300M @ref28 together with a TransICD-style code-wise attention head @ref11 . The encoder first produces contextualized token representations for the input document. For each ICD code, the model learns a separate query that highlights the tokens most relevant to that code. These code-specific weights are then used to build a code-specific document representation, and a sigmoid classifier estimates the probability of that label.

 This design has two advantages. First, it allows the model to focus on different evidence spans for different codes. Second, it supports straightforward explanation through token-level evidence. To strengthen label semantics, the model can also use ICD descriptions, bilingual label text, and synonym expansions @ref18 @ref19 .

 The choice of a label-aware architecture is particularly appropriate here because the benchmark is constructed from weak supervision rather than from expert coding judgments. A pooled document representation can still work as a baseline, but it provides limited insight into which part of the text supports a given code. By contrast, code-wise attention creates a more direct link between the predicted label, the relevant tokens, and the ontology description, which is valuable both for interpretation and for error analysis.

 == Optimization and Training Strategy

 The model is trained as a multi-label classifier. Because the label distribution is highly imbalanced, we consider two losses, namely weighted binary cross-entropy as a stable baseline and LDAM-style margin-aware loss as an imbalance-sensitive alternative @ref13 . In the silver-label setting, each training example can also be weighted by its weak-supervision confidence so that low-confidence silver labels contribute less to the overall loss.

 Long texts are handled through sliding-window chunking or long-context encoding, depending on the backbone. When a document exceeds the model's sequence limit, chunk-level representations are concatenated or pooled before code-wise attention is applied. This choice is motivated by the long-document nature of ICD coding @ref10 @ref17 .

 The training strategy is designed to remain practical under limited hardware conditions. Since the present work is intended as a reproducible public-data study, the computational setup should be realistic enough for follow-up work by other researchers. This is one reason to keep the optimization design straightforward. The contribution of the paper lies more in the coherence of the benchmark and modeling pipeline than in aggressive hyperparameter search.

 == Explainability Design

 Explainability is a first-class design goal. The proposed system provides three complementary explanation signals: code-wise attention over token spans, gradient-based attribution such as Integrated Gradients for verification beyond raw attention @ref24 and label-description alignment that shows which ontology phrase most strongly supports a predicted code.

 At inference time, the model can return the predicted ICD label, a confidence score, the top evidence spans in Vietnamese text, the matched or aligned ICD description, and optional attribution heatmaps for tokens. This is especially valuable because the dataset itself is silver-labeled. Explanations make it easier to distinguish genuine model failure from weak-supervision noise.

 The explanation layer also supports a more careful style of analysis. Instead of treating the final metric table as the only outcome, the study can examine whether the model relies on diagnosis-bearing spans, whether it confuses related labels within the same hierarchy, and whether the ontology descriptions improve semantic grounding for rare codes. These questions are important in a low-resource setting because raw performance alone does not show whether the model has learned clinically plausible behavior or merely exploited lexical overlap.

 = Experiments

 == Research Questions

 The experimental design is intended to answer four questions. The first is whether a silver-labeled public-data benchmark can support meaningful Vietnamese ICD prediction experiments at all. The second is whether a TransICD-style code-wise attention model improves over simpler symbolic and neural baselines. The third is how much ontology semantics, bilingual label descriptions, and individual weak supervision components contribute to performance. The fourth is whether the resulting system produces explanations that are clinically plausible and useful for human review.

 These questions are deliberately ordered from basic to specific. Before asking whether one architecture outperforms another, the study must first establish that the benchmark itself supports stable and interpretable experiments. Only after that does it become meaningful to compare modeling choices and explanation behavior.

 == Experimental Tasks

 We evaluate two tasks. The first is chapter prediction, in which the model predicts one or more ICD chapters for each document. This task is coarser, more stable, and less sparse. The second is 3-character ICD-10 prediction, in which the model predicts one or more 3-character ICD-10 codes. This is the primary task because it is closer to actual coding practice while still remaining feasible under the available data conditions.

 If the full 3-character label space proves too sparse, we will also report results on a restricted subset of labels that appear at least a minimum number of times in the training data. Reporting both the full space and a frequency-restricted space is useful because it separates the challenge of long-tail sparsity from the more general question of whether label-aware modeling helps in Vietnamese medical text.

 == Dataset Splits

 Documents are split into train, validation, and test partitions using a 70/10/20 ratio. When possible, the split is source-aware to reduce train-test leakage from near-duplicate corpora, duplicate-aware to avoid repeated examples across splits, and label-aware so that extremely rare labels are not entirely absent from training. A small manually reviewed subset may also be used as a development-only sanity-check set for threshold tuning and qualitative analysis.

 Careful splitting is especially important in this study because the data come from multiple public sources rather than from one standardized benchmark. If near-duplicate or closely related examples appear across splits, the resulting metrics may look stronger than the true level of generalization. The split design therefore plays a methodological role, not just an administrative one.

 == Baselines

 We compare the proposed model against three baseline categories. The first category includes symbolic and retrieval baselines such as exact match over ontology titles and aliases, normalized exact match, fuzzy string matching, and BM25 label retrieval over code descriptions @ref23 . The second category is a generic neural baseline consisting of the same encoder backbone with a pooled document representation and a linear sigmoid classifier. The third category is the proposed model, which uses the same encoder backbone together with TransICD-style code-wise attention and ontology-aware label semantics.

 This comparison isolates the value of label-aware modeling relative to purely symbolic matching and ordinary document classification. It also helps answer a practical question: in a low-resource setting, does the additional complexity of a code-wise attention model provide meaningful value over simpler retrieval and classification baselines, or do lexical methods remain competitive enough to dominate the cost-benefit tradeoff?

 == Implementation Details

 For the current protocol version, we adopt the following configuration. The encoder is SEA-LION-ModernBERT-300M @ref28 and the corresponding model tokenizer is used. The maximum sequence length is 4096 tokens when supported; otherwise the pipeline falls back to chunked processing with 128-token overlap. Optimization uses AdamW, with a learning rate of 0.00002 for the encoder and 0.0001 for task-specific heads. The batch size is 8 documents, with gradient accumulation if needed. Training runs for up to 10 epochs with early stopping, dropout is set to 0.1, weighted BCE is the default loss, and the LDAM variant is reported separately. Threshold selection is tuned on validation data. The hardware target is one or two modern GPUs with mixed precision.

 These values are concrete enough to make the protocol reproducible while remaining flexible for later tuning. The goal is to provide a realistic implementation target rather than to imply that the current hyperparameters are already fully optimized. This distinction is important because the manuscript is intended to define a transparent experimental roadmap, and reproducibility is more valuable here than presenting an artificially narrow set of final hyperparameter choices.

 == Evaluation Metrics

 For 3-character ICD coding, we report Micro-F1, Macro-F1, Macro-AUC, P@\5, and R@\5. For chapter prediction, we report Accuracy, Macro-F1, and Macro-AUC. Because the benchmark is silver-labeled, quantitative evaluation is complemented by qualitative review of sampled predictions.

 The metric choice reflects the structure of the task. Micro-F1 captures overall performance under class imbalance, Macro-F1 and Macro-AUC better reflect behavior on infrequent labels, and P@\5 and R@\5 are useful for understanding the practical quality of top-ranked predictions. The chapter-level metrics provide a more stable view of performance when fine-grained code prediction is still sparse or noisy.

 == Ablation Studies

 We plan ablations that remove bilingual ICD descriptions, fuzzy matching in weak supervision, ontology hierarchy information, and confidence-aware training. We also compare a pooled encoder baseline against code-wise attention and weighted BCE against LDAM. These ablations isolate which parts of the pipeline matter most under low-resource conditions.

 The ablation design is important because the proposed system has several interacting components, and a raw comparison against external baselines would not show which design choices actually contribute to any observed gain. By removing one component at a time, the study can separate the value of ontology structure, bilingual semantics, weak-supervision design, and label-aware modeling.

 == Qualitative Analysis Protocol

 For qualitative analysis, we sample predicted cases from four groups: high-confidence true positives, high-confidence false positives, false negatives on frequent labels, and false negatives on rare labels. For each case, we record the input document, the predicted label or labels, the top evidence spans, the matched ICD description, the attention distribution, and optional Integrated Gradients attribution.

 The goal is to determine whether the model relies on clinically meaningful evidence or only on brittle lexical overlap. This qualitative layer is essential because the benchmark is silver-labeled. If a prediction looks incorrect under the silver labels but is supported by a plausible diagnosis-bearing span and a semantically aligned ontology description, that case may indicate supervision noise rather than model failure. Conversely, a high-confidence prediction with weak or irrelevant supporting spans may reveal a genuine modeling problem even if the metric does not make it obvious.

 == Reproducibility Considerations

 Because the benchmark is assembled from multiple public resources, reproducibility depends on more than code release alone. The preprocessing pipeline should therefore preserve source identifiers, normalization decisions, candidate-generation signals, and weak-supervision confidence values at each stage. Versioning these intermediate artifacts makes the benchmark easier to audit and easier to extend in later work.

 This point is especially important in weakly supervised research. Small preprocessing differences can change which candidates survive thresholding and therefore alter both the training signal and the reported evaluation. A reproducible paper should make these dependencies visible rather than treating the dataset as an opaque by-product of one internal script.

 = Results, Discussion, and Validity

 == Main Results

 The quantitative results will be inserted once training and evaluation are complete. The reporting format is shown below for completeness.

 // Table 1. Chapter-level prediction results.
 === Chapter-level prediction

 #table(
  columns: 3,
  table.header([*Model*], [*Macro F1*], [*Macro AUC*]),
  [Exact Match], [TBD], [TBD],
  [Fuzzy Match], [TBD], [TBD],
  [BM25], [TBD], [TBD],
  [Encoder + Linear], [TBD], [TBD],
  [TransICD-Vi], [TBD], [TBD],
 )

 // Table 2. Results for 3-character ICD-10 prediction.
 === 3-character ICD-10 prediction

 #table(
  columns: 6,
  table.header([*Model*], [*Micro F1*], [*Macro F1*], [*Macro AUC*], [*P@\5*], [*R@\5*]),
  [Exact Match], [TBD], [TBD], [TBD], [TBD], [TBD],
  [Fuzzy Match], [TBD], [TBD], [TBD], [TBD], [TBD],
  [BM25], [TBD], [TBD], [TBD], [TBD], [TBD],
  [Encoder + Linear], [TBD], [TBD], [TBD], [TBD], [TBD],
  [TransICD-Vi], [TBD], [TBD], [TBD], [TBD], [TBD],
 )

 == Planned Interpretation

 The experiments are designed to test whether ontology-informed supervision and code-wise attention improve Vietnamese ICD prediction under public-data constraints. In particular, the comparison will show whether symbolic baselines remain competitive when lexical overlap is high, whether a pooled encoder improves robustness to surface variation, and whether code-wise attention provides a better balance between prediction quality and explainability.

 The planned interpretation should remain conservative. If the proposed model outperforms symbolic baselines, the result should be interpreted as evidence that label-aware contextual modeling is useful in this benchmark setting, not as proof of hospital-ready coding quality. If the gains are small, that outcome is also informative, because it would suggest that under current public-data constraints the main bottleneck may lie in supervision quality or label ambiguity rather than in model architecture alone.

 == Explainability Analysis Template

 A representative qualitative case can be reported in the following form:

 Document excerpt: `...`. Predicted label: `J18` (Pneumonia, unspecified organism). Confidence: `TBD`. Top evidence spans: `sốt cao`, `khó thở`, `thâm nhiễm phổi`, `viêm phổi`. Matched ontology description: `Viêm phổi, tác nhân không xác định`. Attribution note: the model focused mainly on diagnosis-bearing spans rather than on general history text.

 Such examples help show whether the model remains clinically plausible in the presence of weak-supervision noise. They also make it easier for the reader to judge whether the explanation signal is genuinely useful or merely decorative.

 == Threats to Validity

 The proposed study has several limitations. Silver labels are not gold labels, so performance on the constructed benchmark should not be interpreted as hospital-grade coding accuracy. The labels are generated from imperfect weak supervision and therefore reflect both task difficulty and supervision noise. Public medical text is also not equivalent to discharge summaries. Some public corpora are conversational, spoken, or consumer-facing rather than strictly clinical narrative notes, which makes the benchmark broader but also less directly comparable to hospital EHR coding.

 Ontology-driven supervision can be lexically biased. A weak supervision pipeline may favor codes that are easy to recover through lexical overlap while missing diagnoses that are only implied. Model transfer also remains uncertain. Even if a multilingual or regional encoder supports Vietnamese well, success on public medical text does not guarantee the same behavior on real hospital data.

 These limitations do not invalidate the study. Rather, they define the appropriate scope of the claim. The goal is to establish a reproducible research foundation, not to overclaim a finished clinical product. In that sense, the manuscript should be judged by whether it states the constraints clearly, constructs the benchmark transparently, and tests the proposed model in a way that remains faithful to the available data.

 == Practical Value Despite Limited Supervision

 Despite these limitations, the study remains useful for several reasons. First, it turns a broad idea into a benchmarkable research problem. Second, it creates a reusable ontology and silver-label pipeline that other researchers can improve. Third, it provides an explainable label-aware architecture that can later be transferred or fine-tuned when stronger Vietnamese ICD datasets become available.

 The practical value also lies in its role as an intermediate step. In under-resourced domains, progress often depends on methods that make imperfect but public resources useful enough to support comparison and iteration. A well-documented silver benchmark can therefore play an important role even if it does not replace expert annotation. It enables more grounded discussion, more reproducible baselines, and a clearer path toward future collaboration with clinical institutions when stronger data become available.

 = Conclusion

 This paper presents a research framework for Vietnamese ICD-10 coding under public-data constraints. Instead of assuming access to private discharge summaries with gold ICD labels, the framework combines public Vietnamese medical text, official ICD-10 resources, and ontology-guided weak supervision to construct a silver-labeled benchmark. On top of this benchmark, the study adapts a TransICD-style model with code-wise attention, label semantics, and imbalance-aware training.

 The main contribution of the current work is a clear and reproducible problem formulation for a setting in which public gold-standard data are not yet available. Equally important, the paper argues that benchmark construction, supervision design, and explanation analysis should be treated as central methodological components rather than as secondary implementation details. This perspective is particularly important in low-resource clinical NLP, where the limits of the data strongly shape what can be claimed about the model.

 After the experiments are completed, the final manuscript should report the quantitative results, qualitative case studies, and error analysis in the structure already defined here. Even in its present form, the study offers a concrete roadmap for future Vietnamese ICD coding research that is more transparent and more realistic than simply assuming the existence of a standard supervised benchmark.

 #bibliography("references.yaml")
]

