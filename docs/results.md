# ClinPGx-QA: Evaluating Large Language Models on Pharmacogenomic Reasoning over Biomedical Literature

## Experimental Results

---

## 1. Experimental Setup

### 1.1 Benchmark Overview

ClinPGx-QA evaluates LLMs on pharmacogenomics (PGx) reasoning tasks grounded in full-length PubMed Central articles. The benchmark comprises six task types of increasing complexity, from binary claim verification to end-to-end paper investigation, spanning over 112,000 generated questions derived from 3,064 full-text articles and PharmGKB annotations.

### 1.2 Models Evaluated

We evaluate models from three major providers spanning a range of capabilities and cost profiles:

| Provider | Model | Tier |
|----------|-------|------|
| Anthropic | Claude Opus 4.6 | Frontier |
| Anthropic | Claude Sonnet 4.5 | Mid-tier |
| Anthropic | Claude Haiku 4.5 | Cost-optimized |
| OpenAI | GPT-5 | Frontier |
| OpenAI | GPT-5.2 | Frontier |
| OpenAI | GPT-4o | Mid-tier |
| OpenAI | GPT-4o-mini | Cost-optimized |

### 1.3 Evaluation Runs

We report results from two evaluation campaigns:

- **Pilot evaluation** (February 13, 2026): 10 questions per model per task across all 6 models, designed to validate the evaluation pipeline and identify preliminary trends.
- **Full-scale evaluation** (February 20, 2026): 10,000 questions per model per task. Completed for Claude Opus 4.6 and GPT-4o-mini on all tasks; Anthropic Sonnet/Haiku on study parameters. MCQ full-scale results for remaining models are pending due to API-level output format failures (see Section 3.5).

### 1.4 Dataset Statistics

| Dataset | Questions | Description |
|---------|-----------|-------------|
| Yes/No Claims | 11,981 | Binary pharmacogenomic claim verification |
| Chained Questions | 36,750 | Multi-turn dependent reasoning chains |
| MCQ (Drug) | ~5,000+ | Identify correct drug from 4 options |
| MCQ (Variant) | ~5,000+ | Identify correct genetic variant from 4 options |
| MCQ (Phenotype) | ~5,000+ | Identify correct phenotype category from 4 options |
| Study Parameters | 46,823 | Extract p-values and statistical significance |
| Variant Extraction | 2,922 | Extract all PGx variants from full-text articles |

Questions are derived from PharmGKB annotations (12,798 variant-drug, 14,312 variant-phenotype, 2,123 functional annotations, 35,555 study metadata records) cross-referenced against 32 expert-annotated ground-truth papers.

---

## 2. Main Results

### 2.1 Multiple-Choice Question Accuracy (Pilot, n=10)

Table 1 reports accuracy on MCQ tasks across three entity types (drug, variant, phenotype) and study parameter extraction, evaluated on a 10-question pilot subset.

**Table 1.** MCQ and study parameter accuracy (%, pilot, n=10 per cell).

| Model | Drug MCQ | Variant MCQ | Phenotype MCQ | Study Param |
|-------|:--------:|:-----------:|:-------------:|:-----------:|
| Claude Opus 4.6 | **100** | 80 | **70** | 10 |
| Claude Sonnet 4.5 | **100** | 80 | 60 | 40 |
| Claude Haiku 4.5 | **100** | 60 | 40 | 50 |
| GPT-4o | **100** | **100** | 50 | 40 |
| GPT-5 | 90 | **100** | 60 | 50 |
| GPT-5.2 | **100** | **100** | 60 | 50 |

**Key observations.** Drug MCQ is effectively saturated: all models achieve 90--100%, reflecting the explicit surface-level availability of drug names in article text and tables. Phenotype MCQ exhibits the widest performance spread (40--70%) and serves as the most discriminative task. Variant MCQ reveals a consistent gap between OpenAI models (100%) and Anthropic models (60--80%), driven entirely by "None of the Above" (NOTA) question failures (Section 3.1).

### 2.2 Study Parameter Extraction at Scale (n=10,000)

Table 2 reports study parameter extraction results on the full 10,000-question evaluation, broken down by question type: *correct* (real associations), *modified\_drug* (hallucination test with non-existent drug), and *modified\_variant* (hallucination test with non-existent variant).

**Table 2.** Study parameter extraction accuracy (%, n=10,000). "Both" requires correct p-value *and* significance judgment.

| Model | P-value | Significance | Both Correct | Parse Failures |
|-------|:-------:|:------------:|:------------:|:--------------:|
| GPT-4o-mini | 59.3 | 60.1 | **54.5** | 0.0% |
| Claude Opus 4.6 | 7.3 | 8.2 | 6.4 | 83.9% |
| Claude Sonnet 4.5 | 0.0 | 0.0 | 0.0 | 100.0% |
| Claude Haiku 4.5 | 0.0 | 0.0 | 0.0 | 100.0% |

**Table 3.** Study parameter accuracy by question type (GPT-4o-mini, n=10,000).

| Question Type | P-value | Significance | Both Correct | N |
|---------------|:-------:|:------------:|:------------:|:---:|
| Correct (real association) | 33.7 | 35.6 | 19.2 | 3,334 |
| Modified drug (hallucination test) | 72.7 | 72.9 | 72.7 | 3,333 |
| Modified variant (hallucination test) | 71.5 | 71.9 | 71.5 | 3,333 |

**Table 4.** Study parameter accuracy by question type (Claude Opus 4.6, n=10,000).

| Question Type | P-value | Significance | Both Correct | N |
|---------------|:-------:|:------------:|:------------:|:---:|
| Correct (real association) | 8.8 | 11.5 | 6.3 | 3,334 |
| Modified drug (hallucination test) | 5.8 | 5.7 | 5.7 | 3,333 |
| Modified variant (hallucination test) | 7.4 | 7.3 | 7.3 | 3,333 |

GPT-4o-mini demonstrates a striking asymmetry: it achieves ~72% accuracy on hallucination test questions (correctly returning "not found") but only 19.2% on real associations. Opus's low headline numbers are dominated by JSON parse failures (83.9%), discussed in Section 3.5.

### 2.3 Yes/No Claim Verification (GPT-4o, n=5,000)

Table 5 reports binary claim verification accuracy, stratified by claim manipulation type.

**Table 5.** Yes/No claim verification accuracy (GPT-4o, n=5,000).

| Claim Type | Accuracy (%) |
|------------|:------------:|
| Original claims (unmodified) | 62.3 |
| Association flips (wrong entity) | 86.2 |
| Direction flips (reversed effect) | 93.9 |
| **Overall** | **79.1** |

Models detect manipulated claims far more reliably (86--94%) than they verify unmodified factual claims (62%). This suggests models are better at recognizing inconsistency with article content than at confirming positive associations---a pattern consistent with anomaly detection being easier than factual grounding.

### 2.4 Variant Extraction (n=5 articles)

Table 6 reports variant extraction recall across entity types (rsIDs, star alleles, HLA alleles, metabolizer phenotypes).

**Table 6.** Variant extraction recall (5 articles). Micro recall weights by total variant count; macro recall averages per-article recall.

| Model | Macro Recall | Micro Recall |
|-------|:------------:|:------------:|
| Claude Opus 4.6 | 0.800 | **0.933** |
| GPT-5 | 0.800 | **0.933** |
| GPT-5.2 | 0.800 | **0.933** |
| Claude Sonnet 4.5 | 0.733 | 0.867 |
| Claude Haiku 4.5 | 0.733 | 0.867 |
| GPT-4o | 0.592 | 0.600 |

Frontier models (Opus, GPT-5, GPT-5.2) cluster at 93% micro recall, mid-tier models at 87%, and the weakest model (GPT-4o) drops to 60%. This task primarily tests long-context extraction fidelity and shows clear capability stratification.

### 2.5 Paper Investigation: End-to-End (n=2 articles)

Table 7 reports the composite paper investigation score, which multiplies variant extraction recall by mean question accuracy.

**Table 7.** End-to-end paper investigation scores (2 articles). Paper score = variant recall x mean question accuracy.

| Model | Variant Recall | Question Accuracy | Paper Score |
|-------|:--------------:|:-----------------:|:-----------:|
| Claude Opus 4.6 | **1.000** | 0.616 | **0.609** |
| Claude Haiku 4.5 | 0.833 | 0.674 | 0.583 |
| Claude Sonnet 4.5 | 0.833 | 0.664 | 0.560 |
| GPT-4o-mini | 0.604 | 0.580 | 0.378 |

The best model achieves only 61% on the end-to-end task, demonstrating that errors compound across pipeline stages. Notably, Opus achieves the highest paper score despite lower question accuracy than Haiku and Sonnet, because perfect variant recall (1.000) ensures no downstream questions are missed.

---

## 3. Error Analysis

### 3.1 NOTA Questions Expose Calibration Failures

"None of the Above" (NOTA) variants of MCQ questions replace the correct answer with a distractor and add "None of the options" as the correct choice. These questions test whether models can abstain from answering when the correct entity is absent.

In the pilot evaluation, all Anthropic model failures on variant MCQ cluster exclusively on NOTA questions. When the correct variant is present among options (standard questions), Anthropic models perform comparably to OpenAI models. However, when the correct answer is removed, smaller Anthropic models select a plausible-looking distractor rsID rather than abstaining.

**Example failure (Claude Haiku 4.5):**
> *"\_\_\_\_\_\_ is associated with decreased response to sitagliptin in people with Diabetes Mellitus, Type 2."*
> Options: (a) rs2290910 (b) rs12255372 (c) None of the options (d) rs7903146
> Expected: **(c)** | Model selected: **(a) rs2290910** (a distractor rsID from the same paper)

This pattern---correct on standard questions, incorrect on NOTA variants of the same question---constitutes a *contradiction*, analyzed further in Section 3.2.

### 3.2 Contradiction Analysis

We define contradictions as cases where a model answers a standard MCQ correctly but fails the corresponding NOTA version of the same question (or vice versa), implying logically inconsistent pharmacogenomic associations.

**Table 8.** Contradictions detected in variant MCQ (pilot, 30 question pairs per model).

| Model | Contradictions | Rate |
|-------|:--------------:|:----:|
| Claude Haiku 4.5 | 4 | 13.3% |
| Claude Opus 4.6 | 2 | 6.7% |
| Claude Sonnet 4.5 | 2 | 6.7% |
| GPT-5 | 1 | 3.3% |
| GPT-4o | 0 | 0.0% |
| GPT-5.2 | 0 | 0.0% |

All contradictions concentrate in variant MCQ NOTA questions; no contradictions appear in drug or phenotype tasks. The monotonic relationship between contradiction rate and model capability (Haiku > Sonnet/Opus > GPT-5 > GPT-4o/5.2) suggests that NOTA calibration improves with model scale and training.

### 3.3 Phenotype Classification: Ontological Granularity Failures

Phenotype MCQ is the most discriminative task (40--70% spread) and reveals two distinct failure modes:

**Failure Mode 1: Specificity mismatch.** Models select phenotype terms at a different level of medical specificity than the ground-truth annotation. This occurs in both directions:

- *Too specific:* All 6 models select "hemorrhagic cystitis" when the annotation specifies "Cystitis" (PMC11936550, cyclophosphamide/Fanconi Anemia). The paper explicitly discusses hemorrhagic cystitis, making the models' choice arguably more precise.
- *Too specific:* 3/6 models select "severe neutropenia" vs. annotated "Neutropenia" (PMC11554802, UGT1A1/sacituzumab govitecan).
- *Too broad:* 2/6 models select "Substance-Related Disorders" vs. the more specific "Opioid-Related Disorders" (PMC9261480, ABCB1/loperamide).

These failures highlight a tension between model behavior and annotation conventions: models sometimes make more medically precise choices that are scored as incorrect. This suggests annotation quality and ontological standardization are important confounders in phenotype evaluation.

**Failure Mode 2: Multi-phenotype negation.** Questions asking which phenotypes are *not* associated with a variant require selecting a compound answer listing all non-associated phenotypes. Models anchor on a single clearly non-associated phenotype rather than recognizing that the correct answer encompasses the complete set.

**Example (4/6 models fail):**
> *"HLA-B\*35:08 is NOT associated with \_\_\_\_\_\_ when taking lamotrigine for Epilepsy."*
> Options: (a) Severe Cutaneous Adverse Reactions (b) Maculopapular Exanthema (c) two of three **(d) all three**
> Most models selected (a) or (b)---a single phenotype rather than the complete set.

### 3.4 Study Parameter Extraction: Format vs. Understanding

The low headline accuracy on study parameter extraction (6.4% for Opus, 54.5% for GPT-4o-mini) decomposes into two qualitatively different failure modes:

**Strict string matching inflates error rates.** The evaluation requires exact p-value string match. Models frequently extract the correct numeric value but in a slightly different format:
- Expected: `< 0.001` | Model: `<0.001` (missing space) --- **marked wrong**
- Expected: `= 0.012` | Model: `0.012` (missing equals sign) --- **marked wrong**
- Expected: `< 0.0001` | Model: `<0.0001` (missing space) --- **marked wrong**

When Opus does produce valid JSON on "correct" type questions, its significance judgment accuracy reaches 75%, suggesting the model understands the underlying statistics but is penalized by formatting mismatches. A fuzzy numeric comparison would likely yield substantially higher accuracy.

**Genuine hallucination on modified questions.** Modified questions replace the drug or variant with a non-existent entity, with the expected answer being "not found." GPT-4o-mini correctly identifies non-existent associations ~72% of the time but still hallucinates p-values for ~27% of modified questions, confidently returning statistical values from the paper for associations that were never studied.

**Table 9.** GPT-4o-mini hallucination rates on modified study parameter questions (n=6,666).

| Modification Type | Correct ("not found") | Hallucinated (returned p-value) |
|-------------------|:---------------------:|:-------------------------------:|
| Modified drug | 72.7% | 27.3% |
| Modified variant | 71.5% | 28.5% |

### 3.5 Output Format Compliance as a Systematic Confounder

The most striking result at scale is the near-total failure of Anthropic models on structured output tasks. All three Anthropic models (Opus, Sonnet, Haiku) returned empty or long-form text responses instead of the required JSON format, resulting in 83.9--100% parse failure rates across all MCQ and study parameter tasks at n=10,000.

**Table 10.** JSON/format parse failure rates at scale (n=10,000 per cell).

| Model | MCQ Drug | MCQ Variant | MCQ Phenotype | Study Param |
|-------|:--------:|:-----------:|:-------------:|:-----------:|
| Claude Opus 4.6 | 100% | 100% | 100% | 83.9% |
| Claude Sonnet 4.5 | 100% | 100% | 100% | 100% |
| Claude Haiku 4.5 | 100% | 100% | 100% | 100% |
| GPT-4o-mini | --- | --- | --- | 0.0% |

Inspection of Opus's study parameter responses reveals that the model often *correctly identifies* trick questions (modified drug/variant) and writes detailed reasoning about why the association doesn't exist in the paper---but does so in free text rather than the required JSON schema. This represents a case where better reasoning leads to worse scores: the model's correct identification of the trick triggers a verbose explanation that violates the output format constraint.

This finding has implications for benchmark design: structured output compliance is a confound that can mask genuine reasoning capability, and future evaluations should consider separating format adherence from task understanding.

### 3.6 Asymmetry Between Verification and Grounding

The yes/no claim verification results (Table 5) reveal a consistent asymmetry: models detect manipulated claims (86--94% on flips) far more accurately than they verify unmodified claims (62%). This suggests models are better at detecting *inconsistency* between a claim and article content than at *confirming* that a positive claim is supported.

This has clinical implications: a model deployed for pharmacogenomic literature review would be more reliable at flagging incorrect claims than at confirming correct ones---a useful property for safety-critical applications, but insufficient for autonomous evidence synthesis.

### 3.7 Error Compounding in End-to-End Evaluation

The paper investigation task (Table 7) demonstrates how component-level errors compound in a multi-stage pipeline. Even the best model (Claude Opus 4.6) achieves only 61% on the composite score, despite near-perfect variant extraction (100% recall) and reasonable question accuracy (62%).

The multiplicative scoring formula (variant recall x question accuracy) means that a model with 90% variant recall and 70% question accuracy scores only 63%---losses at any stage propagate through the pipeline. This mirrors real-world deployment scenarios where pharmacogenomic evidence synthesis requires sequential extraction and reasoning steps.

---

## 4. Discussion

### 4.1 Task Difficulty Hierarchy

Our results establish a clear difficulty hierarchy across ClinPGx-QA tasks:

1. **Drug MCQ** (solved, ~100%): Drug names are explicitly stated in article text; this task tests surface-level extraction only.
2. **Variant extraction** (strong, 87--93% for mid-to-frontier models): Requires scanning full-length articles for diverse entity formats (rsIDs, star alleles, HLA alleles).
3. **Yes/No claim verification** (moderate, 79% overall): Requires grounding claims against article content; performance degrades on positive claims.
4. **Variant MCQ** (moderate, 60--100%): Discriminates on NOTA calibration; frontier OpenAI models solve this while Anthropic models show calibration gaps.
5. **Phenotype MCQ** (challenging, 40--70%): Requires ontological reasoning about medical term granularity and multi-entity negation.
6. **Study parameter extraction** (challenging, 19% on real associations): Requires precise numeric extraction from tables and statistical reasoning.
7. **End-to-end paper investigation** (hardest, max 61%): Compounds extraction and reasoning errors across pipeline stages.

### 4.2 Implications for Clinical Deployment

The results suggest that current LLMs can reliably perform *extraction* tasks in pharmacogenomics (drug identification, variant extraction) but struggle with tasks requiring *reasoning* (phenotype classification, statistical interpretation) and *calibration* (NOTA questions, hallucination detection).

For clinical pharmacogenomics applications, this means:
- LLMs can serve as effective first-pass extractors for variant-drug associations from literature.
- Human oversight remains essential for phenotype classification, statistical interpretation, and any task where abstention is the correct response.
- The ~27% hallucination rate on non-existent associations (Table 9) is clinically unacceptable and represents the most concerning failure mode for patient safety.

### 4.3 Limitations

- **Sample sizes**: Pilot results (n=10) should be interpreted cautiously; full-scale results are available only for a subset of model-task combinations.
- **Format compliance confound**: Anthropic models' parse failures at scale prevent direct comparison with GPT-4o-mini on study parameter and MCQ tasks. This is an evaluation infrastructure limitation, not necessarily a model capability limitation.
- **Annotation quality**: Universal failures on specific questions (e.g., cystitis specificity mismatch) suggest some ground-truth labels may not reflect the medically precise answer, inflating apparent error rates.
- **P-value string matching**: The strict string comparison for p-values penalizes format variation rather than numerical understanding. Future evaluations should implement fuzzy numeric matching with tolerance.

---

## 5. Summary of Key Findings

1. **Drug identification is saturated** across all models (90--100%), providing minimal signal for model differentiation.
2. **Phenotype classification is the most discriminative MCQ task** (30-point spread), with failures driven by ontological granularity mismatches and multi-entity negation.
3. **NOTA questions are a key calibration discriminator**: smaller models guess rather than abstain, producing logically inconsistent answers across question variants.
4. **Study parameter extraction reveals a format-vs-understanding gap**: models extract correct values in wrong formats, and hallucinate statistics for ~27% of non-existent associations.
5. **Output format compliance is a systematic confounder** at scale, with Anthropic models showing 84--100% parse failure rates that mask underlying reasoning quality.
6. **Models detect claim inconsistency better than they confirm truth** (94% on direction flips vs. 62% on original claims).
7. **End-to-end performance caps at 61%**, demonstrating that component-level accuracy is insufficient when errors compound across extraction and reasoning stages.
