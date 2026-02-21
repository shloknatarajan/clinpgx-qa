# ClinPGx-QA: Benchmark Overview

ClinPGx-QA is a benchmark for evaluating LLMs on pharmacogenomics (PGx) reasoning over full-length PubMed Central articles. It tests whether models can extract variant-drug associations, interpret study parameters, and perform multi-step evidence-grounded reasoning from biomedical literature.

## Datasets

### Source Data

| Source | Count | Description |
|--------|-------|-------------|
| PubMed Central articles | 3,064 | Full-text markdown-converted papers |
| Variant-drug annotations | 12,798 | PharmGKB variant-drug associations |
| Variant-phenotype annotations | 14,312 | PharmGKB variant-phenotype associations |
| Functional annotations | 2,123 | PharmGKB functional/allele annotations |
| Study parameters | 35,555 | Study-level metadata records |
| Expert annotations | 32 | Hand-annotated ground-truth papers |

### Generated Question Datasets

| Dataset | File | Questions | Description |
|---------|------|-----------|-------------|
| Yes/No Claims | `yes_no_questions.jsonl` | 11,981 | True/false pharmacogenomic claims |
| Chained Questions | `chained_questions.jsonl` | 36,750 | Multi-turn dependent question chains |
| Chained (Curated) | `neurips_chained_questions.jsonl` | 100 | Curated subset for evaluation |
| MCQ Drug | `mcq_options/drug_mcq_options.jsonl` | ~5,000+ | Identify correct drug from options |
| MCQ Variant | `mcq_options/variant_mcq_options.jsonl` | ~5,000+ | Identify correct variant from options |
| MCQ Phenotype | `mcq_options/phenotype_mcq_options.jsonl` | ~5,000+ | Identify correct phenotype category |
| Study Parameters | `study_param_questions.jsonl` | 46,823 | Extract p-values and significance |
| Variant Extraction | `variant_bench.jsonl` | 2,922 | Extract variants from articles |
| Sentence Benchmark | `sentence_bench.jsonl` | 179 | Sentence-level ground truth |

---

## Task Descriptions

### Task 1: Yes/No Claim Verification

Binary classification of pharmacogenomic claims against full-text articles. Includes original claims, association flips (wrong entity), and direction flips (reversed effect).

**Example:**
> Based on PMC11730665, is this statement true or false: Genotype TT is associated with decreased response to sitagliptin in people with Diabetes Mellitus, Type 2.
>
> **Answer:** true

### Task 2: Chained Multi-Turn Reasoning

Dependent multi-turn chains that test claim verification, statistical extraction, and evidence evaluation in sequence.

**Example (3-turn chain):**
> **Turn 1** (claim verification): Based on PMID 15634941, is the following claim supported: CYP3A4 \*17 is associated with decreased metabolism of nifedipine as compared to CYP3A4 \*1. &rarr; **true**
>
> **Turn 2** (statistical extraction): What is the p-value for this association? &rarr; **= 0.05**
>
> **Turn 3** (evidence evaluation): Does the p-value indicate statistical significance? &rarr; **false**

Reasoning types tested: claim verification, evidence provenance, numeric extraction, objective statistical reasoning, negation scope, counterfactual reasoning.

### Task 3: Multiple-Choice Questions

Identify the correct entity (drug, variant, or phenotype) from a set of options, including "none of the above" variants.

**Example (Drug MCQ):**
> Genotype TT is associated with decreased response to ______ in people with Diabetes Mellitus, Type 2.
>
> (a) saxagliptin (b) linagliptin (c) vildagliptin **(d) sitagliptin**

### Task 4: Study Parameter Extraction

Extract p-values and statistical significance judgments from papers. Includes correct associations, modified-variant, and modified-drug question types.

**Example:**
> For the association between rs2909451 and sitagliptin efficacy in PMC11730665, what is the p-value?
>
> **Answer:** < 0.001, significant: yes

### Task 5: Variant Extraction

Extract all pharmacogenomic variants (rsIDs, star alleles, HLA alleles, metabolizer phenotypes) from full-text articles.

**Example:**
> Article PMC11730665 &rarr; Extract: rs1799853, rs4664443, rs7754840, rs3765467, rs2285676, rs2909451, rs6923761, rs163184

### Task 6: Paper Investigation (End-to-End)

Composite evaluation: extract variants from a paper, then answer MCQ and study parameter questions only for successfully extracted variants.

**Scoring:** `paper_score = variant_recall x mean(question_accuracy)`

---

## Model Performance

### Models Evaluated

- **OpenAI:** GPT-5, GPT-5.2, GPT-4o, GPT-4o-mini
- **Anthropic:** Claude Opus 4.6, Claude Sonnet 4.5, Claude Haiku 4.5
- **Google:** Gemini 2.5 Pro

### MCQ Results (Pilot — 10 questions each)

| Model | Drug | Variant | Phenotype | Study Param |
|-------|------|---------|-----------|-------------|
| GPT-5 | 100% | 100% | 50% | 70% |
| GPT-5.2 | 100% | 100% | 60% | 70% |
| Claude Opus 4.6 | 100% | 80% | 70% | 70% |
| Claude Sonnet 4.5 | 100% | 60% | 60% | 80% |
| Claude Haiku 4.5 | 100% | 70% | 40% | 70% |
| GPT-4o-mini | 100% | 50% | 60% | 80% |

**Pilot observations (n=10 per cell):**
- Drug MCQ is trivially easy — all models ~100%.
- Phenotype MCQ is the hardest differentiator (40–70%).
- Variant MCQ: OpenAI models outperform Anthropic, largely due to "None of the Above" (NOTA) questions.
- Study param accuracy is artificially low due to strict p-value string matching (e.g., `< 0.001` vs `<0.001`).

### MCQ Full Run (10,000 questions each) — In Progress

Run: `runs/20260220_025035_mc_study_all_models/`

**Study Parameter Extraction (10,000 questions)**

| Model | P-value Acc | Significance Acc | Both Correct | Parse Failures |
|-------|-------------|------------------|--------------|----------------|
| Claude Opus 4.6 | 7.3% | 8.2% | 6.4% | 83.9% |
| GPT-4o-mini | 59.3% | 60.1% | 54.5% | 0.0% |

**Breakdown by question type (GPT-4o-mini):**

| Question Type | P-value | Significance | Both | N |
|---------------|---------|--------------|------|---|
| correct (real association) | 33.7% | 35.6% | 19.2% | 3,334 |
| modified_drug (hallucination test) | 72.7% | 72.9% | 72.7% | 3,333 |
| modified_variant (hallucination test) | 71.5% | 71.9% | 71.5% | 3,333 |

Key insight: Models score much higher on modified (non-existent) associations by correctly returning "not found" (~72%), but struggle with real associations (~19% both-correct). The low "correct" type accuracy is partly due to strict p-value string matching — models extract the right numeric value but format it differently (e.g., `<0.001` vs `< 0.001`).

**Breakdown by question type (Claude Opus 4.6):**

| Question Type | P-value | Significance | Both | N |
|---------------|---------|--------------|------|---|
| correct | 8.8% | 11.5% | 6.3% | 3,334 |
| modified_drug | 5.8% | 5.7% | 5.7% | 3,333 |
| modified_variant | 7.4% | 7.3% | 7.3% | 3,333 |

Opus had 83.9% JSON parse failures — the model often responded with long-form explanations instead of the required JSON format, especially on modified questions where it detected the mismatch and wrote out its reasoning rather than conforming to the output schema.

**MCQ (Drug/Variant/Phenotype) — pending re-run for Anthropic models**

The Anthropic MCQ pipelines (all 9 combinations of Opus/Sonnet/Haiku × variant/drug/phenotype) returned empty responses due to an API failure during the run. Sonnet and Haiku study_param also affected. GPT-4o-mini and GPT-5.2 MCQ results still in progress.

These need to be re-run once the current OpenAI pipelines complete.

### Yes/No Claim Verification (GPT-4o, 5,000 questions)

| Flip Type | Accuracy |
|-----------|----------|
| Original claims | 62.3% |
| Association flips | 86.2% |
| Direction flips | 93.9% |
| **Overall** | **79.1%** |

### Variant Extraction (5 articles)

| Model | Macro Recall | Micro Recall |
|-------|--------------|--------------|
| Claude Opus 4.6 | 0.800 | 0.933 |
| GPT-5 | 0.800 | 0.933 |
| GPT-5.2 | 0.800 | 0.933 |
| Claude Sonnet 4.5 | 0.733 | 0.867 |
| Claude Haiku 4.5 | 0.733 | 0.867 |
| GPT-4o | 0.592 | 0.600 |

### Paper Investigation (End-to-End, 2 papers)

| Model | Variant Recall | Question Acc | Paper Score |
|-------|---------------|--------------|-------------|
| Claude Opus 4.6 | 1.000 | 0.616 | 0.609 |
| Claude Haiku 4.5 | 0.833 | 0.674 | 0.583 |
| Claude Sonnet 4.5 | 0.833 | 0.664 | 0.560 |
| GPT-4o-mini | 0.604 | 0.580 | 0.378 |

---

## Key Findings

1. **Drug identification is largely solved** -- most models reach 100% on drug MCQs (pilot, n=10).
2. **Phenotype classification remains challenging** -- accuracy ranges 40–70% (pilot, n=10), driven by specificity mismatches (e.g., "hemorrhagic cystitis" vs "Cystitis") and multi-phenotype negation questions.
3. **"None of the above" variants expose calibration gaps** -- smaller Anthropic models guess rather than abstain. NOTA questions account for most contradictions.
4. **Study parameter extraction: format vs understanding** -- at scale (n=10K), GPT-4o-mini reaches 54.5% "both correct" overall, but only 19.2% on real associations vs 72% on hallucination tests (correctly returning "not found"). P-value string matching inflates error rates.
5. **Opus struggles with output format compliance** -- 84% JSON parse failures on study params at scale, despite correctly reasoning about the questions. Format non-compliance is a major confounder.
6. **Variant extraction is strong for top models** -- ~93% micro recall, but drops to 60% for weaker models.
7. **End-to-end reasoning compounds errors** -- best model caps at ~61% paper score, showing a clear gap between entity identification and integrated reasoning.

---

## Usage

```bash
# Generate question datasets
python yes_no_questions.py
python chained_questions.py

# Evaluate a single model
python src/eval/run.py --model gpt-4o-mini
python src/eval/run.py --dataset variant_extraction --model anthropic/claude-opus-4-6

# Run all models
python run_all_models.py --dataset all --limit 100

# Score existing responses
python src/eval/run.py --score-only --dataset yes_no --responses-path runs/<run_dir>/yes_no_responses.jsonl

# Analyze results across runs
python analyze_results.py
```
