# ClinPGx-QA Benchmark Results Summary

**Run:** `20260213_013820_all_models` (Feb 13, 2026)
**Sample size:** 10 questions per model per task (subset of full dataset)
**Models:** Claude Opus 4.6, Claude Sonnet 4.5, Claude Haiku 4.5, GPT-4o, GPT-5, GPT-5.2

---

## Overall Accuracy

| Model | Drug MCQ | Variant MCQ | Phenotype MCQ | Study Params |
|---|---|---|---|---|
| Claude Opus 4.6 | **100%** | 80% | **70%** | 10% |
| Claude Sonnet 4.5 | **100%** | 80% | 60% | 40% |
| Claude Haiku 4.5 | **100%** | 60% | 40% | 50% |
| GPT-4o | **100%** | **100%** | 50% | 40% |
| GPT-5 | 90% | **100%** | 60% | 50% |
| GPT-5.2 | **100%** | **100%** | 60% | 50% |

**Key takeaway:** Drug identification is solved. Variant identification favors OpenAI models. Phenotype classification is the hardest MCQ task. Study parameter extraction is unreliable across the board, but partly due to evaluation strictness (see below).

---

## Task-by-Task Analysis

### 1. Drug MCQ — Near-perfect (90-100%)

All models ace drug identification. The drugs in pharmacogenomics studies (sitagliptin, lamotrigine, cyclophosphamide, loperamide, sacituzumab govitecan) are explicitly named in paper text and tables, making them straightforward to extract.

**The one failure:** GPT-5 missed 1/10 — the only blemish across 60 drug questions. This task is effectively solved and provides minimal signal for model differentiation.

### 2. Variant MCQ — OpenAI perfect, Anthropic struggles with NOTA (60-100%)

OpenAI models (GPT-4o, GPT-5, GPT-5.2) achieve **100%** on variant identification, while Anthropic models score 60-80%.

**Where Anthropic models fail — "None of the Above" (NOTA) variant questions:**
The failures cluster exclusively on NOTA questions from PMC11730665 (sitagliptin/Type 2 Diabetes). When the correct answer is present among the options (standard questions), Anthropic models do fine. But when the correct variant is removed and replaced with "None of the options," models get tricked into selecting a plausible-looking distractor rsID instead.

Example (Haiku failure):
> *"______ is associated with decreased response to sitagliptin in people with Diabetes Mellitus, Type 2."*
> - Correct: rs2909451 (not among the options → answer should be "None of the options")
> - Haiku picked: rs2290910 (a distractor rsID from the same paper)

This reveals a **calibration gap** — smaller Anthropic models are too eager to commit to an answer rather than acknowledging the correct answer isn't listed. Opus shows fewer of these errors (2 vs. Haiku's 4), suggesting scale helps with abstention.

**Contradiction analysis confirms this:** All contradictions detected are Type 1 (correct on standard, wrong on NOTA version of same question). Haiku has 4 contradictions, Opus 2, Sonnet 2. GPT-4o and GPT-5.2 have 0.

### 3. Phenotype MCQ — The Hardest Task (40-70%)

This is where models diverge most, and failure modes are most interesting.

**Failure Mode 1: Specificity mismatch (universal failure)**
All 6 models fail on the cystitis questions from PMC11936550 (cyclophosphamide / Fanconi Anemia):

> *Question: Which phenotype is associated with this gene-drug pair?*
> - Correct answer: "Side Effect: Cystitis"
> - All models chose: "Side Effect: hemorrhagic cystitis"

The paper specifically discusses *hemorrhagic* cystitis, so models reasonably pick the more specific term. But the annotation uses the broader "Cystitis" label. This is likely an **annotation granularity issue** — the models may actually be making a more medically precise choice. (This accounts for 2 wrong answers per model.)

A similar pattern appears with:
- **Neutropenia vs. severe neutropenia** (PMC11554802, UGT1A1/sacituzumab): 3/6 models pick "severe neutropenia" instead of the annotated "Neutropenia"
- **Substance-Related Disorders vs. Opioid-Related Disorders** (PMC9261480, ABCB1/loperamide): 2/6 models pick the broader category instead of the more specific one

These specificity mismatches go both directions — sometimes models are too specific (hemorrhagic cystitis), sometimes too broad (Substance-Related Disorders). The inconsistency suggests models lack a stable ontological grounding for phenotype classification.

**Failure Mode 2: Multi-phenotype "not associated" questions**
Questions from PMC5712579 (HLA-B alleles / lamotrigine / Epilepsy) ask which phenotypes are *not* associated with a variant. The correct answer lists all three non-associated phenotypes (Maculopapular Exanthema + Severe Cutaneous Adverse Reactions + Stevens-Johnson Syndrome), but models often pick a subset:

> *"HLA-B\*35:08 is NOT associated with ______ when taking lamotrigine for Epilepsy."*
> - Options: (a) Severe Cutaneous Adverse Reactions alone, (b) Maculopapular Exanthema alone, (c) two of three, **(d) all three** ← correct
> - 4/6 models picked (a) or (b) — a single phenotype instead of the complete set

Models tend to anchor on one clearly non-associated phenotype rather than recognizing that *all* options apply. This is a reasoning failure where models don't fully process the logical structure of the question.

**Who does best:** Opus 4.6 leads at 70%, correctly handling the Neutropenia and Opioid-Related Disorders questions that trip up other models. It still fails on the cystitis and multi-phenotype questions.

### 4. Study Parameter Extraction — Superficially Low, but Evaluation Issues (10-50%)

Raw "both correct" scores look terrible (Opus at 10%), but the failures decompose into two very different categories:

**Problem 1: P-value string matching is too strict**
For "correct" type questions (where the answer exists in the paper), models frequently get the *right value* but in a slightly different format:
- Expected: `< 0.001` → Model says: `<0.001` (no space) → **Marked wrong**
- Expected: `< 0.001` → Model says: `<.001` (no leading zero) → **Marked wrong**
- Expected: `< 0.001` → Model says: `<0.001>` (trailing bracket) → **Marked wrong**

Significance judgments on these same questions are almost always correct (75% for Opus on "correct" type). The p-value comparison needs fuzzy/numeric matching rather than exact string matching.

**Problem 2: Models hallucinate findings for modified questions**
"Modified_drug" and "modified_variant" questions ask about associations that *don't exist* in the paper (expected answer: "not found"). This tests whether models can recognize when a paper doesn't contain the requested information.

Two patterns emerge:
- **Hallucination (all models):** For modified_drug questions, most models confidently return a p-value from the paper (typically `<0.001`) even though the drug/variant combination was altered to something not studied. They fail to notice the mismatch and just report the closest-looking result.
- **Thoughtful refusal (Opus, occasionally):** Opus sometimes catches the mismatch and writes a long explanation about why the question doesn't match the paper — but then gets a parse_error because the response isn't valid JSON. This is actually *better* behavior (the model correctly identified the trick) but scores 0 due to format non-compliance.

GPT-5 and Haiku score higher (50%) because they correctly return "not found" on modified_variant questions while other models hallucinate.

---

## Contradiction Analysis

Contradictions measure whether a model gives inconsistent answers when the same question is posed in standard vs. NOTA format.

| Model | Contradictions | Out of |
|---|---|---|
| Claude Haiku 4.5 | 4 | 30 |
| Claude Opus 4.6 | 2 | 30 |
| Claude Sonnet 4.5 | 2 | 30 |
| GPT-5 | 1 | 30 |
| GPT-4o | 0 | 30 |
| GPT-5.2 | 0 | 30 |

All contradictions are concentrated in **variant MCQ NOTA questions**. No contradictions appear in drug or phenotype tasks. The pattern is always the same: model gets the standard question right, then fails to select "None of the options" on the NOTA version, instead picking a distractor variant.

---

## Key Findings

1. **Drug MCQ is a ceiling task** — all models hit ~100%. Not useful for differentiation.
2. **Phenotype MCQ is the most discriminative task** — 30-point spread between best (Opus, 70%) and worst (Haiku, 40%), with interesting failure modes around medical term specificity.
3. **Annotation quality matters** — the universal cystitis failure (all 6 models "wrong" but arguably more precise) suggests some ground truth labels need review.
4. **NOTA questions expose calibration gaps** — smaller/cheaper models are more willing to guess than abstain, especially on variant identification.
5. **Study param evaluation needs fixing** — p-value string matching is too strict, masking the fact that models often extract the correct numeric value. Modified questions reveal genuine hallucination problems though.
6. **Sample size caveat** — these are 10-question subsets. Results need validation on the full dataset (112K+ questions) for statistical reliability.
