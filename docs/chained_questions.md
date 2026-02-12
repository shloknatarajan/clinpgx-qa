# NeurIPS Chained Question Generation

This document explains how the multi‑turn "chained question" dataset is generated from PharmGKB variant annotations and study parameters, and shows concrete examples of the resulting question chains.

## Overview

- Input rows from PharmGKB variant annotation TSVs are paired with their study parameters.
- For each pair, the generator builds a 2–4 turn chain in one of two families:
  - **Family A** (`A_claim→modality→stat→eval`, 4 turns): Claim verification → Evidence provenance → Statistical extraction → Evaluation.
  - **Family B** (`B_claim→presence_absence`, 2 turns): Claim verification (with `not_reported` option) → Evidence provenance (presence/absence of quantitative stats).
- Family B chains use negative examples where the claim references a swapped variant, drug, or phenotype that is **not reported** in the paper.
- Output is written to `data/neurips_chained_questions.jsonl`, one chain per line.

## Data Sources

- Variant annotations: `data/raw/variantAnnotations/{var_drug_ann.tsv,var_pheno_ann.tsv,var_fa_ann.tsv}`
- Study parameters: `data/raw/variantAnnotations/study_parameters.tsv`
- Paper full‑texts: loaded via `build_paper_index()` in `src/eval/llm.py`
- Script: `chained_questions.py`

## Pipeline

1. Load study parameters and index by `Variant Annotation ID`.
2. Load each annotation type (drug/pheno/functional annotation).
3. For each annotation (and each linked study parameters row, if any):
   - Build Turn 1 (claim verification): asks whether a pharmacogenomic claim is `supported`, `contradicted`, or (for Family B) `not_reported`.
   - Build Turn 2 (evidence provenance localization): asks about the type of evidence or presence of quantitative statistics.
   - For Family A, build Turn 3 (statistical extraction) and Turn 4 (objective or counterfactual evaluation).
4. Attach full paper text as `context` for each chain.
5. Save each chain as a JSON object to `data/neurips_chained_questions.jsonl`.

## Chain Families

### Family A — `A_claim→modality→stat→eval` (4 turns)

75 chains. Tests claim comprehension, evidence classification, numeric extraction, and statistical reasoning.

| Turn | `reasoning_type` | Description |
|------|-------------------|-------------|
| 1 | `claim_verification` | Is the pharmacogenomic claim `supported` or `contradicted` by the paper? |
| 2 | `evidence_provenance_localization` | Is this a `clinical_association` or `functional_assay`? |
| 3 | `statistical_extraction` | Extract a specific statistic (p‑value, study cases, study controls, etc.) |
| 4 | `objective_evaluation` or `counterfactual_evaluation` | Apply a threshold rule to the extracted statistic |

### Family B — `B_claim→presence_absence` (2 turns)

25 chains. Tests ability to identify claims that are **not reported** in the paper.

| Turn | `reasoning_type` | Description |
|------|-------------------|-------------|
| 1 | `claim_verification` | Is the claim `supported`, `contradicted`, or `not_reported`? |
| 2 | `evidence_provenance_localization` | Does the paper report any quantitative statistic for this claim? (`yes`/`no`) |

All Family B chains use `has_negative: true` with one of three swap strategies:
- `not_reported_variant_swap` — a different variant is substituted
- `not_reported_drug_swap` — a different drug is substituted
- `not_reported_phenotype_swap` — a different phenotype is substituted

## Turn Details

### Turn 1 — Claim Verification

- Questions reference the PMCID and state a pharmacogenomic claim.
- Family A answers: `supported` or `contradicted`.
- Family B answers: `supported`, `contradicted`, or `not_reported`.
- `answer_source_fields`: `["Sentence", "Is/Is Not associated", "Significance"]`

### Turn 2 — Evidence Provenance Localization

- Family A: Asks whether the annotation is a `clinical_association` or `functional_assay`.
- Family B: Asks whether the paper reports any quantitative statistic for the claim (answer: `yes` or `no`).
- `answer_source_fields`: `["source_file"]` (Family A) or `["study_parameters"]` (Family B).

### Turn 3 — Statistical Extraction (Family A only)

- Asks for a specific statistic from the study parameters:
  - P‑value (string, e.g. `"= 0.080"`, `"> 0.05"`)
  - Study cases or controls (integer, e.g. `619`)
- `answer_source_fields`: e.g. `["P Value"]`, `["Study Cases"]`, `["Study Controls"]`

### Turn 4 — Evaluation (Family A only)

Two subtypes:

- **`objective_evaluation`** (65 chains): Applies a threshold to the Turn 3 statistic.
  - P‑value: significant at alpha = 0.05? (`true`/`false`)
  - Study cases/controls: meets N ≥ 30? (`true`/`false`)
- **`counterfactual_evaluation`** (10 chains): Tests robustness with a stricter threshold.
  - "Would this association remain significant at alpha = 0.01?" (`true`/`false`)

## Turn Metadata Fields

Each turn includes rich metadata beyond question/answer:

| Field | Description |
|-------|-------------|
| `answer_source_fields` | Which PharmGKB columns the answer is derived from |
| `evidence_required` | Whether the model must consult the paper text |
| `evidence_granularity` | `document`, `modality`, or `db_record` |
| `negative_type` | Swap type for Family B negatives (or `null`) |
| `ambiguity_risk` | `low`, `medium`, or `high` |
| `metadata` | Additional context (e.g. `Variant Annotation ID`) |

## Capability Tags

Each chain is tagged with the capabilities it tests:

- `claim_verification` — understanding a pharmacogenomic claim
- `evidence_provenance` — classifying evidence type
- `numeric_extraction` — extracting numerical values from text
- `objective_statistical_reasoning` — applying threshold rules
- `counterfactual_reasoning` — reasoning about alternative thresholds
- `negation_scope` — identifying claims not present in the paper

## Output Schema

Each line is a JSON object with the following structure:

```json
{
  "chain_id": "chain_000123",
  "chain_family": "A_claim→modality→stat→eval",
  "pmcid": "PMC1234567",
  "variant_annotation_id": "...",
  "study_parameters_id": "...",
  "summary_annotation_id": null,
  "source_tables": ["var_pheno_ann", "study_parameters"],
  "capability_tags": ["claim_verification", "evidence_provenance", "numeric_extraction", "objective_statistical_reasoning"],
  "num_turns": 4,
  "has_negative": false,
  "turns": [
    {
      "turn": 1,
      "reasoning_type": "claim_verification",
      "question": "...",
      "answer": "supported|contradicted|not_reported",
      "answer_source_fields": ["Sentence", "Is/Is Not associated", "Significance"],
      "evidence_required": true,
      "evidence_granularity": "document",
      "negative_type": null,
      "ambiguity_risk": "low",
      "metadata": {"Variant Annotation ID": "..."}
    },
    {
      "turn": 2,
      "reasoning_type": "evidence_provenance_localization",
      "question": "...",
      "answer": "clinical_association|functional_assay",
      "answer_source_fields": ["source_file"],
      "evidence_required": false,
      "evidence_granularity": "modality",
      "negative_type": null,
      "ambiguity_risk": "low",
      "metadata": {}
    },
    {
      "turn": 3,
      "reasoning_type": "statistical_extraction",
      "question": "...",
      "answer": "string|integer",
      "answer_source_fields": ["P Value"],
      "evidence_required": true,
      "evidence_granularity": "db_record",
      "negative_type": null,
      "ambiguity_risk": "low",
      "metadata": {}
    },
    {
      "turn": 4,
      "reasoning_type": "objective_evaluation|counterfactual_evaluation",
      "question": "...",
      "answer": true|false,
      "answer_source_fields": ["P Value"],
      "evidence_required": true,
      "evidence_granularity": "db_record",
      "negative_type": null,
      "ambiguity_risk": "low",
      "metadata": {}
    }
  ],
  "context": "<full paper text>"
}
```

## How to Run

```bash
python chained_questions.py
# Output: data/neurips_chained_questions.jsonl
```

## Examples (from current dataset)

### Example A — Supported claim, non‑significant p‑value (PMC5948914)

- Turn 1 (`claim_verification`):
  - Q: "Based on PMCID PMC5948914, is the following pharmacogenomic claim supported or contradicted: Genotypes GT + TT are not associated with overall survival when treated with gemcitabine in people with Carcinoma, Non-Small-Cell Lung as compared to genotype GG. Answer one of: supported, contradicted."
  - A: `supported`
- Turn 2 (`evidence_provenance_localization`):
  - Q: "What type of evidence does this annotation represent: clinical association or functional assay? Answer one of: clinical_association, functional_assay."
  - A: `clinical_association`
- Turn 3 (`statistical_extraction`):
  - Q: "For the study of rs2231142 and gemcitabine in Disease:Non-Small Cell Lung Carcinoma (PMCID PMC5948914), what p-value was reported for this association?"
  - A: `= 0.080`
- Turn 4 (`objective_evaluation`):
  - Q: "Is this finding statistically significant at alpha = 0.05? Answer true or false."
  - A: `False`

### Example B — Supported claim, counterfactual evaluation (PMC4640545)

- Turn 1 (`claim_verification`):
  - Q: "Based on PMCID PMC4640545, is the following pharmacogenomic claim supported or contradicted: Genotype CC is not associated with metabolism of carbamazepine in people with Epilepsy as compared to genotypes CT + TT. Answer one of: supported, contradicted."
  - A: `supported`
- Turn 2 (`evidence_provenance_localization`):
  - Q: "What type of evidence does this annotation represent: clinical association or functional assay? Answer one of: clinical_association, functional_assay."
  - A: `clinical_association`
- Turn 3 (`statistical_extraction`):
  - Q: "For the study of rs2298771 and carbamazepine in Disease:Epilepsy (PMCID PMC4640545), what p-value was reported for this association?"
  - A: `> 0.05`
- Turn 4 (`counterfactual_evaluation`):
  - Q: "Would this association remain significant at alpha = 0.01? Answer true or false."
  - A: `False`

### Example C — Not‑reported claim, phenotype swap (PMC8602039)

- Turn 1 (`claim_verification`):
  - Q: "Based on PMCID PMC8602039, is the following pharmacogenomic claim supported, contradicted, or not reported: In PMC8602039, rs74569896 is associated with Efficacy, Toxicity, Metabolism/PK outcomes for aspirin, clopidogrel. Answer one of: supported, contradicted, not_reported."
  - A: `not_reported` (negative_type: `not_reported_phenotype_swap`)
- Turn 2 (`evidence_provenance_localization`):
  - Q: "Does PMCID PMC8602039 report any quantitative statistic (p-value/OR/CI) for this claim?"
  - A: `no`

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total chains | 100 |
| Family A chains | 75 (4 turns each) |
| Family B chains | 25 (2 turns each) |
| Total turns | 350 |
| Chains with negatives | 25 (all Family B) |

**Reasoning type distribution:**

| Type | Count |
|------|-------|
| `claim_verification` | 100 |
| `evidence_provenance_localization` | 100 |
| `statistical_extraction` | 75 |
| `objective_evaluation` | 65 |
| `counterfactual_evaluation` | 10 |
