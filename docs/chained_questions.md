# NeurIPS Chained Question Generation

This document explains how the multi‑turn “chained question” dataset is generated from PharmGKB variant annotations and study parameters, and shows concrete examples of the resulting question chains.

## Overview

- Input rows from PharmGKB variant annotation TSVs are paired with their study parameters.
- For each pair, the generator builds a 1–3 turn chain:
  1) Claim verification (yes/no), 2) Statistical extraction, 3) Evidence evaluation (yes/no).
- About half of chains attempt a flipped (negative) claim for Turn 1 to create balanced supervision.
- Output is written to `data/chained_questions.jsonl`, one chain per line.

## Data Sources

- Variant annotations: `data/raw/variantAnnotations/{var_drug_ann.tsv,var_pheno_ann.tsv,var_fa_ann.tsv}`
- Study parameters: `data/raw/variantAnnotations/study_parameters.tsv`
- Script: `chained_questions.py`

## Pipeline

1. Load study parameters and index by `Variant Annotation ID`.
2. Load each annotation type (drug/pheno/functional annotation).
3. For each annotation (and each linked study parameters row, if any):
   - Build Turn 1 (claim verification), optionally flipped to generate a negative example.
   - If study parameters are present, select one available statistic and build Turn 2 (extraction).
   - Derive a yes/no assessment for that statistic and build Turn 3 (evaluation) when possible.
4. Save each chain as a JSON object to `data/chained_questions.jsonl`.

## Turn Details

### Turn 1 — Claim Verification

- Question templates reference the original annotation `Sentence` and `PMID`.
- Negative examples (~50%) are created by flipping, when possible:
  - Association flip: `associated with` ↔ `not associated with` (respects singular/plural verb forms).
  - Direction flip: `increased` ↔ `decreased` (only if the annotation is a positive association).
- Answers: `True` for the unmodified claim; `False` for a successfully flipped claim.
- Metadata: `flip_type` in `{original, association_flip, direction_flip}`.

### Turn 2 — Statistical Extraction

- Chooses exactly one stat that exists in the study parameters row:
  - `P Value`, `Study Cases`, `Study Controls`, `Ratio Stat` (+ `Ratio Stat Type`),
    `Confidence Interval Start/Stop`, `Study Type`, `Biogeographical Groups`.
- Asks a direct extraction question and returns a typed value (e.g., integer for cases, string for p‑value, float for ratio, CI as `start-stop`).

### Turn 3 — Evidence Evaluation

- Asks a yes/no question about the stat from Turn 2 and derives the answer by simple rules:
  - P Value: significant if `< 0.05` (string‑aware parsing of `=`, `<`, `>` prefixes).
  - Study Cases/Controls: adequate if `≥ 30`.
  - Ratio Stat: strong if `> 2.0` (or `< 0.5` for ORs).
  - Confidence Interval: supportive if the CI does not cross `1.0` (for ratio‑type effects).
  - Study Type: considered controlled if it contains `control` or `rct`.
  - Biogeographical Groups: considered a defined group unless `unknown`, `multiple groups`, or empty.
- If a rule cannot be applied (e.g., unparsable value), Turn 3 is omitted.

## Randomness and Balance

- Uses a fixed RNG seed (`42`) for reproducible template choices and stat selection.
- Negative claims are attempted on alternating indices to target ~50% negatives; if a flip is impossible, the chain remains positive.

## Output Schema

Each line is a JSON object with the following structure:

```json
{
  "chain_id": "chain_000123",
  "pmid": "12345678",
  "variant_annotation_id": "...",
  "study_parameters_id": "..." ,
  "source_file": "var_drug_ann|var_pheno_ann|var_fa_ann",
  "num_turns": 1|2|3,
  "has_negative_claim": true|false,
  "turns": [
    {
      "turn": 1,
      "reasoning_type": "claim_verification",
      "question": "...",
      "answer": true|false,
      "answer_source_field": "Sentence",
      "flip_type": "original|association_flip|direction_flip"
    },
    {
      "turn": 2,
      "reasoning_type": "statistical_extraction",
      "question": "...",
      "answer": "...|number",
      "answer_source_field": "e.g., P Value"
    },
    {
      "turn": 3,
      "reasoning_type": "evidence_evaluation",
      "question": "...",
      "answer": true|false,
      "answer_source_field": "same as Turn 2"
    }
  ]
}
```

## How to Run

```bash
python chained_questions.py
# Output: data/chained_questions.jsonl
```

## Examples (from current dataset)

### Example A — Negative claim, adequate cases (PMID 39792745)

- Turn 1 (claim_verification):
  - Q: “According to the study in PMID 39792745, is it accurate that genotype TT is not associated with decreased response to sitagliptin in people with Diabetes Mellitus, Type 2.”
  - A: `False` (association_flip)
- Turn 2 (statistical_extraction):
  - Q: “How many cases were included in this study from PMID 39792745?”
  - A: `61`
- Turn 3 (evidence_evaluation):
  - Q: “Is the number of cases sufficient for this analysis?”
  - A: `True` (≥ 30)

### Example B — Negative claim, non‑significant p‑value (PMID 31940240)

- Turn 1 (claim_verification):
  - Q: “Based on PMID 31940240, is the following claim supported: Allele T is associated with dose of heroin in people with Heroin Dependence as compared to allele C.”
  - A: `False` (association_flip of original “not associated”)
- Turn 2 (statistical_extraction):
  - Q: “What p-value is reported for this association in PMID 31940240?”
  - A: “`= 0.675`”
- Turn 3 (evidence_evaluation):
  - Q: “Is this finding statistically significant?”
  - A: `False` (0.675 ≥ 0.05)

### Example C — Positive claim, significant p‑value (PMID 27488176)

- Turn 1 (claim_verification):
  - Q: “According to the study in PMID 27488176, is it accurate that CYP2C9 *3 is associated with decreased dose of warfarin as compared to CYP2C9 *1/*1.”
  - A: `True`
- Turn 2 (statistical_extraction):
  - Q: “According to PMID 27488176, what is the p-value for this association?”
  - A: “`= 1.77e-22`”
- Turn 3 (evidence_evaluation):
  - Q: “Is this finding statistically significant?”
  - A: `True`

### Example D — Positive claim, population extraction (PMID 40054571)

- Turn 1 (claim_verification):
  - Q: “Based on PMID 40054571, is the following claim supported: Allele T is associated with increased response to citalopram, escitalopram, fluoxetine, fluvoxamine, paroxetine or sertraline in people with Obsessive-Compulsive Disorder as compared to allele C.”
  - A: `True`
- Turn 2 (statistical_extraction):
  - Q: “What population was studied in PMID 40054571?”
  - A: “`European`”
- Turn 3 (evidence_evaluation):
  - Q: “Is the study population from a defined biogeographical group?”
  - A: `True`

## Notes and Extensibility

- Adding new stat types: extend `STAT_QUESTION_PAIRS` with `field`, `check_fields`, `questions`, and `eval_questions`, plus parsing/evaluation logic if needed.
- Flipping relies on exact substrings; improving NLP robustness (e.g., lemmatization) would expand negative coverage.
- The simple evaluation rules are intentionally transparent; they can be replaced with richer domain logic if desired.

