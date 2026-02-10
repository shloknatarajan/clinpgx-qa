# Generating Fill-in-the-Blank Questions

## Goal

Generate fill-in-the-blank (FITB) pharmacogenomics questions by programmatically ablating structured fields from pre-written sentences in the ClinPgx variant annotation TSVs. Each question masks one field from a sentence and asks an LLM to predict the missing value, producing a large-scale benchmark with verifiable ground-truth answers.

## Data Sources

Three annotation TSVs contain pre-written sentences and structured metadata:

| File | Rows | Focus |
|---|---|---|
| `var_drug_ann.tsv` | 12,798 | Drug-variant associations (metabolism, dosage, PK/PD) |
| `var_fa_ann.tsv` | 2,123 | Functional assay results (in vitro activity) |
| `var_pheno_ann.tsv` | 14,312 | Phenotype associations (toxicity, efficacy, dosage) |

Total: ~29,233 raw annotations across all three files.

Additionally, `data/benchmark_annotations/` contains 32 JSON files (one per paper) with merged annotations, and `sentence_bench.jsonl` contains 179 curated (pmcid, variant, sentences) entries.

Reference documentation: `data/raw/variantAnnotations/README.pdf`

## Sentence Structure

Each sentence follows a templated pattern built from structured columns:

```
[Alleles] [isPlural] [Is/Is Not] associated with [Direction] [PD/PK terms | Functional terms | Phenotype]
[when treated with / when assayed with Drug(s)] in people with [Population Diseases]
[as compared to [Comparison Allele(s)]]
```

Example:
> "Genotypes CT + TT are associated with decreased dose of warfarin in people with Atrial Fibrillation, Heart valve replacement as compared to genotype CC."

## Ablatable Fields

Each annotation type has different fields available for blanking. The field value comes from the structured TSV column and can be located in the sentence via string matching.

### Common across all annotation types
| Field | TSV Column | Example Value |
|---|---|---|
| Variant/Haplotype | `Variant/Haplotypes` | `rs9923231` |
| Gene | `Gene` | `VKORC1` |
| Drug(s) | `Drug(s)` | `warfarin` |
| Allele(s) | `Alleles` | `CT + TT` |
| Direction | `Direction of effect` | `decreased` |
| Association | `Is/Is Not associated` | `Associated with` / `Not associated with` |
| Significance | `Significance` | `yes` / `no` / `not stated` |
| Comparison Allele(s) | `Comparison Allele(s) or Genotype(s)` | `CC` |

### Drug-specific (`var_drug_ann.tsv`)
| Field | TSV Column | Example Value |
|---|---|---|
| PD/PK terms | `PD/PK terms` | `metabolism of`, `dose of`, `clearance of` |
| Population diseases | `Population Phenotypes or diseases` | `Atrial Fibrillation` |

### Functional assay-specific (`var_fa_ann.tsv`)
| Field | TSV Column | Example Value |
|---|---|---|
| Functional terms | `Functional terms` | `activity`, `expression` |
| Assay type | `Assay type` | `in vivo`, `in vitro` |
| Cell type | `Cell type` | `hepatocytes` |

### Phenotype-specific (`var_pheno_ann.tsv`)
| Field | TSV Column | Example Value |
|---|---|---|
| Phenotype | `Phenotype` | `Stevens-Johnson Syndrome` |
| Side effect/efficacy | `Side effect/efficacy/other` | `Toxicity/ADR` |

## Implementation Plan

### Step 1: Define FITB data model

Create a Pydantic model for FITB questions in `src/data_setup/fill_in_blank.py`:

```python
class BlankTarget(BaseModel):
    blanked_field: str              # which field was removed (e.g. "Drug(s)")
    ground_truth: list[str]         # list of acceptable answers (union of comma-separated values)
    field_category: str             # grouping: "variant", "drug", "phenotype", "direction", etc.

class FITBQuestion(BaseModel):
    question_id: str                # unique identifier
    source_file: str                # which TSV the sentence came from
    annotation_id: str              # Variant Annotation ID from TSV
    pmid: str                       # paper reference
    original_sentence: str          # full original sentence
    blanked_sentence: str           # sentence with _____ (one or more blanks)
    blanks: list[BlankTarget]       # one entry per blank in the sentence
    annotation_type: str            # "drug", "functional_assay", "phenotype"
```

### Step 2: Build field extraction + blanking logic

For each annotation type, define a mapping from field name to TSV column. The blanking function should:

1. Read the structured column value for the target field
2. Locate that value (or its transformed form) within the `Sentence` column via string matching
3. Replace the matched span with `_____`
4. Skip rows where the field is empty/null or the value can't be found in the sentence

Edge cases to handle:
- **Multiple drugs**: `Drug(s)` may contain comma-separated values (e.g. `"nifedipine, amlodipine"`) joined by `Multiple drugs And/or` column. Store all values in `ground_truth` as a list — a response is correct if it matches **any** of them (not all required).
- **Multiple phenotypes**: similarly joined by `Multiple phenotypes or diseases And/or`. Same partial-match correctness rule applies.
- **Population diseases**: prefixed with `Disease:` in the TSV but may appear differently in the sentence
- **Allele notation**: values like `*1/*6` or `CT + TT` — use exact string matching
- **isPlural**: affects verb form (`is` vs `are`) — don't blank the verb, just the target field

### Step 3: Generate FITB dataset

Write a script (`src/data_setup/generate_fitb.py`) that:

1. Reads each of the three TSV files
2. For each row, generates FITB questions with **one or more blanks per sentence**:
   - Single-blank questions: one blank per field (one question per non-empty field)
   - Multi-blank questions: blank out 2+ fields from the same sentence simultaneously (e.g., blank both Drug and Direction). Each blank gets its own `BlankTarget` entry in `blanks`.
3. Filters out questions where blanking failed (value not found in sentence)
4. Writes output to `data/fitb_questions.jsonl`

The blanking functions should accept a list of fields to blank, making single-blank just the special case of `len(fields) == 1`. Multi-blank combinations can be generated from predefined field groups (e.g., `[Drug, Direction]`, `[Variant, Gene]`) rather than exhaustive combinatorics.

Expected yield: ~29K rows x ~4-6 single-blank + select multi-blank combos = ~150K-200K questions (after filtering).

### Step 4: Quality filtering

Apply filters to ensure question quality:
- Remove questions where the blanked value is too short (< 2 chars) or too generic
- Remove questions where the sentence is too short (< 10 words)
- Remove duplicate (blanked_sentence, ground_truth) pairs
- Optionally stratify by field type and annotation type for balanced evaluation

### Step 5: LLM evaluation harness

Create `src/eval/fitb_eval.py` using the existing `litellm` dependency:

1. Load questions from `data/fitb_questions.jsonl`
2. For each question, prompt the LLM with the blanked sentence and ask it to fill in the blank
3. Compare LLM output to `ground_truth` (a list of acceptable answers) using:
   - **Partial set match**: correct if the response matches **any** value in `ground_truth` (not all required)
   - **Exact match** (case-insensitive, whitespace-normalized)
   - **Semantic match** (for cases like `"warfarin"` vs `"Warfarin"` or `"SJS"` vs `"Stevens-Johnson Syndrome"`)
   - For multi-blank questions, evaluate each blank independently
4. Report accuracy broken down by:
   - Field type (drug, variant, phenotype, direction, etc.)
   - Annotation type (drug_ann, fa_ann, pheno_ann)
   - Significance level
5. Output results to `data/fitb_results.jsonl`

### Step 6: Benchmark subset selection

From the full FITB dataset, select a curated benchmark subset:
- Restrict to the 32 papers in `data/benchmark_annotations/` for a controlled evaluation
- Balance across field types and annotation types
- Target ~500-1000 questions for the core benchmark
- Write to `data/fitb_benchmark.jsonl`

## File Structure (proposed)

```
src/
├── data_setup/
│   ├── fill_in_blank.py          # FITB data model + blanking logic
│   └── generate_fitb.py          # Script to generate FITB dataset
├── eval/
│   ├── __init__.py
│   └── fitb_eval.py              # LLM evaluation harness
data/
├── fitb_questions.jsonl           # Full generated FITB dataset
├── fitb_benchmark.jsonl           # Curated benchmark subset
└── fitb_results.jsonl             # Evaluation results
```

## MVP: Quick Validation on Benchmark Subset

Before building the full pipeline, run a minimal end-to-end test using the 32 benchmark papers and a cheap, fast model to validate the approach and surface issues early.

### Scope
- **Data**: Only the 32 JSON files in `data/benchmark_annotations/` (~200 sentence-variant pairs across the three annotation types)
- **Blanking**: Single-blank only, targeting 4-5 fields per annotation type (see candidate lists below)
- **Model**: A small/fast model via litellm (e.g., `gpt-4o-mini`, `claude-haiku`, or similar) to keep cost and latency low
- **Eval**: Exact match only (case-insensitive, whitespace-normalized) — skip semantic matching for the MVP

### Candidate fields to blank (per annotation type)

**`var_drug_ann`** — Drug-variant associations:
| # | Field | TSV Column | Why |
|---|---|---|---|
| 1 | Drug | `Drug(s)` | Core clinical question — which drug is affected? |
| 2 | Allele(s) | `Alleles` | Tests variant-level knowledge |
| 3 | Direction | `Direction of effect` | Tests understanding of effect direction (increased/decreased) |
| 4 | PD/PK term | `PD/PK terms` | Tests pharmacological mechanism (metabolism, dose, clearance) |
| 5 | Comparison allele | `Comparison Allele(s) or Genotype(s)` | Tests knowledge of reference genotype |

**`var_fa_ann`** — Functional assay annotations:
| # | Field | TSV Column | Why |
|---|---|---|---|
| 1 | Drug | `Drug(s)` | Which substrate was used in the assay? |
| 2 | Allele(s) | `Alleles` | Which alleles were tested? |
| 3 | Direction | `Direction of effect` | Increased/decreased functional result |
| 4 | Functional term | `Functional terms` | What was measured (activity, expression)? |
| 5 | Comparison allele | `Comparison Allele(s) or Genotype(s)` | Reference allele in comparison |

**`var_pheno_ann`** — Phenotype associations:
| # | Field | TSV Column | Why |
|---|---|---|---|
| 1 | Drug | `Drug(s)` | Which drug triggers the phenotype? |
| 2 | Allele(s) | `Alleles` | Which alleles are implicated? |
| 3 | Direction | `Direction of effect` | Direction of phenotype association |
| 4 | Phenotype | `Phenotype` | The clinical outcome (e.g., Stevens-Johnson Syndrome) |
| 5 | Comparison allele | `Comparison Allele(s) or Genotype(s)` | Reference genotype |

### Steps

1. **Generate questions from benchmark JSONs**
   - Iterate over the 32 JSON files, extract annotations from `var_drug_ann`, `var_fa_ann`, `var_pheno_ann` keys
   - For each annotation, blank one field at a time for the 3-4 target fields
   - Expected yield: ~500-1000 single-blank questions

2. **Prompt the model**
   - Simple zero-shot prompt:
     ```
     Fill in the blank in the following pharmacogenomics statement.
     Respond with ONLY the missing value, nothing else.

     "{blanked_sentence}"
     ```
   - Run all questions through the model, collecting responses

3. **Score and report**
   - Exact match accuracy per field type
   - Exact match accuracy per annotation type
   - Print a confusion-style summary: how often the model gets Drug right vs Gene vs Direction vs Alleles
   - Log all (question, prediction, ground_truth, correct) rows to a results JSONL for manual inspection

4. **Inspect failures**
   - Manually review ~20-30 incorrect predictions to identify:
     - Are blanks ambiguous? (multiple valid answers not captured in ground truth)
     - Does the model hallucinate plausible but wrong answers?
     - Are there sentence patterns where blanking produces degenerate questions?
   - Use findings to refine the blanking logic and correctness criteria before scaling up

### MVP file
- `src/eval/fitb_mvp.py` — single script that generates questions from benchmark JSONs, runs inference, scores, and prints a summary table. Keep it self-contained so it can run with just `python src/eval/fitb_mvp.py`.

### Success criteria
- Pipeline runs end-to-end without errors
- Produces interpretable accuracy numbers per field
- Identifies at least 2-3 concrete improvements to carry into the full implementation (e.g., fields to skip, prompt tweaks, edge cases in blanking)

---

## Design Decisions
- **Partial-match correctness**: For fields with multiple comma-separated values (e.g., multiple drugs or diseases), the ground truth is stored as a list. A response is correct if it matches **any** value in the list, not necessarily all.
- **Multi-blank support**: The blanking logic accepts a list of fields to blank from a single sentence. The data model uses `blanks: list[BlankTarget]` so a single question can have 1..N blanks. Single-blank is the common case; multi-blank is used for harder evaluation variants.