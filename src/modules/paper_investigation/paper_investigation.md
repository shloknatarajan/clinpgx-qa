# Full Paper Investigation

## Concept

Evaluate end-to-end LLM understanding of pharmacogenomics papers by combining **variant extraction recall** with **downstream question accuracy**. The model must first identify which variants are studied in a paper, then correctly answer multiple-choice questions about those variants — producing a single composite score per paper that is guaranteed to be ≤ variant recall.

## Scoring Formula

For each paper (PMCID):

1. **Variant Recall** = |extracted_variants ∩ ground_truth_variants| / |ground_truth_variants|
2. For each recalled (correctly extracted) variant, gather all MC questions that reference that (pmcid, variant) pair
3. **Question Accuracy per variant** = num_correct / num_total questions for that variant
4. **Paper Score** = variant_recall × mean(question_accuracy across recalled variants)

The multiplication by recall guarantees Paper Score ≤ Variant Recall. Missed variants contribute 0 to the question accuracy (they are already penalized via recall), and only recalled variants are tested for deeper understanding.

---

## Implementation Plan

### File Structure

```
src/modules/paper_investigation/
├── paper_investigation.md   # This spec
├── question_index.py        # Step 1: Data linkage — (pmcid, variant) → MC questions
└── paper_investigation.py   # Steps 2–5: Evaluation pipeline + CLI
```

---

### Step 1: Question Index (`question_index.py`)

Build an index mapping **(pmcid, variant) → list of MC questions** across the four MC question types so we can look up every question relevant to a recalled variant.

#### 1a. Unified question model

```python
class UnifiedQuestion(BaseModel):
    """Normalized representation of a question from any MC pipeline."""
    source_pipeline: str          # "mcq_variant" | "mcq_drug" | "mcq_phenotype" | "study_param"
    pmcid: str
    variant: str                  # normalized variant string (e.g. "rs2909451", "CYP2C19*2")
    annotation_id: str            # Variant Annotation ID from PharmGKB
    raw_question: dict            # the full original question record (for passing to eval)
```

#### 1b. Loading logic for each question source

All four MC sources have `pmcid` and `variant` fields directly — no annotation_id resolution needed.

| Source File | Key Fields | Variant Resolution |
|---|---|---|
| `data/mcq_options/variant_mcq_options.jsonl` | `pmcid`, `variant` | Direct — `variant` field already present |
| `data/mcq_options/drug_mcq_options.jsonl` | `pmcid`, `variant` | Direct |
| `data/mcq_options/phenotype_mcq_options.jsonl` | `pmcid`, `variant` | Direct |
| `data/study_param_questions/study_param_questions.jsonl` | `pmcid`, `variant` | Direct |

#### 1c. QuestionIndex class

```python
class QuestionIndex:
    """Maps (pmcid, variant) pairs to all MC questions involving that combination."""

    def __init__(self) -> None:
        # Load all 4 MC question sources
        # Build dict[(pmcid, normalized_variant)] → list[UnifiedQuestion]
        ...

    def get_questions(self, pmcid: str, variant: str) -> list[UnifiedQuestion]:
        """Return all questions for a (pmcid, variant) pair."""
        ...

    def get_paper_variants_with_questions(self, pmcid: str) -> set[str]:
        """Return all variants that have at least one question for this paper."""
        ...

    @property
    def all_pmcids(self) -> set[str]:
        """All PMCIDs that have at least one indexed question."""
        ...
```

Variant normalization: reuse `normalize_variant()` from `src/modules/variant_extraction/variant_extraction.py` for consistent matching between ground truth, extracted, and question-linked variants.

---

### Step 2: Evaluation Pipeline (`paper_investigation.py`)

Single-file pipeline following the existing `generate` / `score` / `run` CLI pattern used by `variant_extraction.py`.

#### 2a. Generate phase

For each paper in `data/variant_bench.jsonl`:

1. **Load paper** text via `load_paper(pmcid, paper_index)` — skip if not found
2. **Extract variants** — call LLM with the variant extraction prompt (reuse `USER_PROMPT_TEMPLATE` and `SYSTEM_PROMPT` from `variant_extraction.py`)
3. **Compute recall** — `parse_variant_list()` → `normalize_variant()` → set intersection with ground truth
4. **Collect questions** — for each recalled variant, use `QuestionIndex.get_questions(pmcid, variant)`
5. **Answer questions** — for each question, format it as the corresponding pipeline would (paper context + question) and call the LLM:
   - **mcq_***: paper text + blanked sentence + options → parse letter (a/b/c/d)
   - **study_param**: paper text + extraction prompt → parse JSON {p_value, significance}
6. **Save response record** to JSONL with all raw data needed for scoring:

```python
{
    "pmcid": str,
    "model": str,
    "ground_truth_variants": list[str],
    "predicted_variants": list[str] | None,
    "recalled_variants": list[str],
    "variant_recall": float,
    "variant_results": {
        "<variant>": {
            "questions_asked": int,
            "questions_correct": int,
            "question_accuracy": float,
            "responses": [
                {
                    "source_pipeline": str,
                    "question": dict,
                    "model_response": str,
                    "correct": bool,
                    "expected_answer": any
                }
            ]
        }
    },
    "paper_score": float  # recall × mean(variant accuracies)
}
```

#### 2b. Scoring each question type

Reuse existing scoring logic from each eval module rather than re-implementing:

| Pipeline | Scoring Reference | Logic |
|---|---|---|
| `mcq_*` | `src/eval/mcq.py` | Extract letter (a/b/c/d) from response, compare to `q["correct_answer"]` |
| `study_param` | `src/eval/study_param.py` | Parse JSON with p_value and significance, numeric comparison with 5% tolerance and inequality support |

Implement a `score_question(question: UnifiedQuestion, response: str) -> bool` dispatcher that calls the right scoring logic based on `source_pipeline`.

#### 2c. Question prompting

Build prompt based on `source_pipeline`:

| Pipeline | Prompt Strategy |
|---|---|
| `mcq_*` | System: PGx expert. User: paper text + blanked sentence + "Options: a) ... b) ... c) ... d) ..." → expects single letter |
| `study_param` | System: PGx expert. User: paper text + extraction prompt for specific variant/drug pair → expects JSON |

Reference the existing `generate` functions in `src/eval/mcq.py` and `src/eval/study_param.py` for exact prompt formatting.

---

### Step 3: Score phase

Read the responses JSONL from generate phase and compute aggregate statistics:

```
================================================================
Paper Investigation Results  |  model=gpt-4o-mini
================================================================

  Papers scored:        N
  Papers skipped:       M  (no paper text or parse failure)

  Avg Variant Recall:   0.XXX
  Avg Question Acc:     0.XXX  (across recalled variants only)
  Avg Paper Score:      0.XXX  (recall × question_acc)

By pipeline breakdown:
  Pipeline           Questions    Correct    Accuracy
  ---------------    ---------    -------    --------
  mcq_variant               50         40      0.800
  mcq_drug                  50         38      0.760
  mcq_phenotype             30         22      0.733
  study_param               40         28      0.700
  ---------------    ---------    -------    --------
  TOTAL                    170        128      0.753

WORST PAPERS (by paper_score):
  [1] PMC12345  recall=0.500  q_acc=0.400  score=0.200
  ...

BEST PAPERS (by paper_score):
  [1] PMC67890  recall=1.000  q_acc=0.950  score=0.950
  ...
```

Output files:
- `{model}_paper_investigation_responses.jsonl` — full per-paper records
- `{model}_paper_investigation_eval_results.jsonl` — scored records
- `{model}_paper_investigation_summary.txt` — human-readable summary

---

### Step 4: CLI interface

Follow the `variant_extraction.py` pattern exactly:

```
python src/modules/paper_investigation/paper_investigation.py run --model gpt-4o-mini --limit 10
python src/modules/paper_investigation/paper_investigation.py generate --model gpt-4o-mini --limit 10
python src/modules/paper_investigation/paper_investigation.py score --responses-path runs/.../paper_investigation_responses.jsonl
```

Arguments:
- `--model` — LiteLLM model identifier (default: `gpt-4o-mini`)
- `--limit` — max papers to process (default: 0 = all)
- `--bench-path` — path to `data/variant_bench.jsonl` (default)
- `--output-dir` — output directory (default: auto-generated timestamped dir under `runs/`)
- `--responses-path` — for `score` subcommand only

---

### Step 5: Integration with multi-model runner

Add `paper_investigation` to:
- `run_all_models.py` → `PIPELINES` dict
- `run_all_models_ve.py` (or create a combined runner)
- `src/eval/run.py` → as an optional `--dataset paper_investigation` choice

---

## Dependencies & Reuse

| What | Reuse From |
|---|---|
| `call_llm()`, `build_paper_index()`, `load_paper()` | `src/eval/llm.py` |
| Variant extraction prompt + parsing | `src/modules/variant_extraction/variant_extraction.py` |
| `normalize_variant()`, `classify_variant()` | `src/modules/variant_extraction/variant_extraction.py` |
| MCQ scoring logic | `src/eval/mcq.py` |
| Study param scoring logic | `src/eval/study_param.py` |
| Pydantic, loguru, tqdm, litellm | existing deps in `pixi.toml` |

## Edge Cases

- **Paper not found**: skip, count as "no paper context"
- **Variant extraction parse failure**: record with `paper_score=None`, count separately
- **Recalled variant has 0 questions**: exclude from question_accuracy mean (variant was in ground truth and extracted, but no questions were generated for it — this can happen if the variant exists in `variant_bench` but not in any question set)
- **Paper has 0 recalled variants with questions**: `paper_score = 0` (recall penalty dominates)
- **MCQ `none_of_the_above` questions**: include — they test the same variant knowledge

## Estimated Scope

- `question_index.py`: ~100 lines (simpler now — all sources have direct variant fields)
- `paper_investigation.py`: ~350 lines
- Total: ~450 lines of new code, heavily reusing existing infrastructure
