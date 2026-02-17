# ClinPGx-QA: A Benchmark for Pharmacogenomic Reasoning from Full-Text Literature

ClinPGx-QA evaluates whether LLMs can read full-length pharmacogenomics papers and correctly answer questions about variant–drug associations, study parameters, and experimental designs — capabilities central to clinical pharmacogenomics.

## Abstract

Modern LLMs excel at short-context QA but struggle with long-context, evidence-grounded reasoning over biomedical literature. We present ClinPGx-QA, a benchmark built from PharmGKB variant annotations linked to PubMed Central full-text articles. The suite spans claim verification, chained multi-turn reasoning, multiple-choice identification, study parameter extraction, variant extraction, and a composite end-to-end paper investigation task. We evaluate contemporary models (Claude 4.x family, GPT-4o/5-series, Gemini 2.5) and observe strong drug identification but persistent weaknesses in phenotype classification, structured numerical extraction, and integrative end-to-end reasoning.

## Introduction

- Problem: Extracting trustworthy, structured pharmacogenomic findings from full-text literature remains challenging for LLMs.
- Gap: Existing benchmarks emphasize abstracts, exam-style QA, or curated triples, not end-to-end understanding of full articles with heterogeneous PGx nomenclature.
- Proposal: ClinPGx-QA measures long-context comprehension and structured reasoning across tasks tied to clinical PGx use cases.

### Contributions

- Full-text PGx benchmark covering six task families, including an end-to-end investigation metric.
- Scalable question generation pipelines grounded in PharmGKB annotations and article text.
- Unified evaluation harness with reproducible prompts, parsers, and scoring logic.
- Baseline results across diverse model families with error analyses highlighting open challenges.

## Related Work (Brief)

- PubMedQA, MedMCQA: short-context, exam-style; limited evidence grounding.
- SciFact: claim verification with abstracts/snippets; narrower clinical PGx scope.
- ClinPGx-QA targets full-text reasoning, entity normalization challenges (rsIDs, star alleles, HLA, phenotypes), and chained, dependent questions.

## Dataset

### Data Sources and Construction

- `data/raw/variantAnnotations/`: PharmGKB TSVs (`var_drug_ann.tsv`, `var_pheno_ann.tsv`, `var_fa_ann.tsv`, `study_parameters.tsv`).
- `data/papers/`: PubMed Central full-text articles (markdown via `pubmed-markdown`).
- `data/benchmark_annotations/`: 32 expert-annotated JSON files for ground-truth associations.
- `data/variant_bench.jsonl`, `data/sentence_bench.jsonl`: Structured benchmarks (variants and sentence-level evidence).

### Corpus and Scale

- Full-text articles (markdown): 3,064
- Human-annotated benchmark papers: 32
- Yes/No questions: 11,981 (`data/yes_no_questions.jsonl`)
- Chained multi-turn questions: 36,750 (`data/chained_questions.jsonl`) and a curated set of 100 (`data/neurips_chained_questions.jsonl`).

## Task Suite

### 1) Yes/No Claim Verification

- Input: full paper text + declarative claim.
- Generation: original supported claims plus flips of association/effect direction.
- Output: `true`/`false`.

### 2) Chained Multi-Turn Reasoning

- Dependent turns combining claim verification, statistical extraction (p-values, CIs), and objective evaluation.
- Chains vary (2–4 turns) based on study parameter availability; ~50% include negative/flipped claims.

### 3) Multiple-Choice Questions (MCQ)

- MCQ Drug: identify drug; MCQ Variant: identify variant (with “none of the options” distractors); MCQ Phenotype: identify phenotype category.

### 4) Study Parameter Extraction

- Extract p-values and significance judgments for specific variant–drug–phenotype statements; robust to inequality formats and numeral styles.

### 5) Variant Extraction

- Extract rsIDs, star alleles, HLA alleles, and metabolizer phenotypes from the full text.
- Scored on recall only; extra variants are not penalized.

### 6) Paper Investigation (End-to-End)

- Composite evaluation: variant extraction → answer MCQ and study-parameter questions for recalled variants.
- Scoring:

```
paper_score = variant_recall × mean(question_accuracy_across_recalled_variants)
```

## Methods

### Data Generation Pipelines (selected)

- Association tables: `src/data_generation/association_table.py`
- Sentence-level: `src/data_generation/sentence_bench_table.py`
- Variant bench: `src/data_generation/variant_bench_table.py`
- Chained questions: `src/data_generation/neurips_chains.py` (curated), `chained_questions.py` (large set)
- MCQ options: `data/mcq_options/*.jsonl`
- Study param question prompts: `src/modules/study_param_questions/generate.py`

### Evaluation Harness

- Unified runner: `src/eval/run.py` orchestrates per-task `generate` and `score` phases.
- LLM interface: `src/eval/llm.py` via LiteLLM, retry/backoff, and paper loading utilities.
- Scoring logic:
  - Yes/No: exact `true`/`false` parse (`src/eval/yes_no.py`).
  - Chained: per-turn parsers for claim verification, numeric extraction with inequality/CI handling, objective evaluation (`src/eval/chained.py`).
  - MCQ: letter parser and type-wise breakdowns (`src/eval/mcq.py`).
  - Study Param: strict JSON parsing then numeric/significance scoring with inequalities and 5% tolerance (`src/eval/study_param.py`).
  - Variant Extraction: recall by type (rsID, star, HLA, phenotype) in `src/modules/variant_extraction/variant_extraction.py`.

### Models and Inference

- Evaluated models: Claude Opus 4.6, Claude Sonnet 4.5, Claude Haiku 4.5, GPT‑5, GPT‑5.2, GPT‑4o, GPT‑4o‑mini, Gemini 2.5 Pro.
- Default model: `gpt-4o-mini`; reasoning models get higher token limits in `call_llm()`.
- Responses constrained to parseable formats (single word/letter or strict JSON) to avoid grading ambiguity.

### Reproducibility

- Environment: `pixi.toml`; set OpenAI/Anthropic keys in `.env`.
- Generate questions:

```bash
python yes_no_questions.py
python chained_questions.py
```

- Evaluate a model across pipelines:

```bash
python src/eval/run.py --model gpt-4o-mini
```

- Score existing results:

```bash
python src/eval/run.py --score-only --dataset yes_no \
  --responses-path runs/<run_dir>/yes_no_responses.jsonl
```

## Results

### Multiple-Choice Questions (10 questions each)

| Model | MCQ Drug | MCQ Variant | MCQ Phenotype | Study Param |
|-------|----------|-------------|---------------|-------------|
| GPT-5 | 100% | 100% | 50% | 70% |
| GPT-5.2 | 100% | 100% | 60% | 70% |
| Claude Opus 4.6 | 100% | 80% | 70% | 70% |
| Claude Sonnet 4.5 | 100% | 60% | 60% | 80% |
| Claude Haiku 4.5 | 100% | 70% | 40% | 70% |
| GPT-4o-mini | 100% | 50% | 60% | 80% |
| Gemini 2.5 Pro | 0%* | 0%* | 0%* | 0%* |

\* JSON parse failures on all outputs.

### Variant Extraction (5 articles)

| Model | Macro Recall | Micro Recall | Phenotype Recall | rsID Recall |
|-------|--------------|--------------|------------------|-------------|
| Claude Opus 4.6 | 0.800 | 0.933 | 1.000 | 0.900 |
| GPT-5 | 0.800 | 0.933 | 1.000 | 0.900 |
| GPT-5.2 | 0.800 | 0.933 | 1.000 | 0.900 |
| Claude Sonnet 4.5 | 0.733 | 0.867 | 0.800 | 0.900 |
| Claude Haiku 4.5 | 0.733 | 0.867 | 0.800 | 0.900 |
| GPT-4o | 0.592 | 0.600 | 0.600 | 0.600 |

### Paper Investigation (End-to-End, 2 papers)

| Model | Avg Variant Recall | Avg Question Acc | Avg Paper Score |
|-------|--------------------|------------------|-----------------|
| Claude Opus 4.6 | 1.000 | 0.616 | 0.609 |
| Claude Haiku 4.5 | 0.833 | 0.674 | 0.583 |
| Claude Sonnet 4.5 | 0.833 | 0.664 | 0.560 |
| GPT-4o-mini | 0.604 | 0.580 | 0.378 |

### Key Findings

- Drug identification is largely solved; most models get 100% on MCQ Drug.
- Phenotype classification remains challenging (40–70%).
- “None of the options” variant MCQs expose guessing bias and entity confusion.
- Study-parameter extraction is brittle to formatting and lexical perturbations.
- Variant extraction recall is strong for top models (≈93%) but drops for GPT‑4o (≈60%).
- End-to-end investigation caps near 61% even for the strongest model, indicating compounding difficulties.

## Limitations

- Partial coverage of PGx entity types beyond rsIDs/star/HLA/phenotypes.
- MCQ sets are derived from association tables and may inherit source biases.
- End-to-end scores currently reported on a small paper subset; scaling is ongoing.

## Ethics and Data Use

- Uses publicly available PMC articles and PharmGKB annotations; follow respective licenses and terms of use.
- Benchmark is for research; not a substitute for clinical decision support.

## Conclusion

ClinPGx-QA highlights clear progress on entity identification but underscores open challenges in phenotype understanding, numerical evidence extraction, and long-context integration. We hope it serves as a practical target for improving trustworthy, evidence-grounded biomedical reasoning.

## Project Structure (Appendix)

```
clinpgx-qa/
├── data/
│   ├── papers/                     # 3,064 markdown-converted PubMed Central articles
│   ├── raw/variantAnnotations/     # PharmGKB annotation TSVs
│   ├── benchmark_annotations/      # 32 expert-annotated ground-truth files
│   ├── yes_no_questions.jsonl      # 11,981 true/false questions
│   ├── chained_questions.jsonl     # 36,750 multi-turn question chains
│   ├── neurips_chained_questions.jsonl  # Curated 100-chain subset
│   ├── mcq_options/                # Variant/drug/phenotype MCQ options
│   ├── study_param_questions/      # Study parameter question JSONL
│   ├── variant_bench.jsonl         # Variant extraction ground truth
│   └── sentence_bench.jsonl        # Sentence-level ground truth
├── src/
│   ├── eval/                       # Evaluation scripts (yes_no, chained, mcq_*, study_param)
│   ├── data_generation/            # Dataset generation from PharmGKB annotations
│   ├── modules/
│   │   ├── variant_extraction/     # Variant extraction module
│   │   ├── paper_investigation/    # End-to-end paper investigation
│   │   ├── mc_questions/           # Multiple-choice question generation
│   │   └── study_param_questions/  # Study parameter question generation
│   └── utils/                      # Paper mapping utilities
├── runs/                           # Timestamped evaluation outputs
├── run_all_models.py               # Run evaluation across all models
├── chained_questions.py            # Generate chained question dataset
├── yes_no_questions.py             # Generate yes/no question dataset
└── analyze_results.py              # Aggregate results across runs
```

## References

- PharmGKB. https://www.pharmgkb.org/
- PubMed Central Open Access Subset. https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- pubmed-markdown. https://github.com/shloknatarajan/pubmed-markdown
