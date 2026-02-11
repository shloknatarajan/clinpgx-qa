# ClinPgx QA Dataset

## Goal
Take human annotated variants and sentences and create a dataset for QA. We also provide markdown files for each of the papers generated using [pubmed-markdown](https://github.com/shloknatarajan/pubmed-markdown).

## Data
The data is in the `data` directory. The `variant_bench.jsonl` and `sentence_bench.jsonl` files contain the human annotated variants and sentences respectively. The `papers` directory contains the markdown files for each of the papers.

## Generating Questions

This project includes scripts to generate the question datasets from the raw PharmGKB annotation files.

### Chained Questions

The `chained_questions.py` script generates multi-turn, dependent question chains from the annotation data.

**How to Run:**
```bash
python chained_questions.py
```
The output will be saved to `data/chained_questions.jsonl`.

### Yes/No Questions

The `yes_no_questions.py` script generates simple true/false questions from the `var_drug_ann.tsv` file by flipping parts of the sentence.

**How to Run:**
```bash
python yes_no_questions.py
```
The output will be saved to `data/yes_no_questions.jsonl`.

---

## Evaluating Models

Once the datasets are generated, you can evaluate a language model's performance on them. All evaluation outputs are saved to a timestamped directory under `runs/`.

### Running the Full Pipeline (Recommended)

The `src/eval/run.py` script is the easiest way to run an evaluation. It handles both generating model responses and scoring them in one step.

```bash
# Run the full evaluation for both datasets using gpt-4o-mini
python src/eval/run.py --model gpt-4o-mini

# Run only the 'chained' dataset with a limit of 50 questions
python src/eval/run.py --model gpt-4o --dataset chained --limit 50

# Re-score an existing response file without calling the LLM
python src/eval/run.py --score-only --dataset yes_no \
    --responses-path runs/20260210_223530_gpt-4o-mini/yes_no_responses.jsonl
```

#### Main Arguments for `run.py`:
- `--model`: The LiteLLM model identifier to use (e.g., `gpt-4o-mini`, `gpt-4o`). Defaults to `gpt-4o-mini`.
- `--limit`: The number of questions to process from each dataset. `0` means all. Defaults to `0`.
- `--dataset`: Which dataset to run: `yes_no`, `chained`, or `all`. Defaults to `all`.
- `--score-only`: A flag to skip the generation step and only score an existing file.
- `--responses-path`: Required when using `--score-only`. Specifies the path to the `..._responses.jsonl` file to score.

### Granular Control (Manual Steps)

You can also run the `generate` and `score` steps manually for each dataset. This is useful for debugging or if you only need to perform one part of the process.

#### Yes/No Questions

**1. Generate Responses:**
```bash
python src/eval/yes_no.py generate --model gpt-4o-mini --limit 100
```
This creates a `yes_no_responses.jsonl` file in a new `runs/...` directory.

**2. Score Responses:**
```bash
python src/eval/yes_no.py score \
    --responses-path runs/<your_run_directory>/yes_no_responses.jsonl
```

#### Chained Questions

**1. Generate Responses:**
```bash
python src/eval/chained.py generate --model gpt-4o-mini --limit 50
```
This creates a `chained_responses.jsonl` file in a new `runs/...` directory.

**2. Score Responses:**
```bash
python src/eval/chained.py score \
    --responses-path runs/<your_run_directory>/chained_responses.jsonl
```
