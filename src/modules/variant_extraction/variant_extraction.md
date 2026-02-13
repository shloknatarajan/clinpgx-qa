# Variant Extraction

Tests how well different models can extract pharmacogenetic variants from scientific articles. Compares LLM-extracted variants against ground truth in `data/variant_bench.jsonl` (2,922 articles).

## Usage

```bash
# Generate + score in one step
python src/modules/variant_extraction/variant_extraction.py run --model gpt-4o-mini --limit 10

# Or run each phase separately
python src/modules/variant_extraction/variant_extraction.py generate --model gpt-4o-mini --limit 10
python src/modules/variant_extraction/variant_extraction.py score --responses-path runs/<run_dir>/variant_responses.jsonl
```

## Pipeline

Two-phase design matching the `src/eval/yes_no.py` pattern:

### 1. Generate

Iterates over `variant_bench.jsonl`, loads each paper's full markdown text from `data/papers/`, sends it to the LLM with the extraction prompt, and saves responses to `variant_responses.jsonl` in a timestamped run directory (`runs/YYYYMMDD_HHMMSS_model_VE/`).

### 2. Score

Focuses exclusively on **recall** — we don't penalize the model for extracting extra variants. Computes:

- **Per-article**: recall (fraction of ground truth variants found)
- **Per-variant-type**: TP/FN and recall for each category
- **Aggregate**: micro-averaged and macro-averaged recall
- **Examples**: worst and best recall articles for debugging

Outputs `variant_eval_results.jsonl` and `variant_summary.txt`.

## Prompt

**System:**
> You are a pharmacogenomics expert. Extract genetic variants from scientific articles with precise formatting.

**User:**
> Extract all pharmacogenetic variants from the following article.
>
> VARIANT TYPES:
> 1. rsIDs: rs followed by numbers (rs9923231, rs1057910)
> 2. Star alleles: Gene\*Number format for pharmacogenes (CYP2C9\*3, NUDT15\*3, UGT1A1\*28)
> 3. HLA alleles: HLA-GENE\*XX:XX format (HLA-B\*58:01, HLA-DRB1\*03:01)
> 4. Metabolizer phenotypes: Gene + phenotype (CYP2D6 poor metabolizer, NAT2 slow acetylator)
>
> NORMALIZATION RULES:
> - Star alleles: Always use GENE\*NUMBER (CYP2C9\*3, not CYP2C9 \*3)
> - HLA: Always include HLA- prefix and use colon separator (HLA-B\*58:01)
> - Include diplotypes if mentioned (e.g., \*1/\*3 should be listed as the individual alleles)
>
> Return ONLY a JSON array of unique variants. No explanations.

## Variant Type Classification

Extracted and ground truth variants are classified into categories for per-type scoring:

| Type | Pattern | Examples |
|------|---------|----------|
| `rsID` | `rs` + digits | rs9923231, rs1057910 |
| `star_allele` | contains `*` + digit | CYP2C9\*3, TPMT\*3A |
| `HLA` | starts with `HLA-` | HLA-B\*58:01 |
| `phenotype` | contains "metabolizer" or "acetylator" | CYP2D6 poor metabolizer |
| `other` | none of the above | anything else |

## Scoring Details

- Only **recall** is measured — extra variants extracted by the model are ignored
- Variants are normalized before comparison (whitespace around `*` is collapsed)
- Set-based matching: each ground truth variant is either a TP (found) or FN (missed)
- Parse failures (LLM didn't return valid JSON) count all ground truth variants as FN

## Output Structure

```
runs/YYYYMMDD_HHMMSS_model_VE/
  variant_responses.jsonl       # Raw LLM responses + ground truth
  variant_eval_results.jsonl    # Scored results with TP/FP/FN per article
  variant_summary.txt           # Human-readable summary with metrics
```

## CLI Arguments

### `run`
Generate + score in one step. Takes the same arguments as `generate`.

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `gpt-4o-mini` | LiteLLM model identifier |
| `--bench-path` | `data/variant_bench.jsonl` | Path to ground truth JSONL |
| `--limit` | `0` (all) | Limit number of articles |

### `generate`
| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `gpt-4o-mini` | LiteLLM model identifier |
| `--bench-path` | `data/variant_bench.jsonl` | Path to ground truth JSONL |
| `--limit` | `0` (all) | Limit number of articles |

### `score`
| Arg | Default | Description |
|-----|---------|-------------|
| `--responses-path` | (required) | Path to `variant_responses.jsonl` |
