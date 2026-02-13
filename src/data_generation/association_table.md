# Association Table

**Goal:** Create a denormalized table joining variant annotations with their corresponding study parameters, scoped to the articles present in `variant_bench.jsonl`.

## Data Sources

| Source | Path | Records | Join Key |
|--------|------|---------|----------|
| variant_bench.jsonl | `data/variant_bench.jsonl` | 2,922 articles | `pmid` |
| var_drug_ann.tsv | `data/raw/variantAnnotations/var_drug_ann.tsv` | 12,797 | `PMID`, `Variant Annotation ID` |
| var_pheno_ann.tsv | `data/raw/variantAnnotations/var_pheno_ann.tsv` | 14,311 | `PMID`, `Variant Annotation ID` |
| study_parameters.tsv | `data/raw/variantAnnotations/study_parameters.tsv` | 35,554 | `Variant Annotation ID` |

## Join Strategy

```
variant_bench.jsonl (article PMIDs)
    │
    ├──► var_drug_ann.tsv     ──┐  (filter by PMID ∈ variant_bench)
    │                           ├──► LEFT JOIN on Variant Annotation ID ──► study_parameters.tsv
    └──► var_pheno_ann.tsv    ──┘
```

1. **Collect PMIDs** from `variant_bench.jsonl` (the 2,922-article scope).
2. **Filter annotations**: Keep only rows from `var_drug_ann.tsv` and `var_pheno_ann.tsv` whose `PMID` is in the variant_bench set. Tag each row with its source (`drug` or `pheno`).
3. **Left-join study parameters**: For each filtered annotation, join `study_parameters.tsv` on `Variant Annotation ID`. Use a left join so annotations without study parameters are preserved (with null study fields).
4. **Result**: One row per (annotation, study_parameter) pair. Annotations with multiple study parameters produce multiple rows.

## Output Schema

### From Variant Annotations (shared fields)
| Column | Description |
|--------|-------------|
| `annotation_source` | `"drug"` or `"pheno"` — which TSV this row came from |
| `Variant Annotation ID` | Unique annotation ID (primary key on the annotation side) |
| `Variant/Haplotypes` | e.g. `rs9923231`, `CYP2C9*3` |
| `Gene` | Pharmacogene |
| `Drug(s)` | Drug name(s) |
| `PMID` | PubMed ID |
| `Phenotype Category` | Dosage, Efficacy, Metabolism/PK, Toxicity, etc. |
| `Significance` | yes / no |
| `Sentence` | Natural-language summary of the finding |
| `Alleles` | Variant alleles tested |
| `Direction of effect` | Direction of pharmacogenetic effect |

### From Study Parameters
| Column | Description |
|--------|-------------|
| `Study Parameters ID` | Unique study parameter ID |
| `Study Type` | e.g. cohort, case/control, meta-analysis |
| `Study Cases` | N cases |
| `Study Controls` | N controls |
| `Characteristics` | Free-text study description |
| `Characteristics Type` | Category (Drug, Study Cohort, etc.) |
| `Frequency In Cases` | Allele frequency in cases |
| `Allele Of Frequency In Cases` | Which allele |
| `Frequency In Controls` | Allele frequency in controls |
| `Allele Of Frequency In Controls` | Which allele |
| `P Value` | Statistical significance |
| `Ratio Stat Type` | OR, RR, etc. |
| `Ratio Stat` | Effect size value |
| `Confidence Interval Start` | CI lower bound |
| `Confidence Interval Stop` | CI upper bound |
| `Biogeographical Groups` | Population ancestry |

## Implementation Plan (`association_table.py`)

### Step 1: Load variant_bench PMIDs
- Read `data/variant_bench.jsonl`, collect the set of all `pmid` values.

### Step 2: Load and filter variant annotations
- Read `var_drug_ann.tsv` and `var_pheno_ann.tsv` into DataFrames.
- Add an `annotation_source` column (`"drug"` / `"pheno"`).
- Concatenate and filter to only PMIDs present in variant_bench.

### Step 3: Load and join study parameters
- Read `study_parameters.tsv` into a DataFrame.
- Left-join onto the filtered annotations on `Variant Annotation ID`.

### Step 4: Write output
- Save as `data/association_table.tsv` (TSV for consistency with other project data).
- Log summary stats: total rows, annotations with/without study params, articles covered.

## Expected Output

- **Scope**: Only articles in variant_bench (2,922 PMIDs)
- **Granularity**: One row per (variant annotation, study parameter) pair
- **Estimated rows**: ~35k–50k (depends on how many annotations in-scope have study params)
- **Output path**: `data/association_table.tsv`
