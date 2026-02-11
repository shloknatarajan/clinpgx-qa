# NeurIPS Chain Families & Evidence Localization

## Evidence Localization

Evidence localization in this dataset is **provenance/modality/aggregation-level**, NOT sentence-level spans. The granularity levels are:

| Granularity   | Meaning                                                       |
|---------------|---------------------------------------------------------------|
| `document`    | Which PMID supports the claim                                 |
| `modality`    | Clinical association vs functional assay vs guideline vs label |
| `aggregation` | Multi-PMID / summary-level (PMID Count, Evidence Count)       |
| `temporal`    | History/update timestamps                                     |
| `db_record`   | Answered purely from ClinPGx structured tables                |
| `heuristic`   | Derived heuristic (must be labeled explicitly)                |

## Chain Families

### Family A: `A_claim->modality->stat->eval` (4 turns)

Positive variant-level chains with statistical evaluation. ~60% of dataset.

| Turn | Type                              | Example                                         |
|------|-----------------------------------|-------------------------------------------------|
| 1    | `claim_verification`              | Is this pharmacogenomic claim supported?         |
| 2    | `evidence_provenance_localization`| Clinical association or functional assay?        |
| 3    | `statistical_extraction`          | What p-value / OR / CI / sample size is reported?|
| 4    | `objective_evaluation` or `counterfactual_evaluation` | Is the p-value significant at alpha=0.05? |

### Family B: `B_claim->presence_absence` (2 turns)

Hard-negative chains using entity swaps. ~20% of dataset.

| Turn | Type                              | Example                                         |
|------|-----------------------------------|-------------------------------------------------|
| 1    | `claim_verification`              | Is this swapped claim reported? Answer: `not_reported` |
| 2    | `evidence_provenance_localization`| Does the PMID report any statistic for this? Answer: `no` |

Negative types: `not_reported_variant_swap`, `not_reported_drug_swap`, `not_reported_phenotype_swap`, `not_reported_population_swap`.

### Family C: `C_summary->aggregate->strength` (3-4 turns)

Summary-level aggregation chains. ~15% of dataset.

| Turn | Type                              | Example                                         |
|------|-----------------------------------|-------------------------------------------------|
| 1    | `evidence_aggregation`            | What is the Level of Evidence (1A-4)?            |
| 2    | `evidence_aggregation`            | How many PMIDs support this annotation?          |
| 3    | `objective_evaluation`            | Is it supported by more than one PMID?           |
| 4*   | `evidence_provenance_localization`| Which evidence types contribute? (optional)      |

### Family D: `D_cross_field_consistency` (3 turns)

Cross-field consistency checks between ratio statistics and annotated direction. ~3% of dataset.

| Turn | Type                        | Example                                           |
|------|-----------------------------|----------------------------------------------------|
| 1    | `statistical_extraction`    | What OR/RR/HR is reported?                         |
| 2    | `statistical_extraction`    | What direction of effect is annotated?             |
| 3    | `cross_field_consistency`   | Is the direction consistent with the ratio stat?   |

Only generated when phenotype context implies a monotonic outcome (risk/likelihood terms or PD/PK terms like metabolism, clearance, response).

### Family E: `E_temporal_summary_history` (2 turns)

Temporal reasoning over summary annotation history. ~2% of dataset.

| Turn | Type                 | Example                                           |
|------|----------------------|----------------------------------------------------|
| 1    | `temporal_reasoning` | Was this annotation updated after 2020?            |
| 2    | `temporal_reasoning` | How many updates/corrections are recorded?         |

## Capability Tags

Each chain is tagged with one or more reasoning capabilities:

- `claim_verification` - Verifying pharmacogenomic claims against evidence
- `evidence_provenance` - Locating which document/modality provides evidence
- `numeric_extraction` - Extracting quantitative statistics from structured data
- `counterfactual_reasoning` - Evaluating claims under alternative conditions
- `objective_statistical_reasoning` - Deriving conclusions from statistics
- `aggregation_reasoning` - Reasoning over multi-source evidence summaries
- `temporal_reasoning` - Reasoning about update history and time
- `consistency_reasoning` - Checking cross-field consistency
- `negation_scope` - Understanding what is NOT reported
