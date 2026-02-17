#!/usr/bin/env python3
"""
Generate chained benchmark questions for pharmacogenomic paper parsing.

Uses var_drug_ann.tsv and var_pheno_ann.tsv, restricted to papers with
markdown files in data/papers/ and where ALL variants have p-values in
study_parameters.tsv.

Each chain has 4 steps:
  1. Gene Inventory — which gene(s) are evaluated
  2. Predictor Inventory Per Gene — genetic predictors grouped by gene
  3. Comparison Extraction — explicit A vs B comparisons per predictor
  4. Evidence Selection — comparison with smallest p-value

Design notes:
  - Variant/Haplotypes values are treated as atomic identifiers exactly as
    encoded in ClinPGx. No comma-splitting or biological deduplication.
  - For Q3, all comparison types (allele, genotype, metabolizer) are
    included without level filtering. Both structured columns and sentence
    fallback are used to build comparison strings.
  - Comparison strings are canonicalized: items within each side are sorted
    lexicographically and joined with " + ", then "lhs vs rhs".
  - Chains are only emitted if they pass a validation checklist (non-empty
    Q1, non-empty Q3, at least one Q4 comparison with a parsed p-value).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "variantAnnotations"
PAPERS_DIR = DATA_DIR / "papers"
OUTPUT_PATH = DATA_DIR / "chained_questions" / "variant_allele_chains.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv_data():
    var_drug = pd.read_csv(RAW_DIR / "var_drug_ann.tsv", sep="\t")
    var_pheno = pd.read_csv(RAW_DIR / "var_pheno_ann.tsv", sep="\t")
    study_params = pd.read_csv(RAW_DIR / "study_parameters.tsv", sep="\t")
    var_drug["_source"] = "var_drug_ann"
    var_pheno["_source"] = "var_pheno_ann"
    return var_drug, var_pheno, study_params


def get_paper_pmids():
    """Return {pmid_int: pmc_id_str} for all papers in data/papers/."""
    pmid_to_pmc = {}
    for pf in PAPERS_DIR.glob("*.md"):
        content = pf.read_text()[:1000]
        m = re.search(r"\*\*PMID:\*\*\s*(\d+)", content)
        if m:
            pmid_to_pmc[int(m.group(1))] = pf.stem
    return pmid_to_pmc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s(val):
    """Safe string from a potentially-NaN value."""
    return str(val).strip() if pd.notna(val) else ""


def parse_pvalue(pval_str):
    """Parse a p-value string like '= 0.05' or '< 1.77e-22' to float."""
    s = _s(pval_str)
    if not s:
        return None
    s = re.sub(r"^[<>=≤≥~\s]+", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_side(side_str):
    """Sort items within one side of a comparison and rejoin with ' + '."""
    parts = [p.strip() for p in side_str.split("+")]
    parts = [p for p in parts if p]
    return " + ".join(sorted(parts))


def normalize_comparison(lhs, rhs):
    """Build a canonical 'A vs B' string with sorted items on each side."""
    return f"{_normalize_side(lhs)} vs {_normalize_side(rhs)}"


# ---------------------------------------------------------------------------
# Comparison extraction helpers
# ---------------------------------------------------------------------------

def get_all_comparison_strings(row):
    """Return all comparison strings for a row (across all types).

    Checks both allele/genotype and metabolizer comparison columns.
    Returns a list of canonical comparison strings.
    """
    comparisons = []

    # Allele / genotype comparison
    alleles = _s(row.get("Alleles"))
    comp_alleles = _s(row.get("Comparison Allele(s) or Genotype(s)"))
    if alleles and comp_alleles:
        comparisons.append(normalize_comparison(alleles, comp_alleles))

    # Metabolizer comparison
    met = _s(row.get("Metabolizer types"))
    comp_met = _s(row.get("Comparison Metabolizer types"))
    if met and comp_met:
        comparisons.append(normalize_comparison(met, comp_met))

    return comparisons


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_chain(chain):
    """Return True if the chain passes all quality checks."""
    turns = chain["turns"]

    # Q1: non-empty gene list
    if not turns[0]["answer"]:
        return False

    # Q3: non-empty comparisons
    if turns[2]["answer"] == "None":
        return False

    # Q4: must have an answer with a parsed p-value
    if turns[3]["answer"] == "None":
        return False
    if turns[3]["answer_p_value_parsed"] is None:
        return False

    return True


# ---------------------------------------------------------------------------
# Chain generation
# ---------------------------------------------------------------------------

def _build_chain(chain_idx, pmid, pmc_id, annotations, sp_lookup,
                 va_pvalues, va_pvalue_strs):
    """Build one 4-turn chain for a paper.

    Returns the chain dict, or None if it fails validation.
    """

    # --- Q1: Gene inventory ---
    unique_genes = sorted(
        annotations["Gene"].dropna().unique().tolist()
    )

    # --- Q2: Predictor inventory per gene ---
    gene_predictors = defaultdict(set)
    for _, row in annotations.iterrows():
        gene = _s(row.get("Gene"))
        predictor = _s(row.get("Variant/Haplotypes"))
        if gene and predictor:
            gene_predictors[gene].add(predictor)
    gene_predictors = {
        g: sorted(preds) for g, preds in sorted(gene_predictors.items())
    }

    # --- Q3: Comparison extraction per predictor ---
    predictor_comparisons = defaultdict(list)  # predictor -> [comp_str]
    all_annotations_with_pval = []             # for Q4

    for _, row in annotations.iterrows():
        predictor_id = _s(row["Variant/Haplotypes"])
        comp_strings = get_all_comparison_strings(row)

        for comp_str in comp_strings:
            if comp_str not in predictor_comparisons[predictor_id]:
                predictor_comparisons[predictor_id].append(comp_str)

        va_id = row["Variant Annotation ID"]
        pval = va_pvalues.get(va_id)
        if pval is not None and comp_strings:
            for comp_str in comp_strings:
                all_annotations_with_pval.append({
                    "predictor": predictor_id,
                    "comparison": comp_str,
                    "va_id": va_id,
                    "p_value": pval,
                    "p_value_str": va_pvalue_strs.get(va_id, ""),
                })

    q3_answer = (
        {k: v for k, v in sorted(predictor_comparisons.items())}
        if predictor_comparisons else "None"
    )

    # --- Q4: smallest p-value ---
    step4_pvalue_raw = None
    step4_pvalue_parsed = None

    if all_annotations_with_pval:
        min_pval = min(a["p_value"] for a in all_annotations_with_pval)
        best = [a for a in all_annotations_with_pval
                if a["p_value"] == min_pval]
        # Deduplicate by (predictor, comparison)
        seen = set()
        deduped = []
        for b in best:
            key = (b["predictor"], b["comparison"])
            if key not in seen:
                seen.add(key)
                deduped.append(b)
        if len(deduped) == 1:
            step4_answer = (
                f"{deduped[0]['predictor']} ({deduped[0]['comparison']})"
            )
        else:
            step4_answer = ", ".join(
                f"{b['predictor']} ({b['comparison']})" for b in deduped
            )
        step4_pvalue_raw = deduped[0]["p_value_str"]
        step4_pvalue_parsed = min_pval
    else:
        step4_answer = "None"

    chain = {
        "chain_id": f"variant_chain_{chain_idx:06d}",
        "pmid": str(pmid),
        "pmc_id": pmc_id,
        "source_files": sorted(annotations["_source"].unique().tolist()),
        "num_turns": 4,
        "turns": [
            {
                "turn": 1,
                "step": "gene_inventory",
                "question": (
                    f"Which gene(s) are evaluated for pharmacogenomic "
                    f"associations in this paper (PMID {pmid})?"
                ),
                "answer": unique_genes,
                "scoring": (
                    "All genes in the TSV must be present in the response."
                ),
            },
            {
                "turn": 2,
                "step": "predictor_inventory_per_gene",
                "question": (
                    "For each gene identified in Q1, list the genetic "
                    "predictors explicitly evaluated (e.g., rsID/variant, "
                    "star allele/haplotype, diplotype/genotype, or "
                    "metabolizer/phenotype group)."
                ),
                "answer": gene_predictors,
                "scoring": (
                    "All gene-predictor mappings must be present. "
                ),
            },
            {
                "turn": 3,
                "step": "comparison_extraction",
                "question": (
                    "For each predictor identified in Q2, list the explicit "
                    "comparisons tested (A vs B) as stated in the paper."
                ),
                "answer": q3_answer,
                "scoring": (
                    "All comparisons must be present per predictor. "
                    "Missing or extra comparisons fail the task."
                ),
            },
            {
                "turn": 4,
                "step": "evidence_selection",
                "question": (
                    "Among the comparisons listed in Q3, which has the "
                    "smallest reported p-value? If multiple comparisons "
                    "are tied for the smallest p-value, list all tied "
                    "comparisons."
                ),
                "answer": step4_answer,
                "answer_p_value_raw": step4_pvalue_raw,
                "answer_p_value_parsed": step4_pvalue_parsed,
                "scoring": (
                    "Exact match required. Incorrect selection or omission "
                    "fails the task."
                ),
            },
        ],
    }

    # Validation gate
    if not _validate_chain(chain):
        return None

    return chain


def generate_chains():
    var_drug, var_pheno, study_params = load_tsv_data()
    pmid_to_pmc = get_paper_pmids()

    # Build study_parameters lookup: variant_annotation_id -> rows
    sp_lookup = defaultdict(list)
    for _, row in study_params.iterrows():
        sp_lookup[row["Variant Annotation ID"]].append(row)

    # Combine annotations from both files (keep all columns, fill NaN)
    all_ann = pd.concat([var_drug, var_pheno], ignore_index=True, sort=False)

    chains = []
    chain_idx = 0
    skipped_no_paper = 0
    skipped_no_pval = 0
    skipped_validation = 0

    for pmid, group in all_ann.groupby("PMID"):
        pmid_int = int(pmid) if pd.notna(pmid) else None
        if pmid_int is None or pmid_int not in pmid_to_pmc:
            skipped_no_paper += 1
            continue

        pmc_id = pmid_to_pmc[pmid_int]

        # Check ALL variant annotation IDs have at least one p-value
        va_ids = group["Variant Annotation ID"].unique()
        va_pvalues = {}      # va_id -> best (smallest) parsed p-value
        va_pvalue_strs = {}  # va_id -> raw string for the best p-value
        all_have_pval = True

        for va_id in va_ids:
            sp_rows = sp_lookup.get(va_id, [])
            best_pval = None
            best_str = None
            for sp_row in sp_rows:
                pv = parse_pvalue(sp_row.get("P Value"))
                if pv is not None:
                    if best_pval is None or pv < best_pval:
                        best_pval = pv
                        best_str = _s(sp_row["P Value"])
            if best_pval is None:
                all_have_pval = False
                break
            va_pvalues[va_id] = best_pval
            va_pvalue_strs[va_id] = best_str

        if not all_have_pval:
            skipped_no_pval += 1
            continue

        # One chain per paper (no level splitting)
        chain = _build_chain(
            chain_idx, pmid_int, pmc_id, group, sp_lookup,
            va_pvalues, va_pvalue_strs,
        )
        if chain is not None:
            chains.append(chain)
            chain_idx += 1
        else:
            skipped_validation += 1

    return chains, skipped_no_paper, skipped_no_pval, skipped_validation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    chains, skip_paper, skip_pval, skip_valid = generate_chains()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for chain in chains:
            f.write(json.dumps(chain) + "\n")

    print(f"\nGenerated {len(chains)} chains → {OUTPUT_PATH}")
    print(f"Skipped (no paper):     {skip_paper}")
    print(f"Skipped (missing p):    {skip_pval}")
    print(f"Skipped (validation):   {skip_valid}")

    if chains:
        avg_genes = sum(
            len(c["turns"][0]["answer"]) for c in chains
        ) / len(chains)
        avg_pred = sum(
            sum(len(v) for v in c["turns"][1]["answer"].values())
            for c in chains
        ) / len(chains)
        print(f"\nAvg genes per chain:      {avg_genes:.1f}")
        print(f"Avg predictors per chain: {avg_pred:.1f}")


if __name__ == "__main__":
    main()
