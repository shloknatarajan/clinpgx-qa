#!/usr/bin/env python3
"""
Generate chained benchmark questions for variant and allele-level reasoning.

Uses var_drug_ann.tsv and var_pheno_ann.tsv, restricted to papers with
markdown files in data/papers/ and where ALL variants have p-values in
study_parameters.tsv.

Each chain has 4 steps:
  1. Predictor Identification
  2. Genetic Comparison Resolution
  3. Lowest-Level Comparison Extraction
  4. Evidence-Based Selection (smallest p-value)

Design notes:
  - Variant/Haplotypes values are treated as atomic identifiers exactly as
    encoded in ClinPGx. No comma-splitting or biological deduplication.
  - Comparison-type classification relies on structured columns first
    (Comparison Allele(s) or Genotype(s), Comparison Metabolizer types,
    Alleles), with Sentence prefix as a last-resort fallback.
  - Comparison strings are canonicalized: items within each side are sorted
    lexicographically and joined with " + ", then "lhs vs rhs".
  - Chains are only emitted if they pass a validation checklist (non-empty
    Step 1, non-empty Step 3, at least one Step 4 comparison with a parsed
    p-value).
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
# Classification  (structured columns first, sentence fallback)
# ---------------------------------------------------------------------------

def classify_comparison(row):
    """Return a list of comparison types for one annotation row.

    Priority order:
      1. If Comparison Metabolizer types populated → metabolizer_or_phenotype_group
      2. If Comparison Allele(s) or Genotype(s) populated →
           Alleles field contains '/' or multi-nucleotide → genotype
           else → allele
      3. If neither → not_specified

    For dual-comparison rows (both fields populated), both types are returned.
    """
    comp_alleles = _s(row.get("Comparison Allele(s) or Genotype(s)"))
    comp_met = _s(row.get("Comparison Metabolizer types"))

    if not comp_alleles and not comp_met:
        return ["not_specified"]

    types = []

    # 1. Metabolizer comparison
    if comp_met:
        types.append("metabolizer_or_phenotype_group")

    # 2. Allele / genotype comparison — structured columns first
    if comp_alleles:
        alleles = _s(row.get("Alleles"))

        # Check structured Alleles field for diplotype or multi-nucleotide
        allele_parts = [a.strip() for a in alleles.split("+")]
        has_diplotype = any("/" in a for a in allele_parts)
        is_multi_nuc = bool(re.match(r"^[ATGC]{2,}$", alleles))

        if has_diplotype or is_multi_nuc:
            types.append("genotype")
        elif alleles:
            # Single allele letter or single haplotype → allele
            types.append("allele")
        else:
            # Alleles field empty but comparison exists — sentence fallback
            sentence = _s(row.get("Sentence")).lower()
            if sentence.startswith("genotype") or sentence.startswith("genotypes"):
                types.append("genotype")
            else:
                types.append("allele")

    return types


LEVEL_PRIORITY = {
    "allele": 0,
    "genotype": 1,
    "metabolizer_or_phenotype_group": 2,
    "not_specified": 3,
}


def get_comparison_string(row, comp_type):
    """Build a canonical 'X vs Y' comparison string for the given type."""
    if comp_type == "metabolizer_or_phenotype_group":
        met = _s(row.get("Metabolizer types"))
        comp_met = _s(row.get("Comparison Metabolizer types"))
        if met and comp_met:
            return normalize_comparison(met, comp_met)
        return None

    alleles = _s(row.get("Alleles"))
    comp = _s(row.get("Comparison Allele(s) or Genotype(s)"))
    if alleles and comp:
        return normalize_comparison(alleles, comp)
    return None


def get_relation_context(annotations):
    """Derive the drug / phenotype relation for question phrasing."""
    drugs = set()
    phenotypes = set()
    for _, row in annotations.iterrows():
        d = _s(row.get("Drug(s)"))
        if d:
            drugs.add(d)
        p = _s(row.get("Phenotype"))
        if p:
            phenotypes.add(p)

    if drugs:
        return ", ".join(sorted(drugs))
    if phenotypes:
        return ", ".join(sorted(phenotypes))
    return "pharmacogenomic"


LEVEL_LABEL = {
    "allele": "allele-level",
    "genotype": "genotype-level",
    "metabolizer_or_phenotype_group": "metabolizer or phenotype-group",
    "not_specified": "genetic",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_chain(chain):
    """Return True if the chain passes all quality checks."""
    turns = chain["turns"]

    # Step 1: non-empty answer
    if not turns[0]["answer"]:
        return False

    # Step 3: non-empty answer (must not be "None")
    if turns[2]["answer"] == "None":
        return False

    # Step 4: must have an answer with a parsed p-value
    if turns[3]["answer"] == "None":
        return False
    if turns[3]["answer_p_value_parsed"] is None:
        return False

    return True


# ---------------------------------------------------------------------------
# Chain generation
# ---------------------------------------------------------------------------

def _build_chain(chain_idx, pmid, pmc_id, annotations, sp_lookup,
                 va_pvalues, va_pvalue_strs, focus_level):
    """Build one 4-turn chain for a paper at a specific comparison level.

    Returns the chain dict, or None if it fails validation.
    """

    # --- Step 1: unique predictors (atomic Variant/Haplotypes values) ---
    unique_predictors = sorted(
        annotations["Variant/Haplotypes"].dropna().unique().tolist()
    )

    # --- Step 2: comparison types per predictor ---
    predictor_comp_types = defaultdict(set)
    for _, row in annotations.iterrows():
        predictor_id = _s(row["Variant/Haplotypes"])
        for ct in classify_comparison(row):
            predictor_comp_types[predictor_id].add(ct)
    predictor_comp_types = {
        k: sorted(v) for k, v in predictor_comp_types.items()
    }

    # --- Step 3: comparisons at focus_level ---
    level_comparisons = defaultdict(list)   # predictor -> [comp_str]
    level_annotations = []                  # for Step 4

    for _, row in annotations.iterrows():
        row_types = classify_comparison(row)
        if focus_level not in row_types:
            continue
        predictor_id = _s(row["Variant/Haplotypes"])
        comp_str = get_comparison_string(row, focus_level)
        if comp_str and comp_str not in level_comparisons[predictor_id]:
            level_comparisons[predictor_id].append(comp_str)

        va_id = row["Variant Annotation ID"]
        pval = va_pvalues.get(va_id)
        if pval is not None and comp_str:
            level_annotations.append({
                "predictor": predictor_id,
                "comparison": comp_str,
                "va_id": va_id,
                "p_value": pval,
                "p_value_str": va_pvalue_strs.get(va_id, ""),
            })

    level_comparisons = dict(level_comparisons) if level_comparisons else "None"

    # --- Step 4: smallest p-value ---
    step4_pvalue_raw = None
    step4_pvalue_parsed = None

    if level_annotations:
        min_pval = min(a["p_value"] for a in level_annotations)
        best = [a for a in level_annotations if a["p_value"] == min_pval]
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

    relation = get_relation_context(annotations)
    level_name = LEVEL_LABEL[focus_level]

    chain = {
        "chain_id": f"variant_chain_{chain_idx:06d}",
        "pmid": str(pmid),
        "pmc_id": pmc_id,
        "source_files": sorted(annotations["_source"].unique().tolist()),
        "num_turns": 4,
        "relation_context": relation,
        "focus_comparison_level": focus_level,
        "turns": [
            {
                "turn": 1,
                "step": "predictor_identification",
                "question": (
                    f"Which genetic predictors (variants, haplotypes, "
                    f"genotypes, or phenotype groups) are explicitly reported "
                    f"in this paper (PMID {pmid}) as being evaluated for "
                    f"{relation}-related outcomes? Respond with all that are "
                    f"reported."
                ),
                "answer": unique_predictors,
                "scoring": (
                    "All predictors in the TSV must be present in the "
                    "response. Extra predictors from the paper not in the "
                    "TSV do not count against the score. "
                    "Variant/Haplotypes values are treated as atomic "
                    "identifiers exactly as encoded in ClinPGx."
                ),
            },
            {
                "turn": 2,
                "step": "comparison_type_classification",
                "question": (
                    "For each predictor identified in Step 1, which type(s) "
                    "of genetic comparison are reported? Select all that "
                    "apply per predictor: allele-level comparison, "
                    "genotype-level comparison, metabolizer or "
                    "phenotype-group comparison, genetic comparison not "
                    "specified."
                ),
                "answer": predictor_comp_types,
                "scoring": (
                    "All predictors must be correctly classified. Any "
                    "misclassification fails the task."
                ),
            },
            {
                "turn": 3,
                "step": "lowest_level_extraction",
                "question": (
                    f"For predictors that have {level_name} comparisons "
                    f"(only predictors classified as having {level_name} "
                    f"comparisons in Step 2 are eligible for this step), "
                    f"list all {level_name} comparisons explicitly reported "
                    f"in the paper. If no {level_name} comparisons are "
                    f"reported, answer: None."
                ),
                "answer": level_comparisons,
                "scoring": (
                    "Exact match required. Missing or extra comparisons "
                    "fail the task."
                ),
            },
            {
                "turn": 4,
                "step": "evidence_based_selection",
                "question": (
                    f"Among the {level_name} comparisons reported in Step 3: "
                    f"Which comparison has the smallest reported p-value "
                    f"(strongest statistical evidence), regardless of "
                    f"association direction? If multiple comparisons are "
                    f"tied for the smallest p-value, list all tied "
                    f"comparisons."
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

        # Collect all comparison types across annotations
        all_types = set()
        has_dual = False
        for _, row in group.iterrows():
            ctypes = classify_comparison(row)
            all_types.update(ctypes)
            if len(ctypes) > 1:
                has_dual = True

        # Determine levels to generate chains for
        levels_present = sorted(
            [t for t in all_types if t != "not_specified"],
            key=lambda t: LEVEL_PRIORITY[t],
        )

        if has_dual:
            # Generate separate chains for each comparison perspective
            for level in levels_present:
                chain = _build_chain(
                    chain_idx, pmid_int, pmc_id, group, sp_lookup,
                    va_pvalues, va_pvalue_strs, level,
                )
                if chain is not None:
                    chains.append(chain)
                    chain_idx += 1
                else:
                    skipped_validation += 1
        else:
            # Single chain at the most granular level
            if levels_present:
                focus = levels_present[0]
            else:
                focus = "not_specified"
            chain = _build_chain(
                chain_idx, pmid_int, pmc_id, group, sp_lookup,
                va_pvalues, va_pvalue_strs, focus,
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
        levels = defaultdict(int)
        for c in chains:
            levels[c["focus_comparison_level"]] += 1
        print("\nFocus-level distribution:")
        for level, count in sorted(levels.items()):
            print(f"  {level}: {count}")

        avg_v = sum(len(c["turns"][0]["answer"]) for c in chains) / len(chains)
        print(f"\nAvg predictors per chain: {avg_v:.1f}")


if __name__ == "__main__":
    main()
