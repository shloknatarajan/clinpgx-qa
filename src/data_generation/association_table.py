"""
Builds two denormalized association tables by joining variant annotations
with study_parameters.tsv, scoped to articles present in variant_bench.jsonl.

Output:
  - data/drug_association_table.tsv  (var_drug_ann + study_parameters)
  - data/pheno_association_table.tsv (var_pheno_ann + study_parameters)

Each table has one row per (variant annotation, study parameter) pair.
Annotations without study parameters are preserved (null study fields).
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "variantAnnotations"

VARIANT_BENCH_PATH = DATA_DIR / "variant_bench.jsonl"
DRUG_ANN_PATH = RAW_DIR / "var_drug_ann.tsv"
PHENO_ANN_PATH = RAW_DIR / "var_pheno_ann.tsv"
STUDY_PARAMS_PATH = RAW_DIR / "study_parameters.tsv"
DRUG_OUTPUT_PATH = DATA_DIR / "drug_association_table.tsv"
PHENO_OUTPUT_PATH = DATA_DIR / "pheno_association_table.tsv"


def load_variant_bench_metadata() -> dict[str, str]:
    """Load PMID -> PMCID mapping from variant_bench.jsonl."""
    pmid_to_pmcid: dict[str, str] = {}
    with open(VARIANT_BENCH_PATH) as f:
        for line in f:
            row = json.loads(line)
            pmid_to_pmcid[str(row["pmid"]).strip()] = str(row["pmcid"]).strip()
    logger.info(f"Loaded {len(pmid_to_pmcid)} PMIDs from {VARIANT_BENCH_PATH}")
    return pmid_to_pmcid


def _load_and_filter(tsv_path: Path, pmid_to_pmcid: dict[str, str]) -> pd.DataFrame:
    """Load a single annotation TSV, filter to variant_bench PMIDs, add PMCID."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    logger.info(f"Loaded {len(df)} rows from {tsv_path}")

    df["PMID"] = df["PMID"].str.strip()
    before = len(df)
    df = df[df["PMID"].isin(pmid_to_pmcid)].copy()
    logger.info(f"Filtered to variant_bench PMIDs: {before} -> {len(df)}")

    df["PMCID"] = df["PMID"].map(pmid_to_pmcid)
    return df


def _join_and_save(
    annotations: pd.DataFrame,
    study_params: pd.DataFrame,
    output_path: Path,
    label: str,
) -> pd.DataFrame:
    """Left-join annotations with study parameters, reorder columns, save."""
    merged = annotations.merge(
        study_params,
        on="Variant Annotation ID",
        how="left",
    )

    # Reorder: put PMID, PMCID up front
    front = ["PMID", "PMCID"]
    cols = front + [c for c in merged.columns if c not in front]
    merged = merged[cols]

    # Summary stats
    n_with_study = merged["Study Parameters ID"].notna().sum()
    n_without_study = merged["Study Parameters ID"].isna().sum()
    n_unique_annotations = merged["Variant Annotation ID"].nunique()
    n_unique_pmids = merged["PMID"].nunique()

    logger.info(f"{label}: {len(merged)} total rows")
    logger.info(f"  {n_unique_annotations} unique variant annotations")
    logger.info(f"  {n_unique_pmids} unique PMIDs")
    logger.info(f"  {n_with_study} rows with study parameters")
    logger.info(f"  {n_without_study} rows without study parameters")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved to {output_path}")

    return merged


def build_association_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build drug and pheno association tables and save to TSV."""
    pmid_to_pmcid = load_variant_bench_metadata()
    study_params = pd.read_csv(STUDY_PARAMS_PATH, sep="\t", dtype=str)
    logger.info(
        f"Loaded {len(study_params)} study parameter rows from {STUDY_PARAMS_PATH}"
    )

    drug_ann = _load_and_filter(DRUG_ANN_PATH, pmid_to_pmcid)
    pheno_ann = _load_and_filter(PHENO_ANN_PATH, pmid_to_pmcid)

    drug_df = _join_and_save(
        drug_ann, study_params, DRUG_OUTPUT_PATH, "Drug associations"
    )
    pheno_df = _join_and_save(
        pheno_ann, study_params, PHENO_OUTPUT_PATH, "Pheno associations"
    )

    return drug_df, pheno_df


if __name__ == "__main__":
    drug_df, pheno_df = build_association_tables()
    print(
        f"\nDrug association table:  {len(drug_df)} rows, {len(drug_df.columns)} columns"
    )
    print(
        f"Pheno association table: {len(pheno_df)} rows, {len(pheno_df.columns)} columns"
    )
