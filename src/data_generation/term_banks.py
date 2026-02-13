"""
Find all possible terms for:
- Variants
- (Gene, Variant) pairs
- Drugs
- Phenotypes
Use var_drug_ann.tsv and var_pheno_ann.tsv to find all possible terms.
Save the banks to new files in `data/term_banks/` directory.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "variantAnnotations"
OUTPUT_DIR = DATA_DIR / "term_banks"

DRUG_ANN_PATH = RAW_DIR / "var_drug_ann.tsv"
PHENO_ANN_PATH = RAW_DIR / "var_pheno_ann.tsv"


def _split_comma_separated(series: pd.Series) -> set[str]:
    """Split a comma-separated column into unique stripped terms."""
    terms: set[str] = set()
    for value in series.dropna():
        for part in value.split(","):
            term = part.strip().strip('"')
            if term and not term.isdigit():
                terms.add(term)
    return terms


def _save_bank(terms: list[str], path: Path, label: str) -> None:
    """Save a sorted list of terms as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(terms, f, indent=2)
    logger.info(f"{label}: {len(terms)} unique terms -> {path}")


def build_term_banks() -> dict[str, list[str]]:
    """Extract unique terms from var_drug_ann and var_pheno_ann, save to JSON files."""
    drug_df = pd.read_csv(DRUG_ANN_PATH, sep="\t", dtype=str)
    pheno_df = pd.read_csv(PHENO_ANN_PATH, sep="\t", dtype=str)
    logger.info(f"Loaded {len(drug_df)} drug rows, {len(pheno_df)} pheno rows")

    # --- Variants ---
    variants = _split_comma_separated(
        drug_df["Variant/Haplotypes"]
    ) | _split_comma_separated(pheno_df["Variant/Haplotypes"])
    variants_sorted = sorted(variants)

    # --- (Gene, Variant) pairs ---
    gene_variant_pairs: set[tuple[str, str]] = set()
    for df in (drug_df, pheno_df):
        for _, row in df.dropna(subset=["Gene", "Variant/Haplotypes"]).iterrows():
            gene = row["Gene"].strip()
            for variant in row["Variant/Haplotypes"].split(","):
                variant = variant.strip()
                if variant:
                    gene_variant_pairs.add((gene, variant))
    gene_variant_sorted = sorted(
        [{"gene": g, "variant": v} for g, v in gene_variant_pairs],
        key=lambda x: (x["gene"], x["variant"]),
    )

    # --- Drugs ---
    drugs = _split_comma_separated(drug_df["Drug(s)"]) | _split_comma_separated(
        pheno_df["Drug(s)"]
    )
    drugs_sorted = sorted(drugs)

    # --- Phenotypes ---
    # Drug annotations use "Phenotype Category" (e.g. "Efficacy", "Toxicity")
    phenotype_categories = _split_comma_separated(drug_df["Phenotype Category"])
    # Pheno annotations use "Phenotype" (e.g. "Side Effect:Stevens-Johnson Syndrome")
    phenotypes = _split_comma_separated(pheno_df["Phenotype"])
    all_phenotypes = sorted(phenotype_categories | phenotypes)

    # --- Save ---
    _save_bank(variants_sorted, OUTPUT_DIR / "variants.json", "Variants")
    _save_bank(drugs_sorted, OUTPUT_DIR / "drugs.json", "Drugs")
    _save_bank(all_phenotypes, OUTPUT_DIR / "phenotypes.json", "Phenotypes")

    # Gene-variant pairs are dicts, save directly
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "gene_variant_pairs.json", "w") as f:
        json.dump(gene_variant_sorted, f, indent=2)
    logger.info(
        f"Gene-Variant pairs: {len(gene_variant_sorted)} unique pairs -> {OUTPUT_DIR / 'gene_variant_pairs.json'}"
    )

    return {
        "variants": variants_sorted,
        "gene_variant_pairs": gene_variant_sorted,
        "drugs": drugs_sorted,
        "phenotypes": all_phenotypes,
    }


if __name__ == "__main__":
    banks = build_term_banks()
    print(f"\nVariants:           {len(banks['variants'])}")
    print(f"Gene-Variant pairs: {len(banks['gene_variant_pairs'])}")
    print(f"Drugs:              {len(banks['drugs'])}")
    print(f"Phenotypes:         {len(banks['phenotypes'])}")
