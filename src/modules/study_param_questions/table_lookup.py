"""
Data access layer for association tables with study parameter fields.

Loads drug_association_table.tsv and pheno_association_table.tsv with the
additional study parameter columns (P Value, Significance, Study Type, etc.)
and exposes query methods for the question generation engine.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

DATA_DIR = _project_root / "data"
DRUG_TABLE_PATH = DATA_DIR / "drug_association_table.tsv"
PHENO_TABLE_PATH = DATA_DIR / "pheno_association_table.tsv"
VARIANT_BANK_PATH = DATA_DIR / "term_banks" / "variants.json"
DRUG_BANK_PATH = DATA_DIR / "term_banks" / "drugs.json"
PHENOTYPE_BANK_PATH = DATA_DIR / "term_banks" / "phenotypes.json"


class StudyParamRow(BaseModel):
    """Row from an association table including study parameter fields."""

    pmid: str
    pmcid: str
    variant_annotation_id: str
    variant_haplotypes: str
    gene: str
    drug: str
    phenotype_category: str
    phenotype: str
    direction_of_effect: str
    sentence: str
    table_type: str  # "drug" | "pheno"
    # Study parameter fields
    significance: str
    study_parameters_id: str
    study_type: str
    p_value: str
    ratio_stat_type: str
    ratio_stat: str
    confidence_interval_start: str
    confidence_interval_stop: str


def _load_table(path: Path, table_type: str) -> list[StudyParamRow]:
    """Read a single association TSV and return rows with study parameters."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows: list[StudyParamRow] = []
    for _, r in df.iterrows():
        phenotype = ""
        if table_type == "pheno":
            phenotype = str(r.get("Phenotype", ""))

        rows.append(
            StudyParamRow(
                pmid=str(r["PMID"]),
                pmcid=str(r["PMCID"]),
                variant_annotation_id=str(r["Variant Annotation ID"]),
                variant_haplotypes=str(r["Variant/Haplotypes"]),
                gene=str(r["Gene"]),
                drug=str(r.get("Drug(s)", "")),
                phenotype_category=str(r.get("Phenotype Category", "")),
                phenotype=phenotype,
                direction_of_effect=str(r.get("Direction of effect", "")),
                sentence=str(r.get("Sentence", "")),
                table_type=table_type,
                significance=str(r.get("Significance", "")),
                study_parameters_id=str(r.get("Study Parameters ID", "")),
                study_type=str(r.get("Study Type", "")),
                p_value=str(r.get("P Value", "")),
                ratio_stat_type=str(r.get("Ratio Stat Type", "")),
                ratio_stat=str(r.get("Ratio Stat", "")),
                confidence_interval_start=str(r.get("Confidence Interval Start", "")),
                confidence_interval_stop=str(r.get("Confidence Interval Stop", "")),
            )
        )
    return rows


class StudyParamTableIndex:
    """In-memory index over both association tables and term banks."""

    def __init__(self) -> None:
        logger.info("Loading association tables with study parameters …")

        drug_rows = _load_table(DRUG_TABLE_PATH, "drug")
        pheno_rows = _load_table(PHENO_TABLE_PATH, "pheno")
        self._all_rows = drug_rows + pheno_rows
        logger.info(
            f"  drug rows: {len(drug_rows)}, pheno rows: {len(pheno_rows)}, "
            f"total: {len(self._all_rows)}"
        )

        # Index by PMID
        self._by_pmid: dict[str, list[StudyParamRow]] = {}
        for row in self._all_rows:
            self._by_pmid.setdefault(row.pmid, []).append(row)

        # Index by Variant Annotation ID
        self._by_annotation_id: dict[str, StudyParamRow] = {}
        for row in self._all_rows:
            self._by_annotation_id[row.variant_annotation_id] = row

        # Term banks
        with open(VARIANT_BANK_PATH) as f:
            self._variant_bank: list[str] = json.load(f)
        with open(DRUG_BANK_PATH) as f:
            self._drug_bank: list[str] = json.load(f)
        with open(PHENOTYPE_BANK_PATH) as f:
            self._phenotype_bank: list[str] = json.load(f)
        logger.info(
            f"  variant bank: {len(self._variant_bank)}, "
            f"drug bank: {len(self._drug_bank)}, "
            f"phenotype bank: {len(self._phenotype_bank)}"
        )

    @property
    def all_rows(self) -> list[StudyParamRow]:
        return self._all_rows

    def get_rows_by_pmid(self, pmid: str) -> list[StudyParamRow]:
        return self._by_pmid.get(pmid, [])

    def get_row_by_annotation_id(self, ann_id: str) -> StudyParamRow | None:
        return self._by_annotation_id.get(ann_id)

    def get_paper_variants(self, pmid: str) -> set[str]:
        """All individual variants (split comma-separated) for a PMID."""
        variants: set[str] = set()
        for row in self.get_rows_by_pmid(pmid):
            for v in row.variant_haplotypes.split(","):
                v = v.strip()
                if v:
                    variants.add(v)
        return variants

    def get_paper_drugs(self, pmid: str) -> set[str]:
        """All individual drugs (split comma-separated) for a PMID."""
        drugs: set[str] = set()
        for row in self.get_rows_by_pmid(pmid):
            for d in row.drug.split(","):
                d = d.strip()
                if d:
                    drugs.add(d)
        return drugs

    def get_paper_phenotypes(self, pmid: str) -> set[str]:
        """All individual phenotypes (split comma-separated) for a PMID."""
        phenotypes: set[str] = set()
        for row in self.get_rows_by_pmid(pmid):
            for p in row.phenotype.split(","):
                p = p.strip()
                if p:
                    phenotypes.add(p)
        return phenotypes

    def get_variant_bank(self) -> list[str]:
        return self._variant_bank

    def get_drug_bank(self) -> list[str]:
        return self._drug_bank

    def get_phenotype_bank(self) -> list[str]:
        return self._phenotype_bank
