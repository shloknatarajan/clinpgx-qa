"""
Data access layer for association tables and term banks.

Loads drug_association_table.tsv, pheno_association_table.tsv, and
term_banks/{variants,drugs,phenotypes}.json once, and exposes query
methods for the distractor generation engines.
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


class AssociationRow(BaseModel):
    """Normalised row from either association table."""

    pmid: str
    pmcid: str
    variant_annotation_id: str
    variant_haplotypes: str  # raw comma-separated
    gene: str
    drug: str
    phenotype_category: str
    phenotype: str  # empty for drug table
    direction_of_effect: str
    sentence: str
    table_type: str  # "drug" | "pheno"


def _load_table(path: Path, table_type: str) -> list[AssociationRow]:
    """Read a single association TSV and return normalised rows."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows: list[AssociationRow] = []
    for _, r in df.iterrows():
        phenotype = ""
        if table_type == "pheno":
            phenotype = str(r.get("Phenotype", ""))

        drug_col = "Drug(s)"
        direction_col = "Direction of effect"

        rows.append(
            AssociationRow(
                pmid=str(r["PMID"]),
                pmcid=str(r["PMCID"]),
                variant_annotation_id=str(r["Variant Annotation ID"]),
                variant_haplotypes=str(r["Variant/Haplotypes"]),
                gene=str(r["Gene"]),
                drug=str(r.get(drug_col, "")),
                phenotype_category=str(r.get("Phenotype Category", "")),
                phenotype=phenotype,
                direction_of_effect=str(r.get(direction_col, "")),
                sentence=str(r.get("Sentence", "")),
                table_type=table_type,
            )
        )
    return rows


class AssociationTableIndex:
    """In-memory index over both association tables and term banks."""

    def __init__(self) -> None:
        logger.info("Loading association tables and term banks …")

        drug_rows = _load_table(DRUG_TABLE_PATH, "drug")
        pheno_rows = _load_table(PHENO_TABLE_PATH, "pheno")
        self._all_rows = drug_rows + pheno_rows
        logger.info(
            f"  drug rows: {len(drug_rows)}, pheno rows: {len(pheno_rows)}, "
            f"total: {len(self._all_rows)}"
        )

        # Index by PMID
        self._by_pmid: dict[str, list[AssociationRow]] = {}
        for row in self._all_rows:
            self._by_pmid.setdefault(row.pmid, []).append(row)

        # Index by Variant Annotation ID
        self._by_annotation_id: dict[str, AssociationRow] = {}
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

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    @property
    def all_rows(self) -> list[AssociationRow]:
        return self._all_rows

    def get_rows_by_pmid(self, pmid: str) -> list[AssociationRow]:
        return self._by_pmid.get(pmid, [])

    def get_row_by_annotation_id(self, ann_id: str) -> AssociationRow | None:
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

    def get_variant_bank(self) -> list[str]:
        return self._variant_bank

    def get_paper_phenotypes(self, pmid: str) -> set[str]:
        """All individual phenotypes (split comma-separated) for a PMID."""
        phenotypes: set[str] = set()
        for row in self.get_rows_by_pmid(pmid):
            for p in row.phenotype.split(","):
                p = p.strip()
                if p:
                    phenotypes.add(p)
        return phenotypes

    def get_drug_bank(self) -> list[str]:
        return self._drug_bank

    def get_phenotype_bank(self) -> list[str]:
        return self._phenotype_bank


# ---------------------------------------------------------------------------
# Simplified MCQ writer
# ---------------------------------------------------------------------------

OPTION_LABELS = ["a", "b", "c", "d"]


def write_simplified_mcqs(
    detailed_path: Path,
    answer_key: str,
) -> Path:
    """Read a detailed MCQ JSONL and write a simplified version next to it.

    The simplified file lives in a ``simplified/`` subdirectory and replaces the
    full options list with ``option_a`` … ``option_d`` plus ``correct_answer``.

    For each row, two entries are written:
      1. **standard** — the original 4 options (one correct, three distractors).
      2. **none_of_the_above** — the 3 distractors as options a–c, with
         option_d = "None of the above" and correct_answer = "d".
    """
    out_dir = detailed_path.parent / "simplified"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / detailed_path.name

    total = 0
    with open(detailed_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)

            base: dict = {}
            for key in row:
                if key in ("options", "generation_notes"):
                    continue
                base[key] = row[key]

            # --- standard entry ---
            standard = {**base, "question_type": "standard"}
            correct_answer = None
            for i, opt in enumerate(row["options"]):
                label = OPTION_LABELS[i]
                standard[f"option_{label}"] = opt[answer_key]
                if opt["role"] == "correct":
                    correct_answer = label
            standard["correct_answer"] = correct_answer
            fout.write(json.dumps(standard) + "\n")
            total += 1

            # --- none-of-the-above entry ---
            distractors = [opt for opt in row["options"] if opt["role"] != "correct"]
            if len(distractors) >= 3:
                nota = {**base, "question_type": "none_of_the_above"}
                for i, opt in enumerate(distractors[:3]):
                    nota[f"option_{OPTION_LABELS[i]}"] = opt[answer_key]
                nota["option_d"] = "None of the above"
                nota["correct_answer"] = "d"
                fout.write(json.dumps(nota) + "\n")
                total += 1

    logger.info(f"Simplified {total} rows → {out_path}")
    return out_path
