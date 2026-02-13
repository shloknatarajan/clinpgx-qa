"""
Study parameter extraction question generation.

Generates free-text extraction questions that ask a model to identify
the p-value and statistical significance for a given pharmacogenomic
association. Two question types:

  1. **correct** — the association is real; answer = actual p-value + significance
  2. **modified** — one entity is swapped; answer = always "not found"

Usage:
    python -m src.modules.study_param_questions.generate
    python -m src.modules.study_param_questions.generate --seed 42 --output data/study_param_questions/study_param_questions.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.modules.study_param_questions.table_lookup import (
    StudyParamRow,
    StudyParamTableIndex,
)

NOT_FOUND = "not found"

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class StudyParamQuestion(BaseModel):
    question_id: int
    annotation_id: str
    pmid: str
    pmcid: str
    table_type: str
    # Association fields
    variant: str
    gene: str
    drug: str
    phenotype_category: str
    phenotype: str
    sentence: str
    # Question
    question_type: str  # "correct" | "modified_variant" | "modified_drug" | "modified_phenotype"
    question_text: str
    # Modification metadata (empty for correct questions)
    modification_type: str  # "" | "variant" | "drug" | "phenotype"
    original_entity: str
    modified_entity: str
    swap_source: str  # "" | "same_paper" | "bank"
    source_annotation_id: str
    # Ground truth (from the original row)
    ground_truth_p_value: str
    ground_truth_significance: str
    # Expected answers
    expected_answer_p_value: str
    expected_answer_significance: str
    generation_notes: str


# ---------------------------------------------------------------------------
# Question text template
# ---------------------------------------------------------------------------


def _build_question_text(
    variant: str,
    gene: str,
    drug: str,
    phenotype_category: str,
    phenotype: str,
    sentence: str,
) -> str:
    """Build the question prompt for a study parameter extraction question."""
    parts = [
        f"Given the following pharmacogenomic association from a published study:\n",
        f"- Variant/Haplotype: {variant}",
        f"- Gene: {gene}",
    ]
    if drug:
        parts.append(f"- Drug: {drug}")
    if phenotype_category:
        parts.append(f"- Phenotype Category: {phenotype_category}")
    if phenotype:
        parts.append(f"- Phenotype: {phenotype}")
    parts.append(f"- Association description: \"{sentence}\"")
    parts.append(
        "\nWhat is the p-value reported for this specific association, "
        "and is the association statistically significant?\n"
        "\n"
        "IMPORTANT: Respond with ONLY a JSON object — no explanation, no reasoning, "
        "no markdown code fences, no text before or after. Just the raw JSON.\n"
        "\n"
        "Use this exact format:\n"
        '{"p_value": "<value>", "significance": "<yes/no/not stated>"}\n'
        "\n"
        "If the exact association is not found in the study, respond with:\n"
        '{"p_value": "not found", "significance": "not found"}'
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entity swap functions
# ---------------------------------------------------------------------------


def _swap_variant(
    row: StudyParamRow,
    index: StudyParamTableIndex,
    rng: random.Random,
) -> tuple[str, str, str] | None:
    """Swap variant: try same-paper first, then fall back to variant bank.

    Returns (new_variant, swap_source, source_annotation_id) or None.
    """
    original = row.variant_haplotypes.strip().lower()

    # Same-paper: find a row with a different variant
    paper_rows = index.get_rows_by_pmid(row.pmid)
    candidates = [
        r
        for r in paper_rows
        if r.variant_haplotypes.strip().lower() != original
        and r.variant_annotation_id != row.variant_annotation_id
    ]
    if candidates:
        pick = rng.choice(candidates)
        return (pick.variant_haplotypes, "same_paper", pick.variant_annotation_id)

    # Bank fallback
    bank = index.get_variant_bank()
    paper_variants = {v.lower() for v in index.get_paper_variants(row.pmid)}
    bank_candidates = [v for v in bank if v.strip().lower() != original and v.strip().lower() not in paper_variants]
    if bank_candidates:
        pick = rng.choice(bank_candidates)
        return (pick, "bank", "")

    return None


def _swap_drug(
    row: StudyParamRow,
    index: StudyParamTableIndex,
    rng: random.Random,
) -> tuple[str, str, str] | None:
    """Swap drug: try same-paper first, then fall back to drug bank.

    Returns (new_drug, swap_source, source_annotation_id) or None.
    """
    original = row.drug.strip().lower()

    # Same-paper: find a row with a different drug
    paper_rows = index.get_rows_by_pmid(row.pmid)
    candidates = [
        r
        for r in paper_rows
        if r.drug.strip().lower() != original
        and r.drug.strip()
        and r.variant_annotation_id != row.variant_annotation_id
    ]
    if candidates:
        pick = rng.choice(candidates)
        return (pick.drug, "same_paper", pick.variant_annotation_id)

    # Bank fallback
    bank = index.get_drug_bank()
    paper_drugs = {d.lower() for d in index.get_paper_drugs(row.pmid)}
    bank_candidates = [d for d in bank if d.strip().lower() != original and d.strip().lower() not in paper_drugs]
    if bank_candidates:
        pick = rng.choice(bank_candidates)
        return (pick, "bank", "")

    return None


def _swap_phenotype(
    row: StudyParamRow,
    index: StudyParamTableIndex,
    rng: random.Random,
) -> tuple[str, str, str] | None:
    """Swap phenotype: try same-paper first, then fall back to phenotype bank.

    Returns (new_phenotype, swap_source, source_annotation_id) or None.
    """
    original = row.phenotype.strip().lower()

    # Same-paper: find a row with a different phenotype
    paper_rows = index.get_rows_by_pmid(row.pmid)
    candidates = [
        r
        for r in paper_rows
        if r.phenotype.strip().lower() != original
        and r.phenotype.strip()
        and r.variant_annotation_id != row.variant_annotation_id
    ]
    if candidates:
        pick = rng.choice(candidates)
        return (pick.phenotype, "same_paper", pick.variant_annotation_id)

    # Bank fallback
    bank = index.get_phenotype_bank()
    paper_phenos = {p.lower() for p in index.get_paper_phenotypes(row.pmid)}
    bank_candidates = [p for p in bank if p.strip().lower() != original and p.strip().lower() not in paper_phenos]
    if bank_candidates:
        pick = rng.choice(bank_candidates)
        return (pick, "bank", "")

    return None


# ---------------------------------------------------------------------------
# Per-row question generation
# ---------------------------------------------------------------------------


def generate_questions_for_row(
    row: StudyParamRow,
    index: StudyParamTableIndex,
    rng: random.Random,
    qid_counter: int,
) -> tuple[list[StudyParamQuestion], int]:
    """Generate 1 correct + up to 3 modified questions for a single row.

    Returns (questions, next_qid_counter).
    """
    questions: list[StudyParamQuestion] = []

    # Ground truth values
    gt_p_value = row.p_value.strip() if row.p_value.strip() else NOT_FOUND
    gt_significance = row.significance.strip() if row.significance.strip() else NOT_FOUND

    # --- 1. Correct question ---
    questions.append(
        StudyParamQuestion(
            question_id=qid_counter,
            annotation_id=row.variant_annotation_id,
            pmid=row.pmid,
            pmcid=row.pmcid,
            table_type=row.table_type,
            variant=row.variant_haplotypes,
            gene=row.gene,
            drug=row.drug,
            phenotype_category=row.phenotype_category,
            phenotype=row.phenotype,
            sentence=row.sentence,
            question_type="correct",
            question_text=_build_question_text(
                row.variant_haplotypes,
                row.gene,
                row.drug,
                row.phenotype_category,
                row.phenotype,
                row.sentence,
            ),
            modification_type="",
            original_entity="",
            modified_entity="",
            swap_source="",
            source_annotation_id="",
            ground_truth_p_value=gt_p_value,
            ground_truth_significance=gt_significance,
            expected_answer_p_value=gt_p_value,
            expected_answer_significance=gt_significance,
            generation_notes="",
        )
    )
    qid_counter += 1

    # --- 2. Modified variant question (always attempted) ---
    swap = _swap_variant(row, index, rng)
    if swap:
        new_variant, swap_source, source_ann_id = swap
        questions.append(
            StudyParamQuestion(
                question_id=qid_counter,
                annotation_id=row.variant_annotation_id,
                pmid=row.pmid,
                pmcid=row.pmcid,
                table_type=row.table_type,
                variant=new_variant,
                gene=row.gene,
                drug=row.drug,
                phenotype_category=row.phenotype_category,
                phenotype=row.phenotype,
                sentence=row.sentence,
                question_type="modified_variant",
                question_text=_build_question_text(
                    new_variant,
                    row.gene,
                    row.drug,
                    row.phenotype_category,
                    row.phenotype,
                    row.sentence,
                ),
                modification_type="variant",
                original_entity=row.variant_haplotypes,
                modified_entity=new_variant,
                swap_source=swap_source,
                source_annotation_id=source_ann_id,
                ground_truth_p_value=gt_p_value,
                ground_truth_significance=gt_significance,
                expected_answer_p_value=NOT_FOUND,
                expected_answer_significance=NOT_FOUND,
                generation_notes="",
            )
        )
        qid_counter += 1

    # --- 3. Modified drug question (only if drug is non-empty) ---
    if row.drug.strip():
        swap = _swap_drug(row, index, rng)
        if swap:
            new_drug, swap_source, source_ann_id = swap
            questions.append(
                StudyParamQuestion(
                    question_id=qid_counter,
                    annotation_id=row.variant_annotation_id,
                    pmid=row.pmid,
                    pmcid=row.pmcid,
                    table_type=row.table_type,
                    variant=row.variant_haplotypes,
                    gene=row.gene,
                    drug=new_drug,
                    phenotype_category=row.phenotype_category,
                    phenotype=row.phenotype,
                    sentence=row.sentence,
                    question_type="modified_drug",
                    question_text=_build_question_text(
                        row.variant_haplotypes,
                        row.gene,
                        new_drug,
                        row.phenotype_category,
                        row.phenotype,
                        row.sentence,
                    ),
                    modification_type="drug",
                    original_entity=row.drug,
                    modified_entity=new_drug,
                    swap_source=swap_source,
                    source_annotation_id=source_ann_id,
                    ground_truth_p_value=gt_p_value,
                    ground_truth_significance=gt_significance,
                    expected_answer_p_value=NOT_FOUND,
                    expected_answer_significance=NOT_FOUND,
                    generation_notes="",
                )
            )
            qid_counter += 1

    # --- 4. Modified phenotype question (only for pheno-table rows with non-empty phenotype) ---
    if row.table_type == "pheno" and row.phenotype.strip():
        swap = _swap_phenotype(row, index, rng)
        if swap:
            new_phenotype, swap_source, source_ann_id = swap
            questions.append(
                StudyParamQuestion(
                    question_id=qid_counter,
                    annotation_id=row.variant_annotation_id,
                    pmid=row.pmid,
                    pmcid=row.pmcid,
                    table_type=row.table_type,
                    variant=row.variant_haplotypes,
                    gene=row.gene,
                    drug=row.drug,
                    phenotype_category=row.phenotype_category,
                    phenotype=new_phenotype,
                    sentence=row.sentence,
                    question_type="modified_phenotype",
                    question_text=_build_question_text(
                        row.variant_haplotypes,
                        row.gene,
                        row.drug,
                        row.phenotype_category,
                        new_phenotype,
                        row.sentence,
                    ),
                    modification_type="phenotype",
                    original_entity=row.phenotype,
                    modified_entity=new_phenotype,
                    swap_source=swap_source,
                    source_annotation_id=source_ann_id,
                    ground_truth_p_value=gt_p_value,
                    ground_truth_significance=gt_significance,
                    expected_answer_p_value=NOT_FOUND,
                    expected_answer_significance=NOT_FOUND,
                    generation_notes="",
                )
            )
            qid_counter += 1

    return questions, qid_counter


# ---------------------------------------------------------------------------
# Simplified output helpers
# ---------------------------------------------------------------------------

SIMPLIFIED_KEYS = [
    "question_id",
    "annotation_id",
    "pmid",
    "pmcid",
    "variant",
    "gene",
    "drug",
    "phenotype_category",
    "phenotype",
    "sentence",
    "table_type",
    "question_type",
    "expected_answer_p_value",
    "expected_answer_significance",
]


def _simplify(q: StudyParamQuestion) -> dict:
    """Extract only the simplified fields from a question."""
    full = q.model_dump()
    return {k: full[k] for k in SIMPLIFIED_KEYS}


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def generate_all_study_param_questions(
    seed: int = 42,
    output_path: str = "data/study_param_questions/study_param_questions.jsonl",
) -> Path:
    """Iterate all rows, generate questions, write simplified JSONL, log stats."""
    rng = random.Random(seed)
    index = StudyParamTableIndex()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    qid = 1
    total = 0
    type_counts: dict[str, int] = {}
    swap_source_counts: dict[str, int] = {}
    correct_with_pvalue = 0
    correct_without_pvalue = 0

    with open(out, "w") as f:
        for row in tqdm(index.all_rows, desc="Generating study param questions", unit="row"):
            questions, qid = generate_questions_for_row(row, index, rng, qid)
            for q in questions:
                f.write(json.dumps(_simplify(q)) + "\n")
                total += 1
                type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
                if q.swap_source:
                    swap_source_counts[q.swap_source] = (
                        swap_source_counts.get(q.swap_source, 0) + 1
                    )
                if q.question_type == "correct":
                    if q.expected_answer_p_value != NOT_FOUND:
                        correct_with_pvalue += 1
                    else:
                        correct_without_pvalue += 1

    logger.info(f"Generated {total} study param questions → {out}")
    logger.info("Question type distribution:")
    for qtype, count in sorted(type_counts.items()):
        logger.info(f"  {qtype}: {count}")
    logger.info("Swap source distribution:")
    for source, count in sorted(swap_source_counts.items()):
        logger.info(f"  {source}: {count}")
    logger.info(
        f"Correct questions: {correct_with_pvalue} with p-value, "
        f"{correct_without_pvalue} without p-value"
    )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate study parameter extraction questions"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="data/study_param_questions/study_param_questions.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()
    generate_all_study_param_questions(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
