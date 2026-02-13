"""
Drug MCQ distractor generation engine.

For each association row, generates exactly 4 multiple-choice options:
  1. The correct drug (ground truth)
  2–3. Same-paper distractors with different associations
  4. Drug-bank distractor chosen by Jaccard similarity

Fallback: when fewer than 2 same-paper distractors are found, remaining
slots are filled from the drug bank.

Usage:
    python -m src.modules.mcq_generator.drug_modifier
    python -m src.modules.mcq_generator.drug_modifier --seed 42 --output data/mcq_options/drug_mcq_options.jsonl
"""

import argparse
import random
import re
import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.modules.mcq_generator.table_lookup import (
    AssociationRow,
    AssociationTableIndex,
    write_simplified_mcqs,
)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class DistractorOption(BaseModel):
    drug: str
    role: str  # "correct" | "same_paper" | "drug_bank"
    source_annotation_id: str | None = None
    jaccard_score: float | None = None
    association_differs_on: str | None = None  # "variant", "direction", "phenotype"


class DrugMCQOptions(BaseModel):
    annotation_id: str
    pmid: str
    pmcid: str
    drug: str
    original_sentence: str
    blanked_sentence: str
    table_type: str
    options: list[DistractorOption]  # exactly 4
    generation_notes: list[str]


# ---------------------------------------------------------------------------
# Sentence blanking
# ---------------------------------------------------------------------------


def blank_drug_in_sentence(sentence: str, drug: str) -> str:
    """Replace the drug name(s) in a templated annotation sentence with ______.

    Multi-drug entries (e.g. "drug1, drug2, drug3") appear in sentences as
    "drug1, drug2 and drug3" or "drug1, drug2 or drug3".  We find the span
    covering all individual drug names and replace it.
    """
    parts = [d.strip() for d in drug.split(",") if d.strip()]
    if not parts:
        return sentence

    # Find the leftmost start and rightmost end of any individual drug
    first_start = len(sentence)
    last_end = 0
    for p in parts:
        idx = sentence.find(p)
        if idx != -1:
            first_start = min(first_start, idx)
            last_end = max(last_end, idx + len(p))

    if last_end > first_start:
        return sentence[:first_start] + "______" + sentence[last_end:]

    return sentence


# ---------------------------------------------------------------------------
# Jaccard similarity for drugs
# ---------------------------------------------------------------------------


def tokenize_drug(drug: str) -> set[str]:
    """Tokenize a drug name for Jaccard similarity.

    Splits on whitespace, hyphens, commas, and common suffixes.
    Also adds character 3-grams for short single-word drugs.
    """
    tokens: set[str] = set()
    parts = re.split(r"[\s\-\,\/]+", drug)
    for p in parts:
        p = p.strip()
        if p:
            tokens.add(p.lower())

    # For single-word drugs, add character 3-grams to capture suffix similarity
    # (e.g. -statin, -prazole, -olol)
    cleaned = drug.strip().lower()
    if " " not in cleaned and len(cleaned) >= 4:
        for i in range(len(cleaned) - 2):
            tokens.add(f"d3g_{cleaned[i : i + 3]}")

    return tokens


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Drug similarity index
# ---------------------------------------------------------------------------


class DrugSimilarityIndex:
    """Precomputed token sets for all bank drugs, supports fast lookup."""

    def __init__(self, bank: list[str]) -> None:
        self._bank = bank
        self._token_sets: list[tuple[str, set[str]]] = [
            (d, tokenize_drug(d)) for d in bank
        ]

    def find_most_similar(
        self,
        query: str,
        exclude: set[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return up to top_k most similar bank drugs, excluding those in exclude."""
        query_tokens = tokenize_drug(query)
        scored: list[tuple[str, float]] = []
        for d, d_tokens in self._token_sets:
            if d in exclude:
                continue
            score = jaccard_similarity(query_tokens, d_tokens)
            scored.append((d, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Drug overlap helpers (Fix 3)
# ---------------------------------------------------------------------------


def _drug_parts(drug: str) -> set[str]:
    """Split a drug field into a set of individual drug names (lowercased)."""
    return {d.strip().lower() for d in drug.split(",") if d.strip()}


def _drugs_overlap(a: str, b: str) -> bool:
    """Return True if any individual drug component appears in both strings."""
    return bool(_drug_parts(a) & _drug_parts(b))


# ---------------------------------------------------------------------------
# Same-paper distractor selection
# ---------------------------------------------------------------------------


def _association_differs_on(
    target: AssociationRow, candidate: AssociationRow
) -> str | None:
    """Return the first field on which the candidate's association differs, or None."""
    if candidate.variant_haplotypes.lower() != target.variant_haplotypes.lower():
        return "variant"
    if candidate.direction_of_effect.lower() != target.direction_of_effect.lower():
        return "direction"
    if candidate.phenotype.lower() != target.phenotype.lower() and (
        target.phenotype or candidate.phenotype
    ):
        return "phenotype"
    return None


def _get_same_paper_distractors(
    target: AssociationRow,
    index: AssociationTableIndex,
) -> list[DistractorOption]:
    """Find distractors from the same paper with different drug AND different association."""
    paper_rows = index.get_rows_by_pmid(target.pmid)
    seen_drugs: set[str] = set()
    distractors: list[DistractorOption] = []

    for row in paper_rows:
        # Must be a different drug
        if row.drug == target.drug:
            continue
        # Skip if drug components overlap
        if _drugs_overlap(row.drug, target.drug):
            continue
        # Must differ on at least one other association field
        differs_on = _association_differs_on(target, row)
        if differs_on is None:
            continue
        # Deduplicate by drug string
        if row.drug in seen_drugs:
            continue
        seen_drugs.add(row.drug)
        distractors.append(
            DistractorOption(
                drug=row.drug,
                role="same_paper",
                source_annotation_id=row.variant_annotation_id,
                association_differs_on=differs_on,
            )
        )

    return distractors


# ---------------------------------------------------------------------------
# MCQ generation for a single row
# ---------------------------------------------------------------------------


def generate_mcq_for_row(
    row: AssociationRow,
    index: AssociationTableIndex,
    sim_index: DrugSimilarityIndex,
    rng: random.Random,
) -> DrugMCQOptions:
    """Generate exactly 4 MCQ options for a single association row."""
    notes: list[str] = []

    # 1. Correct answer
    correct = DistractorOption(
        drug=row.drug,
        role="correct",
        source_annotation_id=row.variant_annotation_id,
    )

    # 2–3. Same-paper distractors
    same_paper = _get_same_paper_distractors(row, index)
    rng.shuffle(same_paper)

    chosen_same_paper = same_paper[:2]
    remaining_slots = 3 - len(chosen_same_paper)

    if len(chosen_same_paper) < 2:
        notes.append(
            f"only {len(chosen_same_paper)} same-paper distractor(s) found; "
            f"filling {remaining_slots} slot(s) from drug bank"
        )

    # Drugs to exclude from bank: correct + all paper drugs
    paper_drugs = index.get_paper_drugs(row.pmid)
    exclude = paper_drugs | {row.drug}

    # 4 + fallback: drug bank distractors
    bank_needed = remaining_slots
    bank_candidates = sim_index.find_most_similar(
        row.drug,
        exclude=exclude,
        top_k=bank_needed + 10,  # increased to compensate for overlap filtering
    )
    # Filter out candidates with overlapping drug components
    bank_candidates = [
        (d, s) for d, s in bank_candidates if not _drugs_overlap(d, row.drug)
    ]

    bank_distractors: list[DistractorOption] = []
    for drug, score in bank_candidates[:bank_needed]:
        bank_distractors.append(
            DistractorOption(
                drug=drug,
                role="drug_bank",
                jaccard_score=round(score, 4),
            )
        )

    if len(bank_distractors) < bank_needed:
        notes.append(
            f"only {len(bank_distractors)} drug bank distractor(s) available "
            f"(needed {bank_needed})"
        )

    # Assemble all 4 options
    options = [correct] + chosen_same_paper + bank_distractors
    if len(options) < 4:
        notes.append(f"could only produce {len(options)} options (expected 4)")
    options = options[:4]

    # Shuffle so correct answer isn't always first
    rng.shuffle(options)

    return DrugMCQOptions(
        annotation_id=row.variant_annotation_id,
        pmid=row.pmid,
        pmcid=row.pmcid,
        drug=row.drug,
        original_sentence=row.sentence,
        blanked_sentence=blank_drug_in_sentence(row.sentence, row.drug),
        table_type=row.table_type,
        options=options,
        generation_notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def generate_all_drug_mcqs(
    seed: int = 42,
    output_path: str = "data/mcq_options/drug_mcq_options.jsonl",
) -> Path:
    """Iterate association rows with non-empty drug, generate MCQs, write JSONL."""
    rng = random.Random(seed)
    index = AssociationTableIndex()
    bank = index.get_drug_bank()
    sim_index = DrugSimilarityIndex(bank)
    logger.info(f"Drug similarity index built with {len(bank)} entries")

    # Only process rows with a non-empty drug field
    drug_rows = [r for r in index.all_rows if r.drug.strip()]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    blank_skipped = 0
    notes_counter: dict[str, int] = {}

    with open(out, "w") as f:
        for row in tqdm(drug_rows, desc="Generating drug MCQs", unit="row"):
            mcq = generate_mcq_for_row(row, index, sim_index, rng)
            if mcq.blanked_sentence == mcq.original_sentence:
                blank_skipped += 1
                continue
            f.write(mcq.model_dump_json() + "\n")
            total += 1
            for note in mcq.generation_notes:
                key = note.split(";")[0].split("(")[0].strip()
                notes_counter[key] = notes_counter.get(key, 0) + 1

    if blank_skipped:
        logger.info(f"Skipped {blank_skipped} rows where blanking failed")
    logger.info(f"Generated {total} drug MCQ entries → {out}")
    if notes_counter:
        logger.info("Generation notes distribution:")
        for key, count in sorted(notes_counter.items(), key=lambda x: -x[1]):
            logger.info(f"  {key}: {count}")

    write_simplified_mcqs(out, answer_key="drug", seed=seed, start_id=30001)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate drug MCQ distractor options")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="data/mcq_options/drug_mcq_options.jsonl",
        help="Output JSONL path (default: data/mcq_options/drug_mcq_options.jsonl)",
    )
    args = parser.parse_args()
    generate_all_drug_mcqs(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
