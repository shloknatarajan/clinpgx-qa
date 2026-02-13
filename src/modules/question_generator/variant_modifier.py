"""
Variant MCQ distractor generation engine.

For each association row, generates exactly 4 multiple-choice options:
  1. The correct variant (ground truth)
  2–3. Same-paper distractors with different associations
  4. Variant-bank distractor chosen by Jaccard similarity

Fallback: when fewer than 2 same-paper distractors are found, remaining
slots are filled from the variant bank.

Usage:
    python -m src.modules.question_generator.variant_modifier
    python -m src.modules.question_generator.variant_modifier --seed 42 --output data/mcq_options/variant_mcq_options.jsonl
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

from src.modules.question_generator.table_lookup import (
    AssociationRow,
    AssociationTableIndex,
    write_simplified_mcqs,
)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class DistractorOption(BaseModel):
    variant: str
    role: str  # "correct" | "same_paper" | "variant_bank"
    source_annotation_id: str | None = None
    jaccard_score: float | None = None
    association_differs_on: str | None = None  # "drug", "direction", "phenotype"


class VariantMCQOptions(BaseModel):
    annotation_id: str
    pmid: str
    pmcid: str
    variant: str
    original_sentence: str
    blanked_sentence: str
    table_type: str
    options: list[DistractorOption]  # exactly 4
    generation_notes: list[str]


# ---------------------------------------------------------------------------
# Sentence blanking
# ---------------------------------------------------------------------------


def blank_sentence(sentence: str) -> str:
    """Replace the variant portion of a templated annotation sentence with ______.

    Sentences follow the pattern:
        [variant info] is/are (not) associated with ...
    We blank everything before the is/are.
    """
    m = re.search(r"\b(is|are)\b\s+(not\s+)?(associated with)", sentence, re.IGNORECASE)
    if m:
        return "______ " + sentence[m.start():]
    return sentence


# ---------------------------------------------------------------------------
# Jaccard similarity — hybrid tokenization
# ---------------------------------------------------------------------------


def tokenize_variant(variant: str) -> set[str]:
    """Tokenize a variant string for Jaccard similarity.

    Splits on whitespace, hyphens, colons, asterisks, slashes.
    For rsIDs, adds character 3-grams of the numeric part.
    """
    tokens: set[str] = set()
    # Split on common delimiters
    parts = re.split(r"[\s\-\:\*\/]+", variant)
    for p in parts:
        p = p.strip()
        if p:
            tokens.add(p.lower())

    # For rsIDs, add character 3-grams of numeric suffix
    rs_match = re.match(r"^rs(\d+)$", variant.strip(), re.IGNORECASE)
    if rs_match:
        digits = rs_match.group(1)
        for i in range(len(digits) - 2):
            tokens.add(f"rs3g_{digits[i:i+3]}")

    return tokens


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Variant similarity index
# ---------------------------------------------------------------------------


class VariantSimilarityIndex:
    """Precomputed token sets for all bank variants, supports fast lookup."""

    def __init__(self, bank: list[str]) -> None:
        self._bank = bank
        self._token_sets: list[tuple[str, set[str]]] = [
            (v, tokenize_variant(v)) for v in bank
        ]

    def find_most_similar(
        self,
        query: str,
        exclude: set[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return up to top_k most similar bank variants, excluding those in exclude."""
        query_tokens = tokenize_variant(query)
        scored: list[tuple[str, float]] = []
        for v, v_tokens in self._token_sets:
            if v in exclude:
                continue
            score = jaccard_similarity(query_tokens, v_tokens)
            scored.append((v, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Same-paper distractor selection
# ---------------------------------------------------------------------------


def _association_differs_on(
    target: AssociationRow, candidate: AssociationRow
) -> str | None:
    """Return the first field on which the candidate's association differs, or None."""
    if candidate.drug.lower() != target.drug.lower():
        return "drug"
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
    """Find distractors from the same paper with different variant AND different association."""
    paper_rows = index.get_rows_by_pmid(target.pmid)
    seen_variants: set[str] = set()
    distractors: list[DistractorOption] = []

    for row in paper_rows:
        # Must be a different variant
        if row.variant_haplotypes == target.variant_haplotypes:
            continue
        # Must differ on at least one association field
        differs_on = _association_differs_on(target, row)
        if differs_on is None:
            continue
        # Deduplicate by variant string
        if row.variant_haplotypes in seen_variants:
            continue
        seen_variants.add(row.variant_haplotypes)
        distractors.append(
            DistractorOption(
                variant=row.variant_haplotypes,
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
    sim_index: VariantSimilarityIndex,
    rng: random.Random,
) -> VariantMCQOptions:
    """Generate exactly 4 MCQ options for a single association row."""
    notes: list[str] = []

    # 1. Correct answer
    correct = DistractorOption(
        variant=row.variant_haplotypes,
        role="correct",
        source_annotation_id=row.variant_annotation_id,
    )

    # 2–3. Same-paper distractors
    same_paper = _get_same_paper_distractors(row, index)
    rng.shuffle(same_paper)

    # We want 2 same-paper distractors ideally
    chosen_same_paper = same_paper[:2]
    remaining_slots = 3 - len(chosen_same_paper)  # need 3 distractors total

    if len(chosen_same_paper) < 2:
        notes.append(
            f"only {len(chosen_same_paper)} same-paper distractor(s) found; "
            f"filling {remaining_slots} slot(s) from variant bank"
        )

    # Variants to exclude from bank: correct + all paper variants
    paper_variants = index.get_paper_variants(row.pmid)
    exclude = paper_variants | {row.variant_haplotypes}

    # 4 + fallback: variant bank distractors
    bank_needed = remaining_slots
    bank_candidates = sim_index.find_most_similar(
        row.variant_haplotypes,
        exclude=exclude,
        top_k=bank_needed + 5,  # grab extras in case of ties
    )

    bank_distractors: list[DistractorOption] = []
    for variant, score in bank_candidates[:bank_needed]:
        bank_distractors.append(
            DistractorOption(
                variant=variant,
                role="variant_bank",
                jaccard_score=round(score, 4),
            )
        )

    if len(bank_distractors) < bank_needed:
        notes.append(
            f"only {len(bank_distractors)} variant bank distractor(s) available "
            f"(needed {bank_needed})"
        )

    # Assemble all 4 options
    options = [correct] + chosen_same_paper + bank_distractors
    # Pad if we somehow still have < 4 (should not happen with 7k bank)
    if len(options) < 4:
        notes.append(f"could only produce {len(options)} options (expected 4)")
    options = options[:4]

    # Shuffle so correct answer isn't always first
    rng.shuffle(options)

    return VariantMCQOptions(
        annotation_id=row.variant_annotation_id,
        pmid=row.pmid,
        pmcid=row.pmcid,
        variant=row.variant_haplotypes,
        original_sentence=row.sentence,
        blanked_sentence=blank_sentence(row.sentence),
        table_type=row.table_type,
        options=options,
        generation_notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def generate_all_variant_mcqs(
    seed: int = 42,
    output_path: str = "data/mcq_options/variant_mcq_options.jsonl",
) -> Path:
    """Iterate all association rows, generate MCQs, write JSONL."""
    rng = random.Random(seed)
    index = AssociationTableIndex()
    bank = index.get_variant_bank()
    sim_index = VariantSimilarityIndex(bank)
    logger.info(f"Variant similarity index built with {len(bank)} entries")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    notes_counter: dict[str, int] = {}

    with open(out, "w") as f:
        for row in tqdm(index.all_rows, desc="Generating MCQs", unit="row"):
            mcq = generate_mcq_for_row(row, index, sim_index, rng)
            f.write(mcq.model_dump_json() + "\n")
            total += 1
            for note in mcq.generation_notes:
                # Bucket notes by prefix for stats
                key = note.split(";")[0].split("(")[0].strip()
                notes_counter[key] = notes_counter.get(key, 0) + 1

    logger.info(f"Generated {total} MCQ entries → {out}")
    if notes_counter:
        logger.info("Generation notes distribution:")
        for key, count in sorted(notes_counter.items(), key=lambda x: -x[1]):
            logger.info(f"  {key}: {count}")

    write_simplified_mcqs(out, answer_key="variant", start_id=1)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate variant MCQ distractor options"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        default="data/mcq_options/variant_mcq_options.jsonl",
        help="Output JSONL path (default: data/mcq_options/variant_mcq_options.jsonl)",
    )
    args = parser.parse_args()
    generate_all_variant_mcqs(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
