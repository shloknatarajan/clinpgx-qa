"""
Phenotype MCQ distractor generation engine.

For each pheno-table association row, generates exactly 4 multiple-choice options:
  1. The correct phenotype (ground truth)
  2–3. Same-paper distractors with different associations
  4. Phenotype-bank distractor chosen by Jaccard similarity

Fallback: when fewer than 2 same-paper distractors are found, remaining
slots are filled from the phenotype bank.

Only processes pheno-table rows (drug-table rows have no phenotype field).

Usage:
    python -m src.modules.mc_questions.phenotype_modifier
    python -m src.modules.mc_questions.phenotype_modifier --seed 42 --output data/mcq_options/phenotype_mcq_options.jsonl
"""

import argparse
import random
import re
import sys
import tempfile
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.modules.mc_questions.table_lookup import (
    AssociationRow,
    AssociationTableIndex,
    write_simplified_mcqs,
)

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class DistractorOption(BaseModel):
    phenotype: str
    role: str  # "correct" | "same_paper" | "phenotype_bank"
    source_annotation_id: str | None = None
    jaccard_score: float | None = None
    association_differs_on: str | None = None  # "variant", "drug", "direction"


class PhenotypeMCQOptions(BaseModel):
    annotation_id: str
    pmid: str
    pmcid: str
    phenotype: str
    original_sentence: str
    blanked_sentence: str
    table_type: str
    options: list[DistractorOption]  # exactly 4
    generation_notes: list[str]


# ---------------------------------------------------------------------------
# Phenotype helpers
# ---------------------------------------------------------------------------


def strip_phenotype_prefix(phenotype: str) -> str:
    """Strip category prefixes like 'Side Effect:', 'Disease:', 'Other:', 'Efficacy:'.

    E.g. 'Side Effect:Neutropenia' -> 'Neutropenia'
    """
    if ":" in phenotype:
        return phenotype.split(":", 1)[1].strip()
    return phenotype.strip()


def phenotype_names(phenotype_field: str) -> list[str]:
    """Extract individual phenotype display names from the raw field.

    'Side Effect:X, Side Effect:Y' -> ['X', 'Y']
    """
    parts = [p.strip() for p in phenotype_field.split(",") if p.strip()]
    return [strip_phenotype_prefix(p) for p in parts]


# ---------------------------------------------------------------------------
# Sentence blanking
# ---------------------------------------------------------------------------


def blank_phenotype_in_sentence(sentence: str, phenotype_field: str) -> str:
    """Replace phenotype name(s) in a sentence with ______.

    Finds the span covering all individual phenotype names (case-insensitive)
    and replaces it.
    """
    names = phenotype_names(phenotype_field)
    if not names:
        return sentence

    sent_lower = sentence.lower()
    first_start = len(sentence)
    last_end = 0
    for name in names:
        idx = sent_lower.find(name.lower())
        if idx != -1:
            first_start = min(first_start, idx)
            last_end = max(last_end, idx + len(name))

    if last_end > first_start:
        return sentence[:first_start] + "______" + sentence[last_end:]

    return sentence


# ---------------------------------------------------------------------------
# Phenotype bank validation (Fix 5)
# ---------------------------------------------------------------------------

_PHENOTYPE_BLOCKLIST = frozenset(
    {
        "other",
        "unknown",
        "n/a",
        "na",
        "toxicity",
        "efficacy",
        "metabolism",
        "dosage",
        "pharmacokinetics",
    }
)


def _is_valid_phenotype(entry: str) -> bool:
    """Return False for junk phenotype bank entries."""
    raw = entry.strip()
    stripped = strip_phenotype_prefix(raw).strip()

    # PK: prefixed entries
    if raw.startswith("PK:"):
        return False

    # Too short after stripping prefix
    if len(stripped) < 4:
        return False

    # Starts with digit
    if stripped[0].isdigit():
        return False

    # Generic blocklist
    if stripped.lower() in _PHENOTYPE_BLOCKLIST:
        return False

    # Prefix-less fragments containing / or )
    if stripped != raw and ("/" in stripped or ")" in stripped):
        return False

    return True


# ---------------------------------------------------------------------------
# Jaccard similarity for phenotypes
# ---------------------------------------------------------------------------


def tokenize_phenotype(phenotype: str) -> set[str]:
    """Tokenize a phenotype string for Jaccard similarity.

    Strips category prefixes, then splits on whitespace, hyphens, commas,
    slashes, and parentheses.
    """
    tokens: set[str] = set()
    # Handle comma-separated multi-phenotype fields
    names = phenotype_names(phenotype) if ":" in phenotype else [phenotype]
    for name in names:
        parts = re.split(r"[\s\-\,\/\(\)]+", name)
        for p in parts:
            p = p.strip()
            if p:
                tokens.add(p.lower())
    return tokens


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Phenotype similarity index
# ---------------------------------------------------------------------------


class PhenotypeSimilarityIndex:
    """Precomputed token sets for all bank phenotypes, supports fast lookup."""

    def __init__(self, bank: list[str]) -> None:
        filtered = [p for p in bank if _is_valid_phenotype(p)]
        logger.info(
            f"Phenotype bank: {len(bank)} raw → {len(filtered)} after filtering"
        )
        self._bank = filtered
        self._token_sets: list[tuple[str, set[str]]] = [
            (p, tokenize_phenotype(p)) for p in filtered
        ]
        # Map stripped lowercase name → set of raw bank entries
        self._by_stripped_name: dict[str, set[str]] = {}
        for p in filtered:
            key = strip_phenotype_prefix(p).strip().lower()
            self._by_stripped_name.setdefault(key, set()).add(p)

    def get_entries_by_stripped_name(self, name: str) -> set[str]:
        """Return all bank entries whose stripped name matches (case-insensitive)."""
        key = strip_phenotype_prefix(name).strip().lower()
        return self._by_stripped_name.get(key, set())

    def find_most_similar(
        self,
        query: str,
        exclude: set[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return up to top_k most similar bank phenotypes, excluding those in exclude."""
        query_tokens = tokenize_phenotype(query)
        scored: list[tuple[str, float]] = []
        for p, p_tokens in self._token_sets:
            if p in exclude:
                continue
            score = jaccard_similarity(query_tokens, p_tokens)
            scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Same-paper distractor selection
# ---------------------------------------------------------------------------


def _association_differs_on(
    target: AssociationRow, candidate: AssociationRow
) -> str | None:
    """Return the first field on which the candidate's association differs, or None."""
    if candidate.variant_haplotypes.lower() != target.variant_haplotypes.lower():
        return "variant"
    if candidate.drug.lower() != target.drug.lower():
        return "drug"
    if candidate.direction_of_effect.lower() != target.direction_of_effect.lower():
        return "direction"
    return None


def _get_same_paper_distractors(
    target: AssociationRow,
    index: AssociationTableIndex,
) -> list[DistractorOption]:
    """Find distractors from the same paper with different phenotype AND different association."""
    paper_rows = index.get_rows_by_pmid(target.pmid)
    seen_phenotypes: set[str] = set()
    seen_stripped: set[str] = set()
    distractors: list[DistractorOption] = []

    target_stripped = strip_phenotype_prefix(target.phenotype).strip().lower()

    for row in paper_rows:
        # Only consider pheno-table rows
        if row.table_type != "pheno":
            continue
        # Must be a different phenotype
        if row.phenotype == target.phenotype:
            continue
        row_stripped = strip_phenotype_prefix(row.phenotype).strip().lower()
        # Skip candidates whose stripped name matches the target
        if row_stripped == target_stripped:
            continue
        # Must have a non-empty phenotype
        if not row.phenotype:
            continue
        # Must differ on at least one other association field
        differs_on = _association_differs_on(target, row)
        if differs_on is None:
            continue
        # Deduplicate by phenotype string and by stripped name
        if row.phenotype in seen_phenotypes or row_stripped in seen_stripped:
            continue
        seen_phenotypes.add(row.phenotype)
        seen_stripped.add(row_stripped)
        distractors.append(
            DistractorOption(
                phenotype=row.phenotype,
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
    sim_index: PhenotypeSimilarityIndex,
    rng: random.Random,
) -> PhenotypeMCQOptions:
    """Generate exactly 4 MCQ options for a single association row."""
    notes: list[str] = []

    # 1. Correct answer
    correct = DistractorOption(
        phenotype=row.phenotype,
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
            f"filling {remaining_slots} slot(s) from phenotype bank"
        )

    # Phenotypes to exclude from bank: correct + all paper phenotypes
    paper_phenotypes = index.get_paper_phenotypes(row.pmid)
    exclude = paper_phenotypes | {row.phenotype}

    # Expand exclusion to all bank entries sharing the same stripped name
    # as the correct answer or any chosen same-paper distractor
    expanded_exclude = set(exclude)
    expanded_exclude |= sim_index.get_entries_by_stripped_name(row.phenotype)
    for sp in chosen_same_paper:
        expanded_exclude |= sim_index.get_entries_by_stripped_name(sp.phenotype)

    # 4 + fallback: phenotype bank distractors
    bank_needed = remaining_slots
    bank_candidates = sim_index.find_most_similar(
        row.phenotype,
        exclude=expanded_exclude,
        top_k=bank_needed + 10,
    )

    # Deduplicate bank candidates by stripped name to avoid prefix-only variants
    seen_bank_stripped: set[str] = set()
    bank_distractors: list[DistractorOption] = []
    for phenotype, score in bank_candidates:
        if len(bank_distractors) >= bank_needed:
            break
        cand_stripped = strip_phenotype_prefix(phenotype).strip().lower()
        if cand_stripped in seen_bank_stripped:
            continue
        seen_bank_stripped.add(cand_stripped)
        bank_distractors.append(
            DistractorOption(
                phenotype=phenotype,
                role="phenotype_bank",
                jaccard_score=round(score, 4),
            )
        )

    if len(bank_distractors) < bank_needed:
        notes.append(
            f"only {len(bank_distractors)} phenotype bank distractor(s) available "
            f"(needed {bank_needed})"
        )

    # Assemble all 4 options
    options = [correct] + chosen_same_paper + bank_distractors
    if len(options) < 4:
        notes.append(f"could only produce {len(options)} options (expected 4)")
    options = options[:4]

    # Shuffle so correct answer isn't always first
    rng.shuffle(options)

    return PhenotypeMCQOptions(
        annotation_id=row.variant_annotation_id,
        pmid=row.pmid,
        pmcid=row.pmcid,
        phenotype=row.phenotype,
        original_sentence=row.sentence,
        blanked_sentence=blank_phenotype_in_sentence(row.sentence, row.phenotype),
        table_type=row.table_type,
        options=options,
        generation_notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def generate_all_phenotype_mcqs(
    seed: int = 42,
    output_path: str = "data/mcq_options/phenotype_mcq_options.jsonl",
) -> Path:
    """Iterate pheno-table association rows, generate MCQs, write JSONL."""
    rng = random.Random(seed)
    index = AssociationTableIndex()
    bank = index.get_phenotype_bank()
    sim_index = PhenotypeSimilarityIndex(bank)
    logger.info(f"Phenotype similarity index built with {len(bank)} entries")

    # Only process pheno-table rows with non-empty phenotype
    pheno_rows = [r for r in index.all_rows if r.table_type == "pheno" and r.phenotype]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    blank_skipped = 0
    notes_counter: dict[str, int] = {}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in tqdm(pheno_rows, desc="Generating phenotype MCQs", unit="row"):
            mcq = generate_mcq_for_row(row, index, sim_index, rng)
            if mcq.blanked_sentence == mcq.original_sentence:
                blank_skipped += 1
                continue
            tmp.write(mcq.model_dump_json() + "\n")
            total += 1
            for note in mcq.generation_notes:
                key = note.split(";")[0].split("(")[0].strip()
                notes_counter[key] = notes_counter.get(key, 0) + 1

    if blank_skipped:
        logger.info(f"Skipped {blank_skipped} rows where blanking failed")
    logger.info(f"Generated {total} phenotype MCQ entries")
    if notes_counter:
        logger.info("Generation notes distribution:")
        for key, count in sorted(notes_counter.items(), key=lambda x: -x[1]):
            logger.info(f"  {key}: {count}")

    write_simplified_mcqs(tmp_path, answer_key="phenotype", seed=seed, start_id=60001, output_path=out)
    tmp_path.unlink()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate phenotype MCQ distractor options"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        default="data/mcq_options/phenotype_mcq_options.jsonl",
        help="Output JSONL path (default: data/mcq_options/phenotype_mcq_options.jsonl)",
    )
    args = parser.parse_args()
    generate_all_phenotype_mcqs(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
