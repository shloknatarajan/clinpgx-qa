"""
Generates true/false questions from var_drug_ann.tsv by flipping annotation categories.

For each annotation, generates:
1. A TRUE question using the original sentence
2. A FALSE question by flipping the association (associated ↔ not associated)
3. A FALSE question by flipping the effect direction (increased ↔ decreased),
   only when the annotation has a positive association (to avoid ambiguity)

Output format: JSONL with fields:
- variant_annotation_id: The unique annotation ID
- pmcid: PubMed Central ID
- question: The full question string
- answer: true or false
- flip_type: "original", "association_flip", or "direction_flip"
- original_sentence: The unmodified sentence from the annotation

Saves to data/yes_no_questions.jsonl
"""

import csv
import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from src.utils.paper_map import build_pmid_to_pmcid


class TrueFalseQuestion(BaseModel):
    variant_annotation_id: str
    pmcid: str
    question: str
    answer: bool
    flip_type: str
    original_sentence: str


def load_var_drug_ann(tsv_path: str | Path) -> list[dict]:
    """Load the var_drug_ann.tsv file and return a list of annotation dicts."""
    tsv_path = Path(tsv_path)
    annotations = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            annotations.append(row)
    logger.info(f"Loaded {len(annotations)} annotations from {tsv_path}")
    return annotations


def flip_association(sentence: str, is_associated: str) -> str | None:
    """Flip 'associated with' to 'not associated with' or vice versa in the sentence.

    Uses the structured field value to determine the current state and applies
    the correct replacement.
    """
    is_associated_lower = is_associated.strip().lower()

    if is_associated_lower == "associated with":
        # Try both "is" and "are" forms
        for verb in [" is associated with ", " are associated with "]:
            not_verb = verb.replace(" associated with ", " not associated with ")
            if verb in sentence:
                return sentence.replace(verb, not_verb, 1)
    elif is_associated_lower == "not associated with":
        for verb in [" is not associated with ", " are not associated with "]:
            assoc_verb = verb.replace(" not associated with ", " associated with ")
            if verb in sentence:
                return sentence.replace(verb, assoc_verb, 1)

    return None


def flip_direction(sentence: str, direction: str) -> str | None:
    """Flip 'increased' to 'decreased' or vice versa in the sentence."""
    direction_lower = direction.strip().lower()

    if direction_lower == "increased":
        if "increased" in sentence:
            return sentence.replace("increased", "decreased", 1)
    elif direction_lower == "decreased":
        if "decreased" in sentence:
            return sentence.replace("decreased", "increased", 1)

    return None


def make_question(pmcid: str, statement: str) -> str:
    return (
        f"Based on the PubMed Central article {pmcid}, "
        f"is this statement true or false: {statement}"
    )


def generate_questions(
    annotations: list[dict],
    pmid_to_pmcid: dict[str, str],
) -> list[TrueFalseQuestion]:
    """Generate true/false questions from annotations by flipping categories."""
    questions = []
    skipped = 0
    filtered_no_paper = 0

    for ann in annotations:
        annotation_id = ann.get("Variant Annotation ID", "").strip()
        pmid = ann.get("PMID", "").strip()
        sentence = ann.get("Sentence", "").strip()
        is_associated = ann.get("Is/Is Not associated", "").strip()
        direction = ann.get("Direction of effect", "").strip()

        if not sentence or not pmid:
            skipped += 1
            continue

        pmcid = pmid_to_pmcid.get(pmid)
        if pmcid is None:
            filtered_no_paper += 1
            continue

        # TRUE question: original sentence
        questions.append(
            TrueFalseQuestion(
                variant_annotation_id=annotation_id,
                pmcid=pmcid,
                question=make_question(pmcid, sentence),
                answer=True,
                flip_type="original",
                original_sentence=sentence,
            )
        )

        # FALSE question: flip association direction
        if is_associated:
            flipped_assoc = flip_association(sentence, is_associated)
            if flipped_assoc:
                questions.append(
                    TrueFalseQuestion(
                        variant_annotation_id=annotation_id,
                        pmcid=pmcid,
                        question=make_question(pmcid, flipped_assoc),
                        answer=False,
                        flip_type="association_flip",
                        original_sentence=sentence,
                    )
                )

        # FALSE question: flip effect direction
        # Only for positive associations to avoid ambiguity:
        # "not associated with decreased X" flipped to "not associated with increased X"
        # is not clearly false, so we skip those.
        if direction and is_associated.strip().lower() == "associated with":
            flipped_dir = flip_direction(sentence, direction)
            if flipped_dir:
                questions.append(
                    TrueFalseQuestion(
                        variant_annotation_id=annotation_id,
                        pmcid=pmcid,
                        question=make_question(pmcid, flipped_dir),
                        answer=False,
                        flip_type="direction_flip",
                        original_sentence=sentence,
                    )
                )

    if skipped:
        logger.warning(f"Skipped {skipped} annotations with missing sentence or PMID")
    if filtered_no_paper:
        logger.warning(
            f"Filtered {filtered_no_paper} annotations with no matching paper file"
        )

    return questions


def save_questions(questions: list[TrueFalseQuestion], output_path: str | Path):
    """Save questions to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump()) + "\n")

    logger.info(f"Saved {len(questions)} questions to {output_path}")


def main():
    tsv_path = Path("data/raw/variantAnnotations/var_drug_ann.tsv")
    output_path = Path("data/yes_no_questions.jsonl")
    papers_dir = Path("data/papers")

    annotations = load_var_drug_ann(tsv_path)
    pmid_to_pmcid = build_pmid_to_pmcid(papers_dir)
    questions = generate_questions(annotations, pmid_to_pmcid)

    true_count = sum(1 for q in questions if q.answer)
    false_count = sum(1 for q in questions if not q.answer)
    assoc_flip_count = sum(1 for q in questions if q.flip_type == "association_flip")
    dir_flip_count = sum(1 for q in questions if q.flip_type == "direction_flip")

    logger.info(f"Generated {len(questions)} total questions")
    logger.info(f"  TRUE: {true_count}")
    logger.info(f"  FALSE: {false_count}")
    logger.info(f"    Association flips: {assoc_flip_count}")
    logger.info(f"    Direction flips: {dir_flip_count}")

    save_questions(questions, output_path)


if __name__ == "__main__":
    main()
