"""
Generate fill-in-the-blank (FITB) questions from benchmark annotation JSONs.

Each annotation sentence has structured fields (Drug, Alleles, Direction, etc.)
whose values appear verbatim in the sentence. For each field we create one
single-blank question by replacing the value with "_____".
"""

import json
import re
import uuid
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BlankTarget(BaseModel):
    blanked_field: str  # human-readable field name, e.g. "Drug(s)"
    ground_truth: list[str]  # acceptable answers
    field_category: str  # grouping label for reporting


class FITBQuestion(BaseModel):
    question_id: str
    source_file: str
    annotation_id: str
    pmid: str
    original_sentence: str
    blanked_sentence: str
    blanks: list[BlankTarget]
    annotation_type: str  # "var_drug_ann" | "var_pheno_ann" | "var_fa_ann"


# ---------------------------------------------------------------------------
# Field configs per annotation type
# Maps (display name) -> (JSON key for that field)
# ---------------------------------------------------------------------------

DRUG_ANN_FIELDS: dict[str, str] = {
    "Drug(s)": "Drug(s)",
    "Alleles": "Alleles",
    "Direction": "Direction of effect",
    "PD/PK terms": "PD/PK terms",
    "Comparison": "Comparison Allele(s) or Genotype(s)",
}

FA_ANN_FIELDS: dict[str, str] = {
    "Drug(s)": "Drug(s)",
    "Alleles": "Alleles",
    "Direction": "Direction of effect",
    "Functional terms": "Functional terms",
    "Comparison": "Comparison Allele(s) or Genotype(s)",
}

PHENO_ANN_FIELDS: dict[str, str] = {
    "Drug(s)": "Drug(s)",
    "Alleles": "Alleles",
    "Direction": "Direction of effect",
    "Phenotype": "Phenotype",
    "Comparison": "Comparison Allele(s) or Genotype(s)",
}

ANN_TYPE_FIELDS: dict[str, dict[str, str]] = {
    "var_drug_ann": DRUG_ANN_FIELDS,
    "var_fa_ann": FA_ANN_FIELDS,
    "var_pheno_ann": PHENO_ANN_FIELDS,
}

# Phenotype values are prefixed with "Side Effect:", "Disease:", "Other:" etc.
_PHENO_PREFIX_RE = re.compile(
    r"^(Side Effect|Disease|Other)\s*:\s*", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Core blanking helpers
# ---------------------------------------------------------------------------


def _strip_phenotype_prefix(value: str) -> str:
    """Remove 'Side Effect:', 'Disease:', 'Other:' prefixes from phenotype values."""
    return _PHENO_PREFIX_RE.sub("", value).strip()


def _parse_ground_truths(field_value: str) -> list[str]:
    """Split comma-separated field values into individual ground truths."""
    parts = [v.strip() for v in field_value.split(",") if v.strip()]
    return parts


def blank_field(sentence: str, field_value: str) -> str | None:
    """Replace *field_value* in *sentence* with '_____' (case-insensitive).

    Returns the blanked sentence, or None if the value is not found.
    """
    if not field_value or not field_value.strip():
        return None

    value = field_value.strip()

    # Case-insensitive search; replace first occurrence only
    idx = sentence.lower().find(value.lower())
    if idx == -1:
        return None

    return sentence[:idx] + "_____" + sentence[idx + len(value) :]


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


def generate_questions_from_annotation(
    annotation: dict,
    ann_type: str,
    source_file: str,
) -> list[FITBQuestion]:
    """Generate single-blank FITB questions from one annotation dict."""
    field_config = ANN_TYPE_FIELDS.get(ann_type)
    if field_config is None:
        logger.warning(f"Unknown annotation type: {ann_type}")
        return []

    sentence = annotation.get("Sentence", "")
    if not sentence or not sentence.strip():
        return []

    ann_id = str(
        annotation.get("Variant Annotation ID_norm")
        or annotation.get("Variant Annotation ID", "")
    )
    pmid = str(annotation.get("PMID_norm") or annotation.get("PMID", ""))

    questions: list[FITBQuestion] = []

    for display_name, json_key in field_config.items():
        raw_value = annotation.get(json_key)
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            continue

        raw_value = str(raw_value).strip()

        # For phenotype fields, strip category prefix before searching
        search_value = raw_value
        if json_key == "Phenotype":
            search_value = _strip_phenotype_prefix(raw_value)
            if not search_value:
                continue

        blanked = blank_field(sentence, search_value)
        if blanked is None:
            continue

        # Build ground truth list
        ground_truths = _parse_ground_truths(search_value)

        questions.append(
            FITBQuestion(
                question_id=str(uuid.uuid4()),
                source_file=source_file,
                annotation_id=ann_id,
                pmid=pmid,
                original_sentence=sentence,
                blanked_sentence=blanked,
                blanks=[
                    BlankTarget(
                        blanked_field=display_name,
                        ground_truth=ground_truths,
                        field_category=display_name,
                    )
                ],
                annotation_type=ann_type,
            )
        )

    return questions


def generate_questions_from_benchmark_json(
    json_path: Path,
) -> list[FITBQuestion]:
    """Load a single benchmark JSON and generate all FITB questions from it."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {json_path}: {e}")
        return []

    source_file = json_path.name
    questions: list[FITBQuestion] = []

    for ann_type in ("var_drug_ann", "var_pheno_ann", "var_fa_ann"):
        for annotation in data.get(ann_type, []):
            questions.extend(
                generate_questions_from_annotation(annotation, ann_type, source_file)
            )

    return questions


def generate_all_questions(
    benchmark_dir: str = "data/benchmark_annotations",
) -> list[FITBQuestion]:
    """Generate FITB questions from all benchmark JSONs in *benchmark_dir*."""
    directory = Path(benchmark_dir)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory {benchmark_dir} does not exist or is not a directory.")

    all_questions: list[FITBQuestion] = []
    for json_file in sorted(directory.glob("*.json")):
        file_questions = generate_questions_from_benchmark_json(json_file)
        all_questions.extend(file_questions)

    return all_questions


# ---------------------------------------------------------------------------
# CLI entry-point — generate questions and print stats
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from collections import Counter

    questions = generate_all_questions()
    print(f"Total FITB questions generated: {len(questions)}")

    # Stats by annotation type
    by_type: Counter[str] = Counter()
    by_field: Counter[str] = Counter()
    for q in questions:
        by_type[q.annotation_type] += 1
        for b in q.blanks:
            by_field[b.blanked_field] += 1

    print("\nBy annotation type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")

    print("\nBy blanked field:")
    for field, count in sorted(by_field.items(), key=lambda x: -x[1]):
        print(f"  {field}: {count}")

    # Save to JSONL
    output_path = Path("data") / "fitb_benchmark_mvp.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump()) + "\n")
    print(f"\nSaved questions to {output_path}")
