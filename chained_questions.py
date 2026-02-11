"""
Generates chained, multi-turn question chains from pharmacogenomics variant annotation data.

For each annotation + study_parameters row, generates a chain of 1-3 dependent questions:
1. Claim Verification (yes/no): Is a pharmacogenomic association claim true?
2. Statistical Extraction: What is a specific reported statistic (p-value, sample size, etc.)?
3. Evidence Evaluation (yes/no): Is the extracted statistic adequate/significant?

~50% of chains use flipped claims (false answers) for negative examples.
Chain length depends on study_parameters data availability.

Output: data/chained_questions.jsonl
"""

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    turn: int
    reasoning_type: str
    question: str
    answer: bool | str | int | float
    answer_source_field: str
    flip_type: str | None = None


class QuestionChain(BaseModel):
    chain_id: str
    pmid: str
    variant_annotation_id: str
    study_parameters_id: str | None
    source_file: str
    num_turns: int
    has_negative_claim: bool
    turns: list[Turn]


# ---------------------------------------------------------------------------
# Annotation Type Config
# ---------------------------------------------------------------------------


@dataclass
class AnnotationTypeConfig:
    source_file: str
    tsv_filename: str


ANNOTATION_TYPES = {
    "var_drug_ann": AnnotationTypeConfig(
        source_file="var_drug_ann",
        tsv_filename="var_drug_ann.tsv",
    ),
    "var_pheno_ann": AnnotationTypeConfig(
        source_file="var_pheno_ann",
        tsv_filename="var_pheno_ann.tsv",
    ),
    "var_fa_ann": AnnotationTypeConfig(
        source_file="var_fa_ann",
        tsv_filename="var_fa_ann.tsv",
    ),
}


# ---------------------------------------------------------------------------
# Question Templates
# ---------------------------------------------------------------------------

CLAIM_TEMPLATES = [
    "Based on PMID {pmid}, is the following claim supported: {sentence}",
    "According to the study in PMID {pmid}, is it accurate that {sentence_lower}",
    "Does PMID {pmid} support the following statement: {sentence}",
    "Is the following pharmacogenomic claim reported in PMID {pmid} true: {sentence}",
    "Per the findings in PMID {pmid}, is this claim correct: {sentence}",
]

STAT_QUESTION_PAIRS: list[dict] = [
    {
        "field": "P Value",
        "check_fields": ["P Value"],
        "questions": [
            "What p-value is reported for this association in PMID {pmid}?",
            "What is the reported statistical significance (p-value) for this finding?",
            "According to PMID {pmid}, what is the p-value for this association?",
        ],
        "eval_questions": [
            "Is this finding statistically significant?",
            "Does the p-value indicate statistical significance?",
            "Based on the reported p-value, is this result significant at the 0.05 level?",
        ],
    },
    {
        "field": "Study Cases",
        "check_fields": ["Study Cases"],
        "questions": [
            "How many cases were included in this study from PMID {pmid}?",
            "What is the case sample size for this finding?",
            "How many subjects were in the case group for this study?",
        ],
        "eval_questions": [
            "Is this sample size adequate for this type of study?",
            "Does the study have enough cases for reliable conclusions?",
            "Is the number of cases sufficient for this analysis?",
        ],
    },
    {
        "field": "Study Controls",
        "check_fields": ["Study Controls"],
        "questions": [
            "How many controls were included in the study from PMID {pmid}?",
            "What is the control group sample size for this study?",
        ],
        "eval_questions": [
            "Is the control group sample size adequate?",
            "Does the study have enough controls for reliable results?",
        ],
    },
    {
        "field": "Ratio Stat",
        "check_fields": ["Ratio Stat", "Ratio Stat Type"],
        "questions": [
            "What is the reported {ratio_stat_type} for this association in PMID {pmid}?",
            "What effect size ({ratio_stat_type}) is reported for this finding?",
        ],
        "eval_questions": [
            "Does the effect size suggest a strong association?",
            "Is the reported effect size indicative of a meaningful association?",
        ],
    },
    {
        "field": "Confidence Interval",
        "check_fields": ["Confidence Interval Start", "Confidence Interval Stop"],
        "questions": [
            "What confidence interval is reported for this finding in PMID {pmid}?",
            "What is the reported confidence interval for this association?",
        ],
        "eval_questions": [
            "Does the confidence interval indicate a precise estimate?",
            "Is the confidence interval narrow enough to suggest a reliable finding?",
        ],
    },
    {
        "field": "Study Type",
        "check_fields": ["Study Type"],
        "questions": [
            "What type of study design was used in PMID {pmid}?",
            "What study design does PMID {pmid} employ for this finding?",
        ],
        "eval_questions": [
            "Is this a controlled study design?",
            "Does the study design include a control group?",
        ],
    },
    {
        "field": "Biogeographical Groups",
        "check_fields": ["Biogeographical Groups"],
        "questions": [
            "What population was studied in PMID {pmid}?",
            "In which biogeographical group was this association studied?",
        ],
        "eval_questions": [
            "Was this studied in a specific biogeographical group?",
            "Is the study population from a defined biogeographical group?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_tsv(tsv_path: str | Path) -> list[dict]:
    """Load a TSV file and return a list of row dicts."""
    tsv_path = Path(tsv_path)
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    logger.info(f"Loaded {len(rows)} rows from {tsv_path}")
    return rows


def load_study_parameters_index(tsv_path: str | Path) -> dict[str, list[dict]]:
    """Load study_parameters.tsv and index by Variant Annotation ID."""
    rows = load_tsv(tsv_path)
    index: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        ann_id = row["Variant Annotation ID"].strip()
        if ann_id:
            index[ann_id].append(row)
    logger.info(f"Indexed {len(rows)} study parameters across {len(index)} annotations")
    return index


# ---------------------------------------------------------------------------
# Flip Logic (reused from yes_no_questions.py)
# ---------------------------------------------------------------------------


def flip_association(sentence: str, is_associated: str) -> str | None:
    """Flip 'associated with' to 'not associated with' or vice versa."""
    is_associated_lower = is_associated.strip().lower()

    if is_associated_lower == "associated with":
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
    """Flip 'increased' to 'decreased' or vice versa."""
    direction_lower = direction.strip().lower()

    if direction_lower == "increased":
        if "increased" in sentence:
            return sentence.replace("increased", "decreased", 1)
    elif direction_lower == "decreased":
        if "decreased" in sentence:
            return sentence.replace("decreased", "increased", 1)

    return None


# ---------------------------------------------------------------------------
# P-Value Parsing
# ---------------------------------------------------------------------------


def parse_p_value(raw: str) -> tuple[str, float | None]:
    """Parse p-value string like '= 0.231' or '< 0.001'.

    Returns (comparator, numeric_value).
    """
    raw = raw.strip()
    if not raw:
        return ("", None)

    for prefix in ["= ", "< ", "> "]:
        if raw.startswith(prefix):
            comparator = prefix.strip()
            try:
                value = float(raw[len(prefix) :])
                return (comparator, value)
            except ValueError:
                return (comparator, None)

    return ("", None)


def is_p_significant(raw_p: str) -> bool | None:
    """Determine if a p-value string indicates statistical significance (< 0.05)."""
    comparator, value = parse_p_value(raw_p)
    if value is None:
        return None
    if comparator == "<":
        return value <= 0.05
    if comparator == "=":
        return value < 0.05
    if comparator == ">":
        return False
    return None


# ---------------------------------------------------------------------------
# Turn Builders
# ---------------------------------------------------------------------------


def build_turn1_claim(
    ann: dict, want_negative: bool, rng: random.Random
) -> tuple[Turn, bool]:
    """Build Turn 1: claim verification.

    Returns (turn, has_negative_claim).
    """
    sentence = ann["Sentence"].strip()
    pmid = ann["PMID"].strip()
    is_associated = ann.get("Is/Is Not associated", "").strip()
    direction = ann.get("Direction of effect", "").strip()

    flipped_sentence = None
    flip_type = "original"

    if want_negative:
        if is_associated:
            flipped_sentence = flip_association(sentence, is_associated)
            if flipped_sentence:
                flip_type = "association_flip"

        if flipped_sentence is None and direction:
            if is_associated.strip().lower() == "associated with":
                flipped_sentence = flip_direction(sentence, direction)
                if flipped_sentence:
                    flip_type = "direction_flip"

    if want_negative and flipped_sentence is not None:
        claim_sentence = flipped_sentence
        answer = False
        has_negative = True
    else:
        claim_sentence = sentence
        answer = True
        has_negative = False
        flip_type = "original"

    template = rng.choice(CLAIM_TEMPLATES)
    question = template.format(
        pmid=pmid,
        sentence=claim_sentence,
        sentence_lower=claim_sentence[0].lower() + claim_sentence[1:]
        if claim_sentence
        else "",
    )

    turn = Turn(
        turn=1,
        reasoning_type="claim_verification",
        question=question,
        answer=answer,
        answer_source_field="Sentence",
        flip_type=flip_type,
    )
    return turn, has_negative


def _sp_field_available(sp_row: dict, check_fields: list[str]) -> bool:
    """Check if all required fields are non-empty in a study_parameters row."""
    return all(sp_row.get(f, "").strip() for f in check_fields)


def _get_stat_answer(sp_row: dict, pair: dict) -> bool | str | int | float:
    """Extract the answer value for a stat question pair."""
    field = pair["field"]

    if field == "P Value":
        return sp_row["P Value"].strip()
    elif field == "Study Cases":
        return int(sp_row["Study Cases"].strip())
    elif field == "Study Controls":
        return int(sp_row["Study Controls"].strip())
    elif field == "Ratio Stat":
        return float(sp_row["Ratio Stat"].strip())
    elif field == "Confidence Interval":
        start = sp_row["Confidence Interval Start"].strip()
        stop = sp_row["Confidence Interval Stop"].strip()
        return f"{start}-{stop}"
    elif field == "Study Type":
        return sp_row["Study Type"].strip()
    elif field == "Biogeographical Groups":
        return sp_row["Biogeographical Groups"].strip()
    return ""


def _derive_eval_answer(sp_row: dict, pair: dict) -> bool | None:
    """Derive the yes/no evaluation answer for Turn 3."""
    field = pair["field"]

    if field == "P Value":
        return is_p_significant(sp_row["P Value"].strip())

    elif field == "Study Cases":
        try:
            return int(sp_row["Study Cases"].strip()) >= 30
        except (ValueError, TypeError):
            return None

    elif field == "Study Controls":
        try:
            return int(sp_row["Study Controls"].strip()) >= 30
        except (ValueError, TypeError):
            return None

    elif field == "Ratio Stat":
        try:
            val = float(sp_row["Ratio Stat"].strip())
            stat_type = sp_row.get("Ratio Stat Type", "").strip()
            if stat_type == "OR":
                return val > 2.0 or val < 0.5
            return val > 2.0
        except (ValueError, TypeError):
            return None

    elif field == "Confidence Interval":
        try:
            ci_start = float(sp_row["Confidence Interval Start"].strip())
            ci_stop = float(sp_row["Confidence Interval Stop"].strip())
            # Precise if CI doesn't cross 1 (for ratio-type stats)
            return not (ci_start <= 1.0 <= ci_stop)
        except (ValueError, TypeError):
            return None

    elif field == "Study Type":
        stype = sp_row["Study Type"].strip().lower()
        return "control" in stype or "rct" in stype

    elif field == "Biogeographical Groups":
        bg = sp_row["Biogeographical Groups"].strip().lower()
        return bg not in ("unknown", "multiple groups", "")

    return None


def build_turns_2_3(ann: dict, sp_row: dict, rng: random.Random) -> list[Turn]:
    """Build Turns 2 and 3 from a study_parameters row.

    Returns 0, 1, or 2 turns depending on data availability.
    """
    pmid = ann["PMID"].strip()
    variant = ann.get("Variant/Haplotypes", "").strip()
    drug = ann.get("Drug(s)", "").strip()

    # Find which stat pairs have data
    eligible_pairs = [
        p for p in STAT_QUESTION_PAIRS if _sp_field_available(sp_row, p["check_fields"])
    ]

    if not eligible_pairs:
        return []

    # Pick one randomly
    pair = rng.choice(eligible_pairs)

    # Build Turn 2: statistical extraction
    ratio_stat_type = (
        sp_row.get("Ratio Stat Type", "effect size").strip() or "effect size"
    )
    q_template = rng.choice(pair["questions"])
    question_2 = q_template.format(
        pmid=pmid,
        variant=variant,
        drug=drug,
        ratio_stat_type=ratio_stat_type,
    )

    try:
        stat_answer = _get_stat_answer(sp_row, pair)
    except (ValueError, TypeError):
        return []

    answer_source = pair["field"]
    if pair["field"] == "Confidence Interval":
        answer_source = "Confidence Interval Start, Confidence Interval Stop"

    turn2 = Turn(
        turn=2,
        reasoning_type="statistical_extraction",
        question=question_2,
        answer=stat_answer,
        answer_source_field=answer_source,
    )

    turns = [turn2]

    # Build Turn 3: evidence evaluation (if derivable)
    eval_answer = _derive_eval_answer(sp_row, pair)
    if eval_answer is not None:
        eq_template = rng.choice(pair["eval_questions"])
        turn3 = Turn(
            turn=3,
            reasoning_type="evidence_evaluation",
            question=eq_template,
            answer=eval_answer,
            answer_source_field=answer_source,
        )
        turns.append(turn3)

    return turns


# ---------------------------------------------------------------------------
# Chain Assembly
# ---------------------------------------------------------------------------


def build_chain(
    ann: dict,
    sp_row: dict | None,
    config: AnnotationTypeConfig,
    chain_index: int,
    rng: random.Random,
) -> QuestionChain:
    """Build a complete question chain for one annotation + study_parameters pair."""
    annotation_id = ann.get("Variant Annotation ID", "").strip()
    pmid = ann.get("PMID", "").strip()
    sp_id = sp_row["Study Parameters ID"].strip() if sp_row else None

    # Turn 1: claim verification
    want_negative = chain_index % 2 == 1
    turn1, has_negative = build_turn1_claim(ann, want_negative, rng)
    turns = [turn1]

    # Turns 2-3: stat extraction + evaluation (if study params exist)
    if sp_row is not None:
        extra_turns = build_turns_2_3(ann, sp_row, rng)
        for i, t in enumerate(extra_turns):
            turns.append(t.model_copy(update={"turn": i + 2}))

    return QuestionChain(
        chain_id=f"chain_{chain_index:06d}",
        pmid=pmid,
        variant_annotation_id=annotation_id,
        study_parameters_id=sp_id,
        source_file=config.source_file,
        num_turns=len(turns),
        has_negative_claim=has_negative,
        turns=turns,
    )


def generate_all_chains(
    annotations_by_type: dict[str, list[dict]],
    sp_index: dict[str, list[dict]],
    seed: int = 42,
) -> list[QuestionChain]:
    """Generate question chains from all annotation types."""
    rng = random.Random(seed)
    chains: list[QuestionChain] = []
    chain_counter = 0
    skipped = 0

    for type_name, config in ANNOTATION_TYPES.items():
        annotations = annotations_by_type.get(type_name, [])
        for ann in annotations:
            sentence = ann.get("Sentence", "").strip()
            pmid = ann.get("PMID", "").strip()

            if not sentence or not pmid:
                skipped += 1
                continue

            ann_id = ann.get("Variant Annotation ID", "").strip()
            sp_rows = sp_index.get(ann_id, [])

            if not sp_rows:
                chain = build_chain(ann, None, config, chain_counter, rng)
                chains.append(chain)
                chain_counter += 1
            else:
                for sp_row in sp_rows:
                    chain = build_chain(ann, sp_row, config, chain_counter, rng)
                    chains.append(chain)
                    chain_counter += 1

    if skipped:
        logger.warning(f"Skipped {skipped} annotations with missing sentence or PMID")

    return chains


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_chains(chains: list[QuestionChain], output_path: str | Path):
    """Save question chains to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chain in chains:
            f.write(json.dumps(chain.model_dump()) + "\n")

    logger.info(f"Saved {len(chains)} chains to {output_path}")


def main():
    base_path = Path("data/raw/variantAnnotations")
    output_path = Path("data/chained_questions.jsonl")

    # Load study parameters index
    sp_index = load_study_parameters_index(base_path / "study_parameters.tsv")

    # Load all annotation types
    annotations_by_type = {}
    for type_name, config in ANNOTATION_TYPES.items():
        annotations_by_type[type_name] = load_tsv(base_path / config.tsv_filename)

    # Generate chains
    chains = generate_all_chains(annotations_by_type, sp_index)

    # Report statistics
    by_source: dict[str, int] = defaultdict(int)
    by_length: dict[int, int] = defaultdict(int)
    negative_count = 0

    for c in chains:
        by_source[c.source_file] += 1
        by_length[c.num_turns] += 1
        if c.has_negative_claim:
            negative_count += 1

    logger.info(f"Generated {len(chains)} total question chains")
    logger.info(f"  By source: {dict(by_source)}")
    logger.info(f"  By chain length: {dict(sorted(by_length.items()))}")
    logger.info(f"  Negative claims: {negative_count}")
    logger.info(f"  Positive claims: {len(chains) - negative_count}")

    save_chains(chains, output_path)


if __name__ == "__main__":
    main()
