"""
Multiple-choice question evaluation — generate LLM responses and score them.

Shared core for variant, drug, and phenotype MCQ pipelines.

Usage (via wrappers):
    python src/eval/mcq_variant.py generate --model gpt-4o-mini --limit 100
    python src/eval/mcq_drug.py score --responses-path runs/.../mcq_drug_responses.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.llm import build_paper_index, call_llm, load_paper

load_dotenv()

SYSTEM_PROMPT = (
    "You are a pharmacogenomics expert. "
    "Read the provided paper and answer the question."
)

DEFAULT_MODEL = "gpt-4o-mini"

LETTERS = ["a", "b", "c", "d"]


def _make_run_dir(model: str) -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    run_dir = Path("runs") / f"{ts}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize_question(q: dict, mcq_type: str) -> dict:
    """Normalize different MCQ JSONL formats to a common structure.

    Variant MCQ already has flat option_a-d and correct_answer.
    Drug/phenotype MCQ have an options array that needs conversion.

    Returns a dict with keys: question_id, annotation_id, pmcid, blanked_sentence,
    option_a-d, correct_answer, question_type, table_type.
    """
    # Variant format: already has option_a-d, correct_answer, question_type, question_id
    if "option_a" in q:
        return q

    # Drug/phenotype format: has options array
    entity_key = mcq_type  # "drug" or "phenotype"
    options = q["options"]

    option_labels = {}
    correct_letter = None
    for idx, opt in enumerate(options):
        letter = LETTERS[idx]
        option_labels[f"option_{letter}"] = opt[entity_key]
        if opt["role"] == "correct":
            correct_letter = letter

    return {
        "question_id": q.get("question_id", q.get("annotation_id")),
        "annotation_id": q["annotation_id"],
        "pmcid": q["pmcid"],
        "blanked_sentence": q["blanked_sentence"],
        "table_type": q.get("table_type", ""),
        "question_type": q.get("question_type", "standard"),
        **option_labels,
        "correct_answer": correct_letter,
    }


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(
    args: argparse.Namespace,
    output_dir: Path | None = None,
    pipeline_name: str = "mcq",
    mcq_type: str = "variant",
) -> Path:
    """Run LLM on MCQ questions, save raw responses to JSONL."""
    questions_path = Path(args.questions_path)
    raw_questions: list[dict] = []
    with open(questions_path) as f:
        for line in f:
            raw_questions.append(json.loads(line))
    logger.info(f"Loaded {len(raw_questions)} questions from {questions_path}")

    # Normalize to common format
    questions = [_normalize_question(q, mcq_type) for q in raw_questions]

    paper_index = build_paper_index()

    if args.limit > 0:
        questions = questions[: args.limit]
        logger.info(f"Limiting to {len(questions)} questions")

    logger.info(f"Generating responses with model={args.model}")

    model_slug = args.model.replace("/", "_")
    if output_dir is None:
        output_dir = _make_run_dir(args.model)
        file_prefix = pipeline_name
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = f"{model_slug}_{pipeline_name}"
    responses_path = output_dir / f"{file_prefix}_responses.jsonl"

    with open(responses_path, "w") as out_f:
        for i, q in enumerate(questions):
            pmcid = q["pmcid"]
            paper_text = load_paper(pmcid, paper_index)

            # Build the options text
            options_text = (
                f"a) {q['option_a']}\n"
                f"b) {q['option_b']}\n"
                f"c) {q['option_c']}\n"
                f"d) {q['option_d']}"
            )

            question_text = (
                f"Fill in the blank in the following sentence from the paper:\n\n"
                f"\"{q['blanked_sentence']}\"\n\n"
                f"Options:\n{options_text}\n\n"
                "Respond with ONLY a single letter (a, b, c, or d). "
                "Do not include any explanation or reasoning."
            )

            if paper_text:
                user_content = (
                    f"## Paper (PMCID {pmcid})\n\n{paper_text}\n\n"
                    f"## Question\n\n{question_text}"
                )
            else:
                user_content = question_text

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            try:
                response = call_llm(messages, args.model)
            except Exception as e:
                logger.error(f"LLM error on question {i}: {e}")
                response = ""

            record = {
                "question_id": q["question_id"],
                "annotation_id": q["annotation_id"],
                "pmcid": pmcid,
                "blanked_sentence": q["blanked_sentence"],
                "option_a": q["option_a"],
                "option_b": q["option_b"],
                "option_c": q["option_c"],
                "option_d": q["option_d"],
                "correct_answer": q["correct_answer"],
                "question_type": q["question_type"],
                "table_type": q["table_type"],
                "response": response,
                "had_paper_context": paper_text is not None,
                "model": args.model,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (i + 1) % 25 == 0 or (i + 1) == len(questions):
                logger.info(f"  [{i + 1}/{len(questions)}]")

    logger.info(f"Responses saved to {responses_path}")
    return responses_path


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def _parse_letter(response: str) -> str | None:
    """Extract a single letter (a/b/c/d) from an LLM response."""
    text = response.strip().lower()
    # Check for just a single letter
    if text in LETTERS:
        return text
    # Check for letter followed by ) or .
    for letter in LETTERS:
        if text.startswith(f"{letter})") or text.startswith(f"{letter}."):
            return letter
    # Check if any letter appears as a word
    for letter in LETTERS:
        if f" {letter} " in f" {text} ":
            return letter
    return None


def score(
    args: argparse.Namespace,
    output_dir: Path | None = None,
    pipeline_name: str = "mcq",
) -> None:
    """Score a responses JSONL file and print summary."""
    responses_path = Path(args.responses_path)
    records: list[dict] = []
    with open(responses_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} responses from {responses_path}")

    results: list[dict] = []
    correct = 0
    total = 0

    for r in records:
        predicted = _parse_letter(r["response"])
        expected = r["correct_answer"].strip().lower()
        is_correct = predicted is not None and predicted == expected

        correct += int(is_correct)
        total += 1

        results.append(
            {
                **r,
                "prediction_parsed": predicted,
                "correct": is_correct,
            }
        )

    # Build summary
    display_name = pipeline_name.replace("_", " ").title()
    model = records[0]["model"] if records else "unknown"
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"{display_name} Results  |  model={model}  |  "
        f"{correct}/{total} ({correct / total:.1%})"
    )
    lines.append("=" * 70)

    # By question_type
    qt_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        qt = r["question_type"]
        qt_stats[qt]["total"] += 1
        qt_stats[qt]["correct"] += int(r["correct"])

    lines.append("")
    lines.append("By question_type:")
    lines.append(f"  {'Type':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
    for qt in sorted(qt_stats):
        s = qt_stats[qt]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {qt:<25} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # By table_type
    tt_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        tt = r["table_type"]
        tt_stats[tt]["total"] += 1
        tt_stats[tt]["correct"] += int(r["correct"])

    lines.append("")
    lines.append("By table_type:")
    lines.append(f"  {'Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")
    for tt in sorted(tt_stats):
        s = tt_stats[tt]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {tt:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # Parse failure stats
    unparsed = sum(1 for r in results if r["prediction_parsed"] is None)
    if unparsed:
        lines.append("")
        lines.append(f"  Parse failures: {unparsed}/{total} ({unparsed / total:.1%})")

    # Paper context stats
    no_paper = sum(1 for r in results if not r["had_paper_context"])
    if no_paper:
        lines.append(
            f"  Questions without paper context: {no_paper}/{total} "
            f"({no_paper / total:.1%})"
        )

    # Example questions (5 correct, 5 incorrect)
    correct_examples = [r for r in results if r["correct"]]
    incorrect_examples = [r for r in results if not r["correct"]]
    random.shuffle(correct_examples)
    random.shuffle(incorrect_examples)

    for label, examples in [
        ("CORRECT", correct_examples[:5]),
        ("INCORRECT", incorrect_examples[:5]),
    ]:
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"Example {label} answers ({len(examples)} randomly chosen)")
        lines.append("-" * 70)
        for i, r in enumerate(examples, 1):
            lines.append("")
            lines.append(
                f"  [{i}] type={r['question_type']}  "
                f"table={r['table_type']}  pmcid={r['pmcid']}"
            )
            lines.append(f"  Sentence: {r['blanked_sentence'][:120]}")
            lines.append(
                f"  Options: a) {r['option_a']}  b) {r['option_b']}  "
                f"c) {r['option_c']}  d) {r['option_d']}"
            )
            lines.append(f"  Expected: {r['correct_answer']}  Got: {r['response']}")

    # Determine output dir
    if output_dir is None:
        output_dir = responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = responses_path.stem.replace("_responses", "")

    # Save eval results JSONL
    results_path = output_dir / f"{file_prefix}_eval_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    lines.append("")
    lines.append(f"Eval results saved to {results_path}")

    summary = "\n".join(lines)
    print(summary)

    # Save summary text
    summary_path = output_dir / f"{file_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")


# ---------------------------------------------------------------------------
# CLI helper for wrappers
# ---------------------------------------------------------------------------


def build_cli(
    pipeline_name: str,
    mcq_type: str,
    default_questions_path: str,
    description: str,
) -> None:
    """Build and run a standard MCQ CLI (used by thin wrappers)."""
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate LLM responses")
    gen_p.add_argument(
        "--model",
        default=os.environ.get("MCQ_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    gen_p.add_argument(
        "--questions-path",
        default=default_questions_path,
        help=f"Path to MCQ questions JSONL (default: {default_questions_path})",
    )
    gen_p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions (0 = all)",
    )
    gen_p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated timestamped dir)",
    )

    # score
    score_p = subparsers.add_parser("score", help="Score saved responses")
    score_p.add_argument(
        "--responses-path",
        required=True,
        help="Path to responses JSONL from generate step",
    )

    args = parser.parse_args()
    if args.command == "generate":
        out_dir = Path(args.output_dir) if args.output_dir else None
        generate(args, output_dir=out_dir, pipeline_name=pipeline_name, mcq_type=mcq_type)
    elif args.command == "score":
        score(args, pipeline_name=pipeline_name)
