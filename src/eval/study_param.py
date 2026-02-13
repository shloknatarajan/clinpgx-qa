"""
Study parameter extraction evaluation — generate LLM responses and score them.

Usage:
    python src/eval/study_param.py generate --model gpt-4o-mini --limit 100
    python src/eval/study_param.py score --responses-path runs/.../study_param_responses.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.llm import build_paper_index, call_llm, load_paper
from src.modules.study_param_questions.generate import _build_question_text

load_dotenv()

SYSTEM_PROMPT = (
    "You are a pharmacogenomics expert. "
    "Read the provided paper and answer the question. "
    "You must respond with ONLY a valid JSON object — no explanation, "
    "no reasoning, no markdown formatting, no extra text before or after the JSON."
)

DEFAULT_MODEL = "gpt-4o-mini"


def _make_run_dir(model: str) -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    run_dir = Path("runs") / f"{ts}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(args: argparse.Namespace, output_dir: Path | None = None) -> Path:
    """Run LLM on study param questions, save raw responses to JSONL."""
    questions_path = Path(args.questions_path)
    questions: list[dict] = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line))
    logger.info(f"Loaded {len(questions)} questions from {questions_path}")

    paper_index = build_paper_index()

    if args.limit > 0:
        questions = questions[: args.limit]
        logger.info(f"Limiting to {len(questions)} questions")

    logger.info(f"Generating responses with model={args.model}")

    model_slug = args.model.replace("/", "_")
    if output_dir is None:
        output_dir = _make_run_dir(args.model)
        file_prefix = "study_param"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = f"{model_slug}_study_param"
    responses_path = output_dir / f"{file_prefix}_responses.jsonl"

    with open(responses_path, "w") as out_f:
        for i, q in enumerate(questions):
            pmcid = q["pmcid"]
            paper_text = load_paper(pmcid, paper_index)

            question_text = _build_question_text(
                variant=q["variant"],
                gene=q["gene"],
                drug=q["drug"],
                phenotype_category=q["phenotype_category"],
                phenotype=q["phenotype"],
                sentence=q["sentence"],
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

            # Parse JSON from response
            resp_p_value, resp_significance = _parse_json_response(response)

            record = {
                "question_id": q["question_id"],
                "annotation_id": q["annotation_id"],
                "pmcid": pmcid,
                "question_type": q["question_type"],
                "table_type": q["table_type"],
                "expected_answer_p_value": q["expected_answer_p_value"],
                "expected_answer_significance": q["expected_answer_significance"],
                "response": response,
                "response_p_value": resp_p_value,
                "response_significance": resp_significance,
                "had_paper_context": paper_text is not None,
                "model": args.model,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (i + 1) % 25 == 0 or (i + 1) == len(questions):
                logger.info(f"  [{i + 1}/{len(questions)}]")

    logger.info(f"Responses saved to {responses_path}")
    return responses_path


def _parse_json_response(response: str) -> tuple[str, str]:
    """Extract p_value and significance from a JSON response string.

    Handles markdown code fences, extra text around JSON, and escaped quotes.
    Returns (p_value, significance) with "parse_error" as fallback.
    """
    text = response.strip()
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()

    # Try to find JSON object in the response
    match = re.search(r"\{[^}]+\}", text)
    if match:
        try:
            obj = json.loads(match.group())
            p_value = str(obj.get("p_value", "parse_error")).strip()
            significance = str(obj.get("significance", "parse_error")).strip()
            return p_value, significance
        except json.JSONDecodeError:
            pass
    return "parse_error", "parse_error"


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

# Reuse numeric comparison utilities from chained.py
from src.eval.chained import _extract_numbers, _parse_inequality, _satisfies_inequality


def _score_p_value(expected: str, response_p_value: str) -> bool:
    """Score a p-value response against expected.

    For "not found" expected: exact match (case-insensitive).
    For actual values: numeric comparison with inequality support
    and 5% relative tolerance.

    When both expected and response are inequalities (e.g. "< 0.001" vs
    "<0.001"), they match if the operator direction is the same and the
    thresholds are close.
    """
    import math

    expected_lower = expected.strip().lower()
    response_lower = response_p_value.strip().lower()

    # "not found" expected → exact match
    if expected_lower == "not found":
        return response_lower == "not found"

    # If response is "not found" but expected isn't
    if response_lower == "not found":
        return False

    # Check for inequality in expected (e.g. "< 0.001")
    exp_op, exp_threshold = _parse_inequality(expected)
    if exp_op and exp_threshold is not None:
        # First check: does the response also contain an inequality with a
        # matching operator and threshold?  e.g. expected "< 0.001" vs
        # response "<0.001" or "< .001"
        resp_op, resp_threshold = _parse_inequality(response_p_value)
        if resp_op and resp_threshold is not None:
            same_direction = (exp_op in ("<", "<=") and resp_op in ("<", "<=")) or (
                exp_op in (">", ">=") and resp_op in (">", ">=")
            )
            if same_direction:
                if exp_threshold == 0:
                    if resp_threshold == 0:
                        return True
                elif math.isclose(resp_threshold, exp_threshold, rel_tol=0.05):
                    return True

        # Second check: does a concrete numeric value in the response
        # satisfy the expected inequality?
        for resp_val in _extract_numbers(response_p_value):
            if _satisfies_inequality(resp_val, exp_op, exp_threshold):
                return True
        return False

    # Numeric comparison with 5% tolerance
    expected_nums = _extract_numbers(expected)
    if expected_nums:
        gt_val = expected_nums[0]
        for resp_val in _extract_numbers(response_p_value):
            if gt_val == 0:
                if resp_val == 0:
                    return True
            elif math.isclose(resp_val, gt_val, rel_tol=0.05):
                return True
        return False

    # Fallback: case-insensitive string match
    return expected_lower == response_lower


def _score_significance(expected: str, response_significance: str) -> bool:
    """Score significance: case-insensitive match."""
    return expected.strip().lower() == response_significance.strip().lower()


def score(args: argparse.Namespace, output_dir: Path | None = None) -> None:
    """Score a responses JSONL file and print summary."""
    responses_path = Path(args.responses_path)
    records: list[dict] = []
    with open(responses_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} responses from {responses_path}")

    results: list[dict] = []
    p_correct = 0
    sig_correct = 0
    both_correct = 0
    total = 0

    for r in records:
        p_ok = _score_p_value(r["expected_answer_p_value"], r["response_p_value"])
        sig_ok = _score_significance(
            r["expected_answer_significance"], r["response_significance"]
        )
        both_ok = p_ok and sig_ok

        p_correct += int(p_ok)
        sig_correct += int(sig_ok)
        both_correct += int(both_ok)
        total += 1

        results.append(
            {
                **r,
                "p_value_correct": p_ok,
                "significance_correct": sig_ok,
                "both_correct": both_ok,
            }
        )

    # Build summary text
    model = records[0]["model"] if records else "unknown"
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"Study Param Results  |  model={model}  |  "
        f"both: {both_correct}/{total} ({both_correct / total:.1%})"
    )
    lines.append("=" * 70)

    lines.append("")
    lines.append("By field:")
    lines.append(f"  {'Field':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")
    for field, cnt in [
        ("p_value", p_correct),
        ("significance", sig_correct),
        ("both", both_correct),
    ]:
        acc = cnt / total if total else 0
        lines.append(f"  {field:<20} {cnt:>8} {total:>8} {acc:>9.1%}")

    # By question_type
    qt_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"p_correct": 0, "sig_correct": 0, "both_correct": 0, "total": 0}
    )
    for r in results:
        qt = r["question_type"]
        qt_stats[qt]["total"] += 1
        qt_stats[qt]["p_correct"] += int(r["p_value_correct"])
        qt_stats[qt]["sig_correct"] += int(r["significance_correct"])
        qt_stats[qt]["both_correct"] += int(r["both_correct"])

    lines.append("")
    lines.append("By question_type:")
    lines.append(
        f"  {'Type':<25} {'P-value':>10} {'Signif.':>10} {'Both':>10} {'Total':>8}"
    )
    lines.append(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
    for qt in sorted(qt_stats):
        s = qt_stats[qt]
        p_acc = s["p_correct"] / s["total"] if s["total"] else 0
        sig_acc = s["sig_correct"] / s["total"] if s["total"] else 0
        both_acc = s["both_correct"] / s["total"] if s["total"] else 0
        lines.append(
            f"  {qt:<25} {p_acc:>9.1%} {sig_acc:>9.1%} {both_acc:>9.1%} {s['total']:>8}"
        )

    # By table_type
    tt_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"both_correct": 0, "total": 0}
    )
    for r in results:
        tt = r["table_type"]
        tt_stats[tt]["total"] += 1
        tt_stats[tt]["both_correct"] += int(r["both_correct"])

    lines.append("")
    lines.append("By table_type:")
    lines.append(f"  {'Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")
    for tt in sorted(tt_stats):
        s = tt_stats[tt]
        acc = s["both_correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {tt:<20} {s['both_correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # Parse failure stats
    parse_failures = sum(
        1
        for r in results
        if r["response_p_value"] == "parse_error"
        or r["response_significance"] == "parse_error"
    )
    if parse_failures:
        lines.append("")
        lines.append(
            f"  JSON parse failures: {parse_failures}/{total} "
            f"({parse_failures / total:.1%})"
        )

    # Paper context stats
    no_paper = sum(1 for r in results if not r["had_paper_context"])
    if no_paper:
        lines.append(
            f"  Questions without paper context: {no_paper}/{total} "
            f"({no_paper / total:.1%})"
        )

    # Example questions (5 correct, 5 incorrect, randomly chosen)
    correct_examples = [r for r in results if r["both_correct"]]
    incorrect_examples = [r for r in results if not r["both_correct"]]
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
            lines.append(f"  Expected p_value:      {r['expected_answer_p_value']}")
            lines.append(f"  Response p_value:      {r['response_p_value']}")
            lines.append(
                f"  Expected significance: {r['expected_answer_significance']}"
            )
            lines.append(f"  Response significance: {r['response_significance']}")
            lines.append(f"  Raw response: {r['response'][:200]}")

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
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study parameter extraction evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate LLM responses")
    gen_p.add_argument(
        "--model",
        default=os.environ.get("STUDY_PARAM_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    gen_p.add_argument(
        "--questions-path",
        default="data/study_param_questions/study_param_questions.jsonl",
        help="Path to study param questions JSONL",
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
        generate(args, output_dir=out_dir)
    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
