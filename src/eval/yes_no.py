"""
Yes/No evaluation — generate LLM responses and score them as separate steps.

Usage:
    python src/eval/yes_no.py generate --model gpt-4o-mini --limit 100
    python src/eval/yes_no.py score --responses-path runs/20250210_120000_gpt-4o-mini/yes_no_responses.jsonl
"""

import argparse
import json
import os
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
    """Run LLM on yes/no questions, save raw responses to JSONL."""
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

    if output_dir is None:
        output_dir = _make_run_dir(args.model)
    responses_path = output_dir / "yes_no_responses.jsonl"

    with open(responses_path, "w") as out_f:
        for i, q in enumerate(questions):
            pmcid = q["pmcid"]
            paper_text = load_paper(pmcid, paper_index)

            if paper_text:
                user_content = (
                    f"## Paper (PMCID {pmcid})\n\n{paper_text}\n\n"
                    f"## Question\n\n{q['question']}\n\n"
                    "Respond with ONLY the single word 'true' or 'false'. "
                    "Do not include any explanation or reasoning."
                )
            else:
                user_content = (
                    f"{q['question']}\n\n"
                    "Respond with ONLY the single word 'true' or 'false'. "
                    "Do not include any explanation or reasoning."
                )

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
                "variant_annotation_id": q["variant_annotation_id"],
                "pmcid": pmcid,
                "question": q["question"],
                "answer": q["answer"],
                "flip_type": q["flip_type"],
                "response": response,
                "had_paper_context": paper_text is not None,
                "model": args.model,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (i + 1) % 25 == 0 or (i + 1) == len(questions):
                logger.info(f"  [{i+1}/{len(questions)}]")

    logger.info(f"Responses saved to {responses_path}")
    return responses_path


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def parse_true_false(response: str) -> bool | None:
    """Extract a boolean from an LLM response. Returns None if unparseable."""
    text = response.strip().lower()
    if "true" in text and "false" not in text:
        return True
    if "false" in text and "true" not in text:
        return False
    return None


def score(args: argparse.Namespace, output_dir: Path | None = None) -> None:
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
        predicted = parse_true_false(r["response"])
        ground_truth = r["answer"]
        is_correct = predicted is not None and predicted == ground_truth

        correct += int(is_correct)
        total += 1

        results.append({
            **r,
            "prediction_parsed": predicted,
            "correct": is_correct,
        })

    # Build summary text
    model = records[0]["model"] if records else "unknown"
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"Yes/No Results  |  model={model}  |  "
        f"{correct}/{total} ({correct/total:.1%})"
    )
    lines.append("=" * 70)

    # By flip_type
    flip_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        flip_stats[r["flip_type"]]["total"] += 1
        flip_stats[r["flip_type"]]["correct"] += int(r["correct"])

    lines.append("")
    lines.append("By flip_type:")
    lines.append(f"  {'flip_type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    for ft in sorted(flip_stats):
        s = flip_stats[ft]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {ft:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # Parse failure stats
    unparsed = sum(1 for r in results if r["prediction_parsed"] is None)
    if unparsed:
        lines.append("")
        lines.append(f"  Parse failures: {unparsed}/{total} ({unparsed/total:.1%})")

    # Paper context stats
    no_paper = sum(1 for r in results if not r["had_paper_context"])
    if no_paper:
        lines.append(
            f"  Questions without paper context: {no_paper}/{total} "
            f"({no_paper/total:.1%})"
        )

    # Determine output dir
    if output_dir is None:
        output_dir = responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eval results JSONL
    results_path = output_dir / "yes_no_eval_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    lines.append("")
    lines.append(f"Eval results saved to {results_path}")

    summary = "\n".join(lines)
    print(summary)

    # Save summary text
    summary_path = output_dir / "yes_no_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Yes/No evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate LLM responses")
    gen_p.add_argument(
        "--model",
        default=os.environ.get("YESNO_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    gen_p.add_argument(
        "--questions-path",
        default="data/yes_no_questions.jsonl",
        help="Path to yes/no questions JSONL",
    )
    gen_p.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of questions (0 = all)",
    )

    # score
    score_p = subparsers.add_parser("score", help="Score saved responses")
    score_p.add_argument(
        "--responses-path", required=True,
        help="Path to responses JSONL from generate step",
    )

    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
