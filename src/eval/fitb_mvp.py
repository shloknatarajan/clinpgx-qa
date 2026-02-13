"""
FITB MVP evaluation script.

1. Generate single-blank questions from the 32 benchmark JSONs
2. Save questions to data/fitb_benchmark_mvp.jsonl
3. Run each question through an LLM via litellm
4. Score with case-insensitive exact match
5. Print summary table and save detailed results
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.data_setup.fill_in_blank import FITBQuestion, generate_all_questions

load_dotenv()

PROMPT_TEMPLATE = (
    "Fill in the blank in the following pharmacogenomics statement.\n"
    "Respond with ONLY the missing value, nothing else.\n\n"
    '"{blanked_sentence}"'
)

DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(blanked_sentence: str, model: str) -> str:
    """Send a single FITB prompt to *model* via litellm and return the response."""
    import litellm

    prompt = PROMPT_TEMPLATE.format(blanked_sentence=blanked_sentence)
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_prediction(prediction: str, ground_truth: list[str]) -> bool:
    """Case-insensitive exact match against any ground truth value."""
    pred_norm = prediction.strip().lower()
    return any(gt.strip().lower() == pred_norm for gt in ground_truth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FITB MVP evaluation")
    parser.add_argument(
        "--model",
        default=os.environ.get("FITB_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmark_annotations",
        help="Directory containing benchmark JSONs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions to evaluate (0 = all)",
    )
    args = parser.parse_args()

    # --- Step 1-2: Generate questions ---
    logger.info("Generating FITB questions...")
    questions = generate_all_questions(args.benchmark_dir)
    logger.info(f"Generated {len(questions)} questions")

    questions_path = Path("data") / "fitb_benchmark_mvp.jsonl"
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(questions_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.model_dump()) + "\n")
    logger.info(f"Saved questions to {questions_path}")

    # Optional limit for testing
    if args.limit > 0:
        questions = questions[: args.limit]
        logger.info(f"Limiting evaluation to {len(questions)} questions")

    # --- Step 3-4: Run LLM + score ---
    logger.info(f"Running evaluation with model={args.model}")
    results: list[dict] = []
    correct = 0
    total = 0

    for i, q in enumerate(questions):
        ground_truth = q.blanks[0].ground_truth
        field = q.blanks[0].blanked_field

        try:
            prediction = call_llm(q.blanked_sentence, args.model)
        except Exception as e:
            print(f"LLM error on question {i}: {e}")
            prediction = ""

        is_correct = score_prediction(prediction, ground_truth)
        correct += int(is_correct)
        total += 1

        result = {
            "question_id": q.question_id,
            "annotation_type": q.annotation_type,
            "blanked_field": field,
            "blanked_sentence": q.blanked_sentence,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
        }
        results.append(result)

        if (i + 1) % 25 == 0 or (i + 1) == len(questions):
            logger.info(
                f"  [{i + 1}/{len(questions)}] running accuracy: {correct}/{total} ({correct / total:.1%})"
            )

    # --- Step 5-6: Summary table ---
    print("\n" + "=" * 70)
    print(
        f"FITB MVP Results  |  model={args.model}  |  {correct}/{total} ({correct / total:.1%})"
    )
    print("=" * 70)

    # Per annotation type
    type_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    field_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for r in results:
        type_stats[r["annotation_type"]]["total"] += 1
        type_stats[r["annotation_type"]]["correct"] += int(r["correct"])
        field_stats[r["blanked_field"]]["total"] += 1
        field_stats[r["blanked_field"]]["correct"] += int(r["correct"])

    print("\nBy annotation type:")
    print(f"  {'Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")
    for t in sorted(type_stats):
        s = type_stats[t]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {t:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    print("\nBy blanked field:")
    print(f"  {'Field':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")
    for field in sorted(field_stats):
        s = field_stats[field]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {field:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # --- Step 7: Save detailed results ---
    results_path = Path("data") / "fitb_mvp_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nDetailed results saved to {results_path}")


if __name__ == "__main__":
    main()
