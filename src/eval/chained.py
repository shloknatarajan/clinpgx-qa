"""
Chained questions evaluation — generate LLM responses and score them as separate steps.

Usage:
    python src/eval/chained.py generate --model gpt-4o-mini --limit 5
    python src/eval/chained.py score --responses-path runs/20250210_120000_gpt-4o-mini/chained_responses.jsonl
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.llm import call_llm

load_dotenv()

SYSTEM_PROMPT = (
    "You are a pharmacogenomics expert. "
    "You will be given a research paper and asked a series of questions about it. "
    "Answer each question concisely and precisely. "
    "When asked whether a claim is 'supported' or 'contradicted', remember: "
    "'supported' means the paper's findings AGREE with the claim as stated, "
    "even if the claim describes a null result (e.g. 'X is not associated with Y'). "
    "'contradicted' means the paper's findings DISAGREE with the claim."
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
    """Run LLM on chained questions, save raw responses to JSONL."""
    chains: list[dict] = []
    with open(args.questions_path) as f:
        for line in f:
            chains.append(json.loads(line))
    logger.info(f"Loaded {len(chains)} chains from {args.questions_path}")

    if args.limit > 0:
        chains = chains[: args.limit]
        logger.info(f"Limiting to {len(chains)} chains")

    logger.info(f"Generating responses with model={args.model}")

    if output_dir is None:
        output_dir = _make_run_dir(args.model)
    responses_path = output_dir / "chained_responses.jsonl"

    with open(responses_path, "w") as out_f:
        for ci, chain in enumerate(chains):
            context = chain.get("context", "")
            turns = chain["turns"]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            if context:
                context_prefix = f"## Paper (PMCID {chain['pmcid']})\n\n{context}\n\n"
            else:
                context_prefix = ""

            turn_records: list[dict] = []

            for turn in turns:
                question = turn["question"]

                if turn["turn"] == 1:
                    user_content = f"{context_prefix}## Question\n\n{question}"
                else:
                    user_content = question

                messages.append({"role": "user", "content": user_content})

                try:
                    response = call_llm(messages, args.model)
                except Exception as e:
                    logger.error(
                        f"LLM error on chain {chain['chain_id']} "
                        f"turn {turn['turn']}: {e}"
                    )
                    response = ""

                messages.append({"role": "assistant", "content": response})

                turn_records.append(
                    {
                        "turn": turn["turn"],
                        "reasoning_type": turn["reasoning_type"],
                        "question": question,
                        "answer": turn["answer"],
                        "response": response,
                    }
                )

            record = {
                "chain_id": chain["chain_id"],
                "chain_family": chain["chain_family"],
                "pmcid": chain.get("pmcid"),
                "num_turns": len(turns),
                "model": args.model,
                "turns": turn_records,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (ci + 1) % 10 == 0 or (ci + 1) == len(chains):
                logger.info(f"  [{ci + 1}/{len(chains)}]")

    logger.info(f"Responses saved to {responses_path}")
    return responses_path


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def parse_claim_verification(response: str) -> str | None:
    """Parse supported / contradicted / not_reported."""
    text = response.strip().lower()
    # Normalize "not reported" (space) to "not_reported" (underscore)
    text = text.replace("not reported", "not_reported")
    for label in ("not_reported", "contradicted", "supported"):
        if label in text:
            return label
    return None


def parse_evidence_provenance(response: str) -> str | None:
    """Parse clinical_association / functional_assay / yes / no."""
    text = response.strip().lower()
    if "clinical_association" in text or "clinical association" in text:
        return "clinical_association"
    if "functional_assay" in text or "functional assay" in text:
        return "functional_assay"
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return None


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text, including decimals and scientific notation."""
    # Normalize unicode characters
    text = text.replace("\u2212", "-").replace("\u2013", "-")
    text = text.replace("\u00d7", "x")  # × → x
    # Convert unicode superscript digits: ⁰¹²³⁴⁵⁶⁷⁸⁹⁻ → 0123456789-
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
    text = text.translate(superscript_map)
    # Remove commas in numbers like "1,678"
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    results: list[float] = []
    # Scientific notation: 1.3x10^-5, 1.3x10-5 (after superscript conversion)
    for m in re.finditer(
        r"(-?\d+\.?\d*)\s*[xX]\s*10\s*\^?\s*\(?\s*(-?\d+)\s*\)?",
        text,
    ):
        try:
            results.append(float(m.group(1)) * 10 ** int(m.group(2)))
        except (ValueError, OverflowError):
            pass
    # E notation: 1.96E-8, 2.2e-63
    for m in re.finditer(r"-?\d+\.?\d*[eE][+-]?\d+", text):
        try:
            results.append(float(m.group()))
        except ValueError:
            pass
    # Plain numbers
    for m in re.findall(r"-?\d+\.?\d*", text):
        try:
            results.append(float(m))
        except ValueError:
            pass
    return results


def _parse_inequality(gt_str: str) -> tuple[str | None, float | None]:
    """Parse an inequality operator and threshold from a ground truth string.

    Returns (operator, threshold) or (None, None) if not an inequality.
    Examples: "< 0.024" → ("<", 0.024), "> 0.05" → (">", 0.05)
    """
    m = re.match(r"^\s*([<>]=?)\s*", gt_str)
    if not m:
        return None, None
    op = m.group(1)
    nums = _extract_numbers(gt_str[m.end() :])
    if not nums:
        return None, None
    return op, nums[0]


def _satisfies_inequality(value: float, op: str, threshold: float) -> bool:
    """Check whether *value* satisfies *op* *threshold*."""
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    return False


def parse_statistical_extraction(response: str, ground_truth) -> bool:
    """Compare numeric or string values.

    For int ground truths: look for matching integer in response.
    For string ground truths with an inequality (e.g. "< 0.024", "> 0.05"):
      check whether any number in the response satisfies the inequality.
    For string ground truths containing a number (e.g. "= 0.004", "0.54"):
      extract the number and compare numerically (5% relative tolerance).
    Fallback: case-insensitive string match.
    """
    text = response.strip()

    if isinstance(ground_truth, int):
        for n in _extract_numbers(text):
            if n == ground_truth:
                return True
        return False

    # String ground truth
    gt_str = str(ground_truth).strip()

    # Check for inequality ground truths (e.g. "< 0.024", "> 0.05")
    op, threshold = _parse_inequality(gt_str)
    if op and threshold is not None:
        for resp_val in _extract_numbers(text):
            if _satisfies_inequality(resp_val, op, threshold):
                return True
        return False

    # Numeric comparison with 5% tolerance
    gt_nums = _extract_numbers(gt_str)
    if gt_nums:
        gt_val = gt_nums[0]
        for resp_val in _extract_numbers(text):
            if gt_val == 0:
                if resp_val == 0:
                    return True
            elif math.isclose(resp_val, gt_val, rel_tol=0.05):
                return True
        return False

    # Fallback: case-insensitive string match
    return text.lower() == gt_str.lower()


def parse_boolean(response: str) -> bool | None:
    """Parse true/false from response."""
    text = response.strip().lower()
    if "true" in text and "false" not in text:
        return True
    if "false" in text and "true" not in text:
        return False
    return None


def score_turn(reasoning_type: str, response: str, ground_truth) -> bool:
    """Score a single turn based on its reasoning_type."""
    if reasoning_type == "claim_verification":
        return parse_claim_verification(response) == ground_truth

    if reasoning_type == "evidence_provenance_localization":
        return parse_evidence_provenance(response) == ground_truth

    if reasoning_type == "statistical_extraction":
        return parse_statistical_extraction(response, ground_truth)

    if reasoning_type in ("objective_evaluation", "counterfactual_evaluation"):
        return parse_boolean(response) == ground_truth

    return response.strip().lower() == str(ground_truth).lower()


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def score(args: argparse.Namespace, output_dir: Path | None = None) -> None:
    """Score a responses JSONL file and print summary."""
    responses_path = Path(args.responses_path)
    chains: list[dict] = []
    with open(responses_path) as f:
        for line in f:
            chains.append(json.loads(line))
    logger.info(f"Loaded {len(chains)} chain responses from {responses_path}")

    results: list[dict] = []
    turn_correct = 0
    turn_total = 0
    chain_correct = 0

    for chain in chains:
        chain_all_correct = True
        scored_turns: list[dict] = []

        for t in chain["turns"]:
            is_correct = score_turn(t["reasoning_type"], t["response"], t["answer"])
            turn_correct += int(is_correct)
            turn_total += 1
            if not is_correct:
                chain_all_correct = False

            scored_turns.append({**t, "correct": is_correct})

        chain_correct += int(chain_all_correct)
        results.append(
            {
                "chain_id": chain["chain_id"],
                "chain_family": chain["chain_family"],
                "pmcid": chain.get("pmcid"),
                "num_turns": chain["num_turns"],
                "model": chain["model"],
                "all_correct": chain_all_correct,
                "turns": scored_turns,
            }
        )

    # Build summary text
    model = chains[0]["model"] if chains else "unknown"
    num_chains = len(results)
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"Chained Results  |  model={model}  |  "
        f"turns: {turn_correct}/{turn_total} ({turn_correct / turn_total:.1%})  |  "
        f"chains: {chain_correct}/{num_chains} ({chain_correct / num_chains:.1%})"
    )
    lines.append("=" * 70)

    # By chain_family
    family_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "turn_correct": 0,
            "turn_total": 0,
            "chain_correct": 0,
            "chain_total": 0,
        }
    )
    for r in results:
        fam = r["chain_family"]
        family_stats[fam]["chain_total"] += 1
        family_stats[fam]["chain_correct"] += int(r["all_correct"])
        for t in r["turns"]:
            family_stats[fam]["turn_total"] += 1
            family_stats[fam]["turn_correct"] += int(t["correct"])

    lines.append("")
    lines.append("By chain_family:")
    lines.append(f"  {'Family':<35} {'Turn Acc':>10} {'Chain Acc':>10}")
    lines.append(f"  {'-' * 35} {'-' * 10} {'-' * 10}")
    for fam in sorted(family_stats):
        s = family_stats[fam]
        tacc = s["turn_correct"] / s["turn_total"] if s["turn_total"] else 0
        cacc = s["chain_correct"] / s["chain_total"] if s["chain_total"] else 0
        lines.append(f"  {fam:<35} {tacc:>9.1%} {cacc:>9.1%}")

    # By reasoning_type
    rt_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        for t in r["turns"]:
            rt_stats[t["reasoning_type"]]["total"] += 1
            rt_stats[t["reasoning_type"]]["correct"] += int(t["correct"])

    lines.append("")
    lines.append("By reasoning_type:")
    lines.append(f"  {'Type':<40} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 10}")
    for rt in sorted(rt_stats):
        s = rt_stats[rt]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {rt:<40} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # By turn number
    tn_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        for t in r["turns"]:
            tn_stats[t["turn"]]["total"] += 1
            tn_stats[t["turn"]]["correct"] += int(t["correct"])

    lines.append("")
    lines.append("By turn number:")
    lines.append(f"  {'Turn':>6} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 10}")
    for tn in sorted(tn_stats):
        s = tn_stats[tn]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {tn:>6} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # Determine output dir
    if output_dir is None:
        output_dir = responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eval results JSONL
    results_path = output_dir / "chained_eval_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    lines.append("")
    lines.append(f"Eval results saved to {results_path}")

    summary = "\n".join(lines)
    print(summary)

    # Save summary text
    summary_path = output_dir / "chained_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Chained questions evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate LLM responses")
    gen_p.add_argument(
        "--model",
        default=os.environ.get("CHAINED_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    gen_p.add_argument(
        "--questions-path",
        default="data/neurips_chained_questions.jsonl",
        help="Path to chained questions JSONL",
    )
    gen_p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of chains (0 = all)",
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
        generate(args)
    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
