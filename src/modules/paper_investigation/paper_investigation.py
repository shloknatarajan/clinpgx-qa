"""
Paper Investigation — end-to-end evaluation combining variant extraction recall
with downstream MC question accuracy per paper.

Usage:
    python src/modules/paper_investigation/paper_investigation.py run --model gpt-4o-mini --limit 10
    python src/modules/paper_investigation/paper_investigation.py generate --model gpt-4o-mini --limit 10
    python src/modules/paper_investigation/paper_investigation.py score --responses-path runs/.../paper_investigation_responses.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.llm import build_paper_index, call_llm, load_paper
from src.eval.mcq import _parse_letter
from src.eval.study_param import (
    _parse_json_response,
    _score_p_value,
    _score_significance,
)
from src.modules.paper_investigation.question_index import (
    QuestionIndex,
    UnifiedQuestion,
)
from src.modules.study_param_questions.generate import _build_question_text
from src.modules.variant_extraction.variant_extraction import (
    SYSTEM_PROMPT as VE_SYSTEM_PROMPT,
)
from src.modules.variant_extraction.variant_extraction import (
    USER_PROMPT_TEMPLATE as VE_USER_PROMPT_TEMPLATE,
)
from src.modules.variant_extraction.variant_extraction import (
    normalize_variant,
    parse_variant_list,
)

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def _make_run_dir(model: str) -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    run_dir = Path("runs") / f"{ts}_{model_slug}_PI"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Batched question prompting
# ---------------------------------------------------------------------------

BATCHED_SYSTEM_PROMPT = (
    "You are a pharmacogenomics expert. "
    "Read the provided paper and answer each numbered question. "
    "For each question, respond with ONLY the answer on its own line, "
    "prefixed by the question number (e.g. '1. a' or '2. {\"p_value\": ...}')."
)


def _format_single_question(uq: UnifiedQuestion) -> str:
    """Format a single question as text (no paper context — that goes in the header)."""
    q = uq.raw_question
    if uq.source_pipeline.startswith("mcq_"):
        options_text = (
            f"a) {q['option_a']}\n"
            f"b) {q['option_b']}\n"
            f"c) {q['option_c']}\n"
            f"d) {q['option_d']}"
        )
        return (
            f"Fill in the blank in the following sentence from the paper:\n"
            f'"{q["blanked_sentence"]}"\n\n'
            f"Options:\n{options_text}\n\n"
            "Respond with ONLY a single letter (a, b, c, or d)."
        )
    # study_param
    return _build_question_text(
        variant=q["variant"],
        gene=q["gene"],
        drug=q["drug"],
        phenotype_category=q["phenotype_category"],
        phenotype=q["phenotype"],
        sentence=q["sentence"],
    )


def _build_batched_prompt(
    questions: list[UnifiedQuestion], paper_text: str | None, pmcid: str
) -> list[dict]:
    """Build a single LLM prompt with all questions for a variant."""
    numbered_questions = []
    for i, uq in enumerate(questions, 1):
        numbered_questions.append(f"### Question {i}\n{_format_single_question(uq)}")

    questions_block = "\n\n".join(numbered_questions)

    if paper_text:
        user_content = (
            f"## Paper (PMCID {pmcid})\n\n{paper_text}\n\n"
            f"## Questions\n\n{questions_block}\n\n"
            f"Answer each question on its own line, prefixed by the question number. "
            f"For multiple-choice questions, respond with only the letter. "
            f'For p-value questions, respond with only the JSON object.\n'
            f"Example:\n1. b\n"
            f'2. {{"p_value": "0.03", "significance": "yes"}}\n3. a'
        )
    else:
        user_content = (
            f"{questions_block}\n\n"
            f"Answer each question on its own line, prefixed by the question number. "
            f"For multiple-choice questions, respond with only the letter. "
            f'For p-value questions, respond with only the JSON object.\n'
            f"Example:\n1. b\n"
            f'2. {{"p_value": "0.03", "significance": "yes"}}\n3. a'
        )

    return [
        {"role": "system", "content": BATCHED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_batched_response(
    response: str, num_questions: int
) -> list[str]:
    """Parse a batched response into individual answer strings.

    Expects lines like '1. a', '2. {"p_value": ...}', etc.
    Returns a list of answer strings (one per question). Missing answers
    are returned as empty strings.
    """
    answers = [""] * num_questions

    # Match lines starting with a number followed by a dot/parenthesis
    for match in re.finditer(
        r"(?m)^(\d+)\s*[.)]\s*(.+)", response
    ):
        idx = int(match.group(1)) - 1  # 0-based
        if 0 <= idx < num_questions:
            answers[idx] = match.group(2).strip()

    return answers


# ---------------------------------------------------------------------------
# Question scoring
# ---------------------------------------------------------------------------


def _score_question(uq: UnifiedQuestion, response: str) -> tuple[bool, str]:
    """Score a single question response.

    Returns (is_correct, expected_answer_repr).
    """
    q = uq.raw_question
    pipeline = uq.source_pipeline

    if pipeline.startswith("mcq_"):
        expected = q["correct_answer"].strip().lower()
        predicted = _parse_letter(response)
        return (predicted is not None and predicted == expected), expected

    if pipeline == "study_param":
        resp_p, resp_sig = _parse_json_response(response)
        p_ok = _score_p_value(q["expected_answer_p_value"], resp_p)
        sig_ok = _score_significance(q["expected_answer_significance"], resp_sig)
        expected_repr = (
            f"p={q['expected_answer_p_value']}, sig={q['expected_answer_significance']}"
        )
        return (p_ok and sig_ok), expected_repr

    return False, "unknown_pipeline"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(args: argparse.Namespace, output_dir: Path | None = None) -> Path:
    """Run paper investigation: extract variants, answer questions, score."""
    bench_path = Path(args.bench_path)
    articles: list[dict] = []
    with open(bench_path) as f:
        for line in f:
            articles.append(json.loads(line))
    logger.info(f"Loaded {len(articles)} articles from {bench_path}")

    if args.limit > 0:
        articles = articles[: args.limit]
        logger.info(f"Limiting to {len(articles)} articles")

    paper_index = build_paper_index()
    question_index = QuestionIndex()

    model_slug = args.model.replace("/", "_")
    if output_dir is None:
        output_dir = _make_run_dir(args.model)
        file_prefix = "paper_investigation"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = f"{model_slug}_paper_investigation"
    responses_path = output_dir / f"{file_prefix}_responses.jsonl"

    logger.info(f"Generating with model={args.model}, output={responses_path}")

    with open(responses_path, "w") as out_f:
        pbar = tqdm(articles, desc=f"Papers ({args.model})", unit="paper")
        for article in pbar:
            pmcid = article["pmcid"]
            ground_truth_raw = article["variants"]
            ground_truth_norm = {normalize_variant(v) for v in ground_truth_raw}

            paper_text = load_paper(pmcid, paper_index)
            if paper_text is None:
                pbar.set_postfix(pmcid=pmcid, status="no paper")
                record = {
                    "pmcid": pmcid,
                    "model": args.model,
                    "ground_truth_variants": ground_truth_raw,
                    "predicted_variants": None,
                    "recalled_variants": [],
                    "variant_recall": None,
                    "variant_results": {},
                    "paper_score": None,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                continue

            # Step 1: Variant extraction
            pbar.set_postfix(pmcid=pmcid, status="extracting variants")
            ve_user = VE_USER_PROMPT_TEMPLATE.format(article_text=paper_text)
            ve_messages = [
                {"role": "system", "content": VE_SYSTEM_PROMPT},
                {"role": "user", "content": ve_user},
            ]
            try:
                ve_response = call_llm(ve_messages, args.model, max_tokens=2048)
            except Exception as e:
                print(f"VE LLM error on {pmcid}: {e}")
                ve_response = ""

            predicted_raw = parse_variant_list(ve_response)
            if predicted_raw is None:
                pbar.set_postfix(pmcid=pmcid, status="parse fail")
                record = {
                    "pmcid": pmcid,
                    "model": args.model,
                    "ground_truth_variants": ground_truth_raw,
                    "predicted_variants": None,
                    "recalled_variants": [],
                    "variant_recall": None,
                    "variant_results": {},
                    "paper_score": None,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                continue

            # Step 2: Compute recall
            predicted_norm = {normalize_variant(v) for v in predicted_raw}
            recalled = predicted_norm & ground_truth_norm
            variant_recall = (
                len(recalled) / len(ground_truth_norm) if ground_truth_norm else 1.0
            )

            # Step 3: Answer questions for recalled variants
            # Cap questions per variant to keep cost/time bounded
            MAX_QUESTIONS_PER_VARIANT = 10
            variant_results: dict[str, dict] = {}
            pbar.set_postfix(pmcid=pmcid, status=f"Q&A ({len(recalled)} variants)")

            for variant in sorted(recalled):
                questions = question_index.get_questions(pmcid, variant)
                if not questions:
                    continue
                questions = questions[:MAX_QUESTIONS_PER_VARIANT]

                # Single batched LLM call for all questions on this variant
                messages = _build_batched_prompt(questions, paper_text, pmcid)
                try:
                    # Scale max_tokens with question count
                    batched_response = call_llm(
                        messages, args.model, max_tokens=256 * len(questions)
                    )
                except Exception as e:
                    print(f"LLM error on {pmcid}/{variant}: {e}")
                    batched_response = ""

                answers = _parse_batched_response(batched_response, len(questions))

                responses_list: list[dict] = []
                q_correct = 0

                for uq, answer in zip(questions, answers):
                    is_correct, expected = _score_question(uq, answer)
                    q_correct += int(is_correct)

                    responses_list.append(
                        {
                            "source_pipeline": uq.source_pipeline,
                            "annotation_id": uq.annotation_id,
                            "model_response": answer,
                            "correct": is_correct,
                            "expected_answer": expected,
                        }
                    )

                q_acc = q_correct / len(questions) if questions else 0.0
                variant_results[variant] = {
                    "questions_asked": len(questions),
                    "questions_correct": q_correct,
                    "question_accuracy": q_acc,
                    "responses": responses_list,
                }

            # Step 4: Compute paper score
            # Only consider recalled variants that had questions
            variants_with_qs = [
                v for v in variant_results if variant_results[v]["questions_asked"] > 0
            ]
            if variants_with_qs:
                mean_q_acc = sum(
                    variant_results[v]["question_accuracy"] for v in variants_with_qs
                ) / len(variants_with_qs)
                paper_score = variant_recall * mean_q_acc
            elif len(recalled) > 0:
                # Recalled variants but none had questions
                paper_score = 0.0
            else:
                paper_score = 0.0

            record = {
                "pmcid": pmcid,
                "model": args.model,
                "ground_truth_variants": ground_truth_raw,
                "predicted_variants": sorted(predicted_norm),
                "recalled_variants": sorted(recalled),
                "variant_recall": variant_recall,
                "variant_results": variant_results,
                "paper_score": paper_score,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    logger.info(f"Responses saved to {responses_path}")
    return responses_path


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def score(args: argparse.Namespace, output_dir: Path | None = None) -> None:
    """Score a responses JSONL and print aggregate summary."""
    responses_path = Path(args.responses_path)
    records: list[dict] = []
    with open(responses_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {responses_path}")

    # Aggregate stats
    scored_records: list[dict] = []
    skipped = 0
    recall_sum = 0.0
    q_acc_sum = 0.0
    paper_score_sum = 0.0
    scored_count = 0

    # Per-pipeline breakdown
    pipeline_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )

    for r in records:
        if r["variant_recall"] is None:
            skipped += 1
            continue

        scored_count += 1
        recall_sum += r["variant_recall"]
        paper_score_sum += r["paper_score"]

        # Collect per-variant question accuracies for this paper
        paper_q_total = 0
        paper_q_correct = 0

        for variant, vr in r["variant_results"].items():
            paper_q_total += vr["questions_asked"]
            paper_q_correct += vr["questions_correct"]
            for resp in vr["responses"]:
                pl = resp["source_pipeline"]
                pipeline_stats[pl]["total"] += 1
                pipeline_stats[pl]["correct"] += int(resp["correct"])

        if paper_q_total > 0:
            q_acc_sum += paper_q_correct / paper_q_total

        scored_records.append(r)

    # Build summary
    model = records[0]["model"] if records else "unknown"
    lines: list[str] = []

    avg_recall = recall_sum / scored_count if scored_count else 0
    avg_q_acc = q_acc_sum / scored_count if scored_count else 0
    avg_paper_score = paper_score_sum / scored_count if scored_count else 0

    lines.append("")
    lines.append("=" * 64)
    lines.append(f"Paper Investigation Results  |  model={model}")
    lines.append("=" * 64)

    lines.append("")
    lines.append(f"  Papers scored:        {scored_count}")
    lines.append(f"  Papers skipped:       {skipped}  (no paper text or parse failure)")

    lines.append("")
    lines.append(f"  Avg Variant Recall:   {avg_recall:.3f}")
    lines.append(
        f"  Avg Question Acc:     {avg_q_acc:.3f}  (across recalled variants only)"
    )
    lines.append(
        f"  Avg Paper Score:      {avg_paper_score:.3f}  (recall x question_acc)"
    )

    # Pipeline breakdown
    lines.append("")
    lines.append("By pipeline breakdown:")
    lines.append(
        f"  {'Pipeline':<20} {'Questions':>10} {'Correct':>10} {'Accuracy':>10}"
    )
    lines.append(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")

    total_q = 0
    total_c = 0
    for pl in sorted(pipeline_stats):
        s = pipeline_stats[pl]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {pl:<20} {s['total']:>10} {s['correct']:>10} {acc:>9.3f}")
        total_q += s["total"]
        total_c += s["correct"]

    lines.append(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    total_acc = total_c / total_q if total_q else 0
    lines.append(f"  {'TOTAL':<20} {total_q:>10} {total_c:>10} {total_acc:>9.3f}")

    # Worst / best papers
    scored_sorted = sorted(scored_records, key=lambda x: x["paper_score"])

    for label, examples in [
        ("WORST PAPERS (by paper_score)", scored_sorted[:3]),
        ("BEST PAPERS (by paper_score)", scored_sorted[-3:]),
    ]:
        lines.append("")
        lines.append(label + ":")
        for i, r in enumerate(examples, 1):
            vr = r["variant_results"]
            total_asked = sum(v["questions_asked"] for v in vr.values())
            total_right = sum(v["questions_correct"] for v in vr.values())
            q_acc = total_right / total_asked if total_asked else 0
            lines.append(
                f"  [{i}] {r['pmcid']}  "
                f"recall={r['variant_recall']:.3f}  "
                f"q_acc={q_acc:.3f}  "
                f"score={r['paper_score']:.3f}"
            )

    # Determine output dir
    if output_dir is None:
        output_dir = responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = responses_path.stem.replace("_responses", "")

    # Save eval results JSONL
    results_path = output_dir / f"{file_prefix}_eval_results.jsonl"
    with open(results_path, "w") as f:
        for r in scored_records:
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
    parser = argparse.ArgumentParser(description="Paper investigation evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run (generate + score)
    run_p = subparsers.add_parser("run", help="Generate and score in one step")
    run_p.add_argument(
        "--model",
        default=os.environ.get("PI_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    run_p.add_argument(
        "--bench-path",
        default="data/variant_bench.jsonl",
        help="Path to variant bench JSONL",
    )
    run_p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of papers (0 = all)",
    )
    run_p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated timestamped dir)",
    )

    # generate
    gen_p = subparsers.add_parser("generate", help="Run variant extraction + Q&A")
    gen_p.add_argument(
        "--model",
        default=os.environ.get("PI_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    gen_p.add_argument(
        "--bench-path",
        default="data/variant_bench.jsonl",
        help="Path to variant bench JSONL",
    )
    gen_p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of papers (0 = all)",
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
    if args.command == "run":
        out_dir = Path(args.output_dir) if args.output_dir else None
        responses_path = generate(args, output_dir=out_dir)
        score(argparse.Namespace(responses_path=str(responses_path)))
    elif args.command == "generate":
        out_dir = Path(args.output_dir) if args.output_dir else None
        generate(args, output_dir=out_dir)
    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
