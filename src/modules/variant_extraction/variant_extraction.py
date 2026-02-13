"""
Variant Extraction — extract pharmacogenetic variants from articles and score
against ground truth.

Usage:
    python src/modules/variant_extraction/variant_extraction.py run --model gpt-4o-mini --limit 10
    python src/modules/variant_extraction/variant_extraction.py generate --model gpt-4o-mini --limit 10
    python src/modules/variant_extraction/variant_extraction.py score --responses-path runs/20260212_120000_gpt-4o-mini/variant_responses.jsonl
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

load_dotenv()

SYSTEM_PROMPT = (
    "You are a pharmacogenomics expert. "
    "Extract genetic variants from scientific articles with precise formatting."
)

USER_PROMPT_TEMPLATE = """\
Extract all pharmacogenetic variants from the following article.

VARIANT TYPES:
1. rsIDs: rs followed by numbers (rs9923231, rs1057910)
2. Star alleles: Gene*Number format for pharmacogenes:
   - CYP genes: CYP2C9*3, CYP2C19*17, CYP2D6*4, CYP3A5*3
   - Other genes: NUDT15*3, TPMT*3A, UGT1A1*28, DPYD*2A, NAT2*5
3. HLA alleles: HLA-GENE*XX:XX format (HLA-B*58:01, HLA-DRB1*03:01)
4. Metabolizer phenotypes: Gene + phenotype (CYP2D6 poor metabolizer, NAT2 slow acetylator)

NORMALIZATION RULES:
- Star alleles: Always use GENE*NUMBER (CYP2C9*3, not CYP2C9 *3)
- HLA: Always include HLA- prefix and use colon separator (HLA-B*58:01)
- Include diplotypes if mentioned (e.g., *1/*3 should be listed as the individual alleles)

Return ONLY a JSON array of unique variants. No explanations.

Article:
{article_text}"""

DEFAULT_MODEL = "gpt-4o-mini"


def _make_run_dir(model: str) -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    run_dir = Path("runs") / f"{ts}_{model_slug}_VE"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate(args: argparse.Namespace, output_dir: Path | None = None) -> Path:
    """Run LLM on articles to extract variants, save raw responses to JSONL."""
    bench_path = Path(args.bench_path)
    articles: list[dict] = []
    with open(bench_path) as f:
        for line in f:
            articles.append(json.loads(line))
    logger.info(f"Loaded {len(articles)} articles from {bench_path}")

    paper_index = build_paper_index()

    if args.limit > 0:
        articles = articles[: args.limit]
        logger.info(f"Limiting to {len(articles)} articles")

    logger.info(f"Generating responses with model={args.model}")

    model_slug = args.model.replace("/", "_")
    if output_dir is None:
        output_dir = _make_run_dir(args.model)
        file_prefix = "variant"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = f"{model_slug}_variant"
    responses_path = output_dir / f"{file_prefix}_responses.jsonl"

    with open(responses_path, "w") as out_f:
        pbar = tqdm(articles, desc=f"Extracting ({args.model})", unit="article")
        for article in pbar:
            pmcid = article["pmcid"]
            paper_text = load_paper(pmcid, paper_index)

            if paper_text is None:
                pbar.set_postfix(pmcid=pmcid, status="no paper")
                record = {
                    "pmcid": pmcid,
                    "article_title": article.get("article_title", ""),
                    "ground_truth": article["variants"],
                    "response": "",
                    "had_paper_context": False,
                    "model": args.model,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                continue

            user_content = USER_PROMPT_TEMPLATE.format(article_text=paper_text)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            try:
                pbar.set_postfix(pmcid=pmcid, status="calling LLM")
                response = call_llm(messages, args.model, max_tokens=2048)
            except Exception as e:
                logger.error(f"LLM error on {pmcid}: {e}")
                response = ""

            record = {
                "pmcid": pmcid,
                "article_title": article.get("article_title", ""),
                "ground_truth": article["variants"],
                "response": response,
                "had_paper_context": True,
                "model": args.model,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    logger.info(f"Responses saved to {responses_path}")
    return responses_path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_variant_list(response: str) -> list[str] | None:
    """Extract a list of variant strings from an LLM response.

    Tries to parse a JSON array from the response. Returns None if
    unparseable.
    """
    text = response.strip()
    if not text:
        return None

    # Try direct JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass

    return None


def classify_variant(variant: str) -> str:
    """Classify a variant string into a type category."""
    v = variant.strip()
    if re.match(r"^rs\d+", v, re.IGNORECASE):
        return "rsID"
    if re.match(r"^HLA-", v, re.IGNORECASE):
        return "HLA"
    if re.search(r"\*\d", v):
        return "star_allele"
    if re.search(r"(metabolizer|acetylator)", v, re.IGNORECASE):
        return "phenotype"
    return "other"


def normalize_variant(variant: str) -> str:
    """Normalize a variant string for comparison."""
    v = variant.strip()
    # Collapse internal whitespace around *
    v = re.sub(r"\s*\*\s*", "*", v)
    return v


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def score(args: argparse.Namespace, output_dir: Path | None = None) -> None:
    """Score a responses JSONL file and print summary."""
    responses_path = Path(args.responses_path)
    records: list[dict] = []
    with open(responses_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} responses from {responses_path}")

    results: list[dict] = []
    total_recall_sum = 0.0
    total_scored = 0
    total_parse_failures = 0

    type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0})

    for r in records:
        predicted_raw = parse_variant_list(r["response"])
        ground_truth_raw = r["ground_truth"]

        if predicted_raw is None:
            total_parse_failures += 1
            results.append(
                {
                    **r,
                    "predicted": None,
                    "recall": None,
                    "parse_failure": True,
                }
            )
            # Count all ground truth as FN for type stats
            for gt_v in ground_truth_raw:
                vtype = classify_variant(gt_v)
                type_stats[vtype]["fn"] += 1
            continue

        # Normalize for comparison
        predicted_normalized = {normalize_variant(v) for v in predicted_raw}
        ground_truth_normalized = {normalize_variant(v) for v in ground_truth_raw}

        tp = predicted_normalized & ground_truth_normalized
        fn = ground_truth_normalized - predicted_normalized

        recall = (
            len(tp) / len(ground_truth_normalized) if ground_truth_normalized else 1.0
        )

        total_recall_sum += recall
        total_scored += 1

        # Per-type stats
        for v in tp:
            vtype = classify_variant(v)
            type_stats[vtype]["tp"] += 1
        for v in fn:
            vtype = classify_variant(v)
            type_stats[vtype]["fn"] += 1

        results.append(
            {
                **r,
                "predicted": sorted(predicted_normalized),
                "true_positives": sorted(tp),
                "false_negatives": sorted(fn),
                "recall": recall,
                "parse_failure": False,
            }
        )

    # Build summary text
    model = records[0]["model"] if records else "unknown"
    lines: list[str] = []

    avg_recall = total_recall_sum / total_scored if total_scored else 0

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"Variant Extraction Results  |  model={model}  |  Recall={avg_recall:.3f}"
    )
    lines.append("=" * 70)

    lines.append("")
    lines.append(f"  Articles scored: {total_scored}/{len(records)}")
    lines.append(f"  Parse failures:  {total_parse_failures}/{len(records)}")

    no_paper = sum(1 for r in results if not r["had_paper_context"])
    if no_paper:
        lines.append(f"  Articles without paper context: {no_paper}/{len(records)}")

    # Per-type breakdown
    lines.append("")
    lines.append("By variant type:")
    lines.append(f"  {'Type':<15} {'TP':>6} {'FN':>6} {'Recall':>10}")
    lines.append(f"  {'-' * 15} {'-' * 6} {'-' * 6} {'-' * 10}")
    for vtype in sorted(type_stats):
        s = type_stats[vtype]
        tp, fn = s["tp"], s["fn"]
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        lines.append(f"  {vtype:<15} {tp:>6} {fn:>6} {r:>9.3f}")

    # Aggregate micro-averaged recall
    total_tp = sum(s["tp"] for s in type_stats.values())
    total_fn = sum(s["fn"] for s in type_stats.values())
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    lines.append(f"  {'-' * 15} {'-' * 6} {'-' * 6} {'-' * 10}")
    lines.append(f"  {'MICRO-AVG':<15} {total_tp:>6} {total_fn:>6} {micro_r:>9.3f}")

    lines.append("")
    lines.append(f"  Macro-avg Recall: {avg_recall:.3f}")

    # Example articles: 3 best, 3 worst by recall
    scored_results = [r for r in results if not r["parse_failure"]]
    scored_results_sorted = sorted(scored_results, key=lambda x: x["recall"])

    for label, examples in [
        ("WORST RECALL", scored_results_sorted[:3]),
        ("BEST RECALL", scored_results_sorted[-3:]),
    ]:
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"Example {label} articles")
        lines.append("-" * 70)
        for i, r in enumerate(examples, 1):
            lines.append("")
            lines.append(f"  [{i}] pmcid={r['pmcid']}  Recall={r['recall']:.3f}")
            lines.append(f"  Title: {r['article_title'][:80]}")
            lines.append(
                f"  Ground truth ({len(r['ground_truth'])}): {r['ground_truth'][:5]}"
            )
            lines.append(f"  Predicted ({len(r['predicted'])}): {r['predicted'][:5]}")
            if r["false_negatives"]:
                lines.append(f"  Missed: {r['false_negatives'][:5]}")

    # Determine output dir
    if output_dir is None:
        output_dir = responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive file prefix from responses filename (e.g. "gpt-4o_variant" or "variant")
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
    parser = argparse.ArgumentParser(description="Variant extraction evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run (generate + score)
    run_p = subparsers.add_parser("run", help="Generate and score in one step")
    run_p.add_argument(
        "--model",
        default=os.environ.get("VARIANT_MODEL", DEFAULT_MODEL),
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
        help="Limit number of articles (0 = all)",
    )
    run_p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated timestamped dir)",
    )

    # generate
    gen_p = subparsers.add_parser("generate", help="Extract variants via LLM")
    gen_p.add_argument(
        "--model",
        default=os.environ.get("VARIANT_MODEL", DEFAULT_MODEL),
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
        help="Limit number of articles (0 = all)",
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
