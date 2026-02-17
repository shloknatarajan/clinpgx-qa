#!/usr/bin/env python3
"""
Run variant-chain evaluation across multiple models.

Self-contained script for the variant_allele_chains.jsonl dataset.
Sends each 4-step chain to the model as a multi-turn conversation
(with the paper as context in turn 1), then scores all responses.

Requirements:  pip install litellm python-dotenv tenacity loguru tqdm

Usage:
    python data/chained_questions/run_all_models.py                        # all models, all chains
    python data/chained_questions/run_all_models.py --limit 20             # first 20 chains
    python data/chained_questions/run_all_models.py --models gpt-4o-mini   # single model
"""

import argparse
import json
import math
import os
import re
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent
PAPERS_DIR = DATA_DIR / "papers"
QUESTIONS_PATH = SCRIPT_DIR / "variant_allele_chains.jsonl"

MODELS = [
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o-mini",
    "gpt-5.2",
]

SYSTEM_PROMPT = (
    "You are a pharmacogenomics expert. "
    "You will be given a research paper and asked a series of questions about "
    "genetic predictors, comparison types, and statistical evidence. "
    "Answer each question concisely and precisely based on the paper content."
)

_LLM_TIMEOUT = 300

_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5-", "gpt-5.")
_REASONING_EXACT = {"gpt-5"}
_REASONING_GEMINI_PREFIXES = ("gemini-2.5", "gemini-3")


# ---------------------------------------------------------------------------
# LLM calling  (matches src/eval/llm.py pattern)
# ---------------------------------------------------------------------------

def _is_reasoning_model(model: str) -> bool:
    bare = model.lower().split("/", 1)[-1]
    if bare in _REASONING_EXACT:
        return True
    if any(bare.startswith(p) for p in _REASONING_PREFIXES):
        return True
    gemini_bare = model.lower().removeprefix("gemini/")
    return any(gemini_bare.startswith(p) for p in _REASONING_GEMINI_PREFIXES)


@retry(
    retry=retry_if_exception_type(
        (litellm.RateLimitError, litellm.exceptions.APIConnectionError, litellm.Timeout)
    ),
    wait=wait_exponential(min=2, max=120),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> str:
    is_reasoning = _is_reasoning_model(model)
    kwargs: dict = {"drop_params": True, "temperature": temperature}
    if is_reasoning:
        kwargs["max_completion_tokens"] = min(max(max_tokens * 8, 4096), 16_000)
    else:
        kwargs["max_tokens"] = max_tokens
    response = litellm.completion(
        model=model, messages=messages, timeout=_LLM_TIMEOUT, **kwargs,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


# ---------------------------------------------------------------------------
# Paper loading
# ---------------------------------------------------------------------------

def _load_paper(pmc_id: str) -> str | None:
    path = PAPERS_DIR / f"{pmc_id}.md"
    if path.exists():
        return path.read_text()
    return None


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate(
    model: str, chains: list[dict], output_dir: Path,
) -> Path:
    """Send each chain to the model and write responses."""
    model_slug = model.replace("/", "_")
    responses_path = output_dir / f"{model_slug}_variant_chain_responses.jsonl"

    logger.info(f"[{model}] Generating {len(chains)} chains → {responses_path}")

    with open(responses_path, "w") as out_f:
        for ci, chain in enumerate(chains):
            pmc_id = chain.get("pmc_id", "")
            paper_text = _load_paper(pmc_id) if pmc_id else None
            turns = chain["turns"]
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            turn_records: list[dict] = []

            for turn in turns:
                question = turn["question"]
                if turn["turn"] == 1 and paper_text:
                    user_content = (
                        f"## Paper (PMCID {pmc_id})\n\n{paper_text}\n\n"
                        f"## Question\n\n{question}"
                    )
                else:
                    user_content = question

                messages.append({"role": "user", "content": user_content})
                try:
                    resp = call_llm(messages, model)
                except Exception as e:
                    logger.error(
                        f"[{model}] Error chain={chain['chain_id']} "
                        f"turn={turn['turn']}: {e}"
                    )
                    resp = ""
                messages.append({"role": "assistant", "content": resp})

                turn_records.append({
                    "turn": turn["turn"],
                    "step": turn["step"],
                    "question": question,
                    "answer": turn["answer"],
                    "response": resp,
                })

            # Carry through p-value metadata for scoring
            if len(turns) == 4:
                turn_records[3]["answer_p_value_parsed"] = turns[3].get(
                    "answer_p_value_parsed"
                )

            record = {
                "chain_id": chain["chain_id"],
                "pmid": chain["pmid"],
                "pmc_id": pmc_id,
                "focus_comparison_level": chain["focus_comparison_level"],
                "num_turns": len(turns),
                "model": model,
                "turns": turn_records,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (ci + 1) % 10 == 0 or (ci + 1) == len(chains):
                logger.info(f"  [{model}] [{ci + 1}/{len(chains)}]")

    logger.info(f"[{model}] Responses saved to {responses_path}")
    return responses_path


# ---------------------------------------------------------------------------
# Scoring helpers  (all exact-match based)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _normalize_comp(s: str) -> str:
    """Canonicalize a comparison string: sort items on each side, lowercase."""
    s = _normalize(s)
    if " vs " in s:
        lhs, rhs = s.split(" vs ", 1)
        lhs = " + ".join(sorted(p.strip() for p in lhs.split("+")))
        rhs = " + ".join(sorted(p.strip() for p in rhs.split("+")))
        return f"{lhs} vs {rhs}"
    return s


def score_step1(response: str, ground_truth: list[str]) -> dict:
    """Step 1 — Predictor identification.

    Exact match: every GT predictor string must appear verbatim
    (case-insensitive) in the response.
    """
    resp_norm = _normalize(response)
    found, missing = [], []
    for pred in ground_truth:
        if _normalize(pred) in resp_norm:
            found.append(pred)
        else:
            missing.append(pred)
    recall = len(found) / len(ground_truth) if ground_truth else 1.0
    return {"correct": len(missing) == 0, "recall": recall,
            "found": found, "missing": missing}


def _parse_comp_types(response: str) -> dict[str, list[str]]:
    """Parse Step 2 response into {predictor: [type_labels]}.

    Expects lines like:
      rs1234: allele-level comparison
      CYP2D6 poor metabolizer: metabolizer or phenotype-group comparison
    """
    result = defaultdict(list)
    type_map = {
        "allele-level": "allele", "allele level": "allele",
        "genotype-level": "genotype", "genotype level": "genotype",
        "metabolizer": "metabolizer_or_phenotype_group",
        "phenotype-group": "metabolizer_or_phenotype_group",
        "phenotype group": "metabolizer_or_phenotype_group",
        "not specified": "not_specified", "not_specified": "not_specified",
    }
    for line in response.splitlines():
        line = line.strip().lstrip("-\u2022*123456789.)")
        if ":" not in line:
            continue
        parts = line.split(":", 1)
        pred = parts[0].strip()
        text = parts[1].lower()
        for pattern, label in type_map.items():
            if pattern in text and label not in result[pred]:
                result[pred].append(label)
    return dict(result)


def score_step2(response: str, ground_truth: dict[str, list[str]]) -> dict:
    """Step 2 — Comparison type classification.

    Parsed types per predictor must equal GT types exactly.
    Falls back to keyword scanning near the predictor mention if structured
    parsing fails.
    """
    parsed = _parse_comp_types(response)
    resp_norm = _normalize(response)
    all_correct = True
    details = {}

    type_keywords = {
        "allele": ["allele-level", "allele level"],
        "genotype": ["genotype-level", "genotype level"],
        "metabolizer_or_phenotype_group": [
            "metabolizer", "phenotype-group", "phenotype group",
        ],
        "not_specified": ["not specified", "not_specified"],
    }

    for pred, expected in ground_truth.items():
        pred_norm = _normalize(pred)
        got = None
        for ppred, ptypes in parsed.items():
            if _normalize(ppred) == pred_norm:
                got = sorted(ptypes)
                break
        if got is None:
            # Fallback: keyword scan in full response
            got = []
            for etype in expected:
                for kw in type_keywords.get(etype, []):
                    if kw in resp_norm and etype not in got:
                        got.append(etype)
            got = sorted(got)

        match = sorted(expected) == got
        if not match:
            all_correct = False
        details[pred] = {"expected": sorted(expected), "got": got, "match": match}

    return {"correct": all_correct, "details": details}


def score_step3(
    response: str, ground_truth: dict[str, list[str]] | str,
) -> dict:
    """Step 3 — Comparison extraction.

    Each GT comparison string must appear in the response.  Tries exact
    canonicalized match first, then falls back to checking that both sides
    of the comparison appear individually.
    """
    if ground_truth == "None":
        return {"correct": "none" in _normalize(response), "found": 0, "expected": 0}

    resp_norm = _normalize(response)
    total_exp, total_found = 0, 0
    details = {}

    for pred, comps in ground_truth.items():
        found, missing = [], []
        for comp in comps:
            total_exp += 1
            cn = _normalize_comp(comp)
            if cn in resp_norm:
                found.append(comp)
                total_found += 1
            elif " vs " in cn:
                # Fuzzy fallback: both sides appear somewhere in response
                lhs, rhs = cn.split(" vs ", 1)
                if lhs in resp_norm and rhs in resp_norm:
                    found.append(comp)
                    total_found += 1
                else:
                    missing.append(comp)
            else:
                missing.append(comp)
        details[pred] = {"found": found, "missing": missing}

    return {"correct": total_found == total_exp, "found": total_found,
            "expected": total_exp, "details": details}


def _extract_pvalue(text: str) -> float | None:
    """Extract the smallest p-value mentioned in text."""
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u00d7", "x")
    sup = str.maketrans(
        "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b",
        "0123456789-",
    )
    text = text.translate(sup)
    vals: list[float] = []
    for m in re.finditer(
        r"(-?\d+\.?\d*)\s*[xX]\s*10\s*\^?\s*\(?\s*(-?\d+)\s*\)?", text,
    ):
        try:
            vals.append(float(m.group(1)) * 10 ** int(m.group(2)))
        except (ValueError, OverflowError):
            pass
    for m in re.finditer(r"-?\d+\.?\d*[eE][+-]?\d+", text):
        try:
            vals.append(float(m.group()))
        except ValueError:
            pass
    for m in re.finditer(r"[pP]\s*[<>=\u2264\u2265~\-]?\s*=?\s*(\d+\.?\d*|\.\d+)", text):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            pass
    return min(vals) if vals else None


def score_step4(response: str, ground_truth: str, gt_pval: float | None) -> dict:
    """Step 4 — Evidence-based selection.

    Primary: GT answer string must appear verbatim in the response.
    Fallback: check that the predictor name and the correct p-value appear.
    """
    if ground_truth == "None":
        return {"correct": "none" in _normalize(response), "method": "none_check"}

    resp_norm = _normalize(response)
    gt_norm = _normalize(ground_truth)

    # Exact string match
    if gt_norm in resp_norm:
        return {"correct": True, "method": "exact_match"}

    # Fuzzy: predictor name present + correct p-value mentioned
    if " (" in gt_norm:
        pred_part = gt_norm.split(" (", 1)[0]
        if pred_part in resp_norm and gt_pval is not None and gt_pval != 0:
            resp_pval = _extract_pvalue(response)
            if resp_pval is not None and math.isclose(resp_pval, gt_pval, rel_tol=0.05):
                return {"correct": True, "method": "fuzzy_match"}

    return {"correct": False, "method": "no_match"}


def score_chain(chain: dict) -> dict:
    """Score all turns of one chain. Returns result dict."""
    scored_turns = []
    all_correct = True

    for t in chain["turns"]:
        step = t["step"]
        resp = t["response"]
        gt = t["answer"]

        if step == "predictor_identification":
            r = score_step1(resp, gt)
        elif step == "comparison_type_classification":
            r = score_step2(resp, gt)
        elif step == "lowest_level_extraction":
            r = score_step3(resp, gt)
        elif step == "evidence_based_selection":
            gt_pval = t.get("answer_p_value_parsed")
            r = score_step4(resp, gt, gt_pval)
        else:
            r = {"correct": False}

        if not r["correct"]:
            all_correct = False
        scored_turns.append({**t, **r})

    return {
        "chain_id": chain["chain_id"],
        "pmid": chain["pmid"],
        "pmc_id": chain.get("pmc_id"),
        "focus_comparison_level": chain.get("focus_comparison_level"),
        "num_turns": chain["num_turns"],
        "model": chain["model"],
        "all_correct": all_correct,
        "turns": scored_turns,
    }


# ---------------------------------------------------------------------------
# Score & summary
# ---------------------------------------------------------------------------

def score_and_summarize(responses_path: Path, output_dir: Path) -> str:
    """Score all responses and return the one-line summary."""
    chains = []
    with open(responses_path) as f:
        for line in f:
            chains.append(json.loads(line))

    results = [score_chain(c) for c in chains]
    model = chains[0]["model"] if chains else "unknown"
    num_chains = len(results)

    turn_correct = sum(
        int(t["correct"]) for r in results for t in r["turns"]
    )
    turn_total = sum(len(r["turns"]) for r in results)
    chain_correct = sum(int(r["all_correct"]) for r in results)

    step_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        for t in r["turns"]:
            step_stats[t["step"]]["total"] += 1
            step_stats[t["step"]]["correct"] += int(t["correct"])

    level_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        lvl = r.get("focus_comparison_level", "unknown")
        level_stats[lvl]["total"] += 1
        level_stats[lvl]["correct"] += int(r["all_correct"])

    # Build summary text
    lines = [""]
    headline = (
        f"Variant Chain Results  |  model={model}  |  "
        f"turns: {turn_correct}/{turn_total} "
        f"({turn_correct / turn_total:.1%})  |  "
        f"chains: {chain_correct}/{num_chains} "
        f"({chain_correct / num_chains:.1%})"
    )
    lines.append("=" * 70)
    lines.append(headline)
    lines.append("=" * 70)

    lines.append("")
    lines.append("By step:")
    lines.append(f"  {'Step':<40} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 10}")
    for step in [
        "predictor_identification", "comparison_type_classification",
        "lowest_level_extraction", "evidence_based_selection",
    ]:
        s = step_stats[step]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {step:<40} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    lines.append("")
    lines.append("By comparison level:")
    lines.append(f"  {'Level':<35} {'Correct':>8} {'Total':>8} {'Chain Acc':>10}")
    lines.append(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 10}")
    for lvl in sorted(level_stats):
        s = level_stats[lvl]
        acc = s["correct"] / s["total"] if s["total"] else 0
        lines.append(f"  {lvl:<35} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    # Sample turns
    all_flat = []
    for r in results:
        for t in r["turns"]:
            all_flat.append({
                "chain_id": r["chain_id"], "pmc_id": r.get("pmc_id"),
                "turn": t["turn"], "step": t["step"],
                "question": t["question"], "response": t["response"],
                "answer": t["answer"], "correct": t["correct"],
            })
    correct_ex = [t for t in all_flat if t["correct"]]
    incorrect_ex = [t for t in all_flat if not t["correct"]]
    random.shuffle(correct_ex)
    random.shuffle(incorrect_ex)
    for label, examples in [("CORRECT", correct_ex[:5]), ("INCORRECT", incorrect_ex[:5])]:
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"Example {label} turns ({len(examples)} shown)")
        lines.append("-" * 70)
        for i, t in enumerate(examples, 1):
            lines.append(f"  [{i}] chain={t['chain_id']}  turn={t['turn']}  step={t['step']}")
            lines.append(f"  Q: {t['question'][:180]}...")
            lines.append(f"  R: {t['response'][:180].replace(chr(10), ' ')}...")
            ans = t["answer"]
            if isinstance(ans, (dict, list)):
                ans = json.dumps(ans)[:180]
            lines.append(f"  GT: {ans}")

    # Save files
    file_prefix = responses_path.stem.replace("_responses", "")
    results_path = output_dir / f"{file_prefix}_eval_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = "\n".join(lines)
    summary_path = output_dir / f"{file_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    print(summary)
    return headline


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "openai"


def _short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def run_single_model(
    model: str, chains: list[dict], output_dir: Path,
) -> dict:
    """Generate + score for one model. Returns summary dict."""
    try:
        resp_path = generate(model, chains, output_dir)
        headline = score_and_summarize(resp_path, output_dir)
        return {"model": model, "status": "ok", "headline": headline}
    except Exception as e:
        logger.error(f"[{model}] FAILED: {e}")
        return {"model": model, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run variant-chain eval across models"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit chains per model (0 = all)",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help="Model identifiers to evaluate",
    )
    parser.add_argument(
        "--questions-path", default=str(QUESTIONS_PATH),
        help="Path to variant chain JSONL",
    )
    args = parser.parse_args()

    # Load chains once
    chains = []
    with open(args.questions_path) as f:
        for line in f:
            chains.append(json.loads(line))
    logger.info(f"Loaded {len(chains)} chains from {args.questions_path}")
    if args.limit > 0:
        chains = chains[: args.limit]
        logger.info(f"Limited to {len(chains)} chains")

    # Create run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{ts}_variant_chain_all_models"
    run_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()

    # Group by provider for parallel execution
    provider_groups: dict[str, list[str]] = defaultdict(list)
    for m in args.models:
        provider_groups[_provider(m)].append(m)

    print(f"Running {len(args.models)} models × {len(chains)} chains")
    for prov, models in provider_groups.items():
        print(f"  {prov}: {[_short(m) for m in models]} (sequential within provider)")
    print(f"Output: {run_dir}\n")

    all_results: list[dict] = []

    def _run_provider(models: list[str]) -> list[dict]:
        group_results = []
        for model in models:
            r = run_single_model(model, chains, run_dir)
            group_results.append(r)
        return group_results

    with ThreadPoolExecutor(max_workers=len(provider_groups)) as pool:
        futures = {
            pool.submit(_run_provider, models): prov
            for prov, models in provider_groups.items()
        }
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Sort to match original model order
    model_order = {m: i for i, m in enumerate(args.models)}
    all_results.sort(key=lambda r: model_order.get(r["model"], 999))

    elapsed = datetime.now() - start

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  ALL MODELS COMPLETE  |  {elapsed.total_seconds():.0f}s elapsed")
    print(f"  Output: {run_dir}")
    print(f"{'=' * 70}\n")

    for r in all_results:
        status = "OK" if r["status"] == "ok" else f"FAILED ({r.get('error', '')[:60]})"
        print(f"  {r['model']:<45} {status}")
        if r.get("headline"):
            print(f"    {r['headline']}")
    print()


if __name__ == "__main__":
    main()
