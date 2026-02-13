"""
Run yes/no evaluation across multiple models.

Usage:
    python run_all_models.py                  # all 6 models, 10k questions
    python run_all_models.py --limit 100      # all 6 models, 100 questions
    python run_all_models.py --models gpt-4o anthropic/claude-opus-4-6
    python run_all_models.py --workers 3      # run 3 models in parallel
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o",
    "gpt-5",
    "gpt-5.2",
]


def run_model(model: str, limit: int) -> dict:
    """Run generate + score for a single model. Returns result summary.

    All output is captured and returned (not printed) so parallel runs
    don't interleave on the terminal.
    """
    log_lines: list[str] = []
    log_lines.append(f"\n{'=' * 70}")
    log_lines.append(f"  Starting: {model}  |  limit={limit}")
    log_lines.append(f"{'=' * 70}\n")

    # Generate
    gen_cmd = [
        sys.executable,
        "src/eval/yes_no.py",
        "generate",
        "--model",
        model,
        "--limit",
        str(limit),
    ]
    gen_result = subprocess.run(gen_cmd, capture_output=True, text=True)
    log_lines.append(gen_result.stderr)

    if gen_result.returncode != 0:
        log_lines.append(f"FAILED (generate): {model}")
        log_lines.append(gen_result.stderr)
        return {
            "model": model,
            "status": "generate_failed",
            "error": gen_result.stderr,
            "log": "\n".join(log_lines),
        }

    # Find the responses path from generate output
    responses_path = None
    for line in gen_result.stderr.splitlines():
        if "Responses saved to" in line:
            responses_path = line.split("Responses saved to")[-1].strip()
            break

    if not responses_path:
        return {
            "model": model,
            "status": "no_responses_path",
            "error": gen_result.stderr,
            "log": "\n".join(log_lines),
        }

    # Score
    score_cmd = [
        sys.executable,
        "src/eval/yes_no.py",
        "score",
        "--responses-path",
        responses_path,
    ]
    score_result = subprocess.run(score_cmd, capture_output=True, text=True)
    log_lines.append(score_result.stderr)
    log_lines.append(score_result.stdout)

    if score_result.returncode != 0:
        log_lines.append(f"FAILED (score): {model}")
        return {
            "model": model,
            "status": "score_failed",
            "error": score_result.stderr,
            "log": "\n".join(log_lines),
        }

    return {
        "model": model,
        "status": "ok",
        "responses_path": responses_path,
        "summary": score_result.stdout,
        "log": "\n".join(log_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Run yes/no eval across models")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of questions per model (default: 10000)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Model identifiers to evaluate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(MODELS),
        help="Max parallel workers (default: number of models)",
    )
    args = parser.parse_args()

    start = datetime.now()
    results: list[dict] = []

    print(f"Running {len(args.models)} models with up to {args.workers} workers\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_model, model, args.limit): model for model in args.models
        }

        for future in as_completed(futures):
            model = futures[future]
            result = future.result()
            results.append(result)
            # Print this model's captured log now that it's done
            print(result.get("log", ""))
            status = (
                "OK" if result["status"] == "ok" else f"FAILED ({result['status']})"
            )
            print(f"  >> {model} finished: {status}\n")

    # Sort results to match the original model order
    model_order = {m: i for i, m in enumerate(args.models)}
    results.sort(key=lambda r: model_order.get(r["model"], 0))

    elapsed = datetime.now() - start

    # Final summary
    print(f"\n\n{'=' * 70}")
    print(f"  ALL MODELS COMPLETE  |  {elapsed.total_seconds():.0f}s elapsed")
    print(f"{'=' * 70}\n")

    for r in results:
        status = "OK" if r["status"] == "ok" else f"FAILED ({r['status']})"
        print(f"  {r['model']:<45} {status}")
        if r["status"] == "ok" and r.get("summary"):
            # Extract the accuracy line
            for line in r["summary"].splitlines():
                if "Yes/No Results" in line:
                    print(f"    {line.strip()}")
                    break

    print()


if __name__ == "__main__":
    main()
