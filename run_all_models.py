"""
Run yes/no evaluation across multiple models.

Usage:
    python run_all_models.py                  # all 6 models, 10k questions
    python run_all_models.py --limit 100      # all 6 models, 100 questions
    python run_all_models.py --models gpt-4o anthropic/claude-opus-4-6
"""

import argparse
import subprocess
import sys
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
    """Run generate + score for a single model. Returns result summary."""
    print(f"\n{'='*70}")
    print(f"  Starting: {model}  |  limit={limit}")
    print(f"{'='*70}\n")

    # Generate
    gen_cmd = [
        sys.executable, "src/eval/yes_no.py", "generate",
        "--model", model,
        "--limit", str(limit),
    ]
    gen_result = subprocess.run(gen_cmd, capture_output=True, text=True)
    print(gen_result.stderr, end="")

    if gen_result.returncode != 0:
        print(f"FAILED (generate): {model}")
        print(gen_result.stderr)
        return {"model": model, "status": "generate_failed", "error": gen_result.stderr}

    # Find the responses path from generate output
    responses_path = None
    for line in gen_result.stderr.splitlines():
        if "Responses saved to" in line:
            responses_path = line.split("Responses saved to")[-1].strip()
            break

    if not responses_path:
        return {"model": model, "status": "no_responses_path", "error": gen_result.stderr}

    # Score
    score_cmd = [
        sys.executable, "src/eval/yes_no.py", "score",
        "--responses-path", responses_path,
    ]
    score_result = subprocess.run(score_cmd, capture_output=True, text=True)
    print(score_result.stderr, end="")
    print(score_result.stdout, end="")

    if score_result.returncode != 0:
        print(f"FAILED (score): {model}")
        return {"model": model, "status": "score_failed", "error": score_result.stderr}

    return {
        "model": model,
        "status": "ok",
        "responses_path": responses_path,
        "summary": score_result.stdout,
    }


def main():
    parser = argparse.ArgumentParser(description="Run yes/no eval across models")
    parser.add_argument(
        "--limit", type=int, default=10000,
        help="Number of questions per model (default: 10000)",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help="Model identifiers to evaluate",
    )
    args = parser.parse_args()

    start = datetime.now()
    results = []

    for model in args.models:
        result = run_model(model, args.limit)
        results.append(result)

    elapsed = datetime.now() - start

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  ALL MODELS COMPLETE  |  {elapsed.total_seconds():.0f}s elapsed")
    print(f"{'='*70}\n")

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
