"""
Run variant extraction across multiple models.

Usage:
    python run_all_models_ve.py                  # all 6 models, all articles
    python run_all_models_ve.py --limit 100      # all 6 models, 100 articles
    python run_all_models_ve.py --models gpt-4o anthropic/claude-opus-4-6
    python run_all_models_ve.py --workers 3      # run 3 models in parallel
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

SCRIPT = "src/modules/variant_extraction/variant_extraction.py"


def _make_run_dir() -> Path:
    """Create a single timestamped run directory for all models."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{ts}_all_models_VE"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_model(model: str, limit: int, output_dir: Path) -> dict:
    """Run variant extraction pipeline for a single model.

    All output is captured and returned (not printed) so parallel runs
    don't interleave on the terminal.
    """
    log_lines: list[str] = []
    log_lines.append(f"\n{'=' * 70}")
    log_lines.append(f"  Starting: {model}  |  limit={limit}")
    log_lines.append(f"{'=' * 70}\n")

    cmd = [
        sys.executable,
        SCRIPT,
        "run",
        "--model",
        model,
        "--limit",
        str(limit),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_lines.append(result.stderr)
    log_lines.append(result.stdout)

    if result.returncode != 0:
        log_lines.append(f"FAILED: {model}")
        return {
            "model": model,
            "status": "failed",
            "error": result.stderr,
            "log": "\n".join(log_lines),
        }

    return {
        "model": model,
        "status": "ok",
        "summary": result.stdout,
        "log": "\n".join(log_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Run variant extraction across models")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of articles per model (0 = all)",
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

    run_dir = _make_run_dir()
    start = datetime.now()
    results: list[dict] = []

    print(f"Running {len(args.models)} models with up to {args.workers} workers")
    print(f"Output directory: {run_dir}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_model, model, args.limit, run_dir): model
            for model in args.models
        }

        for future in as_completed(futures):
            model = futures[future]
            result = future.result()
            results.append(result)
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
    print(f"  Output: {run_dir}")
    print(f"{'=' * 70}\n")

    for r in results:
        status = "OK" if r["status"] == "ok" else f"FAILED ({r['status']})"
        print(f"  {r['model']:<45} {status}")
        if r["status"] == "ok" and r.get("summary"):
            for line in r["summary"].splitlines():
                if "Variant Extraction Results" in line:
                    print(f"    {line.strip()}")
                    break

    print()


if __name__ == "__main__":
    main()
