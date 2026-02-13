"""
Run evaluation across multiple models.

All outputs are saved to a single timestamped directory under runs/.

Usage:
    python run_all_models.py --limit 100                     # yes/no only (default)
    python run_all_models.py --limit 100 --dataset chained   # chained only
    python run_all_models.py --limit 100 --dataset all       # both pipelines
    python run_all_models.py --models gpt-4o anthropic/claude-opus-4-6
    python run_all_models.py --workers 3                     # run 3 models in parallel
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o",
    "gpt-5",
    "gpt-5.2",
]

PIPELINES = {
    "yes_no": {
        "script": "src/eval/yes_no.py",
        "summary_marker": "Yes/No Results",
    },
    "chained": {
        "script": "src/eval/chained.py",
        "summary_marker": "Chained Results",
    },
}


def _make_run_dir() -> Path:
    """Create a single timestamped run directory for all models."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{ts}_all_models"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_pipeline(
    model: str,
    limit: int,
    pipeline_name: str,
    output_dir: Path,
    log_lines: list[str],
) -> dict:
    """Run generate + score for a single pipeline. Returns result dict."""
    cfg = PIPELINES[pipeline_name]
    script = cfg["script"]

    log_lines.append(f"  [{pipeline_name}] generating...")

    # Generate
    gen_cmd = [
        sys.executable,
        script,
        "generate",
        "--model",
        model,
        "--limit",
        str(limit),
        "--output-dir",
        str(output_dir),
    ]
    gen_result = subprocess.run(gen_cmd, capture_output=True, text=True)
    log_lines.append(gen_result.stderr)

    if gen_result.returncode != 0:
        log_lines.append(f"  [{pipeline_name}] FAILED (generate): {model}")
        return {
            "pipeline": pipeline_name,
            "status": "generate_failed",
            "error": gen_result.stderr,
        }

    # Find the responses path from generate output
    responses_path = None
    for line in gen_result.stderr.splitlines():
        if "Responses saved to" in line:
            responses_path = line.split("Responses saved to")[-1].strip()
            break

    if not responses_path:
        return {
            "pipeline": pipeline_name,
            "status": "no_responses_path",
            "error": gen_result.stderr,
        }

    log_lines.append(f"  [{pipeline_name}] scoring...")

    # Score
    score_cmd = [
        sys.executable,
        script,
        "score",
        "--responses-path",
        responses_path,
    ]
    score_result = subprocess.run(score_cmd, capture_output=True, text=True)
    log_lines.append(score_result.stderr)
    log_lines.append(score_result.stdout)

    if score_result.returncode != 0:
        log_lines.append(f"  [{pipeline_name}] FAILED (score): {model}")
        return {
            "pipeline": pipeline_name,
            "status": "score_failed",
            "error": score_result.stderr,
        }

    return {
        "pipeline": pipeline_name,
        "status": "ok",
        "responses_path": responses_path,
        "summary": score_result.stdout,
    }


def run_model(model: str, limit: int, pipelines: list[str], output_dir: Path) -> dict:
    """Run generate + score for all requested pipelines on a single model.

    All output is captured and returned (not printed) so parallel runs
    don't interleave on the terminal.
    """
    log_lines: list[str] = []

    pipeline_results = {}
    for pipeline_name in pipelines:
        result = _run_pipeline(model, limit, pipeline_name, output_dir, log_lines)
        pipeline_results[pipeline_name] = result

    all_ok = all(r["status"] == "ok" for r in pipeline_results.values())

    return {
        "model": model,
        "status": "ok" if all_ok else "partial_failure",
        "pipelines": pipeline_results,
        "log": "\n".join(log_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Run eval across models")
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
    parser.add_argument(
        "--dataset",
        choices=["yes_no", "chained", "all"],
        default="yes_no",
        help="Which pipeline(s) to run (default: yes_no)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        pipelines = list(PIPELINES.keys())
    else:
        pipelines = [args.dataset]

    run_dir = _make_run_dir()
    start = datetime.now()
    results: list[dict] = []

    print(
        f"Running {len(args.models)} models with up to {args.workers} workers"
        f"  |  pipelines: {pipelines}"
    )
    print(f"Output directory: {run_dir}\n")

    pbar = tqdm(total=len(args.models), desc="Models", unit="model")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_model, model, args.limit, pipelines, run_dir): model
            for model in args.models
        }

        for future in as_completed(futures):
            model = futures[future]
            result = future.result()
            results.append(result)
            status = (
                "OK" if result["status"] == "ok" else f"FAILED ({result['status']})"
            )
            pbar.set_postfix_str(f"{model} -> {status}")
            pbar.update(1)

    pbar.close()

    # Sort results to match the original model order
    model_order = {m: i for i, m in enumerate(args.models)}
    results.sort(key=lambda r: model_order.get(r["model"], 0))

    elapsed = datetime.now() - start

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  ALL MODELS COMPLETE  |  {elapsed.total_seconds():.0f}s elapsed")
    print(f"  Output: {run_dir}")
    print(f"{'=' * 70}\n")

    for r in results:
        status = "OK" if r["status"] == "ok" else f"FAILED ({r['status']})"
        print(f"  {r['model']:<45} {status}")
        for pipeline_name in pipelines:
            pr = r["pipelines"].get(pipeline_name, {})
            if pr.get("status") == "ok" and pr.get("summary"):
                marker = PIPELINES[pipeline_name]["summary_marker"]
                for line in pr["summary"].splitlines():
                    if marker in line:
                        print(f"    {line.strip()}")
                        break
            elif pr.get("status") and pr["status"] != "ok":
                print(f"    [{pipeline_name}] {pr['status']}")

    print()


if __name__ == "__main__":
    main()
