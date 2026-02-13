"""
Run evaluation across multiple models.

All outputs are saved to a single timestamped directory under runs/.

Usage:
    python run_all_models.py --limit 100                                # yes/no only (default)
    python run_all_models.py --limit 100 --dataset chained              # chained only
    python run_all_models.py --limit 100 --dataset all                  # all pipelines
    python run_all_models.py --dataset variant_extraction               # variant extraction
    python run_all_models.py --dataset paper_investigation              # paper investigation
    python run_all_models.py --models gpt-4o anthropic/claude-opus-4-6
    python run_all_models.py --workers 3                                # run 3 models in parallel
"""

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o-mini",
    # "gpt-5",
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
    "study_param": {
        "script": "src/eval/study_param.py",
        "summary_marker": "Study Param Results",
    },
    "mcq_variant": {
        "script": "src/eval/mcq_variant.py",
        "summary_marker": "Mcq Variant Results",
    },
    "mcq_drug": {
        "script": "src/eval/mcq_drug.py",
        "summary_marker": "Mcq Drug Results",
    },
    "mcq_phenotype": {
        "script": "src/eval/mcq_phenotype.py",
        "summary_marker": "Mcq Phenotype Results",
    },
    "variant_extraction": {
        "script": "src/modules/variant_extraction/variant_extraction.py",
        "summary_marker": "Variant Extraction Results",
    },
    "paper_investigation": {
        "script": "src/modules/paper_investigation/paper_investigation.py",
        "summary_marker": "Paper Investigation Results",
    },
}


def _make_run_dir(dataset: str) -> Path:
    """Create a single timestamped run directory for all models."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "all_models"
    if dataset == "mc_study":
        suffix = "mc_study_all_models"
    run_dir = Path("runs") / f"{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _short_model(model: str) -> str:
    """Return a short display name for a model identifier."""
    return model.rsplit("/", 1)[-1]


def _stream_stderr(
    proc: subprocess.Popen,
    captured: list[str],
    pbar: tqdm | None,
    model: str,
    pipeline_name: str,
) -> None:
    """Read stderr from a subprocess line-by-line, logging progress in real time."""
    short = _short_model(model)
    for raw_line in proc.stderr:
        line = raw_line.rstrip("\n")
        captured.append(line)
        # Show question-level progress lines (e.g. "  [25/100]")
        if pbar and ("[" in line and "/" in line and "]" in line):
            pbar.write(f"  {short} | {pipeline_name}: {line.strip()}")


def _run_pipeline(
    model: str,
    limit: int,
    pipeline_name: str,
    output_dir: Path,
    log_lines: list[str],
    pbar: tqdm | None = None,
) -> dict:
    """Run generate + score for a single pipeline. Returns result dict."""
    cfg = PIPELINES[pipeline_name]
    script = cfg["script"]
    short = _short_model(model)

    log_lines.append(f"  [{pipeline_name}] generating...")
    if pbar:
        pbar.write(f"  {short} | {pipeline_name}: generating...")

    # Generate — stream stderr so we see per-question progress
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
    gen_proc = subprocess.Popen(
        gen_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    gen_stderr: list[str] = []
    stderr_thread = threading.Thread(
        target=_stream_stderr,
        args=(gen_proc, gen_stderr, pbar, model, pipeline_name),
    )
    stderr_thread.start()
    gen_proc.stdout.read()
    gen_proc.wait()
    stderr_thread.join()
    log_lines.extend(gen_stderr)

    if gen_proc.returncode != 0:
        log_lines.append(f"  [{pipeline_name}] FAILED (generate): {model}")
        if pbar:
            pbar.write(f"  {short} | {pipeline_name}: FAILED (generate)")
        return {
            "pipeline": pipeline_name,
            "status": "generate_failed",
            "error": "\n".join(gen_stderr[-20:]),
        }

    # Find the responses path from this pipeline's generate stderr
    responses_path = None
    for line in gen_stderr:
        if "Responses saved to" in line:
            responses_path = line.split("Responses saved to")[-1].strip()

    if not responses_path:
        return {
            "pipeline": pipeline_name,
            "status": "no_responses_path",
            "error": "\n".join(gen_stderr[-20:]),
        }

    log_lines.append(f"  [{pipeline_name}] scoring...")
    if pbar:
        pbar.write(f"  {short} | {pipeline_name}: scoring...")

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
        if pbar:
            pbar.write(f"  {short} | {pipeline_name}: FAILED (score)")
        return {
            "pipeline": pipeline_name,
            "status": "score_failed",
            "error": score_result.stderr,
        }

    if pbar:
        pbar.write(f"  {short} | {pipeline_name}: done")

    return {
        "pipeline": pipeline_name,
        "status": "ok",
        "responses_path": responses_path,
        "summary": score_result.stdout,
    }


def run_model(
    model: str,
    limit: int,
    pipelines: list[str],
    output_dir: Path,
    pbar: tqdm | None = None,
) -> dict:
    """Run generate + score for all requested pipelines on a single model.

    Progress is streamed via the shared tqdm bar so the user sees
    per-pipeline and per-question updates in real time.
    """
    log_lines: list[str] = []

    pipeline_results = {}
    for pipeline_name in pipelines:
        result = _run_pipeline(
            model,
            limit,
            pipeline_name,
            output_dir,
            log_lines,
            pbar=pbar,
        )
        pipeline_results[pipeline_name] = result
        if pbar:
            pbar.update(1)

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
        choices=[
            "yes_no",
            "chained",
            "study_param",
            "mcq_variant",
            "mcq_drug",
            "mcq_phenotype",
            "variant_extraction",
            "paper_investigation",
            "mc_study",
            "all",
        ],
        default="yes_no",
        help="Which pipeline(s) to run (default: yes_no)",
    )
    args = parser.parse_args()

    MC_STUDY_PIPELINES = ["study_param", "mcq_variant", "mcq_drug", "mcq_phenotype"]
    if args.dataset == "all":
        pipelines = list(PIPELINES.keys())
    elif args.dataset == "mc_study":
        pipelines = MC_STUDY_PIPELINES
    else:
        pipelines = [args.dataset]

    run_dir = _make_run_dir(args.dataset)
    start = datetime.now()
    results: list[dict] = []

    total_pipelines = len(args.models) * len(pipelines)
    print(
        f"Running {len(args.models)} models x {len(pipelines)} pipelines "
        f"({total_pipelines} total) with up to {args.workers} workers"
    )
    print(f"Pipelines: {pipelines}")
    print(f"Output directory: {run_dir}\n")

    pbar = tqdm(total=total_pipelines, desc="Pipelines", unit="pipeline")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_model,
                model,
                args.limit,
                pipelines,
                run_dir,
                pbar,
            ): model
            for model in args.models
        }

        for future in as_completed(futures):
            model = futures[future]
            result = future.result()
            results.append(result)
            status = (
                "OK" if result["status"] == "ok" else f"FAILED ({result['status']})"
            )
            pbar.write(f"  {_short_model(model)}: all pipelines {status}")

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
