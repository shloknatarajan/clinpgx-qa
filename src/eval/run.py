"""
Eval pipeline runner — generate responses then score them.

All outputs are saved to a timestamped directory under runs/.

Usage:
    # Run both yes/no and chained pipelines
    python src/eval/run.py --model gpt-4o-mini --limit 100

    # Run only one dataset
    python src/eval/run.py --model gpt-4o-mini --limit 100 --dataset yes_no
    python src/eval/run.py --model gpt-4o-mini --limit 50 --dataset chained

    # Score-only (re-evaluate existing responses without calling the LLM)
    python src/eval/run.py --score-only --dataset yes_no \
        --responses-path runs/20250210_120000_gpt-4o-mini/yes_no_responses.jsonl
"""

import argparse
import os
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from loguru import logger

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.yes_no import generate as yn_generate, score as yn_score
from src.eval.chained import generate as ch_generate, score as ch_score

DEFAULT_MODEL = "gpt-4o-mini"


def _make_run_dir(model: str) -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    run_dir = Path("runs") / f"{ts}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval pipeline runner")
    parser.add_argument(
        "--model",
        default=os.environ.get("EVAL_MODEL", DEFAULT_MODEL),
        help=f"LiteLLM model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of items per dataset (0 = all)",
    )
    parser.add_argument(
        "--dataset",
        choices=["yes_no", "chained", "all"],
        default="all",
        help="Which dataset to evaluate (default: all)",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Skip generation, only score existing responses",
    )
    parser.add_argument(
        "--responses-path",
        help="Path to existing responses JSONL (used with --score-only)",
    )
    args = parser.parse_args()

    run_yes_no = args.dataset in ("yes_no", "all")
    run_chained = args.dataset in ("chained", "all")

    if args.score_only:
        if not args.responses_path:
            parser.error("--responses-path is required with --score-only")
        score_args = Namespace(responses_path=args.responses_path)
        if run_yes_no:
            logger.info("Scoring yes/no responses...")
            yn_score(score_args)
        if run_chained:
            logger.info("Scoring chained responses...")
            ch_score(score_args)
        return

    # Create a single run directory for this pipeline invocation
    run_dir = _make_run_dir(args.model)
    logger.info(f"Run directory: {run_dir}")

    # --- Yes/No pipeline ---
    if run_yes_no:
        logger.info("=" * 40 + " YES/NO " + "=" * 40)
        gen_args = Namespace(
            model=args.model,
            questions_path="data/yes_no_questions.jsonl",
            limit=args.limit,
        )
        responses_path = yn_generate(gen_args, output_dir=run_dir)
        yn_score(Namespace(responses_path=str(responses_path)), output_dir=run_dir)

    # --- Chained pipeline ---
    if run_chained:
        logger.info("=" * 40 + " CHAINED " + "=" * 39)
        gen_args = Namespace(
            model=args.model,
            questions_path="data/neurips_chained_questions.jsonl",
            limit=args.limit,
        )
        responses_path = ch_generate(gen_args, output_dir=run_dir)
        ch_score(Namespace(responses_path=str(responses_path)), output_dir=run_dir)

    logger.info(f"All outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
