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
from src.eval.study_param import generate as sp_generate, score as sp_score
from src.eval.mcq import generate as mcq_generate, score as mcq_score

DEFAULT_MODEL = "gpt-4o-mini"


def _make_run_dir(model: str, dataset: str) -> Path:
    """Create a timestamped run directory under runs/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "_")
    suffix = model_slug
    if dataset == "mc_study":
        suffix = f"{model_slug}_mc_study"
    run_dir = Path("runs") / f"{ts}_{suffix}"
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
        "--limit",
        type=int,
        default=0,
        help="Limit number of items per dataset (0 = all)",
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
            "mc_study",
            "all",
        ],
        default="all",
        help="Which dataset to evaluate (default: all)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip generation, only score existing responses",
    )
    parser.add_argument(
        "--responses-path",
        help="Path to existing responses JSONL (used with --score-only)",
    )
    args = parser.parse_args()

    _mc_study = ("mc_study", "all")
    run_yes_no = args.dataset in ("yes_no", "all")
    run_chained = args.dataset in ("chained", "all")
    run_study_param = args.dataset in ("study_param", *_mc_study)
    run_mcq_variant = args.dataset in ("mcq_variant", *_mc_study)
    run_mcq_drug = args.dataset in ("mcq_drug", *_mc_study)
    run_mcq_phenotype = args.dataset in ("mcq_phenotype", *_mc_study)

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
        if run_study_param:
            logger.info("Scoring study_param responses...")
            sp_score(score_args)
        if run_mcq_variant:
            logger.info("Scoring mcq_variant responses...")
            mcq_score(score_args, pipeline_name="mcq_variant")
        if run_mcq_drug:
            logger.info("Scoring mcq_drug responses...")
            mcq_score(score_args, pipeline_name="mcq_drug")
        if run_mcq_phenotype:
            logger.info("Scoring mcq_phenotype responses...")
            mcq_score(score_args, pipeline_name="mcq_phenotype")
        return

    # Create a single run directory for this pipeline invocation
    run_dir = _make_run_dir(args.model, args.dataset)
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

    # --- Study param pipeline ---
    if run_study_param:
        logger.info("=" * 40 + " STUDY PARAM " + "=" * 35)
        gen_args = Namespace(
            model=args.model,
            questions_path="data/study_param_questions/study_param_questions.jsonl",
            limit=args.limit,
        )
        responses_path = sp_generate(gen_args, output_dir=run_dir)
        sp_score(Namespace(responses_path=str(responses_path)), output_dir=run_dir)

    # --- MCQ variant pipeline ---
    if run_mcq_variant:
        logger.info("=" * 40 + " MCQ VARIANT " + "=" * 35)
        gen_args = Namespace(
            model=args.model,
            questions_path="data/mcq_options/variant_mcq_options.jsonl",
            limit=args.limit,
        )
        responses_path = mcq_generate(
            gen_args,
            output_dir=run_dir,
            pipeline_name="mcq_variant",
            mcq_type="variant",
        )
        mcq_score(
            Namespace(responses_path=str(responses_path)),
            output_dir=run_dir,
            pipeline_name="mcq_variant",
        )

    # --- MCQ drug pipeline ---
    if run_mcq_drug:
        logger.info("=" * 40 + " MCQ DRUG " + "=" * 39)
        gen_args = Namespace(
            model=args.model,
            questions_path="data/mcq_options/drug_mcq_options.jsonl",
            limit=args.limit,
        )
        responses_path = mcq_generate(
            gen_args, output_dir=run_dir, pipeline_name="mcq_drug", mcq_type="drug"
        )
        mcq_score(
            Namespace(responses_path=str(responses_path)),
            output_dir=run_dir,
            pipeline_name="mcq_drug",
        )

    # --- MCQ phenotype pipeline ---
    if run_mcq_phenotype:
        logger.info("=" * 40 + " MCQ PHENOTYPE " + "=" * 33)
        gen_args = Namespace(
            model=args.model,
            questions_path="data/mcq_options/phenotype_mcq_options.jsonl",
            limit=args.limit,
        )
        responses_path = mcq_generate(
            gen_args,
            output_dir=run_dir,
            pipeline_name="mcq_phenotype",
            mcq_type="phenotype",
        )
        mcq_score(
            Namespace(responses_path=str(responses_path)),
            output_dir=run_dir,
            pipeline_name="mcq_phenotype",
        )

    logger.info(f"All outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
