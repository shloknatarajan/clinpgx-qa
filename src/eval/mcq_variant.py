"""
Variant MCQ evaluation wrapper.

Usage:
    python src/eval/mcq_variant.py generate --model gpt-4o-mini --limit 100
    python src/eval/mcq_variant.py score --responses-path runs/.../mcq_variant_responses.jsonl
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.mcq import build_cli

if __name__ == "__main__":
    build_cli(
        pipeline_name="mcq_variant",
        mcq_type="variant",
        default_questions_path="data/mcq_options/variant_mcq_options.jsonl",
        description="Variant MCQ evaluation",
    )
