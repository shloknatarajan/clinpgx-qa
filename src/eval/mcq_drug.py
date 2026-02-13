"""
Drug MCQ evaluation wrapper.

Usage:
    python src/eval/mcq_drug.py generate --model gpt-4o-mini --limit 100
    python src/eval/mcq_drug.py score --responses-path runs/.../mcq_drug_responses.jsonl
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.eval.mcq import build_cli

if __name__ == "__main__":
    build_cli(
        pipeline_name="mcq_drug",
        mcq_type="drug",
        default_questions_path="data/mcq_options/drug_mcq_options.jsonl",
        description="Drug MCQ evaluation",
    )
