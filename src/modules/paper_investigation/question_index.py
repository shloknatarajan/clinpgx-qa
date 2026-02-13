"""
Question index — maps (pmcid, variant) pairs to MC questions from all pipelines.

Loads the four question JSONL files (variant MCQ, drug MCQ, phenotype MCQ,
study param) and builds a lookup so we can retrieve every question relevant
to a recalled variant in a given paper.

All four files have a `variant` field directly — no annotation resolution needed.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.modules.variant_extraction.variant_extraction import normalize_variant

DATA_DIR = _project_root / "data"

# Question sources
VARIANT_MCQ_PATH = DATA_DIR / "mcq_options" / "variant_mcq_options.jsonl"
DRUG_MCQ_PATH = DATA_DIR / "mcq_options" / "drug_mcq_options.jsonl"
PHENOTYPE_MCQ_PATH = DATA_DIR / "mcq_options" / "phenotype_mcq_options.jsonl"
STUDY_PARAM_PATH = DATA_DIR / "study_param_questions" / "study_param_questions.jsonl"


class UnifiedQuestion(BaseModel):
    """Normalized representation of a question from any MC pipeline."""

    source_pipeline: str  # "mcq_variant" | "mcq_drug" | "mcq_phenotype" | "study_param"
    pmcid: str
    variant: str  # normalized variant string
    annotation_id: str
    raw_question: dict  # full original question record


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class QuestionIndex:
    """Maps (pmcid, variant) pairs to all MC questions involving that combination."""

    def __init__(self) -> None:
        self._index: dict[tuple[str, str], list[UnifiedQuestion]] = defaultdict(list)
        self._pmcids: set[str] = set()
        self._load_all()

    def _load_all(self) -> None:
        sources: list[tuple[Path, str]] = [
            (VARIANT_MCQ_PATH, "mcq_variant"),
            (DRUG_MCQ_PATH, "mcq_drug"),
            (PHENOTYPE_MCQ_PATH, "mcq_phenotype"),
            (STUDY_PARAM_PATH, "study_param"),
        ]

        total = 0
        for path, pipeline in sources:
            if not path.exists():
                logger.warning(f"Question file not found: {path}")
                continue

            records = _load_jsonl(path)
            loaded = 0
            skipped = 0

            for q in records:
                pmcid = q.get("pmcid", "")
                annotation_id = q.get("annotation_id", "")
                variant_raw = q.get("variant", "")

                if not pmcid or not variant_raw:
                    skipped += 1
                    continue

                variant_norm = normalize_variant(variant_raw)
                uq = UnifiedQuestion(
                    source_pipeline=pipeline,
                    pmcid=pmcid,
                    variant=variant_norm,
                    annotation_id=str(annotation_id),
                    raw_question=q,
                )
                self._index[(pmcid, variant_norm)].append(uq)
                self._pmcids.add(pmcid)
                loaded += 1

            total += loaded
            logger.info(
                f"  {pipeline}: loaded {loaded} questions"
                + (f" (skipped {skipped})" if skipped else "")
            )

        logger.info(
            f"QuestionIndex: {total} questions across {len(self._pmcids)} papers"
        )

    def get_questions(self, pmcid: str, variant: str) -> list[UnifiedQuestion]:
        """Return all questions for a (pmcid, variant) pair."""
        key = (pmcid, normalize_variant(variant))
        return self._index.get(key, [])

    def get_paper_variants_with_questions(self, pmcid: str) -> set[str]:
        """Return all variants that have at least one question for this paper."""
        return {v for (p, v) in self._index if p == pmcid}

    @property
    def all_pmcids(self) -> set[str]:
        """All PMCIDs that have at least one indexed question."""
        return self._pmcids
