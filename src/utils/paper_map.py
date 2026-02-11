"""Shared PMID / PMCID mapping utilities.

Paper files are named ``PMC{id}.md`` and contain both identifiers in their
headers.  These helpers build the look-ups needed by data generation and eval
pipelines.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger


def build_pmid_to_pmcid(papers_dir: str | Path) -> dict[str, str]:
    """Scan paper headers, return ``{pmid: pmcid}``."""
    papers_path = Path(papers_dir)
    mapping: dict[str, str] = {}
    if not papers_path.exists():
        logger.warning(f"Papers directory not found: {papers_dir}")
        return mapping

    pmid_pat = re.compile(r"\*\*PMID:\*\*\s*(\d+)")
    pmcid_pat = re.compile(r"\*\*PMCID:\*\*\s*(PMC\d+)")

    for md in sorted(papers_path.glob("*.md")):
        try:
            head = md.read_text(encoding="utf-8")[:2000]
            pmid_m = pmid_pat.search(head)
            pmcid_m = pmcid_pat.search(head)
            if pmid_m and pmcid_m:
                mapping[pmid_m.group(1)] = pmcid_m.group(1)
        except Exception:
            continue

    logger.info(f"Mapped {len(mapping):,} PMIDs to PMCIDs")
    return mapping


def build_pmcid_paper_map(papers_dir: str | Path) -> dict[str, Path]:
    """Return ``{pmcid: filepath}`` derived from filenames (no header parsing)."""
    papers_path = Path(papers_dir)
    mapping: dict[str, Path] = {}
    if not papers_path.exists():
        logger.warning(f"Papers directory not found: {papers_dir}")
        return mapping

    for md in sorted(papers_path.glob("PMC*.md")):
        pmcid = md.stem  # e.g. "PMC10026301"
        mapping[pmcid] = md

    logger.info(f"Built PMCID paper map: {len(mapping):,} papers")
    return mapping
