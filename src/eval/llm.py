"""Reusable LLM calling utilities and paper-loading helpers."""

import re
from pathlib import Path

import litellm
from loguru import logger


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0,
    max_tokens: int = 256,
) -> str:
    """Send messages to any model via litellm and return stripped response text."""
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def build_paper_index(papers_dir: str = "data/papers") -> dict[str, str]:
    """Scan papers/*.md, extract PMID from metadata, return {pmid: file_path}."""
    index: dict[str, str] = {}
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        logger.warning(f"Papers directory not found: {papers_dir}")
        return index

    for md_file in papers_path.glob("*.md"):
        text = md_file.read_text()
        match = re.search(r"\*\*PMID:\*\*\s*(\d+)", text)
        if match:
            index[match.group(1)] = str(md_file)
    logger.info(f"Built paper index: {len(index)} papers")
    return index


def load_paper(pmid: str, index: dict[str, str]) -> str | None:
    """Load paper markdown text for a given PMID using the pre-built index."""
    path = index.get(pmid)
    if path is None:
        return None
    return Path(path).read_text()
