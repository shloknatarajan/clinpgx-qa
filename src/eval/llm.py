"""Reusable LLM calling utilities and paper-loading helpers."""

from pathlib import Path

import litellm
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.utils.paper_map import build_pmcid_paper_map


_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model that needs higher token limits."""
    name = model.lower()
    return any(
        name.startswith(p) or name.startswith(f"openai/{p}")
        for p in _REASONING_PREFIXES
    )


@retry(
    retry=retry_if_exception_type(
        (litellm.RateLimitError, litellm.exceptions.APIConnectionError)
    ),
    wait=wait_exponential(min=2, max=120),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0,
    max_tokens: int = 256,
) -> str:
    """Send messages to any model via litellm and return stripped response text."""
    # Reasoning models use internal chain-of-thought tokens that count against
    # max_tokens, so we need a much higher limit.
    effective_max = 16_000 if _is_reasoning_model(model) else max_tokens
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=effective_max,
        drop_params=True,
    )
    return response.choices[0].message.content.strip()


def build_paper_index(papers_dir: str = "data/papers") -> dict[str, str]:
    """Return {pmcid: file_path} keyed by PMCID (extracted from filename)."""
    pmcid_map = build_pmcid_paper_map(papers_dir)
    return {pmcid: str(path) for pmcid, path in pmcid_map.items()}


def load_paper(pmcid: str, index: dict[str, str]) -> str | None:
    """Load paper markdown text for a given PMCID using the pre-built index."""
    path = index.get(pmcid)
    if path is None:
        return None
    return Path(path).read_text()
