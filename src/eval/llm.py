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


_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5-", "gpt-5.")
_REASONING_EXACT = {"gpt-5"}
_REASONING_GEMINI_PREFIXES = ("gemini-2.5", "gemini-3")

# Timeout in seconds for a single LLM call.
_LLM_TIMEOUT = 300


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model that needs higher token limits."""
    # Strip provider prefix (e.g. "openai/gpt-5" → "gpt-5")
    bare = model.lower().split("/", 1)[-1]
    if bare in _REASONING_EXACT:
        return True
    if any(bare.startswith(p) for p in _REASONING_PREFIXES):
        return True
    # Gemini thinking models (e.g. gemini/gemini-2.5-pro)
    gemini_bare = model.lower().removeprefix("gemini/")
    return any(gemini_bare.startswith(p) for p in _REASONING_GEMINI_PREFIXES)


@retry(
    retry=retry_if_exception_type(
        (litellm.RateLimitError, litellm.exceptions.APIConnectionError, litellm.Timeout)
    ),
    wait=wait_exponential(min=2, max=120),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 256,
) -> str:
    """Send messages to any model via litellm and return stripped response text."""
    # Reasoning models require max_completion_tokens (not max_tokens) and
    # only support temperature=1, so we drop temperature for them.
    is_reasoning = _is_reasoning_model(model)
    kwargs: dict = {"drop_params": True, "temperature": temperature}
    if is_reasoning:
        # Scale up for chain-of-thought overhead, floor at 4096, cap at 16k.
        kwargs["max_completion_tokens"] = min(max(max_tokens * 8, 4096), 16_000)
    else:
        kwargs["max_tokens"] = max_tokens
    response = litellm.completion(
        model=model,
        messages=messages,
        timeout=_LLM_TIMEOUT,
        **kwargs,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


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
