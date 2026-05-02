"""Shared utilities and model configuration for Level 4 agents."""

import json
import os
import re
from datetime import date
from typing import Generator

# ---------------------------------------------------------------------------
# Model configuration (override via environment variables)
# ---------------------------------------------------------------------------
ENGAGEMENT_MANAGER_MODEL = os.getenv("ENGAGEMENT_MANAGER_MODEL", "qwen3.5:9b")
MARKET_RESEARCHER_MODEL = os.getenv("MARKET_RESEARCHER_MODEL", "qwen3.5:35b-a3b")
FINANCIAL_ANALYST_MODEL = os.getenv("FINANCIAL_ANALYST_MODEL", "gpt-oss:20b")
RISK_ANALYST_MODEL = os.getenv("RISK_ANALYST_MODEL", "qwen3.5:9b")
STRATEGY_CONSULTANT_MODEL = os.getenv("STRATEGY_CONSULTANT_MODEL", "gemma4:31b")

AGENT_MODELS = {
    "engagement_manager": ENGAGEMENT_MANAGER_MODEL,
    "market_researcher": MARKET_RESEARCHER_MODEL,
    "financial_analyst": FINANCIAL_ANALYST_MODEL,
    "risk_analyst": RISK_ANALYST_MODEL,
    "strategy_consultant": STRATEGY_CONSULTANT_MODEL,
}

# Ollama's default num_ctx is 4096 tokens, which silently truncates large
# research contexts and prior-stage JSON payloads. Every chat call below
# overrides it so the model actually sees the full input.
NUM_CTX_SMALL = 8192    # small inputs (EM decomposition)
NUM_CTX_LARGE = 32768   # research loops, synthesis, reviews, downstream agents


# ---------------------------------------------------------------------------
# Utility -- robust JSON extraction from LLM output
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling think-tags, code fences, and
    common LLM quirks (trailing commas, single quotes, etc.)."""
    # Strip <think>...</think> blocks (Qwen 3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code fences
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)  # use the extracted block for repair below

    # Try finding the outermost { ... }
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt common repairs
            repaired = _repair_json(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")


def _repair_json(text: str) -> str:
    """Fix common LLM JSON mistakes: trailing commas, unescaped newlines."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Remove control characters except \n \r \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _today() -> str:
    """Return today's date formatted for prompt injection."""
    return date.today().strftime("%B %d, %Y")


def _extract_urls(text: str) -> list[str]:
    """Extract all unique URLs from tool result text, preserving order."""
    urls = re.findall(r'https?://[^\s"\'<>\]\)},]+', text)
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        # Strip trailing punctuation that might have been captured
        u = u.rstrip(".,;:")
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# ---------------------------------------------------------------------------
# Streaming think-tag filter
# ---------------------------------------------------------------------------
def _filter_think_stream(stream) -> Generator[str, None, None]:
    """Filter <think>...</think> blocks from an Ollama streaming response."""
    in_think = False
    for chunk in stream:
        content = chunk["message"]["content"]
        if not content:
            continue

        if in_think:
            if "</think>" in content:
                _, _, after = content.partition("</think>")
                in_think = False
                if after:
                    yield after
        else:
            if "<think>" in content:
                before, _, remainder = content.partition("<think>")
                if before:
                    yield before
                if "</think>" in remainder:
                    _, _, after = remainder.partition("</think>")
                    if after:
                        yield after
                else:
                    in_think = True
            else:
                yield content
