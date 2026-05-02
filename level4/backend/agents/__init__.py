"""
Level 4 — Hybrid Organizational Workflow (BIS).

Five specialized agents operating in a hierarchical hybrid structure:

  1. Engagement Manager  — decomposes AND reviews every intermediate output
  2. Market Researcher   — investigates the market landscape
  3. Financial Analyst   — handles all quantitative/financial analysis
  4. Risk Analyst        — identifies and assesses risks
  5. Strategy Consultant — synthesizes all inputs into a consulting report

Note: The Evaluator is a separate application (not part of this pipeline).

Key differences from Level 3:
  - The EM acts as a Managing Partner: reviews intermediate outputs and can
    send work back with specific revision feedback (max 1 revision per agent).
  - Qwen 3.5 agents use /think or /no_think mode toggles; DeepSeek-R1 emits
    native <think> blocks that are stripped by extract_json(). Gemma 4 has no
    thinking-mode toggle — any trailing /no_think tokens are harmless.
  - Stricter anti-hallucination prompts with mandatory source tracing.
  - Financial Analyst uses GPT-OSS 20B for strong quantitative reasoning.
  - Strategy Consultant uses Gemma 4 31B for top-tier synthesis and writing.

Models:
  Engagement Manager  -> qwen3.5:9b        (fast decomposition + review, fits VRAM)
  Market Researcher   -> qwen3.5:35b-a3b   (35B MoE / 3B active — broad + fast)
  Financial Analyst   -> gpt-oss:20b       (/think for math, /no_think for extraction)
  Risk Analyst        -> qwen3.5:9b        (/think for analytical reasoning)
  Strategy Consultant -> gemma4:31b        (superior writing, GPU+RAM split)

This package was split from a single agents.py file — each agent lives in its
own module for easier debugging. The public API is re-exported here so that
`import agents` / `from agents import ...` continues to work unchanged.
"""

from .common import (
    AGENT_MODELS,
    ENGAGEMENT_MANAGER_MODEL,
    FINANCIAL_ANALYST_MODEL,
    MARKET_RESEARCHER_MODEL,
    NUM_CTX_LARGE,
    NUM_CTX_SMALL,
    RISK_ANALYST_MODEL,
    STRATEGY_CONSULTANT_MODEL,
    extract_json,
    strip_think_tags,
)
from .engagement_manager import EngagementManager
from .financial_analyst import FinancialAnalyst
from .market_researcher import MarketResearcher
from .risk_analyst import RiskAnalyst
from .strategy_consultant import StrategyConsultant
from .tools import assess_risk, read_document, search_web

__all__ = [
    "AGENT_MODELS",
    "ENGAGEMENT_MANAGER_MODEL",
    "MARKET_RESEARCHER_MODEL",
    "FINANCIAL_ANALYST_MODEL",
    "RISK_ANALYST_MODEL",
    "STRATEGY_CONSULTANT_MODEL",
    "NUM_CTX_SMALL",
    "NUM_CTX_LARGE",
    "EngagementManager",
    "MarketResearcher",
    "FinancialAnalyst",
    "RiskAnalyst",
    "StrategyConsultant",
    "extract_json",
    "strip_think_tags",
    "search_web",
    "read_document",
    "assess_risk",
]
