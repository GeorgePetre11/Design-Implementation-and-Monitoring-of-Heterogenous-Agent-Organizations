"""
Level 2 — Four Agents (Core Roles).

Four specialized agents collaborate in a sequential pipeline:
  1. Engagement Manager — decomposes the business question into workstreams
  2. Market Researcher  — investigates the market landscape
  3. Strategy Consultant — synthesizes findings into a consulting report
  4. Evaluator           — independently scores the final report

Each agent has:
  - A dedicated system prompt constraining its role  (soft constraint)
  - A specific LLM model — heterogeneous              (design choice)
  - A defined output schema — JSON or Markdown         (hard constraint)
  - Tool/data restrictions enforced by the orchestrator (hard constraint)

Models (from compass artifact — Qwen 3 family via Ollama):
  Engagement Manager  → qwen3:8b   (fast structured decomposition)
  Market Researcher   → qwen3:14b  (broad knowledge, synthesis)
  Strategy Consultant → qwen3:32b  (superior writing, reasoning)
  Evaluator           → qwen3:14b  (critical assessment)
"""

import json
import os
import re
from typing import Generator

import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Model configuration (override via environment variables)
# ---------------------------------------------------------------------------
ENGAGEMENT_MANAGER_MODEL = os.getenv("ENGAGEMENT_MANAGER_MODEL", "qwen3:8b")
MARKET_RESEARCHER_MODEL = os.getenv("MARKET_RESEARCHER_MODEL", "qwen3:14b")
STRATEGY_CONSULTANT_MODEL = os.getenv("STRATEGY_CONSULTANT_MODEL", "qwen3:32b")
AGENT_MODELS = {
    "engagement_manager": ENGAGEMENT_MANAGER_MODEL,
    "market_researcher": MARKET_RESEARCHER_MODEL,
    "strategy_consultant": STRATEGY_CONSULTANT_MODEL,
}


# ---------------------------------------------------------------------------
# Utility — robust JSON extraction from LLM output
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling think-tags, code fences, and
    common LLM quirks (trailing commas, single quotes, etc.)."""
    # Strip <think>…</think> blocks (Qwen 3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json … ``` code fences
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)  # use the extracted block for repair below

    # Try finding the outermost { … }
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
    # Replace single quotes used as string delimiters with double quotes
    # (only outside of already double-quoted strings — best-effort)
    # Remove control characters except \n \r \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Market Researcher tools — search_web and read_document
# ---------------------------------------------------------------------------

_MR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for information about a topic. "
                "Returns titles, URLs, and text snippets for the top results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to run.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Fetch and read the text content of a web page by URL. "
                "Use this after search_web to read the full content of a relevant result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
_DOC_CHAR_LIMIT = 3000  # truncate fetched pages to this many characters


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a DuckDuckGo text search and return structured results."""
    max_results = min(int(max_results), 10)
    print(f"[tool:search_web] query={query!r} max_results={max_results}", flush=True)
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as exc:
        results.append({"error": str(exc)})
    print(f"[tool:search_web] returned {len(results)} results", flush=True)
    return results


def read_document(url: str) -> str:
    """Fetch a URL and return cleaned plain text, truncated to _DOC_CHAR_LIMIT chars."""
    print(f"[tool:read_document] url={url}", flush=True)
    try:
        resp = requests.get(url, timeout=10, headers=_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        print(f"[tool:read_document] fetched {len(text)} chars (truncated to {_DOC_CHAR_LIMIT})", flush=True)
        return text[:_DOC_CHAR_LIMIT]
    except Exception as exc:
        print(f"[tool:read_document] error: {exc}", flush=True)
        return f"[Error fetching {url}: {exc}]"


def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call by name and return its result as a JSON string."""
    if name == "search_web":
        result = search_web(**args)
    elif name == "read_document":
        result = read_document(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
ENGAGEMENT_MANAGER_PROMPT = """\
You are the Engagement Manager at an AI consulting firm. Your sole \
responsibility is to break down client business questions into a clear, \
actionable analysis plan.

RESPONSIBILITIES:
- Understand the client's core business question
- Decompose it into 3–5 focused workstreams
- For each workstream, define specific sub-questions that need investigation
- Create a structured analysis plan that guides the research team

RESTRICTIONS — You must obey these absolutely:
- Do NOT perform any market research, financial analysis, or risk assessment
- Do NOT write any part of the final consulting report
- Do NOT make strategic recommendations
- ONLY produce the analysis plan — nothing more

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "analysis_plan": {
    "business_question_summary": "A clear restatement of what the client is asking",
    "workstreams": [
      {
        "id": 1,
        "title": "Workstream title",
        "description": "What this workstream investigates",
        "key_questions": ["Specific question 1", "Specific question 2"]
      }
    ]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

MARKET_RESEARCHER_RESEARCH_PROMPT = """\
You are a Market Researcher at an AI consulting firm. Your task right now is \
to gather information using your research tools.

You have two tools available:
- search_web(query, max_results): search the web for relevant information
- read_document(url): fetch and read the full text of a web page

INSTRUCTIONS:
- Use search_web to find sources on market size, competitors, trends, and \
customer segments relevant to the business question
- Use read_document on the most promising URLs to get deeper content
- Make multiple searches with different queries to cover all workstreams
- Stop when you have gathered enough information to produce a thorough analysis

RESTRICTIONS:
- Do NOT produce financial analysis, ROI estimates, or cost projections
- Do NOT assess risks
- Only gather market intelligence\
"""

MARKET_RESEARCHER_SYNTHESIS_PROMPT = """\
You are a Market Researcher at an AI consulting firm. You have already \
conducted web research and the findings are provided below. Now synthesize \
that research into structured findings.

RESTRICTIONS — You must obey these absolutely:
- Do NOT perform financial analysis (costs, ROI, projections)
- Do NOT assess risks
- Do NOT write the final consulting report or strategic recommendations
- Base your output ONLY on the research findings provided — do not invent data

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "market_analysis": {
    "market_overview": "High-level overview of the relevant market",
    "market_size_and_growth": "Current size estimates and growth projections",
    "key_competitors": [
      {
        "name": "Competitor name",
        "description": "What they do and how they compete",
        "market_position": "Their standing (leader/challenger/niche)"
      }
    ],
    "market_trends": ["Trend 1", "Trend 2"],
    "customer_segments": [
      {
        "segment": "Segment name",
        "description": "Characteristics and needs",
        "size_estimate": "Relative or absolute size estimate"
      }
    ],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

STRATEGY_CONSULTANT_PROMPT = """\
You are a senior Strategy Consultant at an AI consulting firm. You synthesize \
research findings into a final consulting recommendation.

RESPONSIBILITIES:
- Analyze the market research findings you receive
- Develop 2–3 distinct strategic options with clear pros, cons, and tradeoffs
- Recommend one option with thorough justification
- Write a complete, professional consulting report

RESTRICTIONS — You must obey these absolutely:
- Do NOT search for new information — work only with what you receive
- Do NOT modify or contradict the research findings
- Base your analysis entirely on the provided market research and analysis plan

Write a professional consulting report in Markdown format with these sections:

# Executive Summary
Brief overview of the situation and your top-line recommendation.

## Situation Analysis
What the client is facing, based on the market research.

## Strategic Options
Present 2–3 options, each with:
- Description of the approach
- Pros and advantages
- Cons and risks
- Estimated effort/complexity

## Recommendation
Your recommended option with detailed justification.

## Implementation Roadmap
Phased action plan: short-term (0–3 months), mid-term (3–12 months), \
long-term (12+ months).

Write in a professional, concise consulting style. Use data from the market \
research to support your points. Be specific and actionable.\
"""



# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------
class EngagementManager:
    """Decomposes the client question into workstreams and an analysis plan."""

    name = "engagement_manager"
    display_name = "Engagement Manager"
    model = ENGAGEMENT_MANAGER_MODEL

    def run(self, question: str) -> dict:
        """Return the analysis plan as a parsed dict."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": ENGAGEMENT_MANAGER_PROMPT},
                {"role": "user", "content": f"Client question:\n{question}"},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class MarketResearcher:
    """Investigates the market landscape using web search tools, then produces
    structured findings via a two-phase approach:
      Phase 1 — Tool loop: LLM calls search_web / read_document to gather data
      Phase 2 — Synthesis: LLM converts collected research into the JSON schema
    """

    name = "market_researcher"
    display_name = "Market Researcher"
    model = MARKET_RESEARCHER_MODEL
    MAX_TOOL_ROUNDS = 8  # cap to prevent runaway loops

    def run(self, question: str, analysis_plan: dict) -> dict:
        """Return the market analysis as a parsed dict.

        Receives only the question and the analysis plan (tool restriction:
        cannot see any other agent's output).
        """
        research_context = self._research_phase(question, analysis_plan)
        return self._synthesis_phase(question, analysis_plan, research_context)

    # ------------------------------------------------------------------
    # Phase 1: tool-calling research loop
    # ------------------------------------------------------------------
    def _research_phase(self, question: str, analysis_plan: dict) -> str:
        """Run the tool loop and return a text summary of all gathered findings."""
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            "Use search_web and read_document to gather the market information "
            "needed to address every workstream in the analysis plan. "
            "When you have enough data, stop calling tools."
        )
        messages = [
            {"role": "system", "content": MARKET_RESEARCHER_RESEARCH_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        tool_results: list[str] = []

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=_MR_TOOLS,
            )
            msg = response["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # LLM finished calling tools
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                result_str = _dispatch_tool(name, args)
                tool_results.append(f"[{name}({args})]\n{result_str}")
                messages.append({"role": "tool", "content": result_str})

        return "\n\n---\n\n".join(tool_results) if tool_results else "(no tool results)"

    # ------------------------------------------------------------------
    # Phase 2: structured synthesis
    # ------------------------------------------------------------------
    def _synthesis_phase(
        self, question: str, analysis_plan: dict, research_context: str
    ) -> dict:
        """Convert the raw research findings into the required JSON schema."""
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"RESEARCH FINDINGS:\n{research_context}\n\n"
            "Synthesize the research findings above into a structured market "
            "analysis that addresses every workstream in the analysis plan."
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": MARKET_RESEARCHER_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class StrategyConsultant:
    """Synthesizes findings into a final consulting report (Markdown, streamed)."""

    name = "strategy_consultant"
    display_name = "Strategy Consultant"
    model = STRATEGY_CONSULTANT_MODEL

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
    ) -> Generator[str, None, None]:
        """Stream the consulting report token-by-token.

        Receives the question, analysis plan, and market research
        (tool restriction: cannot search for new data).
        """
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            "Using the analysis plan and market research above, write a "
            "complete consulting report with your strategic recommendation."
        )
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": STRATEGY_CONSULTANT_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        yield from _filter_think_stream(stream)



# ---------------------------------------------------------------------------
# Streaming think-tag filter
# ---------------------------------------------------------------------------
def _filter_think_stream(stream) -> Generator[str, None, None]:
    """Filter <think>…</think> blocks from an Ollama streaming response."""
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
