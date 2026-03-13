"""
Level 3 — Six Agents (Full Specialization).

Five specialized agents collaborate in a sequential pipeline:
  1. Engagement Manager  — decomposes the business question into workstreams
  2. Market Researcher   — investigates the market landscape
  3. Financial Analyst   — handles all quantitative/financial analysis
  4. Risk Analyst        — identifies and assesses risks
  5. Strategy Consultant — synthesizes all inputs into a consulting report

(Evaluator agent is implemented separately via the Anthropic Claude API.)

Each agent has:
  - A dedicated system prompt constraining its role  (soft constraint)
  - A specific LLM model — heterogeneous              (design choice)
  - A defined output schema — JSON or Markdown         (hard constraint)
  - Tool/data restrictions enforced by the orchestrator (hard constraint)

Models (from Models_Info — mixed Ollama models):
  Engagement Manager  → qwen3:8b          (fast structured decomposition)
  Market Researcher   → qwen3:14b         (broad knowledge, synthesis)
  Financial Analyst   → deepseek-r1:14b   (strong quantitative reasoning)
  Risk Analyst        → qwen3:14b         (edge-case thinking, risk focus)
  Strategy Consultant → qwen3:32b         (superior writing, reasoning)
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
FINANCIAL_ANALYST_MODEL = os.getenv("FINANCIAL_ANALYST_MODEL", "deepseek-r1:14b")
RISK_ANALYST_MODEL = os.getenv("RISK_ANALYST_MODEL", "qwen3:14b")
STRATEGY_CONSULTANT_MODEL = os.getenv("STRATEGY_CONSULTANT_MODEL", "qwen3:32b")

AGENT_MODELS = {
    "engagement_manager": ENGAGEMENT_MANAGER_MODEL,
    "market_researcher": MARKET_RESEARCHER_MODEL,
    "financial_analyst": FINANCIAL_ANALYST_MODEL,
    "risk_analyst": RISK_ANALYST_MODEL,
    "strategy_consultant": STRATEGY_CONSULTANT_MODEL,
}


# ---------------------------------------------------------------------------
# Utility — robust JSON extraction from LLM output
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling think-tags, code fences, and
    common LLM quirks (trailing commas, single quotes, etc.)."""
    # Strip <think>…</think> blocks (Qwen 3 / DeepSeek R1 thinking mode)
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
    # Remove control characters except \n \r \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Shared tools — search_web and read_document (used by MR and RA)
# ---------------------------------------------------------------------------

_SEARCH_TOOLS = [
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


# ---------------------------------------------------------------------------
# Risk Analyst tools — assess_risk (plus search_web / read_document above)
# ---------------------------------------------------------------------------

_RA_TOOLS = _SEARCH_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "assess_risk",
            "description": (
                "Structure a risk assessment for a specific identified risk. "
                "Takes a risk title and description, and returns a formatted "
                "risk entry with probability and impact ratings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title of the risk.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the risk.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["regulatory", "market", "operational", "competitive", "financial"],
                        "description": "Risk category.",
                    },
                    "probability": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Likelihood of the risk materializing.",
                    },
                    "impact": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Severity of impact if the risk materializes.",
                    },
                },
                "required": ["title", "description", "category", "probability", "impact"],
            },
        },
    },
]


def assess_risk(
    title: str,
    description: str,
    category: str,
    probability: str,
    impact: str,
) -> dict:
    """Return a structured risk assessment entry."""
    print(f"[tool:assess_risk] title={title!r} cat={category} prob={probability} impact={impact}", flush=True)
    # Compute a simple risk score: low=1, medium=2, high=3
    score_map = {"low": 1, "medium": 2, "high": 3}
    prob_score = score_map.get(probability, 2)
    impact_score = score_map.get(impact, 2)
    risk_score = prob_score * impact_score
    risk_level = "low" if risk_score <= 2 else "medium" if risk_score <= 4 else "high" if risk_score <= 6 else "critical"
    return {
        "title": title,
        "description": description,
        "category": category,
        "probability": probability,
        "impact": impact,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call by name and return its result as a JSON string."""
    if name == "search_web":
        result = search_web(**args)
    elif name == "read_document":
        result = read_document(**args)
    elif name == "assess_risk":
        result = assess_risk(**args)
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

CRITICAL INSTRUCTIONS — follow this workflow:
1. Start by calling search_web with a broad query about the market
2. IMMEDIATELY call read_document on the 2-3 most relevant URLs from the \
search results — search snippets are NOT enough, you MUST read full pages
3. Then search for specific topics: competitors, market size, trends, etc.
4. For EACH search, call read_document on at least 1-2 of the top URLs
5. Continue until you have covered all workstreams in the analysis plan

You MUST call read_document at least 3-4 times during your research. \
Search snippets alone are too shallow — they contain only 1-2 sentences \
and often have outdated or incomplete data. The full page content from \
read_document is essential for accurate market data.

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
- Every claim MUST be backed by data from the research — cite sources by \
including the URL in parentheses after key facts
- If the research does not contain data for a field, say "Data not available \
from research" — do NOT hallucinate numbers

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "market_analysis": {
    "market_overview": "High-level overview of the relevant market (cite sources)",
    "market_size_and_growth": "Current size estimates and growth projections with source URLs",
    "key_competitors": [
      {
        "name": "Competitor name",
        "description": "What they do and how they compete",
        "market_position": "Their standing (leader/challenger/niche)",
        "source": "URL where this information was found"
      }
    ],
    "market_trends": ["Trend 1 (source URL)", "Trend 2 (source URL)"],
    "customer_segments": [
      {
        "segment": "Segment name",
        "description": "Characteristics and needs",
        "size_estimate": "Relative or absolute size estimate"
      }
    ],
    "key_findings": ["Finding 1 (source)", "Finding 2 (source)", "Finding 3 (source)"],
    "sources": ["URL 1", "URL 2", "URL 3"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

FINANCIAL_ANALYST_PROMPT = """\
You are a Financial Analyst at an AI consulting firm. You perform all \
quantitative and financial analysis for the team.

RESPONSIBILITIES:
- Extract specific numbers from the market research data (market size, growth \
rates, competitor pricing, salary ranges, etc.) and use THOSE as your inputs
- Calculate cost estimates for the proposed business action (setup, operations, \
marketing, staffing, technology)
- Create revenue projections for 3 scenarios (conservative, moderate, aggressive)
- Calculate ROI and break-even timeline
- Perform sensitivity analysis on key assumptions
- Show ALL calculations step by step in your reasoning before outputting JSON

CRITICAL RULES FOR ACCURACY:
- ONLY use numbers that appear in the market research data or can be directly \
derived from it. If the market research says the market is $X billion, use \
that number — do NOT substitute your own estimate.
- For any number you use, state where it came from (e.g., "Based on market \
research: market size is $X")
- If the market research does NOT provide a specific number you need, state \
your assumption explicitly and mark it as "[ASSUMED]"
- Show your math: if you calculate revenue as market_size * capture_rate, \
write out the equation with actual numbers
- Cross-check your numbers: does the total add up? Is break-even consistent \
with revenue and costs?

RESTRICTIONS — You must obey these absolutely:
- Do NOT perform market research — that data is already provided
- Do NOT assess non-financial risks (only flag financial risks)
- Do NOT write the final consulting report or strategic recommendations
- Do NOT invent market data — use only what is provided

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "financial_analysis": {
    "executive_summary": "Brief overview of the financial outlook",
    "data_inputs_used": "List the key numbers extracted from market research",
    "cost_estimates": [
      {
        "category": "Cost category name",
        "amount": "$X,XXX",
        "timeframe": "one-time / monthly / annual",
        "notes": "Key assumptions and source of the number"
      }
    ],
    "revenue_projections": [
      {
        "scenario": "Conservative / Moderate / Aggressive",
        "year_1": "$X,XXX",
        "year_2": "$X,XXX",
        "year_3": "$X,XXX",
        "assumptions": "Key assumptions — reference market research data"
      }
    ],
    "roi_analysis": "Expected ROI with calculation shown",
    "break_even_timeline": "When the investment breaks even, with math shown",
    "sensitivity_analysis": "How results change if key assumptions vary by ±20%",
    "key_financial_risks": ["Financial risk 1", "Financial risk 2"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

RISK_ANALYST_RESEARCH_PROMPT = """\
You are a Risk Analyst at an AI consulting firm. Your task right now is to \
identify and assess risks related to the business question using your tools.

You have three tools available:
- search_web(query, max_results): search the web for risk-related information
- read_document(url): fetch and read the full text of a web page
- assess_risk(title, description, category, probability, impact): structure a \
risk assessment for a specific identified risk

INSTRUCTIONS:
- Use search_web to research regulatory, market, operational, and competitive risks
- Use read_document to get detailed information on specific risk factors
- Use assess_risk for each major risk you identify to create structured entries
- Identify at least 5-8 distinct risks across multiple categories
- Focus on: regulatory risks, market risks, operational risks, competitive \
threats, and financial risks

RESTRICTIONS:
- Do NOT perform market research beyond what's needed for risk identification
- Do NOT do financial analysis
- Do NOT write strategic recommendations
- Only identify and assess risks — do NOT propose full solutions\
"""

RISK_ANALYST_SYNTHESIS_PROMPT = """\
You are a Risk Analyst at an AI consulting firm. You have already researched \
and assessed risks. The findings are provided below. Now synthesize them into \
a structured risk assessment.

RESTRICTIONS — You must obey these absolutely:
- Do NOT perform market research
- Do NOT do financial analysis
- Do NOT write the final consulting report or strategic recommendations
- Base your output ONLY on the risk research provided — do not invent data
- Only identify risks — do NOT propose full solutions, only brief mitigation suggestions

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "risk_assessment": {
    "overall_risk_level": "low / medium / high / critical",
    "risk_summary": "Brief overview of the risk landscape",
    "risks": [
      {
        "id": 1,
        "title": "Risk title",
        "description": "Detailed risk description",
        "category": "regulatory / market / operational / competitive / financial",
        "probability": "low / medium / high",
        "impact": "low / medium / high",
        "mitigation_suggestion": "Brief mitigation idea"
      }
    ],
    "key_risk_factors": ["Key factor 1", "Key factor 2"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

STRATEGY_CONSULTANT_PROMPT = """\
You are a senior Strategy Consultant at an AI consulting firm. You synthesize \
all research inputs — market research, financial analysis, AND risk assessment \
— into a final consulting recommendation.

RESPONSIBILITIES:
- Analyze the market research, financial analysis, and risk assessment findings
- Develop 2–3 distinct strategic options with clear pros, cons, and tradeoffs
- Recommend one option with thorough justification
- Write a complete, professional consulting report

RESTRICTIONS — You must obey these absolutely:
- Do NOT search for new information — work only with what you receive
- Do NOT modify or contradict the research findings
- Base your analysis entirely on the provided market research, financial \
analysis, risk assessment, and analysis plan

Write a professional consulting report in Markdown format with these sections:

# Executive Summary
Brief overview of the situation and your top-line recommendation.

## Situation Analysis
What the client is facing, based on all three research inputs.

## Market Landscape
Key findings from the market research.

## Financial Overview
Key findings from the financial analysis, including projections and ROI.

## Risk Landscape
Key findings from the risk assessment, including the most critical risks.

## Strategic Options
Present 2–3 options, each with:
- Description of the approach
- Pros and advantages
- Cons and risks
- Financial implications
- Risk profile

## Recommendation
Your recommended option with detailed justification that addresses market \
opportunity, financial viability, and risk tolerance.

## Implementation Roadmap
Phased action plan: short-term (0–3 months), mid-term (3–12 months), \
long-term (12+ months).

Write in a professional, concise consulting style. Use data from all three \
research inputs to support your points. Be specific and actionable.\
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
                tools=_SEARCH_TOOLS,
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


class FinancialAnalyst:
    """Handles all quantitative/financial analysis.

    Uses DeepSeek R1 which has strong native math reasoning (93.9% on MATH-500).
    The model performs calculations in its chain-of-thought (think tags) and
    outputs structured JSON — no external tool calling needed.
    """

    name = "financial_analyst"
    display_name = "Financial Analyst"
    model = FINANCIAL_ANALYST_MODEL

    def run(self, question: str, analysis_plan: dict, market_analysis: dict) -> dict:
        """Return the financial analysis as a parsed dict.

        Receives the question, analysis plan, AND market research
        (data restriction: cannot search the web, only analyzes numbers).
        """
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            "STEP 1: Extract all quantitative data from the market research above "
            "(market size, growth rates, competitor revenue, pricing, salaries, etc.)\n\n"
            "STEP 2: Using ONLY those extracted numbers as inputs, calculate:\n"
            "- Cost estimates (setup, operations, marketing, staffing, technology)\n"
            "- Revenue projections for 3 scenarios (conservative, moderate, aggressive)\n"
            "- ROI and break-even timeline\n"
            "- Sensitivity analysis (what happens if key inputs change by ±20%)\n\n"
            "STEP 3: Show all calculations step by step in your thinking, "
            "then output the final structured JSON.\n\n"
            "IMPORTANT: Every number in your output must be traceable to either "
            "the market research data or an explicitly stated assumption."
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": FINANCIAL_ANALYST_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class RiskAnalyst:
    """Identifies and assesses risks using web search and risk assessment tools,
    then produces structured findings via a two-phase approach:
      Phase 1 — Tool loop: LLM calls search_web / read_document / assess_risk
      Phase 2 — Synthesis: LLM converts findings into the JSON schema
    """

    name = "risk_analyst"
    display_name = "Risk Analyst"
    model = RISK_ANALYST_MODEL
    MAX_TOOL_ROUNDS = 8

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
    ) -> dict:
        """Return the risk assessment as a parsed dict.

        Receives question, analysis plan, market research, AND financial
        analysis (to identify financial risks and cross-reference findings).
        """
        research_context = self._research_phase(
            question, analysis_plan, market_analysis, financial_analysis
        )
        return self._synthesis_phase(
            question, analysis_plan, market_analysis, financial_analysis, research_context
        )

    # ------------------------------------------------------------------
    # Phase 1: risk research tool loop
    # ------------------------------------------------------------------
    def _research_phase(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
    ) -> str:
        """Run the risk research tool loop and return all findings."""
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            "Using the information above, identify and assess risks:\n"
            "1. Search for regulatory risks relevant to this business action\n"
            "2. Identify market risks based on the competitive landscape\n"
            "3. Consider operational risks of execution\n"
            "4. Assess competitive threats\n"
            "5. Flag any financial risks from the financial analysis\n\n"
            "Use search_web to research risks, read_document for details, "
            "and assess_risk to structure each identified risk. "
            "Identify at least 5 risks across multiple categories. "
            "When done, stop calling tools."
        )
        messages = [
            {"role": "system", "content": RISK_ANALYST_RESEARCH_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        tool_results: list[str] = []

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=_RA_TOOLS,
            )
            msg = response["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
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
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
        research_context: str,
    ) -> dict:
        """Convert the risk research into the required JSON schema."""
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            f"RISK RESEARCH FINDINGS:\n{research_context}\n\n"
            "Synthesize the risk research above into a structured risk assessment. "
            "Include all risks identified during the research phase."
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": RISK_ANALYST_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class StrategyConsultant:
    """Synthesizes all inputs into a final consulting report (Markdown, streamed)."""

    name = "strategy_consultant"
    display_name = "Strategy Consultant"
    model = STRATEGY_CONSULTANT_MODEL

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
        risk_assessment: dict,
    ) -> Generator[str, None, None]:
        """Stream the consulting report token-by-token.

        Receives the question, analysis plan, market research, financial
        analysis, AND risk assessment (tool restriction: cannot search for
        new data, only reads what it receives).
        """
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            f"RISK ASSESSMENT:\n{json.dumps(risk_assessment, indent=2)}\n\n"
            "Using ALL the inputs above — analysis plan, market research, "
            "financial analysis, and risk assessment — write a complete "
            "consulting report with your strategic recommendation."
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
