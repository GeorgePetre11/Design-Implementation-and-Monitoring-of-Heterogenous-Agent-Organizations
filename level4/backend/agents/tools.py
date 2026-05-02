"""Shared tools for Market Researcher and Risk Analyst agents."""

import json

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


# ---------------------------------------------------------------------------
# Shared tools -- search_web and read_document (used by MR and RA)
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
_DOC_CHAR_LIMIT = 4000  # increased from L3's 3000 for deeper reading


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
# Risk Analyst tools -- assess_risk (plus search_web / read_document above)
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
