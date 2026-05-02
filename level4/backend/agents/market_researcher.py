"""Market Researcher -- investigates the market landscape via web search,
then synthesizes findings into structured JSON."""

import json

import ollama

from .common import (
    MARKET_RESEARCHER_MODEL,
    NUM_CTX_LARGE,
    _extract_urls,
    extract_json,
)
from .tools import _SEARCH_TOOLS, _dispatch_tool


MARKET_RESEARCHER_RESEARCH_PROMPT = """\
You are a Market Researcher at an AI consulting firm. Your task right now is \
to gather information using your research tools.

CURRENT YEAR: 2026. All market size, growth, competitor, and trend data you \
report MUST reflect the state of the market in 2026 (or the most recent \
figures available up to 2026). Your own training data may be older than \
this -- treat the web as the authoritative source, and always prefer newer \
sources over older ones when the numbers conflict. When you call search_web, \
include the year (e.g., "2026" or "2025-2026") in your queries so results \
are current.

You have two tools available:
- search_web(query, max_results): search the web for relevant information
- read_document(url): fetch and read the full text of a web page

CRITICAL INSTRUCTIONS -- follow this workflow:
1. Start by calling search_web with a broad query about the market, \
including "2026" (or "2025-2026") in the query string
2. IMMEDIATELY call read_document on the 2-3 most relevant URLs from the \
search results -- search snippets are NOT enough, you MUST read full pages
3. Then search for specific topics: competitors, market size, trends, etc. \
-- always include the year in each query
4. For EACH search, call read_document on at least 1-2 of the top URLs
5. Continue until you have covered all workstreams in the analysis plan

URL TRACKING RULES -- ABSOLUTE:
- After each search_web call, note the EXACT URLs returned in the results
- When you call read_document, use the EXACT URL from search results
- You will need to provide these EXACT URLs later -- do NOT reconstruct \
URLs from memory, do NOT guess URLs, do NOT modify URLs
- Copy URLs character-by-character from the search results
- If a URL returns an error, note the error and move on -- do NOT \
substitute a different URL you remember from training data

ANTI-HALLUCINATION RULES:
- You MUST call read_document at least 4 times during your research
- Search snippets alone are too shallow -- they contain only 1-2 sentences \
and often have outdated or incomplete data
- The full page content from read_document is essential for accurate data
- If a search returns no useful results, say so -- do NOT make up data
- ONLY report facts you found in actual search results or read pages
- Do NOT supplement search results with information from your training data
- If you cannot find data specific to the client's niche, note this gap \
explicitly -- do NOT substitute data from a broader/different market

RESTRICTIONS:
- Do NOT produce financial analysis, ROI estimates, or cost projections
- Do NOT assess risks
- Only gather market intelligence\
"""

MARKET_RESEARCHER_SYNTHESIS_PROMPT = """\
You are a Market Researcher at an AI consulting firm. You have already \
conducted web research and the findings are provided below. Now synthesize \
that research into structured findings.

CURRENT YEAR: 2026. The market analysis you produce must describe the market \
as of 2026. When the research findings contain figures from multiple years, \
prefer the most recent data point and note the year explicitly (e.g., \
"$4.2B as of 2026 (source: ...)"). Do NOT silently use pre-2026 training \
knowledge to fill gaps -- if the research does not cover a 2026 figure, say \
so rather than reporting stale numbers.

URL CITATION RULES -- these are ABSOLUTE and override everything else:
- You will be given a URL REGISTRY containing every URL found during research
- You may ONLY cite URLs that appear in the URL REGISTRY
- COPY URLs exactly as they appear in the registry -- do NOT modify them
- Do NOT reconstruct URLs from memory (e.g., do NOT guess that a site \
might have a page at /some-path -- only use URLs you actually found)
- Do NOT fabricate URLs that look plausible -- this is the #1 error to avoid
- If you cannot find a URL for a claim, write "(source: research data, \
no specific URL available)" -- this is MUCH better than a fake URL

ANTI-HALLUCINATION RULES -- these are ABSOLUTE:
- ONLY include facts, statistics, and competitor names that appear in the \
RESEARCH FINDINGS below -- do NOT supplement from your training knowledge
- Every statistic or number MUST include the URL where it was found
- If the research does NOT contain data for a field, you MUST write \
"Data not available from research" -- do NOT invent numbers
- Do NOT extrapolate beyond what the sources say
- Do NOT combine partial data from different sources to create new statistics
- If two sources give different numbers, report BOTH with their URLs
- For competitor names: only list companies that appear in the research \
findings -- do NOT add companies you "know" from training data

SEGMENT-RELEVANCE SCOPING -- CRITICAL (this is the #1 reasoning failure mode):
- Every finding, statistic, trend, and competitor you report MUST describe \
the SAME service segment the client actually operates in
- Before including any market statistic, ask: "does this number describe \
the specific product or service the client sells, or a different segment \
of the broader industry?"
- Adjacent-segment data (e.g. reporting AI software CAGR when the client \
sells physical hardware maintenance) is FORBIDDEN -- it causes cascading \
reasoning errors downstream
- If you cannot find segment-specific data, write "No segment-specific data \
found for <segment>; adjacent-segment figures exist but were excluded as \
not comparable" -- do NOT substitute adjacent-segment numbers
- Every entry in `key_findings`, `market_trends`, and `customer_segments` \
must carry a `relevance_to_client_service` string that justifies in one \
sentence why the finding applies to the client's specific service

COMPETITOR IDENTIFICATION -- MANDATORY:
- You MUST identify and name at least 3 real competitors operating in the \
same geography and segment as the client
- Each competitor must have a name, a URL from the URL REGISTRY, and notes \
on service overlap with the client
- If despite focused searching you cannot find 3 competitors, populate \
`key_competitors` with what you found and add an explicit entry \
"searched queries <A>, <B>, <C>; fewer than 3 relevant competitors found" \
in key_findings -- do NOT invent competitors

REGULATORY LANDSCAPE -- MANDATORY when expanding into a new country:
- Surface concrete, named regulations for the target country -- NOT vague \
phrases like "comply with local laws"
- Cover at minimum: tax_regime (corporate tax, VAT, any special regimes), \
labor_law (contributions, notice periods, minimum wage), data_protection \
(GDPR specifics, national DPA), business_registration (entity type, \
registry, minimum capital)
- Cite the URL in the URL REGISTRY for each item; if none, mark [ASSUMED] \
and recommend verification with local counsel

NICHE DATA HONESTY -- CRITICAL:
- If the client's specific niche has no published market data, you MUST \
state this explicitly in market_overview
- Explain what proxy data you are using and why (e.g., "No published data \
exists for physical PC cleaning services in Bucharest. The following \
analysis uses the broader IT services market as a proxy.")
- Do NOT pretend that general market data applies directly to a niche

RESTRICTIONS -- You must obey these absolutely:
- Do NOT perform financial analysis (costs, ROI, projections)
- Do NOT assess risks
- Do NOT write the final consulting report or strategic recommendations
- Base your output ONLY on the research findings provided

OUTPUT FORMAT -- You must respond with valid JSON matching this exact structure:
{
  "market_analysis": {
    "market_overview": "High-level overview. State if niche-specific data was unavailable and what proxy was used. Cite (source URL) for every claim.",
    "market_size_and_growth": "Size estimates with (source URL). If not found, write 'Data not available from research'.",
    "key_competitors": [
      {
        "name": "Competitor name (only from research results)",
        "description": "What they do and how they compete",
        "market_position": "Their standing (leader/challenger/niche)",
        "service_overlap_notes": "How their offering overlaps with the client's specific service",
        "source": "EXACT URL from the URL REGISTRY where this competitor was found"
      }
    ],
    "market_trends": [
      {
        "trend": "Trend description",
        "source": "EXACT URL from URL REGISTRY",
        "relevance_to_client_service": "One sentence justifying why this trend applies to the client's specific segment"
      }
    ],
    "customer_segments": [
      {
        "segment": "Segment name",
        "description": "Characteristics and needs",
        "size_estimate": "Relative or absolute size with (source URL) or 'Estimated based on [assumption]'",
        "relevance_to_client_service": "Why this segment is a realistic customer for the client's specific service"
      }
    ],
    "regulatory_landscape": {
      "tax_regime": "Concrete named rules (corporate tax rate, VAT, special regimes) + (source URL) or [ASSUMED]",
      "labor_law": "Employer contributions, notice periods, minimum wage etc. + (source URL) or [ASSUMED]",
      "data_protection": "GDPR specifics, national DPA, sector-specific rules + (source URL) or [ASSUMED]",
      "business_registration": "Entity type, registry, minimum capital + (source URL) or [ASSUMED]"
    },
    "key_findings": [
      {
        "finding": "Finding text",
        "source": "EXACT URL from URL REGISTRY",
        "relevance_to_client_service": "One sentence justifying why this finding applies to the client's specific service segment"
      }
    ],
    "sources": ["Only URLs from the URL REGISTRY that were actually cited above"]
  }
}

MINIMUM CONTENT REQUIREMENTS (schema-level):
- `key_competitors` MUST contain at least 3 entries OR include an explicit \
"fewer than 3 relevant competitors found" entry in key_findings
- `regulatory_landscape` is REQUIRED and all four subfields must be filled \
(at minimum marked [ASSUMED] if not found)

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""


class MarketResearcher:
    """Investigates the market landscape using web search tools, then produces
    structured findings via a two-phase approach.

    Level 4 changes:
      - /no_think mode during synthesis to reduce hallucination
      - Stricter source citation requirements
      - Supports revision with EM feedback
    """

    name = "market_researcher"
    display_name = "Market Researcher"
    model = MARKET_RESEARCHER_MODEL
    MAX_TOOL_ROUNDS = 8

    def run(
        self,
        question: str,
        analysis_plan: dict,
        revision_feedback: str | None = None,
    ) -> dict:
        """Return the market analysis as a parsed dict.

        If revision_feedback is provided, does a targeted second research pass
        focusing on the gaps identified by the EM.
        """
        research_context = self._research_phase(question, analysis_plan, revision_feedback)
        return self._synthesis_phase(question, analysis_plan, research_context)

    def _research_phase(
        self,
        question: str,
        analysis_plan: dict,
        revision_feedback: str | None = None,
    ) -> str:
        """Run the tool loop and return a text summary of all gathered findings."""
        revision_instruction = ""
        if revision_feedback:
            revision_instruction = (
                f"\n\nREVISION REQUIRED -- The Engagement Manager reviewed your "
                f"previous output and found issues:\n{revision_feedback}\n\n"
                f"Focus your research on addressing these specific gaps. "
                f"Search for the missing data and read relevant pages."
            )

        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            "Use search_web and read_document to gather the market information "
            "needed to address every workstream in the analysis plan. "
            "You MUST call read_document at least 4 times on different URLs. "
            "When you have enough data, stop calling tools."
            f"{revision_instruction}"
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
                options={"num_ctx": NUM_CTX_LARGE},
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

    def _synthesis_phase(
        self, question: str, analysis_plan: dict, research_context: str
    ) -> dict:
        """Convert the raw research into the required JSON schema.
        Uses /no_think to minimize hallucination during factual synthesis.
        Includes a URL registry so the agent can only cite real URLs."""
        urls = _extract_urls(research_context)
        url_list = "\n".join(
            f"  [{i+1}] {u}" for i, u in enumerate(urls)
        ) if urls else "  (no URLs found in research)"

        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"RESEARCH FINDINGS:\n{research_context}\n\n"
            f"URL REGISTRY — you may ONLY cite URLs from this list:\n{url_list}\n\n"
            "Synthesize the research findings above into a structured market "
            "analysis. IMPORTANT: only cite URLs from the URL REGISTRY above. "
            "Copy them exactly. Do NOT reconstruct or fabricate any URLs. "
            "Only include facts found in the RESEARCH FINDINGS — do NOT add "
            "information from your own knowledge.\n\n"
            "/no_think"
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": MARKET_RESEARCHER_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"num_ctx": NUM_CTX_LARGE},
        )
        return extract_json(response["message"]["content"])
