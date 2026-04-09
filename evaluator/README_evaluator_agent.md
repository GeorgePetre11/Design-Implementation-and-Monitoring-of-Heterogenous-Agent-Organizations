# Evaluator Agent — Specification & Integration Guide

Independent quality judge for the AI Consulting Firm multi-agent system.

Model: **Kimi K2.5** (Moonshot AI) — cloud API + Ollama proxy

---

## What the Evaluator Does

The Evaluator Agent has exactly one job: **independently judge the quality of the final consulting report** and produce a structured scorecard. It does not modify the report. It does not interact with other agents during their work. It does not contribute content. It only reads and scores.

The Evaluator exists to solve the self-evaluation problem documented at Level 1: when the same agent that writes a report also scores it, scores are inflated (8–9/10 across all criteria when realistic assessment is 3–5/10). The Evaluator is a *different agent* running a *different model* that never participated in the report generation — it only sees the finished output.

---

## Step-by-Step Behavior

### Step 1 — Receive Inputs

The Evaluator receives exactly two things:

1. **The original client question** — so it can assess whether the report actually answers what was asked
2. **The final consulting report** — the complete output from the Strategy Consultant (L3–L5), the Consultant agent (L2), or the single agent (L1 self-evaluation baseline)

It does NOT receive intermediate outputs, agent logs, system prompts, or any metadata about how the report was produced. It judges the deliverable as a client would receive it.

### Step 2 — Evaluate on 6 Criteria (1–10 each)

The Evaluator scores the report on exactly 6 criteria:

**Completeness (1–10)**
Does the report address all aspects of the client's business question? Are there workstreams or angles that were missed entirely? If the client asked about regulatory requirements and the report doesn't mention them, that's a completeness gap. If the client asked about Bucharest specifically and the report gives generic Romania-wide analysis, that's a partial miss.

- Score 1–3: Major sections missing, question partially or barely addressed
- Score 4–6: Most areas covered but with noticeable gaps or superficial treatment
- Score 7–8: All key areas addressed with reasonable depth
- Score 9–10: Comprehensive coverage, no significant gaps, anticipates follow-up questions

**Accuracy (1–10)**
Are the claims and numbers realistic and internally consistent? Are sources cited? Does the math check out? This is the criterion that catches hallucinations. If costs are €160K and revenue is €120K, the report should not claim an 8-month break-even. If it names competitors, can they plausibly exist? If it cites a market size figure, is a source provided?

- Score 1–3: Majority of data points are fabricated, unsourced, or internally contradictory
- Score 4–6: Mix of plausible and questionable claims; some numbers lack sources; minor inconsistencies
- Score 7–8: Most claims are sourced or clearly marked as estimates; math is internally consistent
- Score 9–10: All claims sourced, calculations verified, no internal contradictions

**Coherence (1–10)**
Does the analysis flow logically from data to recommendation? Do the recommendations actually follow from the evidence presented? Are there contradictions between sections? If the risk section flags fierce competition but the strategy section assumes 20% market capture in year one, that's an incoherence. If the market analysis shows declining demand but the recommendation is aggressive expansion, the logic chain is broken.

- Score 1–3: Major logical contradictions; recommendations disconnected from analysis
- Score 4–6: Generally logical but with notable gaps in reasoning or cross-section contradictions
- Score 7–8: Clear logical flow; recommendations supported by presented evidence
- Score 9–10: Airtight reasoning; every recommendation traces back to specific findings

**Structure (1–10)**
Is it organized like a professional consulting deliverable? Clear sections, logical ordering, executive summary, proper formatting? Are headings descriptive? Is information where you'd expect to find it? This is typically the strongest dimension for LLM-generated reports — models are good at mimicking consulting report structure.

- Score 1–3: Disorganized, missing standard sections, hard to navigate
- Score 4–6: Recognizable structure but with ordering issues or missing elements (e.g., no executive summary)
- Score 7–8: Well-organized with clear headings, logical section order, professional formatting
- Score 9–10: Board-ready; executive summary, clear visual hierarchy, professional-grade organization

**Actionability (1–10)**
Could a real decision-maker act on these recommendations? Are next steps specific — with timelines, budgets, responsibilities, and measurable KPIs? Or are they vague platitudes like "conduct further research" and "build partnerships"? A score of 8+ means a client could hand this report to their operations team and start executing.

- Score 1–3: Recommendations are generic advice that could apply to any company in any market
- Score 4–6: Some specificity but missing concrete timelines, budgets, or responsible parties
- Score 7–8: Specific recommendations with phased timelines and resource estimates
- Score 9–10: Fully actionable roadmap with KPIs, owners, budgets, milestones, and contingencies

**Critical Depth (1–10)**
Does the report consider counterarguments, risks, limitations, and alternative perspectives? Does it include sensitivity analysis (what if revenue is 30% lower than projected)? Does it acknowledge what it doesn't know? Or does it present one rosy scenario as inevitable?

- Score 1–3: No counterarguments, no risk discussion, no limitations acknowledged
- Score 4–6: Risks mentioned but superficially; no sensitivity analysis; limited alternative perspectives
- Score 7–8: Meaningful risk discussion, multiple scenarios considered, limitations acknowledged
- Score 9–10: Thorough sensitivity analysis, explicit assumptions, counterarguments addressed, unknown unknowns flagged

### Step 3 — Produce Justifications

For each criterion, the Evaluator writes a **2–4 sentence justification** explaining the score. The justification must reference specific parts of the report — not vague statements like "the analysis could be deeper." Good justifications look like:

> "The report cites a Bucharest IT market size of €150M annually with no source. The competitor names 'Tech Maintenance Solutions' and 'IT Care Plus' cannot be verified. The break-even calculation claims 8 months, but the model's own numbers show €160K in costs against €120K in revenue — a net loss, not a break-even."

Bad justifications look like:

> "The accuracy could be improved with more reliable data sources."

### Step 4 — Output Structured JSON Scorecard

The Evaluator outputs a single JSON object conforming to the schema below. No markdown, no preamble, no explanation outside the JSON.

### Step 5 — What the Evaluator Must NOT Do

- **Cannot modify or rewrite the report** — it is a judge, not an editor
- **Cannot search the web** — it judges based on internal consistency, plausibility, and whether claims are sourced; it does not fact-check externally
- **Cannot access tools** — no `search_web()`, no `calculate()`, nothing; pure LLM evaluation
- **Cannot communicate with other agents** — it receives the report and produces a scorecard; at L5 hierarchical/hybrid, the Engagement Manager may route its feedback, but the Evaluator itself only produces the scorecard
- **Cannot inflate scores** — the system prompt explicitly instructs against the self-evaluation bias observed at L1
- **Cannot refuse to score** — even a terrible report gets scored (likely with 1–3/10 on most criteria)

---

## Output Schema

```json
{
  "evaluation": {
    "completeness": {
      "score": 6,
      "justification": "The report covers market analysis, financials, and strategy but omits any discussion of regulatory requirements despite the client explicitly requesting this. The risk section is superficial with only three generic risks identified."
    },
    "accuracy": {
      "score": 3,
      "justification": "Multiple data points appear fabricated — the €150M market size has no source, competitor names cannot be verified, and the financial projections are internally inconsistent (costs exceed revenue by €40K yet the report claims 8-month break-even)."
    },
    "coherence": {
      "score": 5,
      "justification": "The report flows logically from market analysis to recommendation, but the strategy section contradicts the risk section. The recommendation for gradual entry is undermined by aggressive Year 1 market share targets of 20%."
    },
    "structure": {
      "score": 7,
      "justification": "Well-organized with clear headings and a logical section order following standard consulting report structure. Would benefit from an executive summary and visual aids."
    },
    "actionability": {
      "score": 4,
      "justification": "The implementation roadmap provides timeframes but lacks specific budget allocations per phase, named responsible parties, or measurable KPIs. A decision-maker would need significantly more detail to act on this."
    },
    "critical_depth": {
      "score": 3,
      "justification": "No sensitivity analysis, no counterarguments to the recommended option, and no discussion of what happens if key assumptions fail. Risk mitigations are generic one-liners rather than actionable strategies."
    }
  },
  "overall_score": 4.7,
  "summary": "The report demonstrates adequate structural organization but suffers from fabricated data, internal contradictions, and insufficient analytical depth. The recommendations are not grounded in verifiable evidence.",
  "strongest_dimension": "structure",
  "weakest_dimension": "accuracy",
  "critical_issues": [
    "Break-even calculation contradicts the report's own cost and revenue figures",
    "Competitor names appear to be fabricated",
    "No sources cited for any market data claims"
  ]
}
```

### Pydantic Schema (for validation in the orchestrator)

```python
from pydantic import BaseModel, Field
from typing import List

class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=10, description="Score from 1 (worst) to 10 (best)")
    justification: str = Field(
        min_length=50, max_length=500,
        description="2-4 sentence justification referencing specific report content"
    )

class EvaluationScorecard(BaseModel):
    evaluation: dict[str, CriterionScore] = Field(
        description="Scores for: completeness, accuracy, coherence, structure, actionability, critical_depth"
    )
    overall_score: float = Field(
        ge=1.0, le=10.0,
        description="Weighted average of all criteria scores"
    )
    summary: str = Field(
        min_length=50, max_length=300,
        description="1-3 sentence overall assessment"
    )
    strongest_dimension: str = Field(
        description="The criterion with the highest score"
    )
    weakest_dimension: str = Field(
        description="The criterion with the lowest score"
    )
    critical_issues: List[str] = Field(
        default_factory=list,
        description="List of the most severe problems found (if any)"
    )
```

---

## System Prompt

```
You are the Evaluator Agent in a multi-agent AI Consulting Firm. Your sole responsibility is to independently assess the quality of a consulting report produced by other agents.

You are NOT the author of this report. You did not participate in its creation. You are an independent judge.

## Your Task

You will receive:
1. The original client business question
2. The final consulting report

You must evaluate the report on exactly 6 criteria, each scored 1-10, with a written justification of 2-4 sentences per criterion.

## Scoring Criteria

1. **Completeness (1-10)**: Does the report address ALL aspects of the client's question? Are any workstreams, topics, or angles missing? Score 1-3 if major sections are missing. Score 7+ only if all key areas are covered with reasonable depth.

2. **Accuracy (1-10)**: Are claims and numbers realistic and internally consistent? Are sources cited? Does the math check out? If the report states costs of €160K and revenue of €120K but claims break-even in 8 months, that is an internal contradiction — score accordingly. If market data, competitor names, or statistics are presented without sources, note this. Score 1-3 if most data appears fabricated or contradictory.

3. **Coherence (1-10)**: Does the analysis flow logically from data to recommendation? Do findings in one section contradict findings in another? If the risk section flags fierce competition but the strategy assumes 20% market share in year one, that is incoherent. Score 7+ only if recommendations clearly follow from the evidence.

4. **Structure (1-10)**: Is the report organized like a professional consulting deliverable? Clear sections, logical ordering, executive summary, proper formatting? Score 7+ if the structure is clean and navigable.

5. **Actionability (1-10)**: Could a decision-maker act on these recommendations? Are there specific timelines, budgets, KPIs, and responsible parties? Or just vague advice like "conduct further research"? Score 7+ only if a client could hand this to their operations team and start executing.

6. **Critical Depth (1-10)**: Does the report consider risks, counterarguments, limitations, and alternative scenarios? Is there sensitivity analysis? Does it acknowledge what it does not know? Score 1-3 if the report presents a single optimistic scenario with no critical examination.

## Scoring Rules

- Be HONEST. Do not inflate scores. A report full of fabricated data should score 2-3 on accuracy, not 7.
- Reference SPECIFIC content from the report in your justifications. Do not write vague statements like "could be improved."
- If the report's own numbers contradict each other, flag this explicitly.
- If claims lack sources, say so.
- If competitor names or data points seem invented, say so.
- The overall_score is the arithmetic mean of all 6 criterion scores, rounded to 1 decimal.
- Identify the strongest and weakest dimensions.
- List critical issues — the 2-5 most severe problems you found.

## Output Format

Respond with ONLY a valid JSON object matching this structure. No markdown fences, no preamble, no text outside the JSON:

{
  "evaluation": {
    "completeness": {"score": <1-10>, "justification": "<2-4 sentences>"},
    "accuracy": {"score": <1-10>, "justification": "<2-4 sentences>"},
    "coherence": {"score": <1-10>, "justification": "<2-4 sentences>"},
    "structure": {"score": <1-10>, "justification": "<2-4 sentences>"},
    "actionability": {"score": <1-10>, "justification": "<2-4 sentences>"},
    "critical_depth": {"score": <1-10>, "justification": "<2-4 sentences>"}
  },
  "overall_score": <float>,
  "summary": "<1-3 sentence overall assessment>",
  "strongest_dimension": "<criterion name>",
  "weakest_dimension": "<criterion name>",
  "critical_issues": ["<issue 1>", "<issue 2>", ...]
}
```

---

## Model: Kimi K2.5 via Cloud API

### Why Kimi K2.5 for the Evaluator

Kimi K2.5 is a frontier-class model (1T total parameters, 32B active per token) with strong analytical reasoning, multilingual support, and an OpenAI-compatible API. For the Evaluator role specifically:

- **Independent provider**: Adds Moonshot AI as a distinct provider in the heterogeneity matrix (alongside Alibaba/Qwen for local agents, and potentially OpenAI/Anthropic). This strengthens the thesis argument about heterogeneous model providers.
- **Strong reasoning**: Scores 96.1% on AIME 2025 and 76.8% on SWE-Bench — sufficient analytical depth for rubric-based evaluation.
- **256K context window**: Can evaluate even very long consulting reports without truncation.
- **Cost**: $0.60/M input tokens, $2.50/M output tokens on the official API. A single evaluation (~4K input + ~1K output) costs roughly $0.005. Running 100 evaluations across all experiments costs about $0.50.
- **Free access available**: NVIDIA NIM provides free API access; kimi.com offers ~30-50 free messages/day.

### Access Options (cheapest to most expensive)

| Method | Cost | Rate Limits | Best For |
|--------|------|-------------|----------|
| **NVIDIA NIM** | Free | No published limits | Development, testing |
| **kimi.com web chat** | Free | ~30-50 msgs/day | Manual testing, qualitative checks |
| **Moonshot API (Tier 0)** | $1 min recharge | 3 RPM, 1 concurrent | Light experiments |
| **Moonshot API (Tier 1)** | $10 cumulative | 200 RPM, 50 concurrent | Production pipeline |
| **OpenRouter** | ~$0.38/M in, $1.72/M out | Varies by plan | If Moonshot is unreachable from RO |

### Recommended: Moonshot Official API

For the production pipeline, use Moonshot's official API at `https://api.moonshot.ai/v1`. A $10 recharge unlocks Tier 1 (200 RPM, 50 concurrent requests) and should cover the entire thesis.

---

## Integration

### Option A — Direct API Call (Recommended for Production)

The Moonshot API is OpenAI SDK-compatible. Drop-in replacement:

```python
"""
Evaluator Agent — calls Kimi K2.5 via Moonshot's OpenAI-compatible API.
"""
import json
from openai import OpenAI

EVALUATOR_SYSTEM_PROMPT = """<paste the full system prompt from above>"""

class EvaluatorAgent:
    def __init__(self, api_key: str, base_url: str = "https://api.moonshot.ai/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "kimi-k2.5"

    def evaluate(self, client_question: str, report: str) -> dict:
        """Score a consulting report on the 6-criterion rubric."""
        user_message = (
            f"## Original Client Question\n{client_question}\n\n"
            f"## Consulting Report to Evaluate\n{report}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,  # Low temp for consistent scoring
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if the model wraps output
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        scorecard = json.loads(raw)

        # Validate overall_score matches criterion average
        scores = [v["score"] for v in scorecard["evaluation"].values()]
        expected_avg = round(sum(scores) / len(scores), 1)
        scorecard["overall_score"] = expected_avg

        return scorecard
```

### Option B — Via Ollama Proxy (for unified local/cloud interface)

If your orchestrator routes all LLM calls through Ollama, you can proxy Kimi K2.5 cloud requests through Ollama using a custom model that forwards to the API. This keeps a uniform `ollama.chat()` interface across local and cloud models.

**Create a Modelfile that proxies to the cloud:**

This approach uses a lightweight local wrapper. Your orchestrator calls Ollama normally, and for the evaluator model, the request is forwarded to the Moonshot API.

**Option B1 — LiteLLM as a unified proxy (recommended):**

LiteLLM provides a single OpenAI-compatible endpoint that routes to any provider:

```bash
pip install litellm
```

```python
"""
Evaluator Agent — calls Kimi K2.5 through LiteLLM for unified routing.
Works alongside Ollama-hosted local models in the same pipeline.
"""
import json
import litellm

EVALUATOR_SYSTEM_PROMPT = """<paste the full system prompt from above>"""

class EvaluatorAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # LiteLLM model string for Moonshot's Kimi K2.5
        self.model = "openai/kimi-k2.5"  # "openai/" prefix = OpenAI-compatible endpoint
        self.api_base = "https://api.moonshot.ai/v1"

    def evaluate(self, client_question: str, report: str) -> dict:
        user_message = (
            f"## Original Client Question\n{client_question}\n\n"
            f"## Consulting Report to Evaluate\n{report}"
        )

        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=0.1,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        return json.loads(raw)
```

**Option B2 — NVIDIA NIM for free access:**

```python
# Same code as Option A, just change base_url and api_key
client = OpenAI(
    api_key="your-nvidia-nim-key",  # Free from build.nvidia.com
    base_url="https://integrate.api.nvidia.com/v1"
)
# model = "moonshotai/kimi-k2.5"  # NVIDIA NIM model identifier
```

### Orchestrator Integration

In your main pipeline orchestrator, the Evaluator is called last:

```python
"""
Example: How the Evaluator fits into the Level 4 pipeline.
"""
async def run_pipeline(client_question: str):
    # Step 1: Engagement Manager decomposes the question
    plan = await engagement_manager.decompose(client_question)

    # Step 2: Specialized agents produce their analyses
    market_research = await market_researcher.analyze(plan.market_workstream)
    financial_analysis = await financial_analyst.analyze(plan.financial_workstream)
    risk_assessment = await risk_analyst.analyze(plan.risk_workstream)

    # Step 3: Strategy Consultant synthesizes the final report
    report = await strategy_consultant.synthesize(
        market_research, financial_analysis, risk_assessment
    )

    # Step 4: Evaluator independently scores the report
    evaluator = EvaluatorAgent(api_key=os.getenv("MOONSHOT_API_KEY"))
    scorecard = evaluator.evaluate(
        client_question=client_question,
        report=report.content  # Only the final report text, nothing else
    )

    # Step 5: Log to monitoring database
    await monitor.log_event(
        event_type="evaluation_complete",
        agent_name="evaluator",
        data=scorecard
    )

    return report, scorecard
```

---

## Behavior Across Complexity Levels

| Level | What the Evaluator Receives | What Changes |
|-------|----------------------------|--------------|
| **L1** | The single agent's self-evaluation (baseline comparison only) | No separate Evaluator exists at L1. The self-eval scores from L1 are compared against the Evaluator's scores at L2+ to demonstrate self-evaluation inflation. |
| **L2** | The Consultant agent's full report + original client question | First level with an independent Evaluator. Scores should be more honest than L1 self-eval. |
| **L3** | The Strategy Consultant's final report + original client question | Same inputs. Report quality may improve due to role specialization (separate Market Researcher, Financial Analyst, etc.). |
| **L4** | The Strategy Consultant's final report + original client question | Same inputs. Report quality may further improve with full specialization (6 agents). |
| **L5 Pipeline** | Same as L4. Evaluator is the last step in the fixed pipeline. | No feedback loop — the Evaluator scores and the pipeline is done. |
| **L5 Hierarchical** | Same as L4. The Engagement Manager decides whether to act on the evaluation. | The EM may use the scorecard to decide if the report needs another revision pass, but the Evaluator itself behaves identically. |
| **L5 Hybrid** | Same as L4. The evaluation can trigger iteration loops. | If scores fall below a threshold (e.g., any criterion < 5), the orchestrator may send the report back for revision. The Evaluator re-evaluates the revised version. |

### L5 Iteration Threshold (Hybrid only)

```python
MINIMUM_ACCEPTABLE_SCORES = {
    "completeness": 5,
    "accuracy": 5,
    "coherence": 5,
    "structure": 5,
    "actionability": 4,
    "critical_depth": 4,
}
MAX_ITERATIONS = 3  # Prevent infinite loops

def needs_revision(scorecard: dict) -> bool:
    """Check if any criterion falls below the minimum threshold."""
    for criterion, min_score in MINIMUM_ACCEPTABLE_SCORES.items():
        actual = scorecard["evaluation"][criterion]["score"]
        if actual < min_score:
            return True
    return False
```

---

## Monitoring Events

The Evaluator logs these events to the SQLite monitoring database:

| Event Type | Data | When |
|------------|------|------|
| `evaluator_start` | `{session_id, level, model: "kimi-k2.5"}` | Evaluator receives the report |
| `evaluator_complete` | `{session_id, scorecard, tokens_used, latency_ms}` | Scorecard produced |
| `evaluator_error` | `{session_id, error_message}` | API call failed or JSON parse error |
| `evaluation_triggered_revision` | `{session_id, failing_criteria, iteration}` | L5 hybrid only: scores below threshold |

---

## Thesis Role

The Evaluator's scores are the **primary dependent variable** in all four thesis experiments:

1. **Progressive Complexity Comparison**: Compare Evaluator scores at L2, L3, L4, L5 to quantify improvement from specialization. Compare against L1 self-evaluation to demonstrate self-assessment inflation.

2. **Organizational Structure Comparison**: At L5, compare Evaluator scores across pipeline vs. hierarchical vs. hybrid to measure which workflow produces the highest-quality reports.

3. **Heterogeneity Impact**: Compare Evaluator scores when all agents use the same model vs. when agents use different models (Qwen, DeepSeek, Kimi, etc.).

4. **Evaluator Consistency**: Run the same report through the Evaluator multiple times (e.g., 5 runs) to measure score variance. Compare Evaluator scores against human evaluation (your own assessment + the "Assessed" column from the L1 analysis document) to validate calibration.

### Key Thesis Argument

The gap between L1 self-evaluation scores (8–9/10) and the independent Evaluator scores (expected 3–5/10 for L1-quality reports) is one of the central findings. This gap directly motivates the architectural decision to separate evaluation from generation — a core principle of the heterogeneous agent organization design.

---

## Configuration

### Environment Variables

```bash
# Moonshot API (primary)
EVALUATOR_API_KEY=sk-your-moonshot-key
EVALUATOR_BASE_URL=https://api.moonshot.ai/v1
EVALUATOR_MODEL=kimi-k2.5

# Or NVIDIA NIM (free alternative)
EVALUATOR_API_KEY=nvapi-your-nvidia-key
EVALUATOR_BASE_URL=https://integrate.api.nvidia.com/v1
EVALUATOR_MODEL=moonshotai/kimi-k2.5
```

### Agent Config (in your pipeline config)

```python
EVALUATOR_CONFIG = {
    "agent_name": "evaluator",
    "role": "Independent Quality Judge",
    "model": os.getenv("EVALUATOR_MODEL", "kimi-k2.5"),
    "provider": "moonshot",
    "base_url": os.getenv("EVALUATOR_BASE_URL", "https://api.moonshot.ai/v1"),
    "api_key": os.getenv("EVALUATOR_API_KEY"),
    "temperature": 0.1,
    "max_tokens": 2000,
    "tools": [],  # No tools — pure evaluation
    "output_schema": "EvaluationScorecard",
    "restrictions": [
        "Cannot modify the report",
        "Cannot search the web",
        "Cannot communicate with other agents",
        "Cannot access any tools",
        "Must score all 6 criteria even for poor reports",
    ],
}
```

### Heterogeneity Matrix (Updated)

| Agent | Model | Provider | Runs On |
|-------|-------|----------|---------|
| Engagement Manager | Qwen 3 8B | Alibaba (Ollama) | MacBook M3 |
| Market Researcher | Qwen 3 14B | Alibaba (Ollama) | Ryzen 7 PC |
| Financial Analyst | DeepSeek R1 14B | DeepSeek (Ollama) | Ryzen 7 PC |
| Risk Analyst | Qwen 3 14B | Alibaba (Ollama) | Ryzen 7 PC |
| Strategy Consultant | Qwen 3 32B | Alibaba (Ollama) | Ryzen 7 PC |
| **Evaluator** | **Kimi K2.5** | **Moonshot AI (Cloud API)** | **Cloud** |

This gives the system **3 distinct model families** (Qwen, DeepSeek, Kimi) across **2 providers** (local Ollama, cloud Moonshot API) on **3 execution environments** (MacBook, Ryzen PC, cloud) — a strong heterogeneity story for the thesis.

---

## Error Handling

```python
import json
import time
from openai import OpenAI, APIError, RateLimitError

class EvaluatorAgent:
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def evaluate(self, client_question: str, report: str) -> dict:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._call_api(client_question, report)
                scorecard = self._parse_response(response)
                self._validate_scorecard(scorecard)
                return scorecard
            except RateLimitError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                raise
            except json.JSONDecodeError as e:
                # Model didn't produce valid JSON — retry with stricter prompt
                if attempt < self.MAX_RETRIES - 1:
                    continue
                raise ValueError(f"Evaluator failed to produce valid JSON after {self.MAX_RETRIES} attempts: {e}")
            except APIError as e:
                raise ConnectionError(f"Moonshot API error: {e}")

    def _validate_scorecard(self, scorecard: dict):
        """Ensure all 6 criteria are present and scores are in range."""
        required = {"completeness", "accuracy", "coherence", "structure", "actionability", "critical_depth"}
        present = set(scorecard.get("evaluation", {}).keys())
        missing = required - present
        if missing:
            raise ValueError(f"Missing criteria in scorecard: {missing}")
        for criterion, data in scorecard["evaluation"].items():
            if not (1 <= data["score"] <= 10):
                raise ValueError(f"{criterion} score {data['score']} out of range [1, 10]")
            if len(data["justification"]) < 30:
                raise ValueError(f"{criterion} justification too short: '{data['justification']}'")
```

---

## Testing the Evaluator

### Quick Smoke Test

Feed it a deliberately bad report and verify it produces low scores:

```python
bad_report = """
# Market Entry Report
## 1. Market Analysis
The market is growing. There are some competitors.
## 2. Recommendation
We recommend entering the market.
"""

scorecard = evaluator.evaluate(
    client_question="Should we expand our German IT services company into Bucharest, Romania?",
    report=bad_report
)

# Expected: all scores 1-3, critical_issues populated
assert scorecard["evaluation"]["completeness"]["score"] <= 3
assert scorecard["evaluation"]["accuracy"]["score"] <= 3
assert len(scorecard["critical_issues"]) > 0
```

### Calibration Test

Feed it the actual L1 Report A and Report B from your experiments. Compare the Evaluator's scores against the "Assessed" scores in your `level1_analysis.docx`. The Evaluator should produce scores roughly aligned with:

| Criterion | Assessed A | Assessed B | Evaluator should be ± 2 of these |
|-----------|-----------|-----------|-----------------------------------|
| Completeness | 5/10 | 6/10 | Yes |
| Accuracy | 3/10 | 3/10 | Yes |
| Coherence | 6/10 | 5/10 | Yes |
| Structure | 7/10 | 7/10 | Yes |
| Actionability | 4/10 | 5/10 | Yes |
| Critical Depth | 3/10 | 4/10 | Yes |

If the Evaluator consistently scores >2 points above these assessed values, it may be exhibiting the same inflation problem — adjust the system prompt to be more aggressive about penalizing fabricated data and missing sources.
