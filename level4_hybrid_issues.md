# Level 4 Hybrid — Issues Report

**Report analyzed:** `Consulting_Report_new.pdf` (Level 4 — Hybrid Hierarchical Workflow with EM Review)
**Client prompt:** German mid-size IT services company, 50 employees, €2M revenue, physical PC cleaning/maintenance, considering Bucharest expansion, €100K first-year budget.
**Recommendation produced:** DO NOT PROCEED
**Overall assessment:** Significant improvement over Level 1 (no fabricated competitors, no math contradictions), but still has substantive analytical errors and layout problems that prevent it from passing as a real consulting deliverable.

---

## Part 1 — Content & Accuracy Issues

Issues grouped by the agent responsible, so you know where to intervene in the code.

### Financial Analyst issues

#### 1. Staffing cost is way too low
€32K/year for two technicians + one admin = ~€890/month gross per person. A junior IT technician in Bucharest costs an employer roughly €1,200–1,800/month fully loaded (gross + employer contributions). Based on the Paylab Romanian IT salary survey, IT workers earn between 4,850 and 17,400 RON net monthly — the agent is either forgetting employer-side contributions entirely, or anchoring on Romanian minimum wage without adjusting for the IT sector. Three people fully loaded should be €45–60K/year minimum.

**Fix:** add to the Financial Analyst's system prompt an explicit instruction to model **fully-loaded employer cost**, not net salary, and to cite a salary source (Paylab, Glassdoor, salaryexplorer) via its search tool. Add a sanity check: "if cost per employee < €15K/year, flag for review."

#### 2. Revenue model conflates one-time and recurring services
"300 clients × €200" = €60K reads like 300 one-time cleaning jobs. But the client is a **corporate maintenance** company — their revenue model is contract-based (€X per workstation per month, or a flat quarterly fee). A corporate maintenance contract for a 50-person office is typically €1,500–5,000/year, not €200 one-time. This single modeling error drives most of the "don't proceed" conclusion.

**Fix:** the Financial Analyst needs to be told explicitly to derive the revenue model **from the client context** (the original brief says "physical PC cleaning and maintenance for corporate clients" — that's recurring). Add to the output schema a required field like `revenue_model_type: "one_time" | "recurring_contract" | "hybrid"` with a justification, so the agent has to consciously pick one.

#### 3. Only one revenue scenario is shown
The report mentions an "Aggressive" scenario but doesn't show Conservative or Moderate alongside it. The Claude.md example explicitly asks for three scenarios.

**Fix:** make `scenarios: [conservative, moderate, aggressive]` a required array of length 3 in the output schema. The agent physically cannot submit without three.

#### 4. No sensitivity analysis
Claude.md specifies sensitivity analysis as part of the Financial Analyst's output. It's missing entirely.

**Fix:** add `sensitivity_analysis` as a required field in the schema.

---

### Market Researcher issues

#### 5. The AI-CAGR finding is a category error
The agent searched for "Romania IT market trends," found the 34.6% AI services CAGR, and passed it along as if it were relevant to physical hardware maintenance. It isn't — AI software and physical cleaning are different markets. This then propagated to the Strategy Consultant, who treated it as evidence that physical maintenance demand is declining.

**Fix:** tighten the Market Researcher's system prompt so that **every finding it reports must be scoped to the specific service the client offers**. Example wording: "Before reporting a market statistic, verify it describes the segment the client operates in. If you cannot find data for that specific segment, say 'no segment-specific data found' rather than substituting data from an adjacent segment." Also add an output field `relevance_to_client_service: string` forcing the agent to justify relevance per finding.

#### 6. No regulatory specifics for Romania
The report says "compliance with local labor laws and tax regulations" without naming anything. A Market Researcher with web search should be surfacing: the Romanian micro-enterprise tax regime (threshold €500K, not relevant at €2M but worth knowing), the IT specialist income tax exemption (partially phased out in 2023, changed again recently), CAS/CASS contribution rates, ONRC registration (Trade Register), and the minimum share capital for an SRL.

**Fix:** either split regulatory research into the Risk Analyst (who in Claude.md also has `search_web`), or add a mandatory `regulatory_landscape` section to the Market Researcher's output schema with required subfields (`tax_regime`, `labor_law`, `data_protection`, `business_registration`).

#### 7. No competitor identification
Report says market is "fragmented with low entry barriers" but doesn't name a single competitor. This should be easy with web search — companies like Ecotronix, PC Garage service centers, and local IT break-fix shops in Bucharest are discoverable.

**Fix:** make `named_competitors: [{name, url, service_overlap_notes}, ...]` a required field with a minimum count (e.g., at least 3). If the agent can't find three, it must explicitly report "searched with queries X, Y, Z, found fewer than 3 relevant competitors."

---

### Strategy Consultant issues

#### 8. Uncritically accepted the AI-CAGR argument
This is the subtler failure. The Strategy Consultant's job is to synthesize, which means evaluating whether the findings it receives actually support the conclusion. It took an irrelevant market stat and used it as a load-bearing argument for "DO NOT PROCEED."

**Fix:** add to the Strategy Consultant's system prompt a **relevance-check step**: before using any finding from another agent as a premise in the recommendation, briefly state *why* that finding is relevant to the specific client question. If you can't articulate relevance in one sentence, drop the finding. This forces an intermediate reasoning step that would have caught the AI-CAGR issue.

#### 9. Alternative strategies have no numbers
The three alternatives at the end (pivot to MSP, hybrid model, lean entry) are reasonable directional ideas but have zero quantification. For a Level 4 report, each alternative should at least sketch an order-of-magnitude budget.

**Fix:** if the Strategy Consultant recommends alternatives, the output schema should require each alternative to have `approximate_budget` and `approximate_revenue` fields, even if marked `~estimate`. This forces the Strategy Consultant to either loop back to the Financial Analyst (in Hybrid mode, it can!) or explicitly mark alternatives as "directional, not costed."

---

### EM / workflow issues (Hybrid-specific)

#### 10. The review loop didn't catch the relevance error
Hybrid mode is supposed to allow the EM to send work back for revision. Either the EM didn't review the Market Researcher's findings for relevance, or the review criteria don't include "is this finding actually about the client's segment?"

**Fix:** give the EM an explicit review checklist when it evaluates intermediate outputs:

```
For each Market Researcher finding, check:
- Does it describe the same service segment the client operates in?
- Is the source cited?
- Could a reasonable consultant disagree with it?

If any answer is "no" or "unclear", send back for revision with specific feedback.
```

#### 11. The Strategy Consultant never requested more financial scenarios
In Hybrid, it's allowed to. The fact that it accepted a single "Aggressive" scenario without asking for Conservative and Moderate suggests the iteration-request mechanism isn't being used, or the agent isn't prompted to use it.

**Fix:** add to the Strategy Consultant's prompt: "If the Financial Analyst's output is missing scenarios, sensitivity analysis, or if any projected number seems to drive the recommendation by itself, you MUST request revision before writing the final report."

---

## Part 2 — Layout & Presentation Issues

These are problems with how the report is rendered/exported, not with the content generation itself. Most trace back to the frontend "Download PDF" flow (browser print-to-PDF from the HTML view).

#### L1. Stray markdown characters rendered literally
Throughout the report, asterisks show up as literal `*` characters instead of being converted to bullets:
```
* Budget Overrun: Total estimated costs for the first year are €102,000...
* Revenue Shortfall: Even under the "Aggressive" scenario...
```
This means markdown is being emitted into HTML that isn't parsing it, or the print CSS isn't styling `<ul>` properly.

**Fix:** make sure the frontend runs the agent's markdown output through a proper renderer (`marked`, `markdown-it`, or similar) before exporting to PDF. Don't just dump raw text into a `<pre>` or `<div>`.

#### L2. Nested bullets flatten to the same level
`Cost Breakdown:` is followed by `Fixed Setup/Ops:` and `Key cost drivers include staffing...` — these should be indented as sub-bullets but render at the same level.

**Fix:** same as L1 — proper markdown-to-HTML rendering preserves nesting.

#### L3. Four stacked titles before content starts
The first page has:
- "AI Consulting Firm"
- "HETEROGENEOUS AGENT ORGANIZATIONS — LEVEL 4 (HYBRID)"
- "Level 4 — Hybrid Hierarchical Workflow with EM Review"
- "Strategic Market Entry Analysis: Bucharest, Romania"

That's four titles for a three-page report. Looks academic-report-heavy rather than consulting-report.

**Fix:** collapse to one title (the analysis topic) + one subtitle (the level/run metadata). Move the "AI Consulting Firm / HETEROGENEOUS AGENT ORGANIZATIONS" branding to a small header or footer.

#### L4. Print artifacts in the PDF
Each page shows:
- Header: `4/21/26, 11:52 AM    Consulting Report`
- Footer: `about:blank 1/3`

This is the browser's default "print to PDF" chrome showing up, and `about:blank` is the URL of the print preview page.

**Fix:** either generate the PDF server-side (WeasyPrint, Puppeteer with `--print-to-pdf` and disabled headers/footers), or inject CSS `@page { margin: 0; }` and set explicit header/footer templates. For the thesis defense, server-side PDF generation is worth the 30 minutes to set up properly.

#### L5. No section numbering
Sections are just H2s: "Executive Summary", "Financial Analysis", "Market Analysis", "Risk Assessment", "Final Recommendation". For a deliverable the evaluator has to grade section-by-section, numbered sections (`1. Executive Summary`, `2. Financial Analysis`, ...) make reference easier and look more professional.

**Fix:** either number in the prompt template, or post-process to auto-number H2s.

#### L6. Number and currency format inconsistency
Within the same report: `€100K`, `€100,000`, `€102,000`, `€32,000`, `€60,000`, `€42,000`, `€5,000/year`. Pick one style (either `€100K` everywhere or `€100,000` everywhere) and enforce it.

**Fix:** add a formatting instruction to the Strategy Consultant's prompt: "Use full currency format (e.g., €100,000) for all amounts in the final report. Do not mix notation styles."

#### L7. No source citations anywhere
The "34.6% CAGR" figure, the "two technicians + one admin" staffing assumption, and the "300 clients × €200" revenue model all appear with no source or justification. For a thesis-graded deliverable, every load-bearing number should be cited or explicitly marked as an assumption.

**Fix:** require the Market Researcher and Financial Analyst to return findings as `{value, source_url_or_assumption, confidence}` tuples in their JSON output. The Strategy Consultant's prompt then instructs it to render these as footnotes or inline `(source: X)` annotations.

#### L8. No visual elements
No tables, no charts, no risk matrix, no financial projection plot — despite the Financial Analyst having a `create_chart()` tool per Claude.md, and despite the Risk Analyst producing structured probability/impact ratings that are perfect for a heatmap.

**Fix:** in the final report template, include placeholders that the Strategy Consultant must fill — at minimum:
- A cost breakdown table
- A scenario comparison table (conservative/moderate/aggressive)
- A risk matrix (probability × impact grid)

For the thesis defense, these visuals are also what makes the demo look convincing.

#### L9. Missing Evaluator scorecard
Per Claude.md, every level ends with an Evaluator Agent scoring the output on six criteria. This PDF stops after "Lean Entry" — no scorecard appended. The evaluator may have run but the results aren't making it into the exported document.

**Fix:** in the frontend, after the Strategy Consultant's report is complete, append the Evaluator's scorecard as a labeled section ("Evaluator Assessment") before the "Download PDF" action becomes available.

#### L10. Metadata line placement
"Generated: 4/21/2026, 11:52:11 AM" sits awkwardly between the question and the executive summary. Should be in a footer or a small muted header line.

**Fix:** style as `<small class="report-meta">` with muted color, placed either at the very top-right or at the very bottom as part of a proper footer.

---

## Priority order if you only have time to fix some

Ranked by impact on report quality:

1. **Revenue model type (#2)** — single biggest driver of the wrong conclusion
2. **Staffing cost methodology (#1)** — biggest quantitative error
3. **Finding-relevance check in Strategy Consultant (#8)** — the architectural fix that prevents propagation of bad findings
4. **Named competitors required (#7)** — easy win, immediately improves credibility
5. **Three scenarios required (#3)** — schema-level fix, one line of Pydantic
6. **Evaluator scorecard in PDF (L9)** — it's probably already being computed, just needs to be surfaced
7. **Markdown rendering in PDF (L1, L2)** — purely a frontend bug fix, no agent changes needed

Fixes #1, #2, #3, #5, and #7 are all **output schema changes** — you can enforce them by making the orchestrator reject outputs that don't match. That's your strongest constraint mechanism per Claude.md (layer 3: output schemas as hard constraint). Fixes #8 and #10 are prompt changes, which are softer but cheaper to iterate on. Fixes L1–L10 are frontend/export concerns that don't touch agent logic at all.

---

## What this means for the thesis narrative

This is actually a useful result for the progressive-complexity argument. Level 1 failed on fabrication (invented competitors, inconsistent math). Level 4 Hybrid has largely fixed those failure modes — no fake companies, no math contradictions, honest about data gaps — but exposed a **new class of failures**: reasoning errors in the synthesis step (the AI-CAGR non-sequitur) and modeling errors in the specialist step (the revenue-per-client assumption). These are exactly the kinds of issues the Evaluator Agent should catch, and the fact that the EM review loop didn't catch them is itself an interesting data point for Chapter 5.

Framing: *"Higher complexity levels do not simply produce better reports; they trade crude failure modes (fabrication, math contradictions) for subtler ones (reasoning errors in synthesis, modeling errors in specialists). The multi-agent architecture reduces failure magnitude but shifts the required evaluation criteria toward reasoning quality."*
