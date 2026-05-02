"""
Level 4 -- Hybrid Hierarchical Orchestrator.

Pure-Python orchestrator (no LLM) that routes data between five agents
in a hierarchical hybrid structure:

  Engagement Manager decomposes the question, then REVIEWS every
  intermediate output before it moves forward. If an output fails
  review, the agent is asked to revise (max 1 revision per agent).

  Flow:
    EM decomposes question
    -> MR researches market    -> EM reviews -> (revision?) -> approved
    -> FA analyzes financials  -> EM reviews -> (revision?) -> approved
    -> RA assesses risks       -> EM reviews -> (revision?) -> approved
    -> SC writes report        -> EM reviews -> (revision?) -> approved

Note: The Evaluator is a separate application (not part of this pipeline).

This module implements constraint layer #4 (orchestrator routing) plus
the hierarchical review loop that makes Level 4 distinct from Level 3.
"""

import json
import os
import re
import time
from typing import Generator

import requests

from agents import (
    EngagementManager,
    MarketResearcher,
    FinancialAnalyst,
    RiskAnalyst,
    StrategyConsultant,
    strip_think_tags,
)
import monitor
import state

MAX_RETRIES = 2                 # retry on JSON parse failure
MAX_REVISIONS = 1               # max EM-requested revisions per agent
MAX_EVAL_ITERATIONS = 2         # max evaluator rounds (up to 1 revision driven by scorecard)
EVAL_CRITERION_THRESHOLD = 5    # any criterion below this triggers a revision
EVALUATOR_URL = os.environ.get("EVALUATOR_URL", "http://host.docker.internal:8005")
EVALUATOR_TIMEOUT = 180         # seconds


def _call_evaluator(question: str, report: str, session_id: str) -> dict:
    """POST to the Gemini 2.5 evaluator service and return the scorecard dict."""
    response = requests.post(
        f"{EVALUATOR_URL}/evaluate",
        json={"question": question, "report": report, "level": 4, "session_id": session_id},
        timeout=EVALUATOR_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("scorecard", {})


def _failing_criteria(scorecard: dict) -> list[tuple[str, int, str]]:
    """Return [(criterion, score, justification), ...] for criteria below threshold."""
    evaluation = scorecard.get("evaluation", {}) or {}
    failing = []
    for name, data in evaluation.items():
        if not isinstance(data, dict):
            continue
        score = data.get("score")
        if isinstance(score, (int, float)) and score < EVAL_CRITERION_THRESHOLD:
            failing.append((name, int(score), data.get("justification", "")))
    return failing


def _build_evaluator_feedback(scorecard: dict, failing: list[tuple[str, int, str]]) -> str:
    """Build a focused feedback string for the Strategy Consultant's revision."""
    lines = ["The independent Evaluator scored the report below acceptable quality on the following criteria:"]
    for name, score, justification in failing:
        pretty = name.replace("_", " ").title()
        lines.append(f"\n- {pretty} ({score}/10): {justification}")
    issues = scorecard.get("critical_issues") or []
    if issues:
        lines.append("\n\nCritical issues identified:")
        for issue in issues:
            lines.append(f"- {issue}")
    lines.append(
        "\n\nRewrite the report to address every point above. Keep every claim grounded "
        "in the market research, financial analysis, and risk assessment already provided. "
        "Do not introduce new facts."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Budget validation helpers (Python-level, not LLM-dependent)
# ---------------------------------------------------------------------------

def _extract_budget(question: str) -> float | None:
    """Try to extract a budget amount from the client question text."""
    patterns = [
        r'budget[^€$£]*?[€$£]\s*([\d,.]+)\s*([KkMm])?',
        r'[€$£]\s*([\d,.]+)\s*([KkMm])?\s*budget',
        r'budget[^€$£]*?([\d,.]+)\s*([KkMm])?\s*(?:EUR|USD|euros?|dollars?)',
        r'[€$£]\s*([\d,.]+)\s*([KkMm])?(?:\s|,|\.|\b)',
    ]
    for pattern in patterns:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            num_str = m.group(1).replace(',', '').replace('.', '', m.group(1).count('.') - 1) if '.' in m.group(1) else m.group(1).replace(',', '')
            try:
                value = float(num_str)
                suffix = m.group(2)
                if suffix and suffix.upper() == 'K':
                    value *= 1_000
                elif suffix and suffix.upper() == 'M':
                    value *= 1_000_000
                if value > 0:
                    return value
            except (ValueError, IndexError):
                continue
    return None


def _parse_amount(amount_str) -> float:
    """Parse a currency string like '€50,000' or '$120K' into a float. Accepts int/float directly."""
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    if not isinstance(amount_str, str):
        return 0.0
    cleaned = re.sub(r'[€$£\s]', '', amount_str)
    m = re.match(r'([\d,.]+)\s*([KkMm])?', cleaned)
    if not m:
        return 0.0
    num_str = m.group(1).replace(',', '')
    try:
        val = float(num_str)
    except ValueError:
        return 0.0
    suffix = m.group(2)
    if suffix and suffix.upper() == 'K':
        val *= 1_000
    elif suffix and suffix.upper() == 'M':
        val *= 1_000_000
    return val


def _validate_budget(question: str, financial_data: dict) -> dict | None:
    """Compare FA cost estimates against the stated budget. Returns warning or None."""
    budget = _extract_budget(question)
    if not budget:
        return None

    fa = financial_data.get('financial_analysis', financial_data)
    costs = fa.get('cost_estimates', [])
    if not costs:
        return None

    total = sum(_parse_amount(c.get('amount', '0')) for c in costs)
    if total <= 0:
        return None

    if total > budget:
        return {
            "budget_exceeded": True,
            "stated_budget": budget,
            "estimated_total_cost": round(total, 2),
            "overrun": round(total - budget, 2),
            "overrun_pct": round((total - budget) / budget * 100, 1),
            "warning": (
                f"Total estimated costs (€{total:,.0f}) exceed the stated "
                f"budget (€{budget:,.0f}) by €{total - budget:,.0f} "
                f"({round((total - budget) / budget * 100, 1)}% over budget)"
            ),
        }
    else:
        return {
            "budget_exceeded": False,
            "stated_budget": budget,
            "estimated_total_cost": round(total, 2),
            "remaining": round(budget - total, 2),
            "info": (
                f"Total estimated costs (€{total:,.0f}) are within the "
                f"stated budget (€{budget:,.0f}). "
                f"€{budget - total:,.0f} remaining."
            ),
        }


REQUIRED_SCENARIOS = {"conservative", "moderate", "aggressive"}


def _financial_gaps(financial_data: dict) -> str:
    """Return a feedback string listing missing required FA content, or '' if OK."""
    fa = financial_data.get("financial_analysis", financial_data) or {}
    gaps: list[str] = []

    projections = fa.get("revenue_projections") or []
    scenarios_present = {
        str(p.get("scenario", "")).strip().lower() for p in projections if isinstance(p, dict)
    }
    missing = REQUIRED_SCENARIOS - scenarios_present
    if missing:
        gaps.append(
            "revenue_projections must contain exactly three scenarios "
            "(Conservative, Moderate, Aggressive). Missing: "
            + ", ".join(sorted(missing))
        )

    if not str(fa.get("sensitivity_analysis", "")).strip():
        gaps.append("sensitivity_analysis is empty -- required field.")

    if not str(fa.get("revenue_model_type", "")).strip():
        gaps.append(
            "revenue_model_type is missing. Choose one_time, recurring_contract, "
            "or hybrid based on the client context and justify it."
        )

    return "\n- ".join(["Schema-level gaps detected:"] + gaps) if gaps else ""


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _run_with_retries(fn, *args, **kwargs):
    """Run a function with retries on parse errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ValueError, Exception) as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                continue
    raise last_err


def run_pipeline(question: str, session_id: str) -> Generator[str, None, None]:
    """
    Execute the five-agent hybrid pipeline and yield SSE event strings.

    SSE event types emitted:
      agent_start    -- an agent begins processing
      agent_output   -- a structured agent (JSON) has produced its output
      review_start   -- EM begins reviewing an agent's output
      review_result  -- EM review verdict (approved/revision needed)
      revision_start -- an agent begins a revision pass
      token          -- a streaming token from the Strategy Consultant
      agent_complete -- a streaming agent finished
      error          -- an error occurred
      done           -- the full pipeline is complete
    """

    em = EngagementManager()
    mr = MarketResearcher()
    fa = FinancialAnalyst()
    ra = RiskAnalyst()
    sc = StrategyConsultant()

    # Initialize pipeline state for the dashboard
    agent_configs = [
        {"name": em.name, "display_name": em.display_name, "model": em.model},
        {"name": mr.name, "display_name": mr.display_name, "model": mr.model},
        {"name": fa.name, "display_name": fa.display_name, "model": fa.model},
        {"name": ra.name, "display_name": ra.display_name, "model": ra.model},
        {"name": sc.name, "display_name": sc.display_name, "model": sc.model},
        {"name": "evaluator", "display_name": "Evaluator", "model": "gemini-2.5-flash"},
    ]

    state.reset_state(session_id, question, agent_configs)

    # Track revisions for the dashboard
    revisions: dict[str, dict] = {}

    # We use a mutable container to capture output from generators,
    # since generators can't return values via `yield from`.
    class OutputCapture:
        value = None

    def run_agent_with_review(agent, run_fn, extra_review_context=""):
        """Wrapper that yields SSE events and captures the agent output."""
        capture = OutputCapture()

        def _inner():
            state.update_agent(agent.name, "working")
            yield _sse({
                "type": "agent_start",
                "agent": agent.name,
                "display_name": agent.display_name,
                "model": agent.model,
            })
            monitor.log_event(
                session_id, "agent_start",
                agent_name=agent.display_name,
                data={"model": agent.model},
            )

            # First run
            t0 = time.time()
            output = _run_with_retries(run_fn)
            elapsed = round(time.time() - t0, 2)

            state.update_agent(agent.name, "done", elapsed=elapsed, output=output)
            monitor.log_event(
                session_id, "agent_complete",
                agent_name=agent.display_name,
                data={"elapsed": elapsed},
            )
            yield _sse({
                "type": "agent_output",
                "agent": agent.name,
                "output": output,
                "elapsed": elapsed,
            })

            # EM Review
            yield _sse({
                "type": "review_start",
                "agent": agent.name,
                "reviewer": em.display_name,
            })
            state.update_agent(agent.name, "reviewing")
            monitor.log_event(
                session_id, "review_start",
                agent_name=agent.display_name,
                data={"reviewer": em.display_name},
            )

            review = em.review_output(
                agent_name=agent.display_name,
                question=question,
                analysis_plan=plan_data,
                agent_output=output,
                extra_context=extra_review_context,
            )

            yield _sse({
                "type": "review_result",
                "agent": agent.name,
                "approved": review.get("approved", False),
                "feedback": review.get("feedback", ""),
            })
            monitor.log_event(
                session_id, "review_result",
                agent_name=agent.display_name,
                data={
                    "approved": review.get("approved"),
                    "feedback": review.get("feedback", ""),
                },
            )

            revisions[agent.name] = {
                "was_revised": False,
                "feedback": review.get("feedback", ""),
                "revision_count": 0,
            }

            # Revision if not approved
            if not review.get("approved", True) and review.get("feedback"):
                revisions[agent.name]["was_revised"] = True
                revisions[agent.name]["revision_count"] = 1

                yield _sse({
                    "type": "revision_start",
                    "agent": agent.name,
                    "feedback": review["feedback"],
                })
                state.update_agent(agent.name, "revising")
                monitor.log_event(
                    session_id, "revision_start",
                    agent_name=agent.display_name,
                    data={"feedback": review["feedback"]},
                )

                t0_rev = time.time()
                output = _run_with_retries(
                    lambda revision_feedback=review["feedback"]: run_fn(revision_feedback=revision_feedback)
                )
                elapsed_rev = round(time.time() - t0_rev, 2)

                state.update_agent(
                    agent.name, "done",
                    elapsed=elapsed + elapsed_rev, output=output,
                )
                monitor.log_event(
                    session_id, "revision_complete",
                    agent_name=agent.display_name,
                    data={"elapsed": elapsed_rev},
                )
                yield _sse({
                    "type": "agent_output",
                    "agent": agent.name,
                    "output": output,
                    "elapsed": elapsed + elapsed_rev,
                    "revised": True,
                })

            state.update_agent(agent.name, "approved")
            capture.value = output

        return _inner, capture

    # ------------------------------------------------------------------
    # Phase 1: Engagement Manager -- decompose the question
    # ------------------------------------------------------------------
    state.update_agent(em.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": em.name,
        "display_name": em.display_name,
        "model": em.model,
    })
    monitor.log_event(
        session_id, "agent_start",
        agent_name=em.display_name,
        data={"model": em.model},
    )

    try:
        t0 = time.time()
        plan_data = _run_with_retries(em.run, question)
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(em.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=em.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": em.name, "error": str(exc)})
        return

    state.update_agent(em.name, "approved", elapsed=elapsed, output=plan_data)
    monitor.log_event(session_id, "agent_complete", agent_name=em.display_name, data={"elapsed": elapsed})
    yield _sse({
        "type": "agent_output",
        "agent": em.name,
        "output": plan_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 2: Market Researcher -- investigate the market (with EM review)
    # ------------------------------------------------------------------
    try:
        gen, capture = run_agent_with_review(
            mr,
            lambda revision_feedback=None: mr.run(question, plan_data, revision_feedback=revision_feedback),
        )
        yield from gen()
        research_data = capture.value
        if research_data is None:
            raise ValueError("Market Researcher produced no output")
    except Exception as exc:
        state.update_agent(mr.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=mr.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": mr.name, "error": str(exc)})
        return

    # ------------------------------------------------------------------
    # Phase 3: Financial Analyst -- quantitative analysis (with EM review)
    # ------------------------------------------------------------------
    try:
        gen, capture = run_agent_with_review(
            fa,
            lambda revision_feedback=None: fa.run(
                question, plan_data, research_data, revision_feedback=revision_feedback
            ),
            extra_review_context=(
                "CROSS-CHECK: Verify that the Financial Analyst's numbers are "
                "traceable to the Market Research data. Flag any numbers that "
                "appear invented or not sourced from the market research."
            ),
        )
        yield from gen()
        financial_data = capture.value
        if financial_data is None:
            raise ValueError("Financial Analyst produced no output")
    except Exception as exc:
        state.update_agent(fa.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=fa.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": fa.name, "error": str(exc)})
        return

    # -- Schema-level guard: ensure 3 scenarios + sensitivity analysis --
    # If the FA omitted content the SC is required to cite, trigger a
    # one-shot Python-driven revision with concrete feedback. This acts
    # as a safety net behind the EM review.
    fa_gaps = _financial_gaps(financial_data)
    if fa_gaps and revisions.get(fa.name, {}).get("revision_count", 0) == 0:
        yield _sse({"type": "revision_start", "agent": fa.name, "feedback": fa_gaps, "source": "schema_guard"})
        state.update_agent(fa.name, "revising")
        monitor.log_event(session_id, "revision_start", agent_name=fa.display_name, data={"feedback": fa_gaps, "source": "schema_guard"})
        try:
            t0_rev = time.time()
            financial_data = _run_with_retries(
                lambda: fa.run(question, plan_data, research_data, revision_feedback=fa_gaps)
            )
            elapsed_rev = round(time.time() - t0_rev, 2)
            revisions[fa.name] = {"was_revised": True, "feedback": fa_gaps, "revision_count": 1, "source": "schema_guard"}
            state.update_agent(fa.name, "approved", elapsed=elapsed_rev, output=financial_data)
            yield _sse({"type": "agent_output", "agent": fa.name, "output": financial_data, "elapsed": elapsed_rev, "revised": True})
        except Exception as exc:
            monitor.log_event(session_id, "agent_error", agent_name=fa.display_name, data={"error": str(exc), "source": "schema_guard"})

    # -- Budget validation (Python-level, not LLM-dependent) --
    budget_check = _validate_budget(question, financial_data)
    if budget_check:
        # Inject into financial data so downstream agents (RA, SC) see it
        fa_inner = financial_data.get('financial_analysis', financial_data)
        fa_inner['budget_validation'] = budget_check
        yield _sse({"type": "budget_validation", **budget_check})
        monitor.log_event(
            session_id, "budget_validation",
            agent_name=fa.display_name,
            data=budget_check,
        )

    # ------------------------------------------------------------------
    # Phase 4: Risk Analyst -- identify and assess risks (with EM review)
    # ------------------------------------------------------------------
    try:
        gen, capture = run_agent_with_review(
            ra,
            lambda revision_feedback=None: ra.run(
                question, plan_data, research_data, financial_data, revision_feedback=revision_feedback
            ),
            extra_review_context=(
                "CROSS-CHECK: Verify that the Risk Analyst's identified risks "
                "are supported by either web research or the market/financial data. "
                "Flag any risks that appear fabricated or unsupported."
            ),
        )
        yield from gen()
        risk_data = capture.value
        if risk_data is None:
            raise ValueError("Risk Analyst produced no output")
    except Exception as exc:
        state.update_agent(ra.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=ra.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": ra.name, "error": str(exc)})
        return

    # ------------------------------------------------------------------
    # Phase 5: Strategy Consultant -- write the report (streamed, with EM review)
    # ------------------------------------------------------------------
    state.update_agent(sc.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": sc.name,
        "display_name": sc.display_name,
        "model": sc.model,
    })
    monitor.log_event(session_id, "agent_start", agent_name=sc.display_name, data={"model": sc.model})

    report_chunks: list[str] = []
    try:
        t0 = time.time()
        for token in sc.run(question, plan_data, research_data, financial_data, risk_data):
            report_chunks.append(token)
            yield _sse({"type": "token", "content": token})
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(sc.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=sc.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": sc.name, "error": str(exc)})
        return

    full_report = strip_think_tags("".join(report_chunks))

    # EM reviews the Strategy Consultant report
    yield _sse({"type": "review_start", "agent": sc.name, "reviewer": em.display_name})
    state.update_agent(sc.name, "reviewing")
    monitor.log_event(session_id, "review_start", agent_name=sc.display_name, data={"reviewer": em.display_name})

    review = em.review_output(
        agent_name=sc.display_name,
        question=question,
        analysis_plan=plan_data,
        agent_output=full_report[:3000],  # truncate for review context
        extra_context=(
            "FACT-CHECK: Verify that the Strategy Consultant's report does NOT "
            "introduce any new claims, statistics, or facts that are not present "
            "in the market research, financial analysis, or risk assessment. "
            "Flag any unsupported claims."
        ),
    )

    yield _sse({
        "type": "review_result",
        "agent": sc.name,
        "approved": review.get("approved", False),
        "feedback": review.get("feedback", ""),
    })
    monitor.log_event(
        session_id, "review_result",
        agent_name=sc.display_name,
        data={"approved": review.get("approved"), "feedback": review.get("feedback", "")},
    )

    revisions[sc.name] = {
        "was_revised": False,
        "feedback": review.get("feedback", ""),
        "revision_count": 0,
    }

    # Revision if not approved
    if not review.get("approved", True) and review.get("feedback"):
        revisions[sc.name]["was_revised"] = True
        revisions[sc.name]["revision_count"] = 1

        yield _sse({"type": "revision_start", "agent": sc.name, "feedback": review["feedback"]})
        state.update_agent(sc.name, "revising")
        monitor.log_event(session_id, "revision_start", agent_name=sc.display_name, data={"feedback": review["feedback"]})

        # Re-stream the revised report
        report_chunks = []
        try:
            t0_rev = time.time()
            for token in sc.run(
                question, plan_data, research_data, financial_data, risk_data,
                revision_feedback=review["feedback"],
            ):
                report_chunks.append(token)
                yield _sse({"type": "token", "content": token})
            elapsed_rev = round(time.time() - t0_rev, 2)
            elapsed += elapsed_rev
        except Exception as exc:
            state.update_agent(sc.name, "error", error=str(exc))
            state.fail_pipeline(str(exc))
            monitor.log_event(session_id, "agent_error", agent_name=sc.display_name, data={"error": str(exc)})
            yield _sse({"type": "error", "agent": sc.name, "error": str(exc)})
            return

        full_report = strip_think_tags("".join(report_chunks))
        monitor.log_event(session_id, "revision_complete", agent_name=sc.display_name, data={"elapsed": elapsed_rev})

    state.update_agent(sc.name, "approved", elapsed=elapsed)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=sc.display_name,
        data={"elapsed": elapsed, "output_length": len(full_report)},
    )
    yield _sse({"type": "agent_complete", "agent": sc.name, "elapsed": elapsed})

    # ------------------------------------------------------------------
    # Phase 6: Evaluator (Gemini 2.5) -- score, optionally trigger SC revision
    # ------------------------------------------------------------------
    scorecard_final: dict | None = None
    total_eval_elapsed = 0.0
    for round_idx in range(MAX_EVAL_ITERATIONS):
        round_num = round_idx + 1
        state.update_agent("evaluator", "working")
        yield _sse({
            "type": "evaluator_start",
            "agent": "evaluator",
            "display_name": "Evaluator (Gemini 2.5)",
            "round": round_num,
        })
        monitor.log_event(
            session_id, "evaluator_start",
            agent_name="Evaluator",
            data={"round": round_num, "model": "gemini-2.5-flash"},
        )

        t0 = time.time()
        try:
            scorecard = _call_evaluator(question, full_report, session_id)
        except Exception as exc:
            state.update_agent("evaluator", "error", error=str(exc))
            monitor.log_event(
                session_id, "evaluator_error",
                agent_name="Evaluator",
                data={"error": str(exc), "round": round_num},
            )
            yield _sse({"type": "evaluator_error", "error": str(exc), "round": round_num})
            break

        eval_elapsed = round(time.time() - t0, 2)
        total_eval_elapsed += eval_elapsed
        scorecard_final = scorecard

        state.update_agent(
            "evaluator", "done",
            elapsed=round(total_eval_elapsed, 2),
            output=scorecard,
        )
        state.set_scorecard(scorecard, round_num)

        yield _sse({
            "type": "evaluator_complete",
            "agent": "evaluator",
            "scorecard": scorecard,
            "elapsed": eval_elapsed,
            "round": round_num,
        })
        monitor.log_event(
            session_id, "evaluator_complete",
            agent_name="Evaluator",
            data={
                "elapsed": eval_elapsed,
                "overall_score": scorecard.get("overall_score"),
                "round": round_num,
            },
        )

        failing = _failing_criteria(scorecard)
        if not failing or round_num >= MAX_EVAL_ITERATIONS:
            break

        feedback = _build_evaluator_feedback(scorecard, failing)
        revisions[sc.name] = {
            "was_revised": True,
            "feedback": feedback,
            "revision_count": revisions.get(sc.name, {}).get("revision_count", 0) + 1,
            "source": "evaluator",
        }

        yield _sse({
            "type": "revision_start",
            "agent": sc.name,
            "feedback": feedback,
            "source": "evaluator",
            "round": round_num,
        })
        state.update_agent(sc.name, "revising")
        monitor.log_event(
            session_id, "revision_start",
            agent_name=sc.display_name,
            data={"feedback": feedback, "source": "evaluator", "round": round_num},
        )

        report_chunks = []
        try:
            t0_rev = time.time()
            for token in sc.run(
                question, plan_data, research_data, financial_data, risk_data,
                revision_feedback=feedback,
            ):
                report_chunks.append(token)
                yield _sse({"type": "token", "content": token})
            elapsed_rev = round(time.time() - t0_rev, 2)
            elapsed += elapsed_rev
        except Exception as exc:
            state.update_agent(sc.name, "error", error=str(exc))
            monitor.log_event(
                session_id, "agent_error",
                agent_name=sc.display_name,
                data={"error": str(exc), "source": "evaluator_revision"},
            )
            yield _sse({"type": "error", "agent": sc.name, "error": str(exc)})
            break

        full_report = strip_think_tags("".join(report_chunks))
        state.update_agent(sc.name, "approved", elapsed=elapsed)
        monitor.log_event(
            session_id, "revision_complete",
            agent_name=sc.display_name,
            data={"elapsed": elapsed_rev, "source": "evaluator", "round": round_num},
        )

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    state.finish_pipeline(revisions)
    monitor.log_event(
        session_id, "session_complete",
        data={"revisions": revisions, "final_scorecard": scorecard_final},
    )
    yield _sse({
        "type": "done",
        "session_id": session_id,
        "revisions": revisions,
        "scorecard": scorecard_final,
    })
