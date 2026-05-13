"""
Level 4 -- Hybrid Hierarchical Orchestrator.

Pure-Python orchestrator (no LLM) that routes data between five agents
in a sequential pipeline, with Evaluator-only quality gating at the end:

  Flow:
    EM decomposes question
    -> MR researches market
    -> FA analyzes financials  (+ Python schema guard + budget check)
    -> RA assesses risks
    -> SC writes report        (+ REQUEST_REVISION upstream loop)
    -> Evaluator scores        (+ optional SC revision on low scores)

This module implements constraint layer #4 (orchestrator routing).
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
MAX_EVAL_ITERATIONS = 2         # max evaluator rounds (up to 1 revision driven by scorecard)
EVAL_CRITERION_THRESHOLD = 5    # any criterion below this triggers a revision
MAX_SC_UPSTREAM_REVISIONS = 1   # max times SC can request upstream agent re-runs


class PipelineCancelled(Exception):
    pass


def _check_cancel():
    if state.is_cancelled():
        raise PipelineCancelled()


def _save_session(session_id, question, status, revisions, scorecard):
    s = state.get_state()
    agent_outputs = {name: ag.output for name, ag in s.agents.items() if ag.output}
    monitor.save_session(
        session_id=session_id,
        question=question,
        status=status,
        started_at=s.started_at,
        agent_outputs=agent_outputs,
        full_report=s.full_report,
        scorecard=scorecard,
        revisions=revisions,
    )

_REQUEST_REVISION_RE = re.compile(
    r"REQUEST_REVISION:\s*(financial_analyst|market_researcher|risk_analyst)"
    r"\s*--\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
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
    Execute the five-agent pipeline and yield SSE event strings.

    Agents run sequentially without mid-pipeline reviews. The only
    quality gate is the Evaluator at the end, which can trigger one
    SC revision on low scores.

    SSE event types emitted:
      agent_start    -- an agent begins processing
      agent_output   -- a structured agent (JSON) has produced its output
      token          -- a streaming token from the Strategy Consultant
      agent_complete -- a streaming agent finished
      upstream_revision_start -- SC requested an upstream agent re-run
      evaluator_start/complete -- evaluator rounds
      revision_start -- SC begins a revision pass (evaluator-driven)
      error          -- an error occurred
      done           -- the full pipeline is complete
    """

    em = EngagementManager()
    mr = MarketResearcher()
    fa = FinancialAnalyst()
    ra = RiskAnalyst()
    sc = StrategyConsultant()

    agent_configs = [
        {"name": em.name, "display_name": em.display_name, "model": em.model},
        {"name": mr.name, "display_name": mr.display_name, "model": mr.model},
        {"name": fa.name, "display_name": fa.display_name, "model": fa.model},
        {"name": ra.name, "display_name": ra.display_name, "model": ra.model},
        {"name": sc.name, "display_name": sc.display_name, "model": sc.model},
        {"name": "evaluator", "display_name": "Evaluator", "model": "gemini-2.5-flash"},
    ]

    state.reset_state(session_id, question, agent_configs)
    state.clear_cancel()
    revisions: dict[str, dict] = {}

    try:
        yield from _run_pipeline_phases(
            question, session_id, em, mr, fa, ra, sc, revisions,
        )
    except PipelineCancelled:
        state.stop_pipeline()
        _save_session(session_id, question, "stopped", revisions, state.get_state().scorecard)
        yield _sse({"type": "stopped", "session_id": session_id})
        return


def _run_pipeline_phases(
    question: str, session_id: str,
    em, mr, fa, ra, sc,
    revisions: dict[str, dict],
) -> Generator[str, None, None]:
    """Inner generator containing all pipeline phases. Raises PipelineCancelled."""

    scorecard_final: dict | None = None

    # ------------------------------------------------------------------
    # Phase 1: Engagement Manager -- decompose the question
    # ------------------------------------------------------------------
    _check_cancel()
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

    state.update_agent(em.name, "done", elapsed=elapsed, output=plan_data)
    monitor.log_event(session_id, "agent_complete", agent_name=em.display_name, data={"elapsed": elapsed})
    yield _sse({
        "type": "agent_output",
        "agent": em.name,
        "output": plan_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 2: Market Researcher -- investigate the market
    # ------------------------------------------------------------------
    _check_cancel()
    state.update_agent(mr.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": mr.name,
        "display_name": mr.display_name,
        "model": mr.model,
    })
    monitor.log_event(session_id, "agent_start", agent_name=mr.display_name, data={"model": mr.model})

    try:
        t0 = time.time()
        research_data = _run_with_retries(
            lambda: mr.run(question, plan_data)
        )
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(mr.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=mr.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": mr.name, "error": str(exc)})
        return

    state.update_agent(mr.name, "done", elapsed=elapsed, output=research_data)
    monitor.log_event(session_id, "agent_complete", agent_name=mr.display_name, data={"elapsed": elapsed})
    yield _sse({
        "type": "agent_output",
        "agent": mr.name,
        "output": research_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 3: Financial Analyst -- quantitative analysis
    # ------------------------------------------------------------------
    _check_cancel()
    state.update_agent(fa.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": fa.name,
        "display_name": fa.display_name,
        "model": fa.model,
    })
    monitor.log_event(session_id, "agent_start", agent_name=fa.display_name, data={"model": fa.model})

    try:
        t0 = time.time()
        financial_data = _run_with_retries(
            lambda: fa.run(question, plan_data, research_data)
        )
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(fa.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=fa.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": fa.name, "error": str(exc)})
        return

    state.update_agent(fa.name, "done", elapsed=elapsed, output=financial_data)
    monitor.log_event(session_id, "agent_complete", agent_name=fa.display_name, data={"elapsed": elapsed})
    yield _sse({
        "type": "agent_output",
        "agent": fa.name,
        "output": financial_data,
        "elapsed": elapsed,
    })

    # -- Schema-level guard: ensure 3 scenarios + sensitivity analysis --
    fa_gaps = _financial_gaps(financial_data)
    if fa_gaps:
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
            state.update_agent(fa.name, "done", elapsed=elapsed_rev, output=financial_data)
            yield _sse({"type": "agent_output", "agent": fa.name, "output": financial_data, "elapsed": elapsed_rev, "revised": True})
        except Exception as exc:
            monitor.log_event(session_id, "agent_error", agent_name=fa.display_name, data={"error": str(exc), "source": "schema_guard"})

    # -- Budget validation (Python-level, not LLM-dependent) --
    budget_check = _validate_budget(question, financial_data)
    if budget_check:
        fa_inner = financial_data.get('financial_analysis', financial_data)
        fa_inner['budget_validation'] = budget_check
        yield _sse({"type": "budget_validation", **budget_check})
        monitor.log_event(
            session_id, "budget_validation",
            agent_name=fa.display_name,
            data=budget_check,
        )

    # ------------------------------------------------------------------
    # Phase 4: Risk Analyst -- identify and assess risks
    # ------------------------------------------------------------------
    _check_cancel()
    state.update_agent(ra.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": ra.name,
        "display_name": ra.display_name,
        "model": ra.model,
    })
    monitor.log_event(session_id, "agent_start", agent_name=ra.display_name, data={"model": ra.model})

    try:
        t0 = time.time()
        risk_data = _run_with_retries(
            lambda: ra.run(question, plan_data, research_data, financial_data)
        )
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(ra.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=ra.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": ra.name, "error": str(exc)})
        return

    state.update_agent(ra.name, "done", elapsed=elapsed, output=risk_data)
    monitor.log_event(session_id, "agent_complete", agent_name=ra.display_name, data={"elapsed": elapsed})
    yield _sse({
        "type": "agent_output",
        "agent": ra.name,
        "output": risk_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 5: Strategy Consultant -- write the report (streamed)
    # ------------------------------------------------------------------
    _check_cancel()
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
            if len(report_chunks) % 50 == 0:
                state.set_report(strip_think_tags("".join(report_chunks)))
                _check_cancel()
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(sc.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(session_id, "agent_error", agent_name=sc.display_name, data={"error": str(exc)})
        yield _sse({"type": "error", "agent": sc.name, "error": str(exc)})
        return

    full_report = strip_think_tags("".join(report_chunks))
    state.set_report(full_report)

    # -- Intercept REQUEST_REVISION from SC (upstream agent re-run) --
    m = _REQUEST_REVISION_RE.search(full_report[:500])
    if m:
        target_agent_name = m.group(1).strip().lower()
        revision_reason = m.group(2).strip()

        agent_map = {
            "financial_analyst": (fa, lambda fb: fa.run(question, plan_data, research_data, revision_feedback=fb)),
            "market_researcher": (mr, lambda fb: mr.run(question, plan_data, revision_feedback=fb)),
            "risk_analyst": (ra, lambda fb: ra.run(question, plan_data, research_data, financial_data, revision_feedback=fb)),
        }

        if target_agent_name in agent_map:
            target_agent, target_run = agent_map[target_agent_name]
            sc_upstream_revisions = revisions.get(f"sc_upstream_{target_agent_name}", {}).get("revision_count", 0)

            if sc_upstream_revisions < MAX_SC_UPSTREAM_REVISIONS:
                yield _sse({
                    "type": "upstream_revision_start",
                    "agent": target_agent.name,
                    "feedback": revision_reason,
                    "source": "strategy_consultant",
                })
                state.update_agent(target_agent.name, "revising")
                monitor.log_event(
                    session_id, "upstream_revision_start",
                    agent_name=target_agent.display_name,
                    data={"feedback": revision_reason, "source": "strategy_consultant"},
                )

                try:
                    t0_upstream = time.time()
                    revised_output = _run_with_retries(lambda: target_run(revision_reason))
                    elapsed_upstream = round(time.time() - t0_upstream, 2)

                    if target_agent_name == "financial_analyst":
                        financial_data = revised_output
                    elif target_agent_name == "market_researcher":
                        research_data = revised_output
                    elif target_agent_name == "risk_analyst":
                        risk_data = revised_output

                    revisions[f"sc_upstream_{target_agent_name}"] = {
                        "was_revised": True,
                        "feedback": revision_reason,
                        "revision_count": sc_upstream_revisions + 1,
                        "source": "strategy_consultant",
                    }
                    state.update_agent(target_agent.name, "done", elapsed=elapsed_upstream, output=revised_output)
                    yield _sse({
                        "type": "agent_output",
                        "agent": target_agent.name,
                        "output": revised_output,
                        "elapsed": elapsed_upstream,
                        "revised": True,
                        "source": "strategy_consultant",
                    })
                    monitor.log_event(
                        session_id, "upstream_revision_complete",
                        agent_name=target_agent.display_name,
                        data={"elapsed": elapsed_upstream, "source": "strategy_consultant"},
                    )
                except Exception as exc:
                    state.update_agent(target_agent.name, "error", error=str(exc))
                    monitor.log_event(
                        session_id, "agent_error",
                        agent_name=target_agent.display_name,
                        data={"error": str(exc), "source": "sc_upstream_revision"},
                    )
                    yield _sse({"type": "error", "agent": target_agent.name, "error": str(exc)})

                # Re-run SC with corrected upstream data
                _check_cancel()
                yield _sse({"type": "agent_start", "agent": sc.name, "display_name": sc.display_name, "model": sc.model, "rerun": True})
                state.update_agent(sc.name, "working")
                monitor.log_event(session_id, "agent_start", agent_name=sc.display_name, data={"model": sc.model, "rerun": True})

                report_chunks = []
                try:
                    t0_rerun = time.time()
                    for token in sc.run(question, plan_data, research_data, financial_data, risk_data):
                        report_chunks.append(token)
                        yield _sse({"type": "token", "content": token})
                        if len(report_chunks) % 50 == 0:
                            state.set_report(strip_think_tags("".join(report_chunks)))
                            _check_cancel()
                    elapsed = round(time.time() - t0_rerun, 2)
                except Exception as exc:
                    state.update_agent(sc.name, "error", error=str(exc))
                    state.fail_pipeline(str(exc))
                    monitor.log_event(session_id, "agent_error", agent_name=sc.display_name, data={"error": str(exc)})
                    yield _sse({"type": "error", "agent": sc.name, "error": str(exc)})
                    return

                full_report = strip_think_tags("".join(report_chunks))
                state.set_report(full_report)

    state.update_agent(sc.name, "done", elapsed=elapsed)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=sc.display_name,
        data={"elapsed": elapsed, "output_length": len(full_report)},
    )
    yield _sse({"type": "agent_complete", "agent": sc.name, "elapsed": elapsed})

    # ------------------------------------------------------------------
    # Phase 6: Evaluator (Gemini 2.5) -- score, optionally trigger SC revision
    # ------------------------------------------------------------------
    total_eval_elapsed = 0.0
    for round_idx in range(MAX_EVAL_ITERATIONS):
        _check_cancel()
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

        _check_cancel()
        report_chunks = []
        try:
            t0_rev = time.time()
            for token in sc.run(
                question, plan_data, research_data, financial_data, risk_data,
                revision_feedback=feedback,
            ):
                report_chunks.append(token)
                yield _sse({"type": "token", "content": token})
                if len(report_chunks) % 50 == 0:
                    state.set_report(strip_think_tags("".join(report_chunks)))
                    _check_cancel()
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
        state.set_report(full_report)
        state.update_agent(sc.name, "done", elapsed=elapsed)
        monitor.log_event(
            session_id, "revision_complete",
            agent_name=sc.display_name,
            data={"elapsed": elapsed_rev, "source": "evaluator", "round": round_num},
        )

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    state.finish_pipeline(revisions)
    _save_session(session_id, question, "done", revisions, scorecard_final)
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
