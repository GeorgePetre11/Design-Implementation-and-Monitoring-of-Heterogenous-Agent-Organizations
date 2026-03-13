"""
Level 3 — Sequential Pipeline Orchestrator.

Pure-Python orchestrator (no LLM) that routes data between six agents
in a fixed sequence:

  Engagement Manager → Market Researcher → Financial Analyst
    → Risk Analyst → Strategy Consultant

(Evaluator is not yet implemented.)

This module implements constraint layer #4 (orchestrator routing):
each agent only receives the data it is allowed to see.
"""

import json
import time
from typing import Generator

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

MAX_RETRIES = 2  # retry once on JSON parse failure


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def run_pipeline(question: str, session_id: str) -> Generator[str, None, None]:
    """
    Execute the five-agent pipeline and yield SSE event strings.

    SSE event types emitted:
      agent_start   — an agent begins processing
      agent_output  — a structured agent (JSON) has produced its output
      token         — a streaming token from the Strategy Consultant
      agent_complete— a streaming agent finished (no payload)
      error         — an error occurred in an agent
      done          — the full pipeline is complete
    """

    em = EngagementManager()
    mr = MarketResearcher()
    fa = FinancialAnalyst()
    ra = RiskAnalyst()
    sc = StrategyConsultant()

    # Initialize pipeline state for the dashboard
    state.reset_state(session_id, question, [
        {"name": em.name, "display_name": em.display_name, "model": em.model},
        {"name": mr.name, "display_name": mr.display_name, "model": mr.model},
        {"name": fa.name, "display_name": fa.display_name, "model": fa.model},
        {"name": ra.name, "display_name": ra.display_name, "model": ra.model},
        {"name": sc.name, "display_name": sc.display_name, "model": sc.model},
    ])

    # ------------------------------------------------------------------
    # Phase 1: Engagement Manager — decompose the question
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
        plan_data = None
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                plan_data = em.run(question)
                break
            except (ValueError, Exception) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    continue
        if plan_data is None:
            raise last_err
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(em.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(
            session_id, "agent_error",
            agent_name=em.display_name,
            data={"error": str(exc)},
        )
        yield _sse({"type": "error", "agent": em.name, "error": str(exc)})
        return

    state.update_agent(em.name, "done", elapsed=elapsed, output=plan_data)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=em.display_name,
        data={"elapsed": elapsed},
    )
    yield _sse({
        "type": "agent_output",
        "agent": em.name,
        "output": plan_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 2: Market Researcher — investigate the market
    # ------------------------------------------------------------------
    state.update_agent(mr.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": mr.name,
        "display_name": mr.display_name,
        "model": mr.model,
    })
    monitor.log_event(
        session_id, "agent_start",
        agent_name=mr.display_name,
        data={"model": mr.model},
    )

    try:
        t0 = time.time()
        # Orchestrator routing: MR receives only question + analysis plan
        research_data = None
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                research_data = mr.run(question, plan_data)
                break
            except (ValueError, Exception) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    continue
        if research_data is None:
            raise last_err
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(mr.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(
            session_id, "agent_error",
            agent_name=mr.display_name,
            data={"error": str(exc)},
        )
        yield _sse({"type": "error", "agent": mr.name, "error": str(exc)})
        return

    state.update_agent(mr.name, "done", elapsed=elapsed, output=research_data)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=mr.display_name,
        data={"elapsed": elapsed},
    )
    yield _sse({
        "type": "agent_output",
        "agent": mr.name,
        "output": research_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 3: Financial Analyst — quantitative analysis
    # ------------------------------------------------------------------
    state.update_agent(fa.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": fa.name,
        "display_name": fa.display_name,
        "model": fa.model,
    })
    monitor.log_event(
        session_id, "agent_start",
        agent_name=fa.display_name,
        data={"model": fa.model},
    )

    try:
        t0 = time.time()
        # Orchestrator routing: FA receives question + plan + market research
        financial_data = None
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                financial_data = fa.run(question, plan_data, research_data)
                break
            except (ValueError, Exception) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    continue
        if financial_data is None:
            raise last_err
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(fa.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(
            session_id, "agent_error",
            agent_name=fa.display_name,
            data={"error": str(exc)},
        )
        yield _sse({"type": "error", "agent": fa.name, "error": str(exc)})
        return

    state.update_agent(fa.name, "done", elapsed=elapsed, output=financial_data)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=fa.display_name,
        data={"elapsed": elapsed},
    )
    yield _sse({
        "type": "agent_output",
        "agent": fa.name,
        "output": financial_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 4: Risk Analyst — identify and assess risks
    # ------------------------------------------------------------------
    state.update_agent(ra.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": ra.name,
        "display_name": ra.display_name,
        "model": ra.model,
    })
    monitor.log_event(
        session_id, "agent_start",
        agent_name=ra.display_name,
        data={"model": ra.model},
    )

    try:
        t0 = time.time()
        # Orchestrator routing: RA receives question + plan + market + financial
        risk_data = None
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                risk_data = ra.run(question, plan_data, research_data, financial_data)
                break
            except (ValueError, Exception) as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    continue
        if risk_data is None:
            raise last_err
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(ra.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(
            session_id, "agent_error",
            agent_name=ra.display_name,
            data={"error": str(exc)},
        )
        yield _sse({"type": "error", "agent": ra.name, "error": str(exc)})
        return

    state.update_agent(ra.name, "done", elapsed=elapsed, output=risk_data)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=ra.display_name,
        data={"elapsed": elapsed},
    )
    yield _sse({
        "type": "agent_output",
        "agent": ra.name,
        "output": risk_data,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Phase 5: Strategy Consultant — write the report (streamed)
    # ------------------------------------------------------------------
    state.update_agent(sc.name, "working")
    yield _sse({
        "type": "agent_start",
        "agent": sc.name,
        "display_name": sc.display_name,
        "model": sc.model,
    })
    monitor.log_event(
        session_id, "agent_start",
        agent_name=sc.display_name,
        data={"model": sc.model},
    )

    report_chunks: list[str] = []
    try:
        t0 = time.time()
        # Orchestrator routing: SC receives everything
        for token in sc.run(question, plan_data, research_data, financial_data, risk_data):
            report_chunks.append(token)
            yield _sse({"type": "token", "content": token})
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        state.update_agent(sc.name, "error", error=str(exc))
        state.fail_pipeline(str(exc))
        monitor.log_event(
            session_id, "agent_error",
            agent_name=sc.display_name,
            data={"error": str(exc)},
        )
        yield _sse({"type": "error", "agent": sc.name, "error": str(exc)})
        return

    full_report = strip_think_tags("".join(report_chunks))
    state.update_agent(sc.name, "done", elapsed=elapsed)
    monitor.log_event(
        session_id, "agent_complete",
        agent_name=sc.display_name,
        data={"elapsed": elapsed, "output_length": len(full_report)},
    )
    yield _sse({
        "type": "agent_complete",
        "agent": sc.name,
        "elapsed": elapsed,
    })

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    state.finish_pipeline()
    monitor.log_event(session_id, "session_complete")
    yield _sse({"type": "done", "session_id": session_id})
