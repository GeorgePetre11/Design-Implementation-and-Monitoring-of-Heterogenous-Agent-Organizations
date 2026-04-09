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

(The Evaluator runs in a separate standalone app powered by Kimi 2.5
and is not invoked from this orchestrator.)

This module implements constraint layer #4 (orchestrator routing) plus
the hierarchical review loop that makes Level 4 distinct from Level 3.
"""

import json
import os
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

MAX_RETRIES = 2       # retry on JSON parse failure
MAX_REVISIONS = 1     # max EM-requested revisions per agent

# Standalone evaluator service (Kimi K2.5 judge). Optional -- if unreachable,
# the pipeline still succeeds and simply emits no evaluation event.
EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://host.docker.internal:8005")
EVALUATOR_TIMEOUT = int(os.getenv("EVALUATOR_TIMEOUT", "180"))


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
    ]

    state.reset_state(session_id, question, agent_configs)

    # Track revisions for the dashboard
    revisions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Helper: run an agent step with EM review + optional revision
    # ------------------------------------------------------------------
    def _agent_step_with_review(
        agent,
        run_fn,
        extra_review_context: str = "",
    ):
        """Run an agent, have EM review, revise if needed. Returns the output."""
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
            data={"approved": review.get("approved"), "feedback": review.get("feedback", "")},
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

            t0 = time.time()
            output = _run_with_retries(run_fn, revision_feedback=review["feedback"])
            elapsed_rev = round(time.time() - t0, 2)

            state.update_agent(agent.name, "done", elapsed=elapsed + elapsed_rev, output=output)
            monitor.log_event(
                session_id, "revision_complete",
                agent_name=agent.display_name,
                data={"elapsed": elapsed_rev, "total_elapsed": elapsed + elapsed_rev},
            )
            yield _sse({
                "type": "agent_output",
                "agent": agent.name,
                "output": output,
                "elapsed": elapsed + elapsed_rev,
                "revised": True,
            })

        state.update_agent(agent.name, "approved")
        return output

    # We need to use a different pattern since generators can't return values.
    # Use a mutable container to capture the output.
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
    # Phase 6: External Evaluator (Kimi K2.5, standalone service)
    # ------------------------------------------------------------------
    yield _sse({"type": "evaluation_start"})
    monitor.log_event(session_id, "evaluator_request", data={"url": EVALUATOR_URL})

    try:
        t0 = time.time()
        resp = requests.post(
            f"{EVALUATOR_URL}/evaluate",
            json={
                "question": question,
                "report": full_report,
                "session_id": session_id,
                "level": 4,
            },
            timeout=EVALUATOR_TIMEOUT,
        )
        resp.raise_for_status()
        eval_payload = resp.json()
        eval_elapsed = round(time.time() - t0, 2)

        monitor.log_event(
            session_id, "evaluator_complete",
            data={
                "overall_score": eval_payload.get("scorecard", {}).get("overall_score"),
                "elapsed": eval_elapsed,
            },
        )
        yield _sse({
            "type": "evaluation",
            "scorecard": eval_payload.get("scorecard"),
            "model": eval_payload.get("model"),
            "elapsed": eval_elapsed,
        })
    except requests.RequestException as exc:
        # Evaluator is optional -- log + emit a soft error, don't fail the pipeline.
        monitor.log_event(session_id, "evaluator_error", data={"error": str(exc)})
        yield _sse({"type": "evaluation_error", "error": str(exc)})

    # ------------------------------------------------------------------
    # Pipeline complete
    # ------------------------------------------------------------------
    state.finish_pipeline(revisions)
    monitor.log_event(session_id, "session_complete", data={"revisions": revisions})
    yield _sse({
        "type": "done",
        "session_id": session_id,
        "revisions": revisions,
    })
