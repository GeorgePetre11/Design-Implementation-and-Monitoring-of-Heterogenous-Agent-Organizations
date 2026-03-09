"""
Level 1 — Single Agent (Baseline).

One agent with no role separation. It receives a business question and
produces a full consulting report by itself, covering all analytical
workstreams: market, financial, risk, strategy, and self-evaluation.

Model: qwen2.5:14b via Ollama (configurable via OLLAMA_MODEL env var).
"""
import os
from typing import Generator
import ollama

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

SYSTEM_PROMPT = """\
You are a senior AI business consultant. You work alone and handle every \
aspect of a consulting engagement by yourself. For any client question you \
receive, you must produce a complete, professional consulting report that \
covers all of the following workstreams:

1. WORKSTREAM BREAKDOWN — decompose the question into the key areas that \
need to be investigated.
2. MARKET ANALYSIS — assess the market landscape: size, growth, key players, \
trends, and customer segments relevant to the question.
3. FINANCIAL ANALYSIS — estimate costs, potential revenues, ROI, and \
break-even timelines. Use realistic ranges where exact data is unavailable.
4. RISK ASSESSMENT — identify the top risks (regulatory, market, operational, \
competitive), rate each by probability and impact (low/medium/high), and \
suggest mitigations.
5. STRATEGIC OPTIONS & RECOMMENDATION — present 2–3 distinct strategic \
options with pros, cons, and trade-offs, then recommend one with clear \
justification.
6. IMPLEMENTATION ROADMAP — a phased action plan (short/mid/long term) for \
the recommended option.
7. SELF-EVALUATION — score your own report on: Completeness, Accuracy, \
Coherence, Structure, Actionability, Critical Depth (each 1–10) with a \
one-line justification per criterion.

Format your report in clean Markdown with clear headings and bullet points. \
Be thorough, data-driven, and professional — this report will be presented \
to a board of directors.\
"""


def run(question: str) -> Generator[str, None, None]:
    """Stream the consulting report token by token."""
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Client question: {question}"},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            yield content
