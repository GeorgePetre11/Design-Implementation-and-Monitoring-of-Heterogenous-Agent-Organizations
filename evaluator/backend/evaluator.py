"""
Evaluator Agent -- Gemini 2.5 via Google AI Studio's OpenAI-compatible API.

Independent quality judge for the AI Consulting Firm multi-agent system.
The Evaluator receives ONLY a client question and a finished consulting
report and produces a structured 6-criterion scorecard. It cannot modify
the report, search the web, call tools, or talk to other agents.

Tuned for the Google AI Studio FREE tier, which enforces tight per-minute
and per-day request limits (e.g. ~10 RPM for gemini-2.5-flash). Retries
use long exponential backoff and respect any Retry-After hint returned by
the API.

See README_evaluator_agent.md for the full specification.
"""
import json
import os
import random
import re
import time

from openai import APIError, OpenAI, RateLimitError

from models import EvaluationScorecard

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Google AI Studio exposes an OpenAI-compatible endpoint, so we can keep the
# OpenAI SDK and only swap base_url + model + key.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"

REQUIRED_CRITERIA = (
    "completeness",
    "accuracy",
    "coherence",
    "structure",
    "actionability",
    "critical_depth",
)


# ---------------------------------------------------------------------------
# System prompt -- verbatim from README_evaluator_agent.md
# ---------------------------------------------------------------------------
EVALUATOR_SYSTEM_PROMPT = """\
You are the Evaluator Agent in a multi-agent AI Consulting Firm. Your sole \
responsibility is to independently assess the quality of a consulting report \
produced by other agents.

You are NOT the author of this report. You did not participate in its \
creation. You are an independent judge.

## Your Task

You will receive:
1. The original client business question
2. The final consulting report

You must evaluate the report on exactly 6 criteria, each scored 1-10, with a \
written justification of 2-4 sentences per criterion.

## Scoring Criteria

1. **Completeness (1-10)**: Does the report address ALL aspects of the \
client's question? Are any workstreams, topics, or angles missing? Score 1-3 \
if major sections are missing. Score 7+ only if all key areas are covered \
with reasonable depth.

2. **Accuracy (1-10)**: Are claims and numbers realistic and internally \
consistent? Are sources cited? Does the math check out? If the report states \
costs of EUR 160K and revenue of EUR 120K but claims break-even in 8 months, \
that is an internal contradiction -- score accordingly. If market data, \
competitor names, or statistics are presented without sources, note this. \
Score 1-3 if most data appears fabricated or contradictory.

3. **Coherence (1-10)**: Does the analysis flow logically from data to \
recommendation? Do findings in one section contradict findings in another? \
If the risk section flags fierce competition but the strategy assumes 20% \
market share in year one, that is incoherent. Score 7+ only if recommendations \
clearly follow from the evidence.

4. **Structure (1-10)**: Is the report organized like a professional \
consulting deliverable? Clear sections, logical ordering, executive summary, \
proper formatting? Score 7+ if the structure is clean and navigable.

5. **Actionability (1-10)**: Could a decision-maker act on these \
recommendations? Are there specific timelines, budgets, KPIs, and responsible \
parties? Or just vague advice like "conduct further research"? Score 7+ only \
if a client could hand this to their operations team and start executing.

6. **Critical Depth (1-10)**: Does the report consider risks, counterarguments, \
limitations, and alternative scenarios? Is there sensitivity analysis? Does it \
acknowledge what it does not know? Score 1-3 if the report presents a single \
optimistic scenario with no critical examination.

## Scoring Rules

- Be HONEST. Do not inflate scores. A report full of fabricated data should \
score 2-3 on accuracy, not 7.
- Reference SPECIFIC content from the report in your justifications. Do not \
write vague statements like "could be improved."
- If the report's own numbers contradict each other, flag this explicitly.
- If claims lack sources, say so.
- If competitor names or data points seem invented, say so.
- The overall_score is the arithmetic mean of all 6 criterion scores, rounded \
to 1 decimal.
- Identify the strongest and weakest dimensions.
- List critical issues -- the 2-5 most severe problems you found.

## Output Format

Respond with ONLY a valid JSON object matching this structure. No markdown \
fences, no preamble, no text outside the JSON:

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
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class EvaluatorAgent:
    """Independent quality judge backed by Gemini 2.5 (Google AI Studio)."""

    name = "evaluator"
    display_name = "Evaluator"
    role = "Independent Quality Judge"

    # Retry budget tuned for Google AI Studio's FREE tier: a single
    # /evaluate call may legitimately wait through several 429s if the
    # per-minute quota was just consumed (10 RPM for gemini-2.5-flash,
    # 5 RPM for gemini-2.5-pro). Daily quota exhaustion is NOT retried.
    MAX_RETRIES = 5
    RATE_LIMIT_BASE_DELAY = 12.0     # seconds; first 429 waits this long
    RATE_LIMIT_MAX_DELAY = 90.0      # cap any single backoff
    PARSE_RETRY_DELAY = 2.0          # short pause between schema-failure retries
    MIN_CALL_INTERVAL = 5.0          # minimum seconds between API calls

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ):
        # Accept GEMINI_API_KEY / GOOGLE_API_KEY as aliases so the standard
        # Google AI Studio env-var names also work.
        self.api_key = (
            api_key
            or os.getenv("EVALUATOR_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        self.base_url = base_url or os.getenv("EVALUATOR_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("EVALUATOR_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Google AI Studio's OpenAI-compat endpoint requires a real API key,
        # but we still allow constructing the client without one so /health
        # can report available=false instead of crashing at import time.
        self._client = OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
        )
        self._last_call_time = 0.0

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        # Google AI Studio always requires a real API key (unlike e.g. a
        # local Ollama proxy), so without one the evaluator cannot serve.
        return bool(self.api_key)

    # ------------------------------------------------------------------
    def evaluate(self, question: str | None, report: str) -> dict:
        """Score a consulting report. Returns a validated scorecard dict.

        Raises ConnectionError if the API is unreachable, ValueError if the
        model fails to produce a valid scorecard after MAX_RETRIES attempts,
        and RuntimeError if the agent has no API key configured or has
        exhausted Google AI Studio's daily free-tier quota.
        """
        if not self.api_key:
            raise RuntimeError(
                "Evaluator has no API key configured. Set EVALUATOR_API_KEY "
                "(or GEMINI_API_KEY) to a Google AI Studio key."
            )

        last_err: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                raw = self._call_api(question, report)
                scorecard = self._parse_response(raw)
                self._validate_scorecard(scorecard)
                self._normalise_scorecard(scorecard)
                return scorecard

            except RateLimitError as e:
                last_err = e
                if self._is_daily_quota_exhaustion(e):
                    raise RuntimeError(
                        "Gemini daily free-tier quota exhausted. Retry "
                        f"tomorrow or upgrade the key. ({e})"
                    )
                if attempt < self.MAX_RETRIES - 1:
                    delay = self._rate_limit_backoff(e, attempt)
                    time.sleep(delay)
                    continue
                raise

            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt < self.MAX_RETRIES - 1:
                    # Brief pause so a flaky JSON response doesn't immediately
                    # burn another RPM slot on the free tier.
                    time.sleep(self.PARSE_RETRY_DELAY)
                    continue
                raise ValueError(
                    f"Evaluator failed to produce a valid scorecard after "
                    f"{self.MAX_RETRIES} attempts: {e}"
                )

            except APIError as e:
                # Some Gemini 429s arrive as a generic APIError rather than
                # RateLimitError depending on SDK version -- detect and
                # retry those too.
                if self._looks_like_rate_limit(e):
                    last_err = e
                    if self._is_daily_quota_exhaustion(e):
                        raise RuntimeError(
                            "Gemini daily free-tier quota exhausted. Retry "
                            f"tomorrow or upgrade the key. ({e})"
                        )
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self._rate_limit_backoff(e, attempt))
                        continue
                raise ConnectionError(f"Gemini API error: {e}")

        # Defensive -- the loop above always returns or raises
        raise RuntimeError(f"Evaluator exhausted retries: {last_err}")

    # ------------------------------------------------------------------
    def _rate_limit_backoff(self, err: Exception, attempt: int) -> float:
        """Pick a sleep duration after a 429.

        Prefers the server's Retry-After / RetryInfo hint when present;
        otherwise falls back to exponential backoff with jitter. Capped
        at RATE_LIMIT_MAX_DELAY to avoid a single client tying up a
        worker for minutes.
        """
        hint = self._extract_retry_after(err)
        if hint is not None:
            # Add a small buffer so we don't race the quota window edge.
            return min(self.RATE_LIMIT_MAX_DELAY, hint + 1.0)

        base = self.RATE_LIMIT_BASE_DELAY * (2 ** attempt)
        jitter = random.uniform(0, 2.0)
        return min(self.RATE_LIMIT_MAX_DELAY, base + jitter)

    @staticmethod
    def _extract_retry_after(err: Exception) -> float | None:
        """Best-effort parse of Retry-After / google.rpc.RetryInfo from a
        Gemini error payload. Returns seconds or None."""
        # 1. HTTP Retry-After header on the response object
        resp = getattr(err, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None) or {}
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except (TypeError, ValueError):
                    pass

        # 2. Google's RetryInfo embedded in the JSON body, e.g.
        #    {"error":{"details":[{"@type":".../RetryInfo","retryDelay":"34s"}]}}
        body = getattr(err, "body", None) or {}
        try:
            details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
            for d in details:
                if isinstance(d, dict) and "retryDelay" in d:
                    s = str(d["retryDelay"]).rstrip("s")
                    return float(s)
        except (AttributeError, TypeError, ValueError):
            pass

        # 3. Last-ditch regex on the stringified error
        m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _looks_like_rate_limit(err: Exception) -> bool:
        status = getattr(err, "status_code", None)
        if status == 429:
            return True
        return "rate" in str(err).lower() and "limit" in str(err).lower()

    @staticmethod
    def _is_daily_quota_exhaustion(err: Exception) -> bool:
        # Google AI Studio free tier surfaces daily exhaustion as a 429 whose
        # body references "PerDay" quota metrics; per-minute exhaustion is
        # safe to retry, per-day is not.
        #
        # IMPORTANT: The quota ID name (e.g. "GenerateRequestsPerDay...")
        # contains "PerDay" even for per-minute throttles. We must also
        # check that the retryDelay is long (>120s) or that there is NO
        # retryDelay hint (true daily exhaustion typically omits it or
        # gives a very long delay). A short retryDelay (e.g. 52s) means
        # it's a per-minute rate limit, not daily exhaustion.
        retry_hint = EvaluatorAgent._extract_retry_after(err)
        if retry_hint is not None and retry_hint < 120:
            return False

        text = str(err)
        body = getattr(err, "body", None)
        if isinstance(body, dict):
            text += " " + json.dumps(body)
        return "PerDay" in text or "per day" in text.lower()

    # ------------------------------------------------------------------
    def _call_api(self, question: str | None, report: str) -> str:
        elapsed = time.time() - self._last_call_time
        if elapsed < self.MIN_CALL_INTERVAL:
            time.sleep(self.MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

        q = (question or "").strip()
        if q:
            user_message = (
                f"## Original Client Question\n{q}\n\n"
                f"## Consulting Report to Evaluate\n{report}"
            )
        else:
            user_message = (
                "## Original Client Question\n"
                "(not provided -- infer the implicit business question from "
                "the report itself, and judge completeness against whatever "
                "scope the report claims to address)\n\n"
                f"## Consulting Report to Evaluate\n{report}"
            )
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        # Gemini 2.5 has "thinking" enabled by default and counts thinking
        # tokens against max_tokens, so a rubric scorecard often gets
        # truncated mid-string before the JSON closes. Disable thinking for
        # this call -- a structured 6-criterion scorecard does not need it.
        # (extra_body is ignored by non-Gemini OpenAI-compat backends.)
        gemini_extra_body = {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_budget": 0,
                        "include_thoughts": False,
                    }
                }
            }
        }

        # Ask for strict JSON + disable thinking when the backend supports
        # them; fall back gracefully on older OpenAI-compat endpoints.
        # IMPORTANT: Only fall back on TypeError (unsupported kwarg) or
        # non-rate-limit APIErrors. Rate-limit (429) errors must propagate
        # so the outer retry loop can handle them with proper backoff
        # instead of burning extra quota slots on fallback attempts.
        try:
            response = self._client.chat.completions.create(
                response_format={"type": "json_object"},
                extra_body=gemini_extra_body,
                **kwargs,
            )
        except (RateLimitError, KeyboardInterrupt):
            raise
        except (APIError, TypeError):
            try:
                response = self._client.chat.completions.create(
                    response_format={"type": "json_object"},
                    **kwargs,
                )
            except (RateLimitError, KeyboardInterrupt):
                raise
            except (APIError, TypeError):
                response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    def _parse_response(self, raw: str) -> dict:
        """Parse the model output into a dict, tolerating common LLM quirks."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

        # Extract the outermost JSON object if there's preamble/trailing text
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = m.group(0) if m else text

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Last-ditch: repair the most common LLM quirk -- raw newlines inside
        # string values -- and try again.
        repaired = self._repair_json_strings(candidate)
        return json.loads(repaired)

    @staticmethod
    def _repair_json_strings(text: str) -> str:
        """Escape raw control chars (newline/tab/cr) that appear inside
        JSON string literals. Leaves structural whitespace untouched."""
        out = []
        in_string = False
        escape = False
        for ch in text:
            if in_string:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    out.append(ch)
                    in_string = False
                    continue
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(ch)
            else:
                if ch == '"':
                    in_string = True
                out.append(ch)
        return "".join(out)

    # ------------------------------------------------------------------
    def _validate_scorecard(self, scorecard: dict) -> None:
        """Strict validation against the README schema."""
        evaluation = scorecard.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("Scorecard is missing the 'evaluation' object.")

        present = set(evaluation.keys())
        missing = set(REQUIRED_CRITERIA) - present
        if missing:
            raise ValueError(f"Missing criteria in scorecard: {sorted(missing)}")

        for criterion in REQUIRED_CRITERIA:
            entry = evaluation[criterion]
            score = entry.get("score")
            justification = entry.get("justification", "")
            if not isinstance(score, int) or not (1 <= score <= 10):
                raise ValueError(f"{criterion} score {score!r} out of range [1, 10]")
            if len(justification) < 30:
                raise ValueError(
                    f"{criterion} justification too short: {justification!r}"
                )

        for field in ("summary", "strongest_dimension", "weakest_dimension"):
            if not scorecard.get(field):
                raise ValueError(f"Scorecard is missing required field: {field}")

        # Final pydantic round-trip catches any remaining schema drift
        EvaluationScorecard.model_validate(scorecard)

    # ------------------------------------------------------------------
    def _normalise_scorecard(self, scorecard: dict) -> None:
        """Recompute overall_score from criterion averages, in place."""
        scores = [scorecard["evaluation"][c]["score"] for c in REQUIRED_CRITERIA]
        scorecard["overall_score"] = round(sum(scores) / len(scores), 1)

        # Recompute strongest/weakest from the actual numbers (defensive)
        ranked = sorted(REQUIRED_CRITERIA, key=lambda c: scorecard["evaluation"][c]["score"])
        scorecard["weakest_dimension"] = ranked[0]
        scorecard["strongest_dimension"] = ranked[-1]
