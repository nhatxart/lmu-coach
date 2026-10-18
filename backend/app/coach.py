"""
Turns a grip_analysis report into a prompt for a local LLM and calls it via
Ollama's OpenAI-compatible endpoint.

Deliberately NOT sending raw per-tick telemetry arrays to the model — that
would blow the context budget for no benefit. We send the *already-computed*
event list (ABS lockups, TC zones, understeer candidates, peak G, slip
numbers), which is exactly what a human coach would want summarized before
giving feedback. The model's job is explanation and prioritization, not
signal processing — we did that in grip_analysis.py.
"""

from __future__ import annotations

import json

import httpx

import os

# Defaults come from the container environment (LLM_* preferred, OLLAMA_* still
# honoured for older compose files). The web UI's Settings normally overrides
# all three per request, so these are only the fallback.
OLLAMA_BASE_URL = (os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
                   or "http://host.docker.internal:8001/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or "local"
# Reasoning models can think for minutes before the first token of the answer
# (a non-streaming request is silent until it is finished), so 120s was too low.
DEFAULT_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "300"))


class LLMError(Exception):
    """The LLM call failed in a way the user can act on; str(e) is the advice."""


class LLMUnavailable(LLMError):
    pass


def _normalize_base_url(url: str | None) -> str:
    base = (url or "").strip() or OLLAMA_BASE_URL
    if not base.lower().startswith(("http://", "https://")):
        raise ValueError("LLM base URL must start with http:// or https://")
    return base.rstrip("/")

SYSTEM_PROMPT = """You are a sim racing engineer coach reviewing telemetry from a Le Mans Ultimate session. \
You are given a structured summary of one lap: lap time, ABS lockup events, traction control \
intervention zones, understeer candidates, peak combined G, per-wheel slip figures, slip \
angle analysis (yaw rate, front/rear axle slip angle), and grip utilization — all derived \
directly from the car's physics, not from comparing against another driver's lap.

On slip_angle_analysis specifically: peak_vehicle_slip_angle_deg is always 0 by construction \
(a modeling simplification, not a real measurement) — ignore it. Trust \
peak_slip_angle_difference_deg and its sign as the primary understeer/oversteer indicator: \
positive means front axle slipping more than rear (understeer tendency) at that moment, \
negative means the reverse (oversteer tendency). Treat the absolute front/rear slip angle \
degree values as approximate magnitude, not precise figures.

On grip_utilization (present once enough history exists for this car — null before that): \
front_grip_utilization / rear_grip_utilization is this lap's sustained peak slip angle on \
that axle divided by the LEARNED optimal slip angle for that axle (from accumulated laps, \
not a spec sheet) — near 1.0 means the driver used the tire close to its own observed limit; \
well above 1.0 means they pushed past the point where more slip angle stops producing more \
grip (into the tire's declining region, which costs time and adds instability, not more \
cornering force); well below 1.0 means grip was left unused. When front/rear_utilization_is_upper_bound is \
true, the model has never seen that axle's grip fall off, so its "optimal" slip angle is only \
a lower bound and the utilization may overstate how hard the tire was used — never call that \
"past the limit". All of these use steady-state cornering samples only (not braking or \
throttle-on phases). cornering_efficiency is this lap's typical-high (p90) grip divided by \
the p90 grip the model has observed near the tire's optimal slip angle, in grip_measure units \
(mu_load_normalized = lateral G per unit of actual axle load, which removes downforce \
differences between slow and fast corners) — this is the more important number for prioritizing advice: \
high grip_utilization with low cornering_efficiency means the driver is asking a lot of the \
tire but converting it poorly into actual cornering force (excess/wasted sliding, likely a \
smoothness or line issue); low grip_utilization with low cornering_efficiency means capability \
is being left on the table (a confidence/commitment issue, not a technique-precision issue). \
These two numbers together are far more actionable than either alone — always read them as a pair.

On front_tire_model / rear_tire_model (when present in the driver's baselines): this is a \
PROBABILISTIC model built from this driver's own accumulated laps on this car, not a fixed \
constant — "peak" is the learned optimal slip angle in degrees, "efficient_range" is the \
slip-angle window where the tire is within 5% of that peak, "confidence" (0-1) reflects how \
much accumulated data supports it (treat anything below ~0.5 as a tentative early estimate, \
not a settled fact — say so if you lean on a low-confidence number), and "sample_count" is \
how many data points it's built from. peak_observed false means the curve was still rising \
at the largest slip angle the driver used: the true peak is higher and the driver has room to \
lean on the tire more. tyre_temperature compares tyre temps with the car's optimal \
temperature; if it carries a warning (e.g. a cold-tyre curve), say that the model and the \
lap reflect tyres outside their window, which is itself a root cause worth naming. When \
front_temp_pressure_load / rear_temp_pressure_load are present, they bucket the learned \
optimal slip angle by temperature/pressure/load — temperature and pressure there are \
confounded with cornering itself (hard cornering heats tyres and raises pressure), so never \
present them as cause and effect or as pressure/temperature targets; they are observed \
patterns in this driver's data, not manufacturer specifications.

Your job:
1. Identify the 2-4 most time-costly or risk-costly moments in this lap, prioritized by \
severity (duration, how many events cluster in one zone, how extreme the numbers are, and — \
when available — how far grip_utilization/cornering_efficiency deviate from a healthy pairing).
2. For each, explain in plain language what physically happened (e.g. "front tires locked \
under braking" not just "ABS fired") and what that likely cost the driver — time, stability, \
or both.
3. Give one concrete technique adjustment per flagged moment — something the driver can \
actually change in inputs (brake release rate, throttle application timing, steering smoothness), \
not vague advice like "be smoother."
4. Keep total output under 300 words. Say where on track each issue occurred. When \
corner_names_available is true, every event already carries a "corner" field (and \
"corner_end" if it runs into the next reference point), computed from its lap distance \
against corner_index ([metres, name] pairs, sorted): use those names exactly as given, \
e.g. "into T3 Apex (Coca-Cola Corner)", and do not work out corners yourself from the \
distances. For follow-up questions about a distance, look it up in corner_index. The \
"(+25 m)" offset means past that point; "between X and Y" means between two reference \
points. When corner_names_available is false, there are no corner names for this track: \
use the lap distance (meters into the lap) instead and never invent corner names.
5. If the data shows a genuinely clean lap with no major issues, say so plainly instead of \
inventing problems — do not manufacture critique to fill space.

Do not mention that you are an AI or that this is telemetry data "provided to you" — just coach, \
the way a race engineer would over the radio after a session."""


def build_user_prompt(grip_report: dict, grip_baselines: dict | None = None) -> str:
    # Strip driver_name before it ever reaches the prompt — the model doesn't
    # need it for coaching, and there's no reason to put a real name in an
    # LLM context window that might get logged by the inference server.
    safe_report = {k: v for k, v in grip_report.items() if k != "driver_name"}

    prompt = (
        f"Lap summary (times are seconds from lap start, distances are meters "
        f"into the lap):\n\n{json.dumps(safe_report, indent=2)}"
    )

    if grip_baselines and grip_baselines.get("total_laps_analyzed", 0) >= 5:
        prompt += (
            f"\n\nThis driver's established grip baselines for this car, built from "
            f"{grip_baselines['total_laps_analyzed']} previously analyzed laps "
            f"(median and 90th-percentile values — use these to judge whether "
            f"something in THIS lap is normal for this driver/car or unusually "
            f"severe, rather than judging severity from general assumptions):\n\n"
            f"{json.dumps({k: v for k, v in grip_baselines.items() if v is not None}, indent=2)}"
        )

    return prompt


def build_followup_messages(
    grip_report: dict,
    conversation_history: list[dict],
    new_question: str,
    grip_baselines: dict | None = None,
) -> list[dict]:
    """
    conversation_history: list of {"role": "user"|"assistant", "content": str}
    from prior turns in this session's chat, so follow-up questions have
    context without re-sending the full report each time.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(grip_report, grip_baselines)},
    ]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": new_question})
    return messages


async def get_coaching_analysis(
    grip_report: dict,
    grip_baselines: dict | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    base_url: str | None = None,
) -> str:
    """First-pass analysis: no conversation history yet."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(grip_report, grip_baselines)},
    ]
    return await _call_ollama(messages, model=model, timeout_s=timeout_s, base_url=base_url)


async def ask_followup(
    grip_report: dict,
    conversation_history: list[dict],
    question: str,
    grip_baselines: dict | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    base_url: str | None = None,
) -> str:
    messages = build_followup_messages(
        grip_report, conversation_history, question, grip_baselines
    )
    return await _call_ollama(messages, model=model, timeout_s=timeout_s, base_url=base_url)


async def _call_ollama(
    messages: list[dict],
    model: str | None = None,
    timeout_s: float | None = None,
    base_url: str | None = None,
) -> str:
    base = _normalize_base_url(base_url)
    model = model or DEFAULT_MODEL
    timeout_s = timeout_s or DEFAULT_TIMEOUT_S
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.4,  # coaching should be fairly consistent, not creative
                },
            )
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise LLMUnavailable(
            f"Could not reach the LLM server at {base}. Start it, or turn on "
            "auto-start in Settings > LLM server."
        ) from e
    except httpx.TimeoutException as e:
        raise LLMError(
            f"The LLM server did not answer within {timeout_s:.0f}s. Reasoning models "
            "can think for a long time: raise 'Request timeout' in Settings > LLM server."
        ) from e

    if resp.status_code == 503:  # llama-server answers 503 while the model is loading
        raise LLMUnavailable("The LLM server is still loading the model (HTTP 503). "
                             "Wait a moment and try again.")
    if resp.status_code >= 400:
        raise LLMError(f"The LLM server returned HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        message = resp.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response from the LLM server: {resp.text[:300]}") from e
    content = (message.get("content") or "").strip()
    if not content:
        if message.get("reasoning_content"):
            raise LLMError(
                "The model used its whole reasoning budget without writing an answer. "
                "Lower the reasoning effort/budget in your launcher script, or try again."
            )
        raise LLMError("The LLM server returned an empty answer.")
    return content
