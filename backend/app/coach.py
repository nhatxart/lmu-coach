"""
Turns a grip_analysis report into a prompt for a local LLM and calls it via
Ollama's OpenAI-compatible endpoint.

Deliberately NOT sending raw per-tick telemetry arrays to the model â€” that
would blow the context budget for no benefit. We send the *already-computed*
event list (ABS lockups, TC zones, understeer candidates, peak G, slip
numbers), which is exactly what a human coach would want summarized before
giving feedback. The model's job is explanation and prioritization, not
signal processing â€” we did that in grip_analysis.py.
"""

from __future__ import annotations

import json

import httpx

import os
from pathlib import Path

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

# The coaching method (persona, tyre/technique model, output format) lives in
# coach_skill/method.md: the race-engineer-coach skill's compressed prompt. This
# file keeps only what is specific to OUR data: how to read the report fields.
# Worked example for few-shot priming: coach_skill/example.md (toggle below).
_SKILL_DIR = Path(__file__).parent / "coach_skill"
USE_FEWSHOT = os.environ.get("LLM_COACH_FEWSHOT", "1").lower() not in ("0", "false", "no", "off")


def _load_skill(name: str) -> str:
    return (_SKILL_DIR / name).read_text(encoding="utf-8").strip()


_DATA_NOTES = """DATA YOU ARE GIVEN (how to read lmu-coach's report)

You are given a structured summary of one lap from a Le Mans Ultimate session: lap time, ABS lockup events, traction control \
intervention zones, understeer candidates, peak combined G, per-wheel slip figures, slip \
angle analysis (yaw rate, front/rear axle slip angle), and grip utilization â€” all derived \
directly from the car's physics, not from comparing against another driver's lap.

On slip_angle_analysis specifically: peak_vehicle_slip_angle_deg is always 0 by construction \
(a modeling simplification, not a real measurement) â€” ignore it. Trust \
peak_slip_angle_difference_deg and its sign as the primary understeer/oversteer indicator: \
positive means front axle slipping more than rear (understeer tendency) at that moment, \
negative means the reverse (oversteer tendency). Treat the absolute front/rear slip angle \
degree values as approximate magnitude, not precise figures.

On grip_utilization (present once enough history exists for this car â€” null before that): \
front_grip_utilization / rear_grip_utilization is this lap's sustained peak slip angle on \
that axle divided by the LEARNED optimal slip angle for that axle (from accumulated laps, \
not a spec sheet) â€” near 1.0 means the driver used the tire close to its own observed limit; \
well above 1.0 means they pushed past the point where more slip angle stops producing more \
grip (into the tire's declining region, which costs time and adds instability, not more \
cornering force); well below 1.0 means grip was left unused. When front/rear_utilization_is_upper_bound is \
true, the model has never seen that axle's grip fall off, so its "optimal" slip angle is only \
a lower bound and the utilization may overstate how hard the tire was used â€” never call that \
"past the limit". All of these use steady-state cornering samples only (not braking or \
throttle-on phases). cornering_efficiency is this lap's typical-high (p90) grip divided by \
the p90 grip the model has observed near the tire's optimal slip angle, in grip_measure units \
(mu_load_normalized = lateral G per unit of actual axle load, which removes downforce \
differences between slow and fast corners) â€” this is the more important number for prioritizing advice: \
high grip_utilization with low cornering_efficiency means the driver is asking a lot of the \
tire but converting it poorly into actual cornering force (excess/wasted sliding, likely a \
smoothness or line issue); low grip_utilization with low cornering_efficiency means capability \
is being left on the table (a confidence/commitment issue, not a technique-precision issue). \
These two numbers together are far more actionable than either alone â€” always read them as a pair.

On front_tire_model / rear_tire_model (when present in the driver's baselines): this is a \
PROBABILISTIC model built from this driver's own accumulated laps on this car, not a fixed \
constant â€” "peak" is the learned optimal slip angle in degrees, "efficient_range" is the \
slip-angle window where the tire is within 5% of that peak, "confidence" (0-1) reflects how \
much accumulated data supports it (treat anything below ~0.5 as a tentative early estimate, \
not a settled fact â€” say so if you lean on a low-confidence number), and "sample_count" is \
how many data points it's built from. peak_observed false means the curve was still rising \
at the largest slip angle the driver used: the true peak is higher and the driver has room to \
lean on the tire more. tyre_temperature compares tyre temps with the car's optimal \
temperature; if it carries a warning (e.g. a cold-tyre curve), say that the model and the \
lap reflect tyres outside their window, which is itself a root cause worth naming. When \
front_temp_pressure_load / rear_temp_pressure_load are present, they bucket the learned \
optimal slip angle by temperature/pressure/load â€” temperature and pressure there are \
confounded with cornering itself (hard cornering heats tyres and raises pressure), so never \
present them as cause and effect or as pressure/temperature targets; they are observed \
patterns in this driver's data, not manufacturer specifications. The baselines' \
load_normalization.laps_by_load_source says which load the grip numbers (mu) rest on: \
"Tire Load" is the game's real tyre load; "Susp Force" is a stand-in used for classes where \
the game provides no tyre load (all GT3 cars so far) and is a somewhat noisier measure of the \
same thing â€” when it is the source, treat absolute grip values as approximate, compare \
laps and corners with each other rather than quoting a friction coefficient, and follow the \
note given there. Do not bring any of this up when the source is Tire Load.

On corner_analysis (the evidence for WHY a limit was hit; use it before anything else): \
available=false means there is no per-corner data (see "reason"); then say the cause is a \
hypothesis. Otherwise each entry in "corners" is one corner measured from the driver's inputs \
at about timing_resolution_s precision (so differences under ~0.1 s are noise): "type" is the \
corner kind (hard_braking_slow, medium, fast_lift_or_flat, direction_change); "braking" has \
the peak, the time to peak, the release time, how many times the brake was re-applied after it \
started to come off (reapplications), seconds of ABS and the most steering held while ABS was \
on; "entry" has the gap between brake release and the first steering \
(brake_release_to_turn_in_s positive = release came first; brake_zero_before_turn_in_s = how \
long the brake was fully off before turning; brake_at_turn_in_pct = how much trail braking \
there was), grip_dip_at_turn_in_pct (a proxy from combined G, only present when the brake was \
nearly off at turn-in), downshifts after turn-in, and the front slip angle peak; "rotation" \
has rotation_share_before_throttle (how much of the corner's heading change was already done \
when the committed throttle pickup began; the target is about 0.5 or more), \
min_speed_marks_spread_s (how far apart the slowest point, the most yaw and the most steering \
were), yaw_plateau_on_part_throttle_s, understeer_onset_s (steering still increasing while yaw \
was flat or falling) and countersteer_events; "exit" has when the committed throttle pickup \
began relative to the slowest point (negative = before it), part_throttle_before_pickup_s, the \
steering already unwound at pickup, how fast throttle built, throttle_lifts, TC seconds and the \
rear slip peak. "flags" lists the diagnoses these numbers support for that corner, and \
"repeated_habits" lists flags that occur in 2 or more corners with their corner types: a \
habit that repeats across a corner type is the first thing to coach. A feature that is absent \
was not measurable: do not guess it. Rear/front slip values here are degrees at the stated \
phase, compare them with the tyre-model peak when one is given. Corners that cannot be named \
(see the naming rule below) can still be referred to by their number "n" and kind.

TRUST AND WORDING RULES (these override the method's example phrasing)
- A diagnosis stated as fact must be supported by a field you were given. If it is only \
inferred, say "likely" and tell the driver what to check on the next lap.
- Never print internal field or flag names (grip_utilization, rotation_share_before_throttle, \
peak_vehicle_slip_angle...). Say it in plain words: "your learned grip limit", "about 40% of \
the turning was done when you got on the power".
- Time cost: do not give a number of seconds per issue unless a reference-lap delta is in the \
data. Write "time cost not measured" or rank by importance instead of inventing one.
- Explain method terms the first time you use them (the "min-speed point" is the slowest \
point of the corner; trail braking is braking into the turn). Do not leave abbreviations bare.
- Each issue block names the misconception behind the habit, uses one image or one-liner from \
the method, and quotes this lap's own numbers.
- Write the answer as markdown: "##" headings, "###" for each issue, "-" bullets with bold \
labels. No other bullet symbols.
- If corner_analysis is available, the Confidence section lists only what is really missing \
(for example tyre-model history or a reference lap), not signals you were given.

YOUR TASK AND OUTPUT
1. Identify the 1-3 most time-costly or risk-costly moments or habits in this lap, prioritized by \
severity (duration, how many events cluster in one zone, how extreme the numbers are, and â€” \
when available â€” how far grip_utilization/cornering_efficiency deviate from a healthy pairing). \
When the same habit shows up in several places, report it once as a habit.
2. Explain in plain language what physically happened (e.g. "front tires locked under braking" \
not just "ABS fired"), what it likely cost, and give one concrete input change per issue with a \
feel cue. Follow the OUTPUT format from the method above: Lap summary, the issue blocks, Drill \
for next run, Confidence.
3. Length: aim for roughly 350-500 words. Be complete but do not pad.
4. Corner names are optional, not required. Use one only when the point is specific to that \
location (a particular braking zone, a one-off lockup). When you are describing a general \
behaviour or habit, describe it in terms of the corner type or phase ("your slow hard-braking \
corners", "on entry") and leave names and distances out. When you do name a location: if \
corner_names_available is true, every event already carries a "corner" field (and "corner_end" \
if it runs into the next reference point), computed from its lap distance against corner_index \
([metres, name] pairs, sorted): use those names exactly as given, e.g. "into T3 Apex \
(Coca-Cola Corner)", and do not work out corners yourself from the distances. For follow-up \
questions about a distance, look it up in corner_index. The "(+25 m)" offset means past that \
point; "between X and Y" means between two reference points. When corner_names_available is \
false, there are no corner names for this track: use the lap distance (meters into the lap) \
instead and never invent corner names. Issue headings use the corner name when there is one, \
otherwise the phase or habit.
   Corner labels may follow the track reference: "Name (number)" such as "Coca-Cola Corner (3)", a \
complex as "Name (3, 4, 5)" (numbers comma-separated), or just "Name" for an unnumbered complex \
such as the Porsche Curves; "Turn 7" when only a number is known. Copy these labels exactly. \
When track_reference is present it lists the layout, the official corner count, the complexes \
(with their parts) and the named straights: use a complex's name when the habit applies to the \
whole sequence or to teach the convention, and a single part when the point is specific to it. \
If track_reference.status is "provisional", the corner names are likely right but not yet \
confirmed against enough laps: do not stress them.
5. If the data shows a genuinely clean lap with no major issues, say so plainly instead of \
inventing problems â€” do not manufacture critique to fill space.
6. The worked example you may see is invented data for style only: never reuse its numbers, \
corners or car.

Do not mention that you are an AI or that this is telemetry data "provided to you" â€” just coach, \
the way a race engineer would over the radio after a session."""

# Method first (who you are, how to think, output format), then our data notes
# and task, which win where they differ (length, optional corner names).
SYSTEM_PROMPT = _load_skill("method.md") + "\n\n" + _DATA_NOTES


def _fewshot_messages() -> list[dict]:
    """The skill's worked example as a user/assistant pair; empty if disabled."""
    if not USE_FEWSHOT:
        return []
    raw = _load_skill("example.md")
    user_part, _, assistant_part = raw.partition("## ASSISTANT")
    user_text = user_part.replace("## USER", "", 1).strip()
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_part.strip()},
    ]


def build_user_prompt(grip_report: dict, grip_baselines: dict | None = None) -> str:
    # Strip driver_name before it ever reaches the prompt â€” the model doesn't
    # need it for coaching, and there's no reason to put a real name in an
    # LLM context window that might get logged by the inference server.
    safe_report = {k: v for k, v in grip_report.items() if k != "driver_name"}

    prompt = (
        f"Lap summary (times are seconds from lap start, distances are meters "
        f"into the lap):\n\n{json.dumps(safe_report, separators=(',', ':'))}"
    )

    if grip_baselines and grip_baselines.get("total_laps_analyzed", 0) >= 5:
        prompt += (
            f"\n\nThis driver's established grip baselines for this car, built from "
            f"{grip_baselines['total_laps_analyzed']} previously analyzed laps "
            f"(median and 90th-percentile values â€” use these to judge whether "
            f"something in THIS lap is normal for this driver/car or unusually "
            f"severe, rather than judging severity from general assumptions):\n\n"
            f"{json.dumps({k: v for k, v in grip_baselines.items() if v is not None}, separators=(',', ':'))}"
        )

    return prompt


def _base_messages(grip_report: dict, grip_baselines: dict | None) -> list[dict]:
    """system prompt, optional few-shot pair, then the real lap."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_fewshot_messages(),
        {"role": "user", "content": build_user_prompt(grip_report, grip_baselines)},
    ]


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
    messages = _base_messages(grip_report, grip_baselines)
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
    messages = _base_messages(grip_report, grip_baselines)
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
