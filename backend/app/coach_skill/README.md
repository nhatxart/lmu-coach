# race-engineer-coach skill

Coaching persona + method for lmu-coach's local LLM, built on Suellio Almeida's publicly taught
driving philosophy (Almeida Racing Academy lessons, blog, interviews), paraphrased and mapped onto
lmu-coach telemetry. No transcripts or book text are included.

```
race-engineer-coach/
  SKILL.md                  full skill entry point (agent harnesses that load skills)
  references/
    philosophy.md           mental model the coach must stay consistent with
    techniques.md           technique library (slip angle, 3 tools, trail braking, string theory, MRP, exit, engine braking...)
    diagnostics.md          symptom → cause → fix, mapped to lmu-coach channels
    voice.md                how to explain + drill templates + a worked example
    progression.md          what to coach first by driver level; practice method
  prompt/
    system_prompt.md        single-block compressed version for llama.cpp / Qwen
```

## Using it with llama-server

Load `prompt/system_prompt.md` as the system message in `coach.py`, and send the per-corner
analysis JSON (from grip_analysis / vehicle_dynamics + the car's tire model) as the user message.
For better style adherence, append the worked example from `references/voice.md` once as a
few-shot pair, or pass `references/diagnostics.md` when you have context to spare.

Suggested sampling for analysis output: temperature 0.3–0.5. If Qwen's thinking mode is on,
keep it on for analysis and strip the reasoning before display.

## Calibrate before trusting thresholds
Numbers in `diagnostics.md` marked *engineering note* are starting heuristics. Tune them per car
against your tire model and best laps.
