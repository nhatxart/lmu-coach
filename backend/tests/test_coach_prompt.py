"""Prompt assembly for the race-engineer-coach skill integration (coach.py).

Run: uv run --no-project --with pytest --with httpx --with fastapi --with duckdb --with numpy --with scipy pytest backend/tests/test_coach_prompt.py
"""
from app import coach


def test_system_prompt_has_method_and_data_notes():
    p = coach.SYSTEM_PROMPT
    assert "Almeida-method" in p                      # skill method
    assert "DATA YOU ARE GIVEN" in p                  # our report notes
    assert "grip_utilization" in p
    assert "## Drill for next run" in p               # skill output format
    assert "never invent corner names" in p
    assert "Corner names are optional" in p


def test_fewshot_pair_is_parsed(monkeypatch):
    monkeypatch.setattr(coach, "USE_FEWSHOT", True)
    pair = coach._fewshot_messages()
    assert [m["role"] for m in pair] == ["user", "assistant"]
    assert pair[0]["content"].startswith("Example lap")
    assert pair[1]["content"].startswith("## Lap summary")
    assert "## USER" not in pair[0]["content"] and "## ASSISTANT" not in pair[1]["content"]


def test_fewshot_can_be_disabled(monkeypatch):
    monkeypatch.setattr(coach, "USE_FEWSHOT", False)
    assert coach._fewshot_messages() == []


def test_message_order_first_pass_and_followup(monkeypatch):
    monkeypatch.setattr(coach, "USE_FEWSHOT", True)
    msgs = coach.build_followup_messages(
        {"lap_time": 90.0, "driver_name": "x"},
        [{"role": "assistant", "content": "first answer"}],
        "why?",
    )
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "user", "assistant", "user"]
    assert "driver_name" not in msgs[3]["content"]
    assert msgs[-1]["content"] == "why?"
