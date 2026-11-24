"""Ideal lap validation tool, design §15 V5 / §15.x 3: the gate row of a pair and
the V2 statistic (the 0.75 quantile, median reported)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "tools" / "ideal_lap_report.py"
_spec = importlib.util.spec_from_file_location("ideal_lap_report", _P)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_gate_rows():
    assert R.gate_row("Hypercar", 2, "medium") == {"gated": True, "row": "main", "reason": None}
    # low confidence stays in the gate, in its own row (§15.x 3)
    assert R.gate_row("LMGT3", 6, "low") == {"gated": True, "row": "low_confidence", "reason": None}
    assert R.gate_row("Hypercar", 1, "low")["row"] == "low_confidence"
    out = R.gate_row("LMGT3", 2, "medium")
    assert not out["gated"] and out["row"] == "excluded" and "< 3" in out["reason"]


def test_v2_statistic_is_the_075_quantile():
    x = np.random.default_rng(3).normal(0.0, 0.05, 4000)
    s = R._stat(x)
    assert s["shift"] == pytest.approx(np.quantile(x, 0.75)) and s["median"] == pytest.approx(np.median(x))
    assert R._stat(np.zeros(0))["shift"] is None
