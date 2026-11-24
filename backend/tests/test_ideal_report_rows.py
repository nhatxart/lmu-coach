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
    assert R.gate_row("Hypercar", 3, "medium") == {"gated": True, "row": "main", "reason": None}
    # low confidence stays in the gate, in its own row (§15.x 3)
    assert R.gate_row("LMGT3", 6, "low") == {"gated": True, "row": "low_confidence", "reason": None}
    assert R.gate_row("Hypercar", 4, "low")["row"] == "low_confidence"


@pytest.mark.parametrize("cls,n,conf", [("LMGT3", 2, "medium"), ("Hypercar", 2, "medium"),   # Genesis Sarthe
                                        ("Hypercar", 1, "low"), ("Hypercar", 0, "medium")])
def test_fewer_than_three_fit_laps_are_excluded_in_every_class(cls, n, conf):
    out = R.gate_row(cls, n, conf)
    assert not out["gated"] and out["row"] == "excluded_few_laps" and "< 3" in out["reason"]


@pytest.mark.parametrize("by,legacy", [("lmu-coach-recorder hf1", False), ("lmu-coach-recorder v4", True),
                                       ("lmu-coach-recorder v3 (full-session)", True),
                                       ("lmu-coach-recorder (shared memory)", True), (None, True)])
def test_legacy_recorder_from_recorded_by(by, legacy):
    assert R.legacy_recorder({} if by is None else {"RecordedBy": by}) is legacy


def test_old_recorder_pairs_are_excluded_with_their_reason():
    """Gate decision 11/23/2026 (§16): old-recorder pairs are out of the gate."""
    out = R.gate_row("Hypercar", 8, "low", legacy=True)
    assert out["gated"] is False and out["row"] == "excluded_legacy_recorder" and "old recorder" in out["reason"]
    assert R.gate_row("Hypercar", 8, "low", legacy=False)["row"] == "low_confidence"


def test_v2_statistic_is_the_075_quantile():
    x = np.random.default_rng(3).normal(0.0, 0.05, 4000)
    s = R._stat(x)
    assert s["shift"] == pytest.approx(np.quantile(x, 0.75)) and s["median"] == pytest.approx(np.median(x))
    assert R._stat(np.zeros(0))["shift"] is None
