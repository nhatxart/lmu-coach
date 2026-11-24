"""Ideal lap, design T9: GET /sessions/{source}/{file}/ideal-lap on a synthetic
session (tests/synthetic_track.py). The ring holds no Susp Force and runs at
one speed, so it cannot be FITTED (that is T2-T8 and T10); the endpoint and
the orchestration are tested with a known car injected through the store, and
the real store's refusal comes back as insufficient_data, never an error."""
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, ideal_lap
from test_ideal_solver import _car
from synthetic_track import write_hilly_session

TOP = {"status", "reason", "car_model", "car_class", "track", "lap_number", "lap_time_s", "fit", "ideal",
       "ceiling", "model_only", "trace", "corners", "note", "public", "public_check"}
TRACE = {"lap_dist", "v_kmh", "v_ref_kmh", "t", "binding", "saturated_ref", "u_front", "u_rear",
         "u_front_ref", "u_rear_ref"}
CORNER = {"n", "name", "s0", "s1", "t_ref", "t_ideal", "gain_s", "gain_min_s", "vmin_ref_kmh", "vmin_ideal_kmh",
          "binding", "u_front_ref", "u_rear_ref", "verdict", "where", "measured_share", "extrapolated"}


def _axle(knee=4.0):
    return {"mu1": 1.5, "k": 0.0, "c": 0.0, "status": "observed", "saturation": "knee", "knee_deg": knee,
            "n_sat": 1000, "ratio_sd": 0.05, "v3_peak_slip_deg": None, "knee_vs_v3": None, "terms": "fitted",
            "v_sat": [10.0, 30.0], "trim": "none", "trim_w": 0.0}


class FakeStore:
    """car_for_lap -> a known car; what ideal_lap reads from a reduction."""

    def __init__(self, car):
        self.car = car
        self.info = {"scope": "track", "key": "synthetic|1200", "n_laps": 3, "n_files": 1, "load_source": "susp",
                     "confidence": "medium", "beta": 0.03, "game_version": "test", "notes": [],
                     "axles": {"front": _axle(), "rear": _axle()}, "s_front": {"value": 0.5, "source": "estimate"},
                     "kappa_brake": {"value": 1.0, "status": "fitted"}, "kappa_trac": {"value": 1.0, "status": "fitted"},
                     "drag": {"d0": 0.0, "d2": 0.0, "source": "prior"}, "selfcheck": None, "transferred": False,
                     "fingerprint": "f0", "setup": {}, "setup_differs": []}

    def car_for_lap(self, path, lap_number=None):
        return self.car, dict(self.info)

    def _reduction(self, path):
        return SimpleNamespace(car_model="Synthetic", car_class="LMGT3", game_version="test")


@pytest.fixture
def session(tmp_path, monkeypatch):
    f = write_hilly_session(tmp_path / "Synthetic Hill Ring-Synthetic Car #1-2026-11-01T10_00_00Z.duckdb",
                            lap_offsets=(0.0, 1.0, -1.0), slip_angles=True, speed_wave=0.15)
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", f.parent)
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", tmp_path / "none")
    from app import grip_profile
    monkeypatch.setattr(grip_profile, "PROFILES_DIR", tmp_path / "profiles")
    ideal_lap._CACHE.clear()
    return f


def _get(f, what, lap=None):
    q = "" if lap is None else f"?lap_number={lap}"
    r = TestClient(main.app).get(f"/sessions/sessions/{quote(f.name)}/{what}{q}")
    assert r.status_code == 200, r.text
    return r.json()


def test_ideal_lap_schema_and_trace_on_the_driven_line(session, monkeypatch):
    # a car that holds the ring's 20 m/s corners easily, so the model is faster
    monkeypatch.setattr(main, "_ideal_store", lambda: FakeStore(_car(mu=(1.6, 1.7), power=0.6)))
    d = _get(session, "ideal-lap", lap=1)
    tm = _get(session, "trackmap", lap=1)
    # `diagnostics` (binding shares, unclamped gain, corner source) is beyond §7, for V10 / V13
    assert set(d) == TOP | {"diagnostics"} and d["status"] == "ok" and d["reason"] is None
    assert d["lap_number"] == 1 and d["lap_time_s"] > 0
    assert set(d["trace"]) == TRACE
    n = len(tm["driven_line"]["lap_dist"])
    assert n > 10 and all(len(v) == n for v in d["trace"].values())
    t = np.asarray(d["trace"]["t"])
    assert (np.diff(t) > 0).all()
    # §7: t ends at ideal.time_s (review minor 8: it ended one 2 m step short)
    assert t[-1] == pytest.approx(d["ideal"]["time_s"], abs=0.005)
    i = d["ideal"]
    assert i["gain_min_s"] <= i["gain_s"] <= i["gain_max_s"]
    assert i["time_s"] <= d["lap_time_s"] + 1e-6                        # never slower than the lap
    assert i["time_s"] == pytest.approx(d["lap_time_s"] - i["gain_s"], abs=1e-3)
    # the ceiling is not clamped (§5.4): it compares with the unclamped plain model
    assert d["ceiling"]["time_s"] <= d["model_only"]["time_s"] + 1e-6
    assert set(d["model_only"]) == {"time_s", "under_driven_share"}
    assert d["corners"] and all(set(c) >= CORNER for c in d["corners"] if c["kind"] == "corner")
    assert {"front", "rear"} == set(d["fit"]["axles"]) and "clamp_conflict" in d["fit"]["axles"]["front"]
    assert isinstance(d["note"], str) and d["public"] is None and d["public_check"] is None
    # §12 Q5: each tyre's share of its axle's load (left, right), per corner
    for c in (c for c in d["corners"] if c["kind"] == "corner"):
        assert set(c["load_share"]) == {"front", "rear"}
        for pair in c["load_share"].values():
            assert len(pair) == 2 and sum(pair) == pytest.approx(1.0, abs=0.002) and min(pair) > 0


def test_lap_omitted_is_the_lap_trackmap_picks(session, monkeypatch):
    monkeypatch.setattr(main, "_ideal_store", lambda: FakeStore(_car(mu=(1.6, 1.7), power=0.6)))
    d = _get(session, "ideal-lap")
    tm = _get(session, "trackmap")
    assert d["status"] == "ok" and d["lap_number"] == tm["lap_number"]
    assert len(d["trace"]["t"]) == len(tm["driven_line"]["lap_dist"])


def test_a_reference_lap_is_indexed_under_its_own_source(tmp_path, monkeypatch):
    """Review minor 9: the public-line lookup indexed every file as 'sessions'."""
    from app import session_index
    ref = tmp_path / "reference_laps"
    ref.mkdir()
    f = write_hilly_session(ref / "Synthetic Hill Ring-Synthetic Car #1-2026-11-02T10_00_00Z.duckdb",
                            lap_offsets=(0.0, 1.0), slip_angles=True)
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", ref)
    monkeypatch.setattr(main, "_ideal_store", lambda: FakeStore(_car(mu=(1.6, 1.7), power=0.6)))
    ideal_lap._CACHE.clear()
    seen = []
    real = session_index.file_entry
    monkeypatch.setattr(session_index, "file_entry", lambda p, source: seen.append(source) or real(p, source))
    r = TestClient(main.app).get(f"/sessions/reference_laps/{quote(f.name)}/ideal-lap?lap_number=0")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert seen and set(seen) == {"reference_laps"}


def test_the_real_store_refuses_this_session_without_an_error(session):
    """No Susp Force on the ring: the store cannot fit a car, and says why."""
    d = _get(session, "ideal-lap", lap=1)
    assert d["status"] == "insufficient_data"
    assert isinstance(d["reason"], str) and d["reason"]
    assert d["ideal"] is None and d["trace"] is None
