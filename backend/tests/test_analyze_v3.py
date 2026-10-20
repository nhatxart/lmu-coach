"""/analyze end to end on a synthetic recording with tyre-model v3 wired in.

The telemetry loader is replaced by synthetic laps (the recorder-style session file the
other tests write has no Slip Angle channels, so it cannot feed the tyre model), the LLM
is replaced by a stub, and PROFILES_DIR is a temp dir: nothing under data/ is written.
"""

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import coach, grip_profile as gp, main, tire_model as tm
from synthetic_session import write_session
from test_tire_model import _lap


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "PROFILES_DIR", tmp_path / "profiles")
    gp.clear_class_prior_cache()
    session = write_session(tmp_path / "s.duckdb")
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", tmp_path)
    grids = []

    def loader(path, grid_hz=20.0):
        grids.append(grid_hz)
        laps = []
        for i in range(4):
            lap = _lap()
            lap.lap_number = i
            # channels build_grip_report reads on top of what the tyre model needs
            n = len(lap.channels["Ground Speed"])
            for ch in ("Lap Dist", "Yaw Rate", "Steering Pos", "Throttle Pos", "Brake Pos"):
                lap.channels.setdefault(ch, np.linspace(0.0, 4000.0, n) if ch == "Lap Dist"
                                        else np.zeros(n))
            lap.meta.update({"VehicleModel": "Test Model", "VehicleClass": "LMGT3"})
            laps.append(lap)
        return laps

    async def fake_llm(report, baselines, **kw):
        fake_llm.baselines = baselines
        return "coach text"

    monkeypatch.setattr(main, "load_telemetry_file", loader)
    monkeypatch.setattr(main, "get_coaching_analysis", fake_llm)
    c = TestClient(main.app)
    c.grids, c.llm, c.session, c.profiles = grids, fake_llm, session, tmp_path / "profiles"
    yield c
    gp.clear_class_prior_cache()


def _post(c, lap=1):
    return c.post("/analyze", json={"file": c.session.name, "source": "sessions", "lap_number": lap})


def test_analyze_returns_grip_utilization_and_v3_model(client):
    r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "grip_utilization" in body["grip_report"]
    m = body["grip_baselines"]["front_tire_model"]
    assert "model" in m and "coverage" in m
    # the profile is keyed by car MODEL and saved after the LLM answered
    assert [p.name for p in client.profiles.glob("*.json")] == ["Test Model.json"]
    # coaching report at the 20 Hz default grid, tyre ingest at the tyre grid
    assert client.grids == [20.0, tm.TIRE_GRID_HZ]


def test_second_analyze_scores_against_the_cached_fit(client):
    assert _post(client).status_code == 200
    r = _post(client, lap=2)
    assert r.status_code == 200
    # laps were already ingested: nothing is added twice
    prof = gp.load_profile("Test Model")
    assert len(prof.tire_model_laps) == len(set(prof.tire_model_laps))


def test_analyze_survives_a_failing_tyre_fit(client, monkeypatch, caplog):
    def boom(self, *a, **k):
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(gp.GripProfile, "refit_tyre_model", boom)
    with caplog.at_level(logging.ERROR):
        r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grip_report"]["grip_utilization"] is None
    for axle in ("front", "rear"):
        m = body["grip_baselines"][f"{axle}_tire_model"]
        assert m["model"] == "binned_v2_fallback" and m["peak"] is None
        assert "RuntimeError" in m["note"]
    assert "fit exploded" in caplog.text            # the failure is logged, not swallowed
    assert body["analysis"] == "coach text"


def test_pre_update_scoring_uses_a_fit_without_this_files_laps(client, monkeypatch):
    seen = []
    real = main.compute_grip_utilization_for_lap

    def spy(lap, front, rear, **kw):
        seen.append((kw.get("front_fit"), front["sample_count"]))
        return real(lap, front, rear, **kw)

    monkeypatch.setattr(main, "compute_grip_utilization_for_lap", spy)
    assert _post(client).status_code == 200
    # first analyze: the profile is empty when the lap is scored, so no fit, no samples
    assert seen == [(None, 0)]


def test_failed_refit_after_ingest_rolls_the_tyre_data_back(client, monkeypatch):
    real = gp.GripProfile.refit_tyre_model
    calls = []

    def flaky(self, *a, **k):
        calls.append(1)
        if len(calls) > 1:                  # the pre-update refit works, the post-ingest one fails
            raise RuntimeError("late failure")
        return real(self, *a, **k)

    monkeypatch.setattr(gp.GripProfile, "refit_tyre_model", flaky)
    r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()
    for axle in ("front", "rear"):
        assert "RuntimeError" in body["grip_baselines"][f"{axle}_tire_model"]["note"]
    saved = gp.load_profile("Test Model")
    assert saved.tire_model_laps == [] and saved.tire_files == {}
    assert saved.lap_count == 1             # the ABS/TC baseline update is still saved
