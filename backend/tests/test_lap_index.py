"""lap_index / real_lap_flags: the lap list without loading the whole file,
and which laps count as "real" (complete, not a short mid-lap join), so the
UI's BEST and auto-pick skip the session's unfinished last lap."""

from urllib.parse import quote

import duckdb
import pytest
from fastapi.testclient import TestClient

from app import main
from app.telemetry import (LapInfo, lap_index, load_telemetry_file,
                           real_lap_flags, reference_lap_time)

from synthetic_track import SPEED_MS, write_hilly_session


def _index(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return lap_index(con)
    finally:
        con.close()


@pytest.fixture(scope="module")
def full_file(tmp_path_factory):
    return write_hilly_session(tmp_path_factory.mktemp("full") / "full.duckdb")


@pytest.fixture(scope="module")
def partial_file(tmp_path_factory):
    return write_hilly_session(tmp_path_factory.mktemp("partial") / "partial.duckdb",
                               trailing_partial_m=300)


def test_lap_index_matches_load_telemetry_file(full_file):
    idx = _index(full_file)
    assert all(isinstance(l, LapInfo) for l in idx)
    want = [(l.lap_number, round(l.lap_time_s, 3), l.is_complete)
            for l in load_telemetry_file(full_file) if l.is_complete]
    got = [(l.lap_number, round(l.lap_time_s, 3), l.complete) for l in idx if l.complete]
    assert len(want) == 3
    assert got == want


def test_trailing_partial_lap_is_not_real(partial_file):
    idx = _index(partial_file)
    assert len(idx) == 4
    last = idx[-1]
    assert last.complete is False
    assert last.real is False
    assert abs(last.lap_time_s - 300 / SPEED_MS) <= 0.2
    assert all(l.complete and l.real for l in idx[:-1])
    # the partial lap is still a lap load_telemetry_file returns
    assert [l.lap_number for l in idx] == [l.lap_number for l in load_telemetry_file(partial_file)]


def test_real_lap_flags_rule():
    assert real_lap_flags([100, 101, 36, 175], [True, True, True, True]) == [True, True, False, True]
    assert real_lap_flags([100, 101, 99], [True, True, False]) == [True, True, False]
    assert real_lap_flags([], []) == []


def test_reference_lap_time_unchanged(partial_file):
    laps = load_telemetry_file(partial_file)
    # the rule as it was before reference_lap_time used real_lap_flags
    times = sorted(l.lap_time_s for l in laps if l.is_complete and l.lap_time_s < 600)
    med = times[len(times) // 2]
    old = [t for t in times if t >= 0.85 * med][0]
    assert reference_lap_time(laps) == old
    assert reference_lap_time([l for l in laps if not l.is_complete]) is None


def test_laps_endpoint_has_flags(partial_file, monkeypatch):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", partial_file.parent)
    r = TestClient(main.app).get(f"/sessions/sessions/{quote(partial_file.name)}/laps")
    assert r.status_code == 200, r.text[:300]
    laps = r.json()["laps"]
    assert len(laps) == 4
    for lap in laps:
        assert {"lap_number", "lap_time_s", "car_name", "track_name", "session_type",
                "complete", "real"} <= set(lap)
    assert laps[0]["car_name"] == "Synthetic Car #1"
    assert laps[0]["track_name"] == "Synthetic Hill Ring"
    assert laps[-1]["complete"] is False
    assert laps[-1]["real"] is False
    assert all(l["real"] for l in laps[:-1])
