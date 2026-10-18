"""Channels must line up by game time, not by row index.

Regression for the 10/14/2026 Fuji HF file: the game clock skipped ~18 s of
ticks in lap 0, so the 100 Hz rows (one per tick) and the 10 Hz rows (one
per 0.1 s of game time) stopped lining up by index fraction. /trackmap put
the lap start 7.8 s early on lap 11 (S/F label and brake points off down the
straight) and telemetry.py analysed the wrong stretch of every lap.
"""

import math
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, telemetry

from synthetic_session import (GPS_REF_LAT, GPS_REF_LON, LON_PER_S, slow_rows,
                               tick_times, write_session)


@pytest.fixture
def session(tmp_path):
    return write_session(tmp_path / "Test Track-Test Car _1-2026-10-14T00_00_00Z.duckdb")


def _trackmap(path, monkeypatch, lap):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", path.parent)
    r = TestClient(main.app).get(
        f"/sessions/sessions/{quote(path.name)}/trackmap", params={"lap_number": lap})
    assert r.status_code == 200, r.text[:300]
    return r.json()


def test_session_time_axes_follow_the_game_clock(session):
    import duckdb
    con = duckdb.connect(str(session), read_only=True)
    try:
        t_fast, t_slow = telemetry.session_time_axes(con)
    finally:
        con.close()
    assert t_fast[0] == 10.0 and t_fast[-1] == 130.0
    # 10 Hz rows are 0.1 s apart in game time even where ticks were dropped
    # (to within one 20 ms tick there: the gate can only fire on a tick).
    assert np.allclose(np.diff(t_slow), 0.1, atol=0.021)
    assert len(t_slow) == len(slow_rows(tick_times()))
    assert t_slow[0] == 10.0


def test_telemetry_lap_channels_are_at_their_own_time(session):
    laps = {l.lap_number: l for l in telemetry.load_telemetry_file(session)}
    lap = laps[1]
    abs_t = lap.time_s + 50.0
    # 100 Hz channel and 10 Hz channel both carry their own time as the value.
    assert np.allclose(lap.channels["Ground Speed"], abs_t, atol=0.02)
    assert np.allclose(lap.channels["Path Lateral"], abs_t, atol=0.11)


def test_trackmap_position_and_channels_share_one_clock(session, monkeypatch):
    d = _trackmap(session, monkeypatch, lap=1)
    dl = d["driven_line"]
    # Position is the 10 Hz track; speed is a 100 Hz time tag. Invert the
    # position back to game time and compare.
    x = np.array(dl["x"])
    t_from_speed = np.array(dl["speed"])
    # The backend's projection: x = (lon - centre_lon) * R * cos(lat), with
    # the centre halfway between the session's first and last 10 Hz rows.
    t_slow = [tick_times()[i] for i in slow_rows(tick_times())]
    centre = GPS_REF_LON + (t_slow[0] + t_slow[-1]) / 2 * LON_PER_S
    per_s = LON_PER_S * 111_320.0 * math.cos(math.radians(GPS_REF_LAT))
    x_expected = (GPS_REF_LON + t_from_speed * LON_PER_S - centre) / LON_PER_S * per_s
    assert np.max(np.abs(x - x_expected)) / per_s < 0.15   # seconds
    # And the lap starts where the Lap event says it does (50.0 s).
    assert t_from_speed[0] == pytest.approx(50.0, abs=0.15)
    assert t_from_speed[-1] == pytest.approx(90.0, abs=0.15)


def test_trackmap_lap_start_position_is_at_lap_event_time(session, monkeypatch):
    d = _trackmap(session, monkeypatch, lap=1)
    x = np.array(d["driven_line"]["x"])
    # Full session spans t=10..130, so the projection centre is t=70; lap 1
    # starts at t=50, i.e. 20 s west of centre, and ends at t=90 (20 s east).
    per_s = (x[-1] - x[0]) / 40.0
    assert x[0] == pytest.approx(-20.0 * per_s, abs=0.15 * abs(per_s))
