"""Lap times: LMU's official (1 ms) value wins over the 10 ms clock difference.

recorder.py swaps a lap's Lap Time row to LMU's mLastLapTime when it
agrees with the clock difference; the backend used to ignore that table for
the lap time it shows, so every lap read as a multiple of 10 ms.
"""

import pytest

import duckdb

from app import telemetry

from synthetic_session import write_session


def _times(path):
    return {l.lap_number: l.lap_time_s for l in telemetry.load_telemetry_file(path)}


def test_official_lap_time_is_used(tmp_path):
    path = write_session(tmp_path / "a.duckdb")   # lap 0 official 39.9937
    times = _times(path)
    assert times[0] == pytest.approx(39.9937, abs=1e-6)
    assert times[1] == pytest.approx(40.0, abs=1e-6)


def test_official_lap_time_far_from_clock_difference_is_ignored(tmp_path):
    # e.g. the recording started mid-lap: LMU's lap is longer than ours.
    path = write_session(tmp_path / "a.duckdb", lap_times=((50.0, 55.0), (90.0, 40.0)))
    assert _times(path)[0] == pytest.approx(40.0, abs=1e-6)


def test_recorder_py_placeholder_row_is_keyed_to_the_lap_that_ended(tmp_path, capsys):
    # the v4 recorder writes (first crossing, 0.0) and then one row per crossing
    # holding the duration of the lap that just ENDED at that crossing.
    path = write_session(tmp_path / "a.duckdb", recorded_by="lmu-coach-recorder v4",
                         lap_times=((10.0, 0.0), (50.0, 39.9937), (90.0, 40.0)))
    con = duckdb.connect(str(path), read_only=True)
    try:
        assert telemetry._read_lap_time_events(con) == {
            0: pytest.approx(39.9937), 1: pytest.approx(40.0)}
    finally:
        con.close()
    assert _times(path)[0] == pytest.approx(39.9937, abs=1e-6)
    assert "duration mismatch" not in capsys.readouterr().out
