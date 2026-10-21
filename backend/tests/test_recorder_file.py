"""A file written by recorder/recorder.py must read correctly in the backend.

Builds the file with the recorder's own code and its test fakes, so no game
and no personal session data are needed.
"""

import contextlib
import io
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

from synthetic_session import write_session

RECORDER_DIR = Path(__file__).resolve().parents[2] / "recorder"
sys.path.insert(0, str(RECORDER_DIR))

hf = pytest.importorskip("recorder")
fakes = pytest.importorskip("test_recorder")

from app import telemetry  # noqa: E402


@pytest.fixture
def hf_file(tmp_path):
    """Three laps: 0 and 1 completed (2.50 s each), 2 still running at stop."""
    script = []
    for i in range(600):
        lap = 0 if i < 250 else (1 if i < 500 else 2)
        llt = 60.0 if i < 510 else 2.4937
        script.append((fakes.fake_telem(et=100.0 + i * 0.01, lap=lap, pos_x=float(i)),
                       fakes.fake_scor(mLastLapTime=llt)))
    fakes.FakeMMap.script = script
    real = hf.MMapControl
    hf.MMapControl = fakes.FakeMMap
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hf.run_recorder(tmp_path, 2000.0)
    finally:
        hf.MMapControl = real
    (path,) = tmp_path.glob("*.duckdb")
    return path


def test_completed_laps_have_both_boundaries(hf_file, capsys):
    con = duckdb.connect(str(hf_file), read_only=True)
    try:
        bounds = telemetry._get_lap_boundaries(con)
    finally:
        con.close()
    assert [(n, round(s, 2), None if e is None else round(e, 2)) for n, s, e in bounds] == [
        (0, 100.0, 102.5),
        (1, 102.5, 105.0),   # the last COMPLETED lap ends at its crossing...
        (2, 105.0, None),    # ...and only the unfinished lap runs to end of file
    ]
    assert "duration mismatch" not in capsys.readouterr().out


def test_lap_time_rows_are_keyed_to_their_own_lap(hf_file):
    con = duckdb.connect(str(hf_file), read_only=True)
    try:
        times = telemetry._read_lap_time_events(con)
    finally:
        con.close()
    assert times == {0: pytest.approx(2.5), 1: pytest.approx(2.4937)}


def test_backend_replays_the_recorders_10hz_gate_exactly():
    # The backend times every 10 Hz row by replaying recorder.py's gate
    # over GPS Time (telemetry.session_time_axes). If the two ever diverge,
    # the map silently drifts against the telemetry again (10/14 Fuji bug).
    # Clock with clean ticks, skipped ticks (20 ms), an odd 12.5 ms step and
    # a pause; each tick's Lap Dist is its own index so rows can be traced.
    assert telemetry.SLOW_PERIOD_S == hf.SLOW_PERIOD_S
    ets, ms = [], 100_000.0
    for i in range(3000):
        ets.append(ms / 1000.0)
        ms += 20.0 if i < 800 else (12.5 if i == 1500 else (5000.0 if i == 2000 else 10.0))
    buf = hf.SessionBuffer("Test Car #1", "Test Track", "Tester", origin_x=0.0,
                           origin_z=0.0, poll_hz=1000.0, session_et_start=ets[0])
    for i, et in enumerate(ets):
        t = fakes.fake_telem(et=et)
        buf.add_sample(t, t.mWheels, fakes.fake_scor(mLapDist=float(i)), et)
    recorded = [int(v) for v in buf.s["Lap Dist"]]
    replayed = telemetry._replay_slow_gate(np.asarray(buf.s["GPS Time"])).tolist()
    assert replayed == recorded


def _axes_output(path, capsys):
    con = duckdb.connect(str(path), read_only=True)
    try:
        telemetry.session_time_axes(con)
    finally:
        con.close()
    return capsys.readouterr().out


def test_warns_when_an_hf_file_does_not_replay(tmp_path, capsys):
    path = write_session(tmp_path / "a.duckdb")
    con = duckdb.connect(str(path))
    con.execute('DELETE FROM "GPS Latitude" WHERE rowid = (SELECT max(rowid) FROM "GPS Latitude")')
    con.close()
    assert "10 Hz" in _axes_output(path, capsys)


def test_no_warning_for_recorder_py_files(tmp_path, capsys):
    # the v4 recorder decimates by tick count, so the replay is not expected to
    # match; the index fallback is exact for that format.
    path = write_session(tmp_path / "a.duckdb", recorded_by="lmu-coach-recorder v4")
    con = duckdb.connect(str(path))
    con.execute('DELETE FROM "GPS Latitude" WHERE rowid = (SELECT max(rowid) FROM "GPS Latitude")')
    con.close()
    assert "WARNING" not in _axes_output(path, capsys)


def test_no_warning_when_an_hf_file_replays(tmp_path, capsys):
    assert "WARNING" not in _axes_output(write_session(tmp_path / "a.duckdb"), capsys)


def test_channel_frequencies_are_readable(hf_file):
    con = duckdb.connect(str(hf_file), read_only=True)
    try:
        freqs = telemetry._read_channel_frequencies(con)
    finally:
        con.close()
    assert freqs["GPS Time"] == pytest.approx(100.0, abs=0.01)
    assert freqs["Local Vel X"] == pytest.approx(100.0, abs=0.01)
    assert "Front Downforce" not in freqs   # all-zero in the fakes: flagged unpopulated
