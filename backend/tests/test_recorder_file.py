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


# -- Driver-input channels (11/26/2026): Brake Pressure, Rear Brake Bias,
# Steering Wheel Range. New recordings carry them; older files must still load.

def _record(tmp_path, telem_over=None, wheel_pressure=None):
    """Recorder file (3 laps as hf_file) with optional telem/wheel overrides."""
    script = []
    for i in range(600):
        lap = 0 if i < 250 else (1 if i < 500 else 2)
        over = dict(telem_over(i) if telem_over else {})
        if wheel_pressure is not None:
            over["mWheels"] = [fakes.fake_wheel(mBrakePressure=p) for p in wheel_pressure(i)]
        script.append((fakes.fake_telem(et=100.0 + i * 0.01, lap=lap, pos_x=float(i), **over),
                       fakes.fake_scor(mLastLapTime=60.0)))
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


def test_input_channels_load_per_lap(tmp_path):
    path = _record(
        tmp_path,
        # bias moved from 0.44 to 0.47 during lap 1 (i = 300 -> et 103.0)
        telem_over=lambda i: {"mRearBrakeBias": 0.44 if i < 300 else 0.47},
        wheel_pressure=lambda i: (0.9, 0.9, 0.6, 0.6) if 50 <= i < 150 else (0.0,) * 4)
    with contextlib.redirect_stdout(io.StringIO()):
        laps = {l.lap_number: l for l in telemetry.load_telemetry_file(path)}
    ch0, ch1 = laps[0].channels, laps[1].channels
    for w in ("FL", "FR", "RL", "RR"):
        assert f"Brake Pressure {w}" in ch0
    assert np.nanmax(ch0["Brake Pressure FL"]) == pytest.approx(0.9)
    assert np.nanmax(ch0["Brake Pressure RR"]) == pytest.approx(0.6)
    # step-held events: lap 0 all 0.44; lap 1 starts at 0.44, ends at 0.47
    assert np.all(ch0["Rear Brake Bias"] == pytest.approx(0.44))
    assert ch1["Rear Brake Bias"][0] == pytest.approx(0.44)
    assert ch1["Rear Brake Bias"][-1] == pytest.approx(0.47)
    assert np.all(ch1["Steering Wheel Range"] == pytest.approx(400.0))
    assert np.all(ch1["Visual Steering Wheel Range"] == pytest.approx(360.0))
    assert laps[0].meta["InputChannels"] == "1"


def test_unpopulated_input_channels_are_left_out(tmp_path):
    # fakes: mBrakePressure 0 on every wheel; here also bias and ranges 0
    path = _record(tmp_path, telem_over=lambda i: {
        "mRearBrakeBias": 0.0, "mPhysicalSteeringWheelRange": 0.0,
        "mVisualSteeringWheelRange": 0.0})
    with contextlib.redirect_stdout(io.StringIO()):
        lap = telemetry.load_telemetry_file(path)[0]
    assert not any(k.startswith("Brake Pressure") for k in lap.channels)
    for k in ("Rear Brake Bias", "Steering Wheel Range", "Visual Steering Wheel Range"):
        assert k not in lap.channels   # not a NaN column


def test_an_old_spool_conversion_gets_no_nan_brake_pressure(tmp_path):
    # A safety copy from before 11/26/2026 has no "Brake Pressure" rows, so the
    # converted file has an EMPTY table with a real unit (not flagged).
    buf = hf.SessionBuffer("Test Car #1", "Test Track", "Tester", origin_x=0.0,
                           origin_z=0.0, poll_hz=1000.0, session_et_start=100.0)
    buf.register_lap(0, 100.0)
    for i in range(300):
        et = 100.0 + i * 0.01
        if i == 250:
            buf.register_lap(1, et)
        t = fakes.fake_telem(et=et, pos_x=float(i), mWheels=[
            fakes.fake_wheel(mBrakePressure=0.5) for _ in range(4)])
        buf.add_sample(t, t.mWheels, fakes.fake_scor(), et)
    buf.s.pop("Brake Pressure")
    with contextlib.redirect_stdout(io.StringIO()):
        path = hf.write_session(buf, tmp_path)
    con = duckdb.connect(str(path), read_only=True)
    try:
        assert con.execute('select count(*) from "Brake Pressure"').fetchone()[0] == 0
    finally:
        con.close()
    with contextlib.redirect_stdout(io.StringIO()):
        lap = telemetry.load_telemetry_file(path)[0]
    assert not any(k.startswith("Brake Pressure") for k in lap.channels)
    assert "Susp Pos FL" in lap.channels       # non-empty wheel tables still load


def test_a_file_without_the_input_channels_still_loads(tmp_path):
    # synthetic_session writes an older-style file: none of the new tables
    path = write_session(tmp_path / "old.duckdb")
    with contextlib.redirect_stdout(io.StringIO()):
        laps = telemetry.load_telemetry_file(path)
    assert laps and "Brake Pos" in laps[0].channels
    assert not any(k.startswith("Brake Pressure") for k in laps[0].channels)
    assert "Rear Brake Bias" not in laps[0].channels