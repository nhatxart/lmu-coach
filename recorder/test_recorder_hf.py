"""Tests for recorder_hf.py. No LMU needed.

Run from the recorder\\ directory:
    uv run python -m unittest test_recorder_hf -v
"""

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb

import recorder_hf as hf


class PacerTest(unittest.TestCase):
    def test_pacer_holds_rate(self):
        pacer = hf.Pacer(0.001)
        t0 = time.perf_counter()
        for _ in range(200):
            pacer.wait()
        elapsed = time.perf_counter() - t0
        self.assertGreater(elapsed, 0.17)
        self.assertLess(elapsed, 0.26)

    def test_pacer_does_not_burst_after_stall(self):
        pacer = hf.Pacer(0.001)
        pacer.wait()
        time.sleep(0.05)  # e.g. a save, or the game sitting at a loading screen
        t0 = time.perf_counter()
        for _ in range(5):
            pacer.wait()
        self.assertGreaterEqual(time.perf_counter() - t0, 0.004)


class PollStatsTest(unittest.TestCase):
    def test_pollstats_counts(self):
        stats = hf.PollStats()
        for et in (1.00, 1.00, 1.01, 1.01, 1.03):
            stats.record(et, False)
        self.assertEqual(stats.polls, 5)
        self.assertEqual(stats.ticks, 3)
        self.assertEqual(stats.histogram(), {"10.0ms": 1, "20.0ms": 1})

    def test_pollstats_data_change_without_tick(self):
        stats = hf.PollStats()
        stats.record(1.0, False)
        stats.record(1.0, True)
        self.assertEqual(stats.data_changed_without_tick, 1)

    def test_pollstats_data_change_with_tick_is_not_counted(self):
        stats = hf.PollStats()
        stats.record(1.0, False)
        stats.record(1.01, True)
        self.assertEqual(stats.data_changed_without_tick, 0)

    def test_summary_reports_rates(self):
        stats = hf.PollStats()
        for i in range(100):
            stats.record(1.0 + (i // 10) * 0.01, False)
        text = "\n".join(stats.summary(wall_s=0.1))
        self.assertIn("1000", text)   # 100 polls in 0.1 s
        self.assertIn("100", text)    # 10 ticks in 0.1 s


class EffectiveHzTest(unittest.TestCase):
    def test_effective_hz(self):
        times = [5.0 + i * 0.01 for i in range(101)]
        self.assertAlmostEqual(hf._effective_hz(times, 1000.0), 100.0, places=6)

    def test_effective_hz_falls_back(self):
        self.assertEqual(hf._effective_hz([], 1000.0), 1000.0)
        self.assertEqual(hf._effective_hz([1.0], 1000.0), 1000.0)
        self.assertEqual(hf._effective_hz([1.0, 1.0], 1000.0), 1000.0)


# ── Fakes for the shared-memory structs (only the fields the recorder reads) ──

def _vec(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def fake_wheel(**over):
    w = dict(
        mSuspensionDeflection=0.03, mRideHeight=0.05, mSuspForce=4000.0,
        mBrakeTemp=600.0, mBrakePressure=0.0, mRotation=150.0,
        mLateralPatchVel=0.4, mLongitudinalPatchVel=-0.2,
        mLateralGroundVel=1.5, mLongitudinalGroundVel=50.0,
        mCamber=-0.04, mLateralForce=3000.0, mLongitudinalForce=-500.0,
        mTireLoad=5000.0, mGripFract=0.2, mPressure=170.0,
        mTemperature=[363.15, 368.15, 373.15], mWear=0.97, mSurfaceType=0,
        mStaticUndeflectedRadius=34, mVerticalTireDeflection=0.012,
        mToe=0.001, mTireCarcassTemperature=358.15, mOptimalTemp=95.0,
    )
    w.update(over)
    return SimpleNamespace(**w)


def fake_telem(et=100.0, lap=0, pos_x=0.0, **over):
    t = dict(
        mID=3, mDeltaTime=0.01, mElapsedTime=et, mLapNumber=lap,
        mVehicleName=b"Test Car #1", mTrackName=b"Test Track",
        mPos=_vec(pos_x, 0.0, 5.0), mLocalVel=_vec(0.8, 0.0, -50.0),
        mLocalAccel=_vec(9.81, 0.0, 0.0), mLocalRot=_vec(0.0, 0.2, 0.0),
        mLocalRotAccel=_vec(0.0, 0.05, 0.0), mGear=4, mEngineRPM=7000.0,
        mUnfilteredThrottle=0.5, mUnfilteredBrake=0.0, mUnfilteredSteering=0.1,
        mSteeringShaftTorque=3.5, mFrontRideHeight=0.045, mRearRideHeight=0.06,
        mDrag=900.0, mFrontDownforce=0.0, mRearDownforce=0.0,
        mABSActive=False, mTCActive=False,
        mFrontTireCompoundName=b"Medium", mRearTireCompoundName=b"Medium",
        mVehicleModel=b"Test Model",
        mWheels=[fake_wheel() for _ in range(4)],
    )
    t.update(over)
    return SimpleNamespace(**t)


def fake_scor(**over):
    s = dict(mID=3, mIsPlayer=True, mDriverName=b"Tester", mInPits=False,
             mInGarageStall=False, mPathLateral=0.5, mTrackEdge=6.0,
             mLapDist=1234.0, mLastLapTime=-1.0)
    s.update(over)
    return SimpleNamespace(**s)


def new_buffer(et=100.0):
    return hf.SessionBuffer("Test Car #1", "Test Track", "Tester",
                            origin_x=0.0, origin_z=0.0, poll_hz=1000.0,
                            session_et_start=et)


NEW_SCALAR = ["Poll Time", "Local Vel X", "Local Vel Y", "Local Vel Z", "Yaw Accel",
              "Front Downforce", "Rear Downforce", "Drag", "Front Ride Height",
              "Rear Ride Height", "Steering Shaft Torque"]
NEW_WHEEL = ["Susp Force", "Ride Height", "Camber", "Toe", "Lateral Patch Vel",
             "Longitudinal Patch Vel", "Lateral Ground Vel", "Longitudinal Ground Vel",
             "Tire Deflection", "Surface Type"]
NEW_WHEEL_10HZ = ["TyresWear", "TyresCarcassTemp", "TyresTempLeft", "TyresTempCenter",
                  "TyresTempRight", "Brake Temp"]


class SessionBufferTest(unittest.TestCase):
    def test_new_channels_recorded(self):
        buf = new_buffer()
        t = fake_telem()
        buf.add_sample(t, t.mWheels, fake_scor(), 100.0, poll_time=0.25)
        for ch in NEW_SCALAR:
            self.assertEqual(len(buf.s[ch]), 1, ch)
        for ch in NEW_WHEEL + NEW_WHEEL_10HZ:
            self.assertEqual(len(buf.s[ch]), 1, ch)
            self.assertEqual(len(buf.s[ch][0]), 4, ch)
        self.assertEqual(buf.s["Poll Time"][0], 0.25)
        self.assertEqual(buf.s["Local Vel X"][0], 0.8)
        self.assertEqual(buf.s["Local Vel Z"][0], -50.0)
        self.assertEqual(buf.s["Susp Force"][0], (4000.0,) * 4)
        self.assertAlmostEqual(buf.s["TyresTempLeft"][0][0], 90.0)
        self.assertAlmostEqual(buf.s["TyresTempRight"][0][0], 100.0)
        self.assertAlmostEqual(buf.s["TyresCarcassTemp"][0][0], 85.0)

    def test_static_metadata_captured(self):
        buf = new_buffer()
        t = fake_telem()
        buf.add_sample(t, t.mWheels, fake_scor(), 100.0)
        self.assertEqual(buf.meta["TyreRadiusCm"], "34,34,34,34")
        self.assertEqual(buf.meta["TyreCompoundFront"], "Medium")
        self.assertEqual(buf.meta["VehicleModel"], "Test Model")

    def test_ten_hz_gate_is_time_based(self):
        buf = new_buffer(et=100.0)
        scor = fake_scor()
        for i in range(100):
            t = fake_telem(et=100.0 + i * 0.01)
            buf.add_sample(t, t.mWheels, scor, t.mElapsedTime)
        self.assertEqual(len(buf.s["GPS Time"]), 100)
        self.assertIn(len(buf.s["Lap Dist"]), (10, 11))
        before = len(buf.s["Lap Dist"])
        # The clock jumps 5 s (pause/stall): one slow row, not a burst of 50.
        for et in (106.0, 106.01, 106.02):
            t = fake_telem(et=et)
            buf.add_sample(t, t.mWheels, scor, et)
        self.assertEqual(len(buf.s["Lap Dist"]), before + 1)

    def test_ten_hz_gate_ignores_sample_rate(self):
        # 200 Hz ticks for 1 s: still ~10 slow rows (a count-based gate gives 20).
        buf = new_buffer(et=100.0)
        scor = fake_scor()
        for i in range(200):
            t = fake_telem(et=100.0 + i * 0.005)
            buf.add_sample(t, t.mWheels, scor, t.mElapsedTime)
        self.assertIn(len(buf.s["Lap Dist"]), (10, 11))


class OfficialLapTimeTest(unittest.TestCase):
    def _buffer_with_one_lap(self, stale=60.0):
        buf = new_buffer(et=100.0)
        buf.register_lap(0, 100.0)
        buf.update_official_lap_time(stale, 100.0)   # value left from before recording
        buf.register_lap(1, 172.46)                  # clock difference = 72.46
        return buf

    def test_official_lap_time_applied(self):
        buf = self._buffer_with_one_lap()
        buf.update_official_lap_time(72.4583, 172.66)
        self.assertEqual(buf.lap_time_events[-1], (172.46, 72.4583))

    def test_stale_value_is_not_applied(self):
        # Scoring lags telemetry: the old lap time is still there at the crossing.
        buf = self._buffer_with_one_lap(stale=72.45)
        buf.update_official_lap_time(72.45, 172.47)
        self.assertAlmostEqual(buf.lap_time_events[-1][1], 72.46, places=9)
        buf.update_official_lap_time(72.4583, 172.66)
        self.assertEqual(buf.lap_time_events[-1][1], 72.4583)

    def test_official_lap_time_rejected_on_mismatch(self):
        # e.g. recorder started mid-lap: LMU's lap is longer than what we recorded.
        buf = self._buffer_with_one_lap()
        buf.update_official_lap_time(95.0, 172.66)
        self.assertAlmostEqual(buf.lap_time_events[-1][1], 72.46, places=9)

    def test_official_lap_time_expires(self):
        buf = self._buffer_with_one_lap()
        buf.update_official_lap_time(72.4583, 172.46 + 3.5)
        self.assertAlmostEqual(buf.lap_time_events[-1][1], 72.46, places=9)

    # 10/14/2026 Fuji session: 8 of 18 laps (incl. the best, 1:42.880) got
    # no official time and the log did not say why. These log lines are
    # there to find out.
    def test_expiry_is_logged_with_the_value_seen(self):
        buf = self._buffer_with_one_lap()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            buf.update_official_lap_time(60.0, 172.46 + 3.5)   # never refreshed
        self.assertIn("no official lap time", out.getvalue())
        self.assertIn("1:00.000", out.getvalue())   # what mLastLapTime still held
        self.assertIn("unchanged", out.getvalue())

    def test_late_value_is_logged_as_late(self):
        buf = self._buffer_with_one_lap()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            buf.update_official_lap_time(72.4583, 172.46 + 3.5)
        self.assertIn("1:12.458", out.getvalue())
        self.assertIn("3.5 s after", out.getvalue())

    def test_mismatch_is_logged(self):
        buf = self._buffer_with_one_lap()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            buf.update_official_lap_time(95.0, 172.66)
        self.assertIn("1:35.000", out.getvalue())
        self.assertIn("1:12.460", out.getvalue())

    def test_first_crossing_has_no_pending_time(self):
        buf = new_buffer(et=100.0)
        buf.register_lap(0, 100.0)
        buf.update_official_lap_time(60.0, 100.0)
        buf.update_official_lap_time(72.0, 100.2)
        # No lap has finished yet, so there is no Lap Time row at all.
        self.assertEqual(buf.lap_time_events, [])


class WriteSessionTest(unittest.TestCase):
    def _recorded_buffer(self, n=300):
        buf = new_buffer(et=100.0)
        scor = fake_scor()
        buf.register_lap(0, 100.0)
        for i in range(n):
            t = fake_telem(et=100.0 + i * 0.01, pos_x=float(i))
            buf.add_sample(t, t.mWheels, scor, t.mElapsedTime, poll_time=i * 0.01)
        return buf

    def test_write_session_roundtrip(self):
        buf = self._recorded_buffer()
        buf.meta["Polls"] = "3000"
        with tempfile.TemporaryDirectory() as d:
            out = hf.write_session(buf, Path(d))
            con = duckdb.connect(str(out), read_only=True)
            try:
                tables = {r[0] for r in con.execute("show tables").fetchall()}
                for ch in NEW_SCALAR + NEW_WHEEL + NEW_WHEEL_10HZ:
                    self.assertIn(ch, tables)
                units = dict(con.execute(
                    "select channelName, unit from channelsList").fetchall())
                freqs = dict(con.execute(
                    "select channelName, frequency from channelsList").fetchall())
                meta = dict(con.execute("select key, value from metadata").fetchall())
                rows = lambda t: con.execute(f'select count(*) from "{t}"').fetchone()[0]

                self.assertEqual(rows("Local Vel X"), 300)
                self.assertEqual(rows("Susp Force"), 300)
                self.assertEqual(rows("TyresWear"), len(buf.s["TyresWear"]))
                # Fake car reports zero downforce: flagged, not written as data.
                self.assertEqual(units["Front Downforce"], "unpopulated_by_lmu")
                self.assertEqual(rows("Front Downforce"), 0)
                self.assertEqual(units["Local Vel X"], "m/s")
                # 0 is a real Surface Type (dry tarmac), not "LMU sends nothing".
                self.assertNotEqual(units["Surface Type"], "unpopulated_by_lmu")
                self.assertEqual(rows("Surface Type"), 300)
                self.assertAlmostEqual(freqs["GPS Time"], 100.0, places=3)
                self.assertAlmostEqual(freqs["Lap Dist"], 10.0, delta=0.5)
                self.assertEqual(meta["RecordedBy"], "lmu-coach-recorder hf1")
                self.assertEqual(meta["Polls"], "3000")
                self.assertEqual(meta["TyreRadiusCm"], "34,34,34,34")
                self.assertAlmostEqual(float(meta["EffectiveHz"]), 100.0, places=2)
            finally:
                con.close()

    def test_spool_roundtrip_keeps_meta(self):
        buf = self._recorded_buffer(n=20)
        with tempfile.TemporaryDirectory() as d:
            restored = hf.load_spool(hf.spool_session(buf, Path(d)))
        self.assertEqual(restored.meta["TyreRadiusCm"], "34,34,34,34")

    def test_safety_copy_cannot_be_mistaken_for_recorder_py_one(self):
        # recorder.py --recover globs ".*.spool" and would rewrite an HF
        # session at the wrong frequency without its extra channels.
        buf = self._recorded_buffer(n=20)
        with tempfile.TemporaryDirectory() as d:
            sp = hf.spool_session(buf, Path(d))
            self.assertTrue(sp.name.endswith(".spool_hf"), sp.name)
            self.assertEqual(list(Path(d).glob(".*.spool")), [])

    def test_recover_leaves_recorder_py_safety_copies_alone(self):
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / ".2026-10-12T00_00_00Z.spool"
            other.write_bytes(b"not ours")
            with contextlib.redirect_stdout(io.StringIO()):
                hf.recover_spools(Path(d))
            self.assertEqual(other.read_bytes(), b"not ours")
            self.assertEqual(list(Path(d).glob("*.duckdb")), [])

    def test_recover_converts_hf_safety_copy(self):
        buf = self._recorded_buffer(n=300)
        with tempfile.TemporaryDirectory() as d:
            sp = hf.spool_session(buf, Path(d))
            with contextlib.redirect_stdout(io.StringIO()):
                hf.recover_spools(Path(d))
            self.assertFalse(sp.exists())
            self.assertEqual(len(list(Path(d).glob("*.duckdb"))), 1)


class FakeMMap:
    """Stands in for MMapControl: replays a scripted list of (telem, scor)."""
    script: list = []

    def __init__(self, *_a, **_k):
        self._i = -1
        self.data = None
        self.update = self._update

    def create(self, access_mode=0):
        pass

    def close(self):
        pass

    def _update(self):
        self._i += 1
        if self._i >= len(FakeMMap.script):
            raise KeyboardInterrupt   # same path as Ctrl+C: flush and exit
        telem, scor = FakeMMap.script[self._i]
        self.data = SimpleNamespace(
            scoring=SimpleNamespace(scoringInfo=SimpleNamespace(mNumVehicles=1),
                                    vehScoringInfo=[scor]),
            telemetry=SimpleNamespace(telemInfo=[telem]))


class RecorderLoopTest(unittest.TestCase):
    def _run(self, script, every_poll=False):
        FakeMMap.script = script
        real = hf.MMapControl
        hf.MMapControl = FakeMMap
        try:
            with tempfile.TemporaryDirectory() as d:
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    hf.run_recorder(Path(d), 2000.0, every_poll)
                files = list(Path(d).glob("*.duckdb"))
                self.assertEqual(len(files), 1, log.getvalue())
                con = duckdb.connect(str(files[0]), read_only=True)
                try:
                    n = con.execute('select count(*) from "GPS Time"').fetchone()[0]
                    meta = dict(con.execute("select key, value from metadata").fetchall())
                    lap_times = con.execute('select value from "Lap Time"').fetchall()
                    self.laps = con.execute('select ts, value from "Lap" order by ts').fetchall()
                finally:
                    con.close()
                return n, meta, lap_times, log.getvalue()
        finally:
            hf.MMapControl = real

    def test_one_sample_per_game_tick(self):
        # 300 ticks, each seen by 3 polls -> 900 polls, 300 stored samples.
        scor = fake_scor()
        script = [(fake_telem(et=100.0 + (i // 3) * 0.01, pos_x=float(i // 3)), scor)
                  for i in range(900)]
        n, meta, _, _ = self._run(script)
        self.assertEqual(n, 300)
        self.assertEqual(meta["Polls"], "900")
        self.assertEqual(meta["UniqueTicks"], "300")
        self.assertEqual(meta["DataChangedWithoutTick"], "0")

    def test_every_poll_stores_all(self):
        scor = fake_scor()
        script = [(fake_telem(et=100.0 + (i // 3) * 0.01), scor) for i in range(900)]
        n, _, _, _ = self._run(script, every_poll=True)
        self.assertEqual(n, 900)

    def test_official_lap_time_reaches_the_file(self):
        script = []
        for i in range(600):
            et = 100.0 + i * 0.01
            lap = 0 if i < 250 else (1 if i < 500 else 2)
            # Lap 0 -> 1 at et 102.50 (partial, recorder started mid-lap);
            # lap 1 -> 2 at et 105.00, a full 2.50 s lap, official 2.4937
            # arriving 0.1 s after the crossing.
            llt = 60.0 if i < 260 else (88.0 if i < 510 else 2.4937)
            script.append((fake_telem(et=et, lap=lap, pos_x=float(i)),
                           fake_scor(mLastLapTime=llt)))
        _, meta, lap_times, log = self._run(script)
        values = [round(v[0], 4) for v in lap_times]
        # One row per COMPLETED lap, in lap order: lap 0 by clock difference
        # (official 88.0 rejected), lap 1 official. Lap 2 was in progress.
        self.assertEqual(values, [2.5, 2.4937], log)
        self.assertEqual(meta["LapCount"], "2")
        # The Lap table keeps the crossing that ENDS the last completed lap
        # (as the start of the unfinished lap 2). Without it a reader cannot
        # tell where lap 1 stopped and counts the tail of lap 2 into it.
        self.assertEqual([(round(ts, 2), lap) for ts, lap in self.laps],
                         [(100.0, 0), (102.5, 1), (105.0, 2)])

    def test_telemetry_slot_miss_does_not_cost_half_a_second(self):
        # Online, a car joining can leave scoring and telemetry disagreeing
        # for one poll. recorder.py waited one poll interval; waiting 0.5 s
        # here would silently drop ~50 ticks.
        scor = fake_scor()
        script = [(fake_telem(et=100.0 + i * 0.01, pos_x=float(i)), scor) for i in range(300)]
        script[150] = (fake_telem(et=101.5, mID=99), scor)   # player's slot not in telemetry
        slept = []
        real_sleep = hf.time.sleep
        hf.time.sleep = lambda s: (slept.append(s), real_sleep(min(s, 0.001)))[1]
        try:
            n, _, _, _ = self._run(script)
        finally:
            hf.time.sleep = real_sleep
        self.assertEqual(n, 299)
        self.assertNotIn(0.5, slept)


if __name__ == "__main__":
    unittest.main()
