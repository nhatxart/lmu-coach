"""Tests for recorder.py. No LMU needed.

Run from the recorder\\ directory:
    uv run python -m unittest test_recorder -v
"""

import contextlib
import io
import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb

import recorder as hf


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
        mPos=_vec(pos_x, 12.5, 5.0),
        mOri=[_vec(1.0, 0.0, 0.0), _vec(0.0, 1.0, 0.0), _vec(0.0, 0.0, 1.0)],
        mLocalVel=_vec(0.8, 0.0, -50.0),
        mLocalAccel=_vec(9.81, 0.0, 0.0), mLocalRot=_vec(0.0, 0.2, 0.0),
        mLocalRotAccel=_vec(0.0, 0.05, 0.0), mGear=4, mEngineRPM=7000.0,
        mUnfilteredThrottle=0.5, mUnfilteredBrake=0.0, mUnfilteredSteering=0.1,
        mSteeringShaftTorque=3.5, mFrontRideHeight=0.045, mRearRideHeight=0.06,
        mDrag=900.0, mFrontDownforce=0.0, mRearDownforce=0.0,
        mABSActive=False, mTCActive=False,
        mFrontTireCompoundName=b"Medium", mRearTireCompoundName=b"Medium",
        mVehicleModel=b"Test Model",
        mRearBrakeBias=0.44, mPhysicalSteeringWheelRange=400.0,
        mVisualSteeringWheelRange=360.0,
        mWheels=[fake_wheel() for _ in range(4)],
    )
    t.update(over)
    return SimpleNamespace(**t)


def fake_scor(**over):
    s = dict(mID=3, mIsPlayer=True, mDriverName=b"Tester", mInPits=False,
             mInGarageStall=False, mPathLateral=0.5, mTrackEdge=6.0,
             mLapDist=1234.0, mLastLapTime=-1.0, mVehicleClass=b"LMGT3",
             mPos=_vec(0.0, 0.0, 0.0))
    s.update(over)
    return SimpleNamespace(**s)


def new_buffer(et=100.0):
    return hf.SessionBuffer("Test Car #1", "Test Track", "Tester",
                            origin_x=0.0, origin_z=0.0, poll_hz=1000.0,
                            session_et_start=et)


NEW_SCALAR = ["Poll Time", "Local Vel X", "Local Vel Y", "Local Vel Z", "Yaw Accel",
              "Front Downforce", "Rear Downforce", "Drag", "Front Ride Height",
              "Rear Ride Height", "Steering Shaft Torque", "Pos Y",
              "Ori R0X", "Ori R0Y", "Ori R0Z", "Ori R1X", "Ori R1Y", "Ori R1Z",
              "Ori R2X", "Ori R2Y", "Ori R2Z"]
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
                self.assertEqual(rows("Pos Y"), 300)
                self.assertEqual(
                    con.execute('select min(value), max(value) from "Pos Y"').fetchone(),
                    (12.5, 12.5))
                # Pose is never flagged empty: 0 is a real height / matrix entry.
                self.assertNotEqual(units["Pos Y"], "unpopulated_by_lmu")
                self.assertEqual(freqs["Ori R2Z"], freqs["Pos X"])
                self.assertEqual(meta["PoseChannels"], "1")
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

    def test_other_cars_written(self):
        buf = new_buffer(et=100.0)
        scor = fake_scor()
        other = fake_scor(mID=7, mPos=_vec(1.0, 2.0, 3.0), mLapDist=55.0,
                          mPathLateral=-1.2, mInPits=0)
        for i in range(300):
            t = fake_telem(et=100.0 + i * 0.01)
            buf.add_sample(t, t.mWheels, scor, t.mElapsedTime, others=[other])
        with tempfile.TemporaryDirectory() as d:
            out = hf.write_session(buf, Path(d))
            con = duckdb.connect(str(out), read_only=True)
            try:
                n_slow = con.execute('select count(*) from "Lap Dist"').fetchone()[0]
                got = con.execute(
                    'select slot, pos_x, pos_y, pos_z, lap_dist, path_lateral, in_pits, ts '
                    'from "Other Cars" order by ts').fetchall()
                self.assertEqual(len(got), n_slow)
                self.assertEqual(got[0][:7], (7, 1.0, 2.0, 3.0, 55.0, -1.2, 0))
                self.assertEqual(got[0][7], 100.0)
                freqs = dict(con.execute(
                    "select channelName, unit from channelsList").fetchall())
                self.assertEqual(freqs["Other Cars"], "pose_rows")
            finally:
                con.close()

    def test_other_cars_absent_is_empty_table(self):
        buf = self._recorded_buffer(n=50)       # add_sample without `others`
        with tempfile.TemporaryDirectory() as d:
            out = hf.write_session(buf, Path(d))
            con = duckdb.connect(str(out), read_only=True)
            try:
                self.assertEqual(
                    con.execute('select count(*) from "Other Cars"').fetchone()[0], 0)
            finally:
                con.close()

    # ── Driver-input channels (11/26/2026, "ideal inputs" Q3) ──────────────
    def _input_file(self, frames):
        """Write a session from (et, telem) frames; return (con, tmpdir)."""
        buf = new_buffer(et=100.0)
        scor = fake_scor()
        buf.register_lap(0, 100.0)
        for et, t in frames:
            buf.add_sample(t, t.mWheels, scor, et)
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        with contextlib.redirect_stdout(io.StringIO()):
            out = hf.write_session(buf, Path(d.name))
        con = duckdb.connect(str(out), read_only=True)
        self.addCleanup(con.close)
        return con

    def test_brake_pressure_is_a_100_hz_wheel_channel(self):
        frames = []
        for i in range(300):
            et = 100.0 + i * 0.01
            p = 0.0 if i < 100 else 0.8
            wheels = [fake_wheel(mBrakePressure=p * k) for k in (1.0, 1.0, 0.7, 0.7)]
            frames.append((et, fake_telem(et=et, pos_x=float(i), mWheels=wheels)))
        con = self._input_file(frames)
        ch = dict((r[0], r[1:]) for r in con.execute(
            "select channelName, frequency, unit from channelsList").fetchall())
        self.assertIn("Brake Pressure", ch)
        self.assertEqual(ch["Brake Pressure"][0], ch["GPS Time"][0])   # same 100 Hz rate
        self.assertIn("0-1", ch["Brake Pressure"][1])                   # raw, unit says so
        rows = con.execute('select value1, value2, value3, value4 from "Brake Pressure" '
                           'order by rowid').fetchall()
        self.assertEqual(len(rows), 300)
        self.assertEqual(rows[0], (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(rows[-1][0], 0.8)
        self.assertAlmostEqual(rows[-1][2], 0.56)
        meta = dict(con.execute("select key, value from metadata").fetchall())
        self.assertEqual(meta["InputChannels"], "1")

    def test_all_zero_brake_pressure_is_flagged_unpopulated(self):
        frames = [(100.0 + i * 0.01, fake_telem(et=100.0 + i * 0.01, pos_x=float(i)))
                  for i in range(300)]                       # fake wheels: mBrakePressure 0
        con = self._input_file(frames)
        units = dict(con.execute("select channelName, unit from channelsList").fetchall())
        self.assertEqual(units["Brake Pressure"], "unpopulated_by_lmu")
        self.assertEqual(con.execute('select count(*) from "Brake Pressure"').fetchone()[0], 0)

    def test_bias_and_steering_range_are_on_change_events(self):
        frames = []
        for i in range(300):
            et = 100.0 + i * 0.01
            bias = 0.44 if i < 150 else 0.46           # driver moves the bias mid-session
            frames.append((et, fake_telem(et=et, pos_x=float(i), mRearBrakeBias=bias)))
        con = self._input_file(frames)
        ev = dict(con.execute("select eventName, unit from eventsList").fetchall())
        for name in ("Rear Brake Bias", "Steering Wheel Range", "Visual Steering Wheel Range"):
            self.assertIn(name, ev)
            self.assertNotEqual(ev[name], "unpopulated_by_lmu", name)
        q = lambda t: [(round(a, 2), round(b, 4)) for a, b in con.execute(
            f'select ts, value from "{t}" order by ts').fetchall()]
        self.assertEqual(q("Rear Brake Bias"), [(100.0, 0.44), (101.5, 0.46)])
        self.assertEqual(q("Steering Wheel Range"), [(100.0, 400.0)])
        self.assertEqual(q("Visual Steering Wheel Range"), [(100.0, 360.0)])

    def test_a_nan_setting_is_one_row_not_one_per_tick(self):
        # NaN != NaN: without care every tick would look like a change.
        buf = new_buffer(et=100.0)
        scor = fake_scor()
        for i in range(1000):
            et = 100.0 + i * 0.01
            t = fake_telem(et=et, mRearBrakeBias=float("nan") if i < 600 else 0.45)
            buf.add_sample(t, t.mWheels, scor, et)
        rows = buf.s["Rear Brake Bias"]
        self.assertEqual(len(rows), 2, rows[:5])
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertEqual((round(rows[1][0], 6), rows[1][1]), (106.0, 0.45))

    def test_all_zero_setting_events_are_flagged_unpopulated(self):
        frames = [(100.0 + i * 0.01, fake_telem(
            et=100.0 + i * 0.01, pos_x=float(i), mRearBrakeBias=0.0,
            mPhysicalSteeringWheelRange=0.0, mVisualSteeringWheelRange=0.0))
            for i in range(300)]
        con = self._input_file(frames)
        ev = dict(con.execute("select eventName, unit from eventsList").fetchall())
        for name in ("Rear Brake Bias", "Steering Wheel Range", "Visual Steering Wheel Range"):
            self.assertEqual(ev[name], "unpopulated_by_lmu", name)
            self.assertEqual(con.execute(f'select count(*) from "{name}"').fetchone()[0], 0)

    def test_a_spool_from_before_the_input_channels_still_converts(self):
        # A safety copy written by an older recorder has none of the new keys.
        buf = self._recorded_buffer(n=300)
        for k in ("Brake Pressure", "Rear Brake Bias", "Steering Wheel Range",
                  "Visual Steering Wheel Range"):
            buf.s.pop(k, None)
        with tempfile.TemporaryDirectory() as d:
            sp = hf.spool_session(buf, Path(d))
            with contextlib.redirect_stdout(io.StringIO()):
                hf.recover_spools(Path(d))
            self.assertFalse(sp.exists())
            self.assertEqual(len(list(Path(d).glob("*.duckdb"))), 1)

    def test_spool_roundtrip_keeps_meta(self):
        buf = self._recorded_buffer(n=20)
        with tempfile.TemporaryDirectory() as d:
            restored = hf.load_spool(hf.spool_session(buf, Path(d)))
        self.assertEqual(restored.meta["TyreRadiusCm"], "34,34,34,34")

    def test_safety_copy_uses_the_plain_spool_suffix(self):
        buf = self._recorded_buffer(n=20)
        with tempfile.TemporaryDirectory() as d:
            sp = hf.spool_session(buf, Path(d))
            self.assertTrue(sp.name.endswith(".spool"), sp.name)

    def test_recover_converts_a_safety_copy(self):
        buf = self._recorded_buffer(n=300)
        with tempfile.TemporaryDirectory() as d:
            sp = hf.spool_session(buf, Path(d))
            with contextlib.redirect_stdout(io.StringIO()):
                hf.recover_spools(Path(d))
            self.assertFalse(sp.exists())
            self.assertEqual(len(list(Path(d).glob("*.duckdb"))), 1)

    def test_recover_still_converts_a_pre_rename_spool_hf_copy(self):
        # Until 10/19/2026 this recorder wrote ".spool_hf"; a copy left behind
        # by a crash before the rename must not be orphaned.
        buf = self._recorded_buffer(n=300)
        with tempfile.TemporaryDirectory() as d:
            sp = hf.spool_session(buf, Path(d))
            legacy = sp.with_name(sp.name + "_hf")  # ".<time>.spool" -> ".<time>.spool_hf"
            sp.rename(legacy)
            self.assertEqual(hf._spool_files(Path(d)), [legacy])
            with contextlib.redirect_stdout(io.StringIO()):
                hf.recover_spools(Path(d))
            self.assertFalse(legacy.exists())
            self.assertEqual(len(list(Path(d).glob("*.duckdb"))), 1)

    def test_stop_file_matches_the_host_agent(self):
        # recorder_control.py asks this recorder to save and exit by creating
        # this exact file; the two names drifting apart would make the Record
        # tab's Stop button silently do nothing until the 300 s force-kill.
        agent = (Path(__file__).parent / "recorder_control.py").read_text(encoding="utf-8")
        self.assertEqual(hf.STOP_FILE.name, ".stop_requested")
        self.assertIn('STOP_FILE = RECORDER_DIR / ".stop_requested"', agent)
        self.assertIn('RECORDER_SCRIPT = RECORDER_DIR / "recorder.py"', agent)


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
            generic=SimpleNamespace(gameVersion=1234),
            scoring=SimpleNamespace(
                scoringInfo=SimpleNamespace(mNumVehicles=1, mTrackTemp=31.5,
                                            mAmbientTemp=22.04),
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

    def test_a_recording_that_lost_updates_says_so(self):
        # 250 stored ticks, each one 30 ms after the last: two updates missing
        # between every pair. The save must say so, in the log and the file.
        scor = fake_scor()
        script = [(fake_telem(et=100.0 + i * 0.03, pos_x=float(i)), scor) for i in range(250)]
        n, meta, _, log = self._run(script)
        self.assertEqual(n, 250)
        self.assertEqual(meta["SkippedTicks"], str(2 * 249))
        self.assertIn("WARNING", log)
        self.assertIn("lost", log)

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

    def test_metadata_has_class_version_and_temps(self):
        scor = fake_scor()
        script = [(fake_telem(et=100.0 + i * 0.01, pos_x=float(i)), scor) for i in range(300)]
        _, meta, _, _ = self._run(script)
        self.assertEqual(meta["VehicleClass"], "LMGT3")
        self.assertEqual(meta["GameVersion"], "1234")
        self.assertEqual(meta["TrackTempC"], "31.5")
        self.assertEqual(meta["AmbientTempC"], "22.0")

    def test_session_meta_missing_sources_are_empty_strings(self):
        # Older/odd shared-memory layouts: key still written, value "".
        m = hf._session_meta(SimpleNamespace(), SimpleNamespace())
        self.assertEqual(m, {"VehicleClass": "", "GameVersion": "",
                             "TrackTempC": "", "AmbientTempC": ""})

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


# ── Tick-synced polling (replaces fixed-rate spinning as the default) ────────
#
# LMU publishes telemetry about every 10 ms. Polling 1000x a second with a
# busy-wait costs a CPU core and copies the whole shared-memory block each
# time; the tick-synced loop sleeps to just before the next update, looks at
# the player's clock in the live memory (no copy), and copies only when it
# has moved. Every stored row still carries the game's own timestamp.

class FakeTime:
    """Deterministic clock: a sleep takes the requested time plus 0.4 ms."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += max(s, 0.0) + 0.0004


class TickSyncTest(unittest.TestCase):
    def _world_peek(self, ft, period=0.01, looks=None):
        """Game clock that moves every `period` of fake time."""
        def peek():
            if looks is not None:
                looks.append(ft.t)
            return 1.0 + int(ft.t / period + 1e-9) * 0.01
        return peek

    def test_sleeps_most_of_the_period_then_catches_the_tick(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        looks = []
        peek = self._world_peek(ft, looks=looks)
        self.assertTrue(sync.wait(peek, last_et=peek()))
        self.assertGreater(ft.sleeps[0], 0.006)          # one long sleep, not a spin
        self.assertLessEqual(len(looks) - 1, 4)          # then only a few looks
        self.assertLess(ft.t, 0.0100 + 0.0035)           # caught within ms of arrival

    def test_a_hundred_ticks_cost_a_handful_of_looks_each(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        looks = []
        peek = self._world_peek(ft, looks=looks)
        last = peek()
        caught = 0
        for _ in range(100):
            if sync.wait(peek, last_et=last):
                caught += 1
            last = peek()
        self.assertEqual(caught, 100)                    # no tick missed
        self.assertLess(len(looks), 100 * 6)             # vs ~10 per tick when spinning
        self.assertAlmostEqual(ft.t, 1.0, delta=0.03)    # and still locked to the game

    def test_a_late_wake_up_returns_at_once_without_sleeping_first(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        peek = self._world_peek(ft)
        last = peek()
        sync.wait(peek, last_et=last)
        last = peek()
        ft.t += 0.030                                    # a stall: three ticks went by
        ft.sleeps.clear()
        self.assertTrue(sync.wait(peek, last_et=last))
        self.assertEqual(ft.sleeps, [])

    def test_gives_up_so_a_frozen_game_is_still_looked_at(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        self.assertFalse(sync.wait(lambda: 5.0, last_et=5.0))
        self.assertGreaterEqual(ft.t, hf.TickSync.GIVE_UP_S)
        self.assertLess(ft.t, hf.TickSync.GIVE_UP_S + 0.01)
        self.assertEqual(sync.gave_up, 1)

    def test_without_a_live_peek_it_falls_back_to_timed_polling(self):
        # peek -> None means the live clock cannot be read (library internals
        # changed). The caller then polls once per expected tick instead of
        # waiting out the give-up time every time.
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        self.assertFalse(sync.wait(lambda: None, last_et=5.0))
        self.assertLess(ft.t, 0.0105)

    def test_diagnostics_report_how_late_the_wake_ups_are(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        peek = self._world_peek(ft)
        last = peek()
        for _ in range(100):
            sync.wait(peek, last_et=last)
            last = peek()
        d = sync.diagnostics()
        self.assertEqual(d["waits"], 100)
        self.assertAlmostEqual(d["wake_late_ms_p50"], 0.4, delta=0.2)   # FakeTime overshoots 0.4 ms
        self.assertLess(d["wake_late_ms_max"], 2.0)
        self.assertGreater(d["looks_per_tick"], 0.9)

    def test_diagnostics_show_a_scheduler_that_wakes_the_thread_late(self):
        # The suspected failure on a loaded PC: a sleep(8.5 ms) that returns
        # ~8 ms late. The numbers must make that visible, not average it away.
        ft = FakeTime()

        def late_sleep(s):
            ft.sleeps.append(s)
            ft.t += s + (0.008 if s > 0.005 else 0.0004)

        sync = hf.TickSync(clock=ft.clock, sleep=late_sleep)
        peek = self._world_peek(ft)
        last = peek()
        for _ in range(60):
            sync.wait(peek, last_et=last)
            last = peek()
        d = sync.diagnostics()
        self.assertGreater(d["wake_late_ms_p50"], 7.0)
        self.assertGreater(d["wake_late_ms_max"], 7.0)

    def test_a_skipped_copy_is_retried_at_once_and_teaches_nothing(self):
        # mmap_ctrl.update() copies only when LMU's update flag is set. When it
        # does not, the recorder is handed the SAME tick again and asks to wait
        # for it a second time while the live clock has already moved on. That
        # must be a quick retry, not a full period sleep, and the short gap
        # must not be learned as the game's tick period.
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        peek = self._world_peek(ft)
        last = peek()
        self.assertTrue(sync.wait(peek, last_et=last))        # tick seen; the copy "fails"
        period, t_before, sleeps_before = sync.period, ft.t, len(ft.sleeps)
        self.assertTrue(sync.wait(peek, last_et=last))        # same tick asked for again
        self.assertEqual(len(ft.sleeps) - sleeps_before, 1)   # one short sleep...
        self.assertLess(ft.t - t_before, 0.003)               # ...not a 8.5 ms one
        self.assertEqual(sync.period, period)

    def test_asking_again_while_the_game_is_paused_does_not_go_hot(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        self.assertFalse(sync.wait(lambda: 5.0, last_et=5.0))   # gives up after 50 ms
        t0 = ft.t
        self.assertFalse(sync.wait(lambda: 5.0, last_et=5.0))   # the same tick again: still paused
        self.assertGreaterEqual(ft.t - t0, hf.TickSync.NOMINAL_S - hf.TickSync.EARLY_S)

    def test_one_skipped_game_tick_does_not_bend_the_learned_period(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        peek = self._world_peek(ft)
        last = peek()
        for _ in range(20):
            sync.wait(peek, last_et=last)
            last = peek()
        settled = sync.period
        ft.t += 0.018                       # one update never arrives (~19 ms gap)
        sync.wait(peek, last_et=last)
        self.assertAlmostEqual(sync.period, settled, delta=0.0002)

    def test_learns_a_game_that_ticks_slower_than_100_hz(self):
        ft = FakeTime()
        sync = hf.TickSync(clock=ft.clock, sleep=ft.sleep)
        peek = self._world_peek(ft, period=0.0104)
        last = peek()
        for _ in range(80):
            sync.wait(peek, last_et=last)
            last = peek()
        self.assertAlmostEqual(sync.period, 0.0104, delta=0.0004)


class PeekElapsedTimeTest(unittest.TestCase):
    def test_reads_the_player_clock_from_live_memory(self):
        live = SimpleNamespace(telemetry=SimpleNamespace(
            telemInfo=[fake_telem(et=1.0), fake_telem(et=123.45)]))
        self.assertEqual(hf.peek_elapsed_time(SimpleNamespace(_realtime=live), 1), 123.45)

    def test_none_when_the_live_view_is_not_available(self):
        self.assertIsNone(hf.peek_elapsed_time(SimpleNamespace(_realtime=None), 0))
        self.assertIsNone(hf.peek_elapsed_time(SimpleNamespace(), 0))           # no such attribute
        live = SimpleNamespace(telemetry=SimpleNamespace(telemInfo=[fake_telem()]))
        self.assertIsNone(hf.peek_elapsed_time(SimpleNamespace(_realtime=live), 7))   # slot gone
        self.assertIsNone(hf.peek_elapsed_time(SimpleNamespace(_realtime=live), hf.INVALID_INDEX))


class PollModeTest(unittest.TestCase):
    # 10/21/2026, A/B on a real drive (same mode, same session):
    #   tick-sync at below-normal priority: 91 Hz caught, 286 skipped, wake-ups
    #     up to 4.9 ms late; at NORMAL priority: 100 Hz, 6 skipped, <= 1.0 ms late.
    # The lowered priority (LMU runs at High) made the sleeping recorder wait for
    # a core. Tick-sync at normal priority is the default; fixed polling is
    # selected by --poll-hz / --every-poll.
    def test_default_is_tick_synced(self):
        self.assertEqual(hf.resolve_poll_mode(None, False), ("tick", 100.0))

    def test_a_poll_rate_or_every_poll_selects_fixed_polling(self):
        self.assertEqual(hf.resolve_poll_mode(200.0, False), ("fixed", 200.0))
        self.assertEqual(hf.resolve_poll_mode(None, True), ("fixed", hf.DEFAULT_POLL_HZ))
        self.assertEqual(hf.resolve_poll_mode(500.0, True), ("fixed", 500.0))

    def test_an_explicit_tick_sync_cannot_be_combined_with_a_fixed_rate(self):
        self.assertEqual(hf.resolve_poll_mode(None, False, tick_sync=True), ("tick", 100.0))
        with self.assertRaises(ValueError):
            hf.resolve_poll_mode(200.0, False, tick_sync=True)
        with self.assertRaises(ValueError):
            hf.resolve_poll_mode(None, True, tick_sync=True)

    def test_command_line(self):
        parser = hf.build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.poll_hz)
        self.assertFalse(args.low_priority)
        self.assertTrue(parser.parse_args(["--low-priority"]).low_priority)
        self.assertEqual(parser.parse_args(["--poll-hz", "200"]).poll_hz, 200.0)
        # flags from the first day of the experiment still parse (they are no-ops now)
        parser.parse_args(["--tick-sync", "--normal-priority"])

    def test_help_text_renders(self):
        # argparse %-formats help strings: a bare "%" in one makes --help crash.
        text = hf.build_parser().format_help()
        self.assertIn("--low-priority", text)
        self.assertIn("--poll-hz", text)

    def test_priority_is_left_alone_unless_asked(self):
        calls, runs = [], []
        real = (hf.lower_priority, hf.run_recorder)
        hf.lower_priority = lambda *a: (calls.append(1), True)[1]
        hf.run_recorder = lambda *a, **k: runs.append(a)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hf.main([])
                self.assertEqual(calls, [])              # default: normal priority
                hf.main(["--low-priority"])
                self.assertEqual(calls, [1])
        finally:
            hf.lower_priority, hf.run_recorder = real
        self.assertEqual(len(runs), 2)


class SkipWarningTest(unittest.TestCase):
    def _stats(self, steps_ms):
        stats = hf.PollStats()
        et = 100.0
        stats.record(et, False)
        for s in steps_ms:
            et += s / 1000.0
            stats.record(et, False)
        return stats

    def test_a_recording_that_lost_a_tenth_of_its_updates_is_flagged(self):
        stats = self._stats([10, 20] * 150)              # every other step skips one
        msg = hf.skip_warning(stats)
        self.assertIsNotNone(msg)
        self.assertIn("lost", msg)
        self.assertIn("--poll-hz 1000", msg)

    def test_a_handful_of_game_skips_is_not_flagged(self):
        stats = self._stats([10] * 2990 + [20, 20, 30, 30])   # 6 skipped in ~3000 (the real 0.2%)
        self.assertEqual(stats.skipped_ticks, 6)
        self.assertIsNone(hf.skip_warning(stats))

    def test_no_data_no_warning(self):
        self.assertIsNone(hf.skip_warning(hf.PollStats()))


class PriorityTest(unittest.TestCase):
    def test_sets_below_normal_priority_class(self):
        calls = []
        k32 = SimpleNamespace(
            GetCurrentProcess=lambda: -1,
            SetPriorityClass=lambda handle, cls: (calls.append((handle, cls)), 1)[1])
        self.assertTrue(hf.lower_priority(k32))
        self.assertEqual(calls, [(-1, 0x00004000)])      # BELOW_NORMAL_PRIORITY_CLASS

    def test_a_failure_is_not_fatal(self):
        def boom(*_a):
            raise OSError("access denied")
        k32 = SimpleNamespace(GetCurrentProcess=lambda: -1, SetPriorityClass=boom)
        self.assertFalse(hf.lower_priority(k32))


class SkippedTicksTest(unittest.TestCase):
    def test_a_30_ms_clock_step_is_two_skipped_updates(self):
        stats = hf.PollStats()
        for et in (1.00, 1.01, 1.02, 1.05, 1.06):
            stats.record(et, False)
        self.assertEqual(stats.skipped_ticks, 2)
        self.assertIn("skipped", "\n".join(stats.summary(1.0)))

    def test_ordinary_jitter_is_not_a_skip(self):
        stats = hf.PollStats()
        for et in (1.000, 1.010, 1.021, 1.030, 1.041):    # 10-11 ms steps
            stats.record(et, False)
        self.assertEqual(stats.skipped_ticks, 0)


class LiveWorld:
    """Game memory whose clock moves one tick every `peeks_per_tick` looks at
    the live clock. Deterministic stand-in for real time: it counts how often
    the recorder looks (cheap) versus copies the whole block (expensive)."""

    def __init__(self, ticks, peeks_per_tick=4):
        self.ticks, self.ppt = ticks, peeks_per_tick
        self.peeks = 0
        self.updates = 0
        self.stale_once = set()     # ticks whose first update() does not copy

    def et(self):
        return 100.0 + min(self.peeks // self.ppt, self.ticks) * 0.01


class FakeLiveMMap:
    world: LiveWorld = None

    def __init__(self, *_a, **_k):
        w = FakeLiveMMap.world

        class _Telem:
            def __getitem__(self_inner, i):
                w.peeks += 1
                return SimpleNamespace(mElapsedTime=w.et())

        self._realtime = SimpleNamespace(telemetry=SimpleNamespace(telemInfo=_Telem()))
        self.data = None
        self.update = self._update

    def create(self, access_mode=0):
        pass

    def close(self):
        pass

    def _update(self):
        w = FakeLiveMMap.world
        w.updates += 1
        k = min(w.peeks // w.ppt, w.ticks)
        if k >= w.ticks:
            raise KeyboardInterrupt        # end of the scripted session
        if k in w.stale_once and self.data is not None:
            w.stale_once.discard(k)        # LMU's update flag was not set: no copy,
            return                         # .data is still the previous tick
        telem = fake_telem(et=w.et(), pos_x=float(k))
        self.data = SimpleNamespace(
            generic=SimpleNamespace(gameVersion=1234),
            scoring=SimpleNamespace(
                scoringInfo=SimpleNamespace(mNumVehicles=1, mTrackTemp=31.5, mAmbientTemp=22.0),
                vehScoringInfo=[fake_scor()]),
            telemetry=SimpleNamespace(telemInfo=[telem]))


class TickModeLoopTest(unittest.TestCase):
    def setUp(self):
        self._saved = {n: getattr(hf.TickSync, n) for n in
                       ("NOMINAL_S", "EARLY_S", "STEP_S", "PERIOD_MIN_S", "GIVE_UP_S")}
        hf.TickSync.NOMINAL_S, hf.TickSync.EARLY_S = 0.001, 0.0004   # keep the test fast
        hf.TickSync.STEP_S, hf.TickSync.PERIOD_MIN_S = 0.0004, 0.0005
        # These tests count copies exactly, but they run on real sleeps. A
        # scheduling stall past the 50 ms give-up (seen once, with the game
        # running) makes the loop copy an extra time and breaks the count.
        # TickSyncTest covers the give-up itself on a fake clock.
        hf.TickSync.GIVE_UP_S = 5.0
        self._mmap = hf.MMapControl

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(hf.TickSync, n, v)
        hf.MMapControl = self._mmap

    def _record(self, mmap_cls, **kw):
        hf.MMapControl = mmap_cls
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()) as log:
                hf.run_recorder(Path(d), None, **kw)
            files = list(Path(d).glob("*.duckdb"))
            self.assertEqual(len(files), 1, log.getvalue())
            con = duckdb.connect(str(files[0]), read_only=True)
            try:
                n = con.execute('select count(*) from "GPS Time"').fetchone()[0]
                meta = dict(con.execute("select key, value from metadata").fetchall())
            finally:
                con.close()
        return n, meta

    def test_copies_the_block_once_per_tick_and_only_looks_in_between(self):
        # 250 ticks: a session under MIN_SAMPLES (200) is discarded by design.
        FakeLiveMMap.world = LiveWorld(ticks=250, peeks_per_tick=3)
        n, meta = self._record(FakeLiveMMap)
        w = FakeLiveMMap.world
        self.assertEqual(n, 250)                          # one stored row per game tick
        self.assertEqual(w.updates, 251)                  # 250 ticks + the one that ends the script
        self.assertGreaterEqual(w.peeks, 2 * 250)         # the waiting was done by looking...
        self.assertEqual(meta["UniqueTicks"], "250")
        self.assertEqual(meta["PollMode"], "tick-sync")
        self.assertEqual(meta["PollHz"], "100")

    def test_a_copy_that_lags_the_live_clock_costs_a_retry_not_a_tick(self):
        FakeLiveMMap.world = LiveWorld(ticks=250, peeks_per_tick=3)
        FakeLiveMMap.world.stale_once = set(range(10, 250, 10))   # 24 ticks arrive "uncopied"
        n, meta = self._record(FakeLiveMMap)
        w = FakeLiveMMap.world
        self.assertEqual(n, 250)                          # every tick still stored
        self.assertEqual(w.updates, 251 + 24)             # one extra copy attempt per lag
        self.assertEqual(meta["UniqueTicks"], "250")

    def test_still_records_when_the_live_clock_cannot_be_read(self):
        # The scripted FakeMMap has no _realtime: every wait falls back to
        # one poll per expected tick, and duplicates are still stored once.
        FakeMMap.script = [(fake_telem(et=100.0 + (i // 2) * 0.01, pos_x=float(i // 2)),
                            fake_scor()) for i in range(500)]   # 250 distinct ticks
        n, meta = self._record(FakeMMap)
        self.assertEqual(n, 250)
        self.assertEqual(meta["PollMode"], "tick-sync")

    def test_fixed_rate_mode_is_unchanged(self):
        FakeMMap.script = [(fake_telem(et=100.0 + (i // 3) * 0.01, pos_x=float(i // 3)),
                            fake_scor()) for i in range(900)]
        hf.MMapControl = FakeMMap
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                hf.run_recorder(Path(d), 2000.0)
            con = duckdb.connect(str(next(Path(d).glob("*.duckdb"))), read_only=True)
            meta = dict(con.execute("select key, value from metadata").fetchall())
            con.close()
        self.assertEqual(meta["PollMode"], "fixed")
        self.assertEqual(meta["PollHz"], "2000")


class ProbeCpuTest(unittest.TestCase):
    def test_probe_reports_cpu_use_so_the_two_modes_can_be_compared(self):
        class TimeWorld(FakeLiveMMap):
            """100 Hz game clock driven by real time."""
            t0 = None

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                TimeWorld.t0 = time.perf_counter()
                w = SimpleNamespace(peeks=0, updates=0)
                TimeWorld.w = w

                class _Telem:
                    def __getitem__(s, i):
                        w.peeks += 1
                        return SimpleNamespace(mElapsedTime=TimeWorld.now())

                self._realtime = SimpleNamespace(telemetry=SimpleNamespace(telemInfo=_Telem()))

            @staticmethod
            def now():
                return 100.0 + int((time.perf_counter() - TimeWorld.t0) / 0.01) * 0.01

            def _update(self):
                TimeWorld.w.updates += 1
                self.data = SimpleNamespace(
                    generic=SimpleNamespace(gameVersion=1234),
                    scoring=SimpleNamespace(
                        scoringInfo=SimpleNamespace(mNumVehicles=1, mTrackTemp=31.5, mAmbientTemp=22.0),
                        vehScoringInfo=[fake_scor()]),
                    telemetry=SimpleNamespace(telemInfo=[fake_telem(et=TimeWorld.now())]))

        real = hf.MMapControl
        hf.MMapControl = TimeWorld
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                hf.run_probe(0.4, None)
        finally:
            hf.MMapControl = real
        text = out.getvalue()
        self.assertIn("CPU", text)
        self.assertIn("tick-synced", text)
        # what the next real drive needs to tell WHY ticks were lost
        self.assertIn("wake-up lateness", text)
        self.assertIn("loop iteration", text)
        # about 40 ticks in 0.4 s; copying the block once per tick, not per look
        self.assertLess(TimeWorld.w.updates, 80)


if __name__ == "__main__":
    unittest.main()
