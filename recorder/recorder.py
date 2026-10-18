"""
LMU-Coach recorder v4 — records the FULL session into a single DuckDB file,
schema-compatible with LMU's own native .duckdb export so that all existing
backend endpoints work without modification.

Key design decisions:
  - GPS Time channel uses telem.mElapsedTime (LMU's per-vehicle, 100Hz-
    updated session elapsed timer) as the master clock — NOT
    scoringInfo.mCurrentET, which only refreshes at ~5Hz and was the
    cause of a precision bug where every recorded timestamp rounded to
    roughly the nearest 0.1-0.2s instead of true ~0.01s precision. This
    is the SAME underlying clock LMU uses in its own native export
    (confirmed by comparing GPS Time precision against a native file), so
    lap duration = Lap[i+1].ts - Lap[i].ts gives the correct wall-clock time.
  - Lap event table stores (mCurrentET, lap_number) at each lap crossing,
    exactly matching LMU's own Lap table format.
  - Lap Time event table stores (mCurrentET, duration) at each crossing,
    computed from the ET delta — matches LMU's "Lap Time" channel.
  - All continuous channels written with their correct row count; wheel
    channels use value1..value4 columns, not a single (v,) tuple.
  - One file per session; filename encodes track + car + session start time.

Sampling:
  100 Hz  — position, speed, inputs, G-forces, yaw, wheel/slip/susp, RPM
  10 Hz   — track geometry (Path Lateral, Track Edge, Lap Dist), tyre thermals
  Event   — Gear, ABS, TC, In Pits, Lap, Lap Time (on change / on crossing)
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import signal
import sys
import tempfile
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent))
from pyLMUSharedMemory import lmu_data
from pyLMUSharedMemory.lmu_mmap import INVALID_INDEX, LMUConstants, MMapControl

# BUG FIX: this used to be Path("../data/sessions") — a RELATIVE path,
# resolved against the process's current working directory at launch, not
# against recorder.py's own location. Launching the recorder from anywhere
# other than exactly the recorder/ folder (a desktop shortcut, a batch
# file, running it from the repo root, from $HOME) silently wrote session
# files into a different "../data/sessions" than the one the user was
# looking in — with no error, since output_dir.mkdir(parents=True,
# exist_ok=True) creates whatever path it's given. This is a very likely
# explanation for "recordings aren't showing up in data/sessions": they
# were probably written to an unexpected sibling folder next to wherever
# the recorder was actually launched from. Anchoring to __file__'s own
# directory (same pattern already used for the pyLMUSharedMemory import
# path above) makes the output location independent of launch method.
DEFAULT_OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "sessions"
DEFAULT_POLL_HZ     = 100.0
# Stop request channel used by recorder_control.py. A plain file is used
# instead of a signal because on Windows Popen.terminate() does NOT deliver
# a signal (it force-kills), so signal handlers can never run there. The
# recorder polls for this file every loop iteration, finishes any save in
# progress, writes the current session, and exits cleanly.
STOP_FILE           = Path(__file__).parent / ".stop_requested"
FREEZE_TIMEOUT_S    = 2.0
SCORING_DECIMATE    = 10          # 10 Hz for geometry + thermal channels
MIN_SAMPLES         = 200         # discard sessions shorter than this

# Pseudo-GPS origin — overridden per session from first mPos sample
GPS_REF_LAT         = 35.37
GPS_REF_LON         = 138.92
R_DEG               = 111_320.0  # metres per degree of latitude


# ── Shared-memory helpers ────────────────────────────────────────────────────

def local_scoring_index(scor_veh) -> int:
    for idx, veh in enumerate(scor_veh):
        if veh.mIsPlayer:
            return idx
    return INVALID_INDEX


def build_tele_index_map(num_vehicles: int, tele_data) -> dict[int, int]:
    return {tele_data.telemInfo[i].mID: i for i in range(num_vehicles)}


def _safe_decode(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _slip_angle(lat_vel: float, lon_vel: float) -> float:
    return math.atan(lat_vel / lon_vel) if lon_vel else 0.0


# ── Connection watchdog ──────────────────────────────────────────────────────

class ConnectionState:
    def __init__(self):
        self.last_et:     float = -1.0
        self.last_wall:   float = 0.0
        self.is_frozen:   bool  = True
        self.last_garage: bool  = False
        self.last_slot:   int   = INVALID_INDEX

    def update(self, et: float, in_garage: bool, slot_id: int) -> tuple[bool, bool]:
        """
        Returns (reset, just_froze).
        reset: a session reset was detected (lap/session counter went
        backwards, car entered garage, or vehicle slot changed) — caller
        should flush and start a new buffer.
        just_froze: mElapsedTime stopped advancing for longer than
        FREEZE_TIMEOUT_S and was NOT already frozen last call — i.e. this
        is the first tick where we can tell the session has gone idle
        (player quit to menu, paused at a loading screen, exited to pits
        and sat, etc). Distinguished from "still frozen" (every tick after
        the first) so the caller flushes exactly once per freeze rather
        than trying on every poll while frozen.

        BUG FIX: previously is_frozen only gated whether the main loop
        kept polling — nothing ever read the frozen->just-froze transition,
        so a session that went idle (quit to menu without the mmap itself
        erroring out, or without ET resetting) was silently held open in
        memory forever. If the recorder process was later killed any way
        other than Ctrl+C (closing the terminal window, Task Manager, a
        crash) — which raises SIGTERM/SIGHUP, NOT KeyboardInterrupt/SIGINT
        — flush() never ran at all, and that entire session's data was
        lost with no error, no warning, nothing on disk. Flushing on the
        freeze transition itself means idle time no longer has to wait for
        an unclean process kill to lose data — by the time that happens,
        the session was already safely on disk.
        """
        reset = False
        just_froze = False
        now   = time.monotonic()
        was_frozen = self.is_frozen
        if et != self.last_et:
            if (self.last_et > et
                    or (not self.last_garage and in_garage)
                    or (self.last_slot != INVALID_INDEX
                        and self.last_slot != slot_id)):
                reset = True
            self.last_wall   = now
            self.last_et     = et
            self.last_garage = in_garage
            self.last_slot   = slot_id
            self.is_frozen   = False
        elif now - self.last_wall > FREEZE_TIMEOUT_S:
            self.is_frozen = True
            just_froze = (not was_frozen)
        return reset, just_froze


# ── Session buffer ───────────────────────────────────────────────────────────

class SessionBuffer:
    """
    Accumulates every sample for a driving session.

    Timestamps use telem.mElapsedTime (LMU's 100Hz per-vehicle session
    elapsed timer — see run_recorder() for why this replaced the lower-
    precision scoringInfo.mCurrentET) so that lap durations computed from
    the Lap event table are accurate:
        lap_time = Lap[i+1].ts - Lap[i].ts
    """

    def __init__(self, car_name: str, track_name: str, driver_name: str,
                 origin_x: float, origin_z: float, poll_hz: float,
                 session_et_start: float):
        self.car_name         = car_name
        self.track_name       = track_name
        self.driver_name      = driver_name
        self.poll_hz          = poll_hz
        self.start_wall       = datetime.now(timezone.utc)
        self.et_start         = session_et_start   # mCurrentET when buffer created
        self.origin_x         = origin_x
        self.origin_z         = origin_z

        # ── Continuous channel lists ──────────────────────────────────────
        # Each list grows at its own Hz:
        #   _100: appended every sample  (100 Hz)
        #   _10:  appended every SCORING_DECIMATE-th sample (10 Hz)
        self.s: dict[str, list] = defaultdict(list)
        self.sample_count = 0
        self.tick         = 0            # increments every poll

        # ── Event tables ──────────────────────────────────────────────────
        # Each entry is (session_et, value) — absolute mCurrentET timestamps
        self.lap_events:      list[tuple[float, int]]   = []
        self.lap_time_events: list[tuple[float, float]] = []
        self.gear_events:     list[tuple[float, int]]   = []
        self.abs_events:      list[tuple[float, float]] = []
        self.tc_events:       list[tuple[float, float]] = []
        self.pits_events:     list[tuple[float, float]] = []

        # ── Last-seen values for event deduplication ──────────────────────
        self._last_lap:   int   | None = None
        self._last_et_at_lap: float    = session_et_start
        self._last_gear:  int   | None = None
        self._last_abs:   float | None = None
        self._last_tc:    float | None = None
        self._last_pits:  float | None = None

    # ── Lap crossing ─────────────────────────────────────────────────────────

    def register_lap(self, lap_number: int, current_et: float) -> None:
        """Record a lap boundary.  Called when mLapNumber changes."""
        # Lap table: (mCurrentET, lap_number) — matches LMU native format
        self.lap_events.append((current_et, lap_number))

        # Lap Time table: duration of the PREVIOUS lap (0 for the first crossing)
        if self._last_lap is not None:
            duration = current_et - self._last_et_at_lap
            self.lap_time_events.append((current_et, duration))
            m, s = divmod(duration, 60)
            print(f"  Lap {self._last_lap}  {int(m)}:{s:06.3f}  ({duration:.3f}s)")
        else:
            self.lap_time_events.append((current_et, 0.0))

        self._last_lap       = lap_number
        self._last_et_at_lap = current_et

    # ── Per-sample recording ─────────────────────────────────────────────────

    def add_sample(self, telem, wheels, player_scor, current_et: float) -> None:
        """Record one poll tick.  current_et = mCurrentET from scoring."""
        s   = self.s                       # alias
        shz = self.poll_hz / SCORING_DECIMATE

        # ── 100 Hz: telemetry-sourced continuous ──────────────────────────
        # GPS Time matches LMU's own convention: absolute session ET in seconds
        s["GPS Time"].append(current_et)

        # Pos X/Z: world metres (recorder-specific extension; backend handles both)
        s["Pos X"].append(telem.mPos.x)
        s["Pos Z"].append(telem.mPos.z)

        spd = math.sqrt(telem.mLocalVel.x**2 +
                        telem.mLocalVel.y**2 +
                        telem.mLocalVel.z**2)
        s["Ground Speed"].append(spd * 3.6)    # km/h — matches LMU unit
        s["G Force Lat"].append(telem.mLocalAccel.x / 9.81)
        s["G Force Long"].append(telem.mLocalAccel.z / 9.81)
        s["G Force Vert"].append(telem.mLocalAccel.y / 9.81)
        s["Steering Pos"].append(telem.mUnfilteredSteering)
        s["Throttle Pos"].append(telem.mUnfilteredThrottle * 100.0)   # 0-100 %
        s["Brake Pos"].append(   telem.mUnfilteredBrake    * 100.0)   # 0-100 %
        s["Yaw Rate"].append(telem.mLocalRot.y)                       # rad/s
        s["Engine RPM"].append(telem.mEngineRPM)

        # Per-wheel 100 Hz — stored as (FL, FR, RL, RR) tuples
        s["Wheel Speed"].append(
            (wheels[0].mRotation, wheels[1].mRotation,
             wheels[2].mRotation, wheels[3].mRotation))
        s["Slip Angle"].append(
            (_slip_angle(wheels[0].mLateralGroundVel, wheels[0].mLongitudinalGroundVel),
             _slip_angle(wheels[1].mLateralGroundVel, wheels[1].mLongitudinalGroundVel),
             _slip_angle(wheels[2].mLateralGroundVel, wheels[2].mLongitudinalGroundVel),
             _slip_angle(wheels[3].mLateralGroundVel, wheels[3].mLongitudinalGroundVel)))
        s["Grip Fraction"].append(
            (wheels[0].mGripFract, wheels[1].mGripFract,
             wheels[2].mGripFract, wheels[3].mGripFract))
        s["Tire Load"].append(
            (wheels[0].mTireLoad, wheels[1].mTireLoad,
             wheels[2].mTireLoad, wheels[3].mTireLoad))
        s["Tire Lateral Force"].append(
            (wheels[0].mLateralForce, wheels[1].mLateralForce,
             wheels[2].mLateralForce, wheels[3].mLateralForce))
        s["Tire Longitudinal Force"].append(
            (wheels[0].mLongitudinalForce, wheels[1].mLongitudinalForce,
             wheels[2].mLongitudinalForce, wheels[3].mLongitudinalForce))
        s["Susp Pos"].append(
            (wheels[0].mSuspensionDeflection, wheels[1].mSuspensionDeflection,
             wheels[2].mSuspensionDeflection, wheels[3].mSuspensionDeflection))

        # ── 10 Hz: scoring-sourced geometry + thermals ────────────────────
        if self.tick % SCORING_DECIMATE == 0:
            # Pseudo-GPS: reproject mPos.x/z to lat/lon degrees using
            # a per-session origin so track shape is self-consistent
            dx_m = telem.mPos.x - self.origin_x
            dz_m = telem.mPos.z - self.origin_z
            cos_lat = math.cos(math.radians(GPS_REF_LAT))
            s["GPS Latitude"].append( GPS_REF_LAT + dz_m / R_DEG)
            s["GPS Longitude"].append(GPS_REF_LON + dx_m / (R_DEG * cos_lat))
            s["GPS Speed"].append(spd)                          # m/s
            s["Path Lateral"].append(player_scor.mPathLateral)
            s["Track Edge"].append(player_scor.mTrackEdge)
            s["Lap Dist"].append(player_scor.mLapDist)
            # Tyre temperature: mean of 3-zone array, Kelvin→Celsius
            s["TyresRubberTemp"].append(
                (sum(wheels[0].mTemperature) / 3.0 - 273.15,
                 sum(wheels[1].mTemperature) / 3.0 - 273.15,
                 sum(wheels[2].mTemperature) / 3.0 - 273.15,
                 sum(wheels[3].mTemperature) / 3.0 - 273.15))
            s["TyresPressure"].append(
                (wheels[0].mPressure, wheels[1].mPressure,
                 wheels[2].mPressure, wheels[3].mPressure))

        # ── Events: write only on change ──────────────────────────────────
        g  = telem.mGear
        ab = float(telem.mABSActive)
        tc = float(telem.mTCActive)
        ip = float(player_scor.mInPits)

        if g  != self._last_gear:
            self.gear_events.append((current_et, g));   self._last_gear = g
        if ab != self._last_abs:
            self.abs_events.append((current_et, ab));   self._last_abs  = ab
        if tc != self._last_tc:
            self.tc_events.append((current_et, tc));    self._last_tc   = tc
        if ip != self._last_pits:
            self.pits_events.append((current_et, ip));  self._last_pits = ip

        self.sample_count += 1
        self.tick         += 1


# ── DuckDB writer ────────────────────────────────────────────────────────────

def write_session(buf: SessionBuffer, output_dir: Path) -> Path:
    """
    Writes the full session to a .duckdb file, ATOMICALLY.

    Root cause of the v3/v4 "silently truncated file" bug: write_session()
    was writing directly to the final output path. If the process is
    interrupted for ANY reason partway through (Ctrl+C, LMU/game crash,
    Windows sleep, antivirus scan, disk hiccup) — even one that has
    nothing to do with a logic bug in this function — DuckDB commits
    whatever tables/rows had landed at that instant, producing a
    corrupt-but-openable file with early channels intact and later ones
    (including the "Lap" event table, written last) missing entirely.
    This is exactly the shape of file the user hit in production: GPS
    Time/Pos X/Pos Z fully written, Ground Speed cut off mid-insert, and
    no Lap table at all — despite in-game lap counters showing several
    completed laps.

    Fix: write everything to a temporary path first, inside a single
    explicit transaction. Only if every table writes successfully do we
    commit and rename the temp file to the real output path. If anything
    goes wrong (interruption, exception, disk error) partway through, the
    temp file is left behind half-written (or is removed) and the real
    output path is never touched — so a partial write can never again
    look like a valid, silently-broken session file to the backend or to
    external tools.
    """
    hz  = buf.poll_hz
    shz = hz / SCORING_DECIMATE                     # 10.0 Hz
    ts  = buf.start_wall.strftime("%Y-%m-%dT%H_%M_%SZ")

    def _safe(name: str) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in name)

    out      = output_dir / f"{_safe(buf.track_name)}-{_safe(buf.car_name)}-{ts}.duckdb"
    tmp_out  = output_dir / f".{out.name}.tmp"
    tmp_out.unlink(missing_ok=True)  # clean up any stale leftover from a prior crash

    con = duckdb.connect(str(tmp_out))

    try:
        con.execute("BEGIN TRANSACTION")

        # ── metadata ─────────────────────────────────────────────────────────
        con.execute("CREATE TABLE metadata (key VARCHAR, value VARCHAR)")
        con.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("Version",          "4"),
            ("DriverName",       buf.driver_name),
            ("RecordingTime",    ts),
            ("TrackName",        buf.track_name),
            ("CarName",          buf.car_name),
            ("RecordedBy",       "lmu-coach-recorder v4"),
            ("TotalSamples",     str(buf.sample_count)),
            ("LapCount",         str(len(buf.lap_events))),
        ])

        # ── channel / event registry ──────────────────────────────────────────
        con.execute("CREATE TABLE channelsList "
                    "(channelName VARCHAR, frequency DOUBLE, unit VARCHAR)")
        con.execute("CREATE TABLE eventsList "
                    "(eventName VARCHAR, unit VARCHAR)")

        def _all_zero_wheel(ch: str) -> bool:
            rows = buf.s.get(ch, [])
            return bool(rows) and all(
                v1 == 0.0 and v2 == 0.0 and v3 == 0.0 and v4 == 0.0
                for v1, v2, v3, v4 in rows
            )

        # PERFORMANCE (root cause of "the recording never saved"): these
        # helpers used to call con.executemany() with one Python-level
        # INSERT per row. Measured: ~2s per 60,000-row channel, so a real
        # 10-25 minute session (~16 channels at 100Hz) spent one to several
        # MINUTES in this function at shutdown. Anything that interrupted
        # it in that window (a stop request with a short timeout, closing a
        # console) left an uncommitted, empty .tmp database. A single
        # "INSERT ... SELECT unnest(?)" per channel does the same rows in
        # the same order ~11x faster (measured 0.9s vs 10.3s for two
        # 150,000-row tables; output verified identical).
        CHUNK = 25_000  # rows formatted per write: bounds memory use
        load_csv = output_dir / f".load_{os.getpid()}.csv"

        def _bulk(ch: str, cols: list) -> None:
            """Load equal-length column lists into table `ch` via CSV + COPY.

            WHY CSV: on the user's Windows PC, inserting Python lists through
            DuckDB's Python parameter conversion cost ~0.36 ms PER VALUE
            (measured: 150,000 values = 54 s, 600,000 values = 240 s, versus
            <1 s on Linux), so a real session took 30-60+ minutes to save and
            was always killed/abandoned mid-write. Writing plain text and
            letting DuckDB's native CSV reader ingest it involves no
            per-value Python-object conversion, and is the bulk-load route
            DuckDB's own documentation recommends. repr(float) round-trips
            exactly. Progress is logged per channel so a slow save is visible.
            """
            t0 = time.monotonic()
            n = len(cols[0])
            try:
                with open(load_csv, "w", newline="") as f:
                    for i in range(0, n, CHUNK):
                        if len(cols) == 1:
                            f.write("\n".join(map(repr, cols[0][i:i + CHUNK])))
                        else:
                            f.write("\n".join(
                                ",".join(map(repr, r))
                                for r in zip(*(c[i:i + CHUNK] for c in cols))))
                        f.write("\n")
                sql_path = load_csv.as_posix().replace("'", "''")
                con.execute(f"COPY \"{ch}\" FROM '{sql_path}' (FORMAT csv, HEADER false)")
            finally:
                load_csv.unlink(missing_ok=True)
            print(f"    {ch}: {n:,} rows, {time.monotonic() - t0:.1f}s", flush=True)

        def _write_scalar(ch: str, freq: float, unit: str) -> None:
            """Write a single-value continuous channel."""
            vals = buf.s.get(ch, [])
            con.execute("INSERT INTO channelsList VALUES (?,?,?)", [ch, freq, unit])
            con.execute(f'CREATE TABLE "{ch}" (value DOUBLE)')
            if vals:
                _bulk(ch, [vals])

        def _write_wheel(ch: str, freq: float, unit: str) -> None:
            """Write a per-wheel (value1..value4) continuous channel."""
            rows = buf.s.get(ch, [])
            # Mark channels LMU doesn't populate for this car
            effective_unit = "unpopulated_by_lmu" if _all_zero_wheel(ch) else unit
            con.execute("INSERT INTO channelsList VALUES (?,?,?)",
                        [ch, freq, effective_unit])
            con.execute(f'CREATE TABLE "{ch}" '
                        f'(value1 DOUBLE, value2 DOUBLE, '
                        f'value3 DOUBLE, value4 DOUBLE)')
            if rows and effective_unit != "unpopulated_by_lmu":
                _bulk(ch, [list(c) for c in zip(*rows)])

        def _write_event_scalar(ch: str, unit: str,
                                rows: list[tuple[float, float | int]]) -> None:
            """Write a scalar event table (ts DOUBLE, value DOUBLE)."""
            con.execute("INSERT INTO eventsList VALUES (?,?)", [ch, unit])
            con.execute(f'CREATE TABLE "{ch}" (ts DOUBLE, value DOUBLE)')
            if rows:
                _bulk(ch, [[float(ts) for ts, _ in rows],
                           [float(v) for _, v in rows]])

        def _write_event_int(ch: str, unit: str,
                             rows: list[tuple[float, int]]) -> None:
            """Write an integer event table (ts DOUBLE, value INTEGER)."""
            con.execute("INSERT INTO eventsList VALUES (?,?)", [ch, unit])
            con.execute(f'CREATE TABLE "{ch}" (ts DOUBLE, value INTEGER)')
            if rows:
                _bulk(ch, [[float(ts) for ts, _ in rows],
                           [int(v) for _, v in rows]])

        # ── 100 Hz scalar channels ────────────────────────────────────────────
        _write_scalar("GPS Time",     hz,  "s")
        _write_scalar("Pos X",        hz,  "m")
        _write_scalar("Pos Z",        hz,  "m")
        _write_scalar("Ground Speed", hz,  "km/h")
        _write_scalar("Steering Pos", hz,  "%")
        _write_scalar("Throttle Pos", hz,  "%")
        _write_scalar("Brake Pos",    hz,  "%")
        _write_scalar("Yaw Rate",     hz,  "rad/s")
        _write_scalar("Engine RPM",   hz,  "RPM")
        _write_scalar("G Force Lat",  hz,  "G")
        _write_scalar("G Force Long", hz,  "G")
        _write_scalar("G Force Vert", hz,  "G")

        # ── 100 Hz wheel channels ─────────────────────────────────────────────
        _write_wheel("Wheel Speed",             hz, "rad/s")
        _write_wheel("Slip Angle",              hz, "rad")
        _write_wheel("Susp Pos",                hz, "m")
        _write_wheel("Grip Fraction",           hz, "")
        _write_wheel("Tire Load",               hz, "N")
        _write_wheel("Tire Lateral Force",      hz, "N")
        _write_wheel("Tire Longitudinal Force", hz, "N")

        # ── 10 Hz scalar channels ─────────────────────────────────────────────
        _write_scalar("GPS Latitude",  shz, "deg")
        _write_scalar("GPS Longitude", shz, "deg")
        _write_scalar("GPS Speed",     shz, "m/s")
        _write_scalar("Path Lateral",  shz, "m")
        _write_scalar("Track Edge",    shz, "m")
        _write_scalar("Lap Dist",      shz, "m")

        # ── 10 Hz wheel channels ──────────────────────────────────────────────
        _write_wheel("TyresRubberTemp", shz, "C")
        _write_wheel("TyresPressure",   shz, "kPa")

        # ── Event channels ────────────────────────────────────────────────────
        # "Lap" table: (ts=mCurrentET, value=lap_number)
        # This is the authoritative source for lap boundaries.
        # The backend's _get_lap_boundaries() reads this and computes
        #   duration = Lap[i+1].ts - Lap[i].ts
        # which gives the true wall-clock lap time (from LMU's timer).
        _write_event_int("Lap",      "",    buf.lap_events)
        _write_event_scalar("Lap Time", "s", buf.lap_time_events)
        _write_event_int("Gear",     "",    buf.gear_events)
        _write_event_scalar("ABS",   "",    buf.abs_events)
        _write_event_scalar("TC",    "",    buf.tc_events)
        _write_event_scalar("In Pits", "",  buf.pits_events)

        print("    committing...", flush=True)
        _tc = time.monotonic()
        con.execute("COMMIT")
        con.close()
        print(f"    committed in {time.monotonic() - _tc:.1f}s", flush=True)
    except BaseException:
        # Catches Exception AND KeyboardInterrupt/SystemExit — a Ctrl+C
        # mid-write must not leave a half-committed file behind either.
        try:
            con.close()
        except Exception:
            pass
        tmp_out.unlink(missing_ok=True)
        raise

    # Only reached if every table above was written and committed
    # successfully. os.replace is atomic on both Windows and POSIX (same
    # filesystem), so readers either see the old absence of the file or
    # the fully-written final file — never a partial one.
    os.replace(tmp_out, out)
    return out


# ── Main loop ────────────────────────────────────────────────────────────────

def run_recorder(output_dir: Path, poll_hz: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / poll_hz
    STOP_FILE.unlink(missing_ok=True)  # a leftover request must not stop a fresh run

    # Recover any .tmp files left behind by a crash between "all tables
    # committed" and "renamed to final path" (this was the exact failure
    # mode of the missing-`os`-import bug: the write itself succeeded, only
    # the rename step raised, leaving a fully-valid file stuck under a
    # ".<name>.tmp" path). A .tmp file that opens cleanly and has a Lap
    # table is real, finished data — recover it. Anything else is left
    # alone so a genuinely mid-write file isn't mistaken for good data.
    for stray in output_dir.glob(".*.duckdb.tmp"):
        final_name = stray.name[1:-4]  # strip leading "." and trailing ".tmp"
        final_path = output_dir / final_name
        if final_path.exists():
            continue
        try:
            con = duckdb.connect(str(stray), read_only=True)
            con.execute('SELECT COUNT(*) FROM "Lap"').fetchone()
            con.close()
        except Exception:
            # No committed Lap table: the process died before the write
            # finished, so there is nothing to recover from this file.
            print(f"Found incomplete temp file {stray.name} (no committed data "
                  f"— an earlier save was interrupted). It cannot be recovered "
                  f"and can be deleted.", flush=True)
            continue
        os.replace(stray, final_path)
        print(f"Recovered orphaned session file: {final_path.name}")

    for _sp in sorted(output_dir.glob(".*.spool")):
        print(f"Found an unconverted session safety copy: {_sp.name}\n"
              f"  Convert it with:  uv run recorder.py --recover", flush=True)

    print(f"lmu-coach recorder v4  |  {poll_hz:.0f}Hz  |  "
          f"10Hz geometry  |  out={output_dir}")
    print("Waiting for LMU...\n")

    mmap_ctrl: MMapControl | None = None
    conn_state                    = ConnectionState()
    buf: SessionBuffer | None     = None
    last_lap: int | None          = None

    origin_x: float | None = None
    origin_z: float | None = None

    # Shared with the signal handler below. "writing" is True only while a
    # session file is being written; "stop_pending" records a stop request
    # that arrived during that window so it is honoured right AFTER the
    # save finishes instead of aborting it.
    _stop_state = {"writing": False, "stop_pending": False}

    def flush(reason: str) -> None:
        nonlocal buf, last_lap
        if buf is None:
            return
        if buf.sample_count < MIN_SAMPLES:
            print(f"  Discarding ({reason}, {buf.sample_count} samples < {MIN_SAMPLES})")
        else:
            # The most recent entry in buf.lap_events marks the START of
            # whatever lap was in progress when we stopped (Ctrl+C, session
            # end, car/track change). That lap was never completed, so it
            # has no real duration — writing it into Lap/Lap Time anyway
            # produces a bogus short "lap" at the end of the file (this is
            # what looked like laps getting merged: a real lap's tail was
            # cut off and saved as its own fake short lap). Drop it before
            # writing, unless it's the session's only lap (nothing to
            # compare it against, and dropping it would discard everything).
            if len(buf.lap_events) > 1:
                incomplete_ts, incomplete_lap_num = buf.lap_events[-1]
                buf.lap_events = buf.lap_events[:-1]
                # lap_time_events has one row per lap_events row in the same
                # order (see register_lap) — drop its matching last row too.
                if buf.lap_time_events:
                    buf.lap_time_events = buf.lap_time_events[:-1]
                print(f"  Dropping incomplete final lap {incomplete_lap_num} "
                      f"(in progress at stop, {reason})")
            # write_session() is isolated in its own try/except so that a
            # write-time bug (bad tuple shape, disk full, etc.) is never
            # mistaken for a shared-memory disconnect by the caller's outer
            # (OSError, ValueError) handler — that outer handler previously
            # swallowed exactly this kind of failure and silently produced
            # a truncated file (the v3 bug). A write failure here is loud,
            # distinct, and does NOT get reinterpreted as "reconnecting".
            print(f"  Saving session ({buf.sample_count} samples, "
                  f"{len(buf.lap_events)} laps) - do not close this window...",
                  flush=True)
            _stop_state["writing"] = True
            spool_path = None
            try:
                # Safety copy FIRST: a plain dump of the raw buffer is fast
                # (~1-2s) and independent of the slower DuckDB conversion.
                # If the conversion below is interrupted or stalls, this
                # file still holds the whole session and can be turned into
                # a .duckdb later with:  uv run recorder.py --recover
                _ts = time.monotonic()
                spool_path = spool_session(buf, output_dir)
                print(f"  Safety copy written: {spool_path.name} "
                      f"({time.monotonic() - _ts:.1f}s)", flush=True)
            except Exception as e:
                print(f"  (could not write safety copy: {type(e).__name__}: {e})",
                      flush=True)
            try:
                path = write_session(buf, output_dir)
                print(f"  Saved: {path.name}  "
                      f"({buf.sample_count} samples, {len(buf.lap_events)} laps)",
                      flush=True)
                if spool_path is not None:
                    spool_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"  !!! WRITE FAILURE ({reason}): session NOT saved "
                      f"correctly — {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                _stop_state["writing"] = False
        buf      = None
        last_lap = None

    # ── Signal handling ──────────────────────────────────────────────────────
    # BUG FIX: previously the ONLY clean-shutdown path was the
    # `except KeyboardInterrupt` block at the bottom of this function, which
    # only fires on Ctrl+C (SIGINT). Any other way of stopping the recorder
    # — closing the terminal/console window, "End Task" in Task Manager,
    # a supervising process sending SIGTERM (exactly what a future web-UI
    # start/stop button would need to do to stop this process cleanly) —
    # does not raise KeyboardInterrupt at all, so flush() never ran and
    # whatever session was in memory was lost with no error. Registering
    # explicit handlers for SIGTERM (any clean "please stop" request) and,
    # on Windows, SIGBREAK (fired when a console window is closed) means
    # every one of those paths now flushes before exiting, same as Ctrl+C
    # already did.
    def _handle_stop_signal(signum, frame):
        # A stop request that lands WHILE the session file is being written
        # must not interrupt the write: raising KeyboardInterrupt inside
        # write_session() rolls the transaction back and deletes the .tmp
        # file, i.e. the whole session is lost — the exact failure seen in
        # practice. Defer instead; the main loop exits right after the save.
        if _stop_state["writing"]:
            _stop_state["stop_pending"] = True
            print("  Stop requested — finishing the save first...", flush=True)
            return
        raise KeyboardInterrupt

    for _sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        _sig = getattr(signal, _sig_name, None)  # SIGBREAK exists on Windows only
        if _sig is None:
            continue
        try:
            signal.signal(_sig, _handle_stop_signal)
        except (ValueError, AttributeError, OSError):
            pass  # not available on this platform/thread — degrade gracefully

    try:
        while True:
            if _stop_state["stop_pending"] or STOP_FILE.exists():
                raise KeyboardInterrupt  # stop requested; any save has finished
            # ── Connect ───────────────────────────────────────────────────
            if mmap_ctrl is None:
                try:
                    mmap_ctrl = MMapControl(LMUConstants.LMU_SHARED_MEMORY_FILE,
                                            lmu_data.LMUObjectOut)
                    mmap_ctrl.create(access_mode=0)
                    print("Connected to LMU shared memory.")
                    origin_x = origin_z = None
                except Exception:
                    time.sleep(2.0)
                    continue

            try:
                mmap_ctrl.update()
                data = mmap_ctrl.data

                num_v    = data.scoring.scoringInfo.mNumVehicles
                si       = local_scoring_index(data.scoring.vehScoringInfo)
                if si == INVALID_INDEX:
                    time.sleep(0.5)
                    continue

                pscor = data.scoring.vehScoringInfo[si]
                tmap  = build_tele_index_map(num_v, data.telemetry)
                ti    = tmap.get(pscor.mID, INVALID_INDEX)
                if ti == INVALID_INDEX:
                    time.sleep(0.5)
                    continue

                telem       = data.telemetry.telemInfo[ti]
                car_name    = _safe_decode(telem.mVehicleName)
                track_name  = _safe_decode(telem.mTrackName)
                driver_name = _safe_decode(pscor.mDriverName)
                lap_number  = telem.mLapNumber
                # Clock source fix: scoringInfo.mCurrentET only refreshes at
                # ~5Hz (LMU's scoring buffer is not physics-critical and is
                # updated far less often than telemetry), which is why every
                # recorded timestamp was landing on one of a handful of
                # values per second — visible as GPS Time jumping in ~0.2s
                # steps instead of the expected 0.01s steps, and Lap/Lap Time
                # durations that rounded to the nearest ~0.1-0.2s versus
                # native LMU's true ~0.01s precision (verified: native's own
                # GPS Time increments every single 100Hz sample; the old
                # scoring-based current_et here only changed every 5th-20th
                # sample). telemInfo (the struct `telem` already comes from,
                # sampled every poll at 100Hz) carries its own per-vehicle
                # mElapsedTime ("game session time") on the SAME clock as
                # scoringInfo.mCurrentET, but refreshed every physics tick —
                # this is almost certainly what LMU's own native exporter
                # reads from, which is why native files have full precision.
                current_et  = float(telem.mElapsedTime)

                if not car_name:
                    time.sleep(0.5)
                    continue

                if origin_x is None:
                    origin_x = telem.mPos.x
                    origin_z = telem.mPos.z

                # ── Session boundary detection ─────────────────────────────
                reset, just_froze = conn_state.update(
                    current_et, pscor.mInGarageStall, pscor.mID
                )
                car_changed = (buf is not None and
                               (buf.car_name   != car_name or
                                buf.track_name != track_name))
                if reset or car_changed:
                    flush("session reset" if reset else "car/track change")
                    origin_x = telem.mPos.x
                    origin_z = telem.mPos.z
                elif just_froze:
                    # Session went idle (quit to menu, paused, sitting at a
                    # loading screen) — flush now rather than holding data
                    # in memory indefinitely. See ConnectionState.update's
                    # docstring for why this matters: it's what protects
                    # against losing a whole session to an unclean process
                    # kill that never reaches the KeyboardInterrupt handler.
                    flush("session went idle")

                if conn_state.is_frozen:
                    time.sleep(interval)
                    continue

                # ── Create new session buffer ──────────────────────────────
                if buf is None:
                    buf = SessionBuffer(
                        car_name, track_name, driver_name,
                        origin_x=origin_x or 0.0,
                        origin_z=origin_z or 0.0,
                        poll_hz=poll_hz,
                        session_et_start=current_et,
                    )
                    last_lap = lap_number
                    buf.register_lap(lap_number, current_et)
                    print(f"Session: {track_name} / {car_name} "
                          f"(driver: {driver_name})")

                # ── Lap crossing ───────────────────────────────────────────
                if lap_number != last_lap:
                    buf.register_lap(lap_number, current_et)
                    last_lap = lap_number

                buf.add_sample(telem, telem.mWheels, pscor, current_et)

            except (OSError, ValueError) as e:
                print(f"Lost shared memory ({e}); reconnecting…")
                flush("connection lost")
                if mmap_ctrl is not None:
                    try:
                        mmap_ctrl.close()
                    except Exception:
                        pass
                mmap_ctrl = None
                conn_state = ConnectionState()
                origin_x = origin_z = None
                time.sleep(2.0)
                continue

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
        flush("user stopped")
        if mmap_ctrl is not None:
            mmap_ctrl.close()
        STOP_FILE.unlink(missing_ok=True)
        print("Recorder exited cleanly.", flush=True)


SPOOL_FIELDS = ("car_name", "track_name", "driver_name", "poll_hz", "start_wall",
                "sample_count", "lap_events", "lap_time_events", "gear_events",
                "abs_events", "tc_events", "pits_events")


def spool_session(buf, output_dir: Path) -> Path:
    """Dump the raw in-memory session to disk as fast as possible."""
    name = buf.start_wall.strftime("%Y-%m-%dT%H_%M_%SZ")
    final = output_dir / f".{name}.spool"
    part = output_dir / f".{name}.spool.part"
    data = {k: getattr(buf, k) for k in SPOOL_FIELDS}
    data["s"] = dict(buf.s)
    with open(part, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(part, final)
    return final


def load_spool(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    s = defaultdict(list)
    s.update(data.pop("s"))
    return types.SimpleNamespace(s=s, **data)


def recover_spools(output_dir: Path) -> None:
    spools = sorted(output_dir.glob(".*.spool"))
    if not spools:
        print(f"No unconverted safety copies in {output_dir}")
        return
    for sp in spools:
        print(f"Converting {sp.name} ...", flush=True)
        try:
            out = write_session(load_spool(sp), output_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}  (safety copy kept)", flush=True)
            continue
        sp.unlink(missing_ok=True)
        print(f"  Recovered: {out.name}", flush=True)


def _probe_methods(rows: int = 5000) -> None:
    """Time the three ways of getting Python floats into DuckDB, on this PC:
    (1) the ORIGINAL recorder's executemany, (2) a list parameter, (3) CSV."""
    import random
    vals = [random.random() for _ in range(rows)]
    con = duckdb.connect(":memory:")
    for t in "abc":
        con.execute(f"CREATE TABLE {t}(v DOUBLE)")
    t = time.monotonic()
    con.executemany("INSERT INTO a VALUES (?)", [(v,) for v in vals])
    t_orig = time.monotonic() - t
    t = time.monotonic()
    con.execute("INSERT INTO b SELECT unnest(?)", [vals])
    t_param = time.monotonic() - t
    p = Path(tempfile.gettempdir()) / f"lmu_probe_{os.getpid()}.csv"
    t = time.monotonic()
    p.write_text("\n".join(map(repr, vals)) + "\n")
    con.execute(f"COPY c FROM '{p.as_posix()}' (FORMAT csv, HEADER false)")
    t_csv = time.monotonic() - t
    p.unlink(missing_ok=True)
    got = [[r[0] for r in con.execute(f"SELECT v FROM {x}").fetchall()] for x in "abc"]
    same = got[0] == got[1] == got[2] == vals
    k = 150000 / rows
    print(f"Method probe ({rows:,} values), identical data: {same}", flush=True)
    print(f"  original executemany : {t_orig:6.2f}s  -> ~{t_orig * k:5.0f}s per 150,000 values", flush=True)
    print(f"  list parameter       : {t_param:6.2f}s  -> ~{t_param * k:5.0f}s per 150,000 values", flush=True)
    print(f"  CSV COPY (now used)  : {t_csv:6.2f}s  -> ~{t_csv * k:5.0f}s per 150,000 values", flush=True)


def selftest(minutes: float) -> None:
    """Write a synthetic session of realistic size on THIS machine and time
    every step. No LMU or driving needed. If this is slow, the problem is the
    save path on this PC, not the session data."""
    import random
    hz = 100.0
    n = int(minutes * 60 * hz)
    n10 = n // 10
    _probe_methods()
    print(f"Self-test: synthetic {minutes:g}-minute session ({n:,} samples)", flush=True)
    s = defaultdict(list)
    for c in ("GPS Time", "Pos X", "Pos Z", "Ground Speed", "Steering Pos", "Throttle Pos",
              "Brake Pos", "Yaw Rate", "Engine RPM", "G Force Lat", "G Force Long", "G Force Vert"):
        s[c] = [random.random() for _ in range(n)]
    for c in ("Wheel Speed", "Slip Angle", "Susp Pos"):
        s[c] = [tuple(random.random() for _ in range(4)) for _ in range(n)]
    for c in ("Grip Fraction", "Tire Load", "Tire Lateral Force", "Tire Longitudinal Force"):
        s[c] = [(0.0, 0.0, 0.0, 0.0)] * n
    for c in ("GPS Latitude", "GPS Longitude", "GPS Speed", "Path Lateral", "Track Edge", "Lap Dist"):
        s[c] = [random.random() for _ in range(n10)]
    for c in ("TyresRubberTemp", "TyresPressure"):
        s[c] = [tuple(random.random() for _ in range(4)) for _ in range(n10)]
    laps = [(0.4, 0), (120.0, 1), (240.0, 2)]
    buf = types.SimpleNamespace(
        poll_hz=hz, start_wall=datetime.now(timezone.utc), track_name="Selftest Track",
        car_name="Selftest Car", driver_name="selftest", sample_count=n, s=s,
        lap_events=laps, lap_time_events=[(0.4, 0.0), (120.0, 119.6), (240.0, 120.0)],
        gear_events=[(0.4, 3)], abs_events=[(0.4, 0.0)], tc_events=[(0.4, 0.0)],
        pits_events=[(0.4, 0.0)])
    tmp = Path(tempfile.mkdtemp(prefix="lmu_selftest_"))
    try:
        t = time.monotonic()
        sp = spool_session(buf, tmp)
        print(f"Safety copy: {time.monotonic() - t:.1f}s ({sp.stat().st_size / 1e6:.0f} MB)", flush=True)
        t = time.monotonic()
        out = write_session(buf, tmp)
        total = time.monotonic() - t
        con = duckdb.connect(str(out), read_only=True)
        rows = con.execute('SELECT COUNT(*) FROM "Ground Speed"').fetchone()[0]
        laps_n = con.execute('SELECT COUNT(*) FROM "Lap"').fetchone()[0]
        con.close()
        ok = rows == n and laps_n == len(laps)
        print(f"DuckDB write: {total:.1f}s total  |  rows={rows:,} (expected {n:,})  "
              f"laps={laps_n}  ->  {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll-hz",    type=float, default=DEFAULT_POLL_HZ)
    parser.add_argument("--recover", action="store_true",
                        help="convert any unconverted safety copies to .duckdb files, then exit")
    parser.add_argument("--selftest", action="store_true",
                        help="time a synthetic-session save on this machine, then exit")
    parser.add_argument("--selftest-minutes", type=float, default=25.0)
    args = parser.parse_args()
    if args.selftest:
        selftest(args.selftest_minutes)
    elif args.recover:
        recover_spools(args.output_dir)
    else:
        run_recorder(args.output_dir, args.poll_hz)
