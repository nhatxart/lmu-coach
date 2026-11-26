"""
LMU-Coach recorder (format HF1) - THE recorder. Run it standalone with
START_RECORDER.ps1, or let the host agent (recorder_control.py, the web UI's
Record tab) start and stop it.

It records automatically and needs no input: it waits for LMU, records every
session it finds, saves the file when you return to the garage or quit the
session, and picks the next session up by itself. See ConnectionState.

History: this file used to be the experimental "recorder_hf.py" next to the
original v4 recorder.py. The two were merged on 10/19/2026 - the v4 recorder was
retired, this one took its name, launcher and stop file. Files written by
either still read in the backend. What this recorder does that v4 did not:
  - Stores ONE sample per game tick (when telem.mElapsedTime advances), so a
    100 Hz game clock yields a clean 100 Hz file instead of ~16% duplicated
    and ~20% skipped ticks (measured on a v4 file). Every row is stamped with
    the game's own clock; lap times come from the game's own mLastLapTime.
  - Follows the game's updates (TickSync, the default): sleep to just before
    the next ~10 ms update, watch the player's clock in the live memory, copy
    the 317 KiB block once. About 1% of a core and 100 updates/s on a real
    drive, versus 96% of a core for fixed 1000 Hz polling (--poll-hz 1000;
    --every-poll also polls at a fixed rate, for raw inspection). It must run
    at NORMAL priority: LMU runs at High, and below-normal it lost 9% to 44%
    of the updates. A save that lost more than 2% says so (skip_warning).
    resolve_poll_mode has the measurements.
  - Measures what the game actually publishes: polls, unique ticks, the
    histogram of clock steps, and polls where car data changed without the
    clock moving. --probe SECONDS does only that and records nothing.
  - 10 Hz channels are gated by game time, not by sample count.
  - channelsList.frequency is the MEASURED rate, not the nominal one.
  - Lap Time uses LMU's own mLastLapTime (1 ms precision) when it agrees
    with the clock difference.
  - Records the extra channels a force-based tyre model needs (local
    velocity, suspension force, camber, patch velocities, wear, ...).
Everything below this paragraph is the original v4 description, still true.

Records the FULL session into a single DuckDB file,
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
  100 Hz  — position, speed, inputs, G-forces, yaw, wheel/slip/susp, RPM,
            Brake Pressure (per wheel, raw; 11/26/2026)
  10 Hz   — track geometry (Path Lateral, Track Edge, Lap Dist), tyre thermals
  Event   — Gear, ABS, TC, In Pits, Lap, Lap Time (on change / on crossing),
            Rear Brake Bias, Steering Wheel Range, Visual Steering Wheel Range
            (on change; 11/26/2026)
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
from collections import Counter, defaultdict, deque
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
DEFAULT_POLL_HZ     = 1000.0
# DO NOT rename. The file format is called HF1 and the backend recognises
# these files by " hf" in this string (telemetry.session_time_axes); every
# recording made so far carries it.
RECORDER_ID         = "lmu-coach-recorder hf1"
# Stop request channel used by recorder_control.py (which writes this exact
# file name). A plain file is used instead of a signal because on Windows
# Popen.terminate() does NOT deliver a signal (it force-kills), so signal
# handlers can never run there. The recorder polls for this file, finishes
# any save in progress, writes the current session, and exits cleanly.
STOP_FILE           = Path(__file__).parent / ".stop_requested"
STOP_CHECK_S        = 0.1         # a file stat per poll is too slow at 1000 Hz
# Safety copies are written as ".<start time>.spool". Until 10/19/2026 this
# recorder used ".spool_hf" so that the v4 recorder's --recover could not
# pick its files up; v4 is gone, so new copies use ".spool", and --recover
# (and the startup notice) still pick up a leftover ".spool_hf".
SPOOL_SUFFIX        = ".spool"
LEGACY_SPOOL_SUFFIXES = (".spool_hf",)
FREEZE_TIMEOUT_S    = 2.0
SCORING_DECIMATE    = 10          # 10 Hz for geometry + thermal channels
SLOW_PERIOD_S       = 0.1         # the same 10 Hz, as a game-time period
OFFICIAL_LAP_WINDOW_S    = 3.0    # how long after a crossing mLastLapTime may arrive
OFFICIAL_LAP_TOLERANCE_S = 0.05   # it must agree with the clock difference this well
MIN_SAMPLES         = 200         # discard sessions shorter than this

# Pseudo-GPS origin — overridden per session from first mPos sample
GPS_REF_LAT         = 35.37
GPS_REF_LON         = 138.92
R_DEG               = 111_320.0  # metres per degree of latitude


# Extra per-wheel channels (table name, LMUWheel field), stored every sample.
HF_WHEEL_FIELDS = (
    ("Susp Force",             "mSuspForce"),
    ("Ride Height",            "mRideHeight"),
    ("Camber",                 "mCamber"),
    ("Toe",                    "mToe"),
    ("Lateral Patch Vel",      "mLateralPatchVel"),
    ("Longitudinal Patch Vel", "mLongitudinalPatchVel"),
    ("Lateral Ground Vel",     "mLateralGroundVel"),
    ("Longitudinal Ground Vel", "mLongitudinalGroundVel"),
    ("Tire Deflection",        "mVerticalTireDeflection"),
    ("Surface Type",           "mSurfaceType"),
)
HF_WHEEL_UNITS = {
    "Susp Force": "N", "Ride Height": "m", "Camber": "rad", "Toe": "rad",
    "Lateral Patch Vel": "m/s", "Longitudinal Patch Vel": "m/s",
    "Lateral Ground Vel": "m/s", "Longitudinal Ground Vel": "m/s",
    "Tire Deflection": "m",
    "Surface Type": "0=dry 1=wet 2=grass 3=dirt 4=gravel 5=kerb 6=special",
}

# Driver-input channels (11/26/2026, for the coach's "ideal inputs" maps; the
# design's Q3). Brake Pressure is per wheel at 100 Hz like Brake Pos, because it
# is what the pedal turns into at each corner. lmu_data.py says mBrakePressure
# is "currently 0.0-1.0 ... will convert to true brake pressure (kPa) in
# future", so it is stored RAW and the unit says which scale it was; a later
# game version may switch to kPa without a recorder change, and a reader must
# check the unit (or the GameVersion) before comparing files.
BRAKE_PRESSURE_UNIT = "raw (0-1 now; LMU plans kPa)"
# Setup values that only change when the driver adjusts them (bias in the car,
# steering lock in the garage), so they are change-only event tables (ts, value)
# like Gear/ABS/TC, NOT 100 Hz or 10 Hz scalars: one row per change instead of
# 360,000 identical rows per hour, the exact time of a mid-lap bias change, and
# lap_cut seeds an event table's value at the start of a kept lap
# (_seed_event), so a kept lap still knows its bias. (table, telem field, unit)
SETTING_EVENT_FIELDS = (
    ("Rear Brake Bias",             "mRearBrakeBias",              "fraction on rear"),
    # Units are not documented in lmu_data.py; rFactor 2 used degrees lock to lock.
    ("Steering Wheel Range",        "mPhysicalSteeringWheelRange", "deg (unverified)"),
    ("Visual Steering Wheel Range", "mVisualSteeringWheelRange",   "deg (unverified)"),
)


# ── Shared-memory helpers ────────────────────────────────────────────────────

def local_scoring_index(scor_veh) -> int:
    for idx, veh in enumerate(scor_veh):
        if veh.mIsPlayer:
            return idx
    return INVALID_INDEX


def build_tele_index_map(num_vehicles: int, tele_data) -> dict[int, int]:
    return {tele_data.telemInfo[i].mID: i for i in range(num_vehicles)}


def _fmt_lap(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


def _safe_decode(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _session_meta(data, player_scor) -> dict[str, str]:
    """Per-session facts that live outside the telemetry slot: car class,
    game build and weather temps. Written once to the metadata table (never
    per sample). WHY: tyre force channels are populated by class and the
    game build changes what LMU exposes, so a reader needs both to know
    which files to trust. A missing source gives "" rather than a missing
    key, so readers can tell "not recorded" from "recorder too old"."""
    def get(fn) -> str:
        try:
            return fn()
        except (AttributeError, IndexError, TypeError, ValueError):
            return ""
    info = getattr(getattr(data, "scoring", None), "scoringInfo", None)
    return {
        "VehicleClass": get(lambda: _safe_decode(player_scor.mVehicleClass)),
        "GameVersion":  get(lambda: str(int(data.generic.gameVersion))),
        "TrackTempC":   get(lambda: f"{float(info.mTrackTemp):.1f}"),
        "AmbientTempC": get(lambda: f"{float(info.mAmbientTemp):.1f}"),
    }


def _slip_angle(lat_vel: float, lon_vel: float) -> float:
    return math.atan(lat_vel / lon_vel) if lon_vel else 0.0


def _find_player(data, si: int, ti: int) -> tuple[int, int]:
    """Return (scoring index, telemetry index) of the player's car, or
    INVALID_INDEX for whichever is not found. si/ti are the previous
    answers: at 1000 Hz, rescanning every vehicle slot on every poll is the
    most expensive thing in the loop, so they are re-checked first and the
    full scan only runs when they stop matching."""
    num_v = data.scoring.scoringInfo.mNumVehicles
    scor = data.scoring.vehScoringInfo
    if not (0 <= si < num_v and scor[si].mIsPlayer):
        si = local_scoring_index(scor)
        if si == INVALID_INDEX:
            return INVALID_INDEX, INVALID_INDEX
    slot = scor[si].mID
    if not (0 <= ti < num_v and data.telemetry.telemInfo[ti].mID == slot):
        ti = build_tele_index_map(num_v, data.telemetry).get(slot, INVALID_INDEX)
    return si, ti


def _effective_hz(times: list[float], fallback: float) -> float:
    """Measured sample rate of a list of game timestamps."""
    if len(times) < 2 or times[-1] <= times[0]:
        return fallback
    return (len(times) - 1) / (times[-1] - times[0])


# ── Poll pacing and measurement ──────────────────────────────────────────────

class Pacer:
    """Holds a fixed poll interval against a perf_counter deadline.

    WHY not time.sleep(interval): measured on this PC, time.sleep(0.001)
    really takes ~1.5 ms, so a sleep-only loop tops out near 650 Hz. The
    pacer sleeps until SPIN_S before the deadline and spins the rest.

    CPU COST: the sleep part only exists when the interval is longer than
    SPIN_S. At 500 Hz and above the whole interval is spin, i.e. one CPU
    core fully busy next to the game. Measured on this PC: 1000 Hz 99%,
    500 Hz 98%, 400 Hz 72%, 100 Hz 19% of one core. That is the price of
    the 1000 Hz test; for everyday recording use --poll-hz 200 or lower.
    """
    SPIN_S = 0.002

    def __init__(self, interval_s: float):
        self.interval = interval_s
        self._next = time.perf_counter() + interval_s

    def wait(self) -> None:
        now = time.perf_counter()
        if now - self._next > 5 * self.interval:
            # Fell well behind (a save, a stall, a frozen session): resync
            # instead of firing a burst of back-to-back polls to catch up.
            self._next = now + self.interval
        remaining = self._next - now
        if remaining > self.SPIN_S:
            time.sleep(remaining - self.SPIN_S)
        while time.perf_counter() < self._next:
            time.sleep(0)  # yield the rest of the timeslice; returns at once
        self._next += self.interval


class TickSync:
    """Waits for the game's next telemetry update without spinning or copying.

    WHY (10/21/2026): the fixed-rate Pacer at 1000 Hz keeps a CPU core 99%
    busy and copies the WHOLE shared-memory block (every car's data) on every
    poll, to catch updates that arrive every ~10 ms. That showed up as stutter
    in LMU even on an undervolted 9800X3D with SMT off. The game's own clock
    (mElapsedTime) is what stamps every row and decides whether a sample is
    new, so all this needs is to look at that one number cheaply:

      1. sleep until just before the next update is expected (a real sleep,
         no spinning),
      2. look at the player's clock in the LIVE memory every STEP_S - a single
         double, no copy - until it changes,
      3. return so the caller copies the block once and stores the row.

    The expected period is learned from the arrivals (the game does not tick
    at exactly 100 Hz). If the clock does not move for GIVE_UP_S (paused game,
    menu, loading screen, or a stale player slot) it returns False and the
    caller polls normally, so the freeze/garage detection keeps running. If
    the live clock cannot be read at all (peek returns None) it degrades to
    one poll per expected tick.

    clock/sleep are injectable so the logic can be tested without real time.
    """
    NOMINAL_S = 0.010
    EARLY_S = 0.0015      # wake this long before the expected update
    STEP_S = 0.001        # time.sleep(0.001) really takes ~1.5 ms on this PC
    GIVE_UP_S = 0.050
    PERIOD_MIN_S = 0.004
    PERIOD_MAX_S = 0.020

    def __init__(self, clock=None, sleep=None):
        self._clock = clock or time.perf_counter
        self._sleep = sleep or time.sleep
        self.period = self.NOMINAL_S
        self.gave_up = 0
        self._t_seen: float | None = None
        self._last_wait_et: float | None = None
        # Diagnostics: how late the long sleep returned (ms) and how many
        # looks each wait needed. The first real drive (10/21/2026) caught
        # only 56 of 100 updates a second with no idea why; these numbers
        # say whether the scheduler woke the thread late.
        self.waits = 0
        self._looks = 0
        self._wake_late_ms: deque = deque(maxlen=20000)

    def wait(self, peek, last_et: float) -> bool:
        """Block until peek() differs from last_et. True: a new update was
        seen. False: gave up (or no live clock) - poll anyway."""
        clock, sleep = self._clock, self._sleep
        self.waits += 1
        # Asked about the SAME tick twice in a row while the live clock has
        # already moved on: the copy did not follow. MMapControl.update()
        # copies only when LMU's update flag is set (and the car counts agree),
        # otherwise the loop is handed the previous tick again. Retry after one
        # short sleep. A full period sleep would cost the next tick, and the
        # short gap must not be learned as the game's period (that collapsed
        # the period toward PERIOD_MIN_S in review). A paused game also asks
        # twice, but its live clock has NOT moved, so it takes the normal path.
        repeat = last_et == self._last_wait_et
        self._last_wait_et = last_et
        if repeat:
            live = peek()
            if live is not None and live != last_et:
                sleep(self.STEP_S)
                self._t_seen = clock()
                return True
        start = self._t_seen if self._t_seen is not None else clock()
        now = clock()
        target = start + self.period - self.EARLY_S
        if target > now:
            sleep(target - now)
            self._wake_late_ms.append((clock() - target) * 1000.0)
        while True:
            self._looks += 1
            et = peek()
            now = clock()
            if et is None:
                self._t_seen = now
                return False
            if et != last_et:
                self._learn(now)
                return True
            if now - start >= self.GIVE_UP_S:
                self.gave_up += 1
                self._t_seen = now
                return False
            sleep(self.STEP_S)

    def diagnostics(self) -> dict:
        v = sorted(self._wake_late_ms)

        def q(p: float) -> float:
            return v[min(len(v) - 1, int(len(v) * p))] if v else 0.0

        return {
            "waits": self.waits,
            "gave_up": self.gave_up,
            "period_ms": self.period * 1000.0,
            "looks_per_tick": self._looks / max(self.waits, 1),
            "wake_late_ms_p50": q(0.5),
            "wake_late_ms_p90": q(0.9),
            "wake_late_ms_p99": q(0.99),
            "wake_late_ms_max": v[-1] if v else 0.0,
        }

    def _learn(self, now: float) -> None:
        if self._t_seen is not None:
            dt = now - self._t_seen
            # Only an ordinary tick-to-tick interval teaches the period. A gap
            # of two or more updates (the game skipped one, or we woke late)
            # is not a slower game; 1.5x leaves room for the wake-up jitter.
            if 0.5 * self.period <= dt <= 1.5 * self.period:
                self.period = min(self.PERIOD_MAX_S,
                                  max(self.PERIOD_MIN_S, 0.9 * self.period + 0.1 * dt))
        self._t_seen = now


def peek_elapsed_time(mmap_ctrl, ti: int) -> float | None:
    """The player's mElapsedTime read straight from the live shared memory,
    without the full copy mmap_ctrl.update() makes. None if it cannot be read.

    Uses MMapControl._realtime (the live view the copy is made from; set in
    the copy access mode this recorder uses). It is a private attribute of the
    vendored library, so every failure maps to None and the caller falls back
    to timed polling instead of crashing."""
    live = getattr(mmap_ctrl, "_realtime", None)
    if live is None or ti is None or ti < 0:    # -1 would index the LAST car
        return None
    try:
        return float(live.telemetry.telemInfo[ti].mElapsedTime)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


TICK_NOMINAL_HZ = 100.0
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def resolve_poll_mode(poll_hz: float | None, every_poll: bool,
                      tick_sync: bool = False) -> tuple[str, float]:
    """("tick", nominal Hz) = follow the game's updates (the default);
    ("fixed", Hz) = poll at a fixed rate. A given --poll-hz, or --every-poll
    (which stores every poll and so needs a poll clock of its own), selects
    fixed polling. An explicit tick_sync together with either is a conflict.

    HISTORY (10/21/2026, real drives with --probe 30): tick-sync first caught
    56 of the game's 100 updates a second (1,323 skipped) and was made opt-in.
    The same session then ran an A/B, one change at a time:
      tick-sync, below-normal priority: 91 Hz, 286 skipped, wake-ups up to 4.9 ms late
      tick-sync, NORMAL priority:       100 Hz,   6 skipped, wake-ups <= 1.0 ms late
      fixed 200 Hz, below-normal:        99 Hz,  38 skipped, one loop stall of 8.9 ms
      fixed 1000 Hz, below-normal:      100 Hz,   1 skipped, 96% of a core
    LMU runs at High priority, so a SLEEPING thread at a lower priority waits
    for a core when it wakes; a spinning one never gives its core up. Hence:
    tick-sync is the default and the recorder keeps normal priority."""
    if tick_sync and (poll_hz is not None or every_poll):
        raise ValueError("--tick-sync cannot be combined with --poll-hz or --every-poll")
    if poll_hz is None and not every_poll:
        return "tick", TICK_NOMINAL_HZ
    return "fixed", poll_hz if poll_hz is not None else DEFAULT_POLL_HZ


SKIP_WARN_FRACTION = 0.02


def skip_warning(stats) -> str | None:
    """A message when more than 2% of the game's updates are missing from a
    recording, else None. The game skips a few itself (about 0.2% on a real
    drive); a larger share means the recorder fell behind, and nothing else
    would have told the driver that rows are missing."""
    seen = stats.ticks + stats.skipped_ticks
    if seen <= 0 or stats.skipped_ticks / seen <= SKIP_WARN_FRACTION:
        return None
    return (f"WARNING: {stats.skipped_ticks:,} of {seen:,} game updates "
            f"({100.0 * stats.skipped_ticks / seen:.1f}%) were lost in this recording. The game "
            f"itself skips about 0.2%; more means the recorder fell behind. Compare with "
            f"--poll-hz 1000 (uv run recorder.py --probe 30 --poll-hz 1000).")


def lower_priority(kernel32=None) -> bool:
    """Run below normal priority so the recorder yields to the game. Windows
    only; any failure is not fatal. kernel32 is injectable for tests."""
    if kernel32 is None:
        if sys.platform != "win32":
            return False
        import ctypes
        kernel32 = ctypes.windll.kernel32
    try:
        return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(),
                                              BELOW_NORMAL_PRIORITY_CLASS))
    except Exception:  # noqa: BLE001 - never let a tuning nicety stop a recording
        return False


class PollStats:
    """Counts what the game actually publishes versus how often we look."""

    def __init__(self):
        self.polls = 0
        self.ticks = 0                       # polls where mElapsedTime was new
        self.data_changed_without_tick = 0   # car data moved, clock did not
        # Game updates that never reached the file: a clock step of 15 ms or
        # more means at least one 10 ms update in between was not seen. It
        # cannot say WHO missed it - the game skips ticks by itself (18 s of
        # them in one Fuji lap), and a recorder that wakes up late misses
        # them too - so compare the two polling modes on the same drive.
        self.skipped_ticks = 0
        self._steps: Counter = Counter()     # clock step in 0.1 ms units
        self._last_et: float | None = None

    TICK_UNITS = 100            # one game update, in 0.1 ms units
    SKIP_THRESHOLD_UNITS = 150  # 15 ms

    def record(self, et: float, data_changed: bool) -> None:
        self.polls += 1
        if self._last_et is None or et != self._last_et:
            if self._last_et is not None and et > self._last_et:
                step = round((et - self._last_et) * 1e4)
                self._steps[step] += 1
                if step >= self.SKIP_THRESHOLD_UNITS:
                    self.skipped_ticks += round(step / self.TICK_UNITS) - 1
            self.ticks += 1
            self._last_et = et
        elif data_changed:
            self.data_changed_without_tick += 1

    def histogram(self) -> dict[str, int]:
        """Clock steps, most common first, e.g. {"10.0ms": 5210}."""
        return {f"{k / 10:.1f}ms": n for k, n in self._steps.most_common()}

    def summary(self, wall_s: float) -> list[str]:
        wall_s = max(wall_s, 1e-9)
        steps = ", ".join(f"{k} x{n:,}" for k, n in list(self.histogram().items())[:6])
        return [
            f"polls: {self.polls:,} in {wall_s:.1f}s = {self.polls / wall_s:.0f} Hz achieved",
            f"unique game ticks: {self.ticks:,} = {self.ticks / wall_s:.0f} Hz published",
            f"clock steps: {steps or 'none'}",
            f"skipped game updates (clock steps >= 15 ms): {self.skipped_ticks:,}",
            f"car data changed without a clock tick: {self.data_changed_without_tick:,} polls",
        ]


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
        self._last_setting: dict[str, float] = {}   # SETTING_EVENT_FIELDS

        # Game time at which the next 10 Hz row is due (see add_sample).
        self._next_slow_et: float = session_et_start

        # Static per-session facts and poll statistics, written to the
        # metadata table. Filled on the first sample and at flush.
        self.meta: dict[str, str] = {}

        # LMU's own lap time (see update_official_lap_time).
        self._official_used: float | None = None      # last mLastLapTime dealt with
        self._official_pending_since: float | None = None
        self._official_watch_since: float | None = None   # missed; watching for a late value

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
            self._official_pending_since = current_et
            self._official_watch_since = None
        # DIFFERENT FROM the retired v4 recorder: no (ts, 0.0) placeholder row for the
        # first crossing. With the placeholder, row i held the duration of
        # lap i-1, while the backend's _read_lap_time_events matches row i
        # to Lap row i - the source of its "duration mismatch" warnings.
        # One row per COMPLETED lap keeps the two tables aligned by position.

        self._last_lap       = lap_number
        self._last_et_at_lap = current_et

    def update_official_lap_time(self, last_lap_time: float, current_et: float) -> None:
        """Swap the newest Lap Time for LMU's own mLastLapTime when it arrives.

        WHY: the clock-difference lap time above can only be as fine as the
        game clock's tick (0.01 s on every file measured so far), while
        mLastLapTime is the timing-loop value to the millisecond. It lives
        in the scoring block, which refreshes only ~5 Hz, so at the moment
        the lap number changes it may still hold the PREVIOUS lap's time.
        A value is therefore only taken when it is new since the last
        crossing, arrives within OFFICIAL_LAP_WINDOW_S, and agrees with the
        clock difference. Disagreement means the two are not the same lap
        (recorder started mid-lap, invalidated lap) and the clock difference
        is kept. Call this on every poll.
        """
        # DIAGNOSTICS (10/14/2026 Fuji session: 8 of 18 laps, including the
        # best, got no official time and nothing said why). Every way a lap
        # can miss out now prints a line: the value never changed, it changed
        # too late (watched until the next crossing), or it disagreed.
        if self._official_used is None:
            self._official_used = last_lap_time   # left over from before recording
            return
        if self._official_watch_since is not None:
            # Window already missed: only report if the value turns up later.
            if last_lap_time != self._official_used:
                print(f"    official lap time {_fmt_lap(last_lap_time)} arrived "
                      f"{current_et - self._official_watch_since:.1f} s after the crossing "
                      f"(window {OFFICIAL_LAP_WINDOW_S:.1f} s, not applied)")
                self._official_used = last_lap_time
                self._official_watch_since = None
            return
        if self._official_pending_since is None:
            return
        diffed = self.lap_time_events[-1][1]
        age = current_et - self._official_pending_since
        if age > OFFICIAL_LAP_WINDOW_S:
            if last_lap_time != self._official_used:
                print(f"    no official lap time: {_fmt_lap(last_lap_time)} arrived "
                      f"{age:.1f} s after the crossing (window {OFFICIAL_LAP_WINDOW_S:.1f} s); "
                      f"keeping clock {_fmt_lap(diffed)}")
            else:
                print(f"    no official lap time: mLastLapTime unchanged at "
                      f"{_fmt_lap(last_lap_time)} {age:.1f} s after the crossing; "
                      f"keeping clock {_fmt_lap(diffed)}")
                self._official_watch_since = self._official_pending_since
            self._official_pending_since = None
            self._official_used = last_lap_time
            return
        if last_lap_time == self._official_used:
            return                                # scoring has not refreshed yet
        self._official_used = last_lap_time
        self._official_pending_since = None
        ts = self.lap_time_events[-1][0]
        if last_lap_time > 0 and abs(last_lap_time - diffed) <= OFFICIAL_LAP_TOLERANCE_S:
            self.lap_time_events[-1] = (ts, last_lap_time)
            print(f"    official lap time {_fmt_lap(last_lap_time)}")
        else:
            print(f"    no official lap time: mLastLapTime {_fmt_lap(last_lap_time)} "
                  f"disagrees with clock {_fmt_lap(diffed)} "
                  f"(tolerance {OFFICIAL_LAP_TOLERANCE_S * 1000:.0f} ms)")

    # ── Per-sample recording ─────────────────────────────────────────────────

    def add_sample(self, telem, wheels, player_scor, current_et: float,
                   poll_time: float = 0.0, session_meta: dict | None = None,
                   others: list | None = None) -> None:
        """Record one sample.  current_et = telem.mElapsedTime; poll_time =
        wall-clock seconds since this session started (when we read it);
        session_meta = extra static facts (see _session_meta), taken once;
        others = scoring structs of the NON-player cars, read only on 10 Hz
        gate ticks (the caller builds the list every sample, so keep it cheap)."""
        s   = self.s                       # alias

        if not self.meta:
            self.meta.update(session_meta or {})
            self.meta.update({
                "TyreRadiusCm": ",".join(
                    str(int(w.mStaticUndeflectedRadius)) for w in wheels),
                "TyreOptimalTempC": ",".join(f"{w.mOptimalTemp:.1f}" for w in wheels),
                "TyreCompoundFront": _safe_decode(telem.mFrontTireCompoundName),
                "TyreCompoundRear":  _safe_decode(telem.mRearTireCompoundName),
                "VehicleModel":      _safe_decode(telem.mVehicleModel),
                # Marks files that carry Pos Y / Ori R* / Other Cars (the
                # track-topography inputs); older files simply lack the key.
                "PoseChannels":      "1",
                # Marks files that carry Brake Pressure / Rear Brake Bias /
                # Steering Wheel Range (11/26/2026); older files lack the key.
                "InputChannels":     "1",
            })

        # ── HF extras, every sample ───────────────────────────────────────
        # Raw values, raw signs: nothing here is derived, so a wrong guess
        # about LMU's axis convention cannot be baked into a recording.
        # Local velocity gives body slip directly (lateral vs forward)
        # instead of reconstructing it from yaw rate and position.
        s["Poll Time"].append(poll_time)
        s["Local Vel X"].append(telem.mLocalVel.x)
        s["Local Vel Y"].append(telem.mLocalVel.y)
        s["Local Vel Z"].append(telem.mLocalVel.z)
        s["Yaw Accel"].append(telem.mLocalRotAccel.y)
        s["Front Downforce"].append(telem.mFrontDownforce)
        s["Rear Downforce"].append(telem.mRearDownforce)
        s["Drag"].append(telem.mDrag)
        s["Front Ride Height"].append(telem.mFrontRideHeight)
        s["Rear Ride Height"].append(telem.mRearRideHeight)
        s["Steering Shaft Torque"].append(telem.mSteeringShaftTorque)
        for ch, attr in HF_WHEEL_FIELDS:
            s[ch].append(tuple(getattr(w, attr) for w in wheels))
        # Orientation matrix rows (mOri[r] = local axes in world), raw like
        # the rest: the chase view decides the axis convention, not the file.
        for r in range(3):
            row = telem.mOri[r]
            s[f"Ori R{r}X"].append(row.x)
            s[f"Ori R{r}Y"].append(row.y)
            s[f"Ori R{r}Z"].append(row.z)

        # ── 100 Hz: telemetry-sourced continuous ──────────────────────────
        # GPS Time matches LMU's own convention: absolute session ET in seconds
        s["GPS Time"].append(current_et)

        # Pos X/Z: world metres (recorder-specific extension; backend handles both)
        s["Pos X"].append(telem.mPos.x)
        s["Pos Y"].append(telem.mPos.y)      # height: the track-elevation source
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
        s["Brake Pressure"].append(tuple(w.mBrakePressure for w in wheels))   # raw

        # ── 10 Hz: scoring-sourced geometry + thermals ────────────────────
        # Gated by GAME TIME, not by sample count: the v4 recorder's
        # `tick % SCORING_DECIMATE` is only 10 Hz if exactly 100 samples
        # arrive per second, which no longer holds when the sample rate is
        # whatever the game turns out to publish.
        if current_et >= self._next_slow_et:
            self._next_slow_et += SLOW_PERIOD_S
            if self._next_slow_et <= current_et:
                # Clock jumped (pause, stall): one row, not a catch-up burst.
                self._next_slow_et = current_et + SLOW_PERIOD_S
            s["TyresWear"].append(tuple(w.mWear for w in wheels))
            s["TyresCarcassTemp"].append(
                tuple(w.mTireCarcassTemperature - 273.15 for w in wheels))
            # Left/center/right across the tread as the game orders them,
            # which is NOT inside/center/outside (see lmu_data.py).
            s["TyresTempLeft"].append(tuple(w.mTemperature[0] - 273.15 for w in wheels))
            s["TyresTempCenter"].append(tuple(w.mTemperature[1] - 273.15 for w in wheels))
            s["TyresTempRight"].append(tuple(w.mTemperature[2] - 273.15 for w in wheels))
            # Stored raw: the struct comment says Celsius, rFactor 2 reported
            # Kelvin here, and LMU has not been checked.
            s["Brake Temp"].append(tuple(w.mBrakeTemp for w in wheels))
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
            # Other cars: one row per car per gate tick, stamped with the
            # player's clock so the backend can join them to this lap.
            for o in others or ():
                s["Other Cars"].append(
                    (current_et, int(o.mID), o.mPos.x, o.mPos.y, o.mPos.z,
                     o.mLapDist, o.mPathLateral, int(o.mInPits)))
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
        # Setup values: rows live in self.s (like Other Cars), so the safety
        # copy carries them with no SPOOL_FIELDS change and an older copy
        # simply has none.
        # BUG FIX (review 11/26/2026, Minor 1): NaN != NaN, so a NaN value
        # looked like a change on EVERY tick (1000 rows for 1000 ticks, 360k
        # rows/h). A NaN now equals the previous NaN: one row, like any value.
        for ch, attr, _ in SETTING_EVENT_FIELDS:
            v = float(getattr(telem, attr))
            last = self._last_setting.get(ch)
            if last is None or (v != last and not (v != v and last != last)):
                s[ch].append((current_et, v));  self._last_setting[ch] = v

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
    # MEASURED rates, not nominal ones. The backend places row N at
    # N / frequency, so a nominal 100 written over data that really arrived
    # at ~96 Hz (v4 recorder) stretches every time axis by a few percent.
    gps_t = buf.s.get("GPS Time", [])
    hz    = _effective_hz(gps_t, buf.poll_hz)
    n_slow = len(buf.s.get("Lap Dist", []))
    span   = gps_t[-1] - gps_t[0] if len(gps_t) > 1 else 0.0
    shz    = (n_slow - 1) / span if n_slow > 1 and span > 0 else 1.0 / SLOW_PERIOD_S
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
            ("RecordedBy",       RECORDER_ID),
            ("TotalSamples",     str(buf.sample_count)),
            # Completed laps. The Lap table has one more row when the
            # session stopped mid-lap (see flush).
            ("LapCount",         str(len(buf.lap_time_events))),
            ("PollHz",           f"{buf.poll_hz:g}"),
            ("EffectiveHz",      f"{hz:.3f}"),
            # getattr: --recover and --selftest pass plain namespaces that
            # may predate the meta field.
            *sorted(getattr(buf, "meta", {}).items()),
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

        def _write_scalar(ch: str, freq: float, unit: str,
                          detect_empty: bool = False) -> None:
            """Write a single-value continuous channel. detect_empty flags an
            all-zero channel as unpopulated (same rule as _write_wheel); it
            is only used for the HF extras, where it is not yet known which
            fields LMU fills in. The original channels are never dropped."""
            vals = buf.s.get(ch, [])
            if detect_empty and vals and not any(vals):
                unit, vals = "unpopulated_by_lmu", []
            con.execute("INSERT INTO channelsList VALUES (?,?,?)", [ch, freq, unit])
            con.execute(f'CREATE TABLE "{ch}" (value DOUBLE)')
            if vals:
                _bulk(ch, [vals])

        def _write_rows(ch: str, columns_sql: str, rows: list,
                        freq: float, unit: str) -> None:
            """Write a multi-column row table (e.g. Other Cars) through the same
            CSV + COPY path as _bulk. columns_sql is the CREATE TABLE column
            list; rows are equal-length tuples in that order. An empty list
            still creates the table, so readers need no existence check."""
            con.execute("INSERT INTO channelsList VALUES (?,?,?)", [ch, freq, unit])
            con.execute(f'CREATE TABLE "{ch}" ({columns_sql})')
            if rows:
                _bulk(ch, [list(c) for c in zip(*rows)])

        def _write_wheel(ch: str, freq: float, unit: str) -> None:
            """Write a per-wheel (value1..value4) continuous channel."""
            rows = buf.s.get(ch, [])
            # Mark channels LMU doesn't populate for this car
            # Surface Type is exempt: 0 means "dry tarmac", so an all-zero
            # channel is a clean run, not a field LMU leaves empty.
            effective_unit = ("unpopulated_by_lmu"
                              if ch != "Surface Type" and _all_zero_wheel(ch) else unit)
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
        _write_scalar("Pos Y",        hz,  "m")   # no detect_empty: 0 is a real height
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
        # all-zero -> unpopulated_by_lmu (the _write_wheel rule), e.g. a car
        # or game version that leaves the field empty
        _write_wheel("Brake Pressure",          hz, BRAKE_PRESSURE_UNIT)

        # ── HF extras ─────────────────────────────────────────────────────────
        _write_scalar("Poll Time",             hz, "s")
        _write_scalar("Local Vel X",           hz, "m/s",    detect_empty=True)
        _write_scalar("Local Vel Y",           hz, "m/s",    detect_empty=True)
        _write_scalar("Local Vel Z",           hz, "m/s",    detect_empty=True)
        _write_scalar("Yaw Accel",             hz, "rad/s2", detect_empty=True)
        _write_scalar("Front Downforce",       hz, "N",      detect_empty=True)
        _write_scalar("Rear Downforce",        hz, "N",      detect_empty=True)
        _write_scalar("Drag",                  hz, "N",      detect_empty=True)
        _write_scalar("Front Ride Height",     hz, "m",      detect_empty=True)
        _write_scalar("Rear Ride Height",      hz, "m",      detect_empty=True)
        _write_scalar("Steering Shaft Torque", hz, "Nm",     detect_empty=True)
        # Raw orientation matrix; like Pos Y, never flagged unpopulated.
        for _r in range(3):
            for _a in "XYZ":
                _write_scalar(f"Ori R{_r}{_a}", hz, "")
        for _ch, _ in HF_WHEEL_FIELDS:
            _write_wheel(_ch, hz, HF_WHEEL_UNITS[_ch])
        _write_wheel("TyresWear",        shz, "fraction")
        _write_wheel("TyresCarcassTemp", shz, "C")
        _write_wheel("TyresTempLeft",    shz, "C")
        _write_wheel("TyresTempCenter",  shz, "C")
        _write_wheel("TyresTempRight",   shz, "C")
        _write_wheel("Brake Temp",       shz, "raw (C or K, unverified)")

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

        # ── Other cars' poses (10 Hz gate ticks) ──────────────────────────────
        _write_rows("Other Cars",
                    "ts DOUBLE, slot INTEGER, pos_x DOUBLE, pos_y DOUBLE, "
                    "pos_z DOUBLE, lap_dist DOUBLE, path_lateral DOUBLE, in_pits INTEGER",
                    buf.s.get("Other Cars", []), shz, "pose_rows")

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
        # Setup events (SETTING_EVENT_FIELDS). Same all-zero rule as the wheel
        # channels: a 0 bias or 0 deg lock is not a real setting, so a session
        # whose only value is 0 is flagged unpopulated_by_lmu in eventsList and
        # its table left empty. An empty list (a safety copy from before these
        # channels) writes an empty table with the normal unit.
        for _ch, _, _unit in SETTING_EVENT_FIELDS:
            _rows = buf.s.get(_ch, [])
            if _rows and not any(v for _, v in _rows):
                _unit, _rows = "unpopulated_by_lmu", []
            _write_event_scalar(_ch, _unit, _rows)

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

def run_probe(seconds: float, poll_hz: float | None, tick_sync: bool = False) -> None:
    """Measure how fast LMU really publishes telemetry. Records nothing.

    Must be run while driving (not in the garage or a menu): the game clock
    only advances on track.

    Probes the same mode the recorder would use: tick-synced by default,
    fixed-rate with poll_hz. Run both in one session to compare CPU cost and
    skipped updates. It also prints how late the tick-sync wake-ups were and
    how long one loop iteration takes, which is what tells WHY a mode lost
    updates.
    """
    mode, poll_hz = resolve_poll_mode(poll_hz, False, tick_sync)
    tick = mode == "tick"
    how = "tick-synced" if tick else f"fixed {poll_hz:.0f}Hz"
    print(f"lmu-coach probe  |  {how} polling for {seconds:g}s")
    print("Waiting for LMU (drive the car, then leave this running)...", flush=True)
    mmap_ctrl = None
    while mmap_ctrl is None:
        try:
            mmap_ctrl = MMapControl(LMUConstants.LMU_SHARED_MEMORY_FILE,
                                    lmu_data.LMUObjectOut)
            mmap_ctrl.create(access_mode=0)
        except Exception:
            mmap_ctrl = None
            time.sleep(2.0)

    stats = PollStats()
    pacer = Pacer(1.0 / poll_hz) if not tick else None
    sync = TickSync() if tick else None
    si = ti = INVALID_INDEX
    last_sig = None
    first_et = None
    t0 = None
    cpu0 = 0.0
    iter_ms: list[float] = []
    dt_min, dt_max = math.inf, 0.0
    try:
        while True:
            t_top = time.perf_counter()
            mmap_ctrl.update()
            data = mmap_ctrl.data
            si, ti = _find_player(data, si, ti)
            if si == INVALID_INDEX or ti == INVALID_INDEX:
                time.sleep(0.5)
                continue
            telem = data.telemetry.telemInfo[ti]
            et = float(telem.mElapsedTime)
            if t0 is None:
                # Start the measurement on the first clock movement, so time
                # spent parked in the garage is not counted as "no ticks".
                if first_et is None:
                    first_et = et
                if et == first_et:
                    time.sleep(0.01)
                    continue
                t0 = time.perf_counter()
                cpu0 = time.process_time()
                if not tick:
                    pacer = Pacer(1.0 / poll_hz)
                print("Clock is running - measuring...", flush=True)
            sig = (telem.mPos.x, telem.mPos.z, telem.mLocalVel.z)
            stats.record(et, sig != last_sig)
            last_sig = sig
            dt = float(telem.mDeltaTime)
            if dt > 0:
                dt_min, dt_max = min(dt_min, dt), max(dt_max, dt)
            if time.perf_counter() - t0 >= seconds:
                break
            iter_ms.append((time.perf_counter() - t_top) * 1000.0)
            if tick:
                sync.wait(lambda: peek_elapsed_time(mmap_ctrl, ti), et)
            else:
                pacer.wait()
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        mmap_ctrl.close()

    if t0 is None or stats.ticks < 2:
        print("No clock movement seen - was the car on track?")
        return
    wall = time.perf_counter() - t0
    cpu = (time.process_time() - cpu0) / max(wall, 1e-9) * 100.0
    for line in stats.summary(wall):
        print("  " + line)
    print(f"  CPU: {cpu:.0f}% of one core while {how}")
    if iter_ms:
        v = sorted(iter_ms)
        q = lambda p: v[min(len(v) - 1, int(len(v) * p))]  # noqa: E731
        print(f"  loop iteration (copy + find player + stats): "
              f"p50 {q(0.5):.2f} / p99 {q(0.99):.2f} / max {v[-1]:.2f} ms")
    if tick:
        d = sync.diagnostics()
        print(f"  wake-up lateness of the long sleep (learned period {d['period_ms']:.2f} ms): "
              f"p50 {d['wake_late_ms_p50']:.2f} / p90 {d['wake_late_ms_p90']:.2f} / "
              f"p99 {d['wake_late_ms_p99']:.2f} / max {d['wake_late_ms_max']:.2f} ms; "
              f"{d['looks_per_tick']:.1f} looks per tick; gave up {d['gave_up']}x")
    if dt_max > 0:
        print(f"  mDeltaTime reported by the game: {dt_min * 1000:.2f}-{dt_max * 1000:.2f} ms")
    published = stats.ticks / wall
    achieved = stats.polls / wall
    if tick:
        print(f"  RESULT: caught {published:.0f} updates/s; {stats.skipped_ticks:,} skipped "
              f"(game or recorder). Run once more with --poll-hz 1000 in the same session "
              f"and compare CPU and skipped.")
        _warn = skip_warning(stats)
        if _warn:
            print("  " + _warn)
    elif achieved < published * 1.5:
        print(f"  INCONCLUSIVE: only polled {achieved:.0f} Hz, too close to the "
              f"{published:.0f} Hz seen. Re-run with a higher --poll-hz.")
    elif stats.data_changed_without_tick > 0.05 * stats.polls:
        print("  RESULT: car data changes BETWEEN clock ticks - the game publishes "
              "faster than its clock shows. Record with --every-poll to capture it.")
    else:
        print(f"  RESULT: LMU publishes ~{published:.0f} Hz. Polling faster than "
              f"that adds no new data.")


def run_recorder(output_dir: Path, poll_hz: float | None, every_poll: bool = False,
                 tick_sync: bool = False) -> None:
    """Default: follow the game's updates with TickSync. poll_hz (or
    every_poll) selects fixed-rate polling with the Pacer instead; see
    resolve_poll_mode for the measurements behind the default."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mode, poll_hz = resolve_poll_mode(poll_hz, every_poll, tick_sync)
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

    for _sp in _spool_files(output_dir):
        print(f"Found an unconverted session safety copy: {_sp.name}\n"
              f"  Convert it with:  uv run recorder.py --recover", flush=True)

    polling = ("tick-synced (follows the game's ~100 updates/s, one memory copy each)"
               if mode == "tick" else f"fixed {poll_hz:.0f}Hz")
    print(f"lmu-coach recorder (HF1)  |  {polling}  |  "
          f"{'every poll stored' if every_poll else 'one sample per game tick'}  |  "
          f"10Hz geometry  |  out={output_dir}")
    if mode == "fixed" and interval <= Pacer.SPIN_S:
        print(f"NOTE: fixed polling at {poll_hz:.0f}Hz keeps one CPU core fully busy while "
              f"driving. Leave out --poll-hz / --every-poll to follow the game's updates "
              f"(about 1% of a core).")
    print("Waiting for LMU...\n")

    mmap_ctrl: MMapControl | None = None
    conn_state                    = ConnectionState()
    buf: SessionBuffer | None     = None
    last_lap: int | None          = None

    # In tick mode the pacer only serves the short retry waits (a car joining
    # leaves scoring and telemetry disagreeing for one poll); TickSync does the
    # real waiting at the bottom of the loop.
    pacer                          = Pacer(interval)
    tick_sync                      = TickSync() if mode == "tick" else None
    stats                          = PollStats()
    stats_t0                       = time.perf_counter()
    si = ti                        = INVALID_INDEX
    last_sample_et: float | None   = None
    last_sig                       = None
    next_stop_check                = 0.0

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
            # DIFFERENT FROM the retired v4 recorder, which at this point dropped the
            # newest row of BOTH lap tables to hide the unfinished final lap.
            # Neither is dropped here:
            #   - The newest Lap row is the crossing that ENDS the last
            #     completed lap. Without it the backend has no end boundary
            #     for that lap and runs it to end-of-file, so its lap time
            #     silently includes the tail of the unfinished lap.
            #   - The newest Lap Time row is that completed lap's duration
            #     (see register_lap); dropping it lost the lap's time.
            # Result: Lap has one row per lap STARTED (the last one is the
            # unfinished lap, which readers see as a lap with no end), and
            # Lap Time has one row per lap COMPLETED, aligned by position.
            if len(buf.lap_events) > 1:
                print(f"  Final lap {buf.lap_events[-1][1]} was in progress at stop "
                      f"({reason}); kept as an unfinished lap")
            # write_session() is isolated in its own try/except so that a
            # write-time bug (bad tuple shape, disk full, etc.) is never
            # mistaken for a shared-memory disconnect by the caller's outer
            # (OSError, ValueError) handler — that outer handler previously
            # swallowed exactly this kind of failure and silently produced
            # a truncated file (the v3 bug). A write failure here is loud,
            # distinct, and does NOT get reinterpreted as "reconnecting".
            for _line in stats.summary(time.perf_counter() - stats_t0):
                print("  " + _line, flush=True)
            _warn = skip_warning(stats)
            if _warn:
                print("  " + _warn, flush=True)
            buf.meta.update({
                "Polls":                  str(stats.polls),
                "UniqueTicks":            str(stats.ticks),
                "DataChangedWithoutTick": str(stats.data_changed_without_tick),
                "SkippedTicks":           str(stats.skipped_ticks),
                "PollMode":               "tick-sync" if mode == "tick" else "fixed",
                "ClockStepHistogram":     ", ".join(
                    f"{k} x{n}" for k, n in stats.histogram().items()),
            })
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
            if _stop_state["stop_pending"]:
                raise KeyboardInterrupt  # stop requested; any save has finished
            _now = time.perf_counter()
            if _now >= next_stop_check:
                next_stop_check = _now + STOP_CHECK_S
                if STOP_FILE.exists():
                    raise KeyboardInterrupt
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

                si, ti = _find_player(data, si, ti)
                if si == INVALID_INDEX:
                    time.sleep(0.5)
                    continue
                if ti == INVALID_INDEX:
                    # Player is scored but not in telemetry THIS poll (the
                    # two blocks can disagree for a moment when a car joins
                    # or leaves). Retry on the next poll: a 0.5 s wait here
                    # would silently drop ~50 ticks mid-session.
                    pacer.wait()
                    continue

                pscor       = data.scoring.vehScoringInfo[si]
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
                    # Idle (menu, pause, loading): a plain sleep, so the
                    # pacer is not spinning a CPU core while nothing happens.
                    time.sleep(0.01)
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
                    stats          = PollStats()
                    stats_t0       = time.perf_counter()
                    last_sample_et = None
                    print(f"Session: {track_name} / {car_name} "
                          f"(driver: {driver_name})")

                # ── Lap crossing ───────────────────────────────────────────
                if lap_number != last_lap:
                    buf.register_lap(lap_number, current_et)
                    last_lap = lap_number

                # One stored sample per game tick. Polling faster than the
                # game publishes re-reads the same tick; storing those would
                # only duplicate rows (the v4 recorder's ~16% duplicates).
                sig = (telem.mPos.x, telem.mPos.z, telem.mLocalVel.z)
                stats.record(current_et, sig != last_sig)
                last_sig = sig
                buf.update_official_lap_time(float(pscor.mLastLapTime), current_et)
                if every_poll or current_et != last_sample_et:
                    # Other cars are only read on the 10 Hz gate ticks, so only
                    # build the list then (100 Hz x 60 struct views is waste).
                    scor  = data.scoring.vehScoringInfo
                    num_v = data.scoring.scoringInfo.mNumVehicles
                    buf.add_sample(telem, telem.mWheels, pscor, current_et,
                                   poll_time=time.perf_counter() - stats_t0,
                                   others=(
                                       [scor[i] for i in range(num_v) if i != si]
                                       if current_et >= buf._next_slow_et else None),
                                   # Only the first sample uses it; skip the work after.
                                   session_meta=None if buf.meta
                                   else _session_meta(data, pscor))
                    last_sample_et = current_et

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
                si = ti = INVALID_INDEX
                time.sleep(2.0)
                continue

            if tick_sync is not None and ti != INVALID_INDEX:
                tick_sync.wait(lambda: peek_elapsed_time(mmap_ctrl, ti), current_et)
            else:
                pacer.wait()

    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
        flush("user stopped")
        if mmap_ctrl is not None:
            mmap_ctrl.close()
        STOP_FILE.unlink(missing_ok=True)
        print("Recorder exited cleanly.", flush=True)


SPOOL_FIELDS = ("car_name", "track_name", "driver_name", "poll_hz", "start_wall",
                "sample_count", "lap_events", "lap_time_events", "gear_events",
                "abs_events", "tc_events", "pits_events", "meta")


def spool_session(buf, output_dir: Path) -> Path:
    """Dump the raw in-memory session to disk as fast as possible."""
    name = buf.start_wall.strftime("%Y-%m-%dT%H_%M_%SZ")
    final = output_dir / f".{name}{SPOOL_SUFFIX}"
    part = output_dir / f".{name}{SPOOL_SUFFIX}.part"
    data = {k: getattr(buf, k) for k in SPOOL_FIELDS if k != "meta"}
    data["meta"] = getattr(buf, "meta", {})  # absent on --selftest's namespace
    data["s"] = dict(buf.s)
    with open(part, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(part, final)
    return final


def load_spool(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    data.setdefault("meta", {})
    s = defaultdict(list)
    s.update(data.pop("s"))
    return types.SimpleNamespace(s=s, **data)


def _spool_files(output_dir: Path) -> list[Path]:
    """Unconverted safety copies: ".*.spool" plus the pre-10/19/2026 ".*.spool_hf"."""
    return sorted(p for suffix in (SPOOL_SUFFIX, *LEGACY_SPOOL_SUFFIXES)
                  for p in output_dir.glob(f".*{suffix}"))


def recover_spools(output_dir: Path) -> None:
    spools = _spool_files(output_dir)
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
    for c in ("GPS Time", "Pos X", "Pos Y", "Pos Z", "Ground Speed", "Steering Pos", "Throttle Pos",
              "Brake Pos", "Yaw Rate", "Engine RPM", "G Force Lat", "G Force Long", "G Force Vert"):
        s[c] = [random.random() for _ in range(n)]
    for c in ("Wheel Speed", "Slip Angle", "Susp Pos", "Brake Pressure"):
        s[c] = [tuple(random.random() for _ in range(4)) for _ in range(n)]
    for c in ("Grip Fraction", "Tire Load", "Tire Lateral Force", "Tire Longitudinal Force"):
        s[c] = [(0.0, 0.0, 0.0, 0.0)] * n
    for c in ("GPS Latitude", "GPS Longitude", "GPS Speed", "Path Lateral", "Track Edge", "Lap Dist"):
        s[c] = [random.random() for _ in range(n10)]
    for c in ("TyresRubberTemp", "TyresPressure"):
        s[c] = [tuple(random.random() for _ in range(4)) for _ in range(n10)]
    # Pose channels (10/25/2026): the nine Ori R.. scalars at 100 Hz and a
    # full grid of other cars at 10 Hz. PREVIOUSLY the selftest left them out,
    # so a slow save caused by the new tables (Other Cars is the largest table
    # in a race: 60 cars x 10 Hz x 8 columns) could not be measured with it.
    for r in range(3):
        for a in "XYZ":
            s[f"Ori R{r}{a}"] = [random.random() for _ in range(n)]
    n_other = 60
    s["Other Cars"] = [(k / 10.0, slot, random.random(), random.random(), random.random(),
                        random.random() * 5000.0, random.random(), 0)
                       for k in range(n10) for slot in range(n_other)]
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
        ori_n = con.execute('SELECT COUNT(*) FROM "Ori R2Z"').fetchone()[0]
        other_n = con.execute('SELECT COUNT(*) FROM "Other Cars"').fetchone()[0]
        con.close()
        ok = (rows == n and laps_n == len(laps) and ori_n == n
              and other_n == len(s["Other Cars"]))
        print(f"DuckDB write: {total:.1f}s total  |  rows={rows:,} (expected {n:,})  "
              f"laps={laps_n}  ori={ori_n:,}  other cars={other_n:,} "
              f"(expected {len(s['Other Cars']):,})  ->  {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll-hz",    type=float, default=None,
                        help="poll at this FIXED rate in Hz instead of following the game's "
                             "updates (default). 500 Hz and up keep one CPU core fully busy; "
                             "1000 Hz is the reference for comparing skipped updates.")
    parser.add_argument("--low-priority", action="store_true",
                        help="run below normal priority. NOT recommended: LMU runs at High "
                             "priority and a sleeping thread below it wakes late (lost 9%% to "
                             "44%% of the updates in the 10/21/2026 drives)")
    # first-day names of the experiment; both are the default now
    parser.add_argument("--tick-sync", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--normal-priority", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recover", action="store_true",
                        help="convert any unconverted safety copies to .duckdb files, then exit")
    parser.add_argument("--selftest", action="store_true",
                        help="time a synthetic-session save on this machine, then exit")
    parser.add_argument("--selftest-minutes", type=float, default=25.0)
    parser.add_argument("--probe", type=float, metavar="SECONDS", default=None,
                        help="measure how fast LMU publishes telemetry for SECONDS "
                             "while you drive, record nothing, then exit")
    parser.add_argument("--every-poll", action="store_true",
                        help="store every poll, not one sample per game tick")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tick_sync and (args.poll_hz is not None or args.every_poll):
        parser.error("--tick-sync cannot be combined with --poll-hz or --every-poll")
    if args.selftest:
        selftest(args.selftest_minutes)
    elif args.recover:
        recover_spools(args.output_dir)
    else:
        if args.low_priority and lower_priority():
            print("Recorder runs at below-normal priority (--low-priority).", flush=True)
        if args.probe is not None:
            run_probe(args.probe, args.poll_hz, args.tick_sync)
        else:
            run_recorder(args.output_dir, args.poll_hz, args.every_poll, args.tick_sync)


if __name__ == "__main__":
    main()
