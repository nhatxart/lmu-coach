"""
Reads LMU's native .duckdb telemetry files and produces a single aligned,
time-indexed table.

Key fact about the LMU format (confirmed by direct inspection, not assumed):
  - Every channel is its own table.
  - "Continuous" channels (G Force Lat, Steering Pos, Wheel Speed, etc.) have
    just a `value` column (or value1..value4 for per-wheel). There is NO
    timestamp column. Row N occurred at time N / frequency, where frequency
    comes from the `channelsList` table (e.g. 10Hz, 50Hz, 100Hz).
  - "Event" channels (ABS, TC, TCCut, Lap, Current LapTime) have explicit
    `ts` + `value` columns and only log on change, not every tick.
  - `metadata` holds session info (driver name, car, track, full setup JSON)
    as a flat key/value table.

Because channels run at different native frequencies, they don't share row
indices. We reconstruct a real time axis per channel and resample everything
onto one common grid (default 20Hz — enough resolution for corner-level
analysis without blowing up token counts when this gets fed to an LLM later).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np


# Channels we care about for driving-technique analysis. Anything else in
# the file (fuel, energy, temps, etc.) is left alone for now — easy to add
# to this list later without touching the alignment logic.
CONTINUOUS_CHANNELS = [
    "G Force Lat",
    "G Force Long",
    "G Force Vert",
    "Steering Pos",
    "Throttle Pos",
    "Brake Pos",
    "Ground Speed",
    "Lap Dist",
    "Path Lateral",
    "Track Edge",  # signed like Path Lateral; track_topography.detect_lateral_sign
    "Gear",  # not in channelsList but present as a plain table; handled defensively
    "Yaw Accel",  # rad/s^2, HF1 files only - tire model's steady-state filter
    # rad/s, recorder files only. vehicle_dynamics.compute_slip_trace and
    # corner_features already prefer it when present, but it was never loaded,
    # so both silently fell back to the lateral-G/speed estimate.
    "Yaw Rate",
    # Chase view: world position incl. height and the raw orientation matrix
    # (recorder files from 10/25/2026 on; vehicle_pose.py decodes the matrix).
    # Pos X/Z are 100 Hz here, unlike the 10 Hz GPS the track map uses.
    "Pos X",
    "Pos Y",
    "Pos Z",
    "Ori R0X", "Ori R0Y", "Ori R0Z",
    "Ori R1X", "Ori R1Y", "Ori R1Z",
    "Ori R2X", "Ori R2Y", "Ori R2Z",
]

# Per-wheel channels: value1..value4 map to FL, FR, RL, RR in LMU's convention.
# Channels below "Wheel Speed" are recorder-sourced only (not in LMU's own
# .duckdb export) — telemetry.py handles missing tables gracefully already,
# so listing them here is safe for both file types.
#
# BUG FIX: "Tire Pressure" and "Tire Temp" below used to point at table
# names that don't exist in EITHER file type — the recorder writes
# "TyresPressure" / "TyresRubberTemp" (confirmed against actual recorder.py
# write_session() calls and real recorded files), so these two entries
# silently matched nothing and lap.channels never contained pressure/temp
# data despite the recorder actually capturing it. Also, "Susp Pos" (used
# directly by main.py's balance metric and grip_analysis.py's slip-ratio
# fallback) was missing from this list entirely — those two call sites
# were reading it via a separate raw DuckDB query rather than through
# LapTelemetry, meaning grip_analysis.py's own lap.channels never had it,
# silently disabling any grip-report logic that expected it there.
WHEEL_CHANNELS = [
    "Wheel Speed",
    "Slip Angle",       # recorder: atan(mLateralGroundVel/mLongitudinalGroundVel) per wheel
    "Susp Pos",          # suspension deflection, metres — load-transfer proxy
    "TyresPressure",     # live pressure in kPa (changes as tires heat up)
    "TyresRubberTemp",   # mean surface temp per wheel, Celsius
    # Vertical load for the tire model, best first (tire_model._axle_load).
    # Tire Load is mTireLoad (N), populated for some cars only; Susp Force is
    # mSuspForce, pushrod load in N (HF1 recorder). Both are dropped by
    # _read_channel_frequencies when the recorder marked them unpopulated.
    "Tire Load",
    "Susp Force",
    # HF1 files only; absent tables are skipped like any missing channel.
    "Camber",             # rad per wheel
    "TyresWear",          # fraction per wheel
    "TyresCarcassTemp",   # Celsius per wheel
    "Surface Type",       # categorical code per wheel (see CATEGORICAL_CHANNELS)
    "Tire Lateral Force", # N; zero/flagged unpopulated on GT3
    "Ride Height",        # m per wheel; chase view's ground probe
    # Recorder files from 11/26/2026 on (metadata InputChannels = "1"): raw
    # mBrakePressure, 0-1 today; LMU says it will become kPa, so check the
    # channelsList unit before comparing files. Flagged unpopulated when all 0.
    "Brake Pressure",
]

# Codes, not quantities: linear interpolation would invent surface types that
# never occurred (e.g. 2.5 between 0 and 5), so these hold the previous sample.
CATEGORICAL_CHANNELS = {"Surface Type"}

# Event channels: sparse, logged on change only. We forward-fill these onto
# the common grid (the value holds until the next logged change).
EVENT_CHANNELS = [
    "ABS",
    "TC",
]

# Setup values the recorder logs on change (11/26/2026; recorder.py
# SETTING_EVENT_FIELDS): rear brake bias (fraction on rear) and the steering
# wheel's physical / visual range (degrees, unverified). Step-held onto the
# grid like ABS/TC; the value before a lap's first change is the last one
# logged before it (kept files are seeded with it by lap_cut._seed_event).
# Unlike EVENT_CHANNELS, a table with no rows (flagged unpopulated_by_lmu in
# eventsList, or a file without the table) gives NO channel rather than a
# NaN column, so callers can test `in lap.channels`.
SETTING_EVENT_CHANNELS = [
    "Rear Brake Bias",
    "Steering Wheel Range",
    "Visual Steering Wheel Range",
]

WHEEL_LABELS = ("FL", "FR", "RL", "RR")


@dataclass
class LapTelemetry:
    """One lap's worth of aligned telemetry, indexed by a common time axis."""

    driver_name: str
    car_name: str
    track_name: str
    session_type: str
    lap_number: int
    lap_time_s: float
    time_s: np.ndarray  # common time grid, seconds from lap start
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    # File metadata (key -> value strings), e.g. TyreOptimalTempC and
    # TyreCompoundFront from the HF1 recorder. Shared by every lap of a file.
    meta: dict[str, str] = field(default_factory=dict)
    # False for the file's last lap: it has no closing Lap event, so it ends
    # wherever the recording stopped (a 1.4 s "lap" was seen). Callers that
    # pick a "fastest" lap must skip these (see reference_lap_time).
    is_complete: bool = True
    # The game's own Lap Dist on the same grid (what /trackmap and the
    # per-track topography look up with). channels["Lap Dist"] is re-zeroed
    # at the lap start and the unwrap across start/finish drops up to one
    # 10 Hz step (4-8 m), so it is not the same number. None without Lap Dist.
    lap_dist_raw: np.ndarray | None = None


def _read_channel_frequencies(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    # exclude channels the recorder flagged as unpopulated_by_lmu — LMU
    # wrote zeros into those shared-memory fields for this car, so the
    # tables exist but contain no useful data; also handles the native
    # export format which stores real units (never 'unpopulated_by_lmu')
    rows = con.execute(
        "SELECT channelName, frequency FROM channelsList WHERE unit != 'unpopulated_by_lmu'"
    ).fetchall()
    return {name: float(freq) for name, freq in rows}


def _read_unpopulated_channels(con: duckdb.DuckDBPyConnection) -> set[str]:
    # The loaders below fall back to a default frequency for channels missing
    # from _read_channel_frequencies, so leaving a channel out of that dict
    # does not stop its (all-zero or empty) table being read. Skip these.
    try:
        rows = con.execute(
            "SELECT channelName FROM channelsList WHERE unit = 'unpopulated_by_lmu'"
        ).fetchall()
    except Exception:
        return set()
    return {r[0] for r in rows}


# ── Real time axes ───────────────────────────────────────────────────────────
# BUG FIX (10/14/2026 Fuji HF file): "row N is at N / frequency" only holds
# if every tick was recorded. The recorders store one 100 Hz row per game
# tick, and the game clock can skip ticks — that file lost ~18 s of them in
# its first lap (1,792 steps of 20 ms). Every later 100 Hz row then sat up
# to 18 s away from N / frequency, while the 10 Hz rows (gated by game time)
# did not drift. Laps were cut at the right ET but read the wrong data:
# /trackmap placed lap 11's start 7.8 s early, and the coach analysed the
# wrong stretch of every lap. Recorder v4 files also start GPS Time at the
# session clock (e.g. 1041.4 s) while N / frequency starts at 0, so laps
# there were sliced outside the data entirely.
# Fix: 100 Hz rows take their time from GPS Time (one value per row), 10 Hz
# rows from replaying the recorder's own 10 Hz gate over GPS Time.
SLOW_PERIOD_S = 0.1   # recorder.py's SLOW_PERIOD_S
SLOW_ROW_TABLES = ("GPS Latitude", "Lap Dist", "Path Lateral")


# The game updates Lap Dist, Path Lateral and Track Edge with its scoring
# data at 5 Hz; recorder.py writes them on its 10 Hz gate, so in real HF1
# files every value is written twice (measured: 49.8 % of the Lap Dist steps
# are zero on the 10/27 Road Atlanta file, the others ~9.7 m at speed). A
# repeated row is the old value at a later time, up to 0.2 s (10 m at
# 180 km/h) stale. Runs longer than this many rows are a car that really
# stands still (or a value that really is constant) and are kept.
SCORING_HOLD_MAX_ROWS = 2
# Track Edge is held too but carries Path Lateral's sign: interpolating a
# held row between +w and -w would pass through 0. Its users take abs()
# per row, which the stale repeat does not hurt, so it is left as recorded.
SCORING_HELD_CHANNELS = ("Lap Dist", "Path Lateral")


# A car above this speed cannot have a standing Lap Dist / Path Lateral, so
# with a speed per row every repeat is a hold, whatever its length.
SCORING_MOVING_KMH = 10.0


def held_rows(values, speed_kmh=None) -> np.ndarray:
    """True for rows that only repeat the previous row because the game had
    not updated the value yet (the first row of a run is kept).

    The rule: a run of equal values is held (all but its first row dropped)
    when it is at most SCORING_HOLD_MAX_ROWS long, or - given `speed_kmh`
    per row - when the car moves (> SCORING_MOVING_KMH anywhere in the run).
    Only a slow car's longer run is kept, as a real stop.
    BUG FIX / PREVIOUSLY: length alone decided, so 10 Hz gate jitter that
    wrote one 5 Hz value three times counted as a stop and its stale rows
    came back as data."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.zeros(len(v), dtype=bool)
    start = np.concatenate([[True], v[1:] != v[:-1]])
    run = np.cumsum(start) - 1
    run_len = np.bincount(run)
    held_run = run_len <= SCORING_HOLD_MAX_ROWS
    if speed_kmh is not None:
        sp = np.nan_to_num(np.asarray(speed_kmh, dtype=float))
        if len(sp) == len(v):
            held_run = held_run | (np.maximum.reduceat(sp, np.nonzero(start)[0]) > SCORING_MOVING_KMH)
    return ~start & held_run[run]


def dehold(values, speed_kmh=None) -> list:
    """`values` with the held repeats (held_rows) replaced by interpolation
    (by row index) between the rows where the game updated the value."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return []
    held = held_rows(v, speed_kmh)
    if not held.any():
        return v.tolist()
    keep = np.nonzero(~held)[0]
    return np.interp(np.arange(len(v)), keep, v[keep]).tolist()


def _run_slow_gate(ticks: list[float], nxt: float) -> tuple[list[int], float]:
    """recorder.py's 10 Hz gate over `ticks` from due time `nxt`: (indices of
    the ticks that wrote a 10 Hz row, due time after the last tick). The one
    copy of the gate that _replay_slow_gate and slow_gate_due_at share."""
    idx = []
    for i, et in enumerate(ticks):
        if et >= nxt:
            nxt += SLOW_PERIOD_S
            if nxt <= et:
                nxt = et + SLOW_PERIOD_S
            idx.append(i)
    return idx, nxt


def _replay_slow_gate(t_fast: np.ndarray, nxt: float | None = None) -> np.ndarray:
    """Indices into t_fast where recorder.py wrote a 10 Hz row: the first
    tick at or after each due time, re-anchored after a clock jump. `nxt` is
    the first due time (default: the first tick, as recorder.py starts)."""
    idx, _ = _run_slow_gate(t_fast.tolist(), float(t_fast[0]) if nxt is None else nxt)
    return np.asarray(idx, dtype=np.int64)


def slow_gate_due_at(t_fast, cut_start_s: float) -> float:
    """The recorder's 10 Hz gate due time in effect when the first tick at or
    after `cut_start_s` is processed, on an original recording's GPS Time.

    The keep-laps writer stores it as metadata GateDue in a kept file, and
    session_time_axes replays the kept file's gate from it, so both use this
    one copy of recorder.py's gate (_run_slow_gate). It must be computed
    on the original: the gate re-anchors after every clock jump, so its
    phase at the cut cannot be rebuilt from the cut alone. A cut at or before
    the recording's start gets the first tick, where recorder.py starts."""
    t = np.asarray(t_fast, dtype=np.float64)
    i = int(np.searchsorted(t, cut_start_s, side="left"))
    _, nxt = _run_slow_gate(t[:i].tolist(), float(t[0]))
    return nxt


def _row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    try:
        return int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
    except Exception:
        return 0


def _gate_due(con: duckdb.DuckDBPyConnection) -> float | None:
    """metadata GateDue (kept files: slow_gate_due_at on the original), or None."""
    try:
        row = con.execute("SELECT value FROM metadata WHERE key = 'GateDue'").fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


def session_time_axes(
    con: duckdb.DuckDBPyConnection,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(t_fast, t_slow): game time (s) of each 100 Hz row and each 10 Hz row.

    t_fast is GPS Time itself. t_slow replays the recorder.py gate (from
    metadata GateDue in a kept file); if
    that does not reproduce the file's 10 Hz row count (the v4 recorder decimated
    by tick count, native LMU is evenly spaced), the 10 Hz rows are spread
    evenly over the 100 Hz rows by index, which is exact for those formats.
    Either is None when the file lacks the tables to build it.
    """
    try:
        rows = con.execute('SELECT value FROM "GPS Time" ORDER BY rowid').fetchall()
    except Exception:
        return None, None
    if len(rows) < 2:
        return None, None
    t_fast = np.array([r[0] for r in rows], dtype=np.float64)
    n_slow = next((n for n in (_row_count(con, t) for t in SLOW_ROW_TABLES) if n), 0)
    if n_slow < 2:
        return t_fast, None
    # PREVIOUSLY the gate was always replayed from the file's first row. A
    # kept file (one lap cut out of a recording, keep laps) starts mid-session,
    # and its first row is not where the recorder's gate started: where ticks
    # were skipped the gate's phase differs, the replay gives the same row
    # count and every 10 Hz row came out up to a tick or more off (0.04 s in
    # test_kept_readers), so the map drifted against the telemetry silently.
    # Fix: replay from the gate's due time at the cut, taken from the
    # original by the writer (metadata GateDue, slow_gate_due_at).
    # PREVIOUSLY (6039d59) it was the original's first GPS Time plus 0.1 s
    # steps, but the gate re-anchors after every clock jump, so after one
    # before the cut that phase was 0.04 s off with the right row count.
    # After any tick p recorder.py leaves p < due <= p + 0.1, so a valid due
    # is below the first kept tick + 0.1; a wrong value or a row count that
    # does not match the 10 Hz tables falls through to the old path.
    due = _gate_due(con)
    if due is not None and due < t_fast[0] + SLOW_PERIOD_S:
        idx = _replay_slow_gate(t_fast, nxt=due)
        if len(idx) == n_slow:
            return t_fast, t_fast[idx]
    idx = _replay_slow_gate(t_fast)
    if len(idx) == n_slow:
        return t_fast, t_fast[idx]
    # For recorder.py files the replay must match; the index fallback
    # below is only exact for v4/native files. A mismatch means the
    # recorder's gate changed (keep SLOW_PERIOD_S and _replay_slow_gate in
    # step with recorder.py) or the file is damaged, and the map will
    # drift against the telemetry wherever ticks were skipped - say so
    # rather than drift silently.
    try:
        row = con.execute("SELECT value FROM metadata WHERE key = 'RecordedBy'").fetchone()
    except Exception:
        row = None
    if row and " hf" in str(row[0]):
        print(f"[telemetry] WARNING: 10 Hz rows do not match the recorder's gate "
              f"(replayed {len(idx)}, file has {n_slow}); falling back to index "
              f"spacing - track position may drift against the telemetry.")
    frac = np.linspace(0.0, len(t_fast) - 1, n_slow)
    return t_fast, np.interp(frac, np.arange(len(t_fast)), t_fast)


def channel_time(n: int, freq_hz: float, t_fast: np.ndarray | None,
                 t_slow: np.ndarray | None) -> np.ndarray:
    """Time axis for a channel with n rows recorded at nominally freq_hz."""
    if t_fast is not None and n == len(t_fast):
        return t_fast
    if t_slow is not None and n == len(t_slow):
        return t_slow
    t0 = float(t_fast[0]) if t_fast is not None else 0.0
    return t0 + np.arange(n, dtype=np.float64) / freq_hz


def _read_continuous(
    con: duckdb.DuckDBPyConnection, table: str, freq_hz: float,
    axes: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (time_s, value) for a plain value-column table."""
    rows = con.execute(f'SELECT value FROM "{table}" ORDER BY rowid').fetchall()
    values = np.array([r[0] for r in rows], dtype=np.float64)
    return channel_time(len(values), freq_hz, *axes), values


def _read_wheel_channel(
    con: duckdb.DuckDBPyConnection, table: str, freq_hz: float,
    axes: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns (time_s, {FL: [...], FR: [...], RL: [...], RR: [...]})."""
    rows = con.execute(
        f'SELECT value1, value2, value3, value4 FROM "{table}" ORDER BY rowid'
    ).fetchall()
    arr = np.array(rows, dtype=np.float64).reshape(-1, 4)  # shape (N, 4)
    t = channel_time(arr.shape[0], freq_hz, *axes)
    return t, {label: arr[:, i] for i, label in enumerate(WHEEL_LABELS)}


def _table_has_column(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    cols = con.execute(f'DESCRIBE "{table}"').fetchall()
    return any(c[0] == column for c in cols)


def _read_event(
    con: duckdb.DuckDBPyConnection, table: str, fallback_freq_hz: float = 10.0,
    axes: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns (ts, value, is_sparse_event). Handles two shapes seen in real
    LMU files for channels we'd expect to be "event-style":
      - true event tables: explicit (ts, value) columns, sparse rows,
        logged only on change -> is_sparse_event=True, caller should
        forward-fill (step-hold) when resampling.
      - plain tables: just `value`, sampled every tick like a continuous
        channel (e.g. TC in some recordings) -> is_sparse_event=False,
        caller should treat it like any continuous channel.
    """
    if _table_has_column(con, table, "ts"):
        rows = con.execute(f'SELECT ts, value FROM "{table}" ORDER BY ts').fetchall()
        if not rows:
            return np.array([]), np.array([]), True
        ts = np.array([r[0] for r in rows], dtype=np.float64)
        values = np.array([r[1] for r in rows], dtype=np.float64)
        return ts, values, True
    else:
        rows = con.execute(f'SELECT value FROM "{table}" ORDER BY rowid').fetchall()
        values = np.array([r[0] for r in rows], dtype=np.float64)
        return channel_time(len(values), fallback_freq_hz, *axes), values, False


def _unwrap_lap_distance(v: np.ndarray, drop_threshold_m: float = -500.0) -> np.ndarray:
    """
    Converts a wrapping Lap Dist signal (resets to ~0 at start/finish each
    lap) into a monotonically increasing total-distance signal, so linear
    interpolation during resampling doesn't cross the wrap discontinuity.
    Standard unwrap: whenever consecutive samples drop by more than
    drop_threshold_m, assume a wrap occurred and add a running offset equal
    to the size of that drop, so the signal keeps rising smoothly through
    the reset point.
    """
    if len(v) == 0:
        return v
    diffs = np.diff(v)
    wrap_mask = diffs < drop_threshold_m
    # at each wrap point, add back what was "lost" in the reset so the
    # signal continues rising instead of jumping down
    corrections = np.where(wrap_mask, -diffs, 0.0)
    offset = np.concatenate([[0.0], np.cumsum(corrections)])
    return v + offset


def _resample_to_grid(
    src_t: np.ndarray, src_v: np.ndarray, grid_t: np.ndarray,
    forward_fill: bool = False, categorical: bool = False,
) -> np.ndarray:
    if len(src_t) == 0:
        return np.full_like(grid_t, np.nan)
    if forward_fill or categorical:
        # step function: value holds until next change (correct for ABS/TC flags)
        idx = np.searchsorted(src_t, grid_t, side="right") - 1
        idx = np.clip(idx, 0, len(src_v) - 1)
        return src_v[idx]
    # linear interpolation for continuous physical quantities
    return np.interp(grid_t, src_t, src_v, left=src_v[0], right=src_v[-1])


def _read_lap_time_events(con: duckdb.DuckDBPyConnection) -> dict[int, float]:
    """
    Reads the v4 recorder's "Lap Time" event table (ts=mCurrentET at lap
    crossing, value=duration_seconds for the lap that just ended), keyed by
    the lap number it belongs to (the lap immediately preceding this event
    in the Lap table). Returns {} for native LMU files or older recorder
    files that don't have this table — callers must treat this as an
    optional cross-check only, never a required source.
    """
    try:
        rows = con.execute('SELECT ts, value FROM "Lap Time" ORDER BY ts').fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    # BUG FIX: this used to match by position (i-th Lap Time row = i-th Lap
    # row). the v4 recorder writes a (first crossing, 0.0) placeholder first, so
    # every lap got the previous lap's time (the "lap 0 duration mismatch …
    # event=0.000s" warnings); recorder.py has no placeholder and only
    # lined up by luck. Both recorders stamp a Lap Time row with the ts of
    # the crossing that ENDED the lap, which is the next lap's Lap row, so
    # match on that instead. Rows at the first crossing have no lap to
    # belong to and are dropped.
    try:
        lap_rows = con.execute('SELECT ts, value FROM "Lap" ORDER BY ts').fetchall()
    except Exception:
        return {}
    out: dict[int, float] = {}
    for ts, dur in rows:
        for j in range(1, len(lap_rows)):
            if abs(float(lap_rows[j][0]) - float(ts)) < 1e-3:
                out[int(lap_rows[j - 1][1])] = float(dur)
                break
    return out


# LMU's official lap time (Lap Time rows) and the clock difference between
# two crossings measure the same lap if they agree this well. The clock
# difference is quantised to the 10 ms game tick (a skipped tick: 20 ms).
OFFICIAL_LAP_TOLERANCE_S = 0.05


def _get_lap_boundaries(con: duckdb.DuckDBPyConnection) -> list[tuple[int, float, float]]:
    """
    Returns [(lap_number, start_ts, end_ts), ...] using the Lap event table.
    Lap table logs (ts, lap_number) each time a new lap begins.
    "Lap" must be quoted — unquoted it conflicts with DuckDB's own reserved
    identifiers in some versions, causing the CatalogError seen in prod.

    Cross-checks each computed duration (end_ts - start_ts) against the v4
    recorder's independent "Lap Time" event table when present. A mismatch
    beyond a small tolerance means a lap was registered out of order or a
    session reset happened mid-lap — this is logged (not raised) since the
    diffed duration is still usable, but it's a signal something upstream
    needs investigating.
    """
    try:
        rows = con.execute('SELECT ts, value FROM "Lap" ORDER BY ts').fetchall()
    except Exception:
        # recorder files written before the quoting fix, or files that
        # genuinely have no Lap table — fall back to treating the whole
        # recording as a single lap starting at t=0
        return []
    if not rows:
        return []

    lap_time_events = _read_lap_time_events(con)

    boundaries = []
    for i, (ts, lap_num) in enumerate(rows):
        end_ts = rows[i + 1][0] if i + 1 < len(rows) else None
        lap_num = int(lap_num)
        if end_ts is not None:
            diffed_duration = float(end_ts) - float(ts)
            reported_duration = lap_time_events.get(lap_num)
            if reported_duration is not None and abs(diffed_duration - reported_duration) > 0.25:
                print(
                    f"[telemetry] WARNING: lap {lap_num} duration mismatch — "
                    f"diffed={diffed_duration:.3f}s vs Lap Time event={reported_duration:.3f}s "
                    f"(tolerance 0.25s). Possible out-of-order lap or mid-lap session reset."
                )
        boundaries.append((lap_num, float(ts), end_ts))
    return boundaries


def kept_lap(meta: dict) -> int | None:
    """The lap a kept file holds (metadata KeptLap), else None.

    A kept file is one lap of a recording plus a 3 s margin either side. It
    keeps the Lap rows n (start) and n+1 (end) and lap n's Lap Time row, so
    read like any recording it would also list an open lap n+1 made of the
    trailing margin. Readers list only lap KeptLap when it is set."""
    try:
        return int(meta["KeptLap"])
    except (KeyError, TypeError, ValueError):
        return None


def kept_lap_of(con: duckdb.DuckDBPyConnection) -> int | None:
    """kept_lap of an open file's metadata; None without a metadata table."""
    try:
        return kept_lap(dict(con.execute("SELECT key, value FROM metadata").fetchall()))
    except Exception:
        return None


def load_telemetry_file(
    path: str | Path, grid_hz: float = 20.0
) -> list[LapTelemetry]:
    """
    Loads an LMU .duckdb telemetry file and returns one LapTelemetry per
    completed lap found in the recording, with all channels aligned onto
    a common time grid at grid_hz.
    """
    path = str(path)
    con = duckdb.connect(path, read_only=True)

    meta_rows = con.execute("SELECT key, value FROM metadata").fetchall()
    meta = {k: v for k, v in meta_rows}

    freqs = _read_channel_frequencies(con)
    unpopulated = _read_unpopulated_channels(con)
    axes = session_time_axes(con)

    # Build full-session time series for every channel first (game time, the
    # same clock as the Lap event timestamps), then slice per lap.
    session_channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    wrap_added: tuple[np.ndarray, np.ndarray] | None = None
    # Speed per row for held_rows (a moving car's scoring value cannot stand still)
    try:
        gs_series = (None if "Ground Speed" in unpopulated else
                     _read_continuous(con, "Ground Speed", freqs.get("Ground Speed", 100.0), axes))
    except duckdb.CatalogException:
        gs_series = None

    for ch in CONTINUOUS_CHANNELS:
        if ch in unpopulated:
            continue
        try:
            freq = freqs.get(ch, 10.0)  # fallback if a channel is missing from channelsList
            t, v = _read_continuous(con, ch, freq, axes)
            if ch in SCORING_HELD_CHANNELS:
                # BUG FIX / PREVIOUSLY: the held repeats (see held_rows) were
                # resampled as data, so on the 50 Hz grid these channels ran
                # in steps: ramp for 0.1 s, flat for 0.1 s. Topography probes
                # then sat up to ~10 m along from where the car was, and the
                # surface heights came out shifted on every slope. Fix: drop
                # the repeats; the grid interpolates between real updates.
                sp = (np.interp(t, gs_series[0], gs_series[1])
                      if gs_series is not None and len(gs_series[0]) else None)
                keep = ~held_rows(v, sp)
                t, v = t[keep], v[keep]
            if ch == "Lap Dist":
                # Lap Dist wraps from ~track_length back to 0 at the start/
                # finish line each lap. Linear interpolation across that
                # discontinuity produces large bogus negative jumps, so we
                # unwrap it into a monotonically increasing "total distance
                # traveled" signal before resampling, then re-wrap per lap
                # later using the known lap length.
                raw_v = v
                v = _unwrap_lap_distance(v)
                # what the unwrap added to each row; subtracting it again
                # gives the game's own Lap Dist back (see lap_dist_raw)
                wrap_added = (t, v - raw_v)
            session_channels[ch] = (t, v)
        except duckdb.CatalogException:
            continue  # channel not present in this file; skip gracefully

    wheel_series: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for ch in WHEEL_CHANNELS:
        if ch in unpopulated:
            continue
        try:
            freq = freqs.get(ch, 100.0)
            t, per_wheel = _read_wheel_channel(con, ch, freq, axes)
            # PREVIOUSLY an empty table (e.g. Brake Pressure in a session
            # converted from a safety copy made before 11/26/2026: real unit,
            # no rows) became four all-NaN columns. Skip it like the settings,
            # so `"Brake Pressure FL" in lap.channels` means there is data.
            if len(t):
                wheel_series[ch] = (t, per_wheel)
        except duckdb.CatalogException:
            continue

    event_series: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
    for ch in EVENT_CHANNELS:
        try:
            freq = freqs.get(ch, 10.0)
            event_series[ch] = _read_event(con, ch, fallback_freq_hz=freq, axes=axes)
        except duckdb.CatalogException:
            continue
    for ch in SETTING_EVENT_CHANNELS:
        try:
            ts, vals, is_sparse = _read_event(con, ch, axes=axes)
        except duckdb.CatalogException:
            continue                 # older file / native export: no such table
        if len(ts):
            event_series[ch] = (ts, vals, is_sparse)

    lap_boundaries = _get_lap_boundaries(con)
    keep = kept_lap(meta)
    if keep is not None:
        # kept file: only its lap, not the open lap of the trailing margin
        lap_boundaries = [b for b in lap_boundaries if b[0] == keep]
    official_times = _read_lap_time_events(con)
    con.close()

    laps: list[LapTelemetry] = []

    for lap_num, start_ts, end_ts in lap_boundaries:
        is_complete = end_ts is not None
        if end_ts is None:
            # Last lap in the file may be incomplete (session ended mid-lap);
            # still include it but caller should check lap_time_s sanity.
            # Use the longest available channel's max time as a fallback end.
            candidate_ends = [t[-1] for t, _ in session_channels.values() if len(t)]
            end_ts = max(candidate_ends) if candidate_ends else start_ts

        duration = end_ts - start_ts
        if duration <= 0:
            continue
        # Shown lap time: LMU's own 1 ms value when it is the same lap.
        lap_time = duration
        official = official_times.get(lap_num)
        if official and abs(official - duration) <= OFFICIAL_LAP_TOLERANCE_S:
            lap_time = official

        grid_t = np.arange(0.0, duration, 1.0 / grid_hz)
        abs_grid_t = grid_t + start_ts  # back to session-relative time for lookup

        channels: dict[str, np.ndarray] = {}

        for ch, (t, v) in session_channels.items():
            channels[ch] = _resample_to_grid(t, v, abs_grid_t, forward_fill=False)

        # Lap Dist in LMU's file is session-relative, not reset per lap, so it
        # can be negative or an arbitrary offset at the start of any given lap.
        # Normalize to 0 at lap start so distance values are actually usable
        # for mapping events to a position on track.
        lap_dist_raw = None
        if "Lap Dist" in channels:
            if wrap_added is not None:
                # The unwrapped signal is raw + a step offset that changes only
                # at a wrap row, so holding that offset (not interpolating it)
                # recovers the raw value between rows. The interval containing
                # the wrap stays at the last pre-wrap value (flat), like the
                # unwrapped signal itself.
                lap_dist_raw = channels["Lap Dist"] - _resample_to_grid(
                    wrap_added[0], wrap_added[1], abs_grid_t, forward_fill=True)
            channels["Lap Dist"] = channels["Lap Dist"] - channels["Lap Dist"][0]

        for ch, (t, per_wheel) in wheel_series.items():
            for wheel_label, v in per_wheel.items():
                channels[f"{ch} {wheel_label}"] = _resample_to_grid(
                    t, v, abs_grid_t, forward_fill=False,
                    categorical=ch in CATEGORICAL_CHANNELS,
                )

        for ch, (t, v, is_sparse) in event_series.items():
            channels[ch] = _resample_to_grid(t, v, abs_grid_t, forward_fill=is_sparse)

        laps.append(
            LapTelemetry(
                driver_name=meta.get("DriverName", "unknown"),
                car_name=meta.get("CarName", "unknown"),
                track_name=meta.get("TrackName", "unknown"),
                session_type=meta.get("SessionType", "unknown"),
                lap_number=lap_num,
                lap_time_s=lap_time,
                time_s=grid_t,
                channels=channels,
                meta=meta,
                is_complete=is_complete,
                lap_dist_raw=lap_dist_raw,
            )
        )

    return laps


def real_lap_flags(times: list[float], complete: list[bool]) -> list[bool]:
    """
    Which laps are REAL laps (one flag per lap, same order): complete, under
    600 s and at least 85 % of the median of the complete laps' times. Shorter
    "laps" are the unfinished last lap or a mid-lap join (see
    reference_lap_time). The median is the upper one of an even count, as
    reference_lap_time has always used.
    """
    ok = [c and t < 600 for t, c in zip(times, complete)]
    pool = sorted(t for t, o in zip(times, ok) if o)
    if not pool:
        return [False] * len(times)
    med = pool[len(pool) // 2]
    return [o and t >= 0.85 * med for t, o in zip(times, ok)]


def reference_lap_time(laps: list[LapTelemetry]) -> float | None:
    """
    The file's fastest REAL lap time, for "fastest lap" picks and the tire
    model's 115 % cutoff. BUG FIX (10/18/2026): min(lap_time_s) used to win
    with the unfinished last lap (Ferrari 14.1 s, Aston 1.4 s) or a mid-lap
    join (Barcelona L19 71.5 s vs ~106 s laps), so ingest_tire_laps rejected
    every real lap. Now: complete laps only, ignoring any shorter than 85 %
    of their median (real_lap_flags). None when no lap qualifies.
    """
    times = [l.lap_time_s for l in laps]
    flags = real_lap_flags(times, [l.is_complete for l in laps])
    real = [t for t, r in zip(times, flags) if r]
    return min(real) if real else None


@dataclass(frozen=True)
class LapInfo:
    """One lap of a file, from the Lap / Lap Time tables only (lap_index)."""

    lap_number: int
    lap_time_s: float
    complete: bool   # closed by a later Lap row (False: the file's open last lap)
    real: bool       # real_lap_flags: a lap to pick as "best"


def _last_session_time(con: duckdb.DuckDBPyConnection) -> float | None:
    """Where the recording stops: the last GPS Time value. Files without GPS
    Time put row N at N / frequency from 0 (channel_time), so their end is
    the longest channel's last row on that axis, as load_telemetry_file has it."""
    try:
        row = con.execute('SELECT value FROM "GPS Time" ORDER BY rowid DESC LIMIT 1').fetchone()
        if row is not None and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    ends = [(n - 1) / f for ch, f in _read_channel_frequencies(con).items()
            if f > 0 and (n := _row_count(con, ch)) > 0]
    return max(ends) if ends else None


def lap_index(con: duckdb.DuckDBPyConnection) -> list[LapInfo]:
    """
    The file's laps without loading its channels: the same lap numbers and
    complete-lap times as load_telemetry_file (LMU's official 1 ms time when
    it agrees within OFFICIAL_LAP_TOLERANCE_S), the open last lap ending at
    the last GPS Time value. Laps of 600 s or more are dropped, as /laps
    always did, and so are zero-length ones (the closing Lap row of a file
    that stops right at the line).
    """
    boundaries = _get_lap_boundaries(con)
    keep = kept_lap_of(con)
    if keep is not None:
        # kept file: only its lap, not the open lap of the trailing margin
        boundaries = [b for b in boundaries if b[0] == keep]
    if not boundaries:
        return []
    official_times = _read_lap_time_events(con)
    session_end = _last_session_time(con) if boundaries[-1][2] is None else None

    rows: list[tuple[int, float, bool]] = []
    for lap_num, start_ts, end_ts in boundaries:
        complete = end_ts is not None
        end = float(end_ts) if complete else (session_end if session_end is not None else start_ts)
        duration = end - start_ts
        if duration <= 0:
            continue
        lap_time = duration
        official = official_times.get(lap_num)
        if official and abs(official - duration) <= OFFICIAL_LAP_TOLERANCE_S:
            lap_time = official
        if lap_time >= 600:
            continue
        rows.append((lap_num, lap_time, complete))

    flags = real_lap_flags([t for _, t, _ in rows], [c for _, _, c in rows])
    return [LapInfo(n, t, c, r) for (n, t, c), r in zip(rows, flags)]


def get_car_setup_json(path: str | Path) -> dict | None:
    """
    Returns the lap file's CarSetup metadata field, parsed from its raw JSON
    string. Used by vehicle_dynamics.py for accurate steering ratio/lock
    values specific to this recording's actual setup, rather than falling
    back to hardcoded per-car defaults.
    """
    con = duckdb.connect(str(path), read_only=True)
    row = con.execute("SELECT value FROM metadata WHERE key = 'CarSetup'").fetchone()
    con.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None


def anonymize_metadata(path: str | Path) -> dict:
    """
    Returns the file's metadata dict with personally-identifying fields
    stripped. Use this instead of raw metadata access anywhere the result
    might get logged, cached to a shared DB, or shown outside this machine.
    """
    con = duckdb.connect(str(path), read_only=True)
    rows = con.execute("SELECT key, value FROM metadata").fetchall()
    con.close()
    meta = {k: v for k, v in rows}
    for sensitive_key in ("DriverName", "SteamID"):
        meta.pop(sensitive_key, None)
    return meta
