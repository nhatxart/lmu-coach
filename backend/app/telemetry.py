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
    "Gear",  # not in channelsList but present as a plain table; handled defensively
    "Yaw Accel",  # rad/s^2, recorder_hf only - tire model's steady-state filter
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
    # mSuspForce, pushrod load in N (recorder_hf). Both are dropped by
    # _read_channel_frequencies when the recorder marked them unpopulated.
    "Tire Load",
    "Susp Force",
    # recorder_hf only; absent tables are skipped like any missing channel.
    "Camber",             # rad per wheel
    "TyresWear",          # fraction per wheel
    "TyresCarcassTemp",   # Celsius per wheel
    "Surface Type",       # categorical code per wheel (see CATEGORICAL_CHANNELS)
    "Tire Lateral Force", # N; zero/flagged unpopulated on GT3
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
    # TyreCompoundFront from recorder_hf. Shared by every lap of a file.
    meta: dict[str, str] = field(default_factory=dict)
    # False for the file's last lap: it has no closing Lap event, so it ends
    # wherever the recording stopped (a 1.4 s "lap" was seen). Callers that
    # pick a "fastest" lap must skip these (see reference_lap_time).
    is_complete: bool = True


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
SLOW_PERIOD_S = 0.1   # recorder_hf.py's SLOW_PERIOD_S
SLOW_ROW_TABLES = ("GPS Latitude", "Lap Dist", "Path Lateral")


def _replay_slow_gate(t_fast: np.ndarray) -> np.ndarray:
    """Indices into t_fast where recorder_hf.py wrote a 10 Hz row: the first
    tick at or after each due time, re-anchored after a clock jump."""
    nxt, idx = float(t_fast[0]), []
    for i, et in enumerate(t_fast.tolist()):
        if et >= nxt:
            nxt += SLOW_PERIOD_S
            if nxt <= et:
                nxt = et + SLOW_PERIOD_S
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def _row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    try:
        return int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
    except Exception:
        return 0


def session_time_axes(
    con: duckdb.DuckDBPyConnection,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(t_fast, t_slow): game time (s) of each 100 Hz row and each 10 Hz row.

    t_fast is GPS Time itself. t_slow replays the recorder_hf.py gate; if
    that does not reproduce the file's 10 Hz row count (recorder.py decimated
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
    idx = _replay_slow_gate(t_fast)
    if len(idx) == n_slow:
        return t_fast, t_fast[idx]
    # For recorder_hf.py files the replay must match; the index fallback
    # below is only exact for recorder.py/native files. A mismatch means the
    # recorder's gate changed (keep SLOW_PERIOD_S and _replay_slow_gate in
    # step with recorder_hf.py) or the file is damaged, and the map will
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
    # row). recorder.py writes a (first crossing, 0.0) placeholder first, so
    # every lap got the previous lap's time (the "lap 0 duration mismatch …
    # event=0.000s" warnings); recorder_hf.py has no placeholder and only
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

    for ch in CONTINUOUS_CHANNELS:
        if ch in unpopulated:
            continue
        try:
            freq = freqs.get(ch, 10.0)  # fallback if a channel is missing from channelsList
            t, v = _read_continuous(con, ch, freq, axes)
            if ch == "Lap Dist":
                # Lap Dist wraps from ~track_length back to 0 at the start/
                # finish line each lap. Linear interpolation across that
                # discontinuity produces large bogus negative jumps, so we
                # unwrap it into a monotonically increasing "total distance
                # traveled" signal before resampling, then re-wrap per lap
                # later using the known lap length.
                v = _unwrap_lap_distance(v)
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

    lap_boundaries = _get_lap_boundaries(con)
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
        if "Lap Dist" in channels:
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
            )
        )

    return laps


def reference_lap_time(laps: list[LapTelemetry]) -> float | None:
    """
    The file's fastest REAL lap time, for "fastest lap" picks and the tire
    model's 115 % cutoff. BUG FIX (10/18/2026): min(lap_time_s) used to win
    with the unfinished last lap (Ferrari 14.1 s, Aston 1.4 s) or a mid-lap
    join (Barcelona L19 71.5 s vs ~106 s laps), so ingest_tire_laps rejected
    every real lap. Now: complete laps only, ignoring any shorter than 85 %
    of their median. None when no lap qualifies.
    """
    times = sorted(l.lap_time_s for l in laps if l.is_complete and l.lap_time_s < 600)
    if not times:
        return None
    med = times[len(times) // 2]
    real = [t for t in times if t >= 0.85 * med]
    return real[0] if real else None


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
