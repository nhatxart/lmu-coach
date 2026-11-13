"""
Keep laps (spec 2026-11-03 section 4): cut one lap of a recorder file into a
file of its own, so the recording can be cleaned out and the lap stays.

A kept file is the lap plus MARGIN_S seconds of the recording before its start
crossing and after its end crossing (clamped to the recording): filters run on
the whole file before the lap is sliced, and the lap needs both crossings. It
has the recorder's schema (same tables, channelsList / eventsList unchanged,
metadata copied plus the Kept* keys and GateDue), so every reader takes it as
it takes a recording; telemetry.kept_lap makes them list only lap KeptLap.

Rows are cut by game time, never by row index or fraction: the game clock skips
ticks and the recorder's 10 Hz gate re-anchors after every clock jump, so row N
is not at N / frequency (see telemetry.session_time_axes). 100 Hz tables are cut
by GPS Time, 10 Hz tables by the original's replayed 10 Hz times over the same
window, and GateDue lets the reader replay the gate in the cut exactly.

The caller (main.keep_laps) writes to .<name>.tmp, os.replace's it and checks
it with check_kept; this module never deletes anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

from . import telemetry

MARGIN_S = 3.0
# Block size of a kept file (bytes); see cut_lap for why not the default.
KEPT_BLOCK_SIZE = 65536
RECORDER_PREFIX = "lmu-coach-recorder"
# Tables cut by their own (ts) column rather than by a time axis
LAP_TABLES = ("Lap", "Lap Time")
POSE_ROWS_TABLE = "Other Cars"
# The original's lap time and the kept file's must agree this well (spec 4.2:
# within 1 ms; both are LMU's official 1 ms value, so normally exactly equal).
CHECK_LAP_TIME_S = 0.001
STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)$")


class CutError(Exception):
    """This lap of this file cannot be kept; the message says why (API 422)."""


@dataclass(frozen=True)
class CutInfo:
    name: str          # kept file name (kept_name)
    lap: int
    lap_time_s: float  # the original's lap time (official 1 ms value when it agrees)
    t_start: float     # game time of lap n's Lap row (start crossing)
    t_end: float       # game time of lap n+1's Lap row (end crossing)


def _safe(name: str) -> str:
    # recorder.py's _safe: the kept file is named like a recording
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in name)


def _lap_time_text(lap_time_s: float) -> str:
    # whole milliseconds first, so 59.9996 reads 1m00.000, not 0m60.000
    ms = int(round(lap_time_s * 1000))
    return f"{ms // 60000}m{ms % 60000 // 1000:02d}.{ms % 1000:03d}"


def kept_name(src_name: str, meta: dict, lap: int, lap_time_s: float) -> str:
    """`<track>-<car> L<n> <m>m<ss.sss>-<original's stamp>.duckdb` (user-approved).
    Deterministic per (original, lap), so keeping a lap twice finds its file."""
    m = STAMP_RE.search(Path(src_name).stem)
    stamp = m.group(1) if m else str(meta.get("RecordingTime", ""))
    if not STAMP_RE.fullmatch(stamp):
        raise CutError(f"{src_name} has no recording time stamp to name the kept file by")
    track = _safe(str(meta.get("TrackName", "unknown")))
    car = _safe(str(meta.get("CarName", "unknown")))
    return f"{track}-{car} L{lap} {_lap_time_text(lap_time_s)}-{stamp}.duckdb"


@dataclass(frozen=True)
class _Plan:
    info: CutInfo
    src_name: str
    meta: list[tuple[str, str]]       # original metadata rows, in order
    lap_rows: list[tuple[float, int]]  # Lap rows n and n+1, as stored
    ws: float                          # window start / end (game time, inclusive)
    we: float
    fast: tuple[int, int]              # [a, b) rows of the 100 Hz tables
    slow: tuple[int, int]              # [a, b) rows of the 10 Hz tables
    n_fast: int
    n_slow: int
    gate_due: float


def _plan(con: duckdb.DuckDBPyConnection, src: Path, lap: int) -> _Plan:
    meta_rows = [(str(k), str(v)) for k, v in
                 con.execute("SELECT key, value FROM metadata ORDER BY rowid").fetchall()]
    meta = dict(meta_rows)
    if not str(meta.get("RecordedBy", "")).startswith(RECORDER_PREFIX):
        raise CutError(f"{src.name} is not an lmu-coach recorder file (native LMU exports "
                       f"cannot be kept)")
    if telemetry.kept_lap(meta) is not None:
        raise CutError(f"{src.name} is already a kept lap (lap {meta.get('KeptLap')})")

    laps = {l.lap_number: l for l in telemetry.lap_index(con)}
    if lap not in laps:
        raise CutError(f"Lap {lap} is not in {src.name}")
    if not laps[lap].complete:
        raise CutError(f"Lap {lap} is not complete (the recording stops before its end crossing)")
    rows = [(float(ts), int(v)) for ts, v in
            con.execute('SELECT ts, value FROM "Lap" ORDER BY ts').fetchall()]
    at = [i for i, (_, v) in enumerate(rows) if v == lap]
    if len(at) != 1 or at[0] + 1 >= len(rows):
        raise CutError(f"Lap {lap} has {len(at)} start crossings in {src.name}")
    i = at[0]
    t_start, t_end = rows[i][0], rows[i + 1][0]

    t_fast, t_slow = telemetry.session_time_axes(con)
    if t_fast is None or t_slow is None:
        raise CutError(f"{src.name} has no GPS Time or no 10 Hz tables to cut by")
    if np.any(np.diff(t_fast) <= 0):
        raise CutError(f"GPS Time in {src.name} does not always increase")
    # The 10 Hz times must be the recorder's gate replayed over GPS Time (what
    # session_time_axes tries first); if it fell back to index spacing (v4
    # recorder, damaged file) they are not the gate's, GateDue would describe
    # nothing and the kept file's 10 Hz rows would drift silently.
    replay = telemetry._replay_slow_gate(t_fast)
    if len(replay) != len(t_slow) or not np.array_equal(t_fast[replay], t_slow):
        raise CutError(f"The 10 Hz rows of {src.name} do not follow the recorder's 10 Hz gate; "
                       f"a cut could not keep their times")

    # Every time-axis table must be one of the two rates, or it cannot be cut
    # by time (found here, so prepare refuses before anything is written).
    unpopulated = telemetry._read_unpopulated_channels(con)
    for (t,) in con.execute("SHOW TABLES").fetchall():
        if t in ("metadata", "channelsList", "eventsList") or t in unpopulated:
            continue
        if "ts" in [c[0] for c in con.execute(f"DESCRIBE {_q(t)}").fetchall()]:
            continue
        n = con.execute(f"SELECT count(*) FROM {_q(t)}").fetchone()[0]
        if n not in (0, len(t_fast), len(t_slow)):
            raise CutError(f"Table {t} of {src.name} has {n} rows, neither the 100 Hz "
                           f"({len(t_fast)}) nor the 10 Hz ({len(t_slow)}) count")

    ws = max(t_start - MARGIN_S, float(t_fast[0]))
    we = min(t_end + MARGIN_S, float(t_fast[-1]))
    fast = (int(np.searchsorted(t_fast, ws, "left")), int(np.searchsorted(t_fast, we, "right")))
    slow = (int(np.searchsorted(t_slow, ws, "left")), int(np.searchsorted(t_slow, we, "right")))
    info = CutInfo(kept_name(src.name, meta, lap, laps[lap].lap_time_s), lap,
                   float(laps[lap].lap_time_s), t_start, t_end)
    return _Plan(info, src.name, meta_rows, rows[i:i + 2], ws, we, fast, slow,
                 len(t_fast), len(t_slow), telemetry.slow_gate_due_at(t_fast, ws))


def prepare(src: Path, lap: int) -> CutInfo:
    """What cut_lap would write for this lap, without writing: raises CutError
    for a file or lap that cannot be kept. The endpoint checks every requested
    lap first, so one bad lap refuses the request before anything is written."""
    src = Path(src)
    con = duckdb.connect(str(src), read_only=True)
    try:
        return _plan(con, src, lap).info
    finally:
        con.close()


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def cut_lap(src: Path, lap: int, dst: Path, tyre_context: dict | None) -> CutInfo:
    """Write lap `lap` of recording `src` (plus MARGIN_S either side) to `dst`,
    a new file (the caller's .<name>.tmp). Raises CutError when the lap cannot
    be kept; on any error the half-written dst is removed (this call made it)."""
    src, dst = Path(src), Path(dst)
    con = duckdb.connect(str(src), read_only=True)
    try:
        plan = _plan(con, src, lap)
        unpopulated = telemetry._read_unpopulated_channels(con)
    finally:
        con.close()
    if dst.exists():
        raise CutError(f"{dst.name} already exists")

    # PREVIOUSLY duckdb.connect(dst) wrote DuckDB's default 256 KiB blocks.
    # Every table takes at least one block, and with ~70 tables a one-lap file
    # was ~40 % padding: Fuji 10/19 L6 17.58 MB vs 10.50 MB at 64 KiB, COTA
    # 11/09 L11 20.20 vs 12.33 MB (duckdb 1.1.3, the Docker pin; 16 KiB was no
    # smaller there). Fix: attach the new file with a 64 KiB block size.
    # Readable by DuckDB 1.1 and later (checked: 1.1.3, 1.2.2, 2.0.0).
    out = duckdb.connect(":memory:")
    try:
        out.execute("ATTACH '" + str(dst).replace("'", "''")
                    + f"' AS kept (BLOCK_SIZE {KEPT_BLOCK_SIZE})")
        out.execute("USE kept")
        out.execute("ATTACH '" + str(src).replace("'", "''") + "' AS src (READ_ONLY)")
        _write(out, plan, unpopulated, tyre_context)
        out.execute("DETACH src")
        out.execute("USE memory")
        out.execute("DETACH kept")
        out.close()
    except BaseException:
        out.close()
        for p in (dst, dst.with_name(dst.name + ".wal")):
            p.unlink(missing_ok=True)
        raise
    return plan.info


def _write(out: duckdb.DuckDBPyConnection, plan: _Plan, unpopulated: set[str],
           tyre_context: dict | None) -> None:
    ws, we = plan.ws, plan.we
    tables = [r[0] for r in out.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = 'src' "
        "ORDER BY table_name").fetchall()]
    out.execute("BEGIN TRANSACTION")
    n_fast_kept = plan.fast[1] - plan.fast[0]
    for t in tables:
        s = f"src.{_q(t)}"
        cols = [c[0] for c in out.execute(f"DESCRIBE {s}").fetchall()]
        empty = f"CREATE TABLE {_q(t)} AS SELECT * FROM {s} LIMIT 0"
        if t in ("channelsList", "eventsList"):
            out.execute(f"CREATE TABLE {_q(t)} AS SELECT * FROM {s} ORDER BY rowid")
        elif t == "metadata":
            out.execute(empty)
            out.executemany(f"INSERT INTO {_q(t)} VALUES (?, ?)",
                            _metadata(plan, n_fast_kept, tyre_context))
        elif t == "Lap":
            out.execute(empty)
            out.executemany(f"INSERT INTO {_q(t)} VALUES (?, ?)", plan.lap_rows)
        elif t == "Lap Time":
            # lap n's row: stamped at the crossing that ended it (Lap row n+1)
            out.execute(f"CREATE TABLE {_q(t)} AS SELECT * FROM {s} "
                        f"WHERE abs(ts - ?) < 1e-3 ORDER BY rowid", [plan.info.t_end])
        elif t in unpopulated:
            out.execute(empty)          # flagged empty by the recorder: stays empty
        elif "ts" in cols:
            # Other Cars (stamped with the player's 10 Hz gate time) and the
            # on-change event tables (ABS, TC, Gear, In Pits): rows in the window
            out.execute(empty)
            if t != POSE_ROWS_TABLE and cols == ["ts", "value"]:
                _seed_event(out, t, s, ws)
            out.execute(f"INSERT INTO {_q(t)} SELECT * FROM {s} "
                        f"WHERE ts >= ? AND ts <= ? ORDER BY rowid", [ws, we])
        else:
            n = out.execute(f"SELECT count(*) FROM {s}").fetchone()[0]
            if n == 0:
                out.execute(empty)
                continue
            if n == plan.n_fast:
                a, b = plan.fast
            elif n == plan.n_slow:
                a, b = plan.slow
            else:
                raise CutError(f"Table {t} has {n} rows, neither the 100 Hz ({plan.n_fast}) "
                               f"nor the 10 Hz ({plan.n_slow}) count")
            # one block of consecutive rows, in the recorded order
            out.execute(f"CREATE TABLE {_q(t)} AS SELECT * EXCLUDE (_rn) FROM ("
                        f"SELECT *, row_number() OVER (ORDER BY rowid) - 1 AS _rn FROM {s}) "
                        f"WHERE _rn >= {a} AND _rn < {b} ORDER BY _rn")
    out.execute("COMMIT")


def _seed_event(out, t: str, s: str, ws: float) -> None:
    """An on-change event table holds its value until the next row, so the
    state at the window start lives in the last row before it: write that
    value at the window start (the table's first row), unless the original
    has a row exactly there or none before it."""
    prev = out.execute(f"SELECT value FROM {s} WHERE ts < ? ORDER BY ts DESC, rowid DESC LIMIT 1",
                       [ws]).fetchone()
    if prev is None or out.execute(f"SELECT count(*) FROM {s} WHERE ts = ?", [ws]).fetchone()[0]:
        return
    out.execute(f"INSERT INTO {_q(t)} VALUES (?, ?)", [ws, prev[0]])


def _metadata(plan: _Plan, n_fast_kept: int, tyre_context: dict | None) -> list[tuple[str, str]]:
    info = plan.info
    new = {
        "TotalSamples": str(n_fast_kept),
        "LapCount": "1",               # completed laps: lap n only
        "KeptFromFile": plan.src_name,
        "KeptLap": str(info.lap),
        "KeptLapTime": f"{info.lap_time_s:.3f}",
        "KeptAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # the original's 10 Hz gate due time at the kept file's first row:
        # session_time_axes replays the kept file's gate from it
        "GateDue": repr(float(plan.gate_due)),
        # the original's tyre-model context (grip_profile.file_tyre_context,
        # read back by grip_profile.tire_file_key); strict JSON: a NaN here
        # would be no context a JSON reader outside Python accepts
        "KeptTyreContext": json.dumps(tyre_context or {}, allow_nan=False, sort_keys=True),
    }
    rows = [(k, new.pop(k, v)) for k, v in plan.meta]
    return rows + list(new.items())


def _axes_of(path: Path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return telemetry.session_time_axes(con)
    finally:
        con.close()


def check_kept(kept: Path, lap: int, lap_time_s: float,
               src: Path) -> tuple[bool, float | None, str | None]:
    """The keep check (spec 4.2): (passed, kept lap time, reason it failed).

    The backend re-reads the kept file like any file: exactly one lap, number
    `lap`, complete, its lap time equal to the original's within 1 ms, and it
    names `src` as its origin. Also its time axes against the original's over
    the kept window: 100 Hz rows identical, 10 Hz times within 1e-9 s. A
    wrong GateDue or 10 Hz cut makes session_time_axes fall back to index
    spacing silently, and the lap time alone would not notice."""
    kept, src = Path(kept), Path(src)
    try:
        con = duckdb.connect(str(kept), read_only=True)
        try:
            laps = telemetry.lap_index(con)
            meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
        finally:
            con.close()
        if [(l.lap_number, l.complete) for l in laps] != [(lap, True)]:
            return False, None, (f"expected one complete lap {lap}, read "
                                 f"{[(l.lap_number, l.complete) for l in laps]}")
        got = float(laps[0].lap_time_s)
        if abs(got - lap_time_s) > CHECK_LAP_TIME_S + 1e-9:
            return False, got, f"lap time {got:.3f} s, the original's is {lap_time_s:.3f} s"
        if meta.get("KeptFromFile") != src.name:
            return False, got, f"kept from {meta.get('KeptFromFile')!r}, not {src.name!r}"
        k_fast, k_slow = _axes_of(kept)
        o_fast, o_slow = _axes_of(src)
        if k_fast is None or k_slow is None or o_fast is None or o_slow is None:
            return False, got, "no GPS Time or 10 Hz rows to check the time axes with"
        lo, hi = float(k_fast[0]), float(k_fast[-1])
        exp_fast = o_fast[(o_fast >= lo) & (o_fast <= hi)]
        exp_slow = o_slow[(o_slow >= lo) & (o_slow <= hi)]
        if not np.array_equal(k_fast, exp_fast):
            return False, got, "its 100 Hz rows are not the original's"
        if len(k_slow) != len(exp_slow) or np.max(np.abs(k_slow - exp_slow)) > 1e-9:
            return False, got, "its 10 Hz times differ from the original's"
        return True, got, None
    except Exception as e:      # unreadable / half-written: a failed check
        return False, None, f"could not read it back: {type(e).__name__}: {e}"
