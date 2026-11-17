"""
Session index: what the Compare tab needs to know about every recording
(track, car class, layout length, coordinate system, laps) without loading
any file's channels.

Why a separate index: picking ghosts means filtering ~55 files by track,
class and layout. /laps per file runs one request each and still says
nothing about the layout or the coordinate system, and load_telemetry_file
takes seconds per big file. Here each file costs its metadata, the lap
tables and one scoring channel (Lap Dist), and the result is cached until
the file changes on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np

from .car_identity import identify_car
from .telemetry import (_get_lap_boundaries, _read_unpopulated_channels, channel_time,
                        kept_lap, lap_index, session_time_axes)
from .track_catalog import find_entry

# abs path -> ((st_mtime_ns, st_size), entry). A recording only changes while
# the recorder writes it (atomic rename) or when the user replaces it, and
# either changes mtime or size.
_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}


def _layout_len_m(con: duckdb.DuckDBPyConnection, tables: set[str]) -> float | None:
    """Median over complete laps of the largest raw Lap Dist inside each
    lap's time window. Raw Lap Dist wraps to 0 at the line, so its largest
    value in a lap is the game's length of the layout driven (to one scoring
    step, ~5-10 m at speed): two layouts of one track name (e.g. Le Mans
    with or without the Mulsanne chicanes) differ here, not in TrackName."""
    if "Lap Dist" not in tables or "Lap Dist" in _read_unpopulated_channels(con):
        return None
    v = con.execute('SELECT value FROM "Lap Dist" ORDER BY rowid').fetchnumpy()["value"]
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return None
    row = con.execute("SELECT frequency FROM channelsList WHERE channelName = 'Lap Dist'").fetchone()
    freq = float(row[0]) if row and row[0] else 10.0
    # Recorder files write Lap Dist on their 10 Hz gate over the game clock,
    # which skips ticks: rows are placed by the same time axes the loader
    # uses, not by N / frequency (see telemetry's "Real time axes" note).
    t = channel_time(len(v), freq, *session_time_axes(con))
    maxima = []
    for _lap, start, end in _get_lap_boundaries(con):
        if end is None:
            continue
        inside = (t >= float(start)) & (t <= float(end))
        if inside.any():
            maxima.append(float(np.nanmax(v[inside])))
    if not maxima:
        return None
    return round(float(np.median(maxima)), 1)


def _read_entry(path: Path, source: str) -> dict:
    con = duckdb.connect(str(path), read_only=True)
    try:
        meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
        tables = _tables(con)
        laps = lap_index(con)
        layout = _layout_len_m(con, tables)
    finally:
        con.close()
    track = meta.get("TrackName")
    car = identify_car(meta, meta.get("CarName", "unknown"))
    entry = find_entry(track) if track else None
    return {
        "source": source,
        "file": path.name,
        "track": track,
        "track_slug": entry["slug"] if entry else None,
        "car_model": car.model,
        "car_class": car.car_class,
        "layout_len_m": layout,
        # /trackmap projects GPS Latitude/Longitude when a file has them and
        # falls back to world Pos X/Z otherwise; lines in the two systems do
        # not overlay, so the Compare tab refuses to mix them.
        "coordinate_system": ("gps_projected" if {"GPS Latitude", "GPS Longitude"} <= tables
                              else "world_metres"),
        # the same rounding as /laps, so the two agree key for key
        "laps": [{"lap_number": l.lap_number, "lap_time_s": round(l.lap_time_s, 3),
                  "complete": l.complete, "real": l.real} for l in laps],
        # Kept laps (spec 4.3): a kept file is one lap of a recording under a
        # new name. The Compare tab and the keep UI need to know which
        # recording and lap it came from (so the same lap is not offered
        # twice, and an original whose laps are kept can be shown as such),
        # and whether a file is ours at all (only recorder files can be kept;
        # native LMU exports have no RecordedBy).
        "kept_from": meta.get("KeptFromFile") or None,
        "kept_lap": kept_lap(meta),
        "recorded_by": str(meta.get("RecordedBy") or ""),
    }


def file_entry(path: Path, source: str) -> dict:
    """The index entry for one file, from the cache while the file's
    (st_mtime_ns, st_size) is unchanged. A file that cannot be read (damaged,
    not a DuckDB file, an older layout without metadata) gets `error` and no
    laps instead of failing the whole index. Callers must not mutate it."""
    path = Path(path)
    key = str(path.resolve())
    try:
        st = os.stat(key)
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError as exc:
        return _error_entry(path, source, exc)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp and hit[1]["source"] == source:
        return hit[1]
    try:
        entry = _read_entry(path, source)
    except Exception as exc:
        entry = _error_entry(path, source, exc)
    _CACHE[key] = (stamp, entry)
    return entry


def _error_entry(path: Path, source: str, exc: Exception) -> dict:
    return {"source": source, "file": path.name, "track": None, "track_slug": None,
            "car_model": None, "car_class": "unknown", "layout_len_m": None,
            "coordinate_system": None, "laps": [],
            "kept_from": None, "kept_lap": None, "recorded_by": "",
            "error": f"{type(exc).__name__}: {exc}"}


def build_index(folders: list[tuple[str, Path]]) -> list[dict]:
    """Entries for every *.duckdb in each (source, folder), folders in the
    given order and files sorted by name inside each: the order /sessions
    lists them in."""
    out: list[dict] = []
    seen: set[str] = set()
    for source, folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for name in sorted(p.name for p in folder.glob("*.duckdb")):
            path = folder / name
            seen.add(str(path.resolve()))
            out.append(file_entry(path, source))
    # forget files that are gone, so the cache does not grow without bound.
    # list() first: another threadpool request may insert while this runs
    # ("dictionary changed size during iteration"); pop, as two requests
    # may prune the same key.
    for key in [k for k in list(_CACHE) if k not in seen]:
        _CACHE.pop(key, None)
    return out
