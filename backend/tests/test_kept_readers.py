"""A kept file (keep laps, spec section 4) is one lap of a recorder file plus
MARGIN_S = 3 s either side. It keeps the Lap rows n and n+1 and lap n's
Lap Time row, so the readers must list only lap n (not an open lap n+1 made
of the trailing margin), and the 10 Hz gate must be replayed from the
original gate's due time at the cut (metadata GateDue), not from the cut's
first row.

The cut is made here by hand from a synthetic recording, the way the keep
step will make it: rows outside [t_start - 3 s, t_end + 3 s] deleted.
"""

import shutil
from urllib.parse import quote

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, telemetry

from synthetic_session import tick_times, write_session

MARGIN_S = 3.0
KEPT = 1
# Lap 1 runs 50.05 -> 90.05. Ticks are dropped (20 ms steps, even ms) until
# 48.0, so the cut's first tick (47.06) is off the original 10 Hz gate's
# phase (47.0, 47.1, ...): anchored at the cut's first row, the gate would
# fire 0.04 s early on every 10 Hz row.
LAPS = ((10.0, 0), (50.05, 1), (90.05, 2))
LAP_TIMES = ((50.05, 40.05), (90.05, 40.0))


def _times():
    return tick_times(start=10.0, end=130.0, drop_until=48.0)


def _jump_times():
    """Clean 10 ms ticks from 10.0 s with one 0.537 s stall at 30.0 s:
    recorder.py's gate re-anchors at et + 0.1 there, so after it the 10 Hz
    phase (.x37) has nothing to do with the first tick (10.0) any more, and
    the cut's first tick (47.057) is on neither phase."""
    out, ms = [], 10_000
    while ms <= 130_000:
        out.append(ms / 1000.0)
        ms += 537 if ms == 30_000 else 10
    return out


def _original(tmp_path, times):
    (tmp_path / "orig").mkdir()
    return write_session(tmp_path / "orig" / "Test Track-Test Car _1-2026-10-14T00_00_00Z.duckdb",
                         laps=LAPS, lap_times=LAP_TIMES, times=times)


@pytest.fixture
def original(tmp_path):
    return _original(tmp_path, _times())


def _axes(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return telemetry.session_time_axes(con)
    finally:
        con.close()


def _cut(src, dst, gate_due=True):
    """Copy src to dst and keep one lap of it, like a kept file."""
    t_fast, t_slow = _axes(src)
    t0, t1 = LAPS[KEPT][0] - MARGIN_S, LAPS[KEPT + 1][0] + MARGIN_S
    shutil.copy(src, dst)
    con = duckdb.connect(str(dst))
    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()]
        for t in tables:
            if t in ("metadata", "channelsList", "Lap", "Lap Time"):
                continue
            n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            axis = t_fast if n == len(t_fast) else t_slow if n == len(t_slow) else None
            assert axis is not None, t
            keep = np.nonzero((axis >= t0) & (axis <= t1))[0]
            a, b = int(keep[0]), int(keep[-1])
            con.execute(f'CREATE TABLE "_cut" AS SELECT * EXCLUDE (rn) FROM ('
                        f'SELECT *, row_number() OVER (ORDER BY rowid) - 1 AS rn FROM "{t}") '
                        f'WHERE rn BETWEEN {a} AND {b} ORDER BY rn')
            con.execute(f'DROP TABLE "{t}"')
            con.execute(f'ALTER TABLE "_cut" RENAME TO "{t}"')
        con.execute(f'DELETE FROM "Lap" WHERE value NOT IN ({KEPT}, {KEPT + 1})')
        con.execute(f'DELETE FROM "Lap Time" WHERE abs(ts - {LAPS[KEPT + 1][0]}) > 1e-3')
        meta = [("KeptFromFile", src.name), ("KeptLap", str(KEPT)),
                ("KeptLapTime", "40.000"), ("KeptAt", "2026-11-10T00:00:00Z")]
        if gate_due:
            meta.append(("GateDue", repr(telemetry.slow_gate_due_at(t_fast, t0))))
        con.executemany("INSERT INTO metadata VALUES (?, ?)", meta)
    finally:
        con.close()
    return dst


@pytest.fixture
def kept(original, tmp_path):
    return _cut(original, tmp_path / "Test Track-Test Car _1 L1 0m40.000-2026-10-14T00_00_00Z.duckdb")


def _lap_index(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return telemetry.lap_index(con)
    finally:
        con.close()


def test_kept_lap_reads_metadata():
    assert telemetry.kept_lap({"KeptLap": "7"}) == 7
    assert telemetry.kept_lap({}) is None
    assert telemetry.kept_lap({"KeptLap": "x"}) is None


def test_lap_index_lists_only_kept_lap(original, kept):
    laps = _lap_index(kept)
    assert [(l.lap_number, l.complete) for l in laps] == [(1, True)]
    orig = {l.lap_number: l for l in _lap_index(original)}
    assert laps[0].lap_time_s == pytest.approx(orig[1].lap_time_s, abs=0.001)


def test_load_telemetry_file_only_kept_lap(kept):
    laps = telemetry.load_telemetry_file(kept)
    assert [l.lap_number for l in laps] == [1]


def test_trackmap_auto_pick_is_kept_lap(kept, monkeypatch):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", kept.parent)
    client = TestClient(main.app)
    url = f"/sessions/sessions/{quote(kept.name)}/trackmap"
    r = client.get(url)
    assert r.status_code == 200, r.text[:300]
    assert r.json()["lap_number"] == 1
    assert client.get(url, params={"lap_number": 2}).status_code == 404


def test_trackmap_kept_lap_without_lap_row_is_422(kept, monkeypatch):
    # A malformed kept file (KeptLap names a lap it has no Lap row for) must
    # not be served as the whole file presented as that lap.
    con = duckdb.connect(str(kept))
    try:
        con.execute("UPDATE metadata SET value = '5' WHERE key = 'KeptLap'")
    finally:
        con.close()
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", kept.parent)
    r = TestClient(main.app).get(f"/sessions/sessions/{quote(kept.name)}/trackmap")
    assert r.status_code == 422, r.text[:300]


def _assert_kept_slow_axis(original, kept, tmp_path):
    _, slow_orig = _axes(original)
    t0, t1 = LAPS[KEPT][0] - MARGIN_S, LAPS[KEPT + 1][0] + MARGIN_S
    expected = slow_orig[(slow_orig >= t0) & (slow_orig <= t1)]
    _, slow_kept = _axes(kept)
    assert len(slow_kept) == len(expected)
    assert np.max(np.abs(slow_kept - expected)) < 1e-9
    # Guard: without GateDue the gate is anchored at the cut's first row and
    # lands off the original's phase, so the test really exercises the anchor.
    bare = _cut(original, tmp_path / "bare.duckdb", gate_due=False)
    _, slow_bare = _axes(bare)
    assert len(slow_bare) != len(expected) or np.max(np.abs(slow_bare - expected)) > 0.01


def test_gate_due_keeps_slow_axis(original, kept, tmp_path):
    _assert_kept_slow_axis(original, kept, tmp_path)


def test_gate_due_after_clock_jump(tmp_path):
    # PREVIOUSLY (6039d59, metadata GateOrigin) the reader walked 0.1 s steps
    # from the original's first tick, so after this stall it replayed the
    # wrong phase (0.04 s off) with the right row count, silently.
    original = _original(tmp_path, _jump_times())
    kept = _cut(original, tmp_path / "Test Track-Test Car _1 L1 0m40.000-2026-10-14T00_00_00Z.duckdb")
    _assert_kept_slow_axis(original, kept, tmp_path)


def test_slow_gate_due_at_cut_start_is_first_tick():
    # A cut clamped at the recording start: the gate starts at the first tick.
    t = np.array(_times())
    assert telemetry.slow_gate_due_at(t, t[0] - 3.0) == t[0]
    assert telemetry.slow_gate_due_at(t, t[0]) == t[0]
