"""Keep laps (spec section 4, amendment 4.4a): POST /sessions/sessions/{file}/keep
cuts each lap into its own file in the kept-laps folder (/data/reference_laps):
the lap plus lap_cut.MARGIN_S either side, recorder schema, written as
.<name>.tmp, os.replace'd and checked by re-reading it.

Every recording here is synthetic and lives in a tmp dir; main's two folders
are monkeypatched onto it.
"""

import os
import re
from pathlib import Path
from urllib.parse import quote

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import lap_cut, main, telemetry

from synthetic_session import tick_times, write_session
from synthetic_track import write_hilly_session

ROOT = Path(__file__).resolve().parents[2]
HILLY = "Synthetic Hill Ring-Synthetic Car _1-2026-10-19T23_18_37Z.duckdb"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    sessions, kept = tmp_path / "sessions", tmp_path / "reference_laps"
    sessions.mkdir()
    kept.mkdir()
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", sessions)
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", kept)
    return sessions, kept


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def hilly(dirs):
    return write_hilly_session(dirs[0] / HILLY)


def _keep(client, name, laps):
    return client.post(f"/sessions/sessions/{quote(name)}/keep", json={"laps": laps})


def _con(path):
    return duckdb.connect(str(path), read_only=True)


def _axes(path):
    con = _con(path)
    try:
        return telemetry.session_time_axes(con)
    finally:
        con.close()


def _laps(path):
    con = _con(path)
    try:
        return telemetry.lap_index(con)
    finally:
        con.close()


def _meta(path):
    con = _con(path)
    try:
        return dict(con.execute("SELECT key, value FROM metadata").fetchall())
    finally:
        con.close()


def _rows(path, sql):
    con = _con(path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _lap_bounds(path, lap):
    rows = _rows(path, 'SELECT ts, value FROM "Lap" ORDER BY ts')
    i = [int(v) for _, v in rows].index(lap)
    return float(rows[i][0]), float(rows[i + 1][0])


def _assert_slow_axis_kept(original, kept, lap):
    """The kept file's 10 Hz times (session_time_axes, replayed from GateDue)
    equal the original's for the kept rows. The keep check compares lap times
    only, so a silent fallback to index spacing would pass it."""
    t_fast, t_slow = _axes(original)
    t0, t1 = _lap_bounds(original, lap)
    ws = max(t0 - lap_cut.MARGIN_S, t_fast[0])
    we = min(t1 + lap_cut.MARGIN_S, t_fast[-1])
    k_fast, k_slow = _axes(kept)
    exp_fast = t_fast[(t_fast >= ws) & (t_fast <= we)]
    exp_slow = t_slow[(t_slow >= ws) & (t_slow <= we)]
    assert len(k_fast) == len(exp_fast) and np.array_equal(k_fast, exp_fast)
    assert len(k_slow) == len(exp_slow)
    assert np.max(np.abs(k_slow - exp_slow)) < 1e-9
    return ws, we


# ── endpoint, happy path ─────────────────────────────────────────────────────

def test_keep_writes_one_lap_file(hilly, dirs, client):
    r = _keep(client, HILLY, [1])
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["ok"] and body["all_checked"]
    (res,) = body["results"]
    assert res["status"] == "kept" and res["lap"] == 1 and res["error"] is None
    assert (dirs[1] / res["kept_file"]).is_file()
    assert sorted(p.name for p in dirs[1].iterdir()) == [res["kept_file"]]   # no .tmp left
    laps = client.get(f"/sessions/reference_laps/{quote(res['kept_file'])}/laps").json()["laps"]
    assert len(laps) == 1
    orig = {l.lap_number: l for l in _laps(hilly)}
    assert laps[0]["lap_number"] == 1 and laps[0]["complete"] and laps[0]["real"]
    assert laps[0]["lap_time_s"] == pytest.approx(orig[1].lap_time_s, abs=0.001)
    assert res["lap_time_original"] == pytest.approx(orig[1].lap_time_s, abs=0.001)
    assert res["lap_time_kept"] == pytest.approx(orig[1].lap_time_s, abs=0.001)


def test_cut_trackmap_matches_original_lap(hilly, dirs, client):
    res = _keep(client, HILLY, [1]).json()["results"][0]
    a = client.get(f"/sessions/sessions/{quote(HILLY)}/trackmap", params={"lap_number": 1}).json()
    b = client.get(f"/sessions/reference_laps/{quote(res['kept_file'])}/trackmap").json()
    assert b["lap_number"] == 1
    assert a["coordinate_origin"] == b["coordinate_origin"]
    la, lb = a["driven_line"], b["driven_line"]
    assert len(la["x"]) == len(lb["x"]) > 100
    for key, tol in (("x", 0.05), ("z", 0.05), ("t", 0.011), ("lap_dist", 0.5),
                     ("speed", 1e-6), ("throttle", 1e-6), ("brake", 1e-6)):
        va, vb = np.asarray(la[key], float), np.asarray(lb[key], float)
        assert len(va) == len(vb) > 0, key
        assert np.max(np.abs(va - vb)) <= tol, key


# ── cut rules ────────────────────────────────────────────────────────────────

LAPS = ((10.0, 0), (50.05, 1), (90.05, 2))
LAP_TIMES = ((50.05, 40.05), (90.05, 40.0))
ORIG = "Test Track-Test Car _1-2026-10-14T00_00_00Z.duckdb"


def _original(dirs, times=None):
    return write_session(dirs[0] / ORIG, laps=LAPS, lap_times=LAP_TIMES,
                         times=tick_times(start=10.0, end=130.0, drop_until=48.0)
                         if times is None else times)


def _cut(src, lap, kept_dir):
    info = lap_cut.prepare(src, lap)
    dst = kept_dir / info.name
    out = lap_cut.cut_lap(src, lap, dst, None)
    assert out == info
    return dst, info


def test_cut_event_seed_rows(dirs):
    src = _original(dirs)
    con = duckdb.connect(str(src))
    con.execute('CREATE TABLE "ABS" (ts DOUBLE, value DOUBLE)')
    con.executemany('INSERT INTO "ABS" VALUES (?, ?)',
                    [(12.0, 1.0), (30.0, 0.0), (45.0, 1.0), (60.0, 0.0), (95.0, 1.0)])
    con.execute('CREATE TABLE "Gear" (ts DOUBLE, value INTEGER)')
    con.executemany('INSERT INTO "Gear" VALUES (?, ?)', [(11.0, 3), (40.0, 5), (70.0, 4)])
    con.execute('CREATE TABLE "TC" (ts DOUBLE, value DOUBLE)')          # nothing before the window
    con.executemany('INSERT INTO "TC" VALUES (?, ?)', [(60.0, 1.0), (61.0, 0.0)])
    con.execute('CREATE TABLE "In Pits" (ts DOUBLE, value DOUBLE)')     # one row, long before
    con.execute('INSERT INTO "In Pits" VALUES (10.0, 0.0)')
    con.close()
    dst, info = _cut(src, 1, dirs[1])
    ws = 50.05 - lap_cut.MARGIN_S
    assert info.t_start == 50.05 and info.t_end == 90.05
    q = lambda t: _rows(dst, f'SELECT ts, value FROM "{t}" ORDER BY rowid')
    assert q("ABS") == [(ws, 1.0), (60.0, 0.0)]      # seed = last value before the window
    assert q("Gear") == [(ws, 5), (70.0, 4)]
    assert q("TC") == [(60.0, 1.0), (61.0, 0.0)]     # no seed without an earlier row
    assert q("In Pits") == [(ws, 0.0)]
    assert q("Lap") == [(50.05, 1), (90.05, 2)]
    assert q("Lap Time") == [(90.05, 40.0)]


def test_cut_after_skipped_ticks_keeps_positions(dirs):
    # Review Focus 1: the original lost every other tick until 48.0 s, so the
    # cut's first tick (47.06) is off the original 10 Hz gate's phase.
    src = _original(dirs)
    dst, _ = _cut(src, 1, dirs[1])
    ws, we = _assert_slow_axis_kept(src, dst, 1)
    # values are time tags (synthetic_session): the kept rows are the same rows
    lon = lambda p: np.array([r[0] for r in _rows(p, 'SELECT value FROM "GPS Longitude" ORDER BY rowid')])
    _, t_slow = _axes(src)
    rows = lon(src)[(t_slow >= ws) & (t_slow <= we)]
    assert np.array_equal(lon(dst), rows)
    assert [l.lap_number for l in _laps(dst)] == [1]


def _jump_times():
    """10 ms ticks with one 0.537 s stall at 30.0 s: recorder.py re-anchors
    its gate there, so the cut's phase cannot come from the first tick."""
    out, ms = [], 10_000
    while ms <= 130_000:
        out.append(ms / 1000.0)
        ms += 537 if ms == 30_000 else 10
    return out


def test_cut_after_clock_jump_keeps_slow_axis(dirs):
    src = _original(dirs, _jump_times())
    dst, _ = _cut(src, 1, dirs[1])
    _assert_slow_axis_kept(src, dst, 1)
    assert float(_meta(dst)["GateDue"]) != float(_axes(dst)[0][0])   # really anchored


def test_cut_first_and_last_lap_clamp_margins(hilly, dirs):
    # Review Focus 2: lap 0 starts at the first tick, lap 2 ends at the last.
    t_fast, _ = _axes(hilly)
    for lap in (0, 2):
        dst, info = _cut(hilly, lap, dirs[1])
        k_fast, _ = _axes(dst)
        if lap == 0:
            assert k_fast[0] == t_fast[0]
            assert float(_meta(dst)["GateDue"]) == t_fast[0]
        else:
            assert k_fast[-1] == t_fast[-1]
        laps = _laps(dst)
        assert [(l.lap_number, l.complete) for l in laps] == [(lap, True)]
        orig = {l.lap_number: l for l in _laps(hilly)}
        assert laps[0].lap_time_s == pytest.approx(orig[lap].lap_time_s, abs=0.001)
        _assert_slow_axis_kept(hilly, dst, lap)


def test_cut_metadata_keys(dirs):
    src = _original(dirs)
    dst, info = _cut(src, 1, dirs[1])
    meta, orig = _meta(dst), _meta(src)
    for k, v in orig.items():
        assert meta[k] == v
    assert meta["KeptFromFile"] == ORIG
    assert meta["KeptLap"] == "1"
    assert meta["KeptLapTime"] == "40.000"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["KeptAt"])
    t_fast, _ = _axes(src)
    assert meta["GateDue"] == repr(telemetry.slow_gate_due_at(t_fast, 50.05 - lap_cut.MARGIN_S))
    assert meta["KeptTyreContext"] == "{}"
    assert meta["LapCount"] == "1"
    assert meta["TotalSamples"] == str(len(_axes(dst)[0]))
    assert info.name == "Test Track-Test Car _1 L1 0m40.000-2026-10-14T00_00_00Z.duckdb"
    # the user-approved example
    assert lap_cut.kept_name(
        "Fuji Speedway-Team WRT 2026 _69_LM-2026-10-19T23_18_37Z.duckdb",
        {"TrackName": "Fuji Speedway", "CarName": "Team WRT 2026 #69:LM"}, 9, 102.983,
    ) == "Fuji Speedway-Team WRT 2026 _69_LM L9 1m42.983-2026-10-19T23_18_37Z.duckdb"


def test_cut_size_small(dirs):
    # Long laps (500 s; 600 s is no lap any more) and three noisy 100 Hz
    # wheel tables, like the real ones: DuckDB stores a table in whole
    # 256 KB blocks, and the smooth synthetic channels compress so well that
    # this fixed cost, not the cut, decides the ratio of small files
    # (measured 0.52 with 300 s laps and no noisy tables, 0.48 with them).
    # Real recordings: one lap of 13 kept in 21 % of the bytes.
    src = write_hilly_session(dirs[0] / HILLY, speed_scale=0.06, with_surface=False)
    con = duckdb.connect(str(src))
    n = con.execute('SELECT count(*) FROM "GPS Time"').fetchone()[0]
    for t in ("Wheel Speed", "Susp Pos", "Susp Force"):
        con.execute(f'CREATE TABLE "{t}" AS SELECT random() AS value1, random() AS value2, '
                    f'random() AS value3, random() AS value4 FROM range({n}) ORDER BY range')
        con.execute("INSERT INTO channelsList VALUES (?, 100.0, '')", [t])
    con.close()
    dst, _ = _cut(src, 1, dirs[1])
    ratio = dst.stat().st_size / src.stat().st_size
    assert ratio < 0.45, ratio


def test_cut_unpopulated_tables_stay_empty(dirs):
    src = _original(dirs)
    con = duckdb.connect(str(src))
    con.execute('CREATE TABLE "Tire Load" (value1 DOUBLE, value2 DOUBLE, value3 DOUBLE, value4 DOUBLE)')
    con.execute("INSERT INTO channelsList VALUES ('Tire Load', 100.0, 'unpopulated_by_lmu')")
    con.close()
    dst, _ = _cut(src, 1, dirs[1])
    assert _rows(dst, 'SELECT count(*) FROM "Tire Load"') == [(0,)]
    assert _rows(dst, "SELECT * FROM channelsList ORDER BY rowid") == \
        _rows(src, "SELECT * FROM channelsList ORDER BY rowid")


def test_cut_refuses_unreplayable_gate(dirs):
    # A 10 Hz table whose rows do not replay recorder.py's gate (v4 recorder:
    # decimated by tick count): its times are not the gate's, so GateDue would
    # describe nothing. Refuse rather than write a file that drifts silently.
    src = _original(dirs)
    con = duckdb.connect(str(src))
    for t in ("GPS Latitude", "GPS Longitude", "Path Lateral"):
        con.execute(f'DELETE FROM "{t}" WHERE rowid = 5')
    con.close()
    with pytest.raises(lap_cut.CutError):
        lap_cut.prepare(src, 1)


def test_cut_refuses_table_of_unknown_rate(dirs):
    src = _original(dirs)
    con = duckdb.connect(str(src))
    con.execute('CREATE TABLE "Odd" AS SELECT range::DOUBLE AS value FROM range(7)')
    con.close()
    with pytest.raises(lap_cut.CutError, match="Odd"):
        lap_cut.prepare(src, 1)


# ── endpoint: repeat, failure, folders, refusals ─────────────────────────────

def test_keep_twice_is_already_kept(hilly, dirs, client):
    first = _keep(client, HILLY, [1]).json()["results"][0]
    path = dirs[1] / first["kept_file"]
    before = path.stat().st_mtime_ns
    r = _keep(client, HILLY, [1]).json()
    assert r["all_checked"]
    (res,) = r["results"]
    assert res["status"] == "already_kept" and res["kept_file"] == first["kept_file"]
    assert path.stat().st_mtime_ns == before


def test_keep_failed_check_removes_file(hilly, dirs, client, monkeypatch):
    monkeypatch.setattr(lap_cut, "check_kept",
                        lambda *a, **k: (False, None, "lap time differs"))
    r = _keep(client, HILLY, [1])
    assert r.status_code == 200
    body = r.json()
    assert body["all_checked"] is False
    (res,) = body["results"]
    assert res["status"] == "failed" and "lap time differs" in res["error"]
    assert list(dirs[1].iterdir()) == []
    assert (dirs[0] / HILLY).is_file()


def test_keep_reports_read_only_folder(hilly, dirs, client, monkeypatch):
    # Review Focus 4: an old compose still mounts the folder :ro.
    def deny(*a, **k):
        raise PermissionError(13, "Read-only file system")
    monkeypatch.setattr(os, "replace", deny)
    r = _keep(client, HILLY, [1])
    assert r.status_code == 409
    assert r.json() == {"ok": False, "error":
                        "The kept-laps folder is read-only: rebuild with the new docker-compose.yml"}
    assert list(dirs[1].iterdir()) == []
    assert (dirs[0] / HILLY).is_file()


def test_keep_refuses_native_and_kept(dirs, client):
    native = write_session(dirs[0] / "Test Track_P_2026-10-14T00_00_00Z.duckdb",
                           recorded_by="LMU")
    r = _keep(client, native.name, [1])
    assert r.status_code == 422 and "recorder" in r.json()["detail"]
    src = _original(dirs)
    res = _keep(client, ORIG, [1]).json()["results"][0]
    kept = dirs[1] / res["kept_file"]
    copy = dirs[0] / kept.name
    copy.write_bytes(kept.read_bytes())
    r = _keep(client, copy.name, [1])
    assert r.status_code == 422 and "kept" in r.json()["detail"]
    r = _keep(client, ORIG, [1, 7])                  # one bad lap: nothing written
    assert r.status_code == 422 and "7" in r.json()["detail"]
    r = _keep(client, ORIG, [2])                     # the open last lap
    assert r.status_code == 422 and "complete" in r.json()["detail"]
    assert sorted(p.name for p in dirs[1].iterdir()) == [kept.name]
    assert src.is_file()


@pytest.mark.parametrize("name", [r"..\x.duckdb", r"C:\x.duckdb", "x.txt", "..x.duckdb",
                                  ".x.duckdb.tmp", " x.duckdb", r"a\b.duckdb"])
def test_keep_rejects_bad_names(dirs, client, name, tmp_path):
    # Review Focus 5: bare names only, resolved inside the sessions folder.
    (tmp_path / "x.duckdb").write_bytes(b"not read")
    r = _keep(client, name, [1])
    assert r.status_code == 400, r.text[:200]
    assert list(dirs[1].iterdir()) == []


def test_keep_rejects_slash_name(dirs, client):
    # "a/b.duckdb" never reaches the handler: the router has no such path.
    r = client.post("/sessions/sessions/a/b.duckdb/keep", json={"laps": [1]})
    assert r.status_code in (400, 404)
    assert list(dirs[1].iterdir()) == []


# ── docker-compose.yml ───────────────────────────────────────────────────────

def test_compose_mounts_writable():
    lines = (ROOT / "docker-compose.yml").read_text(encoding="utf-8").splitlines()
    mount = lambda target: next(l for l in lines if not l.strip().startswith("#")
                                and f":{target}" in l)
    assert ":ro" not in mount("/data/sessions")
    assert ":ro" not in mount("/data/reference_laps")
    assert mount("/data/track_reference").rstrip().rstrip('"').endswith(":ro")
