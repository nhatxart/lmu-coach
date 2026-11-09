"""/trackmap for lap comparison: one coordinate frame per track (a fixed
origin from data/track_reference/sources/<slug>.json) and the game's lap
distance and time per driven-line point."""

import json
import math
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, telemetry, track_reference, trackmap_elevation

from synthetic_track import RING_LENGTH_M, write_hilly_session

ORIGIN = (35.37, 138.92)
R = 111_320.0


@pytest.fixture(autouse=True)
def reference_dir(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    (ref / "sources").mkdir(parents=True)
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(ref))
    track_reference._ORIGIN_CACHE.clear()
    trackmap_elevation._LAP_CACHE.clear()
    return ref


def _write_origin(ref: Path, slug="synthetic_hill_ring", center=ORIGIN):
    (ref / "sources" / f"{slug}.json").write_text(json.dumps({"center": list(center)}))


def sess(tmp_path):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    return d


def _trackmap(path, monkeypatch, lap=1):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", path.parent)
    params = {"lap_number": lap} if lap is not None else {}
    r = TestClient(main.app).get(f"/sessions/sessions/{quote(path.name)}/trackmap", params=params)
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _gps(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        lat = np.array([r[0] for r in con.execute('SELECT value FROM "GPS Latitude" ORDER BY rowid').fetchall()])
        lon = np.array([r[0] for r in con.execute('SELECT value FROM "GPS Longitude" ORDER BY rowid').fetchall()])
        laps = con.execute('SELECT ts, value FROM "Lap" ORDER BY ts').fetchall()
        _, t_slow = telemetry.session_time_axes(con)
    finally:
        con.close()
    return lat, lon, laps, t_slow


def _first_lap_row(path, lap):
    """(lat, lon) of the lap's first position row, as /trackmap slices it."""
    lat, lon, laps, t_slow = _gps(path)
    ts = next(t for t, v in laps if int(v) == lap)
    i = int(np.searchsorted(t_slow, ts, side="left"))
    return lat[i], lon[i]


def _project(lat, lon, c_lat, c_lon):
    return ((lon - c_lon) * R * math.cos(math.radians(c_lat)), -(lat - c_lat) * R)


def test_track_origin_reads_sources_center(reference_dir):
    _write_origin(reference_dir)
    assert track_reference.track_origin("Synthetic Hill Ring") == ORIGIN
    _write_origin(reference_dir, "fuji", (35.371, 138.931))
    assert track_reference.track_origin("Fuji Speedway") == (35.371, 138.931)
    assert track_reference.track_origin("Nowhere Raceway") is None
    (reference_dir / "sources" / "broken_ring.json").write_text("{not json")
    assert track_reference.track_origin("Broken Ring") is None
    (reference_dir / "sources" / "no_centre.json").write_text('{"radius_m": 10}')
    assert track_reference.track_origin("No Centre") is None


def test_track_origin_sees_a_sources_file_written_later(reference_dir):
    # BUILD_TRACK_REFERENCE may run while the backend is up: a first miss
    # must not stick (final review 3), nor an old centre.
    assert track_reference.track_origin("Synthetic Hill Ring") is None
    _write_origin(reference_dir)
    assert track_reference.track_origin("Synthetic Hill Ring") == ORIGIN
    p = reference_dir / "sources" / "synthetic_hill_ring.json"
    st = p.stat()
    _write_origin(reference_dir, center=(35.5, 139.0))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert track_reference.track_origin("Synthetic Hill Ring") == (35.5, 139.0)


def test_trackmap_uses_track_origin(tmp_path, monkeypatch, reference_dir):
    _write_origin(reference_dir)
    f = write_hilly_session(sess(tmp_path) / "Hill-origin.duckdb")
    d = _trackmap(f, monkeypatch)
    assert d["coordinate_origin"] == {"kind": "track", "lat": ORIGIN[0], "lon": ORIGIN[1]}
    x, z = _project(*_first_lap_row(f, 1), *ORIGIN)
    assert d["driven_line"]["x"][0] == pytest.approx(x, abs=0.05)
    assert d["driven_line"]["z"][0] == pytest.approx(z, abs=0.05)


def _max_abs_xz(d):
    dl, s = d["driven_line"], d["surface"]
    return max(max(abs(v) for v in dl["x"]), max(abs(v) for v in dl["z"]),
               max(abs(v) for v in s["x"]), max(abs(v) for v in s["z"]))


def test_game_gps_frame_beats_real_world_centre(tmp_path, monkeypatch, reference_dir):
    """BUG (11/8/2026, user screenshot: Road Atlanta 10/26 #92, chase view
    jagged and jittering): LMU's GPS puts EVERY track near Fuji
    (35.37, 138.92), but the per-track origin was the track's real-world
    centre (sources/road_atlanta.json: 34.15, -83.81), so /trackmap sent
    x ~ 20,500 km. three.js stores positions as float32 (~2 m steps there).
    A recording whose GPS lies near the game's reference must be projected
    around that reference: small coordinates, and still one frame for
    every file of the track."""
    _write_origin(reference_dir, "road_atlanta", (34.1494, -83.8139))
    f = write_hilly_session(sess(tmp_path) / "Road-atl.duckdb",
                            track_name="Michelin Raceway Road Atlanta")
    d = _trackmap(f, monkeypatch)
    assert d["coordinate_origin"]["kind"] == "track"
    assert (d["coordinate_origin"]["lat"], d["coordinate_origin"]["lon"]) == track_reference.LMU_GPS_REF
    assert _max_abs_xz(d) < 5000          # metres; the ring is ~400 m across


def test_far_real_world_centre_is_not_used(tmp_path, monkeypatch, reference_dir):
    """A centre more than GPS_FRAME_RADIUS_M from the file's own GPS is in
    another frame: never project around it (the file's own centre instead)."""
    _write_origin(reference_dir, "road_atlanta", (34.1494, -83.8139))
    f = write_hilly_session(sess(tmp_path) / "Road-far.duckdb",
                            track_name="Michelin Raceway Road Atlanta", gps_ref=(10.0, 20.0))
    d = _trackmap(f, monkeypatch)
    assert d["coordinate_origin"]["kind"] == "file"
    assert _max_abs_xz(d) < 5000


def test_geo_referenced_file_uses_its_real_world_centre(tmp_path, monkeypatch, reference_dir):
    """Should a file's GPS be real-world (near the track's sources centre and
    far from LMU's reference), that centre is the shared origin."""
    _write_origin(reference_dir, "road_atlanta", (34.1494, -83.8139))
    f = write_hilly_session(sess(tmp_path) / "Road-geo.duckdb",
                            track_name="Michelin Raceway Road Atlanta", gps_ref=(34.15, -83.81))
    d = _trackmap(f, monkeypatch)
    assert d["coordinate_origin"] == {"kind": "track", "lat": 34.1494, "lon": -83.8139}
    assert _max_abs_xz(d) < 5000


def test_trackmap_without_origin_unchanged(tmp_path, monkeypatch):
    # GPS far from LMU's reference and no sources centre: the file's own
    # bbox centre, exactly as before the per-track origin.
    f = write_hilly_session(sess(tmp_path) / "Hill-file.duckdb", gps_ref=(10.0, 20.0))
    d = _trackmap(f, monkeypatch)
    lat, lon, _, _ = _gps(f)
    c_lat, c_lon = (lat.min() + lat.max()) / 2, (lon.min() + lon.max()) / 2
    assert d["coordinate_origin"]["kind"] == "file"
    assert d["coordinate_origin"]["lat"] == pytest.approx(c_lat, abs=1e-12)
    assert d["coordinate_origin"]["lon"] == pytest.approx(c_lon, abs=1e-12)
    x, z = _project(*_first_lap_row(f, 1), c_lat, c_lon)
    assert abs(d["driven_line"]["x"][0] - x) < 1e-9
    assert abs(d["driven_line"]["z"][0] - z) < 1e-9


def test_trackmap_without_gps_is_world(tmp_path, monkeypatch):
    f = write_hilly_session(sess(tmp_path) / "Hill-world.duckdb")
    con = duckdb.connect(str(f))
    con.execute('DROP TABLE "GPS Latitude"')
    con.execute('DROP TABLE "GPS Longitude"')
    con.close()
    d = _trackmap(f, monkeypatch)
    assert d["coordinate_system"] == "world_metres"
    assert d["coordinate_origin"] == {"kind": "world", "lat": None, "lon": None}


def test_two_files_share_the_frame(tmp_path, monkeypatch, reference_dir):
    # CONTROLLER RULING: the synthetic ring is symmetric, so two of its files
    # have the same GPS bbox centre whatever their laps are; the guard that
    # the frames really differ without the track origin lives on real data
    # (test_trackmap_real_file.test_real_fuji_files_share_the_frame).
    _write_origin(reference_dir)
    a = write_hilly_session(sess(tmp_path) / "Hill-a.duckdb", lap_offsets=(0.0,))
    b = write_hilly_session(sess(tmp_path) / "Hill-b.duckdb", lap_offsets=(3.0,),
                            trailing_partial_m=300)
    pts = []
    for f in (a, b):
        d = _trackmap(f, monkeypatch, lap=0)
        assert d["coordinate_origin"]["kind"] == "track"
        dl = d["driven_line"]
        i = int(np.argmin(np.abs(np.array(dl["lap_dist"]) - 600.0)))
        pts.append((dl["x"][i], dl["z"][i]))
    gap = math.dist(*pts)
    assert 2.5 < gap < 3.5, gap      # the 3 m lateral offset, nothing more


def test_driven_line_lap_dist_and_time(tmp_path, monkeypatch):
    f = write_hilly_session(sess(tmp_path) / "Hill-dist.duckdb")
    d = _trackmap(f, monkeypatch)
    dl = d["driven_line"]
    n = len(dl["x"])
    assert len(dl["lap_dist"]) == n and len(dl["t"]) == n
    ld = np.array(dl["lap_dist"])
    assert ld[0] == pytest.approx(0.0, abs=2.0)
    body = ld[:-1]                                     # the closing row may wrap to 0
    assert np.all(np.diff(body) >= -0.01)
    assert body.max() == pytest.approx(RING_LENGTH_M, abs=15.0)
    _, _, laps, _ = _gps(f)
    lap_time = next(laps[i + 1][0] - laps[i][0] for i in range(len(laps) - 1) if int(laps[i][1]) == 1)
    assert dl["t"][0] == 0.0
    assert dl["t"][-1] == pytest.approx(lap_time, abs=0.05)
    assert d["lap_len_m"] == pytest.approx(RING_LENGTH_M, abs=15.0)


def test_driven_line_time_from_line_crossing(tmp_path, monkeypatch):
    # The line falls between two 10 Hz rows (0.04 s before one): t counts
    # from the crossing (the Lap row), not from the lap's first position
    # row. PREVIOUSLY t[0] was 0 there, a bias of up to 0.1 s per lap
    # (final review 2).
    f = write_hilly_session(sess(tmp_path) / "Hill-cross.duckdb", cross_offset_ticks=-4)
    d = _trackmap(f, monkeypatch)
    t = d["driven_line"]["t"]
    _, _, laps, t_slow = _gps(f)
    ts0 = next(ts for ts, v in laps if int(v) == 1)
    ts1 = next(ts for ts, v in laps if ts > ts0)
    s = int(np.searchsorted(t_slow, ts0, side="left"))
    e = int(np.searchsorted(t_slow, ts1, side="left"))
    assert t[0] == pytest.approx(t_slow[s] - ts0, abs=1e-3)
    assert t[0] >= 0.03
    assert t[-1] == pytest.approx(t_slow[e] - ts0, abs=0.02)


def _closing_line_time(dl, lap_len):
    """Seconds from the crossing at which the line reaches lap distance
    lap_len (the closing row may read 0: the wrap, unwrapped here)."""
    ld = np.array(dl["lap_dist"], dtype=float)
    k = int(np.argmax(np.diff(ld) < -100)) + 1 if np.any(np.diff(ld) < -100) else len(ld)
    ld[k:] += lap_len
    return float(np.interp(lap_len, ld, dl["t"]))


def test_delta_at_closing_line_matches_lap_times(tmp_path, monkeypatch):
    a = write_hilly_session(sess(tmp_path) / "Hill-ca.duckdb", cross_offset_ticks=-4)
    b = write_hilly_session(sess(tmp_path) / "Hill-cb.duckdb", speed_scale=0.97)
    out = []
    for f in (a, b):
        d = _trackmap(f, monkeypatch)
        _, _, laps, _ = _gps(f)
        ts0 = next(ts for ts, v in laps if int(v) == 1)
        ts1 = next(ts for ts, v in laps if ts > ts0)
        out.append((_closing_line_time(d["driven_line"], RING_LENGTH_M), ts1 - ts0))
    (ta, la), (tb, lb) = out
    assert (ta - tb) == pytest.approx(la - lb, abs=0.02)


def _drop_ticks(path: Path, start_tick: int, n_ticks: int) -> int:
    """Delete n_ticks consecutive 100 Hz rows from start_tick and the 10 Hz
    rows the recorder's gate wrote in that time (as a run of skipped game
    ticks would leave the file). Returns how many 10 Hz rows went.

    The 10 Hz rows kept are the ones the gate replay (telemetry) finds on the
    shortened clock, each matched to the original row nearest in time: the
    gate's due times tie with the 10 ms ticks, so after the re-anchor at the
    hole it can pick a neighbouring tick (0.01 s, ~0.4 m here) or drop the
    closing row; matching keeps the file's 10 Hz rows on the replayed gate."""
    con = duckdb.connect(str(path))
    try:
        t_fast_old = np.array([r[0] for r in con.execute('SELECT value FROM "GPS Time" ORDER BY rowid').fetchall()])
        old_gate = telemetry._replay_slow_gate(t_fast_old)
        old_gate_t = t_fast_old[old_gate]
        keep_fast = np.ones(len(t_fast_old), dtype=bool)
        keep_fast[start_tick:start_tick + n_ticks] = False
        t_new = t_fast_old[keep_fast]
        new_gate_t = t_new[telemetry._replay_slow_gate(t_new)]
        keep_slow = [int(np.argmin(np.abs(old_gate_t - nt))) for nt in new_gate_t]
        assert len(set(keep_slow)) == len(keep_slow)
        assert np.max(np.abs(old_gate_t[keep_slow] - new_gate_t)) <= 0.011
        t0, t1 = t_fast_old[start_tick], t_fast_old[start_tick + n_ticks]
        in_hole = sum(1 for tv in old_gate_t.tolist() if t0 <= tv < t1)
        fast_keep = np.nonzero(keep_fast)[0].tolist()
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        for name in tables:
            if name in ("metadata", "channelsList", "eventsList"):
                continue
            cols = [r[0] for r in con.execute(f'DESCRIBE "{name}"').fetchall()]
            if "ts" in cols:
                continue                                 # event tables keep their times
            cnt = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            keep = fast_keep if cnt == len(t_fast_old) else keep_slow if cnt == len(old_gate) else None
            if keep is None:
                continue
            con.execute("CREATE OR REPLACE TEMP TABLE __keep (r BIGINT)")
            con.execute("INSERT INTO __keep SELECT unnest(?::BIGINT[])", [keep])
            con.execute(f'CREATE TABLE "__tmp" AS SELECT * FROM "{name}" '
                        f'WHERE rowid IN (SELECT r FROM __keep) ORDER BY rowid')
            con.execute(f'DROP TABLE "{name}"')
            con.execute(f'ALTER TABLE "__tmp" RENAME TO "{name}"')
        return in_hole
    finally:
        con.close()


def test_driven_line_time_follows_game_clock(tmp_path, monkeypatch):
    src = write_hilly_session(sess(tmp_path) / "Hill-src.duckdb")
    f = sess(tmp_path) / "Hill-gap.duckdb"
    shutil.copy(src, f)
    con = duckdb.connect(str(f), read_only=True)
    t_fast = np.array([r[0] for r in con.execute('SELECT value FROM "GPS Time" ORDER BY rowid').fetchall()])
    lap1 = next(ts for ts, v in con.execute('SELECT ts, value FROM "Lap" ORDER BY ts').fetchall() if int(v) == 1)
    con.close()
    gate = telemetry._replay_slow_gate(t_fast)
    start = int(gate[np.searchsorted(t_fast[gate], lap1 + 10.0)])  # 10 s into lap 1, on a 10 Hz row
    assert _drop_ticks(f, start, 200) == 20
    con = duckdb.connect(str(f), read_only=True)
    t_fast2, t_slow2 = telemetry.session_time_axes(con)
    n_slow2 = con.execute('SELECT count(*) FROM "GPS Latitude"').fetchone()[0]
    con.close()
    assert len(t_fast2) == len(t_fast) - 200
    assert len(telemetry._replay_slow_gate(t_fast2)) == n_slow2   # gate replay, not index spacing
    d = _trackmap(f, monkeypatch)
    dt = np.diff(d["driven_line"]["t"])
    assert dt.max() == pytest.approx(2.1, abs=0.06)    # the 2 s hole plus one 10 Hz step
    assert np.median(dt) == pytest.approx(0.1, abs=0.01)
    j = int(np.argmax(dt))
    assert d["driven_line"]["t"][j] == pytest.approx(10.0 - 0.1, abs=0.06)
