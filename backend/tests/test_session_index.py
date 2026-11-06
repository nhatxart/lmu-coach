"""/sessions/index: per-file track, car class, layout length and laps from
metadata and the lap tables only (no channel loading), cached per file
until its mtime or size changes. The Compare tab filters ghosts by these."""

import os
from urllib.parse import quote

import duckdb
import pytest
from fastapi.testclient import TestClient

from app import main, session_index

from synthetic_track import write_hilly_session


@pytest.fixture
def folders(tmp_path, monkeypatch):
    own, ref = tmp_path / "own", tmp_path / "ref"
    own.mkdir()
    ref.mkdir()
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", own)
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", ref)
    return own, ref


def _index():
    r = TestClient(main.app).get("/sessions/index")
    assert r.status_code == 200, r.text[:300]
    return r.json()["files"]


def test_index_entry_fields(folders):
    own, _ = folders
    path = write_hilly_session(own / "gt3.duckdb", vehicle_class="LMGT3")
    files = _index()
    assert len(files) == 1
    e = files[0]
    assert e["source"] == "sessions"
    assert e["file"] == "gt3.duckdb"
    assert e["track"] == "Synthetic Hill Ring"
    assert e["track_slug"] is None
    assert e["car_model"] == "Synthetic Car #1"
    assert e["car_class"] == "LMGT3"
    assert e["coordinate_system"] == "gps_projected"
    assert e["layout_len_m"] == pytest.approx(1200, abs=15)
    assert "error" not in e
    laps = TestClient(main.app).get(f"/sessions/sessions/{quote(path.name)}/laps").json()["laps"]
    keys = ("lap_number", "lap_time_s", "complete", "real")
    assert e["laps"] == [{k: l[k] for k in keys} for l in laps]
    assert len(e["laps"]) == 3


def test_index_layout_length_follows_lap_dist(folders):
    own, _ = folders
    write_hilly_session(own / "short.duckdb", lap_dist_scale=0.9)
    (e,) = _index()
    assert e["layout_len_m"] == pytest.approx(1080, abs=15)
    assert e["car_class"] == "unknown"


def test_index_lists_both_folders(folders):
    own, ref = folders
    write_hilly_session(own / "mine.duckdb")
    write_hilly_session(ref / "friend.duckdb")
    files = _index()
    assert [(f["source"], f["file"]) for f in files] == [
        ("sessions", "mine.duckdb"), ("reference_laps", "friend.duckdb")]


def test_index_cache_reuses_and_invalidates(folders, monkeypatch):
    own, _ = folders
    path = write_hilly_session(own / "cached.duckdb")
    opened = []
    real_connect = duckdb.connect

    def counting_connect(database=":memory:", *args, **kwargs):
        opened.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", counting_connect)
    first = _index()
    assert any("cached.duckdb" in p for p in opened)
    opened.clear()
    assert _index() == first
    assert opened == []
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    assert _index() == first
    assert any("cached.duckdb" in p for p in opened)


def test_index_survives_unreadable_file(folders):
    own, _ = folders
    write_hilly_session(own / "good.duckdb")
    (own / "junk.duckdb").write_bytes(os.urandom(4096))
    files = {f["file"]: f for f in _index()}
    assert set(files) == {"good.duckdb", "junk.duckdb"}
    bad = files["junk.duckdb"]
    assert isinstance(bad["error"], str) and bad["error"]
    assert bad["laps"] == []
    good = files["good.duckdb"]
    assert "error" not in good
    assert len(good["laps"]) == 3
    assert good["layout_len_m"] is not None


def test_entry_without_lap_dist_or_gps(tmp_path):
    path = write_hilly_session(tmp_path / "bare.duckdb")
    con = duckdb.connect(str(path))
    for t in ("Lap Dist", "GPS Latitude", "GPS Longitude"):
        con.execute(f'DROP TABLE "{t}"')
    con.close()
    e = session_index.file_entry(path, "sessions")
    assert e["layout_len_m"] is None
    assert e["coordinate_system"] == "world_metres"
    assert len(e["laps"]) == 3
