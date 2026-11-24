"""Deliberate delete of an original (spec 4.4a): POST /sessions/sessions/{file}/delete
removes one recording from the sessions folder, and only when the request
names it exactly (`confirm`) and lists kept laps of it that pass the keep check
(lap_cut.check_kept) against it.

Every file here is synthetic and lives in a tmp dir; main's two folders are
monkeypatched onto it, so no test can reach /data/sessions or the real data/.
"""

import os
import sys
from urllib.parse import quote

import duckdb
import pytest
from fastapi.testclient import TestClient

from app import lap_cut, main, session_index

from synthetic_track import write_hilly_session

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


def _exact_case(path):
    """DuckDB 1.1.3 on Windows lowercases the name of a file it creates with
    duckdb.connect(path); the kept-file name is built from the original's
    (see test_lap_cut._exact_case)."""
    actual = next(p for p in path.parent.iterdir() if p.name.lower() == path.name.lower())
    if actual.name != path.name:
        os.replace(actual, path)
    return path


@pytest.fixture
def kept_one(dirs, client):
    """The hilly original with lap 1 kept (through the real keep route)."""
    src = _exact_case(write_hilly_session(dirs[0] / HILLY))
    r = client.post(f"/sessions/sessions/{quote(HILLY)}/keep", json={"laps": [1]})
    assert r.status_code == 200 and r.json()["all_checked"], r.text[:300]
    kept = dirs[1] / r.json()["results"][0]["kept_file"]
    assert kept.is_file()
    return src, kept


def _delete(client, name, confirm, kept_files):
    return client.post(f"/sessions/sessions/{quote(name)}/delete",
                       json={"confirm": confirm, "kept_files": kept_files})


def _set_meta(path, key, value):
    con = duckdb.connect(str(path))
    try:
        con.execute("UPDATE metadata SET value = ? WHERE key = ?", [value, key])
    finally:
        con.close()


def test_delete_removes_original(kept_one, dirs, client):
    src, kept = kept_one
    names = [(f["source"], f["file"]) for f in client.get("/sessions/index").json()["files"]]
    assert ("sessions", HILLY) in names
    assert str(src.resolve()) in session_index._CACHE

    r = _delete(client, HILLY, HILLY, [kept.name])
    assert r.status_code == 200, r.text[:300]
    assert r.json() == {"ok": True, "deleted": HILLY}
    assert not src.exists()
    assert kept.is_file()                                   # the kept lap stays
    assert str(src.resolve()) not in session_index._CACHE   # entry dropped at once
    files = client.get("/sessions/index").json()["files"]
    assert [(f["source"], f["file"]) for f in files] == [("reference_laps", kept.name)]
    assert HILLY not in client.get("/sessions").json()["sessions"]
    assert sorted(p.name for p in dirs[0].iterdir()) == []


def test_delete_requires_checked_kept_files(kept_one, dirs, client, monkeypatch):
    src, kept = kept_one

    def refused(kept_files, detail_part):
        r = _delete(client, HILLY, HILLY, kept_files)
        assert r.status_code == 412, (kept_files, r.status_code, r.text[:300])
        assert detail_part in r.json()["detail"], r.json()
        assert src.is_file() and kept.is_file()

    refused([], "No kept laps")
    # no kept_files key at all: the same 412, not pydantic's 422
    r = client.post(f"/sessions/sessions/{quote(HILLY)}/delete", json={"confirm": HILLY})
    assert r.status_code == 412 and "No kept laps" in r.json()["detail"]
    assert src.is_file()
    refused(["Not There L1 1m00.000-2026-10-19T23_18_37Z.duckdb"], "not in the kept-laps folder")
    refused([kept.name, "Not There.duckdb"], "not in the kept-laps folder")
    # a kept file that exists in the SESSIONS folder only does not count
    (dirs[0] / "Elsewhere.duckdb").write_bytes(b"")
    refused(["Elsewhere.duckdb"], "not in the kept-laps folder")

    # the check fails: delete must call Task 2's check (not a copy of it)
    calls = []
    real_check = lap_cut.check_kept

    def failing_check(k, lap, t, s):
        calls.append((k, lap, t, s))
        return False, t, "simulated failure"
    monkeypatch.setattr(lap_cut, "check_kept", failing_check)
    refused([kept.name], "simulated failure")
    ((k, lap, t, s),) = calls
    assert k.name == kept.name and lap == 1 and s == src.resolve()
    ok, t_kept, _ = real_check(kept, 1, t, src)
    assert ok and abs(t_kept - t) <= 0.001                  # the original's lap time was passed
    monkeypatch.setattr(lap_cut, "check_kept", real_check)

    # a real check failure: the kept file's lap time no longer matches
    con = duckdb.connect(str(kept))
    try:
        con.execute('UPDATE "Lap" SET ts = ts + 0.5 WHERE value = (SELECT max(value) FROM "Lap")')
        con.execute('UPDATE "Lap Time" SET value = value + 0.5')
    finally:
        con.close()
    refused([kept.name], "failed its check")

    # KeptFromFile names another recording
    fresh = client.post(f"/sessions/sessions/{quote(HILLY)}/keep", json={"laps": [0]}).json()
    other = dirs[1] / fresh["results"][0]["kept_file"]
    assert fresh["all_checked"] and other.is_file()
    _set_meta(other, "KeptFromFile", "Some Other-2026-10-19T00_00_00Z.duckdb")
    refused([other.name], "was kept from")

    # a .duckdb in the kept folder that is not a kept lap at all
    plain = dirs[1] / "Dropped In.duckdb"
    con = duckdb.connect(str(plain))
    try:
        con.execute("CREATE TABLE metadata(key VARCHAR, value VARCHAR)")
    finally:
        con.close()
    refused([_exact_case(plain).name], "was kept from")
    assert src.is_file()


BAD_NAMES = ["..\\x.duckdb", "C:\\x.duckdb", "x.txt", "..x.duckdb", ".x.duckdb.tmp",
             " x.duckdb", "a\\b.duckdb"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_delete_rejects_bad_names_and_confirm(kept_one, dirs, client, name):
    src, kept = kept_one
    before = sorted(p.name for p in dirs[0].parent.rglob("*"))

    # a hostile file name in the URL (confirm equal to it, so only the name is wrong)
    r = _delete(client, name, name, [kept.name])
    assert r.status_code == 400, (name, r.status_code, r.text[:200])
    # a hostile name among the kept files
    r = _delete(client, HILLY, HILLY, [kept.name, name])
    assert r.status_code == 400, (name, r.status_code, r.text[:200])
    # confirm differs by case, whitespace, or is missing / another file
    for confirm in (HILLY.lower(), HILLY.upper(), HILLY + " ", " " + HILLY, HILLY + "\n",
                    "", kept.name, HILLY.replace(".duckdb", "")):
        r = _delete(client, HILLY, confirm, [kept.name])
        assert r.status_code == 400, (repr(confirm), r.status_code, r.text[:200])
    r = client.post(f"/sessions/sessions/{quote(HILLY)}/delete", json={"kept_files": [kept.name]})
    assert r.status_code == 422                              # no confirm at all (still required)
    # a slash never reaches the handler (router 404); either way nothing goes
    r = _delete(client, "a/b.duckdb", "a/b.duckdb", [kept.name])
    assert r.status_code in (400, 404)

    assert src.is_file() and kept.is_file()
    assert sorted(p.name for p in dirs[0].parent.rglob("*")) == before


def test_delete_missing_original_is_404(dirs, client):
    r = _delete(client, "Gone-2026-10-19T23_18_37Z.duckdb",
                "Gone-2026-10-19T23_18_37Z.duckdb", ["k.duckdb"])
    assert r.status_code == 404


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locking")
def test_delete_locked_file_reports_in_use(kept_one, client):
    src, kept = kept_one
    # Probed 11/17/2026 on Windows: a plain open() handle (no FILE_SHARE_DELETE)
    # makes os.remove fail under every DuckDB version; a DuckDB connection does
    # under 1.1.3 (the Docker pin) but not under 2.0, which opens with delete
    # sharing. So the handle here is open(), which locks in both.
    with open(src, "rb"):
        r = _delete(client, HILLY, HILLY, [kept.name])
        assert r.status_code == 409, r.text[:300]
        assert r.json()["detail"] == main.DELETE_IN_USE
        assert src.is_file()
    assert src.is_file() and kept.is_file()
    # once released, the same request deletes it
    assert _delete(client, HILLY, HILLY, [kept.name]).status_code == 200
    assert not src.exists() and kept.is_file()


def test_delete_remove_error_reports_in_use(kept_one, client, monkeypatch):
    """The same 409 on any platform: in Docker (Linux) nothing locks a file, but
    a remove can still fail (permissions, a read-only mount)."""
    src, kept = kept_one
    for exc in (PermissionError(13, "Permission denied"), OSError(30, "Read-only file system")):
        def boom(path, _exc=exc):
            raise _exc
        monkeypatch.setattr(main.os, "remove", boom)
        r = _delete(client, HILLY, HILLY, [kept.name])
        assert r.status_code == 409 and r.json()["detail"] == main.DELETE_IN_USE
        assert src.is_file()


def test_delete_removes_wal_sibling_only(kept_one, dirs, client):
    src, kept = kept_one
    wal = dirs[0] / (HILLY + ".wal")
    wal.write_bytes(b"orphan")
    others = [dirs[0] / n for n in ("." + HILLY + ".tmp", "." + HILLY.replace(".duckdb", ".spool"),
                                    HILLY.replace(".duckdb", ".wal"), HILLY + ".wal.bak")]
    for p in others:
        p.write_bytes(b"x")
    r = _delete(client, HILLY, HILLY, [kept.name])
    assert r.status_code == 200, r.text[:300]
    assert not src.exists() and not wal.exists()
    assert all(p.is_file() for p in others)                  # nothing else touched
    assert kept.is_file()


def test_delete_wal_failure_is_logged_not_an_error(kept_one, dirs, client, monkeypatch, caplog):
    src, kept = kept_one
    wal = dirs[0] / (HILLY + ".wal")
    wal.write_bytes(b"orphan")
    real_remove = os.remove

    def remove(path):
        if str(path).endswith(".wal"):
            raise PermissionError(13, "Permission denied")
        real_remove(path)
    monkeypatch.setattr(main.os, "remove", remove)
    with caplog.at_level("WARNING"):
        r = _delete(client, HILLY, HILLY, [kept.name])
    assert r.status_code == 200 and r.json() == {"ok": True, "deleted": HILLY}
    assert not src.exists() and wal.is_file()
    assert any("could not remove" in m and ".wal" in m for m in caplog.messages)


def test_delete_never_touches_reference_dir(kept_one, dirs, client):
    src, kept = kept_one
    # a kept file's name as the recording to delete: it is not in the sessions
    # folder, so 404, even with itself as its own "kept file"
    r = _delete(client, kept.name, kept.name, [kept.name])
    assert r.status_code == 404, r.text[:300]
    assert kept.is_file() and src.is_file()
    # there is no delete route for the kept-laps folder
    r = client.post(f"/sessions/reference_laps/{quote(kept.name)}/delete",
                    json={"confirm": kept.name, "kept_files": [kept.name]})
    assert r.status_code in (404, 405)
    assert kept.is_file() and src.is_file()


def test_tests_never_use_the_real_folders(dirs, tmp_path):
    # every test here takes `dirs` (directly or through kept_one), which moves
    # both folders into pytest's tmp dir before the app is called
    assert main.OWN_SESSIONS_DIR == tmp_path / "sessions"
    assert main.REFERENCE_LAPS_DIR == tmp_path / "reference_laps"
