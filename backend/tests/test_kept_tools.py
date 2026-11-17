"""Kept laps in the session index and the rebuild tools (spec 2026-11-03
section 4.3 "Rebuild tools" / "Topography", plan Task 4).

A kept lap is the same lap under a new file name in the kept-laps folder
(/data/reference_laps). The index marks it (kept_from, kept_lap, recorded_by),
and ingest_sessions.py / build_track_reference.py read that folder too, so a
cleanout of the sessions folder does not lose what was learned, but a lap
present twice (original + kept copy) counts once. Everything is under tmp_path.
"""

import json
import sys
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from app import grip_profile as gp, lap_cut, main
from app import track_topography as tt
from synthetic_track import write_hilly_session
from synthetic_tyre import write_susp_force_session
from test_lap_cut import _exact_case

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import build_track_reference as btr  # noqa: E402
import ingest_sessions as ing  # noqa: E402

ORIG = "Synthetic Hill Ring-Synthetic Car _1-2026-10-19T23_18_37Z.duckdb"
FUJI = "Fuji Speedway-Synthetic Car _1-2026-10-25T10_00_00Z.duckdb"
LAP = 2


@pytest.fixture
def dirs(tmp_path):
    sessions, kept = tmp_path / "sessions", tmp_path / "reference_laps"
    sessions.mkdir()
    kept.mkdir()
    return sessions, kept


def _keep(original, kept_dir, lap=LAP, context=None):
    info = lap_cut.prepare(original, lap)
    dst = kept_dir / info.name
    lap_cut.cut_lap(original, lap, dst, context or {})
    return dst


# ── session index ────────────────────────────────────────────────────────────

def test_index_marks_kept_files(dirs, monkeypatch):
    sessions, kept_dir = dirs
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", sessions)
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", kept_dir)
    original = _exact_case(write_hilly_session(sessions / ORIG))
    kept = _keep(original, kept_dir)
    native = write_hilly_session(sessions / "native.duckdb")
    con = duckdb.connect(str(native))
    con.execute("DELETE FROM metadata WHERE key = 'RecordedBy'")
    con.close()

    r = TestClient(main.app).get("/sessions/index")
    assert r.status_code == 200, r.text[:300]
    files = {f["file"]: f for f in r.json()["files"]}
    o, k, n = files[original.name], files[kept.name], files["native.duckdb"]

    assert (o["kept_from"], o["kept_lap"]) == (None, None)
    assert o["recorded_by"] == "lmu-coach-recorder hf1"
    assert (k["source"], k["kept_from"], k["kept_lap"]) == ("reference_laps", original.name, LAP)
    assert k["recorded_by"] == "lmu-coach-recorder hf1"      # metadata is copied
    assert [l["lap_number"] for l in k["laps"]] == [LAP]
    assert k["laps"][0]["lap_time_s"] == next(
        l["lap_time_s"] for l in o["laps"] if l["lap_number"] == LAP)
    assert (n["kept_from"], n["kept_lap"], n["recorded_by"]) == (None, None, "")


def test_index_error_entry_has_kept_fields(tmp_path):
    from app import session_index
    junk = tmp_path / "junk.duckdb"
    junk.write_bytes(b"\0" * 4096)
    e = session_index.file_entry(junk, "sessions")
    assert e["error"] and (e["kept_from"], e["kept_lap"], e["recorded_by"]) == (None, None, "")


# ── ingest_sessions.py ───────────────────────────────────────────────────────

def test_select_files_reads_reference_dir_and_dates(dirs):
    sessions, kept_dir = dirs
    (sessions / ORIG).write_bytes(b"")
    k = kept_dir / "Synthetic Hill Ring-Synthetic Car _1 L2 1m02.500-2026-10-19T23_18_37Z.duckdb"
    k.write_bytes(b"")
    (kept_dir / "Other-Car-2026-09-01T10_00_00Z.duckdb").write_bytes(b"")
    names = lambda fs: sorted(p.name for _, p in fs)  # noqa: E731
    assert names(ing.select_files(sessions, kept_dir)) == sorted(
        [ORIG, k.name, "Other-Car-2026-09-01T10_00_00Z.duckdb"])
    # the kept name carries the original's stamp, so the date filters work on it
    assert names(ing.select_files(sessions, kept_dir, month="2026-10")) == sorted([ORIG, k.name])
    assert names(ing.select_files(sessions, kept_dir, since="2026-10-20")) == []
    assert names(ing.select_files(sessions, kept_dir, pattern="*L2*")) == [k.name]
    # "" (or None) = off: the sessions folder only
    assert names(ing.select_files(sessions, None)) == [ORIG]
    # a missing folder is not an error (the default may not exist yet)
    assert names(ing.select_files(sessions, kept_dir / "nope")) == [ORIG]


def _run_ingest(monkeypatch, sessions, kept_dir, profiles):
    monkeypatch.setattr(sys, "argv", ["x", "--sessions-dir", str(sessions),
                                      "--reference-dir", str(kept_dir),
                                      "--profiles-dir", str(profiles), "--save"])
    assert ing.main() == 0
    (prof_path,) = [p for p in profiles.glob("*.json") if not p.name.startswith(".")]
    ledger = json.loads((profiles / ".ingest_ledger.json").read_text())
    return gp.GripProfile.from_dict(json.loads(prof_path.read_text())), ledger


def test_ingest_reads_reference_dir_once(dirs, tmp_path, monkeypatch):
    sessions, kept_dir = dirs
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    monkeypatch.setattr(gp, "PROFILES_DIR", profiles)       # main() sets it; undo restores
    gp.clear_class_prior_cache()
    original = _exact_case(write_susp_force_session(sessions / ORIG))
    from app.telemetry import load_telemetry_file
    from app.tire_model import TIRE_GRID_HZ
    kept = _keep(original, kept_dir,
                 context=gp.file_tyre_context(load_telemetry_file(original, grid_hz=TIRE_GRID_HZ), None))

    # 1) original + kept copy: tyre laps keyed by the original only, baselines
    #    from the original only (the kept file is skipped, not ledgered)
    prof, ledger = _run_ingest(monkeypatch, sessions, kept_dir, profiles)
    keys = prof.tire_model_laps
    assert len(keys) == len(set(keys)) and f"{original.name}#{LAP}" in keys
    assert list(prof.tire_files) == [original.name]
    laps_once = prof.lap_count
    assert laps_once >= 1
    assert ledger == [original.name]

    # 2) the user cleans out the sessions folder (the test's own file): the
    #    kept lap is the only source now, and was already counted in both
    original.unlink()
    prof2, ledger2 = _run_ingest(monkeypatch, sessions, kept_dir, profiles)
    assert prof2.tire_model_laps == keys
    assert prof2.lap_count == laps_once
    assert ledger2 == [original.name]
    gp.clear_class_prior_cache()


def test_ingest_kept_without_original_counts(dirs, tmp_path, monkeypatch):
    sessions, kept_dir = dirs
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    monkeypatch.setattr(gp, "PROFILES_DIR", profiles)
    gp.clear_class_prior_cache()
    original = _exact_case(write_susp_force_session(tmp_path / ORIG))   # never in sessions/
    from app.telemetry import load_telemetry_file
    from app.tire_model import TIRE_GRID_HZ
    kept = _keep(original, kept_dir,
                 context=gp.file_tyre_context(load_telemetry_file(original, grid_hz=TIRE_GRID_HZ), None))
    prof, ledger = _run_ingest(monkeypatch, sessions, kept_dir, profiles)
    assert prof.tire_model_laps == [f"{original.name}#{LAP}"]
    assert prof.lap_count == 1
    assert ledger == [kept.name]
    gp.clear_class_prior_cache()


# ── topography (build_track_reference.py --reference) ────────────────────────

@pytest.fixture
def ref_out(tmp_path, monkeypatch):
    d = tmp_path / "track_reference"
    d.mkdir()
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(d))
    tt._CACHE.clear()
    return d


def _topo_cli(monkeypatch, sessions, kept_dir, ref_out):
    monkeypatch.setattr(sys, "argv", ["x", "--topography-only", "--sessions", str(sessions),
                                      "--reference", str(kept_dir), "--track", "fuji"])
    assert btr.main() == 0
    return json.loads((ref_out / "fuji.topography.json").read_text())


def test_topography_skips_kept_with_original(dirs, ref_out, monkeypatch):
    sessions, kept_dir = dirs
    original = _exact_case(write_hilly_session(sessions / FUJI, track_name="Fuji Speedway"))
    kept = _keep(original, kept_dir)
    assert tt.topography_files([original], [kept]) == [original]
    topo = _topo_cli(monkeypatch, sessions, kept_dir, ref_out)
    want = tt.build_topography([original], "fuji", "x")
    assert [l["sources"] for l in topo["layouts"]] == [l["sources"] for l in want["layouts"]]
    assert topo["layouts"][0]["sources"]["files"] == 1


def test_topography_uses_kept_without_original(dirs, ref_out, monkeypatch, tmp_path):
    sessions, kept_dir = dirs
    original = _exact_case(write_hilly_session(tmp_path / FUJI, track_name="Fuji Speedway"))
    kept = _keep(original, kept_dir)
    friend = write_hilly_session(kept_dir / "Fuji Speedway-Friend-2026-10-01T10_00_00Z.duckdb",
                                 track_name="Fuji Speedway")
    # only kept files join the input; other files in the folder are not ours
    assert tt.topography_files([], [kept, friend]) == [kept]
    topo = _topo_cli(monkeypatch, sessions, kept_dir, ref_out)
    (lay,) = topo["layouts"]
    assert (lay["sources"]["files"], lay["sources"]["laps"]) == (1, 1)


def test_topography_reference_off(dirs, ref_out, monkeypatch, tmp_path):
    sessions, kept_dir = dirs
    original = _exact_case(write_hilly_session(tmp_path / FUJI, track_name="Fuji Speedway"))
    _keep(original, kept_dir)
    monkeypatch.setattr(sys, "argv", ["x", "--topography-only", "--sessions", str(sessions),
                                      "--reference", "", "--track", "fuji"])
    assert btr.main() == 0
    assert not (ref_out / "fuji.topography.json").exists()
