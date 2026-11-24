"""Kept laps and the tyre model (spec 2026-11-03 section 4.3, plan Task 3).

A kept lap is the same lap under a new file name. It must count once (under its
original's file key and lap number) and be worth as much as inside the
recording: the file-level context the tyre model takes from the WHOLE original
(load reference, slow-lap cutoff, deflection calibration) travels with it in
metadata KeptTyreContext, because a one-lap file cannot recompute it.

Fixture: synthetic_tyre.write_susp_force_session, a GT3-style recording (Susp
Force, no Tire Load) whose load falls with time, so a lap's own reference load
differs from the whole file's. Everything is written under tmp_path.
"""

import copy
import json

import duckdb
import pytest
from fastapi.testclient import TestClient

from app import grip_profile as gp, lap_cut, main
from app.telemetry import load_telemetry_file
from app.tire_model import TIRE_GRID_HZ
from synthetic_tyre import write_susp_force_session
from test_lap_cut import _exact_case
from test_tire_model import _lap

ORIG = "Synthetic Hill Ring-Synthetic Car _1-2026-10-19T23_18_37Z.duckdb"
LAP = 2


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    sessions, kept, profiles = (tmp_path / "sessions", tmp_path / "reference_laps",
                                tmp_path / "profiles")
    sessions.mkdir()
    kept.mkdir()
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", sessions)
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", kept)
    monkeypatch.setattr(gp, "PROFILES_DIR", profiles)
    gp.clear_class_prior_cache()
    yield sessions, kept
    gp.clear_class_prior_cache()


@pytest.fixture
def original(dirs):
    return _exact_case(write_susp_force_session(dirs[0] / ORIG))


def _tyre_laps(path):
    return load_telemetry_file(path, grid_hz=TIRE_GRID_HZ)


def _meta(path):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return dict(con.execute("SELECT key, value FROM metadata").fetchall())
    finally:
        con.close()


def _keep(original, kept_dir, lap=LAP, context=None):
    """Cut `lap` the way the keep route does (context from the original)."""
    if context is None:
        context = gp.file_tyre_context(_tyre_laps(original), None)
    info = lap_cut.prepare(original, lap)
    dst = kept_dir / info.name
    lap_cut.cut_lap(original, lap, dst, context)
    return dst


def _ingest_file(profile, path):
    laps = _tyre_laps(path)
    key, context = gp.tire_file_key(path.name, laps[0].meta)
    return profile.ingest_tire_laps(laps, key, context)


def _fingerprint(profile):
    return profile.refit_tyre_model()["fingerprint"]


def test_kept_lap_ingests_like_original(original, dirs):
    full = gp.GripProfile(car_name="Test GT3")
    assert LAP in full.ingest_tire_laps(_tyre_laps(original), original.name)
    want = full.tire_files[original.name]["laps"][str(LAP)]

    kept = _keep(original, dirs[1])
    meta = _meta(kept)
    key, context = gp.tire_file_key(kept.name, meta)
    assert key == original.name
    assert context == json.loads(meta["KeptTyreContext"])
    assert set(context) == {"calibration", "load_ref", "fastest_s"}
    assert context["calibration"] and set(context["load_ref"]) == {"front", "rear"}

    fresh = gp.GripProfile(car_name="Test GT3")
    assert _ingest_file(fresh, kept) == [LAP]
    assert list(fresh.tire_files) == [original.name]
    entry = fresh.tire_files[original.name]
    assert entry["laps"] == {str(LAP): want}
    assert (entry["game_version"], entry["track"]) == (
        full.tire_files[original.name]["game_version"], full.tire_files[original.name]["track"])
    # no calibration refresh from a kept file (it is the original's, in the context)
    assert fresh.deflection_calibration == {}

    # guard: without the original's context the same lap bins differently, so
    # the equality above is the context's doing
    bare = gp.GripProfile(car_name="Test GT3")
    bare.ingest_tire_laps(_tyre_laps(kept), original.name)
    assert bare.tire_files[original.name]["laps"][str(LAP)] != want


def test_kept_after_original_adds_nothing(original, dirs):
    kept = _keep(original, dirs[1])
    prof = gp.GripProfile(car_name="Test GT3")
    assert _ingest_file(prof, original)
    files, fp = copy.deepcopy(prof.tire_files), _fingerprint(prof)
    assert _ingest_file(prof, kept) == []
    assert prof.tire_files == files
    assert _fingerprint(prof) == fp


def test_kept_before_original_counts_lap_once(original, dirs):
    kept = _keep(original, dirs[1])
    prof = gp.GripProfile(car_name="Test GT3")
    assert _ingest_file(prof, kept) == [LAP]
    added = _ingest_file(prof, original)
    assert LAP not in added and added
    assert list(prof.tire_files) == [original.name]
    keys = prof.tire_model_laps
    assert len(keys) == len(set(keys)) and f"{original.name}#{LAP}" in keys


def test_original_deleted_kept_still_counts(original, dirs):
    kept = _keep(original, dirs[1])
    want = gp.GripProfile(car_name="Test GT3")
    want.ingest_tire_laps(_tyre_laps(original), original.name)
    original.unlink()                    # the test's own file: the user cleaned out
    prof = gp.GripProfile(car_name="Test GT3")
    assert _ingest_file(prof, kept) == [LAP]
    assert prof.tire_model_laps == [f"{original.name}#{LAP}"]
    assert prof.tire_files[original.name]["laps"][str(LAP)] == \
        want.tire_files[original.name]["laps"][str(LAP)]


def test_context_cutoff_uses_the_originals_fastest_lap():
    # a lap 20 % slower than the original's fastest is not folded in, even
    # when it is the only lap of the (kept) file
    lap = _lap()
    ctx = {"calibration": None, "load_ref": {"front": 4000.0, "rear": 4000.0},
           "fastest_s": lap.lap_time_s / 1.2}
    assert gp.GripProfile(car_name="x").ingest_tire_laps([lap], "orig", ctx) == []
    ctx["fastest_s"] = lap.lap_time_s
    assert gp.GripProfile(car_name="x").ingest_tire_laps([lap], "orig", ctx) == [1]


def test_tire_file_key_of_a_recording_and_a_bad_context():
    assert gp.tire_file_key("a.duckdb", {"TrackName": "T"}) == ("a.duckdb", {})
    assert gp.tire_file_key("k.duckdb", {"KeptFromFile": "a.duckdb", "KeptLap": "3",
                                         "KeptTyreContext": "not json"}) == ("a.duckdb", {})
    assert gp.tire_file_key("k.duckdb", {"KeptFromFile": "a.duckdb", "KeptLap": "3",
                                         "KeptTyreContext": '{"fastest_s": 90.5}'}) \
        == ("a.duckdb", {"fastest_s": 90.5})


def test_file_tyre_context_is_json_safe_and_falls_back_to_the_profile():
    with_force, no_force = _lap(), _lap(susp_force=False)
    ctx = gp.file_tyre_context([with_force], None)
    assert json.loads(json.dumps(ctx, allow_nan=False)) == ctx
    assert ctx["calibration"]["front"]["r"] > 0.99
    prof_cal = {"front": {"slope": 1.0, "intercept": 0.0, "r": 0.9, "n": 300}}
    assert gp.file_tyre_context([no_force], prof_cal)["calibration"] == prof_cal
    assert gp.file_tyre_context([no_force], None)["calibration"] is None


def test_keep_with_failing_tyre_context_warns(original, dirs, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("context exploded")

    async def fake_llm(report, baselines, **kw):
        return "coach text"

    monkeypatch.setattr(main, "file_tyre_context", boom)
    monkeypatch.setattr(main, "get_coaching_analysis", fake_llm)
    client = TestClient(main.app)
    r = client.post(f"/sessions/sessions/{original.name}/keep", json={"laps": [LAP]})
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["status"] == "kept" and r.json()["all_checked"]
    assert res["warning"] == ("tyre context unavailable: this lap counts in the tyre "
                              "model with a one-lap load reference")
    kept = dirs[1] / res["kept_file"]
    assert _meta(kept)["KeptTyreContext"] == "{}"
    # still keyed under its original, so it counts once
    r = client.post("/analyze", json={"file": kept.name, "source": "reference", "lap_number": LAP})
    assert r.status_code == 200, r.text
    assert gp.load_profile("Test GT3").tire_model_laps == [f"{original.name}#{LAP}"]


def test_analyze_kept_lap_dedupes(original, dirs, monkeypatch):
    async def fake_llm(report, baselines, **kw):
        return "coach text"

    monkeypatch.setattr(main, "get_coaching_analysis", fake_llm)
    client = TestClient(main.app)

    # the keep route stores the original's context in the kept file
    r = client.post(f"/sessions/sessions/{original.name}/keep", json={"laps": [LAP]})
    assert r.status_code == 200 and r.json()["all_checked"], r.text
    kept = dirs[1] / r.json()["results"][0]["kept_file"]
    stored = json.loads(_meta(kept)["KeptTyreContext"])
    assert stored == json.loads(json.dumps(gp.file_tyre_context(_tyre_laps(original), None)))

    def analyze(name, source, lap):
        r = client.post("/analyze", json={"file": name, "source": source, "lap_number": lap})
        assert r.status_code == 200, r.text
        return gp.load_profile("Test GT3")

    p = analyze(kept.name, "reference", LAP)
    assert p.tire_model_laps == [f"{original.name}#{LAP}"]
    p = analyze(original.name, "sessions", 1)
    files, laps = copy.deepcopy(p.tire_files), p.tire_model_laps
    assert list(files) == [original.name] and len(laps) == len(set(laps)) > 1
    fp = p.tire_fit_cache["fingerprint"]
    p = analyze(kept.name, "reference", LAP)
    assert p.tire_files == files and p.tire_fit_cache["fingerprint"] == fp
    # the ABS/TC/slip baselines have no per-lap key and grow per analyze (spec 4.3)
    assert p.lap_count == 3
