"""/trackmap: elevation, attitude and the 3D surface ribbon (chase view)."""

import math
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, track_topography, trackmap_elevation

from synthetic_track import RING_LENGTH_M, truth, write_hilly_session

SLUG = "synthetic_hill_ring"
FLAT_NOTE = "Flat: no elevation recorded for this track yet"


@pytest.fixture(autouse=True)
def reference_dir(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    ref.mkdir()      # (sessions dir below is made by sess())
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(ref))
    monkeypatch.setattr(track_topography, "slug_for_track",
                        lambda name: SLUG if name == "Synthetic Hill Ring" else None)
    track_topography._CACHE.clear()
    trackmap_elevation._LAP_CACHE.clear()
    return ref


def sess(tmp_path):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    return d


def _trackmap(path, monkeypatch, lap=1):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", path.parent)
    r = TestClient(main.app).get(
        f"/sessions/sessions/{quote(path.name)}/trackmap", params={"lap_number": lap})
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _build_and_save(*paths):
    topo = track_topography.build_topography(list(paths), SLUG, "Synthetic Hill Ring")
    track_topography.save_topography(topo)
    return topo


def _check_lengths(d):
    n = len(d["driven_line"]["x"])
    for k in ("y", "pitch_deg", "roll_deg"):
        assert len(d["driven_line"][k]) == n, k
    assert len(d["track"]["left_y"]) == len(d["track"]["left_x"])
    assert len(d["track"]["right_y"]) == len(d["track"]["right_x"])
    s = d["surface"]
    assert s["cols"] == 11
    assert s["rows"] == len(d["track"]["left_x"])
    for k in ("x", "y", "z", "kind"):
        assert len(s[k]) == s["rows"] * s["cols"], k
    for c in d["corner_labels"]:
        assert "y" in c


def test_trackmap_lap_source(tmp_path, monkeypatch):
    f = write_hilly_session(sess(tmp_path) / "Hill-lap.duckdb")
    d = _trackmap(f, monkeypatch)
    el = d["elevation"]
    assert el["source"] == "lap" and el["orientation"] == "recorded"
    assert el["note"] == "Elevation from this session only"
    _check_lengths(d)
    y = np.array(d["driven_line"]["y"])
    assert y.max() - y.min() == pytest.approx(16.0, abs=0.5)
    grade_deg = math.degrees(math.atan(8 * 2 * math.pi / RING_LENGTH_M))
    pitch = np.array(d["driven_line"]["pitch_deg"])
    assert abs(pitch.max() - grade_deg) < 3 and abs(pitch.min() + grade_deg) < 3
    assert np.array(d["track"]["left_y"]).std() > 1.0


def test_trackmap_recorded_source(tmp_path, monkeypatch):
    f = write_hilly_session(sess(tmp_path) / "Hill-rec.duckdb")
    _build_and_save(f)
    d = _trackmap(f, monkeypatch)
    el = d["elevation"]
    assert el["source"] == "recorded" and el["orientation"] == "recorded"
    assert el["note"] == ""
    assert el["topography"]["files"] == 1
    _check_lengths(d)


def test_trackmap_topography_source_for_legacy(tmp_path, monkeypatch):
    rich = write_hilly_session(sess(tmp_path) / "Hill-rich.duckdb")
    topo = _build_and_save(rich)
    legacy = write_hilly_session(sess(tmp_path) / "Hill-legacy.duckdb",
                                 with_pose=False, lap_offsets=(0.5, 0.5, 0.5))
    d = _trackmap(legacy, monkeypatch)
    el = d["elevation"]
    assert el["source"] == "topography" and el["orientation"] == "surface"
    _check_lengths(d)
    assert el["datum_m"] == pytest.approx(topo["datum_m"], abs=1e-6)
    # lateral offset is 0.5 m and bank is 0 outside the second semicircle, so
    # compare the whole line against the truth with that offset (tolerant of
    # the bank region by using the true lat there)
    y = np.array(d["driven_line"]["y"])
    n = len(y)
    # recover s from the lap geometry: the line is evenly timed, lap 1 spans
    # s = 0..L; evenly spaced samples map to s linearly to within a few m
    s = np.linspace(0, RING_LENGTH_M, n)
    rng = np.random.default_rng(1)
    pick = rng.choice(np.arange(5, n - 5), 20, replace=False)
    assert y.max() - y.min() == pytest.approx(16.0, abs=0.6)
    err = np.abs(y[pick] - (truth(s[pick], 0.5) - el["datum_m"]))
    assert np.median(err) < 0.1


def test_trackmap_legacy_file_flat(tmp_path, monkeypatch):
    legacy = write_hilly_session(sess(tmp_path) / "Hill-legacy.duckdb", with_pose=False)
    d = _trackmap(legacy, monkeypatch)
    el = d["elevation"]
    assert el["source"] == "none" and el["orientation"] == "none"
    assert el["note"].startswith(FLAT_NOTE)
    _check_lengths(d)
    assert set(d["driven_line"]["y"]) == {0.0}
    assert set(d["driven_line"]["pitch_deg"]) == {0.0}
    assert set(d["track"]["left_y"]) == {0.0}
    assert set(d["surface"]["y"]) == {0.0}
    # existing keys unchanged
    for k in ("coordinate_system", "lap_number", "has_edges", "lap_len_pts", "susp",
              "slip", "slip_angle", "handling", "car_pos_x", "car_pos_z"):
        assert k in d
    for k in ("x", "z", "throttle", "brake", "heading", "speed"):
        assert k in d["driven_line"]


def test_surface_kind_has_kerb(tmp_path, monkeypatch):
    f = write_hilly_session(sess(tmp_path) / "Hill-kerb.duckdb")
    _build_and_save(f)
    d = _trackmap(f, monkeypatch, lap=1)
    s = d["surface"]
    kind = np.array(s["kind"]).reshape(s["rows"], s["cols"])
    assert (kind == 5).any()
    # kerbs are on the right of travel only
    assert (kind[:, : s["cols"] // 2] == 5).sum() == 0
    assert (kind[:, -3:] == 5).any()
