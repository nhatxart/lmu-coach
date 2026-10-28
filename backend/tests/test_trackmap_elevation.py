"""/trackmap: elevation, attitude and the 3D surface ribbon (chase view)."""

import math
from pathlib import Path
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


def test_lap_source_ignores_out_lap_on_pit_road(tmp_path, monkeypatch):
    """The "lap" topography (built from this file alone) leaves the out-lap
    out too: the surface on the pit-exit stretch is flat across, as the track
    is, not banked by the 2.5 m lower pit road (see test_topography_build)."""
    f = write_hilly_session(sess(tmp_path) / "Hill-out.duckdb",
                            lap_offsets=(0.0, 3.0, 4.5, 3.5), out_lap=True)
    d = _trackmap(f, monkeypatch, lap=1)
    assert d["elevation"]["source"] == "lap"
    sf = d["surface"]
    y = np.array(sf["y"]).reshape(sf["rows"], sf["cols"])
    s = np.array(sf["s"])
    te = 6.0                                  # synthetic Track Edge half-width
    bank = (y[:, -1] - y[:, 0]) / (2 * te)
    m = (s > 40.0) & (s < 260.0)
    assert m.sum() > 5
    assert np.max(np.abs(bank[m])) < 0.01, np.max(np.abs(bank[m]))
    # the out-lap itself still replays, on its own recorded (pit road) height
    d0 = _trackmap(f, monkeypatch, lap=0)
    assert d0["elevation"]["source"] == "lap"


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


def test_failing_elevation_step_is_fail_closed(tmp_path, monkeypatch):
    """A bug or a bad topography file must not turn a working /trackmap into a 500."""
    f = write_hilly_session(sess(tmp_path) / "Hill-boom.duckdb")

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(track_topography, "load_topography", boom)
    monkeypatch.setattr(track_topography, "build_topography", boom)
    d = _trackmap(f, monkeypatch)
    assert d["elevation"]["source"] == "none" and d["elevation"]["orientation"] == "none"
    _check_lengths(d)
    assert set(d["driven_line"]["y"]) == {0.0} and set(d["surface"]["y"]) == {0.0}
    assert len(d["corner_labels"]) == 0 or all(c["y"] == 0.0 for c in d["corner_labels"])


def test_failing_elevation_inside_payload_is_fail_closed(tmp_path, monkeypatch):
    f = write_hilly_session(sess(tmp_path) / "Hill-boom2.duckdb")

    def boom(*a, **k):
        raise ValueError("bad")
    monkeypatch.setattr(trackmap_elevation, "elevation_payload", boom)
    d = _trackmap(f, monkeypatch)
    assert d["elevation"]["source"] == "none"
    _check_lengths(d)


def test_corrupt_topography_file_is_fail_closed(tmp_path, monkeypatch, reference_dir):
    f = write_hilly_session(sess(tmp_path) / "Hill-corrupt.duckdb")
    (reference_dir / f"{SLUG}.topography.json").write_text('{"schema":1,"layouts":[{"len', encoding="utf-8")
    d = _trackmap(f, monkeypatch)
    assert d["elevation"]["source"] == "none"
    _check_lengths(d)


def test_later_laps_do_not_drift_when_lap_dist_ends_short(tmp_path, monkeypatch):
    """The last 10 Hz Lap Dist row before each wrap sits up to 4 m short of the
    line. The first version unwrapped the whole session by adding that short
    value at every wrap, so lap 3's lap distance was ~7 m off."""
    f = write_hilly_session(sess(tmp_path) / "Hill-short.duckdb", cross_offset_ticks=9)
    errs = {}
    for lap in (0, 2):
        d = _trackmap(f, monkeypatch, lap=lap)
        assert d["elevation"]["source"] == "lap"
        s = d["surface"]
        mid = np.array(s["y"]).reshape(s["rows"], s["cols"])[:, 5]     # lat 0 column
        s_est = np.linspace(0.0, RING_LENGTH_M, s["rows"], endpoint=False)
        errs[lap] = float(np.median(np.abs(mid - (truth(s_est, 0.0) - d["elevation"]["datum_m"]))))
        # kerb paint sits where the kerb is, not 7 m away
        # (the layout is shared by all laps, so every lap shows lap 1's kerb)
        kerb_s = s_est[(np.array(s["kind"]).reshape(s["rows"], s["cols"]) == 5).any(axis=1)]
        assert len(kerb_s) and kerb_s.min() > 520 and kerb_s.max() < 720, (kerb_s.min(), kerb_s.max())
    assert errs[0] < 0.1
    # s_est is only an even-spacing estimate (~0.05 m of its own error), so
    # compare the last lap with the first instead of with zero
    assert errs[2] < 0.1 and errs[2] < errs[0] + 0.02, errs


def _true_kerb_s_range():
    """Lap distance range of the synthetic kerb hits: lap 1 (offset 1.5)
    puts its right wheels (lat + 0.8) beyond KERB_LAT_M, and the wheels sit
    1.45 m ahead of / behind the reference point."""
    from synthetic_track import FRONT_M, HALF_TRACK_M, KERB_LAT_M, _lap_lat
    s = np.arange(0.0, RING_LENGTH_M, 0.05)
    on = s[_lap_lat((0.0, 1.5, -1.5), 1, s) + HALF_TRACK_M > KERB_LAT_M]
    return float(on.min()) - FRONT_M, float(on.max()) + FRONT_M


def test_paint_cells_at_true_distance_and_width(tmp_path, monkeypatch):
    """Important 1: paint is sent as the topography's own 2 m x 0.5 m cells,
    not re-sampled onto the ~600-row x 11-column ribbon."""
    f = write_hilly_session(sess(tmp_path) / "Hill-paint.duckdb")
    _build_and_save(f)
    d = _trackmap(f, monkeypatch, lap=1)
    assert d["paint_cell"] == {"along_m": 2.0, "across_m": 0.5}
    s = d["surface"]
    assert len(s["s"]) == s["rows"]
    # lap distance per row: monotone apart from the start/finish wrap (the
    # first and the last row of a lap may each sit on the other side of it)
    ds = np.diff(np.array(s["s"]))
    assert (ds < -RING_LENGTH_M / 2).sum() <= 2 and (ds[ds > -RING_LENGTH_M / 2] >= 0).all()
    assert min(s["s"]) >= 0 and max(s["s"]) < s["length_m"] <= RING_LENGTH_M + 1
    kerb = np.array([c for c in d["paint"] if c[2] == 5], dtype=float)
    assert len(kerb), d["paint"][:5]
    assert all(c[2] in (2, 3, 4, 5) for c in d["paint"])
    lo, hi = _true_kerb_s_range()
    # cell centres (s = row*2 + 1) within one cell of the true hits, both ends
    assert lo - 2.0 <= kerb[:, 0].min() <= lo + 2.0, (kerb[:, 0].min(), lo)
    assert hi - 2.0 <= kerb[:, 0].max() <= hi + 2.0, (kerb[:, 0].max(), hi)
    # across: only the right of travel, beyond the kerb line, at 0.5 m steps
    assert (kerb[:, 1] > 5.0).all() and kerb[:, 1].max() < 7.0
    assert np.allclose(kerb[:, 1] / 0.5, np.round(kerb[:, 1] / 0.5))
    # the compatibility grid is still there
    assert len(s["kind"]) == s["rows"] * s["cols"]


def test_paint_empty_without_layout(tmp_path, monkeypatch):
    legacy = write_hilly_session(sess(tmp_path) / "Hill-legacy.duckdb", with_pose=False,
                                 with_surface=False)
    d = _trackmap(legacy, monkeypatch)
    assert d["paint"] == [] and d["paint_cell"] == {"along_m": 2.0, "across_m": 0.5}
    assert len(d["surface"]["s"]) == d["surface"]["rows"]


def _cut_lap(path, lap, frac):
    """Add a Lap row part-way through `lap` (a reset / a session that ended
    mid-lap): /trackmap's lap `lap` then covers only its first `frac`."""
    import duckdb
    con = duckdb.connect(str(path))
    ts = [r[0] for r in con.execute('SELECT ts FROM "Lap" ORDER BY ts').fetchall()]
    cut = ts[lap] + frac * (ts[lap + 1] - ts[lap])
    con.execute('INSERT INTO "Lap" VALUES (?, ?)', (cut, 99))
    con.close()
    return path


def test_partial_lap_uses_built_topography(tmp_path, monkeypatch):
    """Important 2: a partial lap's Lap Dist span (~half the ring) matches no
    layout length, but all of it fits inside the one layout: use it."""
    rich = write_hilly_session(sess(tmp_path) / "Hill-rich.duckdb")
    topo = _build_and_save(rich)
    legacy = _cut_lap(write_hilly_session(sess(tmp_path) / "Hill-part.duckdb", with_pose=False,
                                          lap_offsets=(0.5, 0.5, 0.5)), 1, 0.5)
    d = _trackmap(legacy, monkeypatch, lap=1)
    el = d["elevation"]
    assert el["source"] == "topography", el
    assert el["note"] == ""
    y = np.array(d["driven_line"]["y"])
    s = np.linspace(0, RING_LENGTH_M / 2, len(y))      # first half of the lap, evenly timed
    err = np.abs(y[5:-5] - (truth(s[5:-5], 0.5) - topo["datum_m"]))
    assert np.median(err) < 0.15, np.median(err)
    # pose file, partial lap: the built topography too ("recorded")
    rec = _cut_lap(write_hilly_session(sess(tmp_path) / "Hill-part-rec.duckdb"), 1, 0.4)
    assert _trackmap(rec, monkeypatch, lap=1)["elevation"]["source"] == "recorded"


def test_partial_lap_without_matching_layout_says_so(tmp_path, monkeypatch, reference_dir):
    rich = write_hilly_session(sess(tmp_path) / "Hill-rich.duckdb")
    topo = track_topography.build_topography([rich], SLUG, "Synthetic Hill Ring")
    lay = topo["layouts"][0]
    # a layout far shorter than the part of the lap driven: nothing fits
    lay["length_m"] = 300.0
    track_topography.save_topography(topo)
    legacy = _cut_lap(write_hilly_session(sess(tmp_path) / "Hill-part.duckdb", with_pose=False), 1, 0.5)
    d = _trackmap(legacy, monkeypatch, lap=1)
    assert d["elevation"]["source"] == "none"
    assert d["elevation"]["note"] == trackmap_elevation.PARTIAL_NOTE
    assert not d["elevation"]["note"].startswith(FLAT_NOTE)


def test_pick_layout_for_partial_lap():
    topo = {"layouts": [{"length_m": 4000.0}, {"length_m": 5000.0}]}
    pick = track_topography.pick_layout_partial
    assert pick(topo, 3500.0)["length_m"] == 4000.0     # fits both: the closest
    assert pick(topo, 4100.0)["length_m"] == 4000.0     # within 3 % of 4000
    assert pick(topo, 4500.0)["length_m"] == 5000.0
    assert pick(topo, 5200.0) is None
    assert pick({"layouts": []}, 100.0) is None


def test_lateral_sign_minus_one_is_applied_to_surface_and_paint(monkeypatch):
    """Minor 1: the ribbon's columns run from the left edge (-te) to the right
    edge (+te) in main.py's frame, which treats Path Lateral as right-positive.
    The topography's frame is lateral_sign * Path Lateral, so a column at
    ribbon offset g looks up lat = sign * g, and a paint cell at topography
    lat sits at ribbon offset sign * lat."""
    L, N = 1200.0, 600
    h = [0.0] * N
    bank = [0.1] * N                                   # rises 0.1 m/m to the topography's right
    lay = {"length_m": L, "row_step_m": 2.0, "rows": N, "h": h, "bank": bank,
           "n": [1] * N, "lat_min": [0.0] * N, "lat_max": [0.0] * N, "paint_cell_m": 0.5,
           "paint": [[100, 12, 5]], "has_heights": True, "coverage": 1.0,
           "lateral_sign": {"value": -1, "confidence": 1.0, "evidence": "test"},
           "sources": {"files": 1, "laps": 1}}
    topo = {"datum_m": 0.0, "built": "", "layouts": [lay]}
    n = 1200
    s_lap = np.linspace(0.0, L - 1.0, n)
    te = np.full(n, 6.0)
    zeros = np.zeros(n)
    monkeypatch.setattr(track_topography, "load_topography", lambda name, length, **k: (topo, lay))
    out = trackmap_elevation.elevation_payload(
        path=Path("unused.duckdb"), track_name="Synthetic Hill Ring",
        lap_len_m=L - 1.0, s_lap=s_lap, path_lat=zeros, ground_rec=None, attitude=None,
        edge_idx=list(range(n)), track_edge=te,
        left_x=zeros, left_z=zeros, right_x=zeros, right_z=zeros,
        labels=[], line_idx=list(range(n)))
    assert out["elevation"]["source"] == "topography"
    # right edge (ribbon +te) = topography lat -te: 0.1 * -6 = -0.6
    assert np.allclose(out["right_y"], -0.6, atol=1e-6)
    assert np.allclose(out["left_y"], 0.6, atol=1e-6)
    sy = np.array(out["surface"]["y"]).reshape(out["surface"]["rows"], out["surface"]["cols"])
    assert np.allclose(sy[:, -1], -0.6, atol=1e-6) and np.allclose(sy[:, 0], 0.6, atol=1e-6)
    # the kerb cell at topography lat +6 (right) sits on the ribbon's left (-6)
    assert out["paint"] == [[201.0, -6.0, 5]]
    kind = np.array(out["surface"]["kind"]).reshape(sy.shape)
    assert (kind[:, -1] == 0).all()
    assert (kind[:, 0] == 5).any()


def test_lap_dist_rows_keeps_raw_values_and_bridges_the_wrap():
    raw = [1190.0, 1196.0, 2.0, 8.0]          # wraps between the 2nd and 3rd sample
    out = trackmap_elevation.lap_dist_rows(raw, np.array([0.0, 1.0, 1.5, 2.0, 3.0]))
    assert out[0] == 1190.0 and out[1] == 1196.0 and out[3] == 2.0 and out[4] == 8.0
    # the row between them stays next to the line, not at half a lap
    assert 1196.0 <= out[2] <= 1200.0
