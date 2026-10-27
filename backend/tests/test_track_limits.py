"""/trackmap for the chase view's car model and track limits (round 2, Task 3),
plus the 5 Hz scoring hold in Lap Dist / Path Lateral that made the 3D
surface a staircase."""

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, telemetry, track_limits, track_topography, trackmap_elevation

from synthetic_track import RING_LENGTH_M, truth, write_hilly_session

SLUG = "synthetic_hill_ring"
MODELS = json.loads((Path(main.__file__).parent / "static" / "car_models.json").read_text(encoding="utf-8"))
FL, FR, RL, RR = 1, 2, 4, 8


@pytest.fixture(autouse=True)
def reference_dir(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    ref.mkdir()
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(ref))
    monkeypatch.setattr(track_topography, "slug_for_track",
                        lambda name: SLUG if name == "Synthetic Hill Ring" else None)
    track_topography._CACHE.clear()
    trackmap_elevation._LAP_CACHE.clear()
    return ref


def _trackmap(path, monkeypatch, lap=1):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", path.parent)
    r = TestClient(main.app).get(
        f"/sessions/sessions/{quote(path.name)}/trackmap", params={"lap_number": lap})
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _file(tmp_path, name, **kw):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    return write_hilly_session(d / name, **kw)


# ── the scoring hold ───────────────────────────────────────────────────────

def test_held_rows_marks_short_repeats_only():
    v = np.array([0, 0, 1, 1, 2, 2, 5, 5, 5, 5, 6], dtype=float)
    held = telemetry.held_rows(v)
    assert held.tolist() == [False, True, False, True, False, True,
                             False, False, False, False, False]      # a 4-row run is a real stop


def test_held_rows_at_speed_drops_any_run_length():
    """10 Hz gate jitter can write one 5 Hz value three times. At speed that
    is still a hold (a moving car's Lap Dist cannot stand still); only a
    slow car's long run is a real stop."""
    v = np.array([0, 0, 0, 10, 10, 20], dtype=float)
    fast = np.full(6, 120.0)
    assert telemetry.held_rows(v, fast).tolist() == [False, True, True, False, True, False]
    slow = np.full(6, 2.0)
    assert telemetry.held_rows(v, slow).tolist() == [False, False, False, False, True, False]
    assert np.allclose(telemetry.dehold(v, fast), [0, 10 / 3, 20 / 3, 10, 15, 20])
    out = trackmap_elevation.lap_dist_rows(v, np.arange(6, dtype=float), speed_kmh=fast)
    assert np.allclose(out, [0, 10 / 3, 20 / 3, 10, 15, 20])


def test_dehold_interpolates_between_updates():
    out = telemetry.dehold([0.0, 0.0, 10.0, 10.0, 20.0])
    assert np.allclose(out, [0, 5, 10, 15, 20])
    assert np.allclose(telemetry.dehold([1.0, 2.0, 3.0]), [1, 2, 3])     # nothing held: unchanged
    assert telemetry.dehold([]) == []


def test_lap_dist_rows_with_held_values():
    raw = [0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0]
    out = trackmap_elevation.lap_dist_rows(raw, np.arange(7, dtype=float))
    assert np.allclose(out, [0, 5, 10, 15, 20, 25, 30])
    # a wrap inside the held pairs still bridges, close to the line
    raw = [1180.0, 1180.0, 1190.0, 1190.0, 4.0, 4.0, 14.0]
    out = trackmap_elevation.lap_dist_rows(raw, np.array([2.0, 3.0, 4.0, 6.0]))
    assert out[0] == 1190.0 and 1190.0 <= out[1] <= 1200.0 and out[2] == 4.0 and out[3] == 14.0


def test_held_scoring_rows_do_not_staircase_the_surface(tmp_path, monkeypatch):
    """Real HF1 files hold Lap Dist / Path Lateral for two 10 Hz rows (5 Hz
    scoring). Every other ribbon row then looked its height up 4 m behind
    where it is, so the surface pitched 0 / 2x the slope row by row and a car
    resting on it would rock."""
    f = _file(tmp_path, "Hill-held.duckdb", scoring_hold=True)
    d = _trackmap(f, monkeypatch)
    assert d["elevation"]["source"] == "lap"
    s = d["surface"]
    R, C = s["rows"], s["cols"]
    mid = np.array(s["y"]).reshape(R, C)[:, C // 2]
    x = np.array(s["x"]).reshape(R, C)[:, C // 2]
    z = np.array(s["z"]).reshape(R, C)[:, C // 2]
    step = np.hypot(np.diff(x), np.diff(z))
    grade = np.diff(mid) / np.maximum(step, 1e-6)
    # the true grade changes slowly (max 0.042, period 1200 m): row to row it
    # moves by < 0.001; the staircase flips it by ~0.04 every row
    jumps = np.abs(np.diff(grade))
    assert np.median(jumps) < 0.004, np.median(jumps)
    sr = np.array(s["s"])
    assert np.median(np.abs(mid - (truth(sr, 0.0) - d["elevation"]["datum_m"]))) < 0.1


# ── car class and dimensions ───────────────────────────────────────────────

def test_class_dims_exact_alias_fallback():
    m = {"classes": {"LMGT3": {"wheelbase": 2.7}, "Hypercar": {"wheelbase": 3.15}},
         "aliases": {"LMH": "Hypercar"}}
    assert track_limits.class_dims(m, "Hypercar") == ("Hypercar", {"wheelbase": 3.15}, "exact")
    assert track_limits.class_dims(m, "LMH") == ("Hypercar", {"wheelbase": 3.15}, "alias")
    assert track_limits.class_dims(m, "unknown") == ("LMGT3", {"wheelbase": 2.7}, "fallback")
    assert track_limits.class_dims(m, None)[2] == "fallback"


def test_car_payload_from_metadata():
    gt3 = track_limits.car_payload({"VehicleClass": "GT3", "VehicleModel": "Porsche 911 GT3 R LMGT3"}, "")
    assert gt3["class"] == "LMGT3" and gt3["source"] == "exact"
    assert gt3["dims"] == MODELS["classes"]["LMGT3"]
    assert gt3["model"] == "Porsche 911 GT3 R LMGT3"
    lmp2 = track_limits.car_payload({"VehicleClass": "LMP2"}, "")
    assert lmp2["class"] == "LMP2" and lmp2["source"] == "exact"
    hyper = track_limits.car_payload({}, "Ferrari AF Corse 2026 #50:WEC")      # livery table
    assert hyper["class"] == "Hypercar" and hyper["source"] == "exact"
    unk = track_limits.car_payload({}, "Somebody #999")
    assert unk["class"] == "LMGT3" and unk["source"] == "fallback" and unk["requested"] == "unknown"


# ── wheels beyond the edge ─────────────────────────────────────────────────

GT3 = MODELS["classes"]["LMGT3"]


def test_wheels_out_geometry():
    lat = np.array([0.0, 5.0, -5.0, 7.0, 4.6])
    edge = np.full(5, 5.5)
    ang = np.array([0.0, 0.0, 0.0, 0.0, 10.0])     # last: nose turned 10 deg to the right
    m = track_limits.wheels_out(lat, edge, ang, GT3)
    # GT3: right wheels at lat + 0.85, inner tyre edge 0.155 in from that
    assert m.tolist() == [0, FR | RR, FL | RL, 15, FR]


def test_trackmap_wheels_out_in_the_kerb_excursion(tmp_path, monkeypatch):
    """Lap 1 runs out to lat 5.0 between s = 600 and 640 m: with a 5.5 m
    half-width the GT3's right tyres (inner edge at 5.695 m) are beyond the
    edge, the left ones are not."""
    f = _file(tmp_path, "Hill-limits.duckdb", track_edge_m=5.5)
    d = _trackmap(f, monkeypatch, lap=1)
    tl = d["track_limits"]
    assert tl["available"] is True and tl["note"]
    dl = d["driven_line"]
    n = len(dl["x"])
    for k in ("lat", "edge_half", "wheels_out"):
        assert len(dl[k]) == n, k
    assert np.allclose(dl["edge_half"], 5.5)
    m = np.array(dl["wheels_out"])
    assert (m & (FL | RL)).sum() == 0
    out = np.nonzero(m)[0]
    assert len(out) > 0
    s = np.linspace(0, RING_LENGTH_M, n)          # lap 1 is evenly timed: s ~ linear in the index
    assert s[out].min() > 540 and s[out].max() < 700, (s[out].min(), s[out].max())
    assert (m[(s > 610) & (s < 630)] == FR | RR).all()
    assert max(dl["lat"]) == pytest.approx(5.0, abs=0.05)
    # lap 0 stays on the centre line: nothing out
    d0 = _trackmap(f, monkeypatch, lap=0)
    assert not any(d0["driven_line"]["wheels_out"])


def test_track_edge_sign_flip_does_not_pinch_the_edge(tmp_path, monkeypatch):
    """Track Edge carries Path Lateral's sign. Where the car crosses the
    centre line a held row sat between +6 and -6, and interpolating it
    (dehold) gave a half-width of ~0 there: all four wheels "out" on the
    centre line (21 single points on the 10/27 Road Atlanta lap)."""
    f = _file(tmp_path, "Hill-flip.duckdb", scoring_hold=True)
    d = _trackmap(f, monkeypatch, lap=1)       # ends blending from +1.5 to -1.5 (crosses 0)
    lat = np.array(d["driven_line"]["lat"])
    assert lat.max() > 1 and lat.min() < -1
    assert min(d["driven_line"]["edge_half"]) > 5.9
    assert not any(d["driven_line"]["wheels_out"])
    # the 2D edges do not pinch either: track width stays ~12 m
    t = d["track"]
    w = np.hypot(np.subtract(t["left_x"], t["right_x"]), np.subtract(t["left_z"], t["right_z"]))
    assert w.min() > 11.5, w.min()


def test_trackmap_without_edges_has_no_track_limits(tmp_path, monkeypatch):
    f = _file(tmp_path, "Hill-noedge.duckdb", with_edges=False)
    d = _trackmap(f, monkeypatch, lap=1)
    assert d["track_limits"]["available"] is False
    assert "Track Edge" in d["track_limits"]["note"]
    assert d["driven_line"]["wheels_out"] == [] and d["driven_line"]["lat"] == []


def test_trackmap_car_class(tmp_path, monkeypatch):
    d = _trackmap(_file(tmp_path, "Hill-car.duckdb"), monkeypatch)
    assert d["car"]["class"] == "LMGT3" and d["car"]["source"] == "fallback"
    d = _trackmap(_file(tmp_path, "Hill-hyper.duckdb", vehicle_class="Hypercar"), monkeypatch)
    assert d["car"]["class"] == "Hypercar" and d["car"]["source"] == "exact"
    assert d["car"]["dims"]["wheelbase"] == MODELS["classes"]["Hypercar"]["wheelbase"]
