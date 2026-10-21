"""Geometry-first corner detection (track_corners.py) on a synthetic track.

Run: uv run --no-project --with pytest --with httpx --with fastapi --with duckdb --with numpy --with scipy --with pydantic pytest backend/tests/test_track_corners.py
"""
import numpy as np

import app.track_corners as tc
from app.telemetry import LapTelemetry

V = 50.0   # m/s
LENGTH = 4000.0


def gauss_corner(s, centre, heading_deg, width=40.0, sign=1.0):
    """Curvature bump that turns the car by heading_deg in total."""
    area = np.radians(heading_deg)
    peak = area / (width * np.sqrt(2 * np.pi))
    return sign * peak * np.exp(-0.5 * ((s - centre) / width) ** 2)


def make_lap(corners, jitter=0.0, braking_at=(), seed=0):
    """corners: [(apex_m, heading_deg, sign)]. 20 Hz at a constant 50 m/s."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, LENGTH / V, 0.05)
    s = t * V
    k = sum(gauss_corner(s, c + rng.normal(0, jitter), h, sign=sg) for c, h, sg in corners)
    brake = np.zeros_like(t)
    for b in braking_at:
        brake[(s > b - 150) & (s < b)] = 80.0
    ch = {
        "Lap Dist": s, "Ground Speed": np.full_like(t, V * 3.6), "Yaw Rate": k * V,
        "G Force Lat": k * V * V / 9.81, "Brake Pos": brake, "Throttle Pos": np.where(brake > 0, 0.0, 100.0),
    }
    return LapTelemetry("d", "Car", "Track", "Race", 1, float(t[-1]), t, ch)


CORNERS = [(600, 90, 1), (1500, 150, -1), (2300, 60, 1), (3200, 120, -1)]


def test_finds_each_corner_near_its_apex_with_the_right_heading():
    prof = tc.curvature_profile(make_lap(CORNERS))
    found = tc.detect_apexes(prof)
    assert len(found) == 4
    for f, (apex, heading, sign) in zip(found, CORNERS):
        assert abs(f["apex_m"] - apex) < 15
        assert abs(f["heading_deg"] - heading) < 0.25 * heading + 5   # smoothing and tails
        assert f["sign"] == sign


def test_gentle_kink_below_minimum_heading_is_ignored():
    prof = tc.curvature_profile(make_lap(CORNERS + [(2800, 8, 1)]))
    assert len(tc.detect_apexes(prof)) == 4


def test_chicane_is_split_by_direction_change():
    prof = tc.curvature_profile(make_lap([(1000, 60, 1), (1120, 60, -1)]))
    found = tc.detect_apexes(prof)
    assert [f["sign"] for f in found] == [1, -1]


def test_demand_label_braking_vs_flat():
    prof = tc.curvature_profile(make_lap(CORNERS, braking_at=(600, 1500)))
    d = [f["demand"] for f in tc.detect_apexes(prof)]
    assert d[:2] == ["braking", "braking"] and d[2:] == ["flat", "flat"]


def test_consensus_survives_lap_to_lap_apex_jitter():
    laps = [tc.curvature_profile(make_lap(CORNERS, jitter=8, seed=i)) for i in range(5)]
    per = [tc.detect_apexes(p) for p in laps]
    cons = tc.consensus_corners(per, [p["length_m"] for p in laps])
    assert [c["n"] for c in cons] == [1, 2, 3, 4]
    assert all(c["seen_in_laps"] == "5/5" for c in cons)


def test_a_bend_seen_in_only_one_of_several_laps_is_dropped():
    good = [tc.curvature_profile(make_lap(CORNERS, seed=i)) for i in range(3)]
    odd = tc.curvature_profile(make_lap(CORNERS + [(2800, 80, 1)]))
    per = [tc.detect_apexes(p) for p in good + [odd]]
    cons = tc.consensus_corners(per, [p["length_m"] for p in good + [odd]])
    assert len(cons) == 4


def test_merge_close_joins_touching_same_direction_bits_only():
    a = {"apex_m": 100, "start_m": 50, "end_m": 120, "radius_m": 60, "heading_deg": 30, "sign": 1,
         "min_speed_kmh": 100, "brake_peak_pct": 0, "min_throttle_pct": 50, "demand": "lift"}
    b = {**a, "apex_m": 160, "start_m": 130, "end_m": 200, "radius_m": 50, "heading_deg": 40, "demand": "braking"}
    assert len(tc.merge_close([a, b], gap_m=5)) == 2
    m = tc.merge_close([a, b], gap_m=20)
    assert len(m) == 1 and m[0]["heading_deg"] == 70 and m[0]["demand"] == "braking"
    assert len(tc.merge_close([a, {**b, "sign": -1}], gap_m=20)) == 2


def test_lateral_g_fallback_and_unusable_laps():
    lap = make_lap(CORNERS)
    del lap.channels["Yaw Rate"]
    prof = tc.curvature_profile(lap)
    assert prof["source"] == "lateral_g" and len(tc.detect_apexes(prof)) == 4
    lap.channels.pop("G Force Lat")
    assert tc.curvature_profile(lap) is None
    assert tc.detect_track_corners([lap])["available"] is False
