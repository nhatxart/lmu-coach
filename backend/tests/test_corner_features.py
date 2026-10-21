"""Per-corner features (corner_features.py) on a synthetic lap with a known corner.

Run: uv run --no-project --with pytest --with httpx --with fastapi --with duckdb --with numpy --with scipy --with pydantic pytest backend/tests/test_corner_features.py
"""
import json

import numpy as np

from app.corner_features import compute_corner_analysis
from app.telemetry import LapTelemetry


def _ramp(t, pts):
    xs, ys = zip(*pts)
    return np.interp(t, xs, ys)


def make_lap(brake_pts, steer_start=8.8, extra=None, drop=()):
    """Straight at 220 km/h, one right-hand corner (min speed ~90 km/h at 10 s),
    throttle coming on from 10.5 s. Brake trace is given by brake_pts."""
    t = np.arange(0.0, 30.0, 0.05)
    speed = _ramp(t, [(0, 220), (5, 220), (10, 90), (16, 200), (30, 200)])
    brake = _ramp(t, brake_pts)
    steer = _ramp(t, [(0, 0), (steer_start, 0), (10, 40), (11, 40), (14, 0), (30, 0)])
    lat_g = 1.6 * steer / 40.0
    ch = {
        "Ground Speed": speed,
        "Brake Pos": brake,
        "Throttle Pos": np.clip((t - 10.5) / 2.0 * 100.0, 0, 100),
        "Steering Pos": steer,
        "G Force Lat": lat_g,
        "G Force Long": np.zeros_like(t),
        "Yaw Rate": lat_g * 9.81 / (speed / 3.6),
        "Lap Dist": np.cumsum(speed / 3.6 * 0.05),
        "ABS": np.zeros_like(t),
        "TC": np.zeros_like(t),
    }
    if extra:
        ch.update(extra)
    for k in drop:
        ch.pop(k)
    return LapTelemetry("d", "Car", "Track", "Race", 1, 30.0, t, ch)


RELEASE_THEN_STEER = [(0, 0), (5, 0), (5.2, 100), (7, 100), (8.0, 0), (30, 0)]
TRAIL_BRAKE = [(0, 0), (5, 0), (5.2, 100), (7, 100), (10, 0), (30, 0)]


def test_release_then_steer_is_flagged_with_evidence():
    r = compute_corner_analysis(make_lap(RELEASE_THEN_STEER))
    assert r["available"] and r["corner_count"] == 1
    c = r["corners"][0]
    assert c["braking"]["peak_pct"] == 100
    assert c["entry"]["brake_zero_before_turn_in_s"] >= 0.5
    assert "brake_released_before_turning" in c["flags"]
    assert c["braking"]["reapplications"] == 0


def test_trail_braking_is_not_flagged_as_release_then_steer():
    c = compute_corner_analysis(make_lap(TRAIL_BRAKE))["corners"][0]
    assert c["entry"]["brake_at_turn_in_pct"] > 20
    assert "brake_released_before_turning" not in c["flags"]


def test_brake_reapplication_is_counted():
    pts = [(0, 0), (5, 0), (5.2, 100), (6.0, 100), (6.4, 40), (6.9, 85), (7.5, 85), (8.5, 0), (30, 0)]
    c = compute_corner_analysis(make_lap(pts))["corners"][0]
    assert c["braking"]["reapplications"] == 1
    assert "brake_reapplied_after_release" in c["flags"]


def test_committed_throttle_pickup_ignores_nothing_before_it():
    c = compute_corner_analysis(make_lap(TRAIL_BRAKE))["corners"][0]
    # throttle ramps 10.5 s -> 12.5 s while min speed is at ~10 s
    assert 0.2 <= c["exit"]["throttle_pickup_vs_min_speed_s"] <= 0.8
    assert 1.5 <= c["exit"]["time_to_90pct_throttle_s"] <= 2.2


def test_tc_seconds_on_exit_are_reported_and_flagged():
    t = np.arange(0.0, 30.0, 0.05)
    tc = ((t >= 11.0) & (t < 11.5)).astype(float)
    c = compute_corner_analysis(make_lap(TRAIL_BRAKE, extra={"TC": tc}))["corners"][0]
    assert c["exit"]["tc_s"] >= 0.4
    assert "tc_on_exit" in c["flags"]


def test_lateral_g_fallback_when_no_yaw_channel():
    r = compute_corner_analysis(make_lap(TRAIL_BRAKE, drop=("Yaw Rate",)))
    assert r["available"] and r["yaw_source"] == "lateral_g_estimate"


def test_unavailable_without_required_channels():
    r = compute_corner_analysis(make_lap(TRAIL_BRAKE, drop=("Brake Pos",)))
    assert r["available"] is False and "Brake Pos" in r["reason"]


def test_result_is_json_serialisable_and_has_no_nulls():
    r = compute_corner_analysis(make_lap(RELEASE_THEN_STEER))
    text = json.dumps(r)
    assert "null" not in text


def test_two_apexes_in_one_same_direction_run_are_split():
    t = np.arange(0.0, 40.0, 0.05)
    speed = _ramp(t, [(0, 200), (5, 200), (9, 90), (13, 170), (17, 90), (22, 200), (40, 200)])
    steer = _ramp(t, [(0, 0), (6, 0), (8, 40), (18, 40), (20, 0), (40, 0)])
    lat_g = 1.6 * steer / 40.0
    ch = {"Ground Speed": speed, "Brake Pos": np.zeros_like(t), "Throttle Pos": np.full_like(t, 50.0),
          "Steering Pos": steer, "G Force Lat": lat_g, "G Force Long": np.zeros_like(t),
          "Yaw Rate": lat_g * 9.81 / (speed / 3.6), "Lap Dist": np.cumsum(speed / 3.6 * 0.05)}
    r = compute_corner_analysis(LapTelemetry("d", "Car", "Track", "Race", 1, 40.0, t, ch))
    assert r["corner_count"] == 2
