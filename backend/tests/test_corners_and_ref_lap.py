"""Corner names for the coach (track_db.annotate_report_with_corners) and the
fastest-real-lap reference (telemetry.reference_lap_time) - 10/18/2026."""

import numpy as np

from app import grip_profile
from app.telemetry import LapTelemetry, reference_lap_time
from app.track_db import annotate_report_with_corners, corner_index_for, locate_corner


def _lap(n, t, complete=True):
    return LapTelemetry("d", "car", "track", "race", n, t, np.zeros(1), is_complete=complete)


# ── corners ──────────────────────────────────────────────────────────────────

def test_known_track_index_is_sorted():
    idx = corner_index_for("Fuji Speedway")
    assert idx and [p["dist"] for p in idx] == sorted(p["dist"] for p in idx)


def test_locate_near_and_between():
    idx = [{"dist": 0.0, "corner": "S/F"}, {"dist": 500.0, "corner": "T1"},
           {"dist": 900.0, "corner": "T2"}]
    assert locate_corner(530, idx) == "T1 (+30 m)"
    assert locate_corner(470, idx) == "T1 (-30 m)"
    assert locate_corner(700, idx) == "between T1 and T2"
    assert locate_corner(1500, idx) == "T2 (+600 m)"  # past the last point
    assert locate_corner(100, []) is None


def test_annotate_events_and_unknown_track():
    report = {
        "abs_events": [{"lap_dist_start_m": 770.0, "lap_dist_end_m": 790.0}],
        "tc_events": [{"lap_dist_start_m": 900.0, "lap_dist_end_m": 1300.0}],
        "understeer_candidates": [],
        "slip_angle_analysis": {"peak_front_slip_angle_deg": {"value": 4.0, "lap_dist_m": 1310.0}},
    }
    annotate_report_with_corners(report, "Fuji Speedway")
    assert report["corner_names_available"] is True
    assert report["abs_events"][0]["corner"].startswith("T1 Apex")
    assert "corner_end" not in report["abs_events"][0]
    assert report["tc_events"][0]["corner_end"].startswith("T3 Apex")
    assert report["slip_angle_analysis"]["peak_front_slip_angle_deg"]["corner"].startswith("T3")

    other = {"abs_events": [{"lap_dist_start_m": 10.0, "lap_dist_end_m": 20.0}]}
    annotate_report_with_corners(other, "Nowhere Ring Zzyzx")
    assert other["corner_names_available"] is False
    assert "corner" not in other["abs_events"][0] and "corner_index" not in other


# ── reference lap ────────────────────────────────────────────────────────────

def test_reference_skips_unfinished_and_join_laps():
    laps = [_lap(19, 71.5), _lap(20, 106.2), _lap(21, 105.4), _lap(22, 119.4),
            _lap(23, 14.1, complete=False)]
    assert reference_lap_time(laps) == 105.4
    assert reference_lap_time([_lap(0, 14.1, complete=False)]) is None


def test_ingest_cutoff_uses_real_fastest(monkeypatch):
    # Before the fix only the 14.1 s partial lap passed the 115 % cutoff.
    laps = [_lap(1, 73.1), _lap(2, 79.9), _lap(3, 73.0), _lap(4, 90.2), _lap(5, 14.1, complete=False)]
    monkeypatch.setattr(grip_profile, "fit_deflection_calibration", lambda lap: {})
    monkeypatch.setattr(grip_profile, "file_load_reference", lambda laps, cal: {})
    monkeypatch.setattr(grip_profile, "extract_axle_observations_from_lap",
                        lambda lap, cal, ref: {"front": grip_profile.AxleSlipObservations(),
                                               "rear": grip_profile.AxleSlipObservations()})
    added = grip_profile.GripProfile(car_name="x").ingest_tire_laps(laps, file_key="f")
    assert added == [1, 2, 3, 5]  # 90.2 s > 73.0 * 1.15; partial lap 5 still in
