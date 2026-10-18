"""Tire model v2 (tire_model.py / grip_profile.py).

Run from the repo root:
    uv run --no-project --with pytest --with fastapi --with duckdb --with numpy `
        --with scipy --with httpx --with pydantic python -m pytest backend/tests -v
"""

from pathlib import Path

import numpy as np
import pytest

from app import grip_profile, tire_model as tm
from app.telemetry import LapTelemetry, load_telemetry_file

FUJI = (Path(__file__).resolve().parents[2] / "data" / "sessions"
        / "Fuji Speedway-Team WRT 2026 _69_LP-2026-10-15T03_10_17Z.duckdb")


def _curve(slip, peak=3.0, top=1.6, falloff=0.08):
    """Saturating curve: linear up to `peak`, then falling `falloff` g/deg."""
    slip = np.abs(slip)
    return np.where(slip <= peak, top * slip / peak, top - falloff * (slip - peak))


# ── curve fit ────────────────────────────────────────────────────────────────

def test_fit_finds_interior_peak():
    rng = np.random.default_rng(0)
    sa = rng.uniform(0, 7, 6000)
    fit = tm._fit_curve(sa, _curve(sa) + rng.normal(0, 0.03, sa.size))
    assert fit["peak_observed"] is True
    assert 2.6 <= fit["peak"] <= 3.4
    assert fit["available_grip"] >= fit["peak_grip"]  # p90 envelope >= mean


def test_thin_tail_bin_cannot_be_the_peak():
    # v1 regression: a 5-sample last bin with a high mean won the argmax
    # (WRT rear "peak" 4.2°). A plateau with a thin high tail must not.
    rng = np.random.default_rng(1)
    sa = rng.uniform(0, 4.0, 3000)
    g = np.minimum(sa / 1.8, 1.0) * 1.45 + rng.normal(0, 0.02, sa.size)
    sa = np.concatenate([sa, np.full(5, 4.2)])
    g = np.concatenate([g, np.full(5, 1.75)])
    fit = tm._fit_curve(sa, g)
    assert fit["peak"] < 4.0


def test_still_rising_curve_reports_peak_not_observed():
    rng = np.random.default_rng(2)
    sa = rng.uniform(0, 4, 3000)
    fit = tm._fit_curve(sa, 0.4 * sa + rng.normal(0, 0.02, sa.size))
    assert fit["peak_observed"] is False
    assert fit["peak"] >= 3.4  # lower bound at the top of the data


def test_too_few_samples_returns_none():
    assert tm._fit_curve(np.linspace(0, 5, 100), np.linspace(0, 1, 100)) is None


# ── per-lap extraction ───────────────────────────────────────────────────────

def _lap(n=4000, susp_force=True, tyre_temp=70.0):
    """Synthetic 20 Hz lap: straights at 80 km/h with no G, corners at
    200 km/h whose axle load is 1.5x the straight (downforce), steady."""
    t = np.arange(n) / 20.0
    cornering = (np.arange(n) // 200) % 2 == 1
    speed = np.where(cornering, 200.0, 80.0)
    slip_rad = np.where(cornering, np.radians(3.0), 0.0)
    lat_g = np.where(cornering, 1.5, 0.0)
    ch = {
        "Ground Speed": speed, "G Force Lat": lat_g, "G Force Long": np.zeros(n),
        "Yaw Accel": np.zeros(n),
    }
    load = np.where(cornering, 1.5, 1.0)
    for w in ("FL", "FR", "RL", "RR"):
        ch[f"Slip Angle {w}"] = slip_rad
        ch[f"TyresRubberTemp {w}"] = np.full(n, tyre_temp)
        ch[f"TyresPressure {w}"] = np.full(n, 170.0)
        ch[f"Susp Pos {w}"] = 0.05 + 0.02 * load  # offset: not proportional to load
        if susp_force:
            ch[f"Susp Force {w}"] = 4000.0 * load
    return LapTelemetry("d", "Car", "Track", "P", 1, n / 20.0, t, ch,
                        meta={"TyreOptimalTempC": "90,90,90,90"})


def test_load_ratio_from_susp_force_and_temp_delta():
    obs = tm.extract_axle_observations_from_lap(_lap())["front"]
    a = obs.arrays()
    corner = np.abs(a["slip_angle_deg"]) > 1
    assert np.allclose(a["load_ratio"][corner], 1.5)
    assert np.allclose(a["temp_delta_c"], -20.0)
    mu, measure, _ = tm.grip_values(a)
    assert measure == "mu_load_normalized"
    assert np.allclose(mu[corner], 1.0)  # 1.5 g on 1.5x load


def test_deflection_alone_gives_no_load_ratio_but_calibration_does():
    lap = _lap(susp_force=False)
    a = tm.extract_axle_observations_from_lap(lap)["front"].arrays()
    assert not np.isfinite(a["load_ratio"]).any()
    cal = tm.fit_deflection_calibration(_lap())  # from a lap with force
    a = tm.extract_axle_observations_from_lap(lap, cal)["front"].arrays()
    corner = np.abs(a["slip_angle_deg"]) > 1
    assert np.allclose(a["load_ratio"][corner], 1.5, atol=0.01)


def test_steady_state_filter_drops_braking_and_yaw_transients():
    lap = _lap()
    n = len(lap.time_s)
    lap.channels["G Force Long"] = np.where(np.arange(n) < n // 2, 1.0, 0.0)
    lap.channels["Yaw Accel"][-100:] = 3.0
    obs = tm.extract_axle_observations_from_lap(lap)["front"]
    assert len(obs) == n - n // 2 - 100


def test_cold_tyres_are_flagged():
    model = tm.build_axle_tire_model(tm.extract_axle_observations_from_lap(_lap())["front"])
    assert "cold-tyre" in model["tyre_temperature"]["warning"]


# ── profile ──────────────────────────────────────────────────────────────────

def test_ingest_adds_each_lap_once_and_skips_slow_laps():
    laps = [_lap(), _lap(), _lap()]
    for i, l in enumerate(laps):
        l.lap_number = i
    laps[2].lap_time_s *= 1.5  # out lap
    p = grip_profile.GripProfile(car_name="Car")
    assert p.ingest_tire_laps(laps, "f.duckdb") == [0, 1]
    n = len(p.front_axle_obs)
    assert p.ingest_tire_laps(laps, "f.duckdb") == []
    assert len(p.front_axle_obs) == n
    assert p.deflection_calibration["front"]["r"] > 0.99


def test_v1_profile_tire_observations_are_discarded():
    v1 = {"car_name": "Car", "lap_count": 3, "tc_active_fractions": [0.5],
          "front_axle_obs": {"slip_angle_deg": [1.0], "lateral_g": [1.0]}}
    p = grip_profile.GripProfile.from_dict(v1)
    assert len(p.front_axle_obs) == 0 and p.lap_count == 3 and p.tc_active_fractions == [0.5]
    p2 = grip_profile.GripProfile.from_dict(p.to_dict())  # v2 round trip keeps samples
    assert p2.tire_model_version == grip_profile.TIRE_MODEL_VERSION


def test_round_trip_preserves_nan_fields():
    p = grip_profile.GripProfile(car_name="Car")
    p.ingest_tire_laps([_lap(susp_force=False)], "f")
    p.deflection_calibration = {}
    d = grip_profile.GripProfile.from_dict(p.to_dict())
    assert len(d.front_axle_obs) == len(p.front_axle_obs)
    assert not np.isfinite(d.front_axle_obs.arrays()["load_ratio"]).any()


# ── real recording ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not FUJI.exists(), reason="Fuji recording not present")
def test_real_fuji_file():
    laps = load_telemetry_file(FUJI)
    p = grip_profile.GripProfile(car_name=laps[0].car_name)
    added = p.ingest_tire_laps(laps, FUJI.name)
    assert len(added) >= 7
    front = tm.build_axle_tire_model(p.front_axle_obs)
    assert front["grip_measure"] == "mu_load_normalized"
    assert front["peak"] is not None
    assert "cold-tyre" in front["tyre_temperature"]["warning"]  # 59-73 °C vs 92
    best = min((l for l in laps if l.lap_time_s < 600), key=lambda l: l.lap_time_s)
    gu = tm.compute_grip_utilization_for_lap(
        best, front, tm.build_axle_tire_model(p.rear_axle_obs), p.deflection_calibration,
        tm.file_load_reference(laps, p.deflection_calibration))
    assert 0.7 < gu["cornering_efficiency"] < 1.2


def test_flying_lap_uses_file_reference():
    # A flying lap with no low-speed straight samples gets its load ratio
    # from another lap of the file (the out lap), not none at all.
    fast = _lap()
    fast.channels["Ground Speed"] = np.where(fast.channels["Ground Speed"] < 100, 250.0, 200.0)
    alone = tm.extract_axle_observations_from_lap(fast)["front"].arrays()
    assert not np.isfinite(alone["load_ratio"]).any()
    ref = tm.file_load_reference([fast, _lap()])
    a = tm.extract_axle_observations_from_lap(fast, None, ref)["front"].arrays()
    assert np.allclose(a["load_ratio"][np.abs(a["slip_angle_deg"]) > 1], 1.5)
