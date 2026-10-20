"""Tire model: per-lap extraction (tire_model.py) and profile ingest (grip_profile.py).

Run from the repo root:
    uv run --no-project --with pytest --with fastapi --with duckdb --with numpy `
        --with scipy --with httpx --with pydantic python -m pytest backend/tests -v
"""

from pathlib import Path

import numpy as np
import pytest

from app import grip_profile, tire_class, tire_model as tm
from app.tire_bins import LapBins, bin_axle_samples
from app.telemetry import LapTelemetry, load_telemetry_file
from synthetic_tyre import make_bins

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
    # the v2 raw-sample builder (kept for validate_tire_model.py) ...
    model = tm.build_v2_axle_model(tm.extract_axle_observations_from_lap(_lap())["front"])
    assert "cold-tyre" in model["tyre_temperature"]["warning"]
    # ... and the v3 path, which reads the temperature from the stored bins
    p = grip_profile.GripProfile(car_name="Car")
    p.ingest_tire_laps([_lap()], "f")
    bins = [LapBins.from_dict(l["front"]) for l in p.tire_files["f"]["laps"].values()]
    v3 = tm.build_axle_tire_model(None, bins)
    assert "cold-tyre" in v3["tyre_temperature"]["warning"]
    assert v3["tyre_temperature"]["median_delta_c"] == pytest.approx(-20.0, abs=2.5)


# ── profile ──────────────────────────────────────────────────────────────────

def test_ingest_adds_each_lap_once_and_skips_slow_laps():
    laps = [_lap(), _lap(), _lap()]
    for i, l in enumerate(laps):
        l.lap_number = i
    laps[2].lap_time_s *= 1.5  # out lap
    p = grip_profile.GripProfile(car_name="Car")
    assert p.ingest_tire_laps(laps, "f.duckdb") == [0, 1]
    n = p.tire_sample_count()
    assert n > 0
    assert p.ingest_tire_laps(laps, "f.duckdb") == []
    assert p.tire_sample_count() == n
    assert p.deflection_calibration["front"]["r"] > 0.99


def test_v1_profile_tire_observations_are_discarded():
    # v1 (no version key) and v2 raw-observation profiles lose their tyre data on load
    # (details in test_grip_profile_v3.py); baselines survive and v3 round-trips.
    v1 = {"car_name": "Car", "lap_count": 3, "tc_active_fractions": [0.5],
          "front_axle_obs": {"slip_angle_deg": [1.0], "lateral_g": [1.0]}}
    p = grip_profile.GripProfile.from_dict(v1)
    assert p.tire_model_laps == [] and p.lap_count == 3 and p.tc_active_fractions == [0.5]
    p2 = grip_profile.GripProfile.from_dict(p.to_dict())
    assert p2.tire_model_version == grip_profile.TIRE_MODEL_VERSION


def test_round_trip_preserves_missing_load_measure():
    # No load source at all: the lap is binned in raw lateral G and stays that way.
    p = grip_profile.GripProfile(car_name="Car")
    p.ingest_tire_laps([_lap(susp_force=False)], "f")
    p.deflection_calibration = {}
    d = grip_profile.GripProfile.from_dict(p.to_dict())
    assert d.tire_files == p.tire_files
    assert d.tire_files["f"]["laps"]["1"]["front"]["measure"] == "lateral_g"

# ── real recording ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not FUJI.exists(), reason="Fuji recording not present")
def test_real_fuji_file():
    laps = load_telemetry_file(FUJI)
    p = grip_profile.GripProfile(car_name=laps[0].car_name)
    added = p.ingest_tire_laps(laps, FUJI.name)
    assert len(added) >= 7
    base = p.compute_baselines()
    front = base["front_tire_model"]
    assert front["grip_measure"] == "mu_load_normalized"
    assert front["peak"] is not None
    best = min((l for l in laps if l.lap_time_s < 600), key=lambda l: l.lap_time_s)
    gu = tm.compute_grip_utilization_for_lap(
        best, front, base["rear_tire_model"], p.deflection_calibration,
        tm.file_load_reference(laps, p.deflection_calibration))
    assert 0.7 < gu["cornering_efficiency"] < 1.2
    # the efficient range stays inside what the driver actually reached
    if front.get("efficient_range") and front.get("max_observed_slip_deg") is not None:
        assert front["efficient_range"][1] <= front["max_observed_slip_deg"]


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


def test_ingest_records_where_the_load_came_from():
    # Susp Force present: that is the source; no force + no calibration: none
    p = grip_profile.GripProfile(car_name="Car")
    with_force = _lap()
    with_force.lap_number = 0
    no_force = _lap(susp_force=False)
    no_force.lap_number = 1
    p.ingest_tire_laps([with_force, no_force], "f")
    assert p.tire_model_load_sources["f#0"] == "Susp Force"
    # the first lap taught the profile a deflection calibration, so the lap
    # without Susp Force falls back one more step: Susp Pos, calibrated
    assert p.tire_model_load_sources["f#1"] == "Susp Pos (calibrated to Susp Force)"
    summary = p.compute_baselines()["load_normalization"]
    assert summary["laps_by_load_source"]["Susp Force"] == 1
    assert "note" in summary and "Susp Force" in summary["note"]

    # nothing to take a load from, and no calibration: none, and no ratio
    bare = grip_profile.GripProfile(car_name="Bare")
    bare.ingest_tire_laps([_lap(susp_force=False)], "g")
    assert bare.tire_model_load_sources["g#1"].startswith("none")
    assert "note" not in bare.compute_baselines()["load_normalization"]


def test_tire_load_wins_over_susp_force_and_needs_no_note():
    lap = _lap()
    for w in ("FL", "FR", "RL", "RR"):
        lap.channels[f"Tire Load {w}"] = lap.channels[f"Susp Force {w}"] * 2.0
    obs = tm.extract_axle_observations_from_lap(lap)
    assert obs["front"].load_source == "Tire Load"
    p = grip_profile.GripProfile(car_name="Car")
    p.ingest_tire_laps([lap], "f")
    assert "note" not in p.compute_baselines()["load_normalization"]


def test_load_sources_round_trip():
    p = grip_profile.GripProfile(car_name="Car")
    p.ingest_tire_laps([_lap()], "f")
    p2 = grip_profile.GripProfile.from_dict(p.to_dict())
    assert p2.tire_model_load_sources == p.tire_model_load_sources == {"f#1": "Susp Force"}

# -- v3 per-sample extraction -------------------------------------------------

def test_kerb_samples_masked():
    lap = _lap()
    base = tm._lap_axle_samples(lap)["front"]["mask"].copy()
    n = len(lap.time_s)
    for w in ("FL", "FR", "RL", "RR"):
        lap.channels[f"Surface Type {w}"] = np.zeros(n)
    lap.channels["Surface Type FL"][1000:1050] = 5.0   # kerb
    s = tm._lap_axle_samples(lap)
    assert not s["front"]["mask"][1000:1050].any()
    assert base[1000:1050].any()                       # was a valid corner sample
    assert (s["rear"]["mask"] == base).all()
    assert (s["front"]["mask"][:1000] == base[:1000]).all()


def test_true_mu_from_forces():
    lap = _lap()
    n = len(lap.time_s)
    for w in ("FL", "FR"):
        lap.channels[f"Tire Lateral Force {w}"] = np.full(n, -6000.0)  # sign-agnostic
        lap.channels[f"Tire Load {w}"] = np.full(n, 4000.0)
    s = tm._lap_axle_samples(lap)
    assert np.allclose(s["front"]["true_mu"], 1.5)
    assert np.isnan(s["rear"]["true_mu"]).all()


def test_true_mu_nan_without_forces():
    s = tm._lap_axle_samples(_lap())
    assert np.isnan(s["front"]["true_mu"]).all()


def test_true_mu_nan_at_low_load():
    lap = _lap()
    n = len(lap.time_s)
    fz = np.full(n, 4000.0)
    fz[:100] = 500.0                                   # < 30 % of the median
    for w in ("FL", "FR"):
        lap.channels[f"Tire Lateral Force {w}"] = np.full(n, 6000.0)
        lap.channels[f"Tire Load {w}"] = fz
    mu = tm._lap_axle_samples(lap)["front"]["true_mu"]
    assert np.isnan(mu[:100]).all() and np.allclose(mu[100:], 1.5)


def test_new_state_fields_present():
    lap = _lap()
    n = len(lap.time_s)
    for w in ("FL", "FR", "RL", "RR"):
        lap.channels[f"Camber {w}"] = np.full(n, -0.05)
        lap.channels[f"TyresWear {w}"] = np.full(n, 0.9)     # remaining tread
        lap.channels[f"TyresCarcassTemp {w}"] = np.full(n, 85.0)
    lap.channels["Camber FL"][:10] = 0.59                    # outlier -> NaN
    lap.channels["TyresCarcassTemp RL"][:10] = -273.15       # sentinel -> NaN
    s = tm._lap_axle_samples(lap)
    f, r = s["front"], s["rear"]
    assert np.allclose(f["camber_deg"][10:], 2.865, atol=0.001)
    assert np.isnan(f["camber_deg"][:10]).all()
    assert np.allclose(f["wear"], 0.1)                       # worn = 1 - remaining
    assert np.allclose(f["carcass_delta_c"], -5.0)
    assert np.isnan(r["carcass_delta_c"][:10]).all()
    assert np.allclose(r["carcass_delta_c"][10:], -5.0)
    assert np.allclose(f["speed_kmh"], lap.channels["Ground Speed"])
    assert np.allclose(f["long_g"], 0.0)


def test_camber_opposite_signs_do_not_cancel():
    lap = _lap()
    n = len(lap.time_s)
    lap.channels["Camber FL"] = np.full(n, 0.05)
    lap.channels["Camber FR"] = np.full(n, -0.05)
    s = tm._lap_axle_samples(lap)
    assert np.allclose(s["front"]["camber_deg"], 2.865, atol=0.001)


def test_new_fields_are_independent_arrays():
    lap = _lap()
    s = tm._lap_axle_samples(lap)
    assert not np.shares_memory(s["front"]["speed_kmh"], lap.channels["Ground Speed"])
    assert not np.shares_memory(s["front"]["long_g"], lap.channels["G Force Long"])
    f = s["front"]
    f["wear"][0] = 1.0
    assert np.isnan(f["camber_deg"][0]) and np.isnan(s["rear"]["wear"][0])


def test_new_fields_nan_when_channels_absent():
    f = tm._lap_axle_samples(_lap())["front"]
    for k in ("carcass_delta_c", "wear", "camber_deg", "true_mu"):
        assert np.isnan(f[k]).all()


# -- new HF channels ----------------------------------------------------------

def test_hf_wheel_channels_registered():
    from app.telemetry import WHEEL_CHANNELS
    for ch in ("Camber", "TyresWear", "TyresCarcassTemp", "Surface Type", "Tire Lateral Force"):
        assert ch in WHEEL_CHANNELS


def test_surface_type_not_interpolated():
    from app.telemetry import _resample_to_grid
    src_t = np.arange(4) / 10.0                      # 10 Hz
    src_v = np.array([0.0, 0.0, 5.0, 5.0])
    grid = np.arange(0, 0.3, 0.01)                   # 100 Hz
    out = _resample_to_grid(src_t, src_v, grid, categorical=True)
    assert set(np.unique(out)) <= {0.0, 5.0}
    # linear would put values strictly between 0 and 5 around t=0.1..0.2
    assert not np.allclose(_resample_to_grid(src_t, src_v, grid), out)


# -- v3 outputs (build_axle_tire_model / state-aware CE) ------------------------

V2_KEYS = {"peak", "efficient_range", "peak_observed", "peak_grip", "available_grip",
           "max_observed_slip_deg", "confidence", "sample_count", "grip_measure",
           "tyre_temperature"}
V3_KEYS = {"model", "peak_slip_ci90", "peak_grip_ci90", "factors", "factor_effects",
           "prior_used", "version_note", "game_version", "coverage"}


def _cached(laps, version="1.0", n_boot=5):
    """One axle's tire_fit_cache entry, built the way GripProfile.refit_tyre_model does."""
    res = tire_class.fit_car_with_prior(laps, None, n_boot=n_boot)
    fit = res["fit"]
    return {"fit": fit.to_dict() if fit else None, "peak_slip_ci90": res["peak_slip_ci90"],
            "peak_mu_ci90": res["peak_mu_ci90"], "n_boot_ok": res["n_boot_ok"],
            "confidence": res["confidence"], "prior_used": res["prior_used"],
            "version_note": None, "fit_game_version": version, "n_fit_laps": len(laps),
            "n_samples": sum(l.n_samples() for l in laps), "measure": laps[0].measure}


def _mf_laps():
    return make_bins(n_laps=3, true_params={"D0": 1.8}, factors=("load",), seed=0,
                     n_per_lap=1500)


def test_v3_output_has_all_v2_keys():
    laps = _mf_laps()
    out = tm.build_axle_tire_model(_cached(laps), laps)
    assert V2_KEYS | V3_KEYS <= set(out)
    assert out["model"] == "magic_formula_v3"
    assert out["peak_slip_ci90"][0] <= out["peak"] <= out["peak_slip_ci90"][1] + 0.5
    assert out["peak_grip_ci90"] is not None and out["game_version"] == "1.0"
    lo, hi = out["efficient_range"]
    assert lo <= out["peak"] <= hi
    assert out["sample_count"] == sum(l.n_samples() for l in laps)
    assert out["factors"]["load"] == "fitted" and out["factors"]["temp"] == "disabled"
    # only fitted factors get an effect; load lowers mu at the peak (kDL < 0 in the truth)
    assert set(out["factor_effects"]) == {"load"}
    eff = out["factor_effects"]["load"]
    assert eff["variable"] == "load_ratio" and len(eff["values"]) == 3
    assert eff["values"] == sorted(eff["values"])
    assert eff["peak_mu"][0] > eff["peak_mu"][1] > eff["peak_mu"][2]
    assert out["coverage"]["available"] is True


def test_fallback_when_no_fit():
    laps = _mf_laps()
    # a profile whose MF fit failed still has bins: the v2 binned curve is the fallback
    out = tm.build_axle_tire_model({**_cached(laps), "fit": None, "confidence": 0.0}, laps)
    assert out["model"] == "binned_v2_fallback"
    assert V2_KEYS | V3_KEYS <= set(out)
    assert out["peak"] is not None and out["peak_slip_ci90"] is None
    assert out["confidence"] <= tm.FALLBACK_MAX_CONFIDENCE
    assert out["factor_effects"] == {}
    # no data at all: every number is None, never a guess
    empty = tm.build_axle_tire_model(None, [])
    assert empty["model"] == "binned_v2_fallback"
    assert empty["peak"] is None and empty["sample_count"] == 0 and empty["confidence"] == 0.0


def test_coverage_lists_under_sampled_load_cells():
    def block(load, n):
        s = {"slip_angle_deg": np.full(n, 3.0), "lateral_g": np.full(n, 1.5 * load),
             "load_ratio": np.full(n, load), "temp_delta_c": np.full(n, -20.0),
             "tire_pressure_kpa": np.full(n, 170.0), "wear": np.full(n, 0.1),
             "camber_deg": np.full(n, 2.0), "carcass_delta_c": np.full(n, 0.0),
             "long_g": np.zeros(n), "true_mu": np.full(n, np.nan)}
        return bin_axle_samples(s, np.ones(n, bool))
    bins = LapBins.merge([block(1.45, 400), block(1.55, 40)])
    cov = tm.coverage_summary(bins)
    assert cov["available"] is True
    thin = [c for c in cov["under_covered"] if c["samples"] == 40]
    assert thin and thin[0]["load_ratio"] == [1.5, 1.6]
    assert all(c["samples"] < tm.UNDER_COVERED_N for c in cov["under_covered"])
    assert cov["n_under_covered"] >= len(cov["under_covered"]) >= 1
    assert "load_ratio" in cov["summary"]


def test_ce_state_aware_used_with_fit():
    laps = _mf_laps()
    cached = _cached(laps)
    front = tm.build_axle_tire_model(cached, laps)
    lap = _lap()
    cal = tm.fit_deflection_calibration(lap)
    ref = tm.file_load_reference([lap], cal)
    v2 = tm.compute_grip_utilization_for_lap(lap, front, front, cal, ref)
    assert v2["ce_method"] == "v2_envelope"
    fit = tm.MFFit.from_dict(cached["fit"])
    v3 = tm.compute_grip_utilization_for_lap(lap, front, front, cal, ref, front_fit=fit)
    assert v3["ce_method"] == "state_aware_mf"
    assert v3["front_grip_utilization"] == v2["front_grip_utilization"]      # GU unchanged
    assert v3["lap_grip_p90"] == v2["lap_grip_p90"]
    # lap mu is 1.0 at load ratio 1.5; the fit's ceiling there (D0 1.8, kDL -0.4) is ~1.6
    assert 0.5 < v3["cornering_efficiency"] < 0.8
    assert v3["cornering_efficiency"] != v2["cornering_efficiency"]


def test_still_rising_curve_stays_inside_the_observed_span():
    laps = _mf_laps()
    cached = _cached(laps)
    peak = tm.MFFit.from_dict(cached["fit"]).peak_slip_deg
    max_obs = round(peak * 0.6, 2)          # the driver never reached the peak
    cached["fit"] = {**cached["fit"], "max_observed_slip_deg": max_obs, "peak_observed": False}
    out = tm.build_axle_tire_model(cached, laps)
    # PREVIOUSLY peak was the raw fit (> max_obs); an unobserved peak is now the lower bound
    assert out["max_observed_slip_deg"] == max_obs == out["peak"] < out["peak_slip_extrapolated"]
    lo, hi = out["efficient_range"]
    assert hi <= max_obs and lo <= hi
    assert out["efficient_range_note"] and "still rising" in out["efficient_range_note"]
    # the same span with the peak observed is a plateau, not "still rising"
    cached["fit"] = {**cached["fit"], "peak_observed": True}
    plateau = tm.build_axle_tire_model(cached, laps)["efficient_range_note"]
    assert plateau and "still rising" not in plateau and "flat" in plateau
    cached["fit"] = {**cached["fit"], "peak_observed": False}
    fit = tm.MFFit.from_dict(cached["fit"])
    bins = tm._merge_usable(laps, fit.measure)
    state = tm._available_state(fit, tm.cell_arrays(bins))
    want = float(tm.predict(fit, np.array([max_obs]), state)[0])
    assert out["available_grip"] == round(want, 3)


def test_unobserved_peak_is_the_lower_bound_inside_the_data():
    laps = _mf_laps()
    cached = _cached(laps)
    raw = tm.MFFit.from_dict(cached["fit"])
    max_obs = round(raw.peak_slip_deg * 0.6, 2)
    cached["fit"] = {**cached["fit"], "max_observed_slip_deg": max_obs, "peak_observed": False,
                     "peak_slip_deg": 15.0}
    out = tm.build_axle_tire_model(cached, laps)
    assert out["peak"] <= out["max_observed_slip_deg"] == max_obs
    assert out["peak_slip_extrapolated"] == 15.0 > out["peak"]
    assert out["peak_at_grid_bound"] is True
    fit = tm.MFFit.from_dict(cached["fit"])
    assert out["peak_grip"] == round(float(tm.predict(fit, np.array([out["peak"]]))[0]), 3)
    assert out["peak_grip_extrapolated"] == round(fit.peak_mu, 3)
    assert out["peak_slip_ci90"] == cached["peak_slip_ci90"]       # still the extrapolation's
    lap = _lap()
    cal = tm.fit_deflection_calibration(lap)
    ref = tm.file_load_reference([lap], cal)
    gu = tm.compute_grip_utilization_for_lap(lap, out, out, cal, ref)
    assert gu["front_utilization_is_upper_bound"] is True
    # GU is against the in-data lower bound, not the 15 deg extrapolation
    assert gu["front_grip_utilization"] == round(min(gu["front_peak_slip_angle_deg"] / out["peak"], 1.5), 3)


def test_observed_peak_keeps_the_raw_fit():
    laps = _mf_laps()
    cached = _cached(laps)
    fit = tm.MFFit.from_dict(cached["fit"])
    cached["fit"] = {**cached["fit"], "peak_observed": True}
    out = tm.build_axle_tire_model(cached, laps)
    assert out["peak"] == out["peak_slip_extrapolated"] == round(fit.peak_slip_deg, 2)
    assert out["peak_at_grid_bound"] is False
