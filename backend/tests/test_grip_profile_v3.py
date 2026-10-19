"""Grip profile v3: per-lap bins, model key, versioned cached MF fit (grip_profile.py)."""

import json

import numpy as np
import pytest

from app import grip_profile as gp
from app import tire_class
from app import tire_bins as tb
from app.telemetry import LapTelemetry
from synthetic_tyre import make_bins
from test_tire_model import _lap

LOAD_ONLY = ("load",)


@pytest.fixture(autouse=True)
def _profiles_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "PROFILES_DIR", tmp_path)
    gp.clear_class_prior_cache()
    yield
    gp.clear_class_prior_cache()


@pytest.fixture
def boot_calls(monkeypatch):
    """Counts bootstrap_fit calls (one per axle fit) and shrinks the bootstrap for speed."""
    calls = []
    orig = tire_class.bootstrap_fit

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(tire_class, "bootstrap_fit", counting)
    real = tire_class.fit_car_with_prior

    def small(lap_bins, prior, weight=tire_class.CLASS_PRIOR_WEIGHT, n_boot=50, prior_kind="class"):
        return real(lap_bins, prior, weight, 5, prior_kind)

    monkeypatch.setattr(gp, "fit_car_with_prior", small)
    return calls


def _profile(files: dict, name="Car", car_class="LMGT3") -> gp.GripProfile:
    """Profile whose tire_files hold the given LapBins: {file: (game_version, [LapBins])}."""
    p = gp.GripProfile(car_name=name, car_class=car_class)
    for fk, (version, laps) in files.items():
        p.tire_files[fk] = {"game_version": version, "track": "T", "laps": {
            str(i): {"load_source": "Susp Force", "front": b.to_dict(), "rear": b.to_dict()}
            for i, b in enumerate(laps)}}
    return p


def _bins(n_laps, d0=1.8, seed=0):
    return make_bins(n_laps=n_laps, true_params={"D0": d0}, factors=LOAD_ONLY, seed=seed,
                     n_per_lap=1500)


# -- storage -------------------------------------------------------------------

def test_v2_profile_loads_and_resets_tyre_data():
    v2 = {"car_name": "Car", "lap_count": 3, "tc_active_fractions": [0.5],
          "tire_model_version": 2, "tire_model_laps": ["f#1"],
          "tire_model_load_sources": {"f#1": "Susp Force"},
          "deflection_calibration": {"front": {"slope": 1.0}},
          "front_axle_obs": {"slip_angle_deg": [1.0], "lateral_g": [1.0]},
          "rear_axle_obs": {"slip_angle_deg": [1.0], "lateral_g": [1.0]}}
    p = gp.GripProfile.from_dict(v2)
    assert p.tire_files == {} and p.tire_fit_cache == {} and p.tire_model_laps == []
    assert p.car_class == "unknown"
    assert p.lap_count == 3 and p.tc_active_fractions == [0.5]      # baselines kept
    assert p.deflection_calibration == {"front": {"slope": 1.0}}
    d = p.to_dict()
    assert d["tire_model_version"] == gp.TIRE_MODEL_VERSION == 3
    assert "front_axle_obs" not in d and "rear_axle_obs" not in d
    # a v1 profile (no version key) resets the same way
    assert gp.GripProfile.from_dict({"car_name": "C", "front_axle_obs": {}}).tire_files == {}


def test_ingest_dedups_and_bins():
    laps = [_lap(), _lap()]
    for i, l in enumerate(laps):
        l.lap_number = i
        l.meta["GameVersion"] = "1.2.3"
    p = gp.GripProfile(car_name="Car")
    assert p.ingest_tire_laps(laps, "f.duckdb") == [0, 1]
    assert p.tire_model_laps == ["f.duckdb#0", "f.duckdb#1"]
    assert p.ingest_tire_laps(laps, "f.duckdb") == []
    assert len(p.tire_model_laps) == 2
    entry = p.tire_files["f.duckdb"]
    # game version and track once per file, never per lap (user requirement)
    assert entry["game_version"] == "1.2.3" and entry["track"] == "Track"
    for lap in entry["laps"].values():
        assert set(lap) == {"load_source", "front", "rear"}
        assert lap["front"]["measure"] == tb.MEASURE_MU and lap["front"]["rows"]
    assert p.tire_model_load_sources["f.duckdb#0"] == "Susp Force"
    assert p.deflection_calibration["front"]["r"] > 0.99
    # survives a JSON round trip
    p2 = gp.GripProfile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert p2.tire_model_laps == p.tire_model_laps and p2.tire_files == p.tire_files


def test_native_lap_adds_nothing():
    lap = _lap()
    for w in ("FL", "FR", "RL", "RR"):
        del lap.channels[f"Slip Angle {w}"]
    p = gp.GripProfile(car_name="Car")
    assert p.ingest_tire_laps([lap], "native.duckdb") == []
    assert p.tire_files == {} and p.tire_model_laps == []


def test_car_class_learned_from_file_metadata():
    lap = _lap()
    lap.meta["VehicleClass"] = "hypercar"
    p = gp.GripProfile(car_name="Car")
    p.ingest_tire_laps([lap], "f")
    assert p.car_class == "Hypercar"


def test_profile_keyed_by_model():
    def lap(car_name):
        l = _lap()
        l.car_name = car_name
        l.meta.update({"VehicleModel": "Aston Martin Valkyrie", "VehicleClass": "Hypercar"})
        return l

    a = gp.load_profile_for_lap(lap("Aston Martin THOR Team 2026 #007:WEC"))
    assert a.car_name == "Aston Martin Valkyrie" and a.car_class == "Hypercar"
    a.ingest_tire_laps([lap("Aston Martin THOR Team 2026 #007:WEC")], "f1")
    gp.save_profile(a)
    b = gp.load_profile_for_lap(lap("Some Other Livery #99"))
    assert b.tire_model_laps == ["f1#1"]
    assert [p.name for p in gp.PROFILES_DIR.glob("*.json")] == ["Aston Martin Valkyrie.json"]


# -- fit cache -----------------------------------------------------------------

def test_cache_reused_when_unchanged(boot_calls):
    p = _profile({"f": ("1.0", _bins(3))})
    base = p.compute_baselines()
    assert len(boot_calls) == 2                      # one fit per axle
    assert base["front_tire_model"]["peak"] is not None
    assert p.tire_fit_cache["fingerprint"]
    p.compute_baselines()
    assert len(boot_calls) == 2                      # unchanged -> no refit
    # the cache is persisted: a reloaded profile does not refit either
    q = gp.GripProfile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.compute_baselines()["front_tire_model"] == base["front_tire_model"]
    assert len(boot_calls) == 2
    # a new lap changes the fit set -> refit
    extra = _bins(1, seed=9)[0]
    q.tire_files["f"]["laps"]["9"] = {"load_source": "x", "front": extra.to_dict(),
                                      "rear": extra.to_dict()}
    q.compute_baselines()
    assert len(boot_calls) == 4


def test_v2_shaped_model_keys():
    m = _profile({"f": ("1.0", _bins(3))}).compute_baselines()["front_tire_model"]
    assert 1.0 < m["peak"] < 8.0 and m["efficient_range"][0] <= m["peak"] <= m["efficient_range"][1]
    assert m["grip_measure"] == tb.MEASURE_MU and m["sample_count"] > 0
    assert 0.0 < m["confidence"] < 1.0
    assert m["prior_used"] == "none" and m["version_note"] is None and m["n_fit_laps"] == 3
    empty = gp.GripProfile(car_name="E").compute_baselines()["front_tire_model"]
    assert empty["peak"] is None and empty["confidence"] == 0.0 and empty["sample_count"] == 0


# -- versions ------------------------------------------------------------------

def test_version_key_orders_numerically_with_unknown_oldest():
    ks = sorted(["1.10", "1.9", "unknown", "0.9.1"], key=gp._version_key)
    assert ks == ["unknown", "0.9.1", "1.9", "1.10"]


def test_newest_version_selected_with_older_prior(boot_calls, monkeypatch):
    old, new = _bins(3, d0=1.2, seed=1), _bins(3, d0=1.8, seed=2)
    p = _profile({"old": ("1.9", old), "new": ("1.10", new)})
    m = p.compute_baselines()["front_tire_model"]
    assert m["game_version"] == "1.10" and m["n_fit_laps"] == 3
    fit = tb.LapBins.merge(new)
    own = p.tire_fit_cache["front"]["fit"]["params"]["D0"]
    assert own > 1.6                                  # the old version's 1.2 did not leak in
    # the newest version has 3 laps, so by Ruling 11b no prior is applied ...
    assert m["prior_used"] == "none"
    # ... but with the lap guard lifted the older version enters as the prior
    monkeypatch.setattr(tire_class, "PRIOR_MAX_LAPS", 10)
    m2 = p.compute_baselines()["front_tire_model"]
    assert m2["prior_used"] == "older_version" and m2["game_version"] == "1.10"
    assert p.tire_fit_cache["front"]["fit"]["params"]["D0"] < own     # pulled toward 1.2
    assert fit.n_samples() > 0


def test_mixed_versions_note(boot_calls):
    p = _profile({"a": ("1.1", _bins(2, seed=1)), "b": ("1.2", _bins(2, seed=2))})
    m = p.compute_baselines()["front_tire_model"]
    assert m["version_note"] == "mixed_versions" and m["n_fit_laps"] == 4
    assert m["game_version"] is None
    # one version below the threshold is not "mixed"
    single = _profile({"a": ("1.1", _bins(2, seed=1))}).compute_baselines()["front_tire_model"]
    assert single["version_note"] is None


def test_mixed_measures_fit_majority_only(boot_calls):
    lat = make_bins(n_laps=1, measure="lat", factors=())
    p = _profile({"a": ("1.0", _bins(3) + lat)})
    m = p.compute_baselines()["front_tire_model"]
    assert m["grip_measure"] == tb.MEASURE_MU and m["n_fit_laps"] == 3


# -- class prior ---------------------------------------------------------------

def _save(name, car_class, bins):
    gp.save_profile(_profile({"f": ("1.0", bins)}, name=name, car_class=car_class))


def test_class_prior_pools_same_class_only(monkeypatch):
    _save("A", "LMGT3", _bins(3, d0=1.4, seed=1))
    _save("B", "LMGT3", _bins(3, d0=1.6, seed=2))
    _save("H", "Hypercar", _bins(3, d0=0.8, seed=3))
    (gp.PROFILES_DIR / ".ingest_ledger.json").write_text("[]")
    prior = gp.class_prior_for("LMGT3", "A", "front")
    assert prior is not None and 1.45 < prior.params["D0"] < 1.75     # B alone, not A or H
    assert gp.class_prior_for("unknown", "A") is None
    assert gp.class_prior_for("Nope", "A") is None
    # cached per process: same profiles -> no second pooled fit
    calls = []
    orig = gp.fit_class_prior
    monkeypatch.setattr(gp, "fit_class_prior", lambda *a: calls.append(1) or orig(*a))
    gp.class_prior_for("LMGT3", "A", "front")
    assert calls == []
    _save("B", "LMGT3", _bins(4, d0=1.6, seed=2))                      # B gained a lap
    gp.class_prior_for("LMGT3", "A", "front")
    assert calls == [1]


def test_class_prior_skips_profiles_in_another_measure():
    _save("A", "LMGT3", _bins(3, d0=1.4, seed=1))
    _save("LAT", "LMGT3", make_bins(n_laps=3, measure="lat", factors=()))
    assert gp.class_prior_for("LMGT3", "A", "front", tb.MEASURE_MU) is None  # only LAT is left
    assert gp.class_prior_for("LMGT3", "A", "front", tb.MEASURE_LAT) is not None


def test_small_car_uses_class_prior_big_car_does_not(boot_calls):
    _save("B", "LMGT3", _bins(3, d0=1.6, seed=2))
    one = _profile({"f": ("1.0", _bins(1, seed=5))}, name="One")
    m = one.compute_baselines()["front_tire_model"]
    assert m["prior_used"] == "class"
    three = _profile({"f": ("1.0", _bins(3, seed=5))}, name="Three")
    assert three.compute_baselines()["front_tire_model"]["prior_used"] == "none"
    # an explicit prior dict is honoured without touching other profiles
    explicit = gp.class_prior_for("LMGT3", "One", "front")
    q = _profile({"f": ("1.0", _bins(1, seed=5))}, name="Q")
    q.refit_tyre_model(class_prior={"front": explicit, "rear": explicit})
    assert q.tire_fit_cache["front"]["prior_used"] == "class"


# -- size ------------------------------------------------------------------------

def test_profile_stores_bins_not_samples():
    # Worst case for the bins: uniform random state variables, so almost every sample is
    # its own cell (~145 B/sample, 11 MB for 25 laps x 3000 samples). Real laps cluster: the
    # Ferrari / Genesis HF files come to 40-73 KB per lap (task 9 report). The point here
    # is only that nothing per-sample is kept beyond one row per occupied cell.
    laps = make_bins(n_laps=5, factors=("load", "temp", "pressure"), seed=1, n_per_lap=3000)
    p = _profile({"f": ("1.0", laps)})
    d = p.to_dict()
    assert set(d) == {"car_name", "car_class", "lap_count", "abs_lockup_durations_s",
                      "tc_active_fractions", "peak_combined_g_values",
                      "peak_front_slip_angle_values", "peak_rear_slip_angle_values",
                      "max_wheel_slip_ratio_values", "tire_model_version",
                      "deflection_calibration", "tire_files", "tire_fit_cache"}
    rows = sum(len(l["front"]["rows"]) for l in d["tire_files"]["f"]["laps"].values())
    assert rows <= sum(b.n_samples() for b in laps)
    json.dumps(d)

# -- v3 outputs in the baselines (Task 10) ---------------------------------------

def test_version_note_says_when_the_newest_version_is_not_used_yet(boot_calls):
    # 3 laps of 1.9 fit; 2 laps of 1.10 are too few and were dropped silently before
    p = _profile({"old": ("1.9", _bins(3, seed=1)), "new": ("1.10", _bins(2, seed=2))})
    m = p.compute_baselines()["front_tire_model"]
    assert m["game_version"] == "1.9" and m["n_fit_laps"] == 3
    assert "1.10" in m["version_note"] and "2 lap" in m["version_note"]
    assert "not used yet" in m["version_note"]
    # the note is not part of the cached fit: adding the 3rd newer lap changes the fit set
    assert "version_note" in p.tire_fit_cache["front"]
    p.tire_files["new"]["laps"]["9"] = p.tire_files["new"]["laps"]["0"]
    m3 = p.compute_baselines()["front_tire_model"]
    assert m3["game_version"] == "1.10" and m3["version_note"] is None


def test_temp_pressure_load_key_carries_factor_effects(boot_calls):
    b = _profile({"f": ("1.0", _bins(3))}).compute_baselines()
    tpl = b["front_temp_pressure_load"]
    assert tpl["available"] is True and "load" in tpl["factor_effects"]
    assert "setup target" in tpl["note"]
    assert tpl["factor_effects"] == b["front_tire_model"]["factor_effects"]
    empty = gp.GripProfile(car_name="E").compute_baselines()
    assert empty["front_temp_pressure_load"]["available"] is False


def test_tire_unavailable_degrades_without_raising(boot_calls):
    p = _profile({"f": ("1.0", _bins(3))})
    b = p.compute_baselines(tire_unavailable="fit failed: RuntimeError")
    for axle in ("front", "rear"):
        m = b[f"{axle}_tire_model"]
        assert m["model"] == "binned_v2_fallback" and m["peak"] is None
        assert "fit failed" in m["note"]
    assert boot_calls == []                        # no fit was attempted
    assert b["tc_active_fraction"] is None         # the rest of the baselines are intact
