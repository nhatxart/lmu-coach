"""Class-pooled MF prior (tire_class)."""

import numpy as np
import pytest

from app import tire_bins as tb
from app import tire_mf as mf
from app import tire_class as tc
from synthetic_tyre import make_bins

LOAD_ONLY = ("load",)


def _car(d0, seed, n_laps=3, **kw):
    # Only the load factor is generated: it is the only one ENABLED_FACTORS fits.
    return make_bins(n_laps=n_laps, true_params={"D0": d0}, factors=LOAD_ONLY, seed=seed, **kw)


def _heldout_rmse(fit, bins):
    """Count-weighted RMSE of `fit` on the cornering cells (slip >= 1.5) of `bins`."""
    a = tb.cell_arrays(bins)
    ok = (a["slip"] >= 1.5) & (a["n"] >= 3)
    state = {k: a[k][ok] for k in ("load_ratio", "temp_delta_c", "pressure_kpa",
                                   "wear", "camber_deg")}
    err = mf.predict(fit, a["slip"][ok], state) - a["mu_mean"][ok]
    return float(np.sqrt(np.sum(a["n"][ok] * err ** 2) / a["n"][ok].sum()))


def test_class_prior_between_cars():
    prior = tc.fit_class_prior({"a": _car(1.4, 1), "b": _car(1.6, 2)}, tb.MEASURE_MU)
    assert prior is not None
    assert 1.4 < prior.params["D0"] < 1.6


def test_equal_car_weighting():
    # Car A has 10x the samples of car B. Pooled by raw counts D0 would sit near 1.4;
    # with equal car weighting it sits near the midpoint of the two cars.
    cars = {"a": _car(1.4, 1, n_laps=10), "b": _car(1.6, 2, n_laps=1)}
    before = {k: [l.n_samples() for l in v] for k, v in cars.items()}
    prior = tc.fit_class_prior(cars, tb.MEASURE_MU)
    assert abs(prior.params["D0"] - 1.5) < 0.05
    # The inputs are not mutated by the scaling.
    assert {k: [l.n_samples() for l in v] for k, v in cars.items()} == before


def test_small_car_borrows_from_prior():
    # Truth D0 1.8 for everybody; the small car sees one short lap that never reaches the
    # peak (slip < 3.2 deg), so on its own it must extrapolate. Three other cars supply
    # the prior. Held-out = a large independent sample over the full slip range.
    others = {f"o{i}": make_bins(n_laps=4, factors=LOAD_ONLY, seed=10 + i) for i in range(3)}
    prior = tc.fit_class_prior(others, tb.MEASURE_MU)
    errs = {0.0: [], 1.0: []}
    for seed in (2, 3, 4):
        small = tb.LapBins.merge(make_bins(n_laps=1, factors=LOAD_ONLY, seed=seed,
                                           n_per_lap=500, slip_range=(0.0, 3.2), noise=0.08))
        test = tb.LapBins.merge(make_bins(n_laps=2, factors=LOAD_ONLY, seed=100 + seed))
        for w in errs:
            fit = mf.fit_mf(small, prior=prior, prior_weight=w)
            assert fit is not None
            errs[w].append(_heldout_rmse(fit, test))
    print("\nheld-out RMSE by weight:", {w: np.round(e, 3) for w, e in errs.items()})
    assert np.mean(errs[1.0]) < np.mean(errs[0.0])


def test_prior_used_label():
    laps = make_bins(n_laps=2, factors=LOAD_ONLY, seed=5)
    prior = tc.fit_class_prior({"a": _car(1.8, 1), "b": _car(1.8, 2)}, tb.MEASURE_MU)
    assert tc.fit_car_with_prior(laps, prior, n_boot=5)["prior_used"] == "class"
    assert tc.fit_car_with_prior(laps, prior, n_boot=5,
                                 prior_kind="older_version")["prior_used"] == "older_version"
    assert tc.fit_car_with_prior(laps, None, n_boot=5)["prior_used"] == "none"
    assert tc.fit_car_with_prior(laps, prior, weight=0.0, n_boot=5)["prior_used"] == "none"
    # A prior fit_mf rejects (other measure) must not be reported as used.
    lat = tc.fit_class_prior({"a": make_bins(n_laps=3, measure="lat", factors=())},
                             tb.MEASURE_LAT)
    out = tc.fit_car_with_prior(laps, lat, n_boot=5)
    assert out["prior_used"] == "none" and out["fit"] is not None
    assert set(out) >= {"fit", "peak_slip_ci90", "peak_mu_ci90", "n_boot_ok", "confidence"}
    with pytest.raises(ValueError):
        tc.fit_car_with_prior(laps, prior, prior_kind="bogus")


def test_prior_not_applied_from_three_laps(monkeypatch):
    # Ruling 11b: held-out scores showed no benefit (even harm) once a car has its own
    # laps, so both prior kinds are switched off at PRIOR_MAX_LAPS laps.
    assert tc.PRIOR_MAX_LAPS == 3 and tc.CLASS_PRIOR_WEIGHT == 0.3
    prior = tc.fit_class_prior({"a": _car(1.8, 1), "b": _car(1.8, 2)}, tb.MEASURE_MU)
    two = make_bins(n_laps=2, factors=LOAD_ONLY, seed=5)
    three = make_bins(n_laps=3, factors=LOAD_ONLY, seed=5)
    for kind in tc.PRIOR_KINDS:
        assert tc.fit_car_with_prior(two, prior, n_boot=3, prior_kind=kind)["prior_used"] == kind
        assert tc.fit_car_with_prior(three, prior, n_boot=3, prior_kind=kind)["prior_used"] == "none"
    # the guard follows the module constant (grip_profile tests lift it to reach older_version)
    monkeypatch.setattr(tc, "PRIOR_MAX_LAPS", 4)
    assert tc.fit_car_with_prior(three, prior, n_boot=3)["prior_used"] == "class"


def test_class_prior_skips_other_measure_and_empty():
    cars = {"a": _car(1.8, 1), "lat": make_bins(n_laps=3, measure="lat", factors=()),
            "none": [tb.LapBins(tb.MEASURE_MU)]}
    assert tc.fit_class_prior(cars, tb.MEASURE_MU).measure == tb.MEASURE_MU
    assert tc.fit_class_prior({"none": [tb.LapBins(tb.MEASURE_MU)]}, tb.MEASURE_MU) is None


# ---------------------------------------------------------------- collapse tool ----------

def _tool():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tools" / "class_collapse.py"
    spec = importlib.util.spec_from_file_location("class_collapse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collapse_tool_scores_each_car_against_the_others_only():
    cc = _tool()
    cars = {f"c{i}": make_bins(n_laps=3, factors=LOAD_ONLY, seed=30 + i) for i in range(3)}
    cars["solo"] = make_bins(n_laps=1, factors=LOAD_ONLY, seed=40)
    res = cc.analyse_axle(cars)
    assert res["measure"] == tb.MEASURE_MU and res["prior"] is not None
    assert res["cars"]["solo"]["score"]["folds"] == 0      # one lap: nothing to train on
    sc = res["cars"]["c0"]["score"]
    assert sc["folds"] == 3
    # Same truth for every car: any weight predicts the held-out lap well.
    assert all(sc["weights"][w]["r2"] > 0.9 for w in cc.WEIGHTS)
    assert res["best"]["weight"] in cc.WEIGHTS and res["best"]["n_cars"] == 3
    text = cc.format_axle("TEST", "front", res)
    assert "| c0 |" in text and "best weight (front)" in text


def test_collapse_tool_leaves_out_other_measure_laps():
    cc = _tool()
    cars = {"a": make_bins(n_laps=3, factors=LOAD_ONLY, seed=1),
            "b": make_bins(n_laps=3, factors=LOAD_ONLY, seed=2),
            "lat": make_bins(n_laps=3, measure="lat", factors=())}
    res = cc.analyse_axle(cars)
    assert "lat" not in res["cars"] and res["dropped"] == {"lat": 3}
