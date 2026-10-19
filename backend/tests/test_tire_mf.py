"""Factorized Magic Formula fit (tire_mf)."""

import json
import time

import numpy as np
import pytest

from app import tire_bins as tb
from app import tire_mf as mf
from synthetic_tyre import GEN_REF, PT_INTERCEPT, PT_SLOPE, TRUE_PARAMS, make_bins, true_mu

GEN_STATE = {"load_ratio": GEN_REF["load_ratio"], "temp_delta_c": GEN_REF["temp_delta_c"],
             "pressure_kpa": PT_INTERCEPT + PT_SLOPE * GEN_REF["temp_delta_c"],
             "wear": GEN_REF["wear"], "camber_deg": GEN_REF["camber_deg"]}


def _true_peak(params):
    a = np.arange(0, 15, 0.01)
    z = np.zeros_like(a)
    mu = true_mu({**TRUE_PARAMS, **params}, a, z + GEN_REF["load_ratio"],
                 z + GEN_REF["temp_delta_c"], z, z + GEN_REF["wear"],
                 z + GEN_REF["camber_deg"], ("load", "temp", "pressure"))
    return float(a[np.argmax(mu)]), float(mu.max())


def _merged(**kw):
    return tb.LapBins.merge(make_bins(**kw))


def test_recovers_known_params():
    t0 = time.perf_counter()
    fit = mf.fit_mf(_merged())
    wall = time.perf_counter() - t0
    assert fit is not None
    print("\nwall %.2fs" % wall)
    for k, v in TRUE_PARAMS.items():
        if k in fit.params:
            print(f"  {k:4s} true {v:9.4f} fitted {fit.params[k]:9.4f}")
    assert fit.params["D0"] == pytest.approx(TRUE_PARAMS["D0"], rel=0.03)
    slip, mu = mf.peak_at(fit, GEN_STATE)
    tslip, tmu = _true_peak({})
    assert abs(slip - tslip) < 0.2
    assert mu == pytest.approx(tmu, rel=0.03)
    assert fit.factors_status["load"] == "fitted"
    assert fit.factors_status["temp"] == "fitted"
    assert fit.factors_status["pressure"] == "fitted"
    assert fit.factors_status["wear"] == "not_identifiable"
    assert fit.factors_status["camber"] == "not_identifiable"
    assert fit.peak_observed is True
    assert fit.rmse < 0.06


def test_recovers_wear_and_camber_factors():
    fit = mf.fit_mf(_merged(factors=mf.ALL_FACTORS))
    assert all(s == "fitted" for s in fit.factors_status.values())
    assert fit.params["kDW"] == pytest.approx(TRUE_PARAMS["kDW"], abs=0.15)
    assert fit.params["kDC"] == pytest.approx(TRUE_PARAMS["kDC"], abs=0.01)
    assert fit.params["D0"] == pytest.approx(TRUE_PARAMS["D0"], rel=0.03)


def test_unidentifiable_factor_fixed_neutral():
    bins = tb.LapBins.merge(make_bins(factors=("load",), td_range=(-32.0, -28.0)))
    fit = mf.fit_mf(bins, factors=("load", "temp"))
    assert fit.factors_status["temp"] == "not_identifiable"
    assert fit.factors_status["load"] == "fitted"
    slip = np.linspace(0.5, 7, 14)
    ref = mf.predict(fit, slip)
    hot = mf.predict(fit, slip, {"temp_delta_c": 40.0})
    assert np.array_equal(ref, hot)


def test_disabled_factor_status():
    fit = mf.fit_mf(_merged(), factors=("load",))
    assert fit.factors_status["load"] == "fitted"
    assert fit.factors_status["temp"] == "disabled"
    assert fit.params["kDP"] == 0.0 and fit.params["kBP"] == 0.0


def test_fit_measure_never_mixed():
    bins = tb.LapBins.merge(make_bins(measure="lat", factors=()))
    assert bins.measure == tb.MEASURE_LAT
    fit = mf.fit_mf(bins)
    assert fit is not None and fit.measure == "lateral_g"
    assert mf.fit_mf(_merged()).measure == "mu_load_normalized"


def test_returns_none_on_few_cells():
    bins = tb.LapBins.merge(make_bins(n_laps=1, n_per_lap=40))
    assert mf.fit_mf(bins) is None


def test_returns_none_on_narrow_slip_span():
    bins = tb.LapBins.merge(make_bins(slip_range=(0.0, 2.0)))
    assert mf.fit_mf(bins) is None


def test_nan_state_is_neutral_not_poison():
    # Wipe the pressure of one lap: those cells must contribute neutrally, no NaN residuals.
    laps = make_bins()
    for c in laps[0].cells.values():
        c.sum_pressure_kpa, c.n_pressure_kpa = 0.0, 0
    fit = mf.fit_mf(tb.LapBins.merge(laps))
    assert fit is not None and np.isfinite(fit.rmse)
    assert np.all(np.isfinite(list(fit.params.values())))


def test_prior_pulls_toward_prior():
    prior = mf.fit_mf(_merged(true_params={"B0": 0.3}, seed=5))
    p_slip = mf.peak_at(prior)[0]
    one_lap = tb.LapBins.merge(make_bins(n_laps=1, seed=2))
    free = mf.fit_mf(one_lap)
    pulled = mf.fit_mf(one_lap, prior=prior, prior_weight=10.0)
    d_free = abs(mf.peak_at(free)[0] - p_slip)
    d_pull = abs(mf.peak_at(pulled)[0] - p_slip)
    assert d_pull < d_free


def test_true_target():
    bins = _merged(true_scale=0.9, factors=("load", "temp"))
    proxy = mf.fit_mf(bins, factors=("load", "temp"))
    true = mf.fit_mf(bins, target="true", factors=("load", "temp"))
    assert true is not None
    assert true.peak_mu == pytest.approx(0.9 * proxy.peak_mu, rel=0.04)


def test_true_target_none_without_true_cells():
    assert mf.fit_mf(_merged(), target="true") is None


def test_round_trip():
    fit = mf.fit_mf(_merged())
    d = json.loads(json.dumps(fit.to_dict()))
    back = mf.MFFit.from_dict(d)
    assert back == fit
    slip = np.linspace(0, 8, 20)
    assert np.allclose(mf.predict(back, slip), mf.predict(fit, slip))


def test_predict_uses_raw_pressure_and_temp():
    fit = mf.fit_mf(_merged())
    slip = np.array([3.0])
    hi = mf.predict(fit, slip, {"pressure_kpa": 175.0})
    lo = mf.predict(fit, slip, {"pressure_kpa": 140.0})
    # kDP is negative in the truth: more pressure (at fixed temperature) -> less grip.
    assert hi[0] < lo[0]
    arr = mf.predict(fit, np.array([3.0, 3.0]), {"load_ratio": np.array([1.0, 1.5])})
    assert arr[0] > arr[1]  # kDL < 0
