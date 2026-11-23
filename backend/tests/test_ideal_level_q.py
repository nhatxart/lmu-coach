"""Ideal lap, design §15 V2 (B1): the grip level of an `observed` axle is the
0.75 quantile of observed / predicted on its saturated set (a limit driver is
saturated where the local capacity dips, so the median reads low under place
noise); a `lower_bound` axle keeps the median of its top-20 % set; the per-file
trim uses the same 0.75 statistic. kb, kt and the engine stay at the median."""
import numpy as np
import pytest

from app import ideal_car as C
from app.ideal_state import FileReduction, State
from test_ideal_axle_fit import _car_samples, MU1, K, CC


def _spread(d, rng, sd):
    """Multiply the saturated samples' |gy| by a log-normal spread: their
    observed / predicted ratio then has known quantiles."""
    d = dict(d)
    d["gy"] = d["gy"] * np.exp(sd * rng.standard_normal(len(d["gy"])))
    return d


def test_observed_level_at_the_075_quantile():
    rng = np.random.default_rng(21)
    sd = 0.05
    d = _spread(_car_samples(rng, 4000, 4000), rng, sd)
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=d["gf"])
    assert C.LEVEL_Q == 0.75
    lam = np.full((4, 1), 1.2)
    for j in (0, 1):
        assert fits[j].status == "observed"
        got = C.axle_capacity(lam, np.array([40.0]), [f.mu1 for f in fits], [f.k for f in fits],
                              [f.c for f in fits])[j][0]
        true = C.axle_capacity(lam, np.array([40.0]), MU1, K, CC)[j][0]
        # q 0.75 of a log-normal ratio with sd 0.05 sits 0.67 sd = 3.4 % above the median
        assert got / true == pytest.approx(np.exp(0.6745 * sd), abs=0.012)


def test_lower_bound_keeps_the_median():
    rng = np.random.default_rng(22)
    d = _car_samples(rng, 4000, 100)
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=d["gf"])
    assert fits[1].status == "lower_bound"
    # the top-20 % set's own median: the fit reproduces it (median of observed / predicted = 1)
    base = (np.abs(d["gy"]) > 0.45 * np.percentile(np.abs(d["gy"]), 99)) & (d["v"] > 15)
    r = np.abs(d["gy"]) / d["lam"][[2, 3]].mean(0)
    m = base & (r >= np.quantile(r[base], 0.8))
    A = C.axle_capacity(d["lam"], d["v"], [f.mu1 for f in fits], [f.k for f in fits], [f.c for f in fits])[1]
    assert np.median(np.abs(d["gy"])[m] / A[m]) == pytest.approx(1.0, abs=1e-6)


def test_trim_at_the_075_quantile():
    rng = np.random.default_rng(23)
    n = 3000
    v = np.full(n, 40.0)
    z = np.zeros(n)
    lm = C.LoadModel(static=np.ones(4), coef=np.zeros((4, 5)), source="susp")
    car = C.CarModel(load=lm, mu1=[1.4, 1.4], k=[0, 0], c=[0, 0], s_front=0.5, kb=1, kt=1, d0=0, d2=0,
                     power_v=[0, 100], power_g=[0.5, 0.5], corner_drag=0.0)
    gy = 1.4 * np.exp(0.03 + 0.05 * rng.standard_normal(n))       # the file's level is 3 % up, spread 5 %
    gf = np.full((4, n), 0.1)
    st = State(lap_number=1, lap_time_s=60, v=v, gx=z, gy=gy, gz=z, grade=z, thr=z, brk=z, abs=z, tc=z, yawacc=z,
               gf=gf)
    red = FileReduction(file="f", mtime=0, size=0, meta={}, track="", car_model="", car_class="", game_version="",
                        load_source="susp", grade_source="", laps=[st])
    ax = [C.AxleFit(mu1=1.4, k=0, c=0, status="observed", saturation="grip_fraction", knee_deg=None, n_sat=n,
                    ratio_sd=0.05) for _ in (0, 1)]
    _, trims = C.axle_trim(car, ax, red)
    x = np.log(gy[np.abs(gy) > 0.45 * np.percentile(np.abs(gy), 99)] / 1.4)
    assert trims[0]["t"] == pytest.approx(float(np.quantile(x, 0.75)), abs=1e-9)
