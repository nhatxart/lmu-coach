"""Per-lap binned sufficient statistics (tire_bins)."""

import json
import math

import numpy as np
import pytest

from app import tire_bins as tb


def _s(n, slip=2.0, lat=1.5, **over):
    d = {
        "slip_angle_deg": np.full(n, slip),
        "lateral_g": np.full(n, lat),
        "load_ratio": np.full(n, 1.0),
        "temp_delta_c": np.full(n, 0.0),
        "tire_pressure_kpa": np.full(n, 160.0),
        "wear": np.full(n, 0.1),
        "camber_deg": np.full(n, 2.0),
        "carcass_delta_c": np.full(n, 1.0),
        "long_g": np.full(n, -0.5),
        "true_mu": np.full(n, np.nan),
    }
    for k, v in over.items():
        d[k] = v
    return d


def _bin(s):
    return tb.bin_axle_samples(s, np.ones(len(s["slip_angle_deg"]), bool))


def test_round_trip_preserves_cells():
    rng = np.random.default_rng(1)
    n = 500
    s = _s(n, slip=rng.uniform(0, 8, n), lat=rng.uniform(0.5, 2, n),
           load_ratio=rng.uniform(0.8, 1.3, n), true_mu=rng.uniform(1, 2, n))
    a = _bin(s)
    b = tb.LapBins.from_dict(json.loads(json.dumps(a.to_dict())))
    assert b.measure == a.measure and set(b.cells) == set(a.cells)
    assert b.n_samples() == a.n_samples() == n
    for k, c in a.cells.items():
        assert b.cells[k].n == c.n
        assert b.cells[k].sum_mu == pytest.approx(c.sum_mu, rel=1e-5)
        assert b.cells[k].sum_load_ratio == pytest.approx(c.sum_load_ratio, rel=1e-5)


def test_merge_sums_counts():
    a = _bin(_s(10, load_ratio=np.full(10, 1.0)))
    b = _bin(_s(30, load_ratio=np.full(30, 1.04)))
    m = tb.LapBins.merge([a, b])
    assert len(m.cells) == 1
    c = next(iter(m.cells.values()))
    assert c.n == 40
    assert c.sum_load_ratio / c.n_load_ratio == pytest.approx((10 * 1.0 + 30 * 1.04) / 40)


def test_merge_rejects_mixed_measures():
    a = _bin(_s(10))
    b = _bin(_s(10, load_ratio=None))
    assert a.measure == "mu_load_normalized" and b.measure == "lateral_g"
    with pytest.raises(ValueError):
        tb.LapBins.merge([a, b])


def test_nan_state_goes_to_unknown_bin():
    a = _bin(_s(10, wear=np.full(10, np.nan)))
    key = next(iter(a.cells))
    assert key[4] == -999
    arr = tb.cell_arrays(a)
    assert np.isnan(arr["wear"][0])


def test_cell_means_are_state_means_not_centres():
    lr = np.array([1.03, 1.07] * 5)
    a = _bin(_s(10, load_ratio=lr))
    assert len(a.cells) == 1
    assert tb.cell_arrays(a)["load_ratio"][0] == pytest.approx(1.05)


def test_true_mu_accumulated_only_where_finite():
    t = np.array([1.5, np.nan, 2.5, np.nan])
    a = _bin(_s(4, true_mu=t))
    arr = tb.cell_arrays(a)
    assert arr["n"][0] == 4 and arr["true_n"][0] == 2
    assert arr["true_mean"][0] == pytest.approx(2.0)


def test_json_size_reasonable():
    rng = np.random.default_rng(2)
    n = 10_000
    # Realistic spread: pressure/wear/camber barely move within one lap.
    s = _s(n, slip=rng.uniform(0, 8, n), lat=rng.uniform(0.2, 2.2, n),
           load_ratio=rng.uniform(0.9, 1.2, n), temp_delta_c=rng.uniform(-5, 5, n),
           tire_pressure_kpa=rng.uniform(160, 164, n), wear=rng.uniform(0.10, 0.14, n),
           camber_deg=rng.uniform(2.0, 2.4, n), true_mu=rng.uniform(1, 2, n))
    assert len(json.dumps(_bin(s).to_dict())) < 200_000


def test_measure_selection_and_drops():
    n = 10
    lr = np.full(n, np.nan)
    lr[:6] = 1.0  # 60 % have load -> mu, keep only those
    a = _bin(_s(n, load_ratio=lr))
    assert a.measure == "mu_load_normalized" and a.n_samples() == 6
    lr[:4] = 1.0
    lr[4:] = np.nan  # 40 % -> lateral_g for all
    a = _bin(_s(n, load_ratio=lr))
    assert a.measure == "lateral_g" and a.n_samples() == n
    slip = np.array([1.0, 15.0, 20.0, np.nan] + [2.0] * 6)
    assert _bin(_s(n, slip=slip)).n_samples() == 7


def test_mu_uses_load_floor():
    a = _bin(_s(4, lat=1.0, load_ratio=np.full(4, 0.1)))
    assert tb.cell_arrays(a)["mu_mean"][0] == pytest.approx(1.0 / 0.2)


def test_coverage():
    s = _s(30, slip=2.1)  # off a bin edge so float division can't tip it
    cov = tb.coverage(_bin(s))
    assert cov["slip_bins_reached"] == [5]
    assert sum(cov["counts"].values()) == 30
