"""Ideal lap, design T4: the envelope never goes below the driven speed,
equals the model where the model is faster, and gain_min <= gain <= gain_max."""
import numpy as np

from app import ideal_solver as SV
from test_ideal_solver import _car, _line


def test_envelope_and_levels():
    rng = np.random.default_rng(11)
    ks = np.convolve(rng.normal(0, 0.03, 1500), np.ones(40) / 40, mode="same")
    ln = _line(ks)
    car = _car(mu=(1.4, 1.55), sf=0.46, power=0.6)
    # a "driven" lap: the model at 97 % grip, with 3 % speed noise
    drv = SV.solve(car.scaled(0.97), ln)["v"] * (1 + 0.03 * np.convolve(rng.normal(0, 1, 1500), np.ones(25) / 5, "same"))
    ln["v"] = drv
    out = SV.levels(car, ln, beta=0.02)
    env, v_m = out["exp"]["env"]["v"], out["exp"]["solve"]["v"]
    assert (env >= drv).all()
    np.testing.assert_array_equal(env[v_m >= drv], v_m[v_m >= drv])
    assert ((v_m < drv) == out["exp"]["env"]["from_ref"]).all()
    assert out["gain_min"] <= out["gain"] <= out["gain_max"]
    assert out["gain_min"] >= 0.0                     # never slower than the reference lap
    # model_only is the plain level-1 model; the envelope is never slower than it
    assert out["model_only_time"] == out["exp"]["solve"]["time"]
    assert out["exp"]["env"]["time"] <= out["model_only_time"] + 1e-9
    assert 0.0 <= out["under_driven_share"] <= 1.0
