"""Ideal lap, design T1: the per-file load model recovers known coefficients,
including the |gy| term (F1), from wheel loads with 3 % noise."""
import numpy as np

from app.ideal_state import State
from app import ideal_car

# Per wheel FL, FR, RL, RR: static c0 and (a v^2, b gx, c gy, d gz, e |gy|).
C0 = np.array([3000.0, 3100.0, 4200.0, 4100.0])
COEF = np.array([
    [1.2e-4, -0.12, +0.30, 0.80, +0.10],
    [1.1e-4, -0.11, -0.28, 0.75, +0.12],
    [1.6e-4, +0.14, +0.25, 0.90, -0.15],
    [1.5e-4, +0.15, -0.26, 0.85, -0.14],
])


def _lap(rng, n=4000):
    v = rng.uniform(20, 80, n)
    gx = rng.uniform(-1.5, 0.8, n)
    gy = rng.uniform(-2.0, 2.0, n)
    gz = rng.uniform(-0.5, 0.6, n)
    X = np.stack([v * v, gx, gy, gz, np.abs(gy)], axis=-1)
    L = C0[:, None] * (1 + COEF @ X.T)
    L = L * (1 + 0.03 * rng.standard_normal(L.shape))
    z = np.zeros(n)
    return State(lap_number=0, lap_time_s=100.0, v=v, gx=gx, gy=gy, gz=gz, grade=z,
                 thr=z, brk=z, abs=z, tc=z, yawacc=z, susp=L)


def test_load_model_recovers_coefficients_with_abs_gy_term():
    rng = np.random.default_rng(1)
    laps = [_lap(rng) for _ in range(6)]
    lm = ideal_car.fit_load_model(laps, "susp")
    assert lm.source == "susp"
    np.testing.assert_allclose(lm.static, C0, rtol=0.05)
    np.testing.assert_allclose(lm.coef, COEF, rtol=0.05)
    # held-out R^2 (fit even laps, score odd laps) is reported per wheel
    assert len(lm.r2) == 4 and all(r > 0.9 for r in lm.r2)


def test_lam_is_ratio_to_static_and_floored():
    rng = np.random.default_rng(2)
    lm = ideal_car.fit_load_model([_lap(rng) for _ in range(4)], "susp")
    lam = lm.lam(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(lam[:, 0], 1.0, atol=1e-9)
    # an absurd state cannot give a negative load ratio
    lam = lm.lam(np.array([0.0]), np.array([-40.0]), np.array([0.0]), np.array([0.0]))
    assert (lam >= 0.02).all()


def test_load_model_round_trips_through_dict():
    rng = np.random.default_rng(3)
    lm = ideal_car.fit_load_model([_lap(rng) for _ in range(4)], "susp")
    back = ideal_car.LoadModel.from_dict(lm.to_dict())
    np.testing.assert_allclose(back.coef, lm.coef)
    np.testing.assert_allclose(back.static, lm.static)
    assert back.source == lm.source
