"""Ideal lap, design §15 V1 (A1, measured power) and §15.x 2: T11.
Where the reference lap was flat out (throttle > 98 %, brake < 1 %, TC not
active), the acceleration it drove is a measurement of the engine at that place:
p_off(s) = gx_ref + grade - power(v_ref, gy_ref) there (5-point circular mean),
0 elsewhere, and the solver's power is curve(v) - cc gy^2 + p_off(s)."""
import numpy as np
import pytest

from app import ideal_solver as SV
from app.ideal_state import G
from test_ideal_solver import _car, _line

PV = [0.0, 20.0, 40.0, 60.0, 90.0]
PG = [0.60, 0.60, 0.42, 0.25, 0.10]          # below the traction limit (0.75 g): power binds


def _power_car():
    c = _car(mu=(1.5, 1.5), power=0.5)
    c.power_v, c.power_g = list(PV), list(PG)
    return c


def _straight_world(bump=0.05):
    """Hairpin (R 15 m, 40 m) + 900 m straight + hairpin + 900 m straight. The
    driven lap is the model's own, except that on the first straight the car
    accelerates 5 % harder over 200 m (a draft, a better gear)."""
    car = _power_car()
    ds, nh, ns = 2.0, 20, 450
    ks = np.concatenate([np.full(nh, 1 / 15), np.zeros(ns), np.full(nh, 1 / 15), np.zeros(ns)])
    ln = _line(ks, ds)
    base = SV.solve(car, ln, clamp=False)
    v = base["v"].copy()
    power_part = np.flatnonzero(base["binding"][nh:nh + ns] == "power") + nh
    first, last = power_part[0], power_part[-1]
    bumped = np.zeros(len(v), bool)
    bumped[first + 60:first + 160] = True
    for i in range(first, last):
        a = np.interp(v[i], PV, PG) * (1 + (bump if bumped[i] else 0.0))
        v[i + 1] = np.sqrt(v[i] ** 2 + 2 * G * a * ds)
    v[last + 1:nh + ns] = np.minimum(v[last + 1:nh + ns], base["v"][last + 1:nh + ns])   # it brakes as the model
    thr = np.where(base["binding"] == "power", 100.0, 0.0)
    thr[first:last + 1] = 100.0
    ln = dict(ln, v=v, thr=thr, brk=np.where(base["binding"] == "brake", 100.0, 0.0), tc=np.zeros(len(v)))
    return car, ln, slice(first, last - 40)          # compare on the flat-out part, clear of the braking


def test_t11_measured_power_reproduces_the_driven_straight():
    car, ln, part = _straight_world()
    plain = SV.solve(car, ln, clamp=False)
    off = np.abs(plain["v"][part] - ln["v"][part]) * 3.6
    assert off.max() > 0.3                            # the median curve cannot follow the bump
    ln["p_off"] = SV.power_offset(car, ln)
    meas = SV.solve(car, ln, clamp=False)
    err = np.abs(meas["v"][part] - ln["v"][part]) * 3.6
    assert err.max() <= 0.3, err.max()
    assert set(meas["binding"][part]) == {"power"}


def test_power_offset_only_where_flat_out_and_not_on_tc():
    car, ln, part = _straight_world()
    ln["tc"][part.start:part.start + 30] = 1.0         # traction control active there
    off = SV.power_offset(car, ln)
    full = (ln["thr"] > 98) & (ln["brk"] < 1) & (ln["tc"] < 0.5)
    assert (off[~full] == 0).all()
    assert (off[part.start:part.start + 30] == 0).all()
    assert np.abs(off[full]).max() > 0.01             # the bump is measured
    # a line without p_off solves exactly as before
    plain = SV.solve(car, dict(ln), clamp=False)
    again = SV.solve(car, {k: v for k, v in ln.items() if k != "p_off"}, clamp=False)
    np.testing.assert_allclose(plain["v"], again["v"])
