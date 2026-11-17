"""Ideal lap, the saturation clamp (design §5.4, §13 R1, §14 R7): T4's round-1
and round-2 cases. Where the reference lap's tyres were measured saturated in
lateral-dominant states, the corner limit is the driven speed."""
import numpy as np
import pytest

from app import ideal_solver as SV
from app.ideal_car import AxleFit
from app.ideal_state import State, GRID_HZ
from test_ideal_solver import _car, _line

G = 9.81


def _axles(sat="grip_fraction", knee=(4.0, 4.0)):
    return [AxleFit(mu1=1.4, k=0, c=0, status="observed", saturation=sat, knee_deg=knee[j], n_sat=1000,
                    ratio_sd=0.05) for j in (0, 1)]


def _state(n, gy, gx=0.0, gf_front=0.6, gf_rear=0.6, slip=None):
    z = np.zeros(n)
    gf = np.vstack([np.full(n, gf_front), np.full(n, gf_front), np.full(n, gf_rear), np.full(n, gf_rear)])
    return State(lap_number=1, lap_time_s=n / GRID_HZ, v=np.full(n, 30.0), gx=z + gx, gy=np.asarray(gy, float),
                 gz=z, grade=z, thr=z, brk=z, abs=z, tc=z, yawacc=z, gf=gf, slip=slip)


def test_saturated_ref_lateral_dominant_runs_only():
    n = 500
    gy = np.full(n, 1.5)
    st = _state(n, gy)
    st.gf[:2, 100:200] = 0.1              # front saturated 2 s: kept
    st.gf[2:, 300:305] = 0.1              # rear 0.1 s: shorter than 0.2 s, dropped
    st.gf[:2, 400:450] = 0.1              # front saturated, but braking hard: not lateral-dominant
    st.gx[400:450] = -1.2
    sat, ax = SV.saturated_ref(st, _car(), _axles(), per_axle=True)
    assert sat[100:200].all() and ax[0, 100:200].all() and not ax[1, 100:200].any()
    assert not sat[300:305].any()
    assert not sat[400:450].any()
    assert sat.sum() == 100


def test_saturated_ref_by_knee_without_grip_fraction():
    n = 300
    slip = np.zeros((4, n))
    slip[2:, 50:150] = 5.0                 # rear past its knee (4 deg) for 2 s
    slip[:2, 200:260] = 9.0                # front beyond knee + 4 deg: not in the set
    st = _state(n, np.full(n, 1.5), slip=slip)
    st.gf = None
    sat, ax = SV.saturated_ref(st, _car(), _axles(sat="knee"), per_axle=True)
    assert sat[50:150].all() and ax[1, 50:150].all() and sat.sum() == 100


def _corner_line(R=100.0, n_straight=300, n_corner=150):
    ks = np.concatenate([np.zeros(n_straight), np.full(n_corner, 1 / R), np.zeros(n_straight)])
    return _line(ks), slice(n_straight, n_straight + n_corner)


def test_clamped_points_hold_the_driven_speed():
    ln, corner = _corner_line()
    car = _car(mu=(1.4, 1.6), power=0.5)
    v_model = SV.solve(car, ln)["v"]
    ln["v"] = v_model * 0.97                      # the driver: 3 % slower everywhere
    sat = np.zeros(len(ln["ks"]), bool)
    sat[corner] = True
    ax = np.vstack([sat, np.zeros_like(sat)])     # the front was saturated
    SV.apply_clamp(ln, sat, ax)
    sol = SV.solve(car, ln)
    env = SV.envelope(sol["v"], ln["v"], ln["ds"])
    np.testing.assert_allclose(env["v"][corner], ln["v"][corner])
    assert (sol["v"][corner] <= ln["v"][corner] + 1e-9).all()
    assert set(sol["binding"][corner]) == {"measured"}
    out = SV.levels(car, ln, beta=0.02)
    assert out["clamped_share"] == pytest.approx(sat.mean())
    assert out["gain"] < out["gain_unclamped"]
    # the clamp does not scale with the grip level
    np.testing.assert_allclose(out["max"]["solve"]["v"][corner], out["min"]["solve"]["v"][corner])
    # the plain model stays unclamped
    assert out["model_only_time"] == pytest.approx(SV.solve(car, ln, clamp=False)["time"])
    # the ceiling is not clamped
    assert "measured" not in set(SV.solve(car, ln, "ceiling")["binding"])


def test_no_saturated_point_gives_the_old_envelope():
    ln, _ = _corner_line()
    car = _car(mu=(1.4, 1.6), power=0.5)
    ln["v"] = SV.solve(car, ln)["v"] * 0.97
    old = SV.levels(car, ln, beta=0.02)
    SV.apply_clamp(ln, np.zeros(len(ln["ks"]), bool), np.zeros((2, len(ln["ks"])), bool))
    new = SV.levels(car, ln, beta=0.02)
    for k in ("gain_min", "gain", "gain_max"):
        assert new[k] == pytest.approx(old[k])
    assert new["clamped_share"] == 0.0


def test_clamp_conflict_only_when_the_other_axle_limits_below_the_driven_speed():
    """§14 R7: front measured saturated at its own limit; the rear model limit
    above v_ref is no conflict; the rear mu1 cut 20 % puts it below v_ref."""
    ln, corner = _corner_line()
    car = _car(mu=(1.4, 1.6), power=0.5)
    ln["v"] = SV.solve(car, ln)["v"]               # the driver at the front limit
    sat = np.zeros(len(ln["ks"]), bool)
    sat[corner] = True
    SV.apply_clamp(ln, sat, np.vstack([sat, np.zeros_like(sat)]))
    ok = SV.clamp_conflict(car, ln)
    assert ok["rear"]["share"] == 0.0 and ok["rear"]["n"] == sat.sum()
    assert ok["front"]["n"] == 0
    weak = _car(mu=(1.4, 1.6 * 0.8), power=0.5)
    bad = SV.clamp_conflict(weak, ln)
    assert bad["rear"]["share"] > 0.9
    # and the result keeps v_I == v_ref at the clamped points (decision R7)
    sol = SV.solve(weak, ln)
    env = SV.envelope(sol["v"], ln["v"], ln["ds"])
    np.testing.assert_allclose(env["v"][corner], ln["v"][corner])


def test_line_of_puts_the_sample_mask_on_the_grid():
    n = 1000
    raw = dict(n=1, time=n / GRID_HZ, v=np.full(n, 40.0), s=np.arange(n) * 40.0 / GRID_HZ,
               gy=np.full(n, 1.0), gz=np.zeros(n), grade=np.zeros(n), thr=np.zeros(n), brk=np.zeros(n))
    sat = np.zeros(n, bool)
    sat[200:400] = True                            # 160 m at 0.8 m per sample
    ax = np.vstack([sat, np.zeros(n, bool)])
    ln = SV.line_of(raw, sat_ref=sat, sat_axles=ax)
    clamped = ln["s"][ln["sat_ref"]]
    assert clamped.min() == pytest.approx(160.0, abs=2.0) and clamped.max() == pytest.approx(318.0, abs=2.0)
    assert np.isnan(ln["v_clamp"][~ln["sat_ref"]]).all()
    np.testing.assert_allclose(ln["v_clamp"][ln["sat_ref"]], 40.0)
    assert (ln["sat_axles"][0] == ln["sat_ref"]).all()
