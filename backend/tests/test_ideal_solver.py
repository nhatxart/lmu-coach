"""Ideal lap solver, design T3 (closed forms) and T5 (ceiling vs balanced)."""
import numpy as np
import pytest

from app import ideal_car as C
from app import ideal_solver as SV

G = 9.81


def _car(mu=(1.5, 1.5), kb=1.0, kt=1.0, power=0.5, sf=0.5):
    """Constant load (lam 1), no load or speed term, no drag: A_j = mu1_j."""
    lm = C.LoadModel(static=np.ones(4), coef=np.zeros((4, 5)), source="susp")
    return C.CarModel(load=lm, mu1=list(mu), k=[0.0, 0.0], c=[0.0, 0.0], s_front=sf, kb=kb, kt=kt,
                      d0=0.0, d2=0.0, power_v=[0.0, 200.0], power_g=[power, power], corner_drag=0.0)


def _line(ks, ds=2.0, v=30.0):
    n = len(ks)
    z = np.zeros(n)
    return dict(s=np.arange(n) * ds, ks=np.asarray(ks, float), kv=z, grade=z, v=np.full(n, v),
                thr=z, brk=z, ds=ds, time=float(n * ds / v))


@pytest.mark.parametrize("R,mu", [(60.0, (1.5, 1.5)), (150.0, (1.3, 1.6)), (400.0, (1.9, 1.7))])
def test_constant_radius_circle_closed_form(R, mu):
    n = int(round(2 * np.pi * R / 2.0))
    sol = SV.solve(_car(mu=mu), _line(np.full(n, 1.0 / R)))
    v_exact = np.sqrt(min(mu) * G * R)
    np.testing.assert_allclose(sol["v"], v_exact, rtol=0.005)
    # the limiting axle is the one with less grip
    want = "front" if mu[0] < mu[1] else "rear"
    if mu[0] != mu[1]:
        assert set(sol["binding"]) == {want}


def test_straight_between_two_hairpins_matches_hand_integration():
    R, Lh, Ls, ds = 15.0, 40.0, 600.0, 2.0
    nh, ns = int(Lh / ds), int(Ls / ds)
    ks = np.concatenate([np.full(nh, 1 / R), np.zeros(ns), np.full(nh, 1 / R), np.zeros(ns)])
    a_pow, b_brk = 0.5, 1.5                       # g: power binds (traction 0.75 g), brake = kb * A
    sol = SV.solve(_car(power=a_pow), _line(ks, ds))
    v = sol["v"]
    vh = np.sqrt(1.5 * G * R)
    seg = v[nh:nh + ns + 1]                        # straight incl. the next hairpin's first point
    s = np.arange(len(seg)) * ds
    # hand integration: v^2 = vh^2 + 2 g a s up, v^2 = vh^2 + 2 g b (Ls - s) down
    s_brake = b_brk * Ls / (a_pow + b_brk)
    v_hand = np.sqrt(np.minimum(vh ** 2 + 2 * G * a_pow * s, vh ** 2 + 2 * G * b_brk * (Ls - s)))
    assert abs(s[np.argmax(seg)] - s_brake) <= 2.0
    # exit: where the hand-integrated speed reaches 40 m/s vs where the solver's does
    s_hand = (40 ** 2 - vh ** 2) / (2 * G * a_pow)
    s_solv = np.interp(40.0, seg[: np.argmax(seg)], s[: np.argmax(seg)])
    assert abs(s_solv - s_hand) <= 2.0
    # whole profile within 2 m of the hand integration: the backward pass applies
    # the decel available at a point over the step before it, and at the hairpin
    # entry (at the lateral limit) that is zero, so braking ends one step early.
    v_early = np.sqrt(np.minimum(vh ** 2 + 2 * G * a_pow * s, vh ** 2 + 2 * G * b_brk * np.maximum(Ls - ds - s, 0)))
    assert (seg <= v_hand * (1 + 1e-3)).all() and (seg >= v_early * (1 - 1e-3)).all()
    codes = sol["binding"][nh:nh + ns]
    assert "power" in set(codes) and "brake" in set(codes)


def test_ceiling_at_least_balanced_and_equal_when_axles_equal():
    rng = np.random.default_rng(3)
    ks = np.convolve(rng.normal(0, 0.02, 1500), np.ones(40) / 40, mode="same")
    ln = _line(ks)
    car = _car(mu=(1.35, 1.6), sf=0.47)
    bal, ceil = SV.solve(car, ln, "balanced"), SV.solve(car, ln, "ceiling")
    assert (ceil["v"] >= bal["v"] - 1e-9).all()
    assert ceil["time"] < bal["time"]
    eq = _car(mu=(1.5, 1.5))
    np.testing.assert_allclose(SV.solve(eq, ln, "ceiling")["v"], SV.solve(eq, ln, "balanced")["v"])
