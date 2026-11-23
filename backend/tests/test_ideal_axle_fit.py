"""Ideal lap, design T2: a synthetic car with known mu1, k, c per axle. The
saturated samples are found by Grip Fraction and, separately, by a slip-angle
knee; mu1 comes back within 2 % both ways. An axle with only 100 saturated
samples comes back `lower_bound`."""
import numpy as np
import pytest

from app import ideal_car as C

MU1, K, CC = (1.40, 1.48), (0.06, 0.03), (0.10, 0.20)
KNEE = 4.0


def _car_samples(rng, n_front_sat, n_rear_sat, n_unsat=20000, rear_v=(0, 999)):
    """Pooled dict as ideal_car.fit_axles takes it. lam per wheel is given (the
    load model's output), every sample is quasi-steady cornering (gx 0).
    `rear_v`: speed band (m/s) of the rear's saturated samples."""
    m = 400000
    v = rng.uniform(20, 70, m)
    lam = np.vstack([rng.uniform(0.9, 1.7, m) for _ in range(4)])
    Af, Ar = C.axle_capacity(lam, v, MU1, K, CC)
    lim = np.minimum(Af, Ar)
    front_lim = Af < Ar
    i_f = np.flatnonzero(front_lim)[:n_front_sat]
    i_r = np.flatnonzero(~front_lim & (v >= rear_v[0]) & (v < rear_v[1]))[:n_rear_sat]
    i_u = np.arange(m)[-n_unsat:]
    idx = np.concatenate([i_f, i_r, i_u])
    u = np.concatenate([np.ones(len(i_f) + len(i_r)), rng.uniform(0.55, 0.98, n_unsat)])
    gy = u * lim[idx] * (1 + 0.03 * rng.standard_normal(len(idx))) * rng.choice([-1, 1], len(idx))
    v, lam, Af, Ar = v[idx], lam[:, idx], Af[idx], Ar[idx]
    sat = np.zeros((2, len(idx)), bool)
    sat[0, :len(i_f)] = True
    sat[1, len(i_f):len(i_f) + len(i_r)] = True
    slip, gf = [], []
    for j, A in enumerate((Af, Ar)):
        uj = np.minimum(np.abs(gy) / A, 0.999)
        s = np.where(sat[j], KNEE + rng.uniform(0.2, 3.0, len(idx)), KNEE * uj)
        g = np.where(sat[j], 0.1, 0.35 + (1 - uj))
        slip += [s, s]; gf += [g, g]
    z = np.zeros(len(idx))
    return dict(v=v, gx=z, gy=gy, gz=z, grade=z, thr=z, brk=z, abs=z, tc=z, yawacc=z, lam=lam,
                slip=np.vstack(slip), gf=np.vstack(gf))


@pytest.mark.parametrize("use_gf", [True, False])
def test_axle_level_recovered_by_grip_fraction_and_by_knee(use_gf):
    d = _car_samples(np.random.default_rng(5), 3000, 3000)
    gxt = np.zeros(len(d["v"]))
    fits = C.fit_axles(d, gxt, slip=d["slip"], gf=d["gf"] if use_gf else None)
    for j in (0, 1):
        assert fits[j].status == "observed"
        assert fits[j].saturation == ("grip_fraction" if use_gf else "knee")
        # §15 V2 (B1): the level is the 0.75 quantile of observed / predicted; with
        # the 3 % gy noise here (|gy| x (1 + 0.03 N)) that is 1 + 0.03 x 0.6745 above
        # the true mu1 (the median calibration recovered mu1 itself)
        assert fits[j].mu1 / MU1[j] == pytest.approx(1 + 0.03 * 0.6745, abs=0.012)
    if not use_gf:
        assert fits[0].knee_deg == pytest.approx(KNEE, abs=0.5)


def test_axle_with_few_saturated_samples_is_lower_bound():
    d = _car_samples(np.random.default_rng(6), 3000, 100)
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=d["gf"])
    assert fits[0].status == "observed"
    assert fits[1].status == "lower_bound"
    assert fits[1].n_sat == 100
    # k, c of a lower-bound axle come from the other axle
    assert (fits[1].k, fits[1].c) == (fits[0].k, fits[0].c)


def test_axle_saturated_in_a_narrow_speed_band_borrows_k_c():
    """Design §4.2 round 1 (§13 R2): an axle saturated only at 20-30 m/s cannot
    tell its speed term; it gets the other axle's k, c and fits only mu1."""
    d = _car_samples(np.random.default_rng(8), 3000, 3000, rear_v=(20, 30))
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=d["gf"])
    assert fits[0].terms == "fitted"
    assert fits[1].status == "observed"
    assert fits[1].terms == "borrowed"
    assert (fits[1].k, fits[1].c) == (fits[0].k, fits[0].c)
    assert fits[1].v_sat[0] >= 20 and fits[1].v_sat[1] <= 30
    # where it was seen saturated, the level is still right
    lam = np.full((4, 1), 1.2)
    got = C.axle_capacity(lam, np.array([25.0]), [f.mu1 for f in fits], [f.k for f in fits], [f.c for f in fits])[1]
    want = C.axle_capacity(lam, np.array([25.0]), MU1, K, CC)[1]
    assert got[0] == pytest.approx(want[0], rel=0.02)


def test_both_axles_unidentifiable_take_the_defaults():
    d = _car_samples(np.random.default_rng(9), 100, 100)
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=d["gf"])
    for f in fits:
        assert f.status == "lower_bound" and f.terms == "default"
        assert (f.k, f.c) == (C.K_DEFAULT, C.C_DEFAULT)


def test_extrapolated_above_the_saturated_speeds():
    d = _car_samples(np.random.default_rng(8), 3000, 3000, rear_v=(20, 30))
    rear = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=d["gf"])[1]
    assert not C.is_extrapolated(rear, rear.v_sat[1] + 4.0)
    assert C.is_extrapolated(rear, rear.v_sat[1] + 6.0)


def test_continuous_knee_is_stable_on_a_flat_plateau():
    """§13 R3: the knee must not jump a whole bin when the plateau is flat:
    removing 5 % of the samples at random moves it by less than 0.3 deg."""
    rng = np.random.default_rng(12)
    n = 6000                                          # 300 samples per bin
    slip = rng.uniform(0, 10, n)
    # 97 % of the plateau is reached at 4.25 deg, the centre of a bin: there a
    # bin-based knee flips between two bins from one sample set to the next
    tau = 4.25 / np.log(10)
    r = (1 - 0.3 * np.exp(-slip / tau)) * (1 + 0.06 * rng.standard_normal(n))
    base = np.ones(n, bool)
    _, k_all = C.knee_mask(slip, r, base)
    assert 3.0 < k_all < 5.5
    moves = []
    for _ in range(20):
        keep = rng.random(n) >= 0.05
        _, k = C.knee_mask(slip, r, base & keep)
        moves.append(abs(k - k_all))
    assert max(moves) < 0.3


def test_knee_vs_tyre_v3_peak():
    d = _car_samples(np.random.default_rng(7), 3000, 3000)
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=None,
                       v3_peaks={"front": 4.2, "rear": 6.0})
    assert fits[0].knee_vs_v3 == "agree"
    assert fits[1].knee_vs_v3 == "differ"
