"""Ideal lap, design T2: a synthetic car with known mu1, k, c per axle. The
saturated samples are found by Grip Fraction and, separately, by a slip-angle
knee; mu1 comes back within 2 % both ways. An axle with only 100 saturated
samples comes back `lower_bound`."""
import numpy as np
import pytest

from app import ideal_car as C

MU1, K, CC = (1.40, 1.48), (0.06, 0.03), (0.10, 0.20)
KNEE = 4.0


def _car_samples(rng, n_front_sat, n_rear_sat, n_unsat=20000):
    """Pooled dict as ideal_car.fit_axles takes it. lam per wheel is given (the
    load model's output), every sample is quasi-steady cornering (gx 0)."""
    m = 400000
    v = rng.uniform(20, 70, m)
    lam = np.vstack([rng.uniform(0.9, 1.7, m) for _ in range(4)])
    Af, Ar = C.axle_capacity(lam, v, MU1, K, CC)
    lim = np.minimum(Af, Ar)
    front_lim = Af < Ar
    i_f = np.flatnonzero(front_lim)[:n_front_sat]
    i_r = np.flatnonzero(~front_lim)[:n_rear_sat]
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
        assert fits[j].mu1 == pytest.approx(MU1[j], rel=0.02)
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


def test_knee_vs_tyre_v3_peak():
    d = _car_samples(np.random.default_rng(7), 3000, 3000)
    fits = C.fit_axles(d, np.zeros(len(d["v"])), slip=d["slip"], gf=None,
                       v3_peaks={"front": 4.2, "rear": 6.0})
    assert fits[0].knee_vs_v3 == "agree"
    assert fits[1].knee_vs_v3 == "differ"
