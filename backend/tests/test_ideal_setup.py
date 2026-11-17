"""Ideal lap, round 2 (design §14 R6/R8): T10. A per-file axle trim moves the
group's grip level to the file's own saturated samples; a setup flag lists
what differs from the group, and an untrimmable axle with a setup difference
widens beta."""
from dataclasses import replace

import numpy as np
import pytest

from app import ideal_car as C
from app import ideal_store as ST
from app.ideal_state import FileReduction, lap_state, setup_fingerprint
from synthetic_ideal import true_car, line, perfect_laps
from test_ideal_store import World, _entry, MODEL

N1 = "Synth-Synthetic GT-2026-11-01T10_00_00Z.duckdb"
N2 = "Synth-Synthetic GT-2026-11-02T10_00_00Z.duckdb"
N3 = "Synth-Synthetic GT-2026-11-03T10_00_00Z.duckdb"
SETUP = dict(camber=[2.0, 1.6], toe=[0.30, 0.20], press=[180.0, 178.0], rh_mid=[52.0, 70.0],
             rh_fast=[45.0, 66.0], compound="Medium/Medium", track_temp=30.0)


def _car(mu):
    """Car B with a speed term on the front only, so that with these levels
    the rear limits when slow and the front at speed: both axles saturate in
    both files (with car B's own terms the 1.30 / 1.40 file never saturates
    its rear)."""
    c = true_car("B")
    c.mu1, c.c = list(mu), [0.20, 0.0]
    return c


@pytest.fixture(scope="module")
def files():
    """8 laps per file, Grip Fraction from the LATERAL usage: T10 tests the trim
    itself. With the combined usage (the game's) the rear's saturated samples
    inside the |gxt| < 0.2 base carry drive, |gy| sits below A_r there and the
    trim follows that bias (file 2 rear 0.994 -> 0.970; batch-b-report.md).
    With 4 laps the rear's weight n / (n + 500) is ~0.5 and leaves 1.1 %."""
    out = {}
    for name, mu, seed in ((N1, (1.30, 1.40), 1), (N2, (1.36, 1.34), 2)):
        car = _car(mu)
        raws, _, _ = perfect_laps(car, line(), 8, np.random.default_rng(seed), place_sd=0.0, lap_sd=0.0,
                                  gf_usage="lateral")
        out[name] = (car, [lap_state(r) for r in raws])
    return out


def _red(name, states, setup=None):
    return FileReduction(file=name, mtime=1.0, size=100, meta={}, track="Synth", car_model=MODEL,
                         car_class="LMGT3", game_version="1.0", load_source="susp", grade_source="pos_y",
                         laps=states, setup=setup if setup is not None else dict(SETUP))


def _level_vs_truth(car, true, states, j):
    """median A_j(car) / A_j(true) on the file's saturated samples of axle j."""
    r = []
    for s in states:
        m = s.gf[[0, 1] if j == 0 else [2, 3]].mean(0) < C.GF_SATURATED
        m &= np.abs(s.gy) > 0.5
        A, T = car.axles(s.v, s.gx, s.gy, s.gz)[j], true.axles(s.v, s.gx, s.gy, s.gz)[j]
        r.append(A[m] / T[m])
    return float(np.median(np.concatenate(r)))


def test_trim_recovers_each_files_level(files):
    reds = [_red(n, st) for n, (_, st) in files.items()]
    g = C.fit_group(reds, "LMGT3")
    axles = [g.axle(0), g.axle(1)]
    for red in reds:
        true = files[red.file][0]
        group_car = g.car_for(red.file)
        trimmed, trims = C.axle_trim(group_car, axles, red)
        for j in (0, 1):
            assert trims[j]["trim"] == "applied" and trims[j]["n"] >= 250
            assert 0 < trims[j]["w"] < 1
            assert _level_vs_truth(trimmed, true, red.laps, j) == pytest.approx(1.0, abs=0.01)
        # untrimmed the pooled level is off for at least one axle of each file
        assert max(abs(_level_vs_truth(group_car, true, red.laps, j) - 1) for j in (0, 1)) > 0.01


def test_a_file_alone_in_its_group_has_zero_trim(files):
    red = _red(N1, files[N1][1])
    g = C.fit_group([red], "LMGT3")
    _, trims = C.axle_trim(g.car_for(N1), [g.axle(0), g.axle(1)], red)
    for t in trims:
        assert abs(t["t"]) < 1e-9


def test_trim_is_capped(files):
    red = _red(N1, files[N1][1])
    g = C.fit_group([red], "LMGT3")
    weak = g.car_for(N1).scaled(0.85)                  # 16 % off: a different car
    _, trims = C.axle_trim(weak, [g.axle(0), g.axle(1)], red)
    assert all(t["trim"] == "capped" and abs(t["t"]) == pytest.approx(C.TRIM_CAP) for t in trims)


def test_setup_differs_thresholds():
    med = dict(SETUP, aero_f=0.44, lltd_f=0.55, stat_f=0.45)
    same = dict(med, camber=[2.2, 1.5], toe=[0.35, 0.15], press=[182.0, 176.0], rh_mid=[54.0, 68.0])
    assert ST.setup_differs(same, med) == []
    diff = dict(med, camber=[2.0, 2.6], toe=[0.45, 0.2], rh_fast=[50.0, 66.0], aero_f=0.46,
                compound="Hard/Hard", track_temp=45.0)
    assert ST.setup_differs(diff, med) == ["camber", "toe", "rh_fast", "aero_f", "compound"]
    # a missing measurement is not a difference
    assert ST.setup_differs(dict(med, camber=None), med) == []


def test_group_median_of_fingerprints():
    fps = [dict(SETUP, camber=[1.0, 1.0], aero_f=0.4), dict(SETUP, camber=[2.0, 3.0], aero_f=0.5),
           dict(SETUP, camber=[3.0, 2.0], aero_f=0.6, compound="Hard/Hard")]
    med = ST.group_median(fps)
    assert med["camber"] == [2.0, 2.0] and med["aero_f"] == pytest.approx(0.5)
    assert med["compound"] == "Medium/Medium"


def test_untrimmable_axle_with_a_setup_difference_widens_beta(tmp_path, files):
    """A file with only 100 saturated rear samples keeps the group rear
    (trim none) and, with a camber difference, gets beta + 0.03 and one step
    less confidence."""
    st3 = [s for s in files[N2][1]]
    keep = 100
    thin = []
    for s in st3:
        gf = s.gf.copy()
        sat = np.flatnonzero(gf[2:].mean(0) < C.GF_SATURATED)
        gf[2:, sat[keep // len(st3):]] = 0.6           # rear saturated on ~100 samples only
        thin.append(replace(s, gf=gf))
    w = World(tmp_path)
    w.add(_red(N1, files[N1][1]), _entry(N1))
    w.add(_red(N2, files[N2][1]), _entry(N2))
    w.add(_red(N3, thin, setup=dict(SETUP, camber=[2.0, 2.4])), _entry(N3))
    st = w.store()
    car1, info1 = st.car_for_lap(w.path(N1))
    car3, info3 = st.car_for_lap(w.path(N3))
    assert info3["axles"]["rear"]["trim"] == "none" and info3["axles"]["front"]["trim"] == "applied"
    assert info3["setup_differs"] == ["camber"]
    assert info3["beta"] == pytest.approx(info1["beta"] + 0.03)
    order = ["high", "medium", "low"]
    assert order.index(info3["confidence"]) == min(order.index(info1["confidence"]) + 1, 2)
    assert any("setup differs" in n for n in info3["notes"])
    assert info1["setup_differs"] == [] and info1["setup"]["camber"] == SETUP["camber"]


def test_setup_fingerprint_adds_the_load_model_shares():
    lm = true_car("B").load
    fp = setup_fingerprint(None, None, lm, measured=dict(SETUP))
    a = lm.coef[:, 0] * lm.static
    lat = (np.abs(lm.coef[:, 2]) + np.abs(lm.coef[:, 4])) * lm.static
    assert fp["aero_f"] == pytest.approx(a[:2].sum() / a.sum())
    assert fp["lltd_f"] == pytest.approx(lat[:2].sum() / lat.sum())
    assert fp["stat_f"] == pytest.approx(lm.static[:2].sum() / lm.static.sum())
    assert fp["camber"] == SETUP["camber"]
