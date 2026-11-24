"""Ideal lap, design T6 (asserts as §15.y, replacing §15 V6 and the round-1 list):
a PERFECT driver. A known car is solved on the synthetic line with its true
capacity multiplied by a noise field over lap distance (place part, same every
lap, log sd 0.035; lap part, new every lap, log sd 0.035; 50 m correlation); the
driver drives each lap at that lap's capacity. gy, speed and the line are exact;
Grip Fraction is low (and the slip angle past the knee) where the usage against
the noisy capacity is >= 0.995. 6 laps, refit, lap 3's line.

Compared with the TRUE MEAN car (the T6 car without the noise field), never with
the noisy driven lap. Limits are ~2.5 sd of the 9-seed spread (opus_p5_t6seeds):
- capacity of each axle at gy 1.6, mean over three speeds spread over the axle's
  OWN v_sat [p02, p98] (never at fixed speeds: car B's rear saturates only at
  25-35 m/s, so 55 m/s is V13's extrapolation, not a fit check):
  observed + fitted terms +-4 % per seed (mean +-2 %), observed + borrowed terms
  +-10 % (mean +-3.5 %), lower_bound in [0.85, 1.02] (mean <= 1.0: it may read
  low, never high);
- braking capacity kb (s_f A_f + s_r A_r) at v 40, gy 0 (§15.z, kb at the
  median): +-7 % per seed, mean +-4 %, noise-free +-3 %; traction capacity
  kt s_r A_r (§15.z, kt at the 0.75 level): +-12 % per seed, mean +-4 %,
  noise-free +-3 %. PREVIOUSLY (kt at the median) traction read 0.863-0.991,
  car B mean 0.936, under limits of -16 % / +4 % (withdrawn). kb, kt are not
  asserted as ratios (they compensate the fitted capacity, review I2);
- plain-model time on lap 3's line (no clamp, no measured power): both axles
  observed +-1.2 % per seed (mean +-0.6 %); an axle lower_bound in [-1.8, +0.5] %
  (mean [-1.0, +0.3] %), conservative by construction;
- phantom gain (clamped, level 1, measured power on as in the app) <= 0.5 % of
  the lap (the §11 trigger), gain_min <= gain <= gain_max;
- without noise: capacities +-4 %, plain +-0.3 %, phantom <= 0.1 %.
Seeds 2026, 7, 99 run by default (per-seed limits); seeds 1-6 and the means over
all nine are marked `slow` (run with --runslow or -m slow). -s prints the numbers."""
import numpy as np
import pytest

from app import ideal_solver as SV
from app import ideal_store as ST
from app.ideal_car import fit_group
from app.ideal_state import FileReduction, lap_state
from synthetic_ideal import true_car, line, perfect_laps

SEEDS_DEFAULT = (2026, 7, 99)
SEEDS_FULL = SEEDS_DEFAULT + (1, 2, 3, 4, 5, 6)
CAP_LIMITS = {"fitted": (0.96, 1.04), "borrowed": (0.90, 1.10), "lower_bound": (0.85, 1.02)}
CAP_MEAN = {"fitted": (0.98, 1.02), "borrowed": (0.965, 1.035), "lower_bound": (0.0, 1.0)}
_CACHE: dict = {}


def _kind(axle) -> str:
    return "lower_bound" if axle.status == "lower_bound" else axle.terms


def _cap(car, j, vs):
    vs = np.asarray(vs, float)
    z = np.zeros(len(vs))
    return float(car.axles(vs, z, z + 1.6, z)[j].mean())


def _brake_trac(car):
    Af, Ar = car.axles(np.array([40.0]), 0.0, np.array([0.0]), 0.0)
    return car.kb * (car.sf * Af[0] + car.sr * Ar[0]), car.kt * car.sr * Ar[0]


def run(kind: str, seed):
    if (kind, seed) in _CACHE:
        return _CACHE[(kind, seed)]
    car = true_car(kind)
    noise = 0.0 if seed is None else 0.035
    raws, _, _ = perfect_laps(car, line(), 6, np.random.default_rng(seed or 0), place_sd=noise, lap_sd=noise)
    states = [lap_state(r) for r in raws]
    red = FileReduction(file="synthetic.duckdb", mtime=0, size=0, meta={}, track="synthetic",
                        car_model="Synthetic", car_class="LMGT3", game_version="test", load_source="susp",
                        grade_source="pos_y", laps=states)
    gfit = fit_group([red], "LMGT3")
    fitted = gfit.car_for("synthetic.duckdb")
    axles = [gfit.axle(0), gfit.axle(1)]
    beta = ST.beta_and_confidence("track", 6, gfit, None)[0]
    plain_line = SV.line_of(raws[3])
    t_fit = SV.solve(fitted, plain_line, clamp=False)["time"]
    t_true = SV.solve(car, plain_line, clamp=False)["time"]
    sat, ax = SV.saturated_ref(states[3], fitted, axles, per_axle=True)
    ln = SV.line_of(raws[3], sat_ref=sat, sat_axles=ax)
    ln["p_off"] = SV.power_offset(fitted, ln)
    out = SV.levels(fitted, ln, beta)
    lap = out["t_ref_grid"]
    bf, tf = _brake_trac(fitted)
    bt, tt = _brake_trac(car)
    cap_in, cap_fix = [], []
    for j in (0, 1):
        vs = np.linspace(*axles[j].v_sat, 3)
        cap_in.append(_cap(fitted, j, vs) / _cap(car, j, vs))
        cap_fix.append(_cap(fitted, j, [25, 40, 55]) / _cap(car, j, [25, 40, 55]))
    res = dict(cap=cap_in, cap_fix=cap_fix, kinds=[_kind(a) for a in axles], brake=bf / bt, trac=tf / tt,
               plain=t_true / t_fit - 1,                  # + = the fitted car is faster
               gain=out["gain"] / lap, gain_min=out["gain_min"] / lap, gain_max=out["gain_max"] / lap,
               gain_s=out["gain"], clamped=out["clamped_share"])
    print("\nT6 car %s seed %s: %s | cap in F %.3f R %.3f (fix %.3f / %.3f) | v_sat F %.0f-%.0f R %.0f-%.0f | "
          "braking %.3f traction %.3f | plain vs true %+.2f %% | gain_min / gain / gain_max %+.2f / %+.2f / %+.2f %% "
          "(%.3f s) | clamped %.1f %%" % (
              kind, seed, "/".join("%s %s" % (a.status, a.terms) for a in axles), *cap_in, *cap_fix,
              *axles[0].v_sat, *axles[1].v_sat, res["brake"], res["trac"], 100 * res["plain"],
              100 * res["gain_min"], 100 * res["gain"], 100 * res["gain_max"], res["gain_s"], 100 * res["clamped"]))
    _CACHE[(kind, seed)] = res
    return res


def _plain_limits(kinds, mean=False):
    if "lower_bound" in kinds:
        return (-0.010, 0.003) if mean else (-0.018, 0.005)
    return (-0.006, 0.006) if mean else (-0.012, 0.012)


SEED_PARAMS = [s if s in SEEDS_DEFAULT else pytest.param(s, marks=pytest.mark.slow) for s in SEEDS_FULL]


@pytest.mark.parametrize("kind", ["A", "B"])
@pytest.mark.parametrize("seed", SEED_PARAMS)
def test_perfect_driver_per_seed(kind, seed):
    r = run(kind, seed)
    for j in (0, 1):
        lo, hi = CAP_LIMITS[r["kinds"][j]]
        assert lo <= r["cap"][j] <= hi, ("capacity in v_sat", j, r["kinds"][j], r["cap"][j])
    assert abs(r["brake"] - 1) <= 0.07, ("braking capacity", r["brake"])
    assert abs(r["trac"] - 1) <= 0.12, ("traction capacity", r["trac"])
    lo, hi = _plain_limits(r["kinds"])
    assert lo <= r["plain"] <= hi, ("plain vs true", r["kinds"], r["plain"])
    assert r["gain"] <= 0.005                            # the phantom gain: design §11 trigger
    assert r["gain_min"] <= r["gain"] <= r["gain_max"]


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["A", "B"])
def test_perfect_driver_means_over_the_seeds(kind):
    rs = [run(kind, s) for s in SEEDS_FULL]
    for j in (0, 1):
        for k, (lo, hi) in CAP_MEAN.items():
            caps = [r["cap"][j] for r in rs if r["kinds"][j] == k]
            if caps:
                assert lo <= np.mean(caps) <= hi, ("mean capacity", j, k, np.mean(caps))
    assert abs(np.mean([r["brake"] for r in rs]) - 1) <= 0.04
    assert abs(np.mean([r["trac"] for r in rs]) - 1) <= 0.04, ("mean traction", np.mean([r["trac"] for r in rs]))
    lb = [r["plain"] for r in rs if "lower_bound" in r["kinds"]]
    ob = [r["plain"] for r in rs if "lower_bound" not in r["kinds"]]
    for xs, mean_lim in ((lb, _plain_limits(["lower_bound"], True)), (ob, _plain_limits([], True))):
        if xs:
            assert mean_lim[0] <= np.mean(xs) <= mean_lim[1], ("mean plain", np.mean(xs))


@pytest.mark.parametrize("kind", ["A", "B"])
def test_perfect_driver_without_noise(kind):
    r = run(kind, None)
    for j in (0, 1):
        assert abs(r["cap"][j] - 1) <= 0.04, ("capacity", j, r["cap"][j])
    assert abs(r["plain"]) <= 0.003
    assert abs(r["brake"] - 1) <= 0.03, ("braking capacity", r["brake"])
    assert abs(r["trac"] - 1) <= 0.03, ("traction capacity", r["trac"])
    assert r["gain"] <= 0.001
