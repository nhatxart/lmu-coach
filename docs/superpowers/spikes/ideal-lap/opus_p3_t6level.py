"""Opus probe 3 (synthetic, T6): which LEVEL rule recovers the true mean car under capacity noise?
Variants of fit_axles (patched in memory): the quantile of the level calibration (q) and the
|gxt| cap of the base mask (D1: drive inside the rear set). Per data set (car A/B, seed, noise):
  cap F/R  : fitted A_j / true A_j at v 25/40/55 m/s, gy 1.6, gx 0, gz 0 (mean of the 3), each car on its own load model
  vs_true  : fitted vs TRUE MEAN car, plain model time on lap 3's line (+ = fitted faster)
  phantom  : clamped envelope gain at level 1 on lap 3 (the perfect driver: should be ~0)
  kbS      : fitted kb*S / true kb*S at v 40 straight (braking capacity), ktR: same for traction
usage: python -I opus_p3_t6level.py"""
import sys, os, inspect, textwrap, itertools
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PERF = Path(os.environ.get("IL_PERF", r"C:\Users\hoang\Projects\lmu-coach-ideal-perf"))
sys.path[:0] = [str(PERF / "backend"), str(PERF / "backend" / "tests")]
from app import ideal_car as C, ideal_solver as SV
from app.ideal_state import FileReduction, lap_state
from synthetic_ideal import true_car, line, perfect_laps

CFG = {"q": 0.5, "cap": 0.2}
src = textwrap.dedent(inspect.getsource(C.fit_axles))
for a, b in (("base = (np.abs(gxt) < 0.2) & (y_all > 0.45 * gymax)", "base = (np.abs(gxt) < CFG['cap']) & (y_all > 0.45 * gymax)"),
             ("mu = m1 * float(np.quantile(y_all[m] / (m1 * shape), q))", "mu = m1 * float(np.quantile(y_all[m] / (m1 * shape), CFG['q']))")):
    assert a in src, a
    src = src.replace(a, b)
C.__dict__["CFG"] = CFG
exec(compile(src, "fit_axles_patched", "exec"), C.__dict__)


def caps(car):
    v = np.array([25.0, 40.0, 55.0]); z = np.zeros(3)
    Af, Ar = car.axles(v, z, z + 1.6, z)
    return np.array([Af.mean(), Ar.mean()])


def brake_trac(car):
    v, z = np.array([40.0]), np.zeros(1)
    Af, Ar = car.axles(v, z, z, z)
    return float(car.kb * (car.sf * Af[0] + car.sr * Ar[0])), float(car.kt * car.sr * Ar[0])


sets = [("A", 2026, 0.035), ("A", 7, 0.035), ("B", 7, 0.035), ("B", 99, 0.035), ("A", 99, 0.05), ("A", None, 0.0), ("B", None, 0.0)]
variants = [(q, cap) for cap in (0.2, 0.08) for q in (0.5, 0.65, 0.75, 0.85)]
rows = {}
for kind, seed, nz in sets:
    car = true_car(kind); ln0 = line()
    raws, _, _ = perfect_laps(car, ln0, 6, np.random.default_rng(seed or 0), place_sd=nz, lap_sd=nz)
    states = [lap_state(r) for r in raws]
    red = FileReduction(file="s", mtime=0, size=0, meta={}, track="s", car_model="S", car_class="LMGT3",
                        game_version="t", load_source="susp", grade_source="pos_y", laps=states)
    ln_plain = SV.line_of(raws[3]); lap = float(np.sum(ln_plain["ds"] / ln_plain["v"]))
    t_true = SV.solve(car, ln_plain, clamp=False)["time"]
    ct = caps(car); bt = brake_trac(car)
    print("== car %s seed %s noise %.3f | lap %.2f s, true mean car %+.2f %% vs lap" % (kind, seed, nz, lap, 100 * (lap - t_true) / lap))
    for q, cap in variants:
        CFG["q"], CFG["cap"] = q, cap
        try:
            g = C.fit_group([red], "LMGT3"); fit = g.car_for("s")
        except Exception as e:
            print("   q %.2f cap %.2f: %s" % (q, cap, e)); continue
        st = lap_state(raws[3]); axles = [g.axle(0), g.axle(1)]
        sat, ax = SV.saturated_ref(st, fit, axles, per_axle=True)
        ln = SV.line_of(raws[3], sat_ref=sat, sat_axles=ax)
        lv = SV.levels(fit, ln, 0.03)
        tf = lv["model_only_time"]
        cf = caps(fit) / ct; bf = brake_trac(fit)
        r = dict(capF=cf[0], capR=cf[1], vs_true=100 * (t_true - tf) / t_true, phantom=100 * lv["gain"] / lap,
                 phantom_min=100 * lv["gain_min"] / lap, kbS=bf[0] / bt[0], ktR=bf[1] / bt[1], nF=axles[0].n_sat, nR=axles[1].n_sat)
        rows.setdefault((q, cap), []).append((nz, r))
        print("   q %.2f cap %.2f: cap F %.3f R %.3f | vs_true %+.2f%% | phantom %.2f%% (min %.2f%%) | kbS %.3f ktR %.3f | nsat %d/%d" % (
            q, cap, r["capF"], r["capR"], r["vs_true"], r["phantom"], r["phantom_min"], r["kbS"], r["ktR"], r["nF"], r["nR"]))
print("== summary over the noisy sets (mean / worst)")
for k, lst in rows.items():
    x = [r for nz, r in lst if nz > 0]
    if not x: continue
    f = lambda key: (np.mean([r[key] for r in x]), max((r[key] for r in x), key=lambda a: abs(a - (1 if key.startswith(('cap', 'kb', 'kt')) else 0))))
    print("   q %.2f cap %.2f: capF %.3f/%.3f capR %.3f/%.3f vs_true %+.2f/%+.2f%% phantom %.2f/%.2f%% kbS %.3f/%.3f ktR %.3f/%.3f" % (
        k[0], k[1], *f("capF"), *f("capR"), *f("vs_true"), *f("phantom"), *f("kbS"), *f("ktR")))
