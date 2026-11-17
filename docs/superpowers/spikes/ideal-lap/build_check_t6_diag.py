"""Diagnose the T6 perfect-driver result (synthetic only, no recordings).
Which part of the refit makes the gain: noise in gy, or the fit itself?
usage: python build_check_t6_diag.py [noise ...]   (noise = log sd of the place and of the lap capacity field, round 1)"""
import sys, os
import numpy as np
from build_check_common import BACKEND
sys.path.insert(0, str(BACKEND / "tests"))
import synthetic_ideal as SI
from app import ideal_car as C, ideal_solver as SV
from app.ideal_state import FileReduction, lap_state

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
car = SI.true_car()
for noise in [float(x) for x in sys.argv[1:]] or [0.0, 0.05]:
    for seed in (2026, 7, 99):
        raws, sols, _ = SI.perfect_laps(car, SI.line(), 6, np.random.default_rng(seed), place_sd=noise, lap_sd=noise); truth = sols[3]
        red = FileReduction(file="s", mtime=0, size=0, meta={}, track="", car_model="", car_class="LMGT3",
                            game_version="", load_source="susp", grade_source="", laps=[lap_state(r) for r in raws])
        g = C.fit_group([red], "LMGT3")
        f = g.car_for("s")
        ln = SV.line_of(raws[3])
        lap = float(np.sum(ln["ds"] / ln["v"]))
        res = {}
        for nm, cm in (("fitted", f), ("true car", car)):
            out = SV.levels(cm, ln, 0.02)
            res[nm] = (lap - out["model_only_time"], out["gain_min"], out["gain"], out["gain_max"])
        # the true car on the TRUE line (no noise in curvature)
        tl = SI.line()
        print("noise %.2f seed %4d | lap %.2f (truth solve %.2f) | fitted: mu1 %s k %s c %s kb %.3f kt %.3f sf %.3f n_sat %d/%d" % (
            noise, seed, lap, truth["time"], np.round(f.mu1, 3), np.round(f.k, 3), np.round(f.c, 3), f.kb, f.kt, f.sf,
            g.info["axles"]["front"]["n_sat"], g.info["axles"]["rear"]["n_sat"]))
        for nm, (mg, a, b, c) in res.items():
            print("     %-8s on lap-3 line: model_only gain %+.3f s (%+.2f %%) | envelope %+.3f / %+.3f / %+.3f s (%.2f / %.2f %%)" % (
                nm, mg, 100 * mg / lap, a, b, c, 100 * a / lap, 100 * b / lap))
        st = red.laps[0]
        Af, Ar = f.axles(st.v, st.gx, st.gy, st.gz); Tf, Tr = car.axles(st.v, st.gx, st.gy, st.gz)
        rows = []
        for lo, hi in ((10, 30), (30, 45), (45, 60), (60, 90)):
            m = (st.v >= lo) & (st.v < hi) & (np.abs(st.gy) > 0.8)
            if m.sum():
                rows.append("%d-%d Af %.3f Ar %.3f" % (lo, hi, np.median(Af[m] / Tf[m]), np.median(Ar[m] / Tr[m])))
        print("     capacity fitted / true (|gy| > 0.8):", " | ".join(rows))
        # kb, kt the fit procedure gets when given the TRUE capacities
        p = C._pool([red], {"s": g.load_models["s"]})
        gxt = p["gx"] + p["grade"] + f.drag(p["v"])
        lamT = car.lam(p["v"], p["gx"], p["gy"], p["gz"])
        TAf, TAr = C.axle_capacity(lamT, p["v"], car.mu1, car.k, car.c)
        kbT = C.fit_braking(p, gxt, TAf, TAr, car.sf, p["gf"]); ktT = C.fit_traction(p, gxt, TAr, car.sf)
        print("     kb / kt from the fit rules on TRUE capacities: %.3f (%s, n %d) / %.3f (%s, n %d); true %.3f / %.3f" % (
            kbT["value"], kbT["rule"], kbT["n"], ktT["value"], ktT["rule"], ktT["n"], car.kb, car.kt))
