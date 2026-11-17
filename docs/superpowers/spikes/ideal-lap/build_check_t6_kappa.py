"""Round-1 T6 diagnosis (synthetic only): kb, kt are ratios to the FITTED
capacity. Are the braking / traction capacities themselves right (kb * S and
kt * s_r * sqrt(Ar^2 - gy^2) at the limit samples, fitted vs true car)?"""
import sys
import numpy as np
from build_check_common import BACKEND
sys.path.insert(0, str(BACKEND / "tests"))
import synthetic_ideal as SI
from app import ideal_car as C
from app.ideal_state import FileReduction, lap_state

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for kind in ("A", "B"):
    for seed in (None, 2026, 7, 99):
        car = SI.true_car(kind)
        nz = 0.0 if seed is None else 0.035
        raws, _, _ = SI.perfect_laps(car, SI.line(), 6, np.random.default_rng(seed or 0), place_sd=nz, lap_sd=nz)
        red = FileReduction(file="s", mtime=0, size=0, meta={}, track="", car_model="", car_class="LMGT3",
                            game_version="", load_source="susp", grade_source="", laps=[lap_state(r) for r in raws])
        g = C.fit_group([red], "LMGT3")
        f = g.car_for("s")
        p = C._pool([red], {"s": g.load_models["s"]})
        v, gx, gy, gz = p["v"], p["gx"], p["gy"], p["gz"]
        gxt = gx + p["grade"] + f.drag(v)

        def caps(cm):
            Af, Ar = cm.axles(v, gx, gy, gz)
            S = cm.sf * np.sqrt(np.maximum(Af ** 2 - gy ** 2, 0)) + cm.sr * np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 0))
            return cm.kb * S, cm.kt * cm.sr * np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 1e-4)), Af, Ar
        bf, tf, Aff, Arf = caps(f)
        bt, tt, Aft, Art = caps(car)
        brk = (np.abs(gy) < 0.3) & (p["brk"] > 40) & (p["abs"] > 0.5)
        trc = (p["tc"] > 0.5) & (v < 45) & (gxt > 0.2)
        hi = np.abs(gy) > 0.8
        print("car %s seed %s | kb %+.1f %% kt %+.1f %% | brake capacity fitted/true median %+.1f %% (n %d) | traction %+.1f %% (n %d) | "
              "Af %+.1f %% Ar %+.1f %% at |gy|>0.8 | rear terms %s, front %s" % (
                  kind, seed, 100 * (f.kb / car.kb - 1), 100 * (f.kt / car.kt - 1),
                  100 * (np.median(bf[brk] / bt[brk]) - 1), brk.sum(), 100 * (np.median(tf[trc] / tt[trc]) - 1), trc.sum(),
                  100 * (np.median(Aff[hi] / Aft[hi]) - 1), 100 * (np.median(Arf[hi] / Art[hi]) - 1),
                  g.axle(1).terms, g.axle(0).terms))
