"""Batch A step 1 check (design §8): the per-file load model of app.ideal_car.
Front/rear axle mean lam in hard cornering, modelled vs measured (target within
2 %, was 6-7 % with the spike's model), and held-out R^2 per wheel (>= 0.93).
usage: python build_check_step1.py [file ...]   (default: Porsche COTA 11/8)"""
import sys
import numpy as np
from build_check_common import load, PORSCHE
from app import ideal_car as C

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for name in sys.argv[1:] or [PORSCHE]:
    d = load(name)
    sts, src = d["states"], d["load_source"]
    lm = C.fit_load_model(sts, src)
    cat = lambda k: np.concatenate([getattr(s, k) for s in sts])
    v, gx, gy, gz = (cat(k) for k in ("v", "gx", "gy", "gz"))
    L = np.concatenate([s.L(src) for s in sts], axis=1)
    meas = L / lm.static[:, None]
    mod = lm.lam(v, gx, gy, gz)
    # the spike's model (no |gy| term) for comparison
    X = np.c_[np.ones_like(v), v * v, gx, gy, gz]
    Cs = np.array([np.linalg.lstsq(X, L[i], rcond=None)[0] for i in range(4)])
    old = (Cs / Cs[:, :1]) @ X.T
    meas_old = L / Cs[:, :1]                         # the spike's own static values
    hard = np.abs(gy) > 0.6 * np.percentile(np.abs(gy), 99)
    print("%s (%s), %d qualifying laps, load source %s" % (d["model"], d["cls"], len(sts), src))
    print("  held-out R2 per wheel FL FR RL RR: %s  (min %.3f, target >= 0.93)" % (np.round(lm.r2, 3), min(lm.r2)))
    print("  coef a,b,c,d,e per wheel:\n%s" % np.array2string(lm.coef, precision=5))
    print("  hard cornering (|gy| > 0.6 p99), axle-mean lam: band | n | front meas / model (bias) / spike model | rear meas / model (bias)")
    for a, b in ((0, 999), (50, 90), (90, 120), (120, 150), (150, 180), (120, 180), (180, 260)):
        m = hard & (v * 3.6 >= a) & (v * 3.6 < b)
        if m.sum() < 30:
            continue
        fm, fo = meas[:2].mean(0)[m].mean(), mod[:2].mean(0)[m].mean()
        fsm, fs = meas_old[:2].mean(0)[m].mean(), old[:2].mean(0)[m].mean()
        rm, ro = meas[2:].mean(0)[m].mean(), mod[2:].mean(0)[m].mean()
        print("   %3d-%3d | %5d | %.3f / %.3f (%+.1f %%) | spike %.3f / %.3f (%+.1f %%) | rear %.3f / %.3f (%+.1f %%)" % (
            a, min(b, 999), m.sum(), fm, fo, 100 * (fo / fm - 1), fsm, fs, 100 * (fs / fsm - 1), rm, ro, 100 * (ro / rm - 1)))
