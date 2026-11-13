"""Fable probe 5 (build round 1): the saturation clamp, the speed span of the saturated sets,
and what the F8 drag order really moves.
Car = spike variant 'both' at the median (as fable_p4). il_core.py is patched in memory only.
  clamp : where the reference lap's own tyres are MEASURED saturated (Grip Fraction < 0.3, else
          slip angle in [knee, knee + 4]) while cornering hard, the corner limit is the driven
          speed (vc = v_ref), not the model's.
usage: python fable_p5.py file [file...]"""
import sys, types, copy
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, sm

import os; GATE = os.environ.get("FABLE_GATE", "none")  # none | base (|gxt| < 0.2) | latdom (|gxt| < 0.5 |gy|)
SRC = (Path(__file__).parent / "il_core.py").read_text(encoding="utf-8")


def module(drag_first=False):
    s = SRC
    reps = [("return np.stack([np.ones_like(v), v * v, gx, gy, gz], axis=-1)", "return np.stack([np.ones_like(v), v * v, gx, gy, gz, np.abs(gy)], axis=-1)"),
            ("lam = L / c0[:, None]", "lam = np.maximum((C / c0[:, None]) @ X.T, 0.02)"),
            ("mu1.append(m1 * scale); kk.append(k); cc_.append(c)",
             "mu1.append(m1 * scale); kk.append(k); cc_.append(c); p.setdefault('knee', []).append(float(knee)); p.setdefault('vspan', []).append([float(x) for x in np.percentile(v[m], [10, 50, 90])]); p.setdefault('lamspan', []).append([float(x) for x in np.percentile(lam[idx].mean(0)[m], [10, 90])]); p.setdefault('nsat', []).append(int(m.sum()))"),
            ("    vs = vc.copy()", "    vc = np.minimum(vc, ln['vcap']) if 'vcap' in ln else vc\n    vs = vc.copy()")]
    if drag_first:
        reps.append(('p["drag0"], p["drag2"] = 0.0, None', 'p["drag0"], p["drag2"] = 0.015, 4.8e-5'))
    for a, b in reps:
        assert a in s, a
        s = s.replace(a, b)
    m = types.ModuleType("il_core_p5"); exec(compile(s, "il_core_p5", "exec"), m.__dict__)
    return m


def sat_mask(M, car, l):
    """Reference lap measured saturated, per 50 Hz sample."""
    st = M.lap_state(l, car.p["load_source"])
    agy = np.abs(st["gy"]); hard = (agy > 0.45 * np.percentile(agy, 99)) & (st["v"] > 15)
    gxt = st["gx"] + st["grade"] + car.drag(st["v"])
    if GATE == "base": hard &= np.abs(gxt) < 0.2
    if GATE == "latdom": hard &= np.abs(gxt) < 0.5 * agy
    out = np.zeros(len(agy), bool)
    for j, idx in enumerate(((0, 1), (2, 3))):
        if st["gf"] is not None:
            out |= hard & (st["gf"][list(idx)].mean(0) < 0.3)
        else:
            k = car.p["knee"][j]; sl = st["slip"][list(idx)].mean(0)
            out |= hard & (sl >= k) & (sl <= k + 4.0)
    return out


def A_at(car, v, lam):
    return [float(car.mu1[j] * max(1 - car.k[j] * (lam - 1) - car.c[j] * (v - 40) / 40, 0.3) * lam) for j in (0, 1)]


for name in sys.argv[1:]:
    f = load_file(name)
    laps = sorted(real_laps(f), key=lambda l: l["time"])[:3]
    print("=" * 110); print(f["model"], "|", f["track"], f["file"][-24:-7])
    for drag_first in ((False, True) if f["laps"][0]["fx"] is None else (False,)):
        M = module(drag_first)
        car = M.fit_car([f], q=0.5, verbose=False)
        p = car.p
        print("  fit (%s): knee %s | mu1 %s k %s c %s | n_sat %s | kb %.3f kt %.3f d2 %.2e" % ("drag prior before the base mask (design F8)" if drag_first else "spike order",
              p["knee"], np.round(p["mu1"], 3), np.round(p["k"], 3), np.round(p["c"], 3), p["nsat"], car.kb, car.kt, p["drag2"]))
        print("      saturated set speed p10/p50/p90 m/s: front %s rear %s | axle-mean lam p10/p90: front %s rear %s" % (np.round(p["vspan"][0], 0), np.round(p["vspan"][1], 0), np.round(p["lamspan"][0], 2), np.round(p["lamspan"][1], 2)))
        print("      capacity A (front, rear) at v 30 / lam 1.05: %s | v 45 / lam 1.2: %s | v 60 / lam 1.4: %s" % (np.round(A_at(car, 30, 1.05), 3), np.round(A_at(car, 45, 1.2), 3), np.round(A_at(car, 60, 1.4), 3)))
        for l in laps:
            ln = M.line_of(l); ds = ln["ds"]; t_ref = float(np.sum(ds / ln["v"]))
            sat = sat_mask(M, car, l)
            sat_g = np.interp(ln["s"], l["s"], sm(sat.astype(float), 5)) > 0.5
            row = []
            for clamp in (False, True):
                g = []
                for lev in (0.98, 1.0, 1.02):
                    q = copy.deepcopy(p); q["mu1"] = [m * lev for m in q["mu1"]]
                    ln2 = dict(ln)
                    if clamp:
                        ln2["vcap"] = np.where(sat_g, ln["v"], np.inf)
                    sol = M.solve(M.Car(q), ln2, "balanced")
                    g.append((t_ref - sol["time"], t_ref - float(np.sum(ds / np.maximum(sol["v"], ln["v"]))), float((sol["v"] < ln["v"] - 2 / 3.6).mean())))
                row.append(g)
            # where does the unclamped envelope gain sit: on points measured saturated?
            q = copy.deepcopy(p); sol = M.solve(M.Car(q), ln, "balanced")
            dt = ds / ln["v"] - ds / np.maximum(sol["v"], ln["v"])
            print("    L%-2d %.2f | saturated %4.1f%% of lap (%.1f s) | envelope gain 0.98/1.00/1.02: no clamp %+.2f/%+.2f/%+.2f | CLAMP %+.2f/%+.2f/%+.2f | plain model: no clamp %+.2f, clamp %+.2f; under driven >2 km/h on %.0f%% / %.0f%% | gain sitting ON saturated points (no clamp) %.2f s" % (
                l["n"], l["time"], 100 * sat_g.mean(), float(np.sum((ds / ln["v"])[sat_g])), row[0][0][1], row[0][1][1], row[0][2][1], row[1][0][1], row[1][1][1], row[1][2][1],
                row[0][1][0], row[1][1][0], 100 * row[0][1][2], 100 * row[1][1][2], float(dt[sat_g].sum())))
