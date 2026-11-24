"""Opus probe 8 (real data, code of this worktree = ideal-lap-kt 5958bfb): is kt at q 0.75
over-optimistic? Same-file fit as build_check_step3 --clamp (R5). Read-only on recordings.
(a) observed drive envelope: all samples of ALL fit laps with throttle > 30 %, tyre demand
    gxt > 0.2 g, v < 45 m/s, |gy| < 0.8 A_r; per bin (v 5 m/s x |gy| 0.3 g): n, TC-active share,
    achieved gxt p50/p95/p99/max, and the model's traction limit kt s_r room (median over the
    bin's own samples) at kt = median and kt = q 0.75. The model exceeds what the car ever did
    if model > p99 (or > max) in a bin.
(b) the fastest lap, A1 on, clamp on: traction-bound share of the ideal (expected level) and of
    the plain model, kt median vs q 0.75, and the gain per corner window that moves.
(c) the TC samples' ratio gxt / (s_r room): log sd, q75/q50, share with |gy|/A_r > 0.5,
    throttle when TC is active.
usage: python opus_p8_trac.py [file ...]"""
import sys, os
import numpy as np
from build_check_common import load, reduction, PORSCHE, FERRARI, WT
sys.path.insert(0, str(WT / "backend" / "tools"))
from app import ideal_car as C, ideal_solver as SV  # noqa: E402
from app.ideal_lap import analyse_lap  # noqa: E402
import ideal_lap_report as R  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for name in sys.argv[1:] or [PORSCHE, FERRARI]:
    d = load(name)
    g = C.fit_group([reduction(d)], d["cls"]); car75 = g.car_for(d["file"])
    st = d["states"]
    cat = lambda k: np.concatenate([getattr(s, k) for s in st])
    v, gx, gy, gz, gr, thr, tc = (cat(k) for k in ("v", "gx", "gy", "gz", "grade", "thr", "tc"))
    gxt = gx + gr + car75.drag(v)
    Af, Ar = car75.axles(v, gx, gy, gz)
    sr = car75.sr
    room = np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 1e-4))
    rest = (gxt > 0.2) & (np.abs(gy) < 0.8 * Ar)
    trl = (tc > 0.5) & (thr > 30) & (v < 45) & rest
    ratio = gxt[trl] / (sr * room[trl])
    k50, k75 = float(np.median(ratio)), float(np.quantile(ratio, 0.75))
    print("=" * 110)
    print("%s | %d laps | kt fit %.3f | TC samples %d: kt q50 %.3f q75 %.3f (x%.3f) | log sd %.3f | |gy|/Ar>0.5 %.0f%% | thr when TC p10/p50 %.0f/%.0f %%" % (
        d["model"], len(st), car75.kt, trl.sum(), k50, k75, k75 / k50, np.log(ratio).std(),
        100 * np.mean(np.abs(gy[trl]) / Ar[trl] > 0.5), *np.percentile(thr[trl], [10, 50])))
    drive = (thr > 30) & (v < 45) & rest
    print("  (a) bin v | |gy|  n   TC%   achieved gxt p50/p95/p99/max   | model kt50  kt75  (g)  | kt75 > p99?")
    over = []
    for v0 in range(5, 45, 5):
        for y0 in (0.0, 0.3, 0.6, 0.9, 1.2):
            m = drive & (v >= v0) & (v < v0 + 5) & (np.abs(gy) >= y0) & (np.abs(gy) < y0 + 0.3)
            if m.sum() < 40:
                continue
            a = gxt[m]; cap = sr * room[m]
            m50, m75 = float(np.median(k50 * cap)), float(np.median(k75 * cap))
            p = np.percentile(a, [50, 95, 99]); tcs = float(np.mean(tc[m] > 0.5))
            flag = "OVER max" if m75 > a.max() else ("over p99" if m75 > p[2] else "")
            if tcs >= 0.2:
                over.append((m75 / p[2], m50 / p[2], m75 / a.max()))
            print("      %2d-%2d | %.1f-%.1f %5d %4.0f%%  %.3f / %.3f / %.3f / %.3f | %.3f  %.3f | %s" % (
                v0, v0 + 5, y0, y0 + 0.3, m.sum(), 100 * tcs, *p, a.max(), m50, m75, flag))
    if over:
        o = np.array(over)
        print("  (a) bins with TC share >= 20 %%: model/p99 kt75 median %.3f max %.3f | kt50 median %.3f | kt75/max median %.3f max %.3f" % (
            np.median(o[:, 0]), o[:, 0].max(), np.median(o[:, 1]), np.median(o[:, 2]), o[:, 2].max()))
    # (b)
    info = R.fit_info(g, len(st))
    lap = sorted(d["raws"], key=lambda r: r["time"])[0]
    res = {}
    for nm, kt in (("kt50", k50), ("kt75", car75.kt)):
        dd = car75.to_dict(); dd["kt"] = kt; cv = C.CarModel.from_dict(dd)
        r = analyse_lap(cv, info, lap, lap["time"], track_name=lap.get("track_name", ""))
        lv = r["_levels"]
        b_ideal = r["diagnostics"]["binding_share"]["traction"]
        b_plain = float(np.mean(np.asarray(lv["plain"]["binding"]) == "traction"))
        res[nm] = r
        print("  (b) %s kt %.3f | gain %.3f (min %.3f) | plain %.3f | traction-bound: ideal %.1f%% plain %.1f%% | ceiling %.3f" % (
            nm, kt, lv["gain"], lv["gain_min"], lv["t_ref_grid"] - lv["model_only_time"], 100 * b_ideal, 100 * b_plain,
            r["ceiling"]["gain_s"]))
    c50 = {c["n"]: c for c in res["kt50"]["corners"]}
    moved = sorted(((c["gain_s"] - c50[c["n"]]["gain_s"], c) for c in res["kt75"]["corners"] if c["n"] in c50), key=lambda x: -abs(x[0]))[:6]
    print("  (b) corners where the gain moves most (kt75 - kt50): " + "; ".join(
        "T%s %s %+.3f s (limit %s, where %s, vmin %s km/h)" % (c.get("n"), c.get("name") or "", dg, c.get("limit", c.get("binding_at_min")),
                                                                c.get("where"), c.get("vmin_ideal_kmh")) for dg, c in moved))
    print("  (b) corner keys:", sorted(res["kt75"]["corners"][0].keys()) if res["kt75"]["corners"] else "no corners")
    tot = sum(c["gain_s"] for c in res["kt75"]["corners"]) - sum(c["gain_s"] for c in res["kt50"]["corners"])
    print("  (b) gain moved inside corner windows %+.3f s of %+.3f s total" % (tot, res["kt75"]["_levels"]["gain"] - res["kt50"]["_levels"]["gain"]))
