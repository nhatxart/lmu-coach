"""Fable probe 1: WHERE and WHY is balanced mode slower than the driver?
At the driven state of the best lap: axle capacities Af, Ar from the fitted car,
usage |gy|/Af, |gy|/Ar, slip angles, yaw accel. Read-only.
usage: python fable_p1.py file [file...]"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, sm, G
from il_core import fit_car, line_of, solve, lap_state, N

for name in sys.argv[1:]:
    f = load_file(name)
    car = fit_car([f])
    l = sorted(real_laps(f), key=lambda x: x["time"])[0]
    src = car.p["load_source"]
    st = lap_state(l, src)
    v, gx, gy, gz, gr = st["v"], st["gx"], st["gy"], st["gz"], st["grade"]
    gxt = gx + gr + car.drag(v)
    Af, Ar = car.axles(v, gx, gy, gz)
    lam_meas = st["L"] / np.array(car.p["static"])[:, None]
    lam_mod = car.lam(v, gx, gy, gz)
    ln = line_of(l)
    solb = solve(car, ln, "balanced"); solc = solve(car, ln, "ceiling")
    s = l["s"]
    vb = np.interp(s, ln["s"], solb["v"]); vcl = np.interp(s, ln["s"], solb["vc"]); vce = np.interp(s, ln["s"], solc["v"])
    uF, uR = np.abs(gy) / Af, np.abs(gy) / Ar
    tot = car.sf * Af + car.sr * Ar
    print("lap L%d %.2f  model %s  sf %.3f" % (l["n"], l["time"], f["model"], car.sf))
    print("whole lap, |gy|>0.6*p99: usage quantiles p50/p90/p97/p99.5")
    hard = np.abs(gy) > 0.6 * np.percentile(np.abs(gy), 99)
    for nm, u in (("front", uF), ("rear", uR), ("min-axle", np.maximum(uF, uR)), ("ceiling", np.abs(gy) / tot)):
        print("   %-8s %s   share of hard samples with usage > 1: %.1f%%" % (nm, np.round(np.percentile(u[hard], [50, 90, 97, 99.5]), 3), 100 * (u[hard] > 1).mean()))
    print("usage>1 by speed band (hard samples): band | n | front>1 | rear>1 | median uF uR | median slipF slipR | lamF lamR meas | lamF lamR model | gxt")
    for a, b in ((50, 90), (90, 120), (120, 150), (150, 180), (180, 210), (210, 260)):
        m = hard & (v * 3.6 >= a) & (v * 3.6 < b)
        if m.sum() < 30: continue
        print("   %3d-%3d | %4d | %4.0f%% | %4.0f%% | %.2f %.2f | %.1f %.1f | %.2f %.2f | %.2f %.2f | %+.2f" % (a, b, m.sum(), 100 * (uF[m] > 1).mean(), 100 * (uR[m] > 1).mean(),
              np.median(uF[m]), np.median(uR[m]), np.median(st["slip"][:2].mean(0)[m]), np.median(st["slip"][2:].mean(0)[m]),
              np.median(lam_meas[:2].mean(0)[m]), np.median(lam_meas[2:].mean(0)[m]), np.median(lam_mod[:2].mean(0)[m]), np.median(lam_mod[2:].mean(0)[m]), np.median(gxt[m])))
    print("zones where balanced is > 4 km/h under the driver (50 m rows):")
    print("    s    v_drv v_bal v_cornerlimit v_ceil |  gy    gxt   gz  | Af   Ar  | uF   uR  | slipF slipR | thr brk | yawacc | lamF lamR(meas)")
    step = np.searchsorted(s, np.arange(0, s[-1], 50.0))
    for i in step:
        if vb[i] < v[i] - 4 / 3.6:
            print("  %5.0f  %5.1f %5.1f %5.1f %5.1f | %+.2f %+.2f %+.2f | %.2f %.2f | %.2f %.2f | %4.1f %4.1f | %3.0f %3.0f | %+.2f | %.2f %.2f" % (
                s[i], v[i] * 3.6, vb[i] * 3.6, vcl[i] * 3.6, vce[i] * 3.6, gy[i], gxt[i], gz[i], Af[i], Ar[i], uF[i], uR[i],
                st["slip"][:2].mean(0)[i], st["slip"][2:].mean(0)[i], st["thr"][i], st["brk"][i], st["yawacc"][i], lam_meas[:2].mean(0)[i], lam_meas[2:].mean(0)[i]))
    if st["fy"] is not None:
        Fy, Fz = st["fy"], st["load"]
        share = Fy[:2].sum(0) / Fy.sum(0)
        print("front share of lateral force (hard samples) by condition: all %.3f | thr>90 %.3f | brk>10 %.3f | coasting %.3f" % (
            np.median(share[hard]), np.median(share[hard & (st["thr"] > 90)]), np.median(share[hard & (st["brk"] > 10)]), np.median(share[hard & (st["thr"] < 20) & (st["brk"] < 5)])))
        for a, b in ((50, 90), (90, 120), (120, 150), (150, 180), (180, 210), (210, 260)):
            m = hard & (v * 3.6 >= a) & (v * 3.6 < b)
            if m.sum() > 30: print("   %3d-%3d front Fy share p10/p50/p90 %s" % (a, b, np.round(np.percentile(share[m], [10, 50, 90]), 3)))
    print()
