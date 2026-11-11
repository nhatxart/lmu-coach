"""Why is the balanced model slow in fast corners? Axle proxies by speed band."""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps
from il_core import lap_state, stack, X_load, FRONT, REAR

for name in sys.argv[1:]:
    f = load_file(name); laps = real_laps(f)
    src = "load" if laps[0]["load"] is not None else "susp"
    st = [lap_state(l, src) for l in laps]
    v, gx, gy, gz, gr, yawacc = (stack(st, k) for k in ("v", "gx", "gy", "gz", "grade", "yawacc"))
    L = stack(st, "L"); slip = stack(st, "slip")
    C = np.array([np.linalg.lstsq(X_load(v, gx, gy, gz), L[i], rcond=None)[0] for i in range(4)])
    lam = L / C[:, :1]
    print("=" * 100); print(f["model"], "| loads from", src, "| aero front share %.2f | lam at 200 km/h: front %.2f rear %.2f" % (
        C[:2, 1].sum() / C[:, 1].sum(), 1 + C[:2, 1].sum() * 55.6 ** 2 / C[:2, 0].sum(), 1 + C[2:, 1].sum() * 55.6 ** 2 / C[2:, 0].sum()))
    qs = (np.abs(gx + gr) < 0.2) & (np.abs(yawacc) < 0.4)
    has = st[0]["fy"] is not None
    if has:
        Fy, Fz, gf = stack(st, "fy"), stack(st, "load"), stack(st, "gf")
    print("speed band | n | |gy| p50 p90 | slip F/R p50 | lamF lamR | rF p50 p90 | rR p50 p90" + (" | true muF muR p50 | gfF gfR | front Fy share" if has else ""))
    for a, b in ((60, 90), (90, 120), (120, 150), (150, 180), (180, 210), (210, 250)):
        band = qs & (v * 3.6 >= a) & (v * 3.6 < b)
        if band.sum() < 100: continue
        m = band & (np.abs(gy) > np.percentile(np.abs(gy)[band], 70))        # the hard-cornering part of the band
        rF = np.abs(gy) / lam[:2].mean(0); rR = np.abs(gy) / lam[2:].mean(0)
        s = "%3d-%3d | %5d | %.2f %.2f | %.1f / %.1f | %.2f %.2f | %.2f %.2f | %.2f %.2f" % (a, b, m.sum(), np.median(np.abs(gy)[m]), np.percentile(np.abs(gy)[m], 90),
            np.median(slip[:2].mean(0)[m]), np.median(slip[2:].mean(0)[m]), np.median(lam[:2].mean(0)[m]), np.median(lam[2:].mean(0)[m]),
            np.median(rF[m]), np.percentile(rF[m], 90), np.median(rR[m]), np.percentile(rR[m], 90))
        if has:
            s += " | %.2f %.2f | %.2f %.2f | %.3f" % (np.median((np.abs(Fy[:2].sum(0)) / Fz[:2].sum(0))[m]), np.median((np.abs(Fy[2:].sum(0)) / Fz[2:].sum(0))[m]),
                                                  np.median(gf[:2].mean(0)[m]), np.median(gf[2:].mean(0)[m]), np.median((Fy[:2].sum(0) / Fy.sum(0))[m]))
        print(s)
    # outer vs inner tyre slip and load in hard cornering
    hard = qs & (np.abs(gy) > 1.2)
    right = gy > 0
    for nm, m in (("gy>0", hard & right), ("gy<0", hard & ~right)):
        print("  %s n=%d: lam FL FR RL RR %s | slip %s" % (nm, m.sum(), np.round(np.median(lam[:, m], axis=1), 2), np.round(np.median(slip[:, m], axis=1), 1)))
