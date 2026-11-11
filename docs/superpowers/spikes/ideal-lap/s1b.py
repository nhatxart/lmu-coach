import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, sm, G, HZ

f = load_file(sys.argv[1]); l = min(real_laps(f), key=lambda x: x["time"])
v, gx, gz, t = l["v"], l["gx"], l["gz"], l["t"]
print("dt unique", np.unique(np.round(np.diff(t), 4))[:5], "n", len(t))
print("v first 30 (km/h):", np.round(v[1000:1030] * 3.6, 2))
print("repeats in v: %.0f%% of consecutive samples equal" % (100 * np.mean(np.diff(v) == 0)))
for n in (1, 5, 10, 25, 50):
    dv = np.gradient(sm(v, n), 1 / HZ) / G; g2 = sm(gx, n)
    best = max(((np.corrcoef(np.roll(dv, k)[100:-100], g2[100:-100])[0, 1], k) for k in range(-25, 26)))
    print("smooth %2d: corr %.4f at lag 0, best %.4f at lag %d samples; sd dv %.3f sd gx %.3f" % (n, np.corrcoef(dv[100:-100], g2[100:-100])[0, 1], best[0], best[1], dv.std(), g2.std()))
# speed from position
if l["posx"] is not None:
    px, py, pz = (sm(l[k], 5) for k in ("posx", "posy", "posz"))
    vp = np.sqrt(np.gradient(px, 1 / HZ) ** 2 + np.gradient(py, 1 / HZ) ** 2 + np.gradient(pz, 1 / HZ) ** 2)
    vh = np.sqrt(np.gradient(px, 1 / HZ) ** 2 + np.gradient(pz, 1 / HZ) ** 2)
    print("speed from position vs Ground Speed: 3D ratio %.4f, horizontal ratio %.4f, corr %.5f" % (np.median(vp[100:-100] / v[100:-100]), np.median(vh[100:-100] / v[100:-100]), np.corrcoef(vp[100:-100], v[100:-100])[0, 1]))
    for n in (10, 25):
        dvp = np.gradient(sm(vp, n), 1 / HZ) / G
        print("  d(pos speed)/dt vs gx smooth %d: corr %.4f slope %.3f" % (n, np.corrcoef(dvp[100:-100], sm(gx, n)[100:-100])[0, 1], np.polyfit(sm(gx, n)[100:-100], dvp[100:-100], 1)[0]))
    # grade and vertical curvature
    n = 25
    vy = np.gradient(sm(l["posy"], n), 1 / HZ)
    th = np.arcsin(np.clip(vy / np.maximum(sm(vp, n), 5), -0.5, 0.5))
    gz_pred = sm(vp, n) * np.gradient(sm(th, n), 1 / HZ) / G           # v * dtheta/dt
    gzs = sm(gz, n)
    k = np.polyfit(gz_pred[200:-200], gzs[200:-200], 1)
    print("gz vs v*dtheta/dt: slope %.3f offset %.3f corr %.3f" % (k[0], k[1], np.corrcoef(gz_pred[200:-200], gzs[200:-200])[0, 1]))
    dvp = np.gradient(sm(vp, n), 1 / HZ) / G
    res = sm(gx, n) - dvp
    k = np.polyfit(np.sin(th)[200:-200], res[200:-200], 1)
    print("gx - d(speed)/dt vs sin(grade): slope %.3f corr %.3f | residual sd %.4f g" % (k[0], np.corrcoef(np.sin(th)[200:-200], res[200:-200])[0, 1], res[200:-200].std()))
    print("grade %.3f..%.3f" % (np.sin(th).min(), np.sin(th).max()))
