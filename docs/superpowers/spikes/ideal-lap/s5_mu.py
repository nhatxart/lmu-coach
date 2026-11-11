"""Per-tyre saturated mu on a car with forces: load or speed (or temperature)?"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, sm, W4
from il_core import lap_state, stack, X_load, N

files = [load_file(x) for x in sys.argv[1:]]
laps = [l for f in files for l in real_laps(f)]
st = [lap_state(l, "load") for l in laps]
v, gx, gy, gz, gr = (stack(st, k) for k in ("v", "gx", "gy", "gz", "grade"))
Fz, Fx, Fy, gf = (stack(st, k) for k in ("load", "fx", "fy", "gf"))
print(files[0]["model"], "| files", len(files), "| laps", len(laps))
C0 = np.array([np.linalg.lstsq(X_load(v, gx, gy, gz), Fz[i], rcond=None)[0][0] for i in range(4)])
mu = np.hypot(Fx, Fy) / np.maximum(Fz, 300); lam = Fz / C0[:, None]
ang = np.degrees(np.arctan2(np.abs(Fx), np.abs(Fy)))
for i, w in enumerate(W4):
    s = (gf[i] < 0.25) & (Fz[i] > 1500) & (ang[i] < 30)
    A = np.c_[np.ones(s.sum()), lam[i][s] - 1, (v[s] - 40) / 40]
    c = np.linalg.lstsq(A, mu[i][s], rcond=None)[0]
    c1 = np.linalg.lstsq(A[:, :2], mu[i][s], rcond=None)[0]; c2 = np.linalg.lstsq(A[:, [0, 2]], mu[i][s], rcond=None)[0]
    res = lambda cc, AA: np.std(mu[i][s] - AA @ cc)
    print("%s lateral-saturated n=%d corr(lam,v) %.2f | mu = %.3f %+.3f*(lam-1) %+.3f*(v-40)/40 [sd %.3f] | load only: %.3f %+.3f [sd %.3f] | speed only: %.3f %+.3f [sd %.3f] | raw sd %.3f" % (
        w, s.sum(), np.corrcoef(lam[i][s], v[s])[0, 1], *c, res(c, A), *c1, res(c1, A[:, :2]), *c2, res(c2, A[:, [0, 2]]), mu[i][s].std()))
    # 2D table: mean mu by load band x speed band
    lq = np.percentile(lam[i][s], [0, 33, 66, 100]); vq = [15, 30, 42, 55, 90]
    rows = []
    for a, b in zip(lq[:-1], lq[1:]):
        rows.append("lam %.1f-%.1f: " % (a, b) + " ".join("%s" % ("%.2f(%d)" % (mu[i][s & (lam[i] >= a) & (lam[i] <= b) & (v >= c_) & (v < d_)].mean(), (s & (lam[i] >= a) & (lam[i] <= b) & (v >= c_) & (v < d_)).sum()) if (s & (lam[i] >= a) & (lam[i] <= b) & (v >= c_) & (v < d_)).sum() > 15 else "  -  ") for c_, d_ in zip(vq[:-1], vq[1:])))
    print("     speed bands (km/h) 54-108 | 108-151 | 151-198 | 198-324:  " + "  ||  ".join(rows))
