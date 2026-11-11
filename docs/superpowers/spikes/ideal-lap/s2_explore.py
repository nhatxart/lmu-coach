"""Stage 2 exploration: load model variants; tyre limit with forces (A) and without (B)."""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, cat, sm, G, HZ, W4

N = 10  # 0.2 s


def grade_of(l, n=25):
    if l["posy"] is None:
        return np.zeros(len(l["v"]))
    return np.clip(np.gradient(sm(l["posy"], n), 1 / HZ) / np.maximum(sm(l["v"], n), 5), -0.3, 0.3)


def state(laps):
    v = cat(laps, "v", N); gx = cat(laps, "gx", N); gy = cat(laps, "gy", N); gz = cat(laps, "gz", N)
    gr = np.concatenate([sm(grade_of(l), N) for l in laps])
    return v, gx, gy, gz, gr


FEATS = {
    "F0 1,v2,gx,gy,gz": lambda v, gx, gy, gz, gr: np.c_[np.ones_like(v), v * v, gx, gy, gz],
    "F1 gx->gx+grade": lambda v, gx, gy, gz, gr: np.c_[np.ones_like(v), v * v, gx + gr, gy, gz],
    "F2 F1+gz*v2": lambda v, gx, gy, gz, gr: np.c_[np.ones_like(v), v * v, gx + gr, gy, gz, gz * v * v],
    "F3 F1+gx*v2,|gy|v2": lambda v, gx, gy, gz, gr: np.c_[np.ones_like(v), v * v, gx + gr, gy, gz, (gx + gr) * v * v, np.abs(gy) * v * v],
    "F4 F3+gy*gx,gy^2": lambda v, gx, gy, gz, gr: np.c_[np.ones_like(v), v * v, gx + gr, gy, gz, (gx + gr) * v * v, np.abs(gy) * v * v, gy * (gx + gr), gy * gy],
}

for name in sys.argv[1:]:
    f = load_file(name); laps = real_laps(f)
    print("=" * 110); print(f["file"][:64], "|", f["model"], "|", f["cls"], "| laps", len(laps))
    tr, te = laps[0::2], laps[1::2]
    Sa, Sb = state(tr), state(te)
    for src in ("load", "susp"):
        if laps[0][src] is None: continue
        print(f"[load model] {src}: held-out R2 per wheel (rmse %% of mean)")
        for fn, F in FEATS.items():
            Xa, Xb = F(*Sa), F(*Sb); o = []
            for i in range(4):
                ya, yb = cat(tr, src, N, i), cat(te, src, N, i)
                c, *_ = np.linalg.lstsq(Xa, ya, rcond=None); p = Xb @ c
                o.append("%.3f (%.1f%%)" % (1 - np.sum((yb - p) ** 2) / np.sum((yb - yb.mean()) ** 2), 100 * np.sqrt(np.mean((yb - p) ** 2)) / yb.mean()))
            print("   %-22s " % fn + "  ".join(o))
    v, gx, gy, gz, gr = state(laps)
    if laps[0]["fx"] is not None:
        Fz = np.vstack([cat(laps, "load", N, i) for i in range(4)]); Fx = np.vstack([cat(laps, "fx", N, i) for i in range(4)]); Fy = np.vstack([cat(laps, "fy", N, i) for i in range(4)])
        gf = np.vstack([cat(laps, "gf", N, i) for i in range(4)])
        mu = np.hypot(Fx, Fy) / np.maximum(Fz, 300)
        print("[A] per tyre: combined mu vs grip fraction (mean mu | share of samples), load > 2 kN")
        bands = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .8), (.8, 1.01)]
        for i, w in enumerate(W4):
            ok = Fz[i] > 2000
            print("   %s " % w + "  ".join("%.1f-%.1f: %.2f|%2.0f%%" % (a, b, mu[i][ok & (gf[i] >= a) & (gf[i] < b)].mean() if (ok & (gf[i] >= a) & (gf[i] < b)).any() else np.nan, 100 * np.mean((gf[i] >= a) & (gf[i] < b))) for a, b in bands))
        print("[A] saturated samples (grip fraction < 0.25): mu vs load, and by force direction")
        for i, w in enumerate(W4):
            s = (gf[i] < 0.25) & (Fz[i] > 2000)
            ang = np.degrees(np.arctan2(np.abs(Fx[i]), np.abs(Fy[i])))      # 0 = pure lateral, 90 = pure longitudinal
            k = np.polyfit(Fz[i][s] / 1000, mu[i][s], 1)
            qs = np.percentile(Fz[i][s], [0, 25, 50, 75, 100])
            print("   %s n=%d  mu = %.2f %+.3f/kN | by load quartile: " % (w, s.sum(), k[1], k[0]) + ", ".join("%.1f kN: %.2f" % ((a + b) / 2000, mu[i][s & (Fz[i] >= a) & (Fz[i] <= b)].mean()) for a, b in zip(qs[:-1], qs[1:]))
                  + " | lateral(<25deg) %.2f, mixed %.2f, braking(>65deg) %.2f, traction(>65deg) %.2f" % (
                      mu[i][s & (ang < 25)].mean(), mu[i][s & (ang >= 25) & (ang <= 65)].mean(),
                      mu[i][s & (ang > 65) & (gx < 0)].mean() if (s & (ang > 65) & (gx < 0)).sum() > 20 else np.nan,
                      mu[i][s & (ang > 65) & (gx > 0)].mean() if (s & (ang > 65) & (gx > 0)).sum() > 20 else np.nan))
        # axle shares in quasi-steady cornering
        yawacc = cat(laps, "yawacc", N)
        qs_ = (np.abs(gy) > 1.0) & (np.abs(gx + gr) < 0.2) & (np.abs(yawacc) < 0.15)
        fs = Fy[:2].sum(0) / Fy.sum(0)
        print("[A] quasi-steady corners (%d samples): front share of lateral force median %.3f (p10 %.3f, p90 %.3f); front share of load there %.3f" % (qs_.sum(), np.median(fs[qs_]), *np.percentile(fs[qs_], [10, 90]), np.median((Fz[:2].sum(0) / Fz.sum(0))[qs_])))
    # ── B: without forces. Axle proxy r = |gy| / mean load ratio of the axle, against the axle's slip angle
    src = "load" if laps[0]["load"] is not None else "susp"
    X = FEATS["F1 gx->gx+grade"](v, gx, gy, gz, gr)
    L = np.vstack([cat(laps, src, N, i) for i in range(4)])
    c0 = np.array([np.linalg.lstsq(X, L[i], rcond=None)[0][0] for i in range(4)])
    lam = L / c0[:, None]
    slip = np.degrees(np.abs(np.vstack([cat(laps, "slip", N, i) for i in range(4)])))
    qs_ = (np.abs(gy) > 0.5) & (np.abs(gx + gr) < 0.2) & (v > 18)
    print("[B] static loads (%s) %s | axle proxy r = |gy| / mean lambda, by axle slip angle (deg): median | p90 | n" % (src, np.round(c0)))
    for ax_, idx in (("front", (0, 1)), ("rear", (2, 3))):
        a = slip[list(idx)].mean(0); r = np.abs(gy) / lam[list(idx)].mean(0)
        edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 14]
        print("   %-5s " % ax_ + "  ".join("%d-%d: %.2f|%.2f|%d" % (lo, hi, np.median(r[qs_ & (a >= lo) & (a < hi)]), np.percentile(r[qs_ & (a >= lo) & (a < hi)], 90), (qs_ & (a >= lo) & (a < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:]) if (qs_ & (a >= lo) & (a < hi)).sum() > 30))
        if laps[0]["fx"] is not None:
            mu_ax = np.abs(Fy[list(idx)].sum(0)) / Fz[list(idx)].sum(0)
            gfa = gf[list(idx)].mean(0)
            print("         true axle mu:   " + "  ".join("%d-%d: %.2f|gf %.2f" % (lo, hi, np.median(mu_ax[qs_ & (a >= lo) & (a < hi)]), np.median(gfa[qs_ & (a >= lo) & (a < hi)])) for lo, hi in zip(edges[:-1], edges[1:]) if (qs_ & (a >= lo) & (a < hi)).sum() > 30))
