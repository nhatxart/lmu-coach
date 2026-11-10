"""Feasibility: theoretical best lap from tyre/load data. Read-only.
usage: python feas.py <backend_dir> <file.duckdb>"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, sys.argv[1])
from app.telemetry import load_telemetry_file

G = 9.81
W4 = ("FL", "FR", "RL", "RR")
laps = [l for l in load_telemetry_file(sys.argv[2], grid_hz=50.0) if l.is_complete]
best = min(laps, key=lambda l: l.lap_time_s)
real = [l for l in laps if l.lap_time_s < best.lap_time_s * 1.05]
print("FILE", sys.argv[2][-62:]); print("model", best.meta.get("VehicleModel"), "| laps used", len(real), "| best %.3f (L%d)" % (best.lap_time_s, best.lap_number))


def ch(l, k):
    return np.nan_to_num(np.asarray(l.channels[k], float))


SM = 10   # 0.2 s: kerb strikes and bumps are not what the model predicts


def cat(k, ls, sm=False):
    f = (lambda a: np.convolve(a, np.ones(SM) / SM, mode="same")) if sm else (lambda a: a)
    return np.concatenate([f(ch(l, k)) for l in ls])


def feats(ls):
    v = cat("Ground Speed", ls, True) / 3.6
    ax, ay, az = cat("G Force Long", ls, True), cat("G Force Lat", ls, True), cat("G Force Vert", ls, True)
    return v, ax, ay, az


def X_of(v, ax, ay, az):
    return np.c_[np.ones_like(v), v * v, ax, ay, az, az * v * v]


# ── 1. wheel load from speed and G ───────────────────────────────────────────
train, test = real[0::2], real[1::2]
load_fit = {}
for src in ("Tire Load", "Susp Force"):
    if f"{src} FL" not in best.channels or np.nanmax(np.abs(ch(best, f"{src} FL"))) < 1:
        print(f"[load] {src}: not recorded"); continue
    Xa, Xb = X_of(*feats(train)), X_of(*feats(test))
    out = []
    for w in W4:
        ya, yb = cat(f"{src} {w}", train, True), cat(f"{src} {w}", test, True)
        c, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        p = Xb @ c
        r2 = 1 - np.sum((yb - p) ** 2) / np.sum((yb - yb.mean()) ** 2)
        out.append("%s R2 %.3f rmse %.0f N (%.0f%% of mean)" % (w, r2, np.sqrt(np.mean((yb - p) ** 2)), 100 * np.sqrt(np.mean((yb - p) ** 2)) / yb.mean()))
        load_fit[(src, w)] = c
    print(f"[load] {src} = c0 + c1 v^2 + c2 longG + c3 latG, held-out laps:\n      " + "\n      ".join(out))
    c = np.array([load_fit[(src, w)] for w in W4])
    print("      static sum %.0f N (=%.0f kg) | front share %.3f | downforce at 200 km/h %.0f N | front aero share %.2f" % (
        c[:, 0].sum(), c[:, 0].sum() / G, c[:2, 0].sum() / c[:, 0].sum(), c[:, 1].sum() * (200 / 3.6) ** 2, c[:2, 1].sum() / c[:, 1].sum()))

# ── 2. what the car has shown: envelopes ─────────────────────────────────────
v, ax, ay, az = feats(real)
thr, brk = cat("Throttle Pos", real), cat("Brake Pos", real)
if thr.max() <= 1.5: thr = thr * 100
if brk.max() <= 1.5: brk = brk * 100
kmh = v * 3.6
edges = np.arange(60, 320, 20)


def binned(mask, val, q, min_n=80):
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = mask & (kmh >= a) & (kmh < b)
        if m.sum() >= min_n:
            xs.append((a + b) / 2 / 3.6); ys.append(np.percentile(val[m], q)); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns


xl, yl, nl = binned((np.abs(ax) < 0.3) & (np.abs(ay) > 0.8), np.abs(ay), 95)
xb, yb_, nb = binned(brk > 80, ax, 90)
xt, yt, nt = binned((thr > 98) & (np.abs(ay) < 0.3), -ax, 50)
cl = np.polyfit(xl ** 2, yl, 1); cb = np.polyfit(xb ** 2, yb_, 1)
print("[env] cornering p95 |latG| by speed (km/h: g, n):", ", ".join("%d: %.2f (%d)" % (x * 3.6, y, n) for x, y, n in zip(xl, yl, nl)))
print("      fit latmax = %.2f + %.2f*(v/200kmh)^2" % (cl[1], cl[0] * (200 / 3.6) ** 2))
print("[env] braking p90 longG, brake>80%%:", ", ".join("%d: %.2f (%d)" % (x * 3.6, y, n) for x, y, n in zip(xb, yb_, nb)))
print("      fit brakemax = %.2f + %.2f*(v/200kmh)^2 | brake>80%% samples %d (%.1f s)" % (cb[1], cb[0] * (200 / 3.6) ** 2, (brk > 80).sum(), (brk > 80).sum() / 50))
print("[env] full-throttle accel median g:", ", ".join("%d: %.2f (%d)" % (x * 3.6, y, n) for x, y, n in zip(xt, yt, nt)))
lat_env = lambda u: np.interp(u, xl, yl)
brk_env = lambda u: np.interp(u, xb, yb_)
acc_env = lambda u: np.interp(u, xt, yt)
A_TR = float(np.percentile(-ax[(thr > 98) & (kmh < 140)], 95)) if ((thr > 98) & (kmh < 140)).sum() > 50 else float(yt.max())
print("      traction limit (p95 full-throttle accel below 140 km/h): %.2f g" % A_TR)

# ── 3. per-tyre grip vs load (cars that record tyre forces) ──────────────────
mu_fit = None
if ("Tire Load", "FL") in load_fit and "Tire Lateral Force FL" in best.channels:
    mu_fit = {}
    print("[tyre] peak lateral mu (p95 of |Fy|/Fz, |longG|<0.25, speed>60) by load:")
    for w in W4:
        fz, fy = cat(f"Tire Load {w}", real, True), np.abs(cat(f"Tire Lateral Force {w}", real, True))
        m = (fz > 500) & (np.abs(ax) < 0.25)
        mu = fy[m] / fz[m]; fzm = fz[m]
        qs = np.percentile(fzm, np.linspace(40, 100, 7))   # the loaded half: where the limit is reached
        xs, ys = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            mm = (fzm >= a) & (fzm < b)
            if mm.sum() > 100: xs.append(np.mean(fzm[mm])); ys.append(np.percentile(mu[mm], 98))
        c = np.polyfit(xs, ys, 1); mu_fit[w] = c
        print("      %s: " % w + ", ".join("%.1f kN: %.2f" % (x / 1000, y) for x, y in zip(xs, ys)) + " | mu = %.2f %+.3f per kN" % (c[1], c[0] * 1000))
    # do the recorded forces add up to the car's lateral acceleration?
    fy_sum = sum(cat(f"Tire Lateral Force {w}", real, True) for w in W4)
    mass = sum(load_fit[("Tire Load", w)][0] for w in W4) / G
    k = np.polyfit(ay, fy_sum, 1)
    print("      check: sum Fy vs latG slope %.0f N/g -> mass %.0f kg (static loads say %.0f kg), corr %.3f" % (abs(k[0]), abs(k[0]) / G, mass, abs(np.corrcoef(ay, fy_sum)[0, 1])))

# ── 4. prototype lap simulation on the best lap's line ───────────────────────
bv = ch(best, "Ground Speed") / 3.6
yaw = ch(best, "Yaw Rate")
t = np.asarray(best.time_s, float)
s = np.concatenate([[0], np.cumsum(0.5 * (bv[1:] + bv[:-1]) * np.diff(t))])
k_raw = np.abs(yaw) / np.maximum(bv, 5)
k_raw = np.convolve(k_raw, np.ones(15) / 15, mode="same")          # 0.3 s
DS = 2.0
sg = np.arange(0, s[-1], DS)
kap = np.interp(sg, s, k_raw); v_act = np.interp(sg, s, bv)
kv = np.interp(sg, s, np.convolve(ch(best, "G Force Vert"), np.ones(25) / 25, mode="same") * G / np.maximum(bv, 10) ** 2)   # vertical G per v^2
print("[track] vertical G on the best lap (0.5 s mean): min %.2f max %.2f" % ((kv * v_act ** 2 / G).min(), (kv * v_act ** 2 / G).max()))
t_act = np.sum(DS / v_act)


def simulate(latf0, brkf0, label, uses_vert=False):
    cur = {"i": slice(None)}
    vg = lambda u: np.clip(kv[cur["i"]] * np.asarray(u) ** 2 / G, -0.7, 1.0)
    latf = (lambda u: latf0(u, vg(u))) if uses_vert else latf0
    brkf = (lambda u: brkf0(u, vg(u))) if uses_vert else brkf0
    n = len(sg); vc = np.full(n, 110.0)
    for _ in range(30):
        vc = np.minimum(110.0, np.sqrt(G * latf(vc) / np.maximum(kap, 1e-6)))
    vs = vc.copy()
    for _ in range(2):
        for i in list(range(n)) + [0]:                       # forward: accelerate
            j = (i + 1) % n; cur["i"] = i
            u = min(1.0, vs[i] ** 2 * kap[i] / (G * latf(vs[i])))
            a = min(acc_env(vs[i]), A_TR * np.sqrt(1 - u * u)) if acc_env(vs[i]) > 0 else acc_env(vs[i])
            vs[j] = min(vs[j], np.sqrt(max(vs[i] ** 2 + 2 * G * a * DS, 1.0)))
        for i in list(range(n - 1, -1, -1)) + [n - 1]:       # backward: brake
            j = (i - 1) % n; cur["i"] = i
            u = min(1.0, vs[i] ** 2 * kap[i] / (G * latf(vs[i])))
            a = brkf(vs[i]) * np.sqrt(1 - u * u)
            vs[j] = min(vs[j], np.sqrt(vs[i] ** 2 + 2 * G * a * DS))
    cur["i"] = slice(None)
    tt = np.sum(DS / vs)
    d = np.cumsum(DS / v_act - DS / vs)                       # time gained along the lap
    # where: 500 m blocks
    blocks = [(int(a), d[min(n - 1, int((a + 500) / DS))] - d[int(a / DS)]) for a in np.arange(0, sg[-1] - 500, 500)]
    top = sorted(blocks, key=lambda b: -b[1])[:4]
    print("[sim] %-34s %.2f s vs driven %.2f s (integral of the same grid) -> %.2f s (%.1f%%) faster; vmax %.0f vs %.0f km/h; min corner %.0f vs %.0f" % (
        label, tt, t_act, t_act - tt, 100 * (t_act - tt) / t_act, vs.max() * 3.6, v_act.max() * 3.6, vs.min() * 3.6, v_act.min() * 3.6))
    print("      biggest gains per 500 m: " + ", ".join("%d m: %.2f s" % b for b in top) + " | slower than driven on %.0f%% of the lap" % (100 * np.mean(vs < v_act - 0.5)))
    return vs


simulate(lat_env, brk_env, "proven grip (p95 by speed, no fit)")

if mu_fit:
    C = {w: load_fit[("Tire Load", w)] for w in W4}
    mass = sum(C[w][0] for w in W4) / G
    fshare = (C["FL"][0] + C["FR"][0]) / (mass * G)

    def loads(u, axg, ayg, azg=0.0):
        return {w: np.maximum(0.0, C[w][0] + C[w][1] * u * u + C[w][2] * axg + C[w][3] * ayg + C[w][4] * azg + C[w][5] * azg * u * u) for w in W4}

    def cap(fz, ws):
        return sum((mu_fit[w][1] + mu_fit[w][0] * fz[w]) * fz[w] for w in ws)

    def lat_tyre(u, balanced, azg=0.0):
        u = np.asarray(u, float); res = []
        for sgn in (1.0, -1.0):
            a = np.full_like(u, 1.5)
            for _ in range(40):
                fz = loads(u, 0.0, sgn * a, azg)
                if balanced:
                    new = np.minimum(cap(fz, ("FL", "FR")) / (fshare * mass * G), cap(fz, ("RL", "RR")) / ((1 - fshare) * mass * G))
                else:
                    new = cap(fz, W4) / (mass * G)
                a = 0.5 * a + 0.5 * new
            res.append(a)
        return np.maximum(0.5, 0.5 * (res[0] + res[1]))

    def brk_tyre(u, azg=0.0):
        u = np.asarray(u, float); a = np.full_like(u, 1.5)
        for _ in range(40):
            a = 0.5 * a + 0.5 * cap(loads(u, a, 0.0, azg), W4) / (mass * G)
        return np.maximum(0.5, a)

    us = np.array([80, 120, 160, 200, 240, 280]) / 3.6
    print("[tyre] cornering limit g at 80..280 km/h: ceiling " + " ".join("%.2f" % x for x in lat_tyre(us, False)) + " | balanced " + " ".join("%.2f" % x for x in lat_tyre(us, True)) + " | driven p95 fit " + " ".join("%.2f" % x for x in lat_env(us)))
    print("[tyre] braking limit g (tyres only, no drag): " + " ".join("%.2f" % x for x in brk_tyre(us)) + " | driven p90 fit " + " ".join("%.2f" % x for x in brk_env(us)))
    from scipy.interpolate import RegularGridInterpolator as RGI
    ug, zg = np.linspace(10, 110, 41), np.linspace(-0.7, 1.0, 18)
    UU, ZZ = np.meshgrid(ug, zg, indexing="ij")
    tabs = {k: RGI((ug, zg), f(UU, ZZ), bounds_error=False, fill_value=None) for k, f in (("bal", lambda a, b: lat_tyre(a, True, b)), ("max", lambda a, b: lat_tyre(a, False, b)), ("brk", brk_tyre))}
    T = lambda k: (lambda u, z: tabs[k](np.c_[np.clip(np.atleast_1d(u), 10, 110), np.atleast_1d(z)]).reshape(np.shape(u)) if np.ndim(u) else float(tabs[k]([[min(max(u, 10), 110), float(z)]])[0]))
    simulate(lambda u: T("bal")(u, np.zeros(np.shape(u)) if np.ndim(u) else 0.0), lambda u: T("brk")(u, np.zeros(np.shape(u)) if np.ndim(u) else 0.0), "per-tyre grip, balanced, flat track")
    vsb = simulate(T("bal"), T("brk"), "per-tyre grip, balanced, with hills", True)
    simulate(T("max"), T("brk"), "per-tyre, all four at max, hills", True)
    # where the model is slower than the driver: its grip is underestimated there
    lo = vsb < v_act - 0.5
    print("      slower-than-driven points: median speed %.0f km/h, median |latG demand| %.2f, share in braking (driven) %.0f%%" % (np.median(v_act[lo]) * 3.6, np.median(v_act[lo] ** 2 * kap[lo] / G), 100 * np.mean(np.gradient(v_act)[lo] < -0.05)))
