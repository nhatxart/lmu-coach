"""Ideal-lap spike core: car model fitted from recordings, and the lap solver.

Units: accelerations in g. gx kinematic longitudinal (+ speeding up), gy lateral
as recorded, gz vertical (crest negative, compression positive), grade = dh/ds.
gxt = longitudinal force the tyres supply, in g: gx + grade + drag(v).
lam = wheel load / its static value. Axle "A" = the lateral g the car holds when
that axle is saturated = 1/2 * sum over its two tyres of mu(lam) * lam,
mu(lam, v) = mu1 * (1 - k * (lam - 1) - c * (v - 40) / 40): the game's tyres lose grip with
speed (measured on Hypercar forces: about -0.11..-0.28 per 40 m/s), and only weakly with load.
"""
import numpy as np
from il_data import sm, G, HZ, real_laps

N = 10          # 0.2 s smoothing of every signal used in fits
FRONT, REAR = (0, 1), (2, 3)


def grade_of(l, n=25):
    if l.get("grade") is not None:
        return l["grade"]
    if l["posy"] is None:
        return np.zeros(len(l["v"]))
    return np.clip(np.gradient(sm(l["posy"], n), 1 / HZ) / np.maximum(sm(l["v"], n), 5), -0.3, 0.3)


def lap_state(l, src):
    d = dict(v=sm(l["v"], N), gx=sm(l["gx"], N), gy=sm(l["gy"], N), gz=sm(l["gz"], N), grade=sm(grade_of(l), N),
             thr=l["thr"], brk=l["brk"], abs=sm(l["abs"], 5), tc=sm(l["tc"], 5), yawacc=sm(l["yawacc"], N))
    d["L"] = np.vstack([sm(l[src][i], N) for i in range(4)])
    d["slip"] = np.degrees(np.abs(np.vstack([sm(l["slip"][i], N) for i in range(4)]))) if l["slip"] is not None else None
    for k in ("fx", "fy", "gf", "load"):
        d[k] = np.vstack([sm(l[k][i], N) for i in range(4)]) if l.get(k) is not None else None
    return d


def stack(states, k):
    return np.concatenate([s[k] for s in states], axis=-1)


def X_load(v, gx, gy, gz):
    v, gx, gy, gz = np.broadcast_arrays(*(np.asarray(a, float) for a in (v, gx, gy, gz)))
    return np.stack([np.ones_like(v), v * v, gx, gy, gz], axis=-1)


class Car:
    """Fitted car model. All parameters are plain numbers (JSON-able)."""

    def __init__(self, p):
        self.p = p
        self.C = np.array(p["load_coef"])            # 4 x 5, already divided by the static load
        self.mu1 = np.array(p["mu1"]); self.k = np.array(p["k"]); self.c = np.array(p["c"])      # [front, rear]
        self.sf = p["s_front"]; self.sr = 1 - self.sf
        self.kb, self.kt = p["kappa_brake"], p["kappa_trac"]
        self.pv = np.array(p["power_v"]); self.pa = np.array(p["power_g"])
        self.cc = p["corner_drag"]; self.d0, self.d2 = p["drag0"], p["drag2"]

    def lam(self, v, gx, gy, gz):
        return np.maximum(np.einsum("ij,...j->i...", self.C, X_load(v, gx, gy, gz)), 0.02)

    def axles(self, v, gx, gy, gz):
        lam = self.lam(v, gx, gy, gz)
        out = []
        for j, idx in enumerate((FRONT, REAR)):
            a = 0.0
            for i in idx:
                a = a + 0.5 * self.mu1[j] * np.maximum(1 - self.k[j] * (lam[i] - 1) - self.c[j] * (v - 40) / 40, 0.3) * lam[i]
            out.append(a)
        return out[0], out[1]

    def drag(self, v):
        return self.d0 + self.d2 * v * v

    def power(self, v, gy):
        return np.interp(v, self.pv, self.pa) - self.cc * gy * gy


def _knee_mask(slip_ax, r, base):
    """Saturated = at or past the slip angle where r stops rising (no force data needed)."""
    edges = np.arange(0, 14.5, 0.5)
    med, mid = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = base & (slip_ax >= a) & (slip_ax < b)
        if m.sum() >= 60:
            med.append(np.median(r[m])); mid.append(a)
    if len(med) < 3:
        return base & False, np.nan
    med = np.array(med); plateau = med.max()
    knee = mid[int(np.argmax(med >= 0.97 * plateau))]
    return base & (slip_ax >= knee) & (slip_ax <= knee + 4.0), knee


def fit_car(files, q=0.75, sat="auto", verbose=True):
    laps = [l for f in files for l in real_laps(f)]
    has_force = all(l.get("fx") is not None and l.get("load") is not None for l in laps)
    src = "load" if all(l.get("load") is not None for l in laps) else "susp"
    st = [lap_state(l, src) for l in laps]
    v, gx, gy, gz, gr = (stack(st, k) for k in ("v", "gx", "gy", "gz", "grade"))
    thr, brk, abs_, tc, yawacc = (stack(st, k) for k in ("thr", "brk", "abs", "tc", "yawacc"))
    L = stack(st, "L"); slip = stack(st, "slip")
    say = print if verbose else (lambda *a, **k: None)
    p = dict(model=files[0]["model"], cls=files[0]["cls"], load_source=src, n_laps=len(laps), q=q)

    # 1. loads
    X = X_load(v, gx, gy, gz)
    C = np.array([np.linalg.lstsq(X, L[i], rcond=None)[0] for i in range(4)])
    c0 = C[:, 0].copy()
    p["static"] = c0.tolist(); p["load_coef"] = (C / c0[:, None]).tolist()
    lam = L / c0[:, None]
    wf_static = c0[:2].sum() / c0.sum()

    # 2. drag and axle share: measured with forces, else fitted with braking below
    if has_force:
        Fx, Fy = stack(st, "fx"), stack(st, "fy")
        ok = v > 15
        mass = abs(np.polyfit(gy[ok], Fy.sum(0)[ok], 1)[0]) / G
        sgn = np.sign(np.polyfit(gx[ok], Fx.sum(0)[ok], 1)[0])
        D = (sgn * Fx.sum(0) - mass * G * (gx + gr)) / (mass * G)
        d = np.linalg.lstsq(np.c_[np.ones(ok.sum()), v[ok] ** 2], D[ok], rcond=None)[0]
        p["drag0"], p["drag2"] = float(max(d[0], 0)), float(max(d[1], 0))
        qs = (np.abs(gy) > 1.0) & (np.abs(gx + gr) < 0.2) & (np.abs(yawacc) < 0.15)
        p["s_front"] = float(np.median((Fy[:2].sum(0) / Fy.sum(0))[qs])); p["mass"] = float(mass)
    else:
        p["drag0"], p["drag2"] = 0.0, None
        p["s_front"] = float(wf_static)
    drag_now = p["drag0"] + (p["drag2"] or 0.0) * v * v
    gxt = gx + gr + drag_now

    # 3. lateral grip per axle
    gymax = np.percentile(np.abs(gy), 99)
    base = (np.abs(gxt) < 0.2) & (np.abs(gy) > 0.45 * gymax) & (v > 15) & (np.abs(yawacc) < 0.4)
    gf = stack(st, "gf") if has_force and st[0]["gf"] is not None else None
    mu1, kk, cc_, info = [], [], [], []
    for j, idx in enumerate((FRONT, REAR)):
        idx = list(idx)
        r = np.abs(gy) / lam[idx].mean(0)
        m_knee, knee = _knee_mask(slip[idx].mean(0), r, base)
        m_gf = base & (gf[idx].mean(0) < 0.3) if gf is not None else None
        m = m_gf if (sat == "gf" or (sat == "auto" and gf is not None)) else m_knee
        P = 0.5 * lam[idx].sum(0); Q = 0.5 * ((lam[idx] - 1) * lam[idx]).sum(0); R = P * (v - 40) / 40
        y = np.abs(gy)[m]
        # |gy| = mu1*P - mu1*k*Q - mu1*c*R, with k in [0, 0.15] and c in [0, 0.45]: try the pinned corners too
        best = None
        for k_fix in (None, 0.0, 0.15):
            for c_fix in (None, 0.0, 0.45):
                cols = [P[m]] + ([] if k_fix is not None else [-Q[m]]) + ([] if c_fix is not None else [-R[m]])
                # with a pinned k or c the level multiplies (P - k*Q - c*R)
                if k_fix is not None or c_fix is not None:
                    cols[0] = P[m] - (k_fix or 0.0) * Q[m] * (k_fix is not None) - (c_fix or 0.0) * R[m] * (c_fix is not None)
                sol = np.linalg.lstsq(np.c_[tuple(cols)], y, rcond=None)[0]
                m1 = float(sol[0]); it = iter(sol[1:])
                k = k_fix if k_fix is not None else float(next(it) / m1)
                c = c_fix if c_fix is not None else float(next(it) / m1)
                if m1 <= 0 or not (0.0 <= k <= 0.15) or not (0.0 <= c <= 0.45):
                    continue
                err = float(np.std(y - m1 * (P[m] - k * Q[m] - c * R[m])))
                if best is None or err < best[0] - 1e-9:
                    best = (err, m1, k, c)
        _, m1, k, c = best
        ratio = y / (m1 * (P[m] - k * Q[m] - c * R[m]))
        scale = float(np.quantile(ratio, q))
        mu1.append(m1 * scale); kk.append(k); cc_.append(c)
        info.append("%s: n=%d knee %.1f deg, mu1 %.3f (LS) x %.3f (q%.0f) = %.3f, k %.3f, c %.3f, ratio p50/p75/p90/p97 %s%s" % (
            ("front", "rear")[j], m.sum(), knee, m1, scale, q * 100, m1 * scale, k, c, np.round(np.quantile(ratio, [.5, .75, .9, .97]), 3),
            "" if gf is None else " | knee-set n=%d, its median r/pred %.3f" % (m_knee.sum(), np.median(np.abs(gy)[m_knee] / (m1 * (P[m_knee] - k * Q[m_knee] - c * R[m_knee]))) if m_knee.sum() else np.nan)))
    p["mu1"], p["k"], p["c"] = mu1, kk, cc_
    say("[fit] %s (%s), %d laps, loads from %s, static front share %.3f, lateral front share %.3f" % (p["model"], p["cls"], len(laps), src, wf_static, p["s_front"]))
    for s in info: say("      " + s)

    def axle_A(j):
        idx = list((FRONT, REAR)[j])
        return sum(0.5 * mu1[j] * np.maximum(1 - kk[j] * (lam[i] - 1) - cc_[j] * (v - 40) / 40, 0.3) * lam[i] for i in idx)
    Af, Ar = axle_A(0), axle_A(1)
    sf, sr = p["s_front"], 1 - p["s_front"]

    # 4. braking (ideal bias: both axles on their circle)
    S = sf * np.sqrt(np.maximum(Af ** 2 - gy ** 2, 0)) + sr * np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 0))
    straight = np.abs(gy) < 0.3
    if has_force and gf is not None:
        lim = straight & (brk > 40) & (gf[:2].mean(0) < 0.3); how = "front grip fraction < 0.3"
    elif (abs_ > 0.5).sum() > 150:
        lim = straight & (abs_ > 0.5) & (brk > 40); how = "ABS active"
    else:
        lim = straight & (brk > 90); how = "brake > 90 %"
    if lim.sum() < 100:
        lim = straight & (brk > 85); how += " (too few: brake > 85 %)"
    if p["drag2"] is None:
        # |gx + grade| = kb * S + d2 * v^2: joint, d2 bounded by coasting
        coast = (thr < 2) & (brk < 1) & (np.abs(gy) < 0.3) & (v > 40)
        d2_max = float(np.median((-(gx + gr))[coast] / v[coast] ** 2)) if coast.sum() > 50 else 6e-5
        best = None
        for d2 in np.linspace(0, max(d2_max, 0), 13):
            kb_ = np.quantile((-(gx + gr)[lim] - d2 * v[lim] ** 2) / S[lim], 0.5)
            err = np.std(-(gx + gr)[lim] - d2 * v[lim] ** 2 - kb_ * S[lim])
            if best is None or err < best[0]: best = (err, d2)
        p["drag2"] = float(best[1])
        say("      drag (no forces): d2 %.2e per (m/s)^2 = %.2f g at 250 km/h (coasting bound %.2e)" % (p["drag2"], p["drag2"] * (250 / 3.6) ** 2, d2_max))
        gxt = gx + gr + p["drag0"] + p["drag2"] * v * v
    rb = -gxt[lim] / S[lim]
    p["kappa_brake"] = float(np.quantile(rb, q))
    say("      braking: %s, n=%d (%.0f s), kappa_b %.3f (p50/p75/p90/p97 %s)" % (how, lim.sum(), lim.sum() / HZ, p["kappa_brake"], np.round(np.quantile(rb, [.5, .75, .9, .97]), 3)))

    # 5. traction (rear axle circle)
    room = np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 1e-4))
    trl = (tc > 0.5) & (thr > 30) & (v < 45) & (gxt > 0.2) & (np.abs(gy) < 0.8 * Ar)
    how = "TC active"
    if trl.sum() < 100:
        trl = (thr > 95) & (v < 40) & (gxt > 0.2) & (np.abs(gy) < 0.8 * Ar); how = "full throttle below 144 km/h"
    rt = gxt[trl] / (sr * room[trl])
    p["kappa_trac"] = float(np.quantile(rt, q)) if trl.sum() > 30 else 1.0
    say("      traction: %s, n=%d, kappa_t %.3f (p50/p75/p90/p97 %s)" % (how, trl.sum(), p["kappa_trac"], np.round(np.quantile(rt, [.5, .75, .9, .97]), 3) if trl.sum() > 30 else "n/a"))

    # 6. engine: flat-road full-throttle acceleration by speed, and the cost of cornering
    ft = (thr > 98) & (tc < 0.05)
    es = np.arange(max(8.0, np.floor(v[ft].min())), v[ft].max() + 2.5, 2.5)
    pv, pa = [], []
    for a, b in zip(es[:-1], es[1:]):
        m = ft & (np.abs(gy) < 0.25) & (v >= a) & (v < b)
        if m.sum() >= 25:
            pv.append((a + b) / 2); pa.append(float(np.quantile((gx + gr)[m], q)))
    pv, pa = np.array(pv), np.array(pa)
    for _ in range(2):                              # light smoothing, keeps the ends
        pa[1:-1] = 0.25 * pa[:-2] + 0.5 * pa[1:-1] + 0.25 * pa[2:]
    # extend to higher speed with the last slope, to lower speed flat
    slope = (pa[-1] - pa[-4]) / (pv[-1] - pv[-4]) if len(pv) > 4 else 0.0
    pv = np.concatenate([[0.0], pv, [pv[-1] + 40]]); pa = np.concatenate([[pa[0]], pa, [pa[-1] + 40 * min(slope, 0.0)]])
    p["power_v"], p["power_g"] = pv.tolist(), pa.tolist()
    mc = ft & (np.abs(gy) > 0.8) & (v > pv[1])
    cc = float(np.median(((np.interp(v, pv, pa) - (gx + gr)) / gy ** 2)[mc])) if mc.sum() > 100 else 0.0
    p["corner_drag"] = float(np.clip(cc, 0.0, 0.08))
    say("      engine: %d speed bins %.0f-%.0f km/h, accel %.2f g at %.0f, %.2f g at %.0f km/h; cornering drag %.4f g per g^2 (n=%d)" % (
        len(pv) - 2, pv[1] * 3.6, pv[-2] * 3.6, pa[1], pv[1] * 3.6, pa[-2], pv[-2] * 3.6, p["corner_drag"], mc.sum()))
    return Car(p)


# ── the track line of one lap ────────────────────────────────────────────────
DS = 2.0


def line_of(l, ds=DS):
    v = sm(l["v"], N); s = l["s"]
    ks = sm(G * l["gy"] / np.maximum(l["v"], 8) ** 2, 15)        # signed path curvature (as gy)
    kv = sm(G * l["gz"] / np.maximum(l["v"], 8) ** 2, 25)        # vertical curvature
    gr = sm(grade_of(l), 25)
    sg = np.arange(0, s[-1], ds)
    f = lambda a: np.interp(sg, s, a)
    return dict(s=sg, ks=f(ks), kv=f(kv), grade=f(gr), v=f(v), thr=f(l["thr"]), brk=f(l["brk"]), ds=ds, time=l["time"])


VG = np.arange(6.0, 115.01, 0.5)
GXT_UP = np.linspace(0.0, 2.0, 81)
GXT_DN = np.linspace(0.0, -5.5, 221)


def solve(car, ln, mode="balanced"):
    """Quasi-steady lap on a fixed line. mode: 'balanced' (each axle carries its
    share) or 'ceiling' (cornering uses front and rear capacity together)."""
    ks, kv, gr, ds = ln["ks"], ln["kv"], ln["grade"], ln["ds"]
    n = len(ks)

    def lat_ok(v, gxt, gy, gz):
        """(feasible, Af, Ar) for tyre long demand gxt at this state."""
        Af, Ar = car.axles(v, gxt - car.drag(v), gy, gz)
        ay = np.abs(gy)
        if mode == "ceiling":
            tot = car.sf * Af + car.sr * Ar
            return ay <= tot, np.maximum(Af, tot), np.maximum(Ar, tot)
        return (ay <= Af) & (ay <= Ar), Af, Ar

    def feasible(v, gxt, gy, gz):
        """Can the tyres supply gxt (drive +, brake -) while cornering at gy?"""
        ok, Af, Ar = lat_ok(v, gxt, gy, gz)
        rf = np.sqrt(np.maximum(Af ** 2 - gy * gy, 0)); rr = np.sqrt(np.maximum(Ar ** 2 - gy * gy, 0))
        return ok & np.where(gxt >= 0, gxt <= car.kt * car.sr * rr + 1e-9, -gxt <= car.kb * (car.sf * rf + car.sr * rr) + 1e-9)

    # 1. corner limit: the highest v at which SOME tyre long demand is feasible.
    # PREVIOUSLY: only "holding speed" was tried, so a front-limited car, which
    # corners harder while slowing (fronts loaded), was up to 30 km/h too slow
    # through a sequence taken on a falling speed (COTA esses).
    V = VG[None, :]; gy = ks[:, None] * V * V / G; gz = np.clip(kv[:, None] * V * V / G, -0.9, 1.5)
    ok = np.zeros(gy.shape, bool)
    for g_try in (0.3, 0.1, 0.0, -0.15, -0.3, -0.5, -0.75, -1.0, -1.5, -2.0):
        ok |= feasible(V, g_try, gy, gz)
    first_bad = np.argmax(~ok, axis=1)
    vc = np.where(ok.all(1), VG[-1], VG[np.maximum(first_bad - 1, 0)])
    vs = vc.copy()
    GX_ALL = np.concatenate([GXT_DN[::-1], GXT_UP[1:]])

    def accel(i, v):
        gy = ks[i] * v * v / G; gz = min(max(kv[i] * v * v / G, -0.9), 1.5)
        ok = feasible(v, GX_ALL, gy, gz)
        t = GX_ALL[ok].max() if ok.any() else 0.0
        return min(car.power(v, gy) - gr[i], t - car.drag(v) - gr[i])

    def brake(i, v):
        gy = ks[i] * v * v / G; gz = min(max(kv[i] * v * v / G, -0.9), 1.5)
        ok = feasible(v, GXT_DN, gy, gz)
        t = GXT_DN[ok].min() if ok.any() else 0.0
        return t - car.drag(v) - gr[i]            # kinematic gx (negative)

    for _ in range(2):
        for i in list(range(n)) + [0]:
            j = (i + 1) % n
            a = accel(i, vs[i])
            vs[j] = min(vs[j], np.sqrt(max(vs[i] ** 2 + 2 * G * a * ds, 4.0)))
        for i in list(range(n - 1, -1, -1)) + [n - 1]:
            j = (i - 1) % n
            a = brake(i, vs[i])                    # decel available at i, applied over the step before it
            vs[j] = min(vs[j], np.sqrt(max(vs[i] ** 2 - 2 * G * a * ds, 4.0)))
    return dict(v=vs, vc=vc, time=float(np.sum(ds / vs)))


def compare(ln, sol, tol=0.5):
    """Driven vs ideal on the same grid."""
    va, vi, ds = ln["v"], sol["v"], ln["ds"]
    t_act, t_id = float(np.sum(ds / va)), sol["time"]
    slow = vi < va - tol
    reg = np.where(ln["brk"] > 10, 0, np.where(ln["thr"] > 90, 1, 2))
    lost = ds / vi - ds / va
    return dict(t_act=t_act, t_ideal=t_id, gain=t_act - t_id, slow_share=float(slow.mean()),
                slow_by=[float((slow & (reg == k)).mean()) for k in range(3)],
                slow_time=float(lost[slow].sum()), worst=float((va - vi).max() * 3.6))
