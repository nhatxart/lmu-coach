"""Ideal lap, the solver (design 2026-11-10-ideal-lap-design.md §5.1-5.5):
quasi-steady lap of a fitted ideal_car.CarModel on the fixed driven line of a
reference lap, the envelope with the driven lap, and grip used per axle.
Ported from the spike's il_core.line_of / solve. No I/O.

Units as ideal_car: accelerations in g, v in m/s, s in m on the line's own
distance (integrated speed), ks / kv path / vertical curvature in 1/m.
"""
from __future__ import annotations

import numpy as np

from .ideal_state import G, smooth, SMOOTH_N, raw_grade

DS = 2.0
VG = np.arange(6.0, 115.01, 0.5)           # corner-limit speed grid (m/s)
GXT_UP = np.linspace(0.0, 2.0, 81)          # tyre longitudinal demand tried, drive
GXT_DN = np.linspace(0.0, -5.5, 221)        # and braking (0.025 g steps)
GX_ALL = np.concatenate([GXT_DN[::-1], GXT_UP[1:]])
# Tyre demands tried for the corner limit. A front-limited car corners harder
# while slowing (fronts loaded), so holding speed alone is not enough.
G_TRY = (0.3, 0.1, 0.0, -0.15, -0.3, -0.5, -0.75, -1.0, -1.5, -2.0)
GZ_MIN, GZ_MAX = -0.9, 1.5
VC_BISECT = 12                              # 0.5 m/s / 2^12: vc exact to ~1e-4 m/s
BINDING = ("front", "rear", "brake", "traction", "power")


def line_of(raw: dict, ds: float = DS) -> dict:
    """The reference lap's line on a ds grid of its own distance (il_core.line_of):
    path curvature ks = g gy / v^2 (0.3 s), vertical curvature kv = g gz / v^2
    (0.5 s), grade (0.5 s), the driven speed v and pedals. `raw`: ideal_state.lap_raw."""
    v = smooth(raw["v"], SMOOTH_N)
    s = raw["s"]
    ks = smooth(G * raw["gy"] / np.maximum(raw["v"], 8) ** 2, 15)
    kv = smooth(G * raw["gz"] / np.maximum(raw["v"], 8) ** 2, 25)
    grade = raw.get("grade")
    if grade is None:
        grade, _ = raw_grade(raw)
    gr = smooth(grade, 25)
    sg = np.arange(0, s[-1], ds)
    f = lambda a: np.interp(sg, s, a)
    out = dict(s=sg, ks=f(ks), kv=f(kv), grade=f(gr), v=f(v), thr=f(raw["thr"]), brk=f(raw["brk"]),
               ds=ds, time=float(raw["time"]))
    if raw.get("lapdist") is not None:
        out["lap_dist"] = f(np.asarray(raw["lapdist"], float))
    return out


def _lat_ok(car, mode, v, gxt, gy, gz):
    """(feasible lateral, Af, Ar) for tyre demand gxt at this state; lam at the
    kinematic gx = gxt - drag (il_core: grade is not taken out here)."""
    Af, Ar = car.axles(v, gxt - car.drag(v), gy, gz)
    ay = np.abs(gy)
    if mode == "ceiling":
        tot = car.sf * Af + car.sr * Ar
        return ay <= tot, np.maximum(Af, tot), np.maximum(Ar, tot)
    return (ay <= Af) & (ay <= Ar), Af, Ar


def _feasible(car, mode, v, gxt, gy, gz):
    """Can the tyres supply gxt (drive +, brake -) while cornering at gy?"""
    ok, Af, Ar = _lat_ok(car, mode, v, gxt, gy, gz)
    rf = np.sqrt(np.maximum(Af ** 2 - gy * gy, 0))
    rr = np.sqrt(np.maximum(Ar ** 2 - gy * gy, 0))
    return ok & np.where(gxt >= 0, gxt <= car.kt * car.sr * rr + 1e-9,
                         -gxt <= car.kb * (car.sf * rf + car.sr * rr) + 1e-9)


def _any_feasible(car, mode, v, ks, kv):
    gy = ks * v * v / G
    gz = np.clip(kv * v * v / G, GZ_MIN, GZ_MAX)
    ok = np.zeros(np.broadcast(v, ks).shape, bool)
    for g_try in G_TRY:
        ok |= _feasible(car, mode, v, g_try, gy, gz)
    return ok


def corner_limit(car, ln: dict, mode: str = "balanced") -> tuple[np.ndarray, np.ndarray]:
    """(vc, axle) per grid point: the highest v at which SOME tyre demand is
    feasible, and which axle limits there (0 front, 1 rear, -1 none).
    PREVIOUSLY (spike): vc was the last 0.5 m/s grid speed that was feasible,
    up to 0.5 m/s low (1.3 % on a 38 m/s corner). Fix: bisect between the last
    feasible and the first infeasible grid speed."""
    ks, kv = ln["ks"], ln["kv"]
    ok = _any_feasible(car, mode, VG[None, :], ks[:, None], kv[:, None])
    allok = ok.all(1)
    first_bad = np.argmax(~ok, axis=1)
    vc = np.where(allok, VG[-1], VG[np.maximum(first_bad - 1, 0)])
    lo, hi = vc.copy(), np.where(allok, VG[-1], VG[first_bad])
    act = ~allok & (first_bad > 0)
    for _ in range(VC_BISECT):
        mid = 0.5 * (lo + hi)
        good = _any_feasible(car, mode, mid, ks, kv)
        lo = np.where(act & good, mid, lo)
        hi = np.where(act & ~good, mid, hi)
    vc = np.where(act, lo, vc)
    # Which axle runs out at vc: among the demands that are feasible there, the
    # one with the most lateral room left; the axle with less room at it binds.
    # (In ceiling mode this is informative only: both axles act together.)
    gy = ks * vc * vc / G
    gz = np.clip(kv * vc * vc / G, GZ_MIN, GZ_MAX)
    best, axle = np.full(len(ks), -np.inf), np.zeros(len(ks), int)
    for g_try in G_TRY:
        Af, Ar = car.axles(vc, g_try - car.drag(vc), gy, gz)
        mf, mr = Af - np.abs(gy), Ar - np.abs(gy)
        room = np.where(_feasible(car, mode, vc, g_try, gy, gz), np.minimum(mf, mr), -np.inf)
        take = room > best
        best = np.where(take, room, best)
        axle = np.where(take, np.where(mf <= mr, 0, 1), axle)
    axle = np.where(allok, -1, axle)
    return vc, axle


def solve(car, ln: dict, mode: str = "balanced", refine: bool = True) -> dict:
    """Quasi-steady lap on the fixed line (design §5.2 / §5.3).
    mode 'balanced': each axle carries its share (|gy| <= Af and <= Ar).
    mode 'ceiling': |gy| <= s_f Af + s_r Ar (both axles at the limit at once).
    Returns {v, vc, binding, gxt, time}: v and vc in m/s per grid point, the
    binding code per point (front | rear | brake | traction | power), the tyre
    longitudinal demand gxt of the solved profile (g) and the lap time.
    `refine`: TEST ONLY (False = the spike's 0.5 m/s corner limit, for the
    parity check); the app always refines."""
    ks, kv, gr, ds = ln["ks"], ln["kv"], ln["grade"], ln["ds"]
    n = len(ks)
    if refine:
        vc, axle = corner_limit(car, ln, mode)
    else:
        ok = _any_feasible(car, mode, VG[None, :], ks[:, None], kv[:, None])
        first_bad = np.argmax(~ok, axis=1)
        vc = np.where(ok.all(1), VG[-1], VG[np.maximum(first_bad - 1, 0)])
        axle = np.where(ok.all(1), -1, 0)
    vs = vc.copy()
    code = np.where(axle == 0, 0, np.where(axle == 1, 1, 4)).astype(int)

    def accel(i, v):
        gy = ks[i] * v * v / G
        gz = min(max(kv[i] * v * v / G, GZ_MIN), GZ_MAX)
        ok = _feasible(car, mode, v, GX_ALL, gy, gz)
        t = GX_ALL[ok].max() if ok.any() else 0.0
        p, g = car.power(v, gy) - gr[i], t - car.drag(v) - gr[i]
        return (p, 4) if p <= g else (g, 3)

    def brake(i, v):
        gy = ks[i] * v * v / G
        gz = min(max(kv[i] * v * v / G, GZ_MIN), GZ_MAX)
        ok = _feasible(car, mode, v, GXT_DN, gy, gz)
        t = GXT_DN[ok].min() if ok.any() else 0.0
        return t - car.drag(v) - gr[i]            # kinematic gx (negative)

    # Values only ever decrease, so the last bound that lowered a point is the
    # one that binds there (the code follows it).
    for _ in range(2):
        for i in list(range(n)) + [0]:
            j = (i + 1) % n
            a, why = accel(i, vs[i])
            vn = np.sqrt(max(vs[i] ** 2 + 2 * G * a * ds, 4.0))
            if vn < vs[j]:
                vs[j], code[j] = vn, why
        for i in list(range(n - 1, -1, -1)) + [n - 1]:
            j = (i - 1) % n
            a = brake(i, vs[i])                    # decel available at i, applied over the step before it
            vn = np.sqrt(max(vs[i] ** 2 - 2 * G * a * ds, 4.0))
            if vn < vs[j]:
                vs[j], code[j] = vn, 2
    # tyre demand of the solved profile: kinematic gx over each step + grade + drag
    gx = (np.roll(vs, -1) ** 2 - vs ** 2) / (2 * G * ds)
    gxt = gx + gr + car.drag(vs)
    return dict(v=vs, vc=vc, binding=np.array(BINDING, dtype=object)[code], gxt=gxt,
                time=float(np.sum(ds / vs)))


def envelope(v_model: np.ndarray, v_ref: np.ndarray, ds: float) -> dict:
    """Design §5.4: v_I = max(v_M, v_ref), T_I = sum ds / v_I. Where the two
    traces cross they are in the same state (s, v), so every piece of the
    envelope is either allowed by the model or was actually driven."""
    v_i = np.maximum(v_model, v_ref)
    return dict(v=v_i, time=float(np.sum(ds / v_i)), from_ref=v_ref > v_model)


def levels(car, ln: dict, beta: float, mode: str = "balanced") -> dict:
    """Design §5.4: solve at grip level 1-beta, 1, 1+beta and take the envelope
    of each with the driven speed. gain_* = T_ref_grid - T_I (positive = faster).
    Also the plain model time at level 1 (`model_only`) and the share of the lap
    where the model is more than 2 km/h under the driven speed."""
    ds, v_ref = ln["ds"], ln["v"]
    t_ref = float(np.sum(ds / v_ref))
    out = {"t_ref_grid": t_ref}
    for name, lev in (("min", 1 - beta), ("exp", 1.0), ("max", 1 + beta)):
        sol = solve(car.scaled(lev), ln, mode)
        env = envelope(sol["v"], v_ref, ds)
        out[name] = dict(level=lev, solve=sol, env=env, gain=t_ref - env["time"],
                         model_gain=t_ref - sol["time"])
    out["gain_min"], out["gain"], out["gain_max"] = out["min"]["gain"], out["exp"]["gain"], out["max"]["gain"]
    sol1 = out["exp"]["solve"]
    out["model_only_time"] = sol1["time"]
    out["under_driven_share"] = float((sol1["v"] < v_ref - 2 / 3.6).mean())
    return out


def usage(car, v, gxt, gy, gz) -> tuple[np.ndarray, np.ndarray]:
    """Design §5.5: grip used per axle, u_j = sqrt(gy^2 + (gxt_j / kappa)^2) / A_j.
    Braking: both axles see gxt, kappa = kb (by construction of §4.4).
    Driving: the rear sees gxt / s_r with kappa = kt; the front sees none."""
    v, gxt, gy, gz = (np.asarray(a, float) for a in (v, gxt, gy, gz))
    Af, Ar = car.axles(v, gxt - car.drag(v), gy, gz)
    brk = gxt < 0
    gf_ = np.where(brk, gxt / car.kb, 0.0)
    gr_ = np.where(brk, gxt / car.kb, gxt / (car.sr * car.kt))
    uf = np.sqrt(gy * gy + gf_ * gf_) / np.maximum(Af, 1e-6)
    ur = np.sqrt(gy * gy + gr_ * gr_) / np.maximum(Ar, 1e-6)
    return uf, ur


def usage_on_line(car, ln: dict, v: np.ndarray, gxt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Usage of a speed profile on the line (the ideal's own gxt)."""
    gy = ln["ks"] * v * v / G
    gz = np.clip(ln["kv"] * v * v / G, GZ_MIN, GZ_MAX)
    return usage(car, v, gxt, gy, gz)
