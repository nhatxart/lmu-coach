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
GX_STEP = 0.025                             # the demand grids' step (g)
DEMAND_BISECT = 4                           # 0.025 g / 2^4 = 0.0016 g
BINDING = ("front", "rear", "brake", "traction", "power", "measured")
SAT_MIN_RUN = 10                            # samples (0.2 s at 50 Hz): shorter saturated runs are dropped
CONFLICT_MARGIN = 1.0                       # m/s, §14 R7


def _drop_short_runs(m: np.ndarray, n_min: int = SAT_MIN_RUN) -> np.ndarray:
    m = np.asarray(m, bool).copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < n_min:
            m[a:b] = False
    return m


def saturated_ref(state, car, axles, per_axle: bool = False):
    """Design §5.4 (round 1): the reference lap's samples where its own tyres were
    measured saturated in a lateral-dominant state:
      |gy| > 0.45 p99(|gy|) of the lap, v > 15, |gxt| < 0.5 |gy|, and an axle
      saturated: mean Grip Fraction < 0.3 when recorded, else knee_j <= mean
      |slip| <= knee_j + 4 deg (the fit's own knee). Runs under 0.2 s dropped.
    `axles`: [AxleFit front, rear]. per_axle: also the (2, n) mask per axle."""
    from .ideal_car import GF_SATURATED, KNEE_SPAN_DEG, FRONT, REAR
    gy = np.abs(state.gy)
    gxt = state.gx + state.grade + car.drag(state.v)
    base = (gy > 0.45 * np.percentile(gy, 99)) & (state.v > 15) & (np.abs(gxt) < 0.5 * gy)
    ax = np.zeros((2, len(gy)), bool)
    for j, idx in enumerate((FRONT, REAR)):
        idx = list(idx)
        if state.gf is not None:
            m = state.gf[idx].mean(0) < GF_SATURATED
        elif state.slip is not None and axles[j].knee_deg is not None:
            sl = state.slip[idx].mean(0)
            m = (sl >= axles[j].knee_deg) & (sl <= axles[j].knee_deg + KNEE_SPAN_DEG)
        else:
            continue
        ax[j] = _drop_short_runs(base & m)
    sat = _drop_short_runs(ax.any(0))
    ax &= sat[None, :]
    return (sat, ax) if per_axle else sat


def _to_grid(s: np.ndarray, m: np.ndarray, sg: np.ndarray, ds: float) -> np.ndarray:
    """A grid point is set when more than half of its samples (the samples
    nearest to it) are; a point without samples takes its nearest sample."""
    k = np.clip(np.rint(np.asarray(s) / ds).astype(int), 0, len(sg) - 1)
    n = np.bincount(k, minlength=len(sg))
    on = np.bincount(k, weights=np.asarray(m, float), minlength=len(sg))
    out = on > 0.5 * n
    empty = n == 0
    if empty.any():
        out[empty] = np.interp(sg[empty], s, np.asarray(m, float)) > 0.5
    return out


def apply_clamp(ln: dict, sat_grid: np.ndarray, axle_grid: np.ndarray | None = None) -> dict:
    """Design §5.4 saturation clamp on a line: at the grid points where the
    reference lap was measured saturated, the corner limit is the driven speed
    (`v_clamp`; NaN elsewhere). `sat_axles` (2 x n) says which axle was."""
    sat = np.asarray(sat_grid, bool)
    ln["sat_ref"] = sat
    ln["v_clamp"] = np.where(sat, ln["v"], np.nan)
    ln["sat_axles"] = (np.asarray(axle_grid, bool) if axle_grid is not None
                       else np.zeros((2, len(sat)), bool))
    return ln


def line_of(raw: dict, ds: float = DS, sat_ref: np.ndarray | None = None,
            sat_axles: np.ndarray | None = None) -> dict:
    """The reference lap's line on a ds grid of its own distance (il_core.line_of):
    path curvature ks = g gy / v^2 (0.3 s), vertical curvature kv = g gz / v^2
    (0.5 s), grade (0.5 s), the driven speed v and pedals. `raw`: ideal_state.lap_raw.
    `sat_ref` / `sat_axles`: saturated_ref() of the same lap's state (per 50 Hz
    sample); put on the grid and applied as the saturation clamp."""
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
    if sat_ref is not None:
        ax = None
        if sat_axles is not None:
            ax = np.vstack([_to_grid(s, sat_axles[j], sg, ds) for j in (0, 1)])
        sat = _to_grid(s, sat_ref, sg, ds)
        apply_clamp(out, sat, None if ax is None else ax & sat[None, :])
    return out


def _lat_ok(car, mode, v, gxt, gy, gz, sc=1.0):
    """(feasible lateral, Af, Ar) for tyre demand gxt at this state; lam at the
    kinematic gx = gxt - drag (il_core: grade is not taken out here)."""
    Af, Ar = car.axles(v, gxt - car.drag(v), gy, gz)
    Af, Ar = Af * sc, Ar * sc
    ay = np.abs(gy)
    if mode == "ceiling":
        tot = car.sf * Af + car.sr * Ar
        return ay <= tot, np.maximum(Af, tot), np.maximum(Ar, tot)
    return (ay <= Af) & (ay <= Ar), Af, Ar


def _feasible(car, mode, v, gxt, gy, gz, sc=1.0):
    """Can the tyres supply gxt (drive +, brake -) while cornering at gy?
    `sc`: capacity scale per point (the line's `cap_scale`, synthetic tests only)."""
    ok, Af, Ar = _lat_ok(car, mode, v, gxt, gy, gz, sc)
    rf = np.sqrt(np.maximum(Af ** 2 - gy * gy, 0))
    rr = np.sqrt(np.maximum(Ar ** 2 - gy * gy, 0))
    return ok & np.where(gxt >= 0, gxt <= car.kt * car.sr * rr + 1e-9,
                         -gxt <= car.kb * (car.sf * rf + car.sr * rr) + 1e-9)


def _any_feasible(car, mode, v, ks, kv, sc=1.0):
    gy = ks * v * v / G
    gz = np.clip(kv * v * v / G, GZ_MIN, GZ_MAX)
    ok = np.zeros(np.broadcast(v, ks).shape, bool)
    for g_try in G_TRY:
        ok |= _feasible(car, mode, v, g_try, gy, gz, sc)
    return ok


def _cap_scale(ln: dict) -> np.ndarray:
    """Per-point multiplier on both axles' capacity. Only the synthetic
    perfect-driver test sets it (design §8 T6: the true limit moves with place
    and lap); a real line has none (1.0)."""
    sc = ln.get("cap_scale")
    return np.ones(len(ln["ks"])) if sc is None else np.asarray(sc, float)


VC_BISECT_DIRECT = 21                       # 109 m/s / 2^21: 5e-5 m/s


def corner_limit(car, ln: dict, mode: str = "balanced", scan: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(vc, axle) per grid point: the highest v at which SOME tyre demand is
    feasible, and which axle limits there (0 front, 1 rear, -1 none).
    PREVIOUSLY (spike): vc was the last 0.5 m/s grid speed that was feasible,
    up to 0.5 m/s low (1.3 % on a 38 m/s corner). Fix: bisect between the last
    feasible and the first infeasible grid speed (`scan=True`, the reference).
    Review I1: the grid scan evaluated all 219 grid speeds x 10 demands per point
    (0.7 s per COTA solve). The lateral demand grows with v^2 faster than the
    capacity does (load ~ 1 + a v^2, mu falling with v), so feasibility only gets
    harder with speed and vc is bisected directly between 6 and 115 m/s."""
    ks, kv = ln["ks"], ln["kv"]
    sc = _cap_scale(ln)
    if scan:
        ok = _any_feasible(car, mode, VG[None, :], ks[:, None], kv[:, None], sc[:, None])
        allok = ok.all(1)
        first_bad = np.argmax(~ok, axis=1)
        vc = np.where(allok, VG[-1], VG[np.maximum(first_bad - 1, 0)])
        lo, hi = vc.copy(), np.where(allok, VG[-1], VG[first_bad])
        act = ~allok & (first_bad > 0)
        iters = VC_BISECT
    else:
        n = len(ks)
        allok = _any_feasible(car, mode, np.full(n, VG[-1]), ks, kv, sc)
        none = ~_any_feasible(car, mode, np.full(n, VG[0]), ks, kv, sc)
        vc = np.where(allok, VG[-1], VG[0])
        lo, hi = np.full(n, VG[0]), np.full(n, VG[-1])
        act = ~allok & ~none
        iters = VC_BISECT_DIRECT
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        good = _any_feasible(car, mode, mid, ks, kv, sc)
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
        mf, mr = Af * sc - np.abs(gy), Ar * sc - np.abs(gy)
        room = np.where(_feasible(car, mode, vc, g_try, gy, gz, sc), np.minimum(mf, mr), -np.inf)
        take = room > best
        best = np.where(take, room, best)
        axle = np.where(take, np.where(mf <= mr, 0, 1), axle)
    axle = np.where(allok, -1, axle)
    return vc, axle


TABLE_K = 32                                # speed nodes per grid point, dense near the corner limit
TABLE_BISECT = 12                           # 7.5 g / 2^12 = 0.0018 g on the demand edge (the per-step search: 0.0016)
GX_MIN, GX_MAX = float(GXT_DN[-1]), float(GXT_UP[-1])


def demand_tables(car, ln: dict, mode: str, vc: np.ndarray) -> dict:
    """Review I1: per grid point i and speed node V[i, k] (from 2 m/s to the
    point's corner limit vc[i], spaced quadratically towards vc where the
    feasible demands change fastest), the strongest feasible drive demand `up`
    and braking demand `dn` (g), and whether any demand is feasible there.
    Fully vectorised: a feasible seed among G_TRY, then the edge bisected from
    it (up towards 2 g, down towards -5.5 g); a node without a feasible seed is
    `ok = False` (the passes then fall back to the per-step search).
    WHY: the per-step search called _feasible ~40,000 times per solve on tiny
    arrays (2.3 of 2.5 s); here it is ~40 calls on (n x K) arrays."""
    ks, kv = ln["ks"], ln["kv"]
    sc = _cap_scale(ln)[:, None]
    f = 1.0 - (1.0 - np.linspace(0.0, 1.0, TABLE_K)) ** 2
    lo_v = 2.0
    V = lo_v + (np.maximum(vc, lo_v + 1e-3) - lo_v)[:, None] * f[None, :]
    gy = ks[:, None] * V * V / G
    gz = np.clip(kv[:, None] * V * V / G, GZ_MIN, GZ_MAX)
    feas = np.stack([_feasible(car, mode, V, g, gy, gz, sc) for g in G_TRY])
    ok = feas.any(0)
    gt = np.asarray(G_TRY, float)[:, None, None]
    seed_up = np.where(feas, gt, -np.inf).max(0)
    seed_dn = np.where(feas, gt, np.inf).min(0)
    seed_up = np.where(ok, seed_up, 0.0)
    seed_dn = np.where(ok, seed_dn, 0.0)

    def edge(seed, bound):
        top = _feasible(car, mode, V, np.full(V.shape, bound), gy, gz, sc)
        lo, hi = seed.copy(), np.full(V.shape, bound)
        for _ in range(TABLE_BISECT):
            mid = 0.5 * (lo + hi)
            good = _feasible(car, mode, V, mid, gy, gz, sc)
            lo, hi = np.where(good, mid, lo), np.where(good, hi, mid)
        return np.where(top, bound, lo)

    up = edge(seed_up, GX_MAX)
    dn = edge(seed_dn, GX_MIN)
    # the per-step search took braking demands in [-5.5, 0] only: a feasible set
    # entirely above 0 gave no braking (0)
    dn = np.where(dn <= 0.0, dn, 0.0)
    return dict(V=V, up=up, dn=dn, ok=ok)


def _lookup(tab: dict, key: str, i: int, v: float):
    """Interpolated demand at point i and speed v, or None when v is outside
    the nodes or a bracketing node has no feasible demand."""
    Vi = tab["V"][i]
    if v > Vi[-1] + 1e-9:
        return None
    k = int(np.searchsorted(Vi, v))
    if k <= 0:
        return float(tab[key][i, 0]) if tab["ok"][i, 0] else None
    if k >= len(Vi):
        k = len(Vi) - 1
    if not (tab["ok"][i, k] and tab["ok"][i, k - 1]):
        return None
    w = (v - Vi[k - 1]) / (Vi[k] - Vi[k - 1])
    a = tab[key][i]
    return float(a[k - 1] + w * (a[k] - a[k - 1]))


def solve(car, ln: dict, mode: str = "balanced", refine: bool = True, clamp: bool = True,
          fast: bool = True) -> dict:
    """Quasi-steady lap on the fixed line (design §5.2 / §5.3).
    mode 'balanced': each axle carries its share (|gy| <= Af and <= Ar).
    mode 'ceiling': |gy| <= s_f Af + s_r Ar (both axles at the limit at once).
    Returns {v, vc, binding, gxt, time}: v and vc in m/s per grid point, the
    binding code per point (front | rear | brake | traction | power | measured),
    the tyre longitudinal demand gxt of the solved profile (g) and the lap time.
    `clamp`: in balanced mode, a line with `v_clamp` (apply_clamp) has its corner
    limit replaced by the driven speed at the clamped points (code `measured`,
    §5.4); the ceiling is never clamped. clamp=False gives the plain model.
    `refine`: TEST ONLY (False = the spike's 0.5 m/s corner limit and 0.025 g
    demand grid, for the parity check); the app always refines.
    `fast`: the demands from demand_tables (review I1); False = the per-step
    search (the reference of test_ideal_solver_fast; refine=False implies it)."""
    ks, kv, gr, ds = ln["ks"], ln["kv"], ln["grade"], ln["ds"]
    n = len(ks)
    fast = fast and refine
    if refine:
        vc, axle = corner_limit(car, ln, mode, scan=not fast)
    else:
        ok = _any_feasible(car, mode, VG[None, :], ks[:, None], kv[:, None], _cap_scale(ln)[:, None])
        first_bad = np.argmax(~ok, axis=1)
        vc = np.where(ok.all(1), VG[-1], VG[np.maximum(first_bad - 1, 0)])
        axle = np.where(ok.all(1), -1, 0)
    code = np.where(axle == 0, 0, np.where(axle == 1, 1, 4)).astype(int)
    tab = demand_tables(car, ln, mode, vc) if fast else None      # on the MODEL's corner limit
    if clamp and mode == "balanced" and ln.get("v_clamp") is not None:
        # The measurement replaces the model where there is one: at a point the
        # lap's own tyres were saturated, v_M / v_ref = sqrt(A / |gy|) is the
        # model's own residual (§13 R1). Up or down, and not scaled by the level.
        m = np.asarray(ln["sat_ref"], bool)
        vc = np.where(m, ln["v_clamp"], vc)
        code = np.where(m, 5, code)
    vs = vc.copy()
    scs = _cap_scale(ln)

    def edge(v, gy, gz, t, step, sc):
        """Bisect from the last feasible grid demand t towards t + step.
        PREVIOUSLY the solver used the grid value itself: up to 0.025 g (1-2 %)
        less than the fitted kb / kt, so a car refitted from the solver's own
        laps read kb, kt 1.5-2.5 % low (design §13 R4). Fix: 4 halvings."""
        lo, hi = t, t + step
        for _ in range(DEMAND_BISECT):
            mid = 0.5 * (lo + hi)
            if _feasible(car, mode, v, mid, gy, gz, sc):
                lo = mid
            else:
                hi = mid
        return lo

    def accel(i, v):
        gy = ks[i] * v * v / G
        t = _lookup(tab, "up", i, v) if fast else None
        if t is None:
            gz = min(max(kv[i] * v * v / G, GZ_MIN), GZ_MAX)
            ok = _feasible(car, mode, v, GX_ALL, gy, gz, scs[i])
            t = 0.0
            if ok.any():
                k = int(np.flatnonzero(ok)[-1])
                t = GX_ALL[k] if (k == len(GX_ALL) - 1 or not refine) else edge(v, gy, gz, GX_ALL[k], GX_STEP, scs[i])
        p, g = car.power(v, gy) - gr[i], t - car.drag(v) - gr[i]
        return (p, 4) if p <= g else (g, 3)

    def brake(i, v):
        t = _lookup(tab, "dn", i, v) if fast else None
        if t is None:
            gy = ks[i] * v * v / G
            gz = min(max(kv[i] * v * v / G, GZ_MIN), GZ_MAX)
            ok = _feasible(car, mode, v, GXT_DN, gy, gz, scs[i])
            t = 0.0
            if ok.any():
                k = int(np.flatnonzero(ok)[-1])
                t = GXT_DN[k] if (k == len(GXT_DN) - 1 or not refine) else edge(v, gy, gz, GXT_DN[k], -GX_STEP, scs[i])
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
    """Design §5.4: solve at grip level 1-beta, 1, 1+beta (clamped where the line
    carries the saturation clamp) and take the envelope of each with the driven
    speed. gain_* = T_ref_grid - T_I (positive = faster).
    Also, unclamped at level 1: the plain model time (`model_only`), the share of
    the lap where it is more than 2 km/h under the driven speed, and its envelope
    gain (`gain_unclamped`, V13); and `clamped_share`."""
    ds, v_ref = ln["ds"], ln["v"]
    t_ref = float(np.sum(ds / v_ref))
    out = {"t_ref_grid": t_ref}
    for name, lev in (("min", 1 - beta), ("exp", 1.0), ("max", 1 + beta)):
        sol = solve(car.scaled(lev), ln, mode)
        env = envelope(sol["v"], v_ref, ds)
        out[name] = dict(level=lev, solve=sol, env=env, gain=t_ref - env["time"],
                         model_gain=t_ref - sol["time"])
    out["gain_min"], out["gain"], out["gain_max"] = out["min"]["gain"], out["exp"]["gain"], out["max"]["gain"]
    clamped = mode == "balanced" and ln.get("v_clamp") is not None and bool(np.any(ln["sat_ref"]))
    plain = solve(car, ln, mode, clamp=False) if clamped else out["exp"]["solve"]
    out["plain"] = plain
    out["model_only_time"] = plain["time"]
    out["under_driven_share"] = float((plain["v"] < v_ref - 2 / 3.6).mean())
    out["gain_unclamped"] = t_ref - envelope(plain["v"], v_ref, ds)["time"]
    out["clamped_share"] = float(np.mean(ln["sat_ref"])) if ln.get("sat_ref") is not None else 0.0
    return out


def axle_limit(car, ln: dict, j: int, iters: int = 20) -> np.ndarray:
    """Per grid point, the highest speed at which axle j alone holds the
    corner (|gy| <= A_j for some tyre demand of G_TRY), bisected in [6, 115]."""
    ks, kv = ln["ks"], ln["kv"]

    def ok(v):
        gy = ks * v * v / G
        gz = np.clip(kv * v * v / G, GZ_MIN, GZ_MAX)
        good = np.zeros(len(ks), bool)
        for g_try in G_TRY:
            A = car.axles(v, g_try - car.drag(v), gy, gz)[j]
            good |= np.abs(gy) <= A
        return good

    lo, hi = np.full(len(ks), VG[0]), np.full(len(ks), VG[-1])
    top = ok(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = ok(mid)
        lo, hi = np.where(g, mid, lo), np.where(g, hi, mid)
    return np.where(top, VG[-1], lo)


def clamp_conflict(car, ln: dict) -> dict:
    """§14 R7 diagnostic: at the clamped points where axle j was NOT measured
    saturated, the model's own limit of axle j should be >= v_ref (the lap
    proves it); a point where it is below v_ref - 1 m/s is a conflict, the
    signature of a pooled level that does not fit this file's setup.
    {front|rear: {n, conflicts, share (None when n = 0)}}."""
    out = {}
    sat = np.asarray(ln.get("sat_ref") if ln.get("sat_ref") is not None else np.zeros(len(ln["ks"]), bool))
    axes = ln.get("sat_axles") if ln.get("sat_axles") is not None else np.zeros((2, len(sat)), bool)
    for j, name in enumerate(("front", "rear")):
        m = sat & ~np.asarray(axes[j], bool)
        if not m.any():
            out[name] = {"n": 0, "conflicts": 0, "share": None}
            continue
        sub = {k: ln[k][m] for k in ("ks", "kv")}
        lim = axle_limit(car, sub, j)
        bad = lim < ln["v"][m] - CONFLICT_MARGIN
        out[name] = {"n": int(m.sum()), "conflicts": int(bad.sum()), "share": float(bad.mean())}
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
