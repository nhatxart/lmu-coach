"""Ideal inputs probe 1: input maps fitted on EVEN laps, scored on ODD laps of one file.
Read-only on recordings; uses this worktree's ideal_* modules (fit_group for drag, power, kb).
Signals at 50 Hz, smoothed 0.2 s; tyre-side longitudinal gxt = gx + grade + drag(v).
BRAKE   samples: brk > 3 %, thr < 5 %.  lag L_b (pedal leads decel) searched 0..0.3 s.
        map: -gxt = a * brk/100 + e(v)  on non-ABS samples (abs < 0.5) with |gy| < 0.4
        (e(v) = engine braking/coast decel, linear in v).  ABS share per brake-% bin.
        Front share of braking force (forces recorded): median Fx_front / Fx_total by brk bin.
THROTTLE samples: brk < 1 %, |gy| < 0.4, TC off. f = (gxa - coast(v)) / (power(v) - coast(v)),
        gxa = gx + grade (kinematic + grade, the engine curve's own quantity); f per thr bin;
        map thr -> f by a monotone table; held-out RMS of predicted gxa.
STEER   samples: v > 12, |gy| > 0.05.  k = g*gy/v^2 (path curvature, kinematic).  lag searched.
        map: steer = c1 * k + c2 * gy + c0  (Ackermann + understeer gradient); also with yaw-rate
        curvature k_yaw = yaw / v when Yaw Rate exists.  held-out RMS, R2, implied c1 per car.
usage: python inp_p1_maps.py <file substring> [...]"""
import sys, os, io, contextlib
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT / "backend"))
os.environ.setdefault("TRACK_REFERENCE_DIR", r"C:\Users\hoang\Projects\lmu-coach\data\track_reference")
SESS = Path(r"C:\Users\hoang\Projects\lmu-coach\data\sessions")
from app import ideal_car as C  # noqa: E402
from app import ideal_state as S  # noqa: E402
from app.ideal_state import smooth  # noqa: E402
from app.car_identity import identify_car  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
G = 9.81; HZ = S.GRID_HZ; N = 10


def load(sub):
    path = next(p for p in sorted(SESS.glob("*.duckdb")) if sub in p.name)
    S._ensure_wheel_channels()
    from app.telemetry import load_telemetry_file
    from app import track_topography as tt
    with contextlib.redirect_stdout(io.StringIO()):
        laps = load_telemetry_file(path, grid_hz=HZ)
    q = S.qualifying_laps(laps, tt._read_pit_laps(path))
    raws, steers = [], []
    for l in q:
        layout = None
        if "Pos Y" not in l.channels:
            hit = tt.load_topography(l.track_name, tt._lap_length_m(l)); layout = hit[1] if hit else None
        raws.append(S.lap_raw(l, layout))
        steers.append(np.nan_to_num(np.asarray(l.channels.get("Steering Pos", np.zeros(len(l.time_s))), float)))
    ident = identify_car(dict(laps[0].meta), laps[0].car_name)
    return path.name, ident, raws, steers


def shift(a, k):
    """a delayed by k samples (input leads response: compare input[t-k] with response[t])."""
    return a if k == 0 else np.concatenate([np.full(k, a[0]), a[:-k]])


def r2(y, p):
    return 1 - np.var(y - p) / np.var(y)


for sub in sys.argv[1:]:
    name, ident, raws, steers = load(sub)
    states = [S.lap_state(r) for r in raws]
    red = S.FileReduction(file=name, mtime=0, size=0, meta={}, track="", car_model=ident.model, car_class=ident.car_class,
                          game_version="", load_source=S.file_load_source(states), grade_source="pos_y", laps=states)
    g = C.fit_group([red], ident.car_class); car = g.car_for(name)
    print("=" * 110); print("%s | %s (%s) | %d laps | kb %.3f kt %.3f d2 %.2e" % (name[-40:-7], ident.model, ident.car_class, len(raws), car.kb, car.kt, car.d2))
    lapd = []
    for r, st, sr in zip(raws, states, steers):
        v = st.v; gxa = st.gx + st.grade
        d = dict(v=v, gxa=gxa, gxt=gxa + car.drag(v), gy=st.gy, brk=smooth(r["brk"], N), thr=smooth(r["thr"], N),
                 abs=st.abs, tc=st.tc, steer=smooth(sr, N), yaw=None if r.get("yaw") is None else smooth(r["yaw"], N),
                 fx=st.fx)
        lapd.append(d)
    ev, od = lapd[0::2], lapd[1::2]
    cat = lambda L, k, lag=0: np.concatenate([shift(x[k], lag) for x in L])
    resp = lambda L, k: np.concatenate([x[k] for x in L])

    # ---------- BRAKE
    best = None
    for lag in range(0, 16):
        b = cat(ev, "brk", lag); y = -resp(ev, "gxt"); m = (b > 3) & (cat(ev, "thr", lag) < 5) & (resp(ev, "abs") < 0.5) & (np.abs(resp(ev, "gy")) < 0.4)
        X = np.c_[b[m] / 100, resp(ev, "v")[m], np.ones(m.sum())]
        co = np.linalg.lstsq(X, y[m], rcond=None)[0]; e = r2(y[m], X @ co)
        if best is None or e > best[0]: best = (e, lag, co)
    _, lb, co = best
    b = cat(od, "brk", lb); y = -resp(od, "gxt"); v = resp(od, "v")
    m = (b > 3) & (cat(od, "thr", lb) < 5) & (resp(od, "abs") < 0.5) & (np.abs(resp(od, "gy")) < 0.4)
    p = np.c_[b / 100, v, np.ones(len(b))] @ co
    print("  BRAKE lag %.2f s | -gxt = %.3f*brk + %.5f*v + %.3f | held-out non-ABS straight: R2 %.3f RMS %.3f g (n %d) | in brk-%% units RMS %.1f %%" % (
        lb / HZ, *co, r2(y[m], p[m]), np.sqrt(np.mean((y[m] - p[m]) ** 2)), m.sum(), 100 * np.sqrt(np.mean((y[m] - p[m]) ** 2)) / co[0]))
    ab = resp(od, "abs") > 0.5; ball = (b > 3) & (cat(od, "thr", lb) < 5)
    rows = []
    for lo in range(0, 100, 20):
        mm = ball & (b >= lo) & (b < lo + 20)
        if mm.sum() > 30:
            rows.append("%d-%d%%: n%d decel p50 %.2f ABS %.0f%%" % (lo, lo + 20, mm.sum(), np.median(y[mm]), 100 * ab[mm].mean()))
    print("     by pedal: " + " | ".join(rows))
    allb = cat(lapd, "brk"); print("     pedal shown: p95 %.0f %% p99 %.0f %% max %.0f %% | brake zones peak pedal median %.0f %%" % (
        *np.percentile(allb[allb > 3], [95, 99]), allb.max(), np.median([x["brk"].max() for x in lapd])))
    # speed dependence: residual vs v at high pedal
    hm = m & (b > 50)
    if hm.sum() > 50:
        print("     high-pedal residual vs v: slope %.4f g per m/s (v %.0f-%.0f)" % (np.polyfit(v[hm], (y - p)[hm], 1)[0], *np.percentile(v[hm], [5, 95])))
    if lapd[0]["fx"] is not None:
        fx = np.concatenate([x["fx"] for x in od], axis=1)
        tot = fx.sum(0); mf = ball & (np.abs(tot) > 2000)
        sh = fx[:2].sum(0)[mf] / tot[mf]
        print("     front share of braking force (forces): median %.3f p10/p90 %.3f/%.3f" % (np.median(sh), *np.percentile(sh, [10, 90])))

    # ---------- THROTTLE
    def coast_and_f(L, lag):
        t = cat(L, "thr", lag); v = resp(L, "v"); ga = resp(L, "gxa")
        base = (cat(L, "brk", lag) < 1) & (np.abs(resp(L, "gy")) < 0.4) & (resp(L, "tc") < 0.05)
        cm = base & (t < 2)
        cf = np.polyfit(v[cm], ga[cm], 2) if cm.sum() > 50 else np.zeros(3)
        coast = np.polyval(cf, v); pw = car.power(v, resp(L, "gy"))
        f = (ga - coast) / np.maximum(pw - coast, 0.05)
        return t, v, ga, base, cf, coast, pw, f
    bestt = None
    for lag in range(0, 16):
        t, v, ga, base, cf, coast, pw, f = coast_and_f(ev, lag)
        mm = base & (t > 2)
        edges = np.arange(0, 101, 10); tab = [np.median(f[mm & (t >= a) & (t < a + 10)]) if (mm & (t >= a) & (t < a + 10)).sum() > 20 else np.nan for a in edges[:-1]]
        tab = np.maximum.accumulate(np.nan_to_num(np.array(tab), nan=0.0))
        pred = coast + np.interp(t, edges[:-1] + 5, tab) * (pw - coast)
        e = r2(ga[base], pred[base])
        if bestt is None or e > bestt[0]: bestt = (e, lag, cf, tab)
    _, lt, cf, tab = bestt
    t, v, ga, base, _, _, pw, f = coast_and_f(od, lt)
    coast = np.polyval(cf, v); pred = coast + np.interp(t, np.arange(5, 100, 10), tab) * (pw - coast)
    full = base & (t > 98)
    part = base & (t > 5) & (t < 95)
    print("  THROTTLE lag %.2f s | f(thr) table 5..95%%: %s" % (lt / HZ, np.round(tab, 2).tolist()))
    print("     held-out (brk off, |gy|<0.4, TC off): R2 %.3f RMS %.3f g | full throttle RMS %.3f g (n %d) | part throttle RMS %.3f g (n %d, %.0f%% of drive samples)" % (
        r2(ga[base], pred[base]), np.sqrt(np.mean((ga - pred)[base] ** 2)), np.sqrt(np.mean((ga - pred)[full] ** 2)), full.sum(),
        np.sqrt(np.mean((ga - pred)[part] ** 2)), part.sum(), 100 * part.sum() / max((base & (t > 5)).sum(), 1)))
    tcm = resp(od, "tc") > 0.5
    print("     TC active on %.1f%% of drive samples; pedal when TC active p10/p50 %.0f/%.0f %%" % (
        100 * tcm[t > 5].mean(), *np.percentile(t[tcm], [10, 50])) if tcm.any() else "     TC never active")

    # ---------- STEERING
    def stfit(L, lag, use_yaw=False):
        s = cat(L, "steer", lag); v = resp(L, "v"); gy = resp(L, "gy")
        k = (resp(L, "yaw") / np.maximum(v, 5)) if use_yaw else G * gy / np.maximum(v, 5) ** 2
        m = (v > 12) & (np.abs(gy) > 0.05)
        return s, k, gy, m
    out = []
    for use_yaw in ((False, True) if lapd[0]["yaw"] is not None else (False,)):
        bests = None
        for lag in range(0, 16):
            s, k, gy, m = stfit(ev, lag, use_yaw)
            X = np.c_[k, gy, np.ones(len(k))]
            co = np.linalg.lstsq(X[m], s[m], rcond=None)[0]; e = r2(s[m], X[m] @ co)
            if bests is None or e > bests[0]: bests = (e, lag, co)
        _, ls, co = bests
        s, k, gy, m = stfit(od, ls, use_yaw)
        X = np.c_[k, gy, np.ones(len(k))]; p = X @ co
        hard = m & (np.abs(gy) > 1.0)
        X2 = np.c_[k, np.ones(len(k))]; co2 = np.linalg.lstsq(np.c_[stfit(ev, ls, use_yaw)[1], np.ones(len(stfit(ev, ls, use_yaw)[1]))][stfit(ev, ls, use_yaw)[3]], stfit(ev, ls, use_yaw)[0][stfit(ev, ls, use_yaw)[3]], rcond=None)[0]
        p2 = X2 @ co2
        out.append("%s lag %.2f s: steer = %.2f*k %+.4f*gy %+.4f | held-out R2 %.3f RMS %.4f (= %.1f %% of the p95 |steer| %.3f) | hard |gy|>1: RMS %.4f | without the gy term R2 %.3f RMS %.4f" % (
            "yaw-k" if use_yaw else "kin-k", ls / HZ, *co, r2(s[m], p[m]), np.sqrt(np.mean((s - p)[m] ** 2)),
            100 * np.sqrt(np.mean((s - p)[m] ** 2)) / np.percentile(np.abs(s[m]), 95), np.percentile(np.abs(s[m]), 95),
            np.sqrt(np.mean((s - p)[hard] ** 2)), r2(s[m], p2[m]), np.sqrt(np.mean((s - p2)[m] ** 2))))
    for o in out:
        print("  STEER " + o)
