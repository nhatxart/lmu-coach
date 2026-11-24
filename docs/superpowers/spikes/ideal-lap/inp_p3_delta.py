"""Ideal inputs probe 3: ABSOLUTE vs DELTA input maps, scored as "predict lap j's inputs".
Maps fitted on the even laps (brake: -gxt = a brk + e(v); throttle: f table over the engine
curve; steer: steer = a_s k + b_s US(gy) + c0 with the US map of inp_p2). Test on pairs of odd
laps (i = the "your lap", j = the "other lap" playing the ideal), aligned by Lap Dist (2 m):
  ABSOLUTE: input_j = map^-1(state_j)
  DELTA   : input_j = input_i + [map^-1(state_j) - map^-1(state_i)]   when both points are in the
            same mode (both braking, both driving, both coasting); else ABSOLUTE
  NONE    : input_j = input_i (the trivial baseline: "the same as your lap")
States from the recorded channels (gxt, gy, k = yaw/v); inputs smoothed 0.2 s; lags from probe 1
(brake 0, throttle 0.04 s, steer 0.10 s). Scores: RMS over the lap per input (brake / throttle in
% points, steer in % of full lock) and RMS restricted to the points where the two laps differ
(|v_j - v_i| > 2 km/h).
usage: python inp_p3_delta.py <file substring> [...]"""
import sys, os, io, contextlib, itertools
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT / "backend"))
os.environ.setdefault("TRACK_REFERENCE_DIR", r"C:\Users\hoang\Projects\lmu-coach\data\track_reference")
SESS = Path(r"C:\Users\hoang\Projects\lmu-coach\data\sessions")
from app import ideal_car as C  # noqa: E402
from app import ideal_state as S  # noqa: E402
from app.ideal_state import smooth  # noqa: E402
from app.ideal_solver import unwrap_lap_dist  # noqa: E402
from app.car_identity import identify_car  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
G = 9.81; HZ = S.GRID_HZ; N = 10; LAG = dict(brk=0, thr=2, steer=5)
rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


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
    return path.name, identify_car(dict(laps[0].meta), laps[0].car_name), raws, steers


def lead(a, k):
    """input aligned to its response: the response at t belongs to the input at t - k."""
    return a if k == 0 else np.concatenate([np.full(k, a[0]), a[:-k]])


for sub in sys.argv[1:]:
    name, ident, raws, steers = load(sub)
    states = [S.lap_state(r) for r in raws]
    red = S.FileReduction(file=name, mtime=0, size=0, meta={}, track="", car_model=ident.model, car_class=ident.car_class,
                          game_version="", load_source=S.file_load_source(states), grade_source="pos_y", laps=states)
    car = C.fit_group([red], ident.car_class).car_for(name)
    L = []
    for r, st, sr in zip(raws, states, steers):
        slip = np.vstack([smooth(r["slip"][i], N) for i in range(4)])
        v = st.v
        L.append(dict(v=v, gxt=st.gx + st.grade + car.drag(v), gxa=st.gx + st.grade, gy=st.gy,
                      brk=lead(smooth(r["brk"], N), LAG["brk"]), thr=lead(smooth(r["thr"], N), LAG["thr"]),
                      steer=lead(smooth(sr, N), LAG["steer"]), tc=st.tc, abs=st.abs,
                      k=smooth(r["yaw"], N) / np.maximum(v, 5), da=slip[:2].mean(0) - slip[2:].mean(0),
                      ld=unwrap_lap_dist(np.asarray(r["lapdist"], float)) if r.get("lapdist") is not None else r["s"]))
    ev, od = L[0::2], L[1::2]
    cat = lambda LL, k: np.concatenate([x[k] for x in LL])
    # brake map (all braking samples, ABS off)
    b, y, v = cat(ev, "brk"), -cat(ev, "gxt"), cat(ev, "v")
    m = (b > 3) & (cat(ev, "thr") < 5) & (cat(ev, "abs") < 0.5)
    bco = np.linalg.lstsq(np.c_[b[m] / 100, v[m], np.ones(m.sum())], y[m], rcond=None)[0]
    # throttle map
    t, v, ga, gy = cat(ev, "thr"), cat(ev, "v"), cat(ev, "gxa"), cat(ev, "gy")
    base = (cat(ev, "brk") < 1) & (cat(ev, "tc") < 0.05) & (v > 8)
    X = lambda v, gy: np.c_[np.ones(len(v)), v, v * v, gy * gy]
    cco = np.linalg.lstsq(X(v, gy)[base & (t < 2)], ga[base & (t < 2)], rcond=None)[0]
    f = (ga - X(v, gy) @ cco) / np.maximum(car.power(v, gy) - X(v, gy) @ cco, 0.05)
    cen = np.arange(5, 100, 10)
    tab = np.array([np.median(f[base & (t >= a) & (t < a + 10) & (t > 2)]) if (base & (t >= a) & (t < a + 10) & (t > 2)).sum() >= 20 else np.nan for a in range(0, 100, 10)])
    okk = ~np.isnan(tab); tab = np.maximum.accumulate(np.interp(cen, cen[okk], tab[okk]))
    # steer map
    s, k, gy, da, v = cat(ev, "steer"), cat(ev, "k"), cat(ev, "gy"), cat(ev, "da"), cat(ev, "v")
    ms = (v > 12) & (np.abs(gy) > 0.05)
    U = lambda gy: np.c_[gy, gy * np.abs(gy)]
    uco = np.linalg.lstsq(U(gy)[ms], da[ms], rcond=None)[0]
    sco = np.linalg.lstsq(np.c_[k, U(gy) @ uco, np.ones(len(k))][ms], s[ms], rcond=None)[0]
    lock = float(np.percentile(np.abs(s), 99.5))

    def inv(d):
        """map^-1(state) on one lap: brake %, throttle %, steer, and the mode per sample."""
        gxt, v, gy = d["gxt"], d["v"], d["gy"]
        brk = np.clip((-gxt - bco[1] * v - bco[2]) / bco[0] * 100, 0, 100)
        coast = X(v, gy) @ cco; pw = car.power(v, gy)
        fr = np.clip((d["gxa"] - coast) / np.maximum(pw - coast, 0.05), 0, 1)
        thr = np.interp(fr, tab, cen); thr = np.where(fr >= tab[-1] - 1e-6, 100.0, thr)
        mode = np.where(-gxt > bco[1] * v + bco[2] + 0.05, -1, np.where(d["gxa"] > coast + 0.03, 1, 0))
        brk = np.where(mode == -1, brk, 0.0); thr = np.where(mode == 1, thr, 0.0)
        st = sco[0] * d["k"] + sco[1] * (U(gy) @ uco) + sco[2]
        return dict(brk=brk, thr=thr, steer=st, mode=mode)

    res = {kk: {"ABS": [], "DELTA": [], "NONE": []} for kk in ("brk", "thr", "steer")}
    resd = {kk: {"ABS": [], "DELTA": [], "NONE": []} for kk in ("brk", "thr", "steer")}
    for i, j in itertools.permutations(range(len(od)), 2):
        A, B = od[i], od[j]
        grid = np.arange(max(A["ld"][0], B["ld"][0]) + 5, min(A["ld"][-1], B["ld"][-1]) - 5, 2.0)
        on = lambda d, a: np.interp(grid, d["ld"], a)
        ia, ib = inv(A), inv(B)
        diff = np.abs(on(B, B["v"]) - on(A, A["v"])) > 2 / 3.6
        for kk in ("brk", "thr", "steer"):
            ya, yb = on(A, A[kk]), on(B, B[kk])
            pa, pb = on(A, ia[kk]), on(B, ib[kk])
            same = on(A, ia["mode"].astype(float)) == on(B, ib["mode"].astype(float))
            delta = np.where(same, ya + (pb - pa), pb) if kk != "steer" else ya + (pb - pa)
            if kk != "steer":
                delta = np.clip(delta, 0, 100)
            sc = 100 / lock if kk == "steer" else 1.0
            for nm, p in (("ABS", pb), ("DELTA", delta), ("NONE", ya)):
                res[kk][nm].append(sc * (p - yb)); resd[kk][nm].append(sc * (p - yb)[diff])
    print("=" * 110)
    print("%s | %s (%s) | %d odd laps -> %d ordered pairs | brake %.2f g/100%% | full lock (p99.5 |steer|) %.3f" % (
        name[-40:-7], ident.model, ident.car_class, len(od), len(od) * (len(od) - 1), bco[0], lock))
    for kk, unit in (("brk", "% pts"), ("thr", "% pts"), ("steer", "% of lock")):
        print("  %-5s RMS (%s) whole lap: ABS %.1f  DELTA %.1f  NONE %.1f | where the laps differ (>2 km/h): ABS %.1f  DELTA %.1f  NONE %.1f" % (
            kk, unit, *(rms(np.concatenate(res[kk][n])) for n in ("ABS", "DELTA", "NONE")),
            *(rms(np.concatenate(resd[kk][n])) for n in ("ABS", "DELTA", "NONE"))))
