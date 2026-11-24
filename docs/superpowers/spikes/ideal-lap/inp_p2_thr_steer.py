"""Ideal inputs probe 2 (even laps fit, odd laps scored; read-only):
THROTTLE over ALL lateral g (part throttle lives in corners): f = (gxa - coast(v)) / (power(v, gy)
  - coast(v)) per 10 % pedal bin (TC off, brake off); monotone table; held-out RMS of gxa, and the
  pedal-error view: invert the table at the measured gxa -> predicted pedal vs recorded pedal.
  Pedal use: share of drive time at 0-5 / 5-95 / 95-100 %.
STEER bicycle form: steer = a * k + b * (alpha_f - alpha_r) + c0, k = yaw / v (path curvature),
  alpha = mean front / rear slip angle (rad, signed). Implied wheelbase L = a / b, lock = 1 / b
  (road-wheel rad per unit steer). Then the usable form for the ideal (slip angles unknown there):
  US map: (alpha_f - alpha_r) = u1 * gy + u2 * gy * |gy| (learned), steer = a k + b US(gy) + c0.
  Held-out R2 / RMS for: measured slips (upper bound), US map, and k only.
  Driver corrections: RMS of (steer smoothed 0.2 s - steer smoothed 1.0 s) on the same samples.
usage: python inp_p2_thr_steer.py <file substring> [...]"""
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
    return path.name, identify_car(dict(laps[0].meta), laps[0].car_name), raws, steers


def shift(a, k):
    return a if k == 0 else np.concatenate([np.full(k, a[0]), a[:-k]])


r2 = lambda y, p: 1 - np.var(y - p) / np.var(y)
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))

for sub in sys.argv[1:]:
    name, ident, raws, steers = load(sub)
    states = [S.lap_state(r) for r in raws]
    red = S.FileReduction(file=name, mtime=0, size=0, meta={}, track="", car_model=ident.model, car_class=ident.car_class,
                          game_version="", load_source=S.file_load_source(states), grade_source="pos_y", laps=states)
    car = C.fit_group([red], ident.car_class).car_for(name)
    L = []
    for r, st, sr in zip(raws, states, steers):
        slip = None if r.get("slip") is None else np.vstack([smooth(r["slip"][i], N) for i in range(4)])
        L.append(dict(v=st.v, gxa=st.gx + st.grade, gy=st.gy, brk=smooth(r["brk"], N), thr=smooth(r["thr"], N), tc=st.tc,
                      steer=smooth(sr, N), steer1=smooth(sr, 50), yaw=None if r.get("yaw") is None else smooth(r["yaw"], N),
                      af=None if slip is None else slip[:2].mean(0), ar=None if slip is None else slip[2:].mean(0)))
    ev, od = L[0::2], L[1::2]
    cat = lambda LL, k, lag=0: np.concatenate([shift(x[k], lag) for x in LL])
    print("=" * 110); print("%s | %s (%s) | %d laps" % (name[-40:-7], ident.model, ident.car_class, len(L)))
    # ---------- THROTTLE
    lt = 2
    def prep(LL):
        t = cat(LL, "thr", lt); v = cat(LL, "v"); ga = cat(LL, "gxa"); gy = cat(LL, "gy")
        base = (cat(LL, "brk", lt) < 1) & (cat(LL, "tc") < 0.05) & (v > 8)
        return t, v, ga, gy, base
    t, v, ga, gy, base = prep(ev)
    cm = base & (t < 2)
    X = lambda v, gy: np.c_[np.ones(len(v)), v, v * v, gy * gy]
    cco = np.linalg.lstsq(X(v, gy)[cm], ga[cm], rcond=None)[0]
    coast = X(v, gy) @ cco; pw = car.power(v, gy)
    f = (ga - coast) / np.maximum(pw - coast, 0.05)
    cen = np.arange(5, 100, 10); tab = []
    for a in range(0, 100, 10):
        mm = base & (t >= a) & (t < a + 10) & (t > 2)
        tab.append(np.median(f[mm]) if mm.sum() >= 20 else np.nan)
    tab = np.array(tab); ok = ~np.isnan(tab)
    tab = np.maximum.accumulate(np.interp(cen, cen[ok], tab[ok]))
    t, v, ga, gy, base = prep(od)
    coast = X(v, gy) @ cco; pw = car.power(v, gy)
    pred = coast + np.interp(t, cen, tab) * (pw - coast)
    drive = base & (t > 2)
    fr = np.clip((ga - coast) / np.maximum(pw - coast, 0.05), 0, 1.2)
    ped = np.interp(fr, tab, cen)
    part = drive & (t > 5) & (t < 95)
    use = [np.mean((t[drive] < 5)), np.mean((t[drive] >= 5) & (t[drive] < 95)), np.mean(t[drive] >= 95)]
    print("  THROTTLE f table: %s | pedal use while driving 0-5/5-95/95-100 %%: %.0f/%.0f/%.0f %%" % (np.round(tab, 2).tolist(), *(100 * np.array(use))))
    print("     held-out gxa: all drive R2 %.3f RMS %.3f g | part throttle RMS %.3f g (n %d) | pedal from gxa (inverse) on part throttle: RMS %.0f %% pts, bias %+.0f" % (
        r2(ga[drive], pred[drive]), rms((ga - pred)[drive]), rms((ga - pred)[part]), part.sum(), rms((ped - t)[part]), np.mean((ped - t)[part])))
    # ---------- STEER
    if L[0]["yaw"] is None or L[0]["af"] is None:
        print("  STEER: no yaw / slip"); continue
    ls = 5
    def sprep(LL):
        s = cat(LL, "steer", ls); s1 = cat(LL, "steer1", ls); v = cat(LL, "v"); gy = cat(LL, "gy")
        k = cat(LL, "yaw") / np.maximum(v, 5); da = cat(LL, "af") - cat(LL, "ar")
        m = (v > 12) & (np.abs(gy) > 0.05)
        return s, s1, v, gy, k, da, m
    s, s1, v, gy, k, da, m = sprep(ev)
    A = np.c_[k, da, np.ones(len(k))]; co = np.linalg.lstsq(A[m], s[m], rcond=None)[0]
    U = np.c_[gy, gy * np.abs(gy)]; uco = np.linalg.lstsq(U[m], da[m], rcond=None)[0]
    B = np.c_[k, U @ uco, np.ones(len(k))]; cob = np.linalg.lstsq(B[m], s[m], rcond=None)[0]
    K = np.c_[k, np.ones(len(k))]; cok = np.linalg.lstsq(K[m], s[m], rcond=None)[0]
    print("  STEER fit: steer = %.3f k %+.3f (af-ar) %+.4f | implied wheelbase %.2f m, lock %.1f deg road wheel per full steer | US map da = %.4f gy %+.4f gy|gy| (R2 %.2f on fit laps)" % (
        *co, co[0] / co[1], np.degrees(1 / co[1]), *uco, r2(da[m], (U @ uco)[m])))
    s, s1, v, gy, k, da, m = sprep(od)
    p_meas = np.c_[k, da, np.ones(len(k))] @ co
    p_us = np.c_[k, np.c_[gy, gy * np.abs(gy)] @ uco, np.ones(len(k))] @ cob
    p_k = np.c_[k, np.ones(len(k))] @ cok
    hard = m & (np.abs(gy) > 1.0)
    p95 = np.percentile(np.abs(s[m]), 95)
    for nm, p in (("measured slips", p_meas), ("US map", p_us), ("k only", p_k)):
        print("     held-out %-14s R2 %.3f RMS %.4f (%.1f %% of p95 %.3f) | hard RMS %.4f" % (nm, r2(s[m], p[m]), rms((s - p)[m]), 100 * rms((s - p)[m]) / p95, p95, rms((s - p)[hard])))
    print("     driver corrections (0.2 s vs 1.0 s smoothing): RMS %.4f; US map vs the 1.0 s steer: RMS %.4f" % (rms((s - s1)[m]), rms((s1 - p_us)[m])))
