"""Fable probe 3: is the residual of the grip model NOISE or PLACE?
On saturated samples: log(|gy| / min(Af, Ar)), split into between-location (25 m bins of
Lap Dist, median over laps) and within-location (lap to lap) spread; how much banking
(topography) explains; and whether a per-location correction learned on even laps
holds on odd laps. Car model = variant 'both' of fable_p2 (|gy| load term, capacity on
modelled lam). Read-only.  usage: python fable_p3.py file [file...]"""
import sys, types, os
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, BACKEND, REF

s_ = (Path(__file__).parent / "il_core.py").read_text(encoding="utf-8")
for a, b in (("return np.stack([np.ones_like(v), v * v, gx, gy, gz], axis=-1)", "return np.stack([np.ones_like(v), v * v, gx, gy, gz, np.abs(gy)], axis=-1)"),
             ("lam = L / c0[:, None]", "lam = np.maximum((C / c0[:, None]) @ X.T, 0.02)")):
    assert a in s_; s_ = s_.replace(a, b)
M = types.ModuleType("il_core_both"); exec(compile(s_, "il_core_both", "exec"), M.__dict__)
os.environ.setdefault("TRACK_REFERENCE_DIR", REF)
if BACKEND not in sys.path: sys.path.insert(0, BACKEND)
from app import track_topography as tt

BIN = 25.0
for name in sys.argv[1:]:
    f = load_file(name); laps = real_laps(f)
    car = M.fit_car([f], q=0.5, verbose=False)
    rows = []
    for li, l in enumerate(laps):
        st = M.lap_state(l, car.p["load_source"])
        v, gx, gy, gz, gr = st["v"], st["gx"], st["gy"], st["gz"], st["grade"]
        Af, Ar = car.axles(v, gx, gy, gz)
        gxt = gx + gr + car.drag(v)
        ld = l["lapdist"] if l["lapdist"] is not None else l["s"]
        rows.append(np.c_[np.full(len(v), li), ld, np.log(np.abs(gy) / np.minimum(Af, Ar) + 1e-9), np.abs(gy), gy, gz, gxt, v,
                          st["slip"][:2].mean(0), st["slip"][2:].mean(0), st["yawacc"]])
    R = np.vstack(rows)
    lap, ld, lu, agy, gy, gz, gxt, v, sF, sR, ya = R.T
    hard = (agy > 0.6 * np.percentile(agy, 99)) & (np.abs(gxt) < 0.25) & (v > 15)
    sat = hard & ((sF >= np.percentile(sF[hard], 60)) | (sR >= np.percentile(sR[hard], 60)))
    L = float(np.max(ld))
    topo = tt.load_topography(f["track"], L)
    bank = tt.grade_bank(topo[1], ld)[1] if topo else np.zeros(len(ld))
    print("=" * 100); print("%s | %s | %d laps | saturated samples %d (%.0f s) | topography %s" % (f["model"], f["track"], len(laps), sat.sum(), sat.sum() / 50, "yes" if topo else "NO"))
    print("log-ratio over saturated samples: median %+.3f sd %.3f  p75 %+.3f p90 %+.3f p97 %+.3f" % (np.median(lu[sat]), lu[sat].std(), *np.percentile(lu[sat], [75, 90, 97])))
    b = (ld // BIN).astype(int)
    cell = {}
    for i in np.where(sat)[0]:
        cell.setdefault((b[i], int(lap[i])), []).append(i)
    cm = {k: (np.median(lu[ix]), ix) for k, ix in cell.items() if len(ix) >= 5}        # per (bin, lap) median
    bins = {}
    for (bb, ll), (m, ix) in cm.items():
        bins.setdefault(bb, []).append((ll, m, ix))
    bm = {bb: np.median([m for _, m, _ in v_]) for bb, v_ in bins.items() if len(v_) >= 3}
    within = np.concatenate([[m - bm[bb] for _, m, _ in v_] for bb, v_ in bins.items() if bb in bm])
    between = np.array(list(bm.values()))
    print("25 m bins with >= 3 laps saturated: %d (%.0f m of %.0f m)" % (len(bm), len(bm) * BIN, L))
    print("  BETWEEN locations (sd of bin medians): %.3f   WITHIN a location, lap to lap (sd): %.3f   -> place explains %.0f%% of the variance" % (
        between.std(), within.std(), 100 * between.var() / (between.var() + within.var())))
    print("  bin medians: p10 %+.3f p50 %+.3f p90 %+.3f max %+.3f" % (*np.percentile(between, [10, 50, 90]), between.max()))
    # banking
    if topo:
        bk = np.array([np.median((bank * np.sign(gy))[np.concatenate([ix for _, _, ix in bins[bb]])]) for bb in bm])
        gzb = np.array([np.median(gz[np.concatenate([ix for _, _, ix in bins[bb]])]) for bb in bm])
        for nm, x in (("bank*sign(gy)", bk), ("gz", gzb)):
            if x.std() > 0:
                r = np.corrcoef(x, between)[0, 1]; sl = np.polyfit(x, between, 1)
                print("  bin median vs %-13s: r %+.2f, slope %+.2f per unit (x sd %.3f) -> sd after removing it %.3f" % (nm, r, sl[0], x.std(), (between - np.polyval(sl, x)).std()))
    top = sorted(bm.items(), key=lambda kv: -kv[1])[:8]
    print("  highest locations (model too LOW there):  " + "  ".join("%dm:%+.2f(n%d)" % (bb * BIN, m, len(bins[bb])) for bb, m in top))
    low = sorted(bm.items(), key=lambda kv: kv[1])[:6]
    print("  lowest locations  (driver under the model): " + "  ".join("%dm:%+.2f(n%d)" % (bb * BIN, m, len(bins[bb])) for bb, m in low))
    # held-out by laps: map from even laps, scored on odd laps
    ev = {}
    for (bb, ll), (m, ix) in cm.items():
        if ll % 2 == 0: ev.setdefault(bb, []).append(m)
    raw, cor = [], []
    for (bb, ll), (m, ix) in cm.items():
        if ll % 2 == 1:
            n = len(ev.get(bb, []))
            e = (np.median(ev[bb]) * n / (n + 1.0)) if n else 0.0
            raw.append(lu[ix]); cor.append(lu[ix] - e)
    raw, cor = np.concatenate(raw), np.concatenate(cor)
    print("  held-out laps (odd), location map from even laps, shrink n/(n+1):")
    print("     without map: sd %.3f  p75 %+.3f p90 %+.3f p97 %+.3f p99.5 %+.3f" % (raw.std(), *np.percentile(raw, [75, 90, 97, 99.5])))
    print("     with map   : sd %.3f  p75 %+.3f p90 %+.3f p97 %+.3f p99.5 %+.3f" % (cor.std(), *np.percentile(cor, [75, 90, 97, 99.5])))
