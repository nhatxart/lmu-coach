"""Probe 7 (build round 2, Q10): does the SETUP change the grip level between files of one
car on one track, by more than the model's own noise?  Read-only on recordings.

Recordings carry no setup JSON (fable_p6_meta.py: metadata has compound, temperatures, tyre
radius, game version; Front/Rear Downforce, Drag, Front/Rear Ride Height are empty tables).
So the setup is MEASURED per file from what it does:
  camber / toe per axle (deg, straight-line median), hot pressure per axle (kPa), ride height
  per axle at 30-45 m/s and above 60 m/s (mm, wheel 'Ride Height'), and from the per-file load
  model (design 4.1): aero front share (a_i * c0_i), lateral load-transfer front share (|c_i| *
  c0_i, LLTD in the load source's units), static front share.
Grip level per axle (design 4.2, spike variant 'both', q = 0.5):
  cross : mu1/k/c fitted on file A, applied with file B's OWN load model to B's own saturated
          set: level_j = median log(|gy| / A_j).  (Self = 0 by construction.)
  noise : the same between the even and the odd laps of one file (V2 odd/even).
usage: python fable_p7_setup.py <group-name>=<substr>,<substr>,... [...]"""
import sys, types, os, copy
from pathlib import Path
import numpy as np
import duckdb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps, SESS

SRC = (Path(__file__).parent / "il_core.py").read_text(encoding="utf-8")
for a, b in (("return np.stack([np.ones_like(v), v * v, gx, gy, gz], axis=-1)", "return np.stack([np.ones_like(v), v * v, gx, gy, gz, np.abs(gy)], axis=-1)"),
             ("lam = L / c0[:, None]", "lam = np.maximum((C / c0[:, None]) @ X.T, 0.02)"),
             ("mu1.append(m1 * scale); kk.append(k); cc_.append(c)",
              "mu1.append(m1 * scale); kk.append(k); cc_.append(c); p.setdefault('_m', []).append(m); p['_arr'] = (v, gx, gy, gz); p.setdefault('knee', []).append(float(knee))")):
    assert a in SRC, a
    SRC = SRC.replace(a, b)
M = types.ModuleType("il_core_p7"); exec(compile(SRC, "il_core_p7", "exec"), M.__dict__)


def fit(f, laps=None):
    g = dict(f); g["laps"] = laps if laps is not None else f["laps"]
    return M.fit_car([g], q=0.5, verbose=False)


def level(A, B):
    """A's grip on B's loads and B's saturated sets; per axle median log ratio."""
    p = dict(B.p); p["mu1"], p["k"], p["c"] = A.p["mu1"], A.p["k"], A.p["c"]
    H = M.Car(p); v, gx, gy, gz = B.p["_arr"]
    Af, Ar = H.axles(v, gx, gy, gz)
    return [float(np.median(np.log(np.abs(gy)[m] / Aj[m]))) if m.sum() >= 50 else np.nan
            for m, Aj in zip(B.p["_m"], (Af, Ar))]


def fingerprint(path, car):
    c = duckdb.connect(str(path), read_only=True)
    q = lambda t: np.array(c.execute(f'select * from "{t}"').fetchall(), float)
    v = q("Ground Speed")[:, 0] / 3.6; gy = q("G Force Lat")[:, 0]; gx = q("G Force Long")[:, 0]
    out = {}
    st = (v > 30) & (v < 60) & (np.abs(gy) < 0.15) & (np.abs(gx) < 0.15)
    for name, scale in (("Camber", 57.2958), ("Toe", 57.2958)):
        try:
            a = q(name)
            out[name.lower()] = [float(np.median(np.abs(a[st][:, i]) * scale)) for i in range(4)]
        except Exception:
            out[name.lower()] = None
    try:
        rh = q("Ride Height") * 1000
        out["rh_mid"] = [float(np.median(rh[st & (v < 45)][:, idx].mean(1))) for idx in ([0, 1], [2, 3])]
        hs = (v > 60) & (np.abs(gy) < 0.15) & (np.abs(gx) < 0.15)
        out["rh_fast"] = [float(np.median(rh[hs][:, idx].mean(1))) for idx in ([0, 1], [2, 3])] if hs.sum() > 100 else None
    except Exception:
        out["rh_mid"] = out["rh_fast"] = None
    try:
        pr = q("TyresPressure"); pr = pr[(pr > 50).all(1)]
        pr = pr[len(pr) // 5:]                                  # skip the out-lap warm-up
        out["press"] = [float(np.median(pr[:, idx].mean(1))) for idx in ([0, 1], [2, 3])]
    except Exception:
        out["press"] = None
    md = dict(c.execute("select key, value from metadata").fetchall()); c.close()
    out["cond"] = "%s/%s %sC v%s" % (md.get("TyreCompoundFront", "?"), md.get("TyreCompoundRear", "?"), md.get("TrackTempC", "?"), md.get("GameVersion", "?"))
    C = np.array(car.p["load_coef"]); c0 = np.array(car.p["static"])
    aero = C[:, 1] * c0; lat = np.abs(C[:, 3]) * c0 + np.abs(C[:, 5]) * c0
    out["aero_f"] = float(aero[:2].sum() / aero.sum()); out["aero_g250"] = float(aero.sum() / c0.sum() * (250 / 3.6) ** 2)
    out["lltd_f"] = float(lat[:2].sum() / lat.sum()); out["stat_f"] = float(c0[:2].sum() / c0.sum())
    return out


def fmt(x, n=1):
    return "-" if x is None else "/".join(("%." + str(n) + "f") % a for a in x)


for arg in sys.argv[1:]:
    gname, subs = arg.split("=")
    paths = sorted(p for p in Path(SESS).glob("*.duckdb") if any(s in p.name for s in subs.split(",")))
    print("=" * 120); print("GROUP", gname)
    fits = []
    for p in paths:
        f = load_file(p.name); laps = sorted(real_laps(f), key=lambda l: l["n"])
        if len(laps) < 4 or f["laps"][0]["slip"] is None:
            print("  skip %s (%d laps)" % (p.name[-30:-7], len(laps))); continue
        car = fit(f, laps); fp = fingerprint(p, car)
        ev, od = fit(f, laps[0::2]), fit(f, laps[1::2])
        noise = [np.nanmean(np.abs([a, b])) for a, b in zip(level(ev, od), level(od, ev))]
        best = min(l["time"] for l in laps)
        print("  %s %-6s laps %2d best %.2f | %s | camber %s toe %s | press %s | rh mid %s fast %s | aero_f %.3f (%.2f g@250) lltd_f %.3f stat_f %.3f"
              % (p.name[-30:-7], f["model"][:6], len(laps), best, fp["cond"], fmt(fp["camber"], 2), fmt(fp["toe"], 2), fmt(fp["press"], 0),
                 fmt(fp["rh_mid"]), fmt(fp["rh_fast"]), fp["aero_f"], fp["aero_g250"], fp["lltd_f"], fp["stat_f"]))
        print("      mu1 F/R %.3f/%.3f  k %s c %s  knee %s  nsat %d/%d | odd/even level |F| %.3f |R| %.3f"
              % (car.p["mu1"][0], car.p["mu1"][1], fmt(car.p["k"], 2), fmt(car.p["c"], 2), fmt(car.p["knee"]), car.p["_m"][0].sum(), car.p["_m"][1].sum(), *noise))
        fits.append((p.name[-30:-7], car, noise, fp))
    for i in range(len(fits)):
        for j in range(len(fits)):
            if i == j: continue
            a, b = fits[i], fits[j]
            lv = level(a[1], b[1])
            print("  fit %s -> test %s : level F %+.3f R %+.3f  (F-R %+.3f)" % (a[0], b[0], lv[0], lv[1], lv[0] - lv[1]))
