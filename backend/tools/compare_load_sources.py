"""
Pilot: can Susp Force stand in for Tire Load in the grip proxy? (read-only)

    python -I backend/tools/compare_load_sources.py "*Genesis*" "*Ferrari AF Corse*"

GT3 cars (BMW M4, Porsche, Mustang, ...) get ZERO Tire Load / Tire Lateral
Force from LMU (a Studio 397 side issue, see DevNotes 10/18/2026), so their
tyre model takes load from Susp Force. That substitute can only be tested on
cars that have BOTH, plus the game's own Tire Lateral Force: today the
10/18 hf1 Hypercar files. A GT3 run would only prove the files lack the data.

For each axle it compares, on steady-state samples against the game's own
mu = sum|Fy| / sum Fz:
  - raw lateral G (no load normalisation)
  - lat G / load_ratio, load from Tire Load (the best case)
  - lat G / load_ratio, load from Susp Force (what GT3 files can use)
Susp Force is pushrod load, so its size differs from wheel load (x1.5-2.3
here); only the RATIO to a low-speed straight-line reference is used, as in
tire_model.file_load_reference.
Args are globs on file names in data/sessions (quote them; "*Ferrari*" alone
also matches the Imola file).
"""
import glob, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

repo = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(repo / "backend"))
from app import telemetry, tire_model as tm
telemetry.WHEEL_CHANNELS.append("Tire Lateral Force")


def pair(lap, ch, w):
    a, b = lap.channels.get(f"{ch} {w[0]}"), lap.channels.get(f"{ch} {w[1]}")
    return None if a is None or b is None else (a + b) / 2


def ref_load(laps, ch, w):
    vals = []
    for lap in laps:
        m = tm._ref_mask(lap); v = pair(lap, ch, w)
        if m is not None and v is not None:
            vals.append(v[m])
    v = np.concatenate(vals) if vals else np.array([])
    return float(np.median(v)) if len(v) >= tm.REF_LOAD_MIN_SAMPLES and np.median(v) > 0 else None


if len(sys.argv) < 2:
    sys.exit(__doc__)
for pat in sys.argv[1:]:
    found = glob.glob(str(repo / "data" / "sessions" / pat))
    if not found:
        print(f"no file matches {pat}"); continue
    path = found[0]
    laps = [l for l in telemetry.load_telemetry_file(Path(path)) if l.lap_time_s < 600]
    print(f"\n=== {laps[0].car_name} / {laps[0].meta.get('VehicleModel')}  {os.path.basename(path)[-30:-7]}  ({len(laps)} laps)")
    for axle, w in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        ref_tl, ref_sf = ref_load(laps, "Tire Load", w), ref_load(laps, "Susp Force", w)
        if not ref_tl or not ref_sf:
            print(f" {axle}: no low-speed straight reference (TL {ref_tl}, SF {ref_sf}) - skipped"); continue
        S = {k: [] for k in ("fz_tl", "fz_sf", "lat", "sa", "mu_true", "steady")}
        for lap in laps:
            sp, lat, lg = lap.channels.get("Ground Speed"), lap.channels.get("G Force Lat"), lap.channels.get("G Force Long")
            fz_l, fz_r = lap.channels.get(f"Tire Load {w[0]}"), lap.channels.get(f"Tire Load {w[1]}")
            fy_l, fy_r = lap.channels.get(f"Tire Lateral Force {w[0]}"), lap.channels.get(f"Tire Lateral Force {w[1]}")
            tl, sf, sa = pair(lap, "Tire Load", w), pair(lap, "Susp Force", w), pair(lap, "Slip Angle", w)
            if any(x is None for x in (sp, lat, lg, fz_l, fz_r, fy_l, fy_r, tl, sf, sa)):
                continue
            ya = lap.channels.get("Yaw Accel")
            steady = (sp > tm.MIN_SPEED_KMH_FOR_MODEL) & (np.abs(lg) < tm.STEADY_MAX_LONG_G)
            if ya is not None:
                steady &= np.abs(ya) < tm.STEADY_MAX_YAW_ACCEL
            mu = (np.abs(fy_l) + np.abs(fy_r)) / np.maximum(fz_l + fz_r, 1.0)
            mu[(fz_l + fz_r) < 0.3 * np.nanmedian(fz_l + fz_r)] = np.nan
            for k, v in (("fz_tl", tl), ("fz_sf", sf), ("lat", np.abs(lat)), ("sa", np.degrees(sa)), ("mu_true", mu), ("steady", steady)):
                S[k].append(v)
        if not S["sa"]:
            print(f" {axle}: file lacks Tire Load / Susp Force / Tire Lateral Force - nothing to compare"); continue
        S = {k: np.concatenate(v) for k, v in S.items()}
        ok_all = np.isfinite(S["fz_tl"]) & np.isfinite(S["fz_sf"])
        r_load = np.corrcoef(S["fz_tl"][ok_all], S["fz_sf"][ok_all])[0, 1]
        ratio = np.median(S["fz_sf"][ok_all] / np.maximum(S["fz_tl"][ok_all], 1))
        lr_tl, lr_sf = S["fz_tl"] / ref_tl, S["fz_sf"] / ref_sf
        print(f" {axle}: load r(TireLoad, SuspForce)={r_load:.3f}  SF/TL median={ratio:.3f}  "
              f"load_ratio range TL {np.nanpercentile(lr_tl,[5,95]).round(2)} vs SF {np.nanpercentile(lr_sf,[5,95]).round(2)}")
        m = S["steady"] & np.isfinite(S["mu_true"]) & np.isfinite(S["sa"]) & np.isfinite(lr_tl) & np.isfinite(lr_sf)
        sa, mu_true = S["sa"][m], S["mu_true"][m]
        proxies = {
            "raw lat G (no load)": S["lat"][m],
            "lat G / load_ratio(Tire Load)": S["lat"][m] / np.maximum(lr_tl[m], 0.2),
            "lat G / load_ratio(Susp Force)": S["lat"][m] / np.maximum(lr_sf[m], 0.2),
        }
        ft = tm._fit_curve(sa, mu_true)
        print(f"   n={m.sum()} steady samples; game mu peak {ft and ft['peak']} (observed {ft and ft['peak_observed']}) avail {ft and ft['available_grip']}")
        corner = np.abs(sa) > 1.5
        for name, g in proxies.items():
            r, rho = np.corrcoef(g, mu_true)[0, 1], spearmanr(g, mu_true)[0]
            rc = np.corrcoef(g[corner], mu_true[corner])[0, 1]
            bias = np.median(g[corner] / np.maximum(mu_true[corner], 1e-3))
            fp = tm._fit_curve(sa, g)
            print(f"   {name:34s} r={r:.3f} spearman={rho:.3f} | cornering r={rc:.3f} proxy/game={bias:.3f} | "
                  f"peak {fp and fp['peak']} (obs {fp and fp['peak_observed']}) avail {fp and fp['available_grip']}")
