"""Fable probe 2: does a CONSISTENT load model fix balanced mode?
Variants of il_core (source patched in memory, il_core.py itself untouched):
  base      : il_core as is (capacity fitted on MEASURED lam, solved with MODELLED lam)
  mlam      : capacity fitted on the MODELLED lam (same quantity the solver uses)
  absgy     : load model gets a |gy| term per wheel (axle mean load moves in cornering), capacity on measured lam
  both      : |gy| term and capacity fitted on modelled lam
usage: python fable_p2.py [-q 0.75] [--laps 2] [--variants base,mlam,absgy,both] fitfile[,fitfile2] [simfile]"""
import sys, argparse, types
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps

SRC = (Path(__file__).parent / "il_core.py").read_text(encoding="utf-8")


def variant(name):
    s = SRC
    if name in ("absgy", "both"):
        a = "return np.stack([np.ones_like(v), v * v, gx, gy, gz], axis=-1)"
        assert a in s
        s = s.replace(a, "return np.stack([np.ones_like(v), v * v, gx, gy, gz, np.abs(gy)], axis=-1)")
    if name in ("mlam", "both"):
        a = "lam = L / c0[:, None]"
        assert a in s
        s = s.replace(a, "lam = np.maximum((C / c0[:, None]) @ X.T, 0.02)")
    m = types.ModuleType("il_core_" + name)
    exec(compile(s, "il_core_" + name, "exec"), m.__dict__)
    return m


ap = argparse.ArgumentParser(); ap.add_argument("-q", type=float, default=0.75); ap.add_argument("--laps", type=int, default=2)
ap.add_argument("--variants", default="base,mlam,absgy,both"); ap.add_argument("fit"); ap.add_argument("sim", nargs="?")
a = ap.parse_args()
fit_files = [load_file(x) for x in a.fit.split(",")]
simf = load_file(a.sim) if a.sim else fit_files[0]
laps = sorted(real_laps(simf), key=lambda l: l["time"])[: a.laps]
print("fit on %s | sim on %s %s | q %.2f" % ([f["file"][-24:-7] for f in fit_files], simf["track"], simf["file"][-24:-7], a.q))
for name in a.variants.split(","):
    M = variant(name)
    print("=" * 30, name)
    car = M.fit_car(fit_files, q=a.q)
    for l in laps:
        st = M.lap_state(l, car.p["load_source"])
        v, gx, gy, gz = st["v"], st["gx"], st["gy"], st["gz"]
        Af, Ar = car.axles(v, gx, gy, gz)
        hard = np.abs(gy) > 0.6 * np.percentile(np.abs(gy), 99)
        uF, uR = np.abs(gy)[hard] / Af[hard], np.abs(gy)[hard] / Ar[hard]
        um = np.maximum(uF, uR)
        print("  L%d %.2f usage(hard) front p50/p90/p97 %s >1: %2.0f%% | rear %s >1: %2.0f%% | min-axle p97 %.3f p99.5 %.3f | rear is the limiting axle on %.0f%%" % (
            l["n"], l["time"], np.round(np.percentile(uF, [50, 90, 97]), 3), 100 * (uF > 1).mean(), np.round(np.percentile(uR, [50, 90, 97]), 3), 100 * (uR > 1).mean(),
            np.percentile(um, 97), np.percentile(um, 99.5), 100 * (uR > uF).mean()))
        ln = M.line_of(l)
        for mode in ("balanced", "ceiling"):
            c = M.compare(ln, M.solve(car, ln, mode))
            c2 = M.compare(ln, M.solve(car, ln, mode), tol=2 / 3.6)
            print("      %-8s ideal %.2f gain %+.2f s | slower than driven (>0.5 km/h) on %4.1f%% (brake %.1f / power %.1f / other %.1f), (>2 km/h) on %4.1f%%, worst %.0f km/h" % (
                mode, c["t_ideal"], c["gain"], 100 * c["slow_share"], *(100 * x for x in c["slow_by"]), 100 * c2["slow_share"], c["worst"]))
