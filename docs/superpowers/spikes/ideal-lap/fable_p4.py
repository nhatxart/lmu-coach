"""Fable probe 4: the headline definition. Car = variant 'both' (|gy| load term, capacity on
modelled lam), calibrated at the MEDIAN (q 0.5). Grip level scaled by 0.98 / 1.00 / 1.02.
Per lap: plain model time (may be slower than driven somewhere) and ENVELOPE time
(v = max(model, driven): never slower than the reference lap), balanced and ceiling.
usage: python fable_p4.py [-q 0.5] [--laps 3] fitfile[,fitfile2] [simfile]"""
import sys, types, argparse, copy
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps

s_ = (Path(__file__).parent / "il_core.py").read_text(encoding="utf-8")
for a, b in (("return np.stack([np.ones_like(v), v * v, gx, gy, gz], axis=-1)", "return np.stack([np.ones_like(v), v * v, gx, gy, gz, np.abs(gy)], axis=-1)"),
             ("lam = L / c0[:, None]", "lam = np.maximum((C / c0[:, None]) @ X.T, 0.02)")):
    assert a in s_; s_ = s_.replace(a, b)
M = types.ModuleType("il_core_both"); exec(compile(s_, "il_core_both", "exec"), M.__dict__)

ap = argparse.ArgumentParser(); ap.add_argument("-q", type=float, default=0.5); ap.add_argument("--laps", type=int, default=3)
ap.add_argument("fit"); ap.add_argument("sim", nargs="?")
a = ap.parse_args()
fit_files = [load_file(x) for x in a.fit.split(",")]
simf = load_file(a.sim) if a.sim else fit_files[0]
car = M.fit_car(fit_files, q=a.q, verbose=False)
laps = sorted(real_laps(simf), key=lambda l: l["time"])[: a.laps]
print("%s | fit on %s | sim on %s %s | q %.2f | mu1 %s kb %.3f kt %.3f" % (car.p["model"], [f["file"][-24:-7] for f in fit_files], simf["track"], simf["file"][-24:-7], a.q,
                                                                    np.round(car.p["mu1"], 3), car.kb, car.kt))
for l in laps:
    ln = M.line_of(l); ds = ln["ds"]; t_ref = float(np.sum(ds / ln["v"]))
    print("  L%d driven %.2f" % (l["n"], l["time"]))
    for mode in ("balanced", "ceiling"):
        out = []
        for lev in (0.98, 1.0, 1.02):
            p = copy.deepcopy(car.p); p["mu1"] = [m * lev for m in p["mu1"]]
            sol = M.solve(M.Car(p), ln, mode)
            v_env = np.maximum(sol["v"], ln["v"])
            slow = (sol["v"] < ln["v"] - 0.5 / 3.6).mean()
            out.append((t_ref - sol["time"], t_ref - float(np.sum(ds / v_env)), slow, (ln["v"] - sol["v"]).max() * 3.6))
        print("      %-8s gain plain %+.2f / %+.2f / %+.2f s | ENVELOPE gain %+.2f / %+.2f / %+.2f s (level 0.98 / 1.00 / 1.02) | at 1.00: model under driven on %.0f%% of lap, worst %.0f km/h" % (
            mode, out[0][0], out[1][0], out[2][0], out[0][1], out[1][1], out[2][1], 100 * out[1][2], out[1][3]))
