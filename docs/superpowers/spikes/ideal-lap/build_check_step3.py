"""Batch A step 3 check (design §8): envelope and levels of app.ideal_solver at
the median calibration, levels 0.98 / 1.00 / 1.02, against fable_p4.py (design
table F5, +-0.1 s). Same-file fit, best 3 laps.
Variants: 'app' (design fit, bisected corner limit: what the app returns),
'grid' (design fit, the spike's 0.5 m/s corner grid), and with --spike-d2 the
spike's drag order for the GT3 (parity only).
usage: python build_check_step3.py [--spike-d2 4.0e-5] [file ...]"""
import sys, argparse, time
import numpy as np
from build_check_common import load, PORSCHE, FERRARI
from app import ideal_car as C, ideal_solver as SV
from build_check_common import reduction, spike_parity_fit

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser(); ap.add_argument("--spike-d2", type=float); ap.add_argument("--laps", type=int, default=3)
ap.add_argument("files", nargs="*")
a = ap.parse_args()

for name in a.files or [PORSCHE, FERRARI]:
    d = load(name)
    gfit = C.fit_group([reduction(d)], d["cls"])
    variants = [("app", gfit.car_for(d["file"]), True), ("grid", gfit.car_for(d["file"]), False)]
    if a.spike_d2 is not None and d["cls"] == "LMGT3":
        variants.append(("spike-parity", spike_parity_fit(d, a.spike_d2, 0.5)[0], False))
    for vn, car, refine in variants:
        print("%s | %s | mu1 %s kb %.3f kt %.3f" % (d["model"], vn, np.round(car.mu1, 3), car.kb, car.kt))
        for r in sorted(d["raws"], key=lambda r: r["time"])[: a.laps]:
            ln = SV.line_of(r)
            ds, t_ref = ln["ds"], float(np.sum(ln["ds"] / ln["v"]))
            for mode in ("balanced", "ceiling"):
                out, t0 = [], time.time()
                for lev in (0.98, 1.0, 1.02):
                    sol = SV.solve(car.scaled(lev), ln, mode, refine=refine)
                    env = SV.envelope(sol["v"], ln["v"], ds)
                    out.append((t_ref - sol["time"], t_ref - env["time"], (sol["v"] < ln["v"] - 0.5 / 3.6).mean()))
                print("  L%d %.2f %-8s gain plain %+.2f / %+.2f / %+.2f s | ENVELOPE gain %+.2f / %+.2f / %+.2f s | under driven %2.0f%% | %.1f s" % (
                    r["n"], r["time"], mode, out[0][0], out[1][0], out[2][0], out[0][1], out[1][1], out[2][1], 100 * out[1][2], time.time() - t0))
