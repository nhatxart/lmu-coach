"""Batch A step 2 check (design §8): app.ideal_car.fit_group + app.ideal_solver
balanced / ceiling against the spike's fable_p2.py --variants both at q 0.75
(Porsche COTA L9 balanced 129.90 s, Ferrari COTA L11 115.75 s, +-0.15 s),
then at the median for good. Same-file fit, best 2 laps.
usage: python build_check_step2.py [-q 0.75] [--no-refine] [file ...]"""
import sys, time, argparse
import numpy as np
from build_check_common import load, PORSCHE, FERRARI, reduction, spike_parity_fit
from app import ideal_car as C, ideal_solver as SV

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser(); ap.add_argument("-q", type=float, default=0.75)
ap.add_argument("--no-refine", action="store_true"); ap.add_argument("--spike-d2", type=float); ap.add_argument("--laps", type=int, default=2)
ap.add_argument("files", nargs="*")
a = ap.parse_args()


for name in a.files or [PORSCHE, FERRARI]:
    d = load(name)
    if a.spike_d2 is not None and d["cls"] == "LMGT3":
        car, i = spike_parity_fit(d, a.spike_d2, a.q)
    else:
        gfit = C.fit_group([reduction(d)], d["cls"], q=a.q)
        car, i = gfit.car_for(d["file"]), gfit.info
    print("%s | q %.2f | refine %s | %d laps | mu1 %s k %s c %s | sf %.3f (%s) | kb %.3f (%s, n %d) | kt %.3f (%s, n %d) | drag %s %.4f + %.2e v2 | cc %.4f" % (
        d["model"], a.q, not a.no_refine, i["n_laps"], np.round(car.mu1, 3), np.round(car.k, 3), np.round(car.c, 3),
        car.sf, i["s_front"]["source"], car.kb, i["kappa_brake"]["rule"], i["kappa_brake"]["n"], car.kt, i["kappa_trac"]["rule"],
        i["kappa_trac"]["n"], i["drag"]["source"], car.d0, car.d2, car.corner_drag))
    for ax in ("front", "rear"):
        x = i["axles"][ax]
        print("   %-5s %s n_sat %d knee %s mu1 %.3f k %.3f c %.3f (%s) v_sat %s ratio_sd %.3f" % (ax, x["status"], x["n_sat"], x["knee_deg"], x["mu1"], x["k"], x["c"], x.get("terms"), x.get("v_sat"), x["ratio_sd"]))
    for r in sorted(d["raws"], key=lambda r: r["time"])[: a.laps]:
        ln = SV.line_of(r)
        t_ref = float(np.sum(ln["ds"] / ln["v"]))
        for mode in ("balanced", "ceiling"):
            t0 = time.time()
            sol = SV.solve(car, ln, mode, refine=not a.no_refine)
            dt = time.time() - t0
            slow = (sol["v"] < ln["v"] - 0.5 / 3.6).mean()
            codes = {c: round(100 * float((sol["binding"] == c).mean()), 1) for c in SV.BINDING}
            print("  L%d %.2f (grid %.2f) %-8s ideal %.3f gain %+.3f s | slower than driven (>0.5 km/h) on %4.1f%%, worst %.0f km/h | %.1f s | binding %% %s" % (
                r["n"], r["time"], t_ref, mode, sol["time"], t_ref - sol["time"], 100 * slow, (ln["v"] - sol["v"]).max() * 3.6, dt, codes))
