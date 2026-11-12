"""Batch A step 2 check (design §8): app.ideal_car.fit_group + app.ideal_solver
balanced / ceiling against the spike's fable_p2.py --variants both at q 0.75
(Porsche COTA L9 balanced 129.90 s, Ferrari COTA L11 115.75 s, +-0.15 s),
then at the median for good. Same-file fit, best 2 laps.
usage: python build_check_step2.py [-q 0.75] [--no-refine] [file ...]"""
import sys, time, argparse
import numpy as np
from build_check_common import load, PORSCHE, FERRARI
from app import ideal_car as C, ideal_solver as SV
from app.ideal_state import FileReduction

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ap = argparse.ArgumentParser(); ap.add_argument("-q", type=float, default=0.75)
ap.add_argument("--no-refine", action="store_true"); ap.add_argument("--spike-d2", type=float); ap.add_argument("--laps", type=int, default=2)
ap.add_argument("files", nargs="*")
a = ap.parse_args()


def reduction(d):
    return FileReduction(file=d["file"], mtime=0, size=0, meta={}, track="", car_model=d["model"],
                         car_class=d["cls"], game_version="", load_source=d["load_source"],
                         grade_source="pos_y", laps=d["states"])


def spike_parity_fit(d, d2, q):
    """The spike's order for a car without forces (parity only, not the design):
    the axle base mask with NO drag (the spike's joint drag fit came after the
    axles), then braking, traction and the solve with drag d0 0 + d2 v^2 as the
    spike's joint fit returned it."""
    red = reduction(d)
    lm = C.fit_load_model(red.laps, red.load_source)
    p = C._pool([red], {red.file: lm})
    gf = p["gf"] if red.has_grip_fraction else None
    axles = C.fit_axles(p, p["gx"] + p["grade"], p["slip"], gf, q=q)
    share = C.fit_share(p, red.has_forces, [lm.static])
    mu1, kk, cc = [x.mu1 for x in axles], [x.k for x in axles], [x.c for x in axles]
    Af, Ar = C.axle_capacity(p["lam"], p["v"], mu1, kk, cc)
    gxt = p["gx"] + p["grade"] + d2 * p["v"] ** 2
    kb = C.fit_braking(p, gxt, Af, Ar, share["value"], gf, q)
    kt = C.fit_traction(p, gxt, Ar, share["value"], q)
    eng = C.fit_engine(p, q)
    info = dict(n_laps=len(red.laps), axles={n: x.to_dict() for n, x in zip(C.AXLES, axles)}, s_front=share,
                kappa_brake=kb, kappa_trac=kt, drag=dict(source="spike-parity", d0=0.0, d2=d2))
    return C.CarModel(load=lm, mu1=mu1, k=kk, c=cc, s_front=share["value"], kb=kb["value"], kt=kt["value"], d0=0.0,
                      d2=d2, power_v=eng["power_v"], power_g=eng["power_g"], corner_drag=eng["corner_drag"]), info


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
        print("   %-5s %s n_sat %d knee %s mu1 %.3f k %.3f c %.3f ratio_sd %.3f" % (ax, x["status"], x["n_sat"], x["knee_deg"], x["mu1"], x["k"], x["c"], x["ratio_sd"]))
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
