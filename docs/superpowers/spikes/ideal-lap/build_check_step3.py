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
ap.add_argument("--clamp", action="store_true", help="round 1 (R5): the app's clamped envelope, plain unclamped")
ap.add_argument("files", nargs="*")
a = ap.parse_args()


def clamped(d):
    """R5: same-file fit, the app's solver with the saturation clamp, levels
    0.98 / 1.00 / 1.02 (as design §13 R1's table). Prints per lap: plain
    (unclamped) gains, clamped envelope gains, unclamped envelope gain at 1.00,
    clamped share, clamped ideal time; per file the spread of the ideal."""
    from app.ideal_state import lap_state
    gfit = C.fit_group([reduction(d)], d["cls"])
    car = gfit.car_for(d["file"])
    axles = [gfit.axle(0), gfit.axle(1)]
    print("%s | R5 clamped | mu1 %s k %s c %s terms %s knee %s kb %.3f kt %.3f" % (
        d["model"], np.round(car.mu1, 3), np.round(car.k, 3), np.round(car.c, 3), [x.terms for x in axles],
        [None if x.knee_deg is None else round(x.knee_deg, 2) for x in axles], car.kb, car.kt))
    ideals = []
    for r in sorted(d["raws"], key=lambda r: r["time"])[: a.laps]:
        t0 = time.time()
        sat, ax = SV.saturated_ref(lap_state(r), car, axles, per_axle=True)
        ln = SV.line_of(r, sat_ref=sat, sat_axles=ax)
        lv = SV.levels(car, ln, 0.02)
        plain = [lv["t_ref_grid"] - SV.solve(car.scaled(x), ln, clamp=False)["time"] for x in (0.98, 1.0, 1.02)]
        ideal = r["time"] - lv["gain"]
        ideals.append(ideal)
        # the probe's clamp (fable_p5): min(vc, v_ref) at clamped points, only down
        probe = []
        for x in (0.98, 1.0, 1.02):
            cx = car.scaled(x)
            vc = SV.corner_limit(cx, ln)[0]
            ln2 = dict(ln, v_clamp=np.where(ln["sat_ref"], np.minimum(ln["v"], vc), np.nan))
            probe.append(lv["t_ref_grid"] - SV.envelope(SV.solve(cx, ln2)["v"], ln["v"], ln["ds"])["time"])
        print("      probe-style clamp min(vc, v_ref): %+.2f / %+.2f / %+.2f s" % tuple(probe))
        print("  L%d %.2f | plain %+.2f / %+.2f / %+.2f s | CLAMPED envelope %+.2f / %+.2f / %+.2f s | unclamped env %+.2f s"
              " | clamped %.1f%% | ideal %.2f | %.1f s" % (
                  r["n"], r["time"], *plain, lv["gain_min"], lv["gain"], lv["gain_max"], lv["gain_unclamped"],
                  100 * lv["clamped_share"], ideal, time.time() - t0))
    print("  clamped ideal over the laps: %s, spread %.2f %%" % (
        " / ".join("%.2f" % x for x in ideals), 100 * (max(ideals) - min(ideals)) / np.mean(ideals)))


if a.clamp:
    for name in a.files or [PORSCHE, FERRARI]:
        clamped(load(name))
    sys.exit(0)

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
