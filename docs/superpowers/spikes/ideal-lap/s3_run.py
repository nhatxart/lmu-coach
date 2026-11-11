"""Stage 3: fit a car on file(s), solve the ideal lap on driven lines, validate.
usage: python s3_run.py [-q 0.75] [--sat knee|gf|auto] [--laps N] fitfile[,fitfile2] [simfile]"""
import sys, time, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from il_data import load_file, real_laps
from il_core import fit_car, line_of, solve, compare

ap = argparse.ArgumentParser(); ap.add_argument("-q", type=float, default=0.75); ap.add_argument("--sat", default="auto")
ap.add_argument("--laps", type=int, default=3); ap.add_argument("fit"); ap.add_argument("sim", nargs="?")
a = ap.parse_args()
fit_files = [load_file(x) for x in a.fit.split(",")]
car = fit_car(fit_files, q=a.q, sat=a.sat)
simf = load_file(a.sim) if a.sim else fit_files[0]
laps = sorted(real_laps(simf), key=lambda l: l["time"])[: a.laps]
print("[sim] on %s (%s)" % (simf["track"], simf["file"][-24:-7]))
for l in laps:
    ln = line_of(l); t0 = time.time()
    for mode in ("balanced", "ceiling"):
        sol = solve(car, ln, mode); c = compare(ln, sol)
        d = np.cumsum(ln["ds"] / ln["v"] - ln["ds"] / sol["v"]); nb = int(500 / ln["ds"])
        blocks = sorted(((i * 500, d[min(len(d) - 1, (i + 1) * nb)] - d[i * nb]) for i in range(len(d) // nb)), key=lambda b: -b[1])
        print("  L%-2d %.3f  %-8s ideal %.2f  gain %+.2f s (%.1f%%) | ideal slower than driven on %4.1f%% of lap (brake %.1f / power %.1f / other %.1f), costing %.2f s, worst %.0f km/h | top gains %s | worst %s  [%.1fs]" % (
            l["n"], l["time"], mode, c["t_ideal"], c["gain"], 100 * c["gain"] / c["t_act"], 100 * c["slow_share"], *(100 * x for x in c["slow_by"]), c["slow_time"], c["worst"],
            " ".join("%d:%.2f" % b for b in blocks[:3]), " ".join("%d:%.2f" % b for b in blocks[-2:]), time.time() - t0))
