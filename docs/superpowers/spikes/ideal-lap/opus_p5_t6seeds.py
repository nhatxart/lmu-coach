"""Opus probe 5 (synthetic T6, built code at f730f45): 10 seeds per car, B1 as built
(lower_bound median) vs q 0.75 on every axle. Per seed and axle: status, terms, v_sat, capacity
vs the true mean car at the fixed states (v 25/40/55, gy 1.6) AND inside the axle's own v_sat
(3 speeds spread over [p02, p98]); plain model vs true mean car; phantom gain.
usage: python opus_p5_t6seeds.py [n_seeds]"""
import inspect, sys, textwrap, os
from pathlib import Path
import numpy as np
WT = Path(os.environ.get("IL_WT", r"C:\Users\hoang\Projects\lmu-coach-ideal"))
sys.path[:0] = [str(WT / "backend"), str(WT / "backend" / "tests")]
from app import ideal_car as C, ideal_solver as SV  # noqa: E402
from app.ideal_state import FileReduction, lap_state  # noqa: E402
from synthetic_ideal import true_car, line, perfect_laps  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ORIG = C.fit_axles
src = textwrap.dedent(inspect.getsource(C.fit_axles))
a = 'qx = q if q is not None else (LEVEL_Q if status == "observed" else 0.5)'
assert a in src
ALL = {}
exec(compile(src.replace(a, "qx = LEVEL_Q"), "fit_axles_all", "exec"), C.__dict__.copy() | {"__name__": "x"}, ALL)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEEDS = [2026, 7, 99] + list(range(1, N - 2))


def cap_at(car, j, v):
    v = np.asarray(v, float); z = np.zeros(len(v))
    return car.axles(v, z, z + 1.6, z)[j].mean()


res = {}
for kind in ("A", "B"):
    for seed in SEEDS:
        car = true_car(kind)
        raws, _, _ = perfect_laps(car, line(), 6, np.random.default_rng(seed), place_sd=0.035, lap_sd=0.035)
        red = FileReduction(file="s", mtime=0, size=0, meta={}, track="s", car_model="S", car_class="LMGT3",
                            game_version="t", load_source="susp", grade_source="pos_y", laps=[lap_state(r) for r in raws])
        ln0 = SV.line_of(raws[3]); lap = float(np.sum(ln0["ds"] / ln0["v"]))
        t_true = SV.solve(car, ln0, clamp=False)["time"]
        for name, fa in (("built", ORIG), ("q75all", ALL["fit_axles"])):
            C.fit_axles = fa
            g = C.fit_group([red], "LMGT3"); f = g.car_for("s"); ax = [g.axle(0), g.axle(1)]
            st = lap_state(raws[3]); sat, sa = SV.saturated_ref(st, f, ax, per_axle=True)
            ln = SV.line_of(raws[3], sat_ref=sat, sat_axles=sa)
            lv = SV.levels(f, ln, 0.03)
            row = dict(plain=100 * (t_true / lv["model_only_time"] - 1), phantom=100 * lv["gain"] / lap)
            for j in (0, 1):
                vs = np.linspace(*ax[j].v_sat, 3)
                row[f"fix{j}"] = cap_at(f, j, [25, 40, 55]) / cap_at(car, j, [25, 40, 55])
                row[f"in{j}"] = cap_at(f, j, vs) / cap_at(car, j, vs)
                row[f"st{j}"] = ax[j].status[:5] + "/" + ax[j].terms[:4]
                row[f"vs{j}"] = ax[j].v_sat
            res.setdefault((kind, name), []).append(row)
            print("%s %4d %-6s | F %s fix %.3f in %.3f vsat %.0f-%.0f | R %s fix %.3f in %.3f vsat %.0f-%.0f | plain %+.2f%% phantom %.2f%%" % (
                kind, seed, name, row["st0"], row["fix0"], row["in0"], *row["vs0"], row["st1"], row["fix1"], row["in1"], *row["vs1"],
                row["plain"], row["phantom"]), flush=True)
        C.fit_axles = ORIG
print("== summary (mean, sd, min, max over seeds)")
for k, rows in res.items():
    for key in ("fix0", "in0", "fix1", "in1", "plain", "phantom"):
        x = np.array([r[key] for r in rows])
        print("  %s %-6s %-7s mean %.3f sd %.3f min %.3f max %.3f" % (k[0], k[1], key, x.mean(), x.std(ddof=1), x.min(), x.max()))
