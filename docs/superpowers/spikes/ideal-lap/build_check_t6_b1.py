"""Validation round 1, T6 diagnosis (synthetic only): B1 as built (observed q 0.75,
lower_bound median) against q 0.75 on every axle (what opus_p3 measured: its
patch replaced the one quantile call). Per car / seed: axle status, capacity
vs the true mean car, plain-model time vs the true mean car.
usage: python build_check_t6_b1.py"""
import inspect, sys, textwrap
import numpy as np
from build_check_common import BACKEND
sys.path.insert(0, str(BACKEND / "tests"))
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


def caps(car):
    v = np.array([25.0, 40.0, 55.0]); z = np.zeros(3)
    Af, Ar = car.axles(v, z, z + 1.6, z)
    return np.array([Af.mean(), Ar.mean()])


for kind in ("A", "B"):
    for seed in (2026, 7, 99):
        car = true_car(kind)
        raws, _, _ = perfect_laps(car, line(), 6, np.random.default_rng(seed), place_sd=0.035, lap_sd=0.035)
        red = FileReduction(file="s", mtime=0, size=0, meta={}, track="s", car_model="S", car_class="LMGT3",
                            game_version="t", load_source="susp", grade_source="pos_y", laps=[lap_state(r) for r in raws])
        ln = SV.line_of(raws[3])
        t_true = SV.solve(car, ln, clamp=False)["time"]
        out = []
        for name, fa in (("as built", ORIG), ("q75 all axles", ALL["fit_axles"])):
            C.fit_axles = fa
            g = C.fit_group([red], "LMGT3")
            f = g.car_for("s")
            st = [g.axle(j).status for j in (0, 1)]
            out.append("%s: %s cap F %.3f R %.3f plain vs true %+.2f %%" % (
                name, "/".join(s[:5] for s in st), *(caps(f) / caps(car)), 100 * (t_true / SV.solve(f, ln, clamp=False)["time"] - 1)))
        C.fit_axles = ORIG
        print("car %s seed %4s | %s" % (kind, seed, " || ".join(out)))
