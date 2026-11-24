"""Opus probe 7 (synthetic T6, built code at 9504f1a): are the braking / traction CAPACITIES
(kb * S, kt * s_r * A_r) moved by the level quantile, and is traction low by selection?
Variants (fit_axles / fit_group patched in a module copy):
  built   : axles q 0.75, kb/kt median
  ax50    : axles median, kb/kt median           (built vs ax50 = the capacity-quantile effect)
  kq75    : axles q 0.75, kb/kt q 0.75
Selection: mean log capacity scale (the noise field) at the grid points where the TRUE solve is
bound by traction / braking, against the lap mean (0).
usage: python opus_p7_kbkt.py"""
import inspect, sys, textwrap, os
from pathlib import Path
import numpy as np
WT = Path(os.environ.get("IL_WT", r"C:\Users\hoang\Projects\lmu-coach-ideal"))
sys.path[:0] = [str(WT / "backend"), str(WT / "backend" / "tests")]
from app import ideal_car as C  # noqa: E402
from app.ideal_state import FileReduction, lap_state  # noqa: E402
from synthetic_ideal import true_car, line, perfect_laps  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def variant(qa, qm):
    ns = C.__dict__.copy(); ns["__name__"] = "v"
    fa = textwrap.dedent(inspect.getsource(C.fit_axles)).replace(
        "qx = q if q is not None else LEVEL_Q", f"qx = {qa}")
    fg = textwrap.dedent(inspect.getsource(C.fit_group)).replace(
        "qm = 0.5 if q is None else q", f"qm = {qm}")
    assert f"qx = {qa}" in fa and f"qm = {qm}" in fg
    exec(compile(fa, "fa", "exec"), ns); exec(compile(fg, "fg", "exec"), ns)
    return ns["fit_group"]


V = {"built": variant(0.75, 0.5), "ax50": variant(0.5, 0.5), "kq75": variant(0.75, 0.75)}


def bt(car, v):
    v = np.array([float(v)]); z = np.zeros(1)
    Af, Ar = car.axles(v, z, z, z)
    return float(car.kb * (car.sf * Af[0] + car.sr * Ar[0])), float(car.kt * car.sr * Ar[0])


rows = {}
for kind in ("A", "B"):
    for seed in (2026, 7, 99, 1, 2, 3, 4, 5, 6, None):
        car = true_car(kind); nz = 0.0 if seed is None else 0.035
        raws, sols, scales = perfect_laps(car, line(), 6, np.random.default_rng(seed or 0), place_sd=nz, lap_sd=nz)
        red = FileReduction(file="s", mtime=0, size=0, meta={}, track="s", car_model="S", car_class="LMGT3",
                            game_version="t", load_source="susp", grade_source="pos_y", laps=[lap_state(r) for r in raws])
        sel = {}
        for code in ("traction", "brake"):
            x = np.concatenate([np.log(sc)[np.asarray(so["binding"]) == code] for so, sc in zip(sols, scales)])
            sel[code] = float(x.mean()) if len(x) else float("nan")
        tb40, tt40 = bt(car, 40); tb25, tt25 = bt(car, 25)
        out = []
        for name, fg in V.items():
            g = fg([red], "LMGT3"); f = g.car_for("s")
            fb40, ft40 = bt(f, 40); fb25, ft25 = bt(f, 25)
            r = dict(b40=fb40 / tb40, t40=ft40 / tt40, t25=ft25 / tt25, kb=f.kb, kt=f.kt)
            rows.setdefault((kind, name, seed is None), []).append(r)
            out.append("%s brk %.3f trac40 %.3f trac25 %.3f (kb %.3f kt %.3f)" % (name, r["b40"], r["t40"], r["t25"], f.kb, f.kt))
        print("%s %4s | sel log-scale trac %+.3f brake %+.3f | %s" % (kind, seed, sel["traction"], sel["brake"], " | ".join(out)), flush=True)
print("== noisy seeds: mean / sd / min / max")
for (kind, name, clean), rr in rows.items():
    if clean: continue
    for k in ("b40", "t40", "t25"):
        x = np.array([r[k] for r in rr])
        print("  %s %-5s %-4s mean %.3f sd %.3f min %.3f max %.3f" % (kind, name, k, x.mean(), x.std(ddof=1), x.min(), x.max()))
