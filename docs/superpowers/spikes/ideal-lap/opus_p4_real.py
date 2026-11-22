"""Opus probe 4: decisions A1 (measured power) + B1 (axle level at the 0.75 quantile) on real pairs.
Patches in memory: ideal_solver.solve (p_off, as opus_p2_power) and ideal_car.fit_axles (level
quantile QL for OBSERVED axles; lower_bound keeps the median). Reports per variant:
  V2 (median and q75 of the held-out log ratio per axle), V3 (sd), V5, V6 (unclamped spread),
  model_only vs driven, expected gain and gain_min.
usage: python -I opus_p4_real.py "<fit specs>" "<sim spec>" [odd]"""
import sys, os, io, contextlib, tempfile, inspect, textwrap
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PERF = Path(os.environ.get("IL_PERF", r"C:\Users\hoang\Projects\lmu-coach-ideal-perf"))
SESS = Path(r"C:\Users\hoang\Projects\lmu-coach\data\sessions")
sys.path.insert(0, str(PERF / "backend")); sys.path.insert(0, str(PERF / "backend" / "tools"))
os.environ.setdefault("TRACK_REFERENCE_DIR", r"C:\Users\hoang\Projects\lmu-coach\data\track_reference")
import ideal_lap_report as R
from app import ideal_car as C, ideal_solver as SV
from app.ideal_lap import analyse_lap
from app.ideal_state import GRID_HZ, lap_raw, load_reduction, qualifying_laps, _ensure_wheel_channels

CFG = {"pw": False, "ql": 0.5}


def p_off_of(car, ln):
    v, gr, ds = ln["v"], ln["grade"], ln["ds"]
    gx = (np.roll(v, -1) ** 2 - v ** 2) / (2 * SV.G * ds)
    gy = ln["ks"] * v * v / SV.G
    full = (ln["thr"] > 98) & (ln["brk"] < 1)
    off = np.where(full, gx + gr - car.power(v, gy), 0.0)
    off = np.convolve(np.pad(off, 2, mode="wrap"), np.ones(5) / 5, mode="valid")
    return np.where(full, off, 0.0)


s = textwrap.dedent(inspect.getsource(SV.solve))
s = s.replace("        p, g = car.power(v, gy) - gr[i], t - car.drag(v) - gr[i]",
              "        p, g = car.power(v, gy) + poff[i] - gr[i], t - car.drag(v) - gr[i]")
s = s.replace("    vs = vc.copy()\n", "    vs = vc.copy()\n    poff = p_off_of(car, ln) if CFG['pw'] else np.zeros(n)\n", 1)
assert "poff[i]" in s and "p_off_of(car" in s
SV.__dict__.update(p_off_of=p_off_of, CFG=CFG); exec(compile(s, "solve_p4", "exec"), SV.__dict__)
f = textwrap.dedent(inspect.getsource(C.fit_axles))
a = "        mu = m1 * float(np.quantile(y_all[m] / (m1 * shape), q))"
assert a in f
f = f.replace(a, "        mu = m1 * float(np.quantile(y_all[m] / (m1 * shape), CFG['ql'] if status == 'observed' else 0.5))")
C.__dict__["CFG"] = CFG; exec(compile(f, "fit_axles_p4", "exec"), C.__dict__)

CACHE = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "ideal_lap_cache"
fits = [R.resolve(SESS, x) for x in sys.argv[1].split(",")]
sim = R.resolve(SESS, sys.argv[2]); odd = len(sys.argv) > 3
reds = {p.name: load_reduction(p, CACHE) for p in set(fits) | {sim}}
sr = reds[sim.name]; q = sorted(sr.laps, key=lambda x: x.lap_number)
fit_reds, test = ([R.subset(sr, q[0::2])], q[1::2]) if odd else ([reds[p.name] for p in fits if p != sim], q)
_ensure_wheel_channels()
from app.telemetry import load_telemetry_file
from app import track_topography as tt
with contextlib.redirect_stdout(io.StringIO()):
    laps = load_telemetry_file(sim, grid_hz=GRID_HZ)
keep = {x.lap_number for x in test}
tl = sorted((l for l in qualifying_laps(laps) if l.lap_number in keep), key=lambda l: l.lap_time_s)[:3]
raws = []
for lap in tl:
    lap_len = tt._lap_length_m(lap); layout = None
    if "Pos Y" not in lap.channels:
        hit = tt.load_topography(lap.track_name, lap_len); layout = hit[1] if hit else None
    raws.append((lap, lap_raw(lap, layout), lap_len))
print("fit %s -> %s | %s (%d fit laps)" % ([r.file[-24:-7] for r in fit_reds], sim.name[-24:-7], sr.car_model, sum(len(r.laps) for r in fit_reds)))


def held(car, axles, states):
    out = []
    for j, idx in enumerate((C.FRONT, C.REAR)):
        x = []
        for st in states:
            gxt = st.gx + st.grade + car.drag(st.v); ay = np.abs(st.gy)
            base = (np.abs(gxt) < 0.2) & (ay > 0.45 * np.percentile(ay, 99)) & (st.v > 15) & (np.abs(st.yawacc) < 0.4)
            ax = axles[j]
            if ax.saturation == "grip_fraction" and st.gf is not None:
                sat = st.gf[list(idx)].mean(0) < C.GF_SATURATED
            elif ax.knee_deg is not None and st.slip is not None:
                sl = st.slip[list(idx)].mean(0); sat = (sl >= ax.knee_deg) & (sl <= ax.knee_deg + C.KNEE_SPAN_DEG)
            else:
                continue
            m = base & sat
            x.append(np.log(ay[m] / car.axles(st.v, st.gx, st.gy, st.gz)[j][m]))
        x = np.concatenate(x) if x else np.zeros(0)
        out.append("%s n%d med %+.3f q75 %+.3f sd %.3f (%s)" % ("FR"[j], len(x), np.median(x), np.quantile(x, 0.75), x.std(), axles[j].status[:5]) if len(x) else "%s -" % "FR"[j])
    return " | ".join(out)


for ql, pw in ((0.5, False), (0.5, True), (0.75, True)):
    CFG["ql"], CFG["pw"] = ql, pw
    g = C.fit_group(fit_reds, sr.car_class)
    lm = g.load_models.get(sim.name) if odd else C.fit_load_model(q, sr.load_source)
    car = R.car_with(g, lm); info = R.fit_info(g, sum(len(r.laps) for r in fit_reds)); axles = [g.axle(0), g.axle(1)]
    print(" variant level q%.2f power_%s | mu1 %s | held-out %s" % (ql, "loc" if pw else "med", np.round(car.mu1, 3).tolist(), held(car, axles, test)))
    unc, rows = [], []
    for lap, raw, lap_len in raws:
        res = analyse_lap(car, info, raw, lap.lap_time_s, track_name=lap.track_name, lap_len=lap_len)
        ln, lv = res["_line"], res["_levels"]
        p = SV.solve(car.scaled(1.10), ln, clamp=False)
        u = ln["v"] - p["v"] > 2 / 3.6
        unc.append(lap.lap_time_s - lv["gain_unclamped"])
        rows.append("L%d V5 %.1f%% | model_only %+.2f%% | gain %.2f (min %.2f, max %.2f) unclamped %.2f clamped %.0f%%" % (
            lap.lap_number, 100 * u.mean(), 100 * (lap.lap_time_s - lv["model_only_time"]) / lap.lap_time_s,
            lv["gain"], lv["gain_min"], lv["gain_max"], lv["gain_unclamped"], 100 * lv["clamped_share"]))
    T = np.mean([l.lap_time_s for l, _, _ in raws])
    print("    V6 unclamped spread %.2f%% | " % (100 * (max(unc) - min(unc)) / T) + " || ".join(rows))
