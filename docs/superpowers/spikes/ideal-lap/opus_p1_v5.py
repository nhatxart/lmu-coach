"""Opus probe 1 (validation round 1): WHERE is the plain model under the driven lap (V5)?
Runs the built code of the perf worktree (read-only on recordings, imports only).
Per test lap: V5 share at level x1.10 split by the solver's binding code, and the same
with variants of the car:
  base      : as built
  pwr_q90   : engine curve at the 0.9 quantile per speed bin (instead of the median)
  pwr_x110  : power also x1.10 (what V5 'level x1.10' was meant to give)
  grip_q75  : mu1 x exp(q75 - q50 of the saturated log ratio) per axle (level above the median)
usage: python -I opus_p1_v5.py "<fit spec>[,<fit spec>]" "<sim spec>" [odd]"""
import sys, os, io, contextlib, tempfile
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

CACHE = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "ideal_lap_cache"
fits = [R.resolve(SESS, s) for s in sys.argv[1].split(",")]
sim = R.resolve(SESS, sys.argv[2]); odd = len(sys.argv) > 3
reds = {p.name: load_reduction(p, CACHE) for p in set(fits) | {sim}}
sr = reds[sim.name]; q = sorted(sr.laps, key=lambda s: s.lap_number)
if odd:
    fit_reds, test = [R.subset(sr, q[0::2])], q[1::2]
else:
    fit_reds, test = [reds[p.name] for p in fits if p != sim], q
g = C.fit_group(fit_reds, sr.car_class)
lm = g.load_models.get(sim.name) if odd else C.fit_load_model(q, sr.load_source)
car = R.car_with(g, lm); info = R.fit_info(g, sum(len(r.laps) for r in fit_reds)); axles = [g.axle(0), g.axle(1)]

# saturated log ratios on the FIT laps (for the grip_q75 variant)
fit_states = [s for r in fit_reds for s in r.laps]
qs = []
for j, idx in enumerate((C.FRONT, C.REAR)):
    x = []
    for s in fit_states:
        gxt = s.gx + s.grade + car.drag(s.v); ay = np.abs(s.gy)
        base = (np.abs(gxt) < 0.2) & (ay > 0.45 * np.percentile(ay, 99)) & (s.v > 15) & (np.abs(s.yawacc) < 0.4)
        ax = axles[j]
        if ax.saturation == "grip_fraction" and s.gf is not None:
            sat = s.gf[list(idx)].mean(0) < C.GF_SATURATED
        else:
            sl = s.slip[list(idx)].mean(0); sat = (sl >= ax.knee_deg) & (sl <= ax.knee_deg + C.KNEE_SPAN_DEG)
        m = base & sat
        if m.any():
            x.append(np.log(ay[m] / car.axles(s.v, s.gx, s.gy, s.gz)[j][m]))
    x = np.concatenate(x)
    qs.append((np.quantile(x, 0.5), np.quantile(x, 0.75), np.quantile(x, 0.9), len(x)))
print("fit %s -> %s | %s | sat log-ratio q50/q75/q90 F %s R %s" % ([r.file[-24:-7] for r in fit_reds], sim.name[-24:-7], sr.car_model,
      np.round(qs[0][:3], 3), np.round(qs[1][:3], 3)))

# engine curve variants on the fit laps
cat = lambda k: np.concatenate([getattr(s, k) for s in fit_states])
d = {k: cat(k) for k in ("v", "gx", "gy", "grade", "thr", "tc")}
e90 = C.fit_engine(d, q=0.9)


def variant(name):
    dd = car.to_dict()
    if name == "pwr_q90":
        dd["power_v"], dd["power_g"] = e90["power_v"], e90["power_g"]
    if name == "pwr_x110":
        dd["power_g"] = [p * 1.10 for p in dd["power_g"]]
    if name == "grip_q75":
        dd["mu1"] = [m * np.exp(qs[j][1] - qs[j][0]) for j, m in enumerate(dd["mu1"])]
    return C.CarModel.from_dict(dd)


_ensure_wheel_channels()
from app.telemetry import load_telemetry_file
from app import track_topography as tt
with contextlib.redirect_stdout(io.StringIO()):
    laps = load_telemetry_file(sim, grid_hz=GRID_HZ)
keep = {s.lap_number for s in test}
tl = sorted((l for l in qualifying_laps(laps) if l.lap_number in keep), key=lambda l: l.lap_time_s)[:3]
for lap in tl:
    lap_len = tt._lap_length_m(lap); layout = None
    if "Pos Y" not in lap.channels:
        hit = tt.load_topography(lap.track_name, lap_len); layout = hit[1] if hit else None
    raw = lap_raw(lap, layout)
    out = []
    for name in ("base", "pwr_q90", "pwr_x110", "grip_q75"):
        cv = variant(name)
        res = analyse_lap(cv, info, raw, lap.lap_time_s, track_name=lap.track_name, lap_len=lap_len)
        ln = res["_line"]
        p = SV.solve(cv.scaled(1.10), ln, clamp=False)
        p1 = SV.solve(cv, ln, clamp=False)
        under = ln["v"] - p["v"]; u = under > 2 / 3.6
        b = np.asarray(p["binding"])
        split = {c: round(100 * float((u & (b == c)).mean()), 1) for c in ("front", "rear", "brake", "traction", "power") if (u & (b == c)).any()}
        out.append("%s: V5 %.1f%% (%s) worst %.0f | lvl1 under %.0f%% model %.2f | ideal %.2f unclamped %.2f" % (
            name, 100 * u.mean(), split, under.max() * 3.6, 100 * (ln["v"] - p1["v"] > 2 / 3.6).mean(),
            float(np.sum(np.diff(ln["s"]) / (0.5 * (p1["v"][1:] + p1["v"][:-1])))) if "s" in ln else float("nan"),
            res["ideal"]["time_s"], lap.lap_time_s - res["_levels"]["gain_unclamped"]))
    print("  L%d %.3f" % (lap.lap_number, lap.lap_time_s)); [print("     " + o) for o in out]
