"""Opus probe 2: 'measured power' on the reference lap's own line.
Where the reference lap was flat out (thr > 98 %, brake < 1 %), the engine curve is offset per
grid point so that the model reproduces the DRIVEN acceleration at the driven speed:
    p_off(s) = (gx_ref + grade)(s) - power(v_ref(s), gy_ref(s))      (smoothed 5 points = 10 m)
    power(s, v) = curve(v) - cc gy^2 + p_off(s);   p_off = 0 where the lap was not flat out.
(The analogue of the saturation clamp for the engine: gear, draft, ERS and wind are measured.)
SV.solve is patched in memory only (the built code is untouched).
Prints per test lap: V5 (x1.10, unclamped) with its binding split, model_only, ideal, unclamped
envelope, for base and power_loc; then V6 (spread of the unclamped envelope over the laps).
usage: python -I opus_p2_power.py "<fit specs>" "<sim spec>" [odd]"""
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

FLAG = {"on": False}


def p_off_of(car, ln):
    v, gr, ds = ln["v"], ln["grade"], ln["ds"]
    gx = (np.roll(v, -1) ** 2 - v ** 2) / (2 * SV.G * ds)
    gy = ln["ks"] * v * v / SV.G
    full = (ln["thr"] > 98) & (ln["brk"] < 1)
    off = np.where(full, gx + gr - car.power(v, gy), 0.0)
    k = np.ones(5) / 5
    off = np.convolve(np.pad(off, 2, mode="wrap"), k, mode="valid")
    return np.where(full, off, 0.0)


src = textwrap.dedent(inspect.getsource(SV.solve))
a = "        p, g = car.power(v, gy) - gr[i], t - car.drag(v) - gr[i]"
assert a in src
src = src.replace(a, "        p, g = car.power(v, gy) + poff[i] - gr[i], t - car.drag(v) - gr[i]")
b = "    vs = vc.copy()\n"
assert b in src
src = src.replace(b, b + "    poff = p_off_of(car, ln) if FLAG['on'] else np.zeros(n)\n", 1)
ns = SV.__dict__; ns["p_off_of"] = p_off_of; ns["FLAG"] = FLAG
exec(compile(src, "solve_patched", "exec"), ns)            # replaces SV.solve (levels() calls it by name)

CACHE = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "ideal_lap_cache"
fits = [R.resolve(SESS, s) for s in sys.argv[1].split(",")]
sim = R.resolve(SESS, sys.argv[2]); odd = len(sys.argv) > 3
reds = {p.name: load_reduction(p, CACHE) for p in set(fits) | {sim}}
sr = reds[sim.name]; q = sorted(sr.laps, key=lambda s: s.lap_number)
fit_reds, test = ([R.subset(sr, q[0::2])], q[1::2]) if odd else ([reds[p.name] for p in fits if p != sim], q)
g = C.fit_group(fit_reds, sr.car_class)
lm = g.load_models.get(sim.name) if odd else C.fit_load_model(q, sr.load_source)
car = R.car_with(g, lm); info = R.fit_info(g, sum(len(r.laps) for r in fit_reds))
print("fit %s -> %s | %s (%d fit laps)" % ([r.file[-24:-7] for r in fit_reds], sim.name[-24:-7], sr.car_model, sum(len(r.laps) for r in fit_reds)))
_ensure_wheel_channels()
from app.telemetry import load_telemetry_file
from app import track_topography as tt
with contextlib.redirect_stdout(io.StringIO()):
    laps = load_telemetry_file(sim, grid_hz=GRID_HZ)
keep = {s.lap_number for s in test}
tl = sorted((l for l in qualifying_laps(laps) if l.lap_number in keep), key=lambda l: l.lap_time_s)[:3]
unc = {False: [], True: []}
for lap in tl:
    lap_len = tt._lap_length_m(lap); layout = None
    if "Pos Y" not in lap.channels:
        hit = tt.load_topography(lap.track_name, lap_len); layout = hit[1] if hit else None
    raw = lap_raw(lap, layout)
    print("  L%d %.3f" % (lap.lap_number, lap.lap_time_s))
    for on in (False, True):
        FLAG["on"] = on
        res = analyse_lap(car, info, raw, lap.lap_time_s, track_name=lap.track_name, lap_len=lap_len)
        ln = res["_line"]; lv = res["_levels"]
        p = SV.solve(car.scaled(1.10), ln, clamp=False)
        u = ln["v"] - p["v"] > 2 / 3.6; b = np.asarray(p["binding"])
        split = {c: round(100 * float((u & (b == c)).mean()), 1) for c in ("front", "rear", "brake", "traction", "power") if (u & (b == c)).any()}
        ut = lap.lap_time_s - lv["gain_unclamped"]; unc[on].append(ut)
        full = float(np.mean((ln["thr"] > 98) & (ln["brk"] < 1)))
        print("     %-9s V5 %4.1f%% %s worst %.0f | model_only %.2f under %.0f%% | ideal %.2f (%.2f/%.2f/%.2f) unclamped %.2f | full-throttle %.0f%%" % (
            "power_loc" if on else "base", 100 * u.mean(), split, (ln["v"] - p["v"]).max() * 3.6, lv["model_only_time"],
            100 * lv["under_driven_share"], res["ideal"]["time_s"], lv["gain_min"], lv["gain"], lv["gain_max"], ut, 100 * full))
for on in (False, True):
    x = np.array(unc[on]); T = np.array([l.lap_time_s for l in tl])
    print("  V6 %-9s unclamped spread %.2f%%" % ("power_loc" if on else "base", 100 * (x.max() - x.min()) / T.mean()))
