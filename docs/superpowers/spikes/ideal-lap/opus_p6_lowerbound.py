"""Opus probe 6 (real, built code): lower_bound axles at the median (B1 as built) vs q 0.75.
Per pair: axle status, mu1, the expected gain / gain_min of the 3 test laps, the share of the
lap the rear/front corner limit binds, V5. Read-only on recordings; fit_axles patched in memory.
usage: python -I opus_p6_lowerbound.py "<fit specs>" "<sim spec>" [odd]"""
import sys, os, io, contextlib, tempfile, inspect, textwrap
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WT = Path(os.environ.get("IL_WT", r"C:\Users\hoang\Projects\lmu-coach-ideal"))
SESS = Path(r"C:\Users\hoang\Projects\lmu-coach\data\sessions")
sys.path[:0] = [str(WT / "backend"), str(WT / "backend" / "tools")]
os.environ.setdefault("TRACK_REFERENCE_DIR", r"C:\Users\hoang\Projects\lmu-coach\data\track_reference")
import ideal_lap_report as R
from app import ideal_car as C, ideal_solver as SV
from app.ideal_lap import analyse_lap
from app.ideal_state import GRID_HZ, lap_raw, load_reduction, qualifying_laps, _ensure_wheel_channels

ORIG = C.fit_axles
src = textwrap.dedent(inspect.getsource(C.fit_axles))
a = 'qx = q if q is not None else (LEVEL_Q if status == "observed" else 0.5)'
assert a in src
ALL = {}
exec(compile(src.replace(a, "qx = LEVEL_Q"), "fit_axles_all", "exec"), C.__dict__.copy() | {"__name__": "x"}, ALL)

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
print("fit %s -> %s | %s (%d fit laps)" % ([r.file[-24:-7] for r in fit_reds], sim.name[-24:-7], sr.car_model, sum(len(r.laps) for r in fit_reds)))
for name, fa in (("built", ORIG), ("q75all", ALL["fit_axles"])):
    C.fit_axles = fa
    g = C.fit_group(fit_reds, sr.car_class)
    lm = g.load_models.get(sim.name) if odd else C.fit_load_model(q, sr.load_source)
    car = R.car_with(g, lm); info = R.fit_info(g, sum(len(r.laps) for r in fit_reds))
    out = []
    for lap in tl:
        lap_len = tt._lap_length_m(lap); layout = None
        if "Pos Y" not in lap.channels:
            hit = tt.load_topography(lap.track_name, lap_len); layout = hit[1] if hit else None
        res = analyse_lap(car, info, lap_raw(lap, layout), lap.lap_time_s, track_name=lap.track_name, lap_len=lap_len)
        lv = res["_levels"]; b = res["diagnostics"]["binding_share"]
        out.append("L%d gain %.2f (min %.2f) bind F %.0f%% R %.0f%%" % (lap.lap_number, lv["gain"], lv["gain_min"], 100 * b["front"], 100 * b["rear"]))
    print("  %-6s status %s mu1 %s | %s" % (name, [g.axle(j).status for j in (0, 1)], np.round(car.mu1, 3).tolist(), " || ".join(out)))
C.fit_axles = ORIG
