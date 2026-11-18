"""Batch B step 4 check: app.ideal_state.load_reduction and its npz cache on a
real recording, and app.ideal_store end to end, read-only on the recordings.
Cache and profiles go to %TEMP%\\il_store_check (never under data/).
usage: python build_check_step4.py [file]   (default: Ferrari COTA 10/30)"""
import sys, time, shutil
import numpy as np
from build_check_common import load, FERRARI, SESS
from pathlib import Path
import os
from app import ideal_state as S, ideal_store as ST

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
name = sys.argv[1] if len(sys.argv) > 1 else FERRARI
root = Path(os.environ.get("TEMP", ".")) / "il_store_check"
shutil.rmtree(root, ignore_errors=True)
cache = root / "ideal_lap" / "cache"
path = SESS / name

t0 = time.time(); red = S.load_reduction(path, cache); t1 = time.time()
again = S.load_reduction(path, cache); t2 = time.time()
npz = list(cache.glob("*.npz"))
print("%s\n  first load %.1f s, from npz %.2f s; npz files %s (%.1f MB)" % (
    name, t1 - t0, t2 - t1, [p.name for p in npz], sum(p.stat().st_size for p in npz) / 1e6))
print("  car %s (%s), version %s, load source %s, grade %s, laps %s, forces %s, grip fraction %s, slip %s" % (
    red.car_model, red.car_class, red.game_version, red.load_source, red.grade_source,
    [s.lap_number for s in red.laps], red.has_forces, red.has_grip_fraction, red.has_slip))
same = all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True) if getattr(a, f) is not None else getattr(b, f) is None
           for a, b in zip(red.laps, again.laps) for f in S.SCALAR_KEYS + S.WHEEL_KEYS)
print("  npz round trip identical: %s; meta keys %d" % (same, len(again.meta)))
ref = load(name)                                     # the build checks' own loader (same lap_raw / lap_state)
print("  laps vs build-check loader: %s vs %s" % ([s.lap_number for s in red.laps], [s.lap_number for s in ref["states"]]))
ok = all(np.allclose(a.v, b.v) and np.allclose(a.gy, b.gy) for a, b in zip(red.laps, ref["states"]))
print("  arrays equal to the build-check loader: %s" % ok)

store = ST.IdealStore(root / "ideal_lap", [SESS])
t0 = time.time(); out = store.car_for_lap(path); dt = time.time() - t0
if isinstance(out, ST.Refusal):
    print("  store: REFUSED %s" % out.reason)
else:
    car, info = out
    print("  store: %.0f s, scope %s key %s, %d laps / %d files, version %s, beta %.2f, confidence %s, selfcheck %s, load model %s" % (
        dt, info["scope"], info["key"], info["n_laps"], info["n_files"], info["game_version"], info["beta"],
        info["confidence"], info["selfcheck"], info["load_model"]))
    for ax in ("front", "rear"):
        a = info["axles"][ax]
        print("    %s %s terms %s mu1 %.3f k %.3f c %.3f v_sat %s n_sat %d" % (ax, a["status"], a["terms"], a["mu1"], a["k"], a["c"], np.round(a["v_sat"], 1), a["n_sat"]))
    print("    notes %s" % info["notes"])
    print("    setup %s\n    setup_differs %s | trims %s" % (
        {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in info["setup"].items()},
        info["setup_differs"], {a: (info["axles"][a]["trim"], info["axles"][a]["trim_t"], round(info["axles"][a]["trim_w"], 2))
                                for a in ("front", "rear")}))
    t0 = time.time(); store2 = ST.IdealStore(root / "ideal_lap", [SESS]); store2.car_for_lap(path)
    print("  warm (new store, profile + npz): %.1f s, fits %d; profile %s" % (time.time() - t0, store2.fits_done,
          [p.name for p in (root / "ideal_lap").glob("*.json")]))
