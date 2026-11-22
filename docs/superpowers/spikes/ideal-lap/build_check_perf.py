"""Review I1: time of one /ideal-lap request on a real recording, the old
per-step solver against the tabled one. Read-only on the recordings; profiles
and caches in %TEMP%\\il_perf_check (never under data/).
'warm' = profile and npz caches present, the response cache cleared (what a
new lap of an already fitted car costs).
usage: python build_check_perf.py [file]   (default: Ferrari COTA 10/30)"""
import os, sys, time, shutil
from pathlib import Path
from urllib.parse import quote
from build_check_common import SESS, FERRARI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from fastapi.testclient import TestClient  # noqa: E402
from app import main, grip_profile, ideal_lap, ideal_solver as SV  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else FERRARI
root = Path(os.environ.get("TEMP", ".")) / "il_perf_check"
shutil.rmtree(root, ignore_errors=True)
main.OWN_SESSIONS_DIR = SESS
main.REFERENCE_LAPS_DIR = root / "no_reference_laps"
grip_profile.PROFILES_DIR = root / "profiles"
client = TestClient(main.app)
url = f"/sessions/sessions/{quote(name)}/ideal-lap"
orig = SV.solve


def run(label, fast):
    SV.solve = (lambda *a, **k: orig(*a, **{**k, "fast": fast}))
    ideal_lap._CACHE.clear()
    t0 = time.time()
    d = client.get(url).json()
    dt = time.time() - t0
    print("%-22s %6.1f s | status %s lap %s ideal %s gain %s" % (
        label, dt, d["status"], d.get("lap_number"), d["ideal"] and d["ideal"]["time_s"],
        d["ideal"] and (d["ideal"]["gain_min_s"], d["ideal"]["gain_s"], d["ideal"]["gain_max_s"])))
    return d


run("cold (fit + npz), new", True)
old = run("warm, old solver", False)
new = run("warm, tabled solver", True)
print("ideal time old %.3f new %.3f diff %+.4f s" % (old["ideal"]["time_s"], new["ideal"]["time_s"],
                                                     new["ideal"]["time_s"] - old["ideal"]["time_s"]))
