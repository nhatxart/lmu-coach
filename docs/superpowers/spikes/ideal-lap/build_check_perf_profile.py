"""Review I1: where a warm /ideal-lap request spends its time (cProfile).
Same setup as build_check_perf.py (read-only recordings, caches in %TEMP%)."""
import cProfile, io, os, pstats, shutil, sys, time
from pathlib import Path
from urllib.parse import quote
from build_check_common import SESS, FERRARI

from fastapi.testclient import TestClient  # noqa: E402
from app import main, grip_profile, ideal_lap  # noqa: E402

root = Path(os.environ.get("TEMP", ".")) / "il_perf_check"
main.OWN_SESSIONS_DIR = SESS
main.REFERENCE_LAPS_DIR = root / "no_reference_laps"
grip_profile.PROFILES_DIR = root / "profiles"
client = TestClient(main.app)
url = f"/sessions/sessions/{quote(sys.argv[1] if len(sys.argv) > 1 else FERRARI)}/ideal-lap"
client.get(url)                                     # cold: fit, npz, in-memory reductions
ideal_lap._CACHE.clear()
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
client.get(url)
pr.disable()
print("warm request %.1f s" % (time.time() - t0))
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(40)
for line in s.getvalue().splitlines():
    if any(k in line for k in ("cumtime", "ideal_", "main.py", "telemetry.py", "session_index", "trackmap",
                               "track_topography", "duckdb")):
        print(line[:200])
