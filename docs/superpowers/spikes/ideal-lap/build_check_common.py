"""Shared loader for the build checks of the ideal-lap app modules (Batch A).
Runs the NEW app modules (backend/app/ideal_*.py of this worktree) on real
recordings, read-only. Raw lap arrays are cached in %TEMP%\\il_build_cache
(pickle, keyed by file name, mtime, size), never under data/."""
import os, sys, io, pickle, hashlib, contextlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]                                  # worktree root
BACKEND = WT / "backend"
SESS = Path(os.environ.get("IL_SESSIONS", r"C:\Users\hoang\Projects\lmu-coach\data\sessions"))
REF = os.environ.get("IL_REF", r"C:\Users\hoang\Projects\lmu-coach\data\track_reference")
CACHE = Path(os.environ.get("TEMP", ".")) / "il_build_cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("TRACK_REFERENCE_DIR", REF)
sys.path.insert(0, str(BACKEND))

from app import ideal_state as S  # noqa: E402

PORSCHE = "Circuit of the Americas-The Bend Manthey 2026 _92_LM-2026-11-09T04_29_26Z.duckdb"
FERRARI = "Circuit of the Americas-Ferrari AF Corse 2026 _50_WEC-2026-10-30T03_57_04Z.duckdb"


def load(name):
    """{file, model, cls, raws (qualifying laps, lap_raw dicts), states, load_source}"""
    path = SESS / name
    st = path.stat()
    key = hashlib.sha1(f"{name}|{st.st_mtime}|{st.st_size}|1".encode()).hexdigest()[:16]
    cp = CACHE / f"{key}.pkl"
    if cp.exists():
        d = pickle.loads(cp.read_bytes())
    else:
        S._ensure_wheel_channels()
        from app.telemetry import load_telemetry_file
        from app import track_topography as tt
        from app.car_identity import identify_car
        with contextlib.redirect_stdout(io.StringIO()):
            laps = load_telemetry_file(path, grid_hz=S.GRID_HZ)
        q = S.qualifying_laps(laps, tt._read_pit_laps(path))
        ident = identify_car(dict(laps[0].meta), laps[0].car_name)
        d = dict(file=name, model=ident.model, cls=ident.car_class, raws=[S.lap_raw(l) for l in q])
        cp.write_bytes(pickle.dumps(d))
    d["states"] = [S.lap_state(r) for r in d["raws"]]
    d["load_source"] = S.file_load_source(d["states"])
    return d
