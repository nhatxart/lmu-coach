"""Run the backend on a folder of .duckdb files, for the browser tests.

    python serve_for_ui.py <sessions_dir> <port> [<track_reference_dir> [<reference_laps_dir>]]

The track reference dir is set BEFORE app.main is imported: track_reference
reads TRACK_REFERENCE_DIR once at import time. Without it the repo's
data/track_reference would be used, and the tests must not depend on that.
"""

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]

# The synthetic sessions conftest writes (a parseable "<track>-<car>-<UTC>" name).
HILLY = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-10-25T10_00_00Z.duckdb"
LEGACY = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-10-24T09_00_00Z.duckdb"
# Track Edge 5.5 m: lap 1's run out to lat 5.0 puts the right tyres past it.
LIMITS = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-10-26T10_00_00Z.duckdb"
# Per-wheel Slip Angle with known values and a known steer angle (tyre slip arrows).
SLIP = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-10-27T10_00_00Z.duckdb"
# Ends 300 m into one more lap (session left mid-lap): the short last lap must not be BEST.
PARTIAL = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-10-28T10_00_00Z.duckdb"
# Lap comparison (test_compare.py). All "Synthetic Hill Ring".
CMP_A = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-01T10_00_00Z.duckdb"       # LMGT3, laps at lat 0 / 1 / -1
CMP_B = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-02T10_00_00Z.duckdb"       # LMGT3, 3 % slower
CMP_HYPER = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-03T10_00_00Z.duckdb"   # Hypercar
CMP_LAYOUT = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-04T10_00_00Z.duckdb"  # LMGT3, Lap Dist x 0.9
CMP_KEPT = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-05T10_00_00Z.duckdb"    # copy of CMP_B, reference laps
CMP_V1 = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-06T10_00_00Z.duckdb"      # LMGT3, no GPS tables
CMP_NOLD = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-07T10_00_00Z.duckdb"    # LMGT3, no Lap Dist
CMP_WAVE = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-08T10_00_00Z.duckdb"    # LMGT3, 5 % slower, speed varies
# The same ring under a CATALOG track name ("Fuji Speedway", slug "fuji"): the
# index names it by slug, the file name by "Fuji Speedway". FUJI_A's crossings
# after the first fall 0.04 s before a 10 Hz row (cross_offset_ticks -4), as
# in real files; FUJI_B's sit on rows.
FUJI_A = "Fuji_Speedway-Synthetic_Car_1-2026-11-09T10_00_00Z.duckdb"     # LMGT3, laps at lat 0 / 1 / -1
FUJI_B = "Fuji_Speedway-Synthetic_Car_1-2026-11-10T10_00_00Z.duckdb"     # LMGT3, 3 % slower
FUJI_OTHER = "Fuji_Speedway-Synthetic_Car_2-2026-11-11T10_00_00Z.duckdb"  # LMGT3, another car model, 5 % slower
# Keep laps (test_keep.py): written per test into WRITABLE tmp folders (the
# keep_server fixture), because keeping writes files and delete removes one.
KEEP_ORIG = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-12T10_00_00Z.duckdb"   # recorder file, laps 0-2
KEEP_NATIVE = "Synthetic_Hill_Ring_P_2026-11-13T10_00_00Z.duckdb"               # no RecordedBy (an LMU export)
# Ideal lap (test_ideal_lap.py, the ideal_server fixture). The ring cannot be
# FITTED (no Susp Force, one speed), so with UI_IDEAL_FAKE=1 the server's
# ideal-lap store is a fake with a known car per file (as test_ideal_endpoint);
# everything after the store (solver, corners, trace on the driven line) is real.
IDEAL_OK = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-14T10_00_00Z.duckdb"      # medium confidence
IDEAL_LOW = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-15T10_00_00Z.duckdb"     # low: car scope, setup differs, public line
IDEAL_REFUSED = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-16T10_00_00Z.duckdb"  # insufficient_data
IDEAL_SLOW = "Synthetic_Hill_Ring-Synthetic_Car_1-2026-11-17T10_00_00Z.duckdb"    # the store takes IDEAL_DELAY_S
IDEAL_DELAY_S = 4.0
IDEAL_REFUSAL = "3 clean laps on this track needed, have 1"


def install_fake_ideal_store(main_mod) -> None:
    """UI_IDEAL_FAKE=1: main._ideal_store -> a store with a known car per file."""
    import time
    sys.path.insert(0, str(BACKEND / "tests"))
    from types import SimpleNamespace
    from app import ideal_lap
    from app.ideal_store import Refusal
    from test_ideal_solver import _car

    def axle():
        return {"mu1": 1.5, "k": 0.0, "c": 0.0, "status": "observed", "saturation": "knee", "knee_deg": 4.0,
                "n_sat": 1000, "ratio_sd": 0.05, "v3_peak_slip_deg": None, "knee_vs_v3": None, "terms": "fitted",
                "v_sat": [10.0, 30.0], "trim": "none", "trim_w": 0.0}

    def info(**kw):
        d = {"scope": "track", "key": "synthetic|1200", "n_laps": 6, "n_files": 2, "load_source": "susp",
             "confidence": "medium", "beta": 0.03, "game_version": "test", "notes": [],
             "axles": {"front": axle(), "rear": axle()}, "s_front": {"value": 0.5, "source": "estimate"},
             "kappa_brake": {"value": 1.0, "status": "fitted"}, "kappa_trac": {"value": 1.0, "status": "fitted"},
             "drag": {"d0": 0.0, "d2": 0.0, "source": "prior"}, "selfcheck": None, "transferred": False,
             "fingerprint": "f-ok", "setup": {}, "setup_differs": []}
        d.update(kw)
        return d

    # PREVIOUSLY mu 5.0 / 5.2: at the ring's default 40 m/s the bends are 2.6 g,
    # so a car with ordinary grip was slower everywhere and every gain read 0.00.
    # Now the ideal files are driven at speed_scale 0.6 (24 m/s, ~0.9 g in the
    # bends, below this car's limit) and a realistic car gains ~1.6 s of 50 s
    # (probe, 11/24/2026).
    car = _car(mu=(1.5, 1.6), power=0.6)

    # Named corners for the ring, as the map shows them (track_db ref_points in
    # game Lap Dist): T1 and T2 in the first bend (400-600 m), T3 in the second
    # (1000-1200 m). The ideal-lap table must use the same names.
    from app import track_db
    track_db.TRACK_REGISTRY["Synthetic Hill Ring"] = {
        "display_name": "Synthetic Hill Ring", "aliases": [], "country": "Nowhere",
        "layouts": {"Default": {"aliases": [], "ref_points": [
            {"dist": 0, "alt": 0.0, "corner": "Start/Finish Line"},
            {"dist": 470, "alt": 0.0, "corner": "T1 Apex (North)"},
            {"dist": 540, "alt": 0.0, "corner": "T2 Apex"},
            {"dist": 1100, "alt": 0.0, "corner": "T3 Apex (Hairpin)"}]}}}

    class FakeStore:
        def car_for_lap(self, path, lap_number=None):
            name = Path(path).name
            if name == IDEAL_REFUSED:
                return Refusal(IDEAL_REFUSAL)
            if name == IDEAL_SLOW:
                time.sleep(IDEAL_DELAY_S)
                return car, info(fingerprint="f-slow")
            if name == IDEAL_LOW:
                return car, info(scope="car", transferred=True, confidence="low", beta=0.05, fingerprint="f-low",
                                 setup_differs=["camber_f", "press_r"],
                                 notes=["transferred: estimate from other tracks"])
            return car, info()

        def _reduction(self, path):
            return SimpleNamespace(car_model="Synthetic Car LMGT3", car_class="LMGT3", game_version="test")

    real_public = ideal_lap.public_block
    state = {"low": False}

    def public_block(car_model, car_class, slug, layout_len_m, game_version, lap_time_s, ideal_s, ceiling_s):
        # IDEAL_LOW gets a public line (the real one reads public_laptimes.json, absent here)
        if state["low"]:
            best = round(lap_time_s * 0.96, 3)
            return ({"car_best_s": best, "class_ref_s": best, "patch": "1.4", "patch_match": True,
                     "driven_pct": round(100 * lap_time_s / best, 2), "ideal_pct": round(100 * ideal_s / best, 2),
                     "band": "Midpack"}, "ok")
        return real_public(car_model, car_class, slug, layout_len_m, game_version, lap_time_s, ideal_s, ceiling_s)

    real_build = ideal_lap.build_ideal_lap

    def build(path, lap_number=None, store=None, points=None, source="sessions"):
        state["low"] = Path(path).name == IDEAL_LOW
        return real_build(path, lap_number, store=store, points=points, source=source)
    ideal_lap.public_block = public_block
    ideal_lap.build_ideal_lap = build
    fake = FakeStore()
    main_mod._ideal_store = lambda: fake


def main(argv: list[str]) -> None:
    """argv: sessions dir, port, [track reference dir], [reference laps dir]."""
    if len(argv) < 2:
        raise SystemExit(__doc__)
    sessions, port = Path(argv[0]).resolve(), int(argv[1])
    if len(argv) > 2:
        os.environ["TRACK_REFERENCE_DIR"] = str(Path(argv[2]).resolve())
    sys.path.insert(0, str(BACKEND))
    import uvicorn
    import app.main

    app.main.OWN_SESSIONS_DIR = sessions
    if len(argv) > 3:
        app.main.REFERENCE_LAPS_DIR = Path(argv[3]).resolve()
    if os.environ.get("UI_IDEAL_FAKE") == "1":
        install_fake_ideal_store(app.main)
    uvicorn.run(app.main.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main(sys.argv[1:])
