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
    uvicorn.run(app.main.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main(sys.argv[1:])
