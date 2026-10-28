"""Run the backend on a folder of .duckdb files, for the browser tests.

    python serve_for_ui.py <sessions_dir> <port> [<track_reference_dir>]

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


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    sessions, port = Path(argv[0]).resolve(), int(argv[1])
    if len(argv) > 2:
        os.environ["TRACK_REFERENCE_DIR"] = str(Path(argv[2]).resolve())
    sys.path.insert(0, str(BACKEND))
    import uvicorn
    import app.main

    app.main.OWN_SESSIONS_DIR = sessions
    uvicorn.run(app.main.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main(sys.argv[1:])
