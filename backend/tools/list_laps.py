"""
List laps per session file, as the backend sees them (read-only).

    python -I backend/tools/list_laps.py                 # every file
    python -I backend/tools/list_laps.py "*2026-10-*"    # glob on the file name

Marks: * = unfinished last lap (no closing Lap event), j = shorter than 85 %
of the file's median (mid-lap join), R = the reference lap
(telemetry.reference_lap_time) that /analyze and the tyre model use.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.telemetry import load_telemetry_file, reference_lap_time  # noqa: E402


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    files = sorted((REPO / "data" / "sessions").glob(f"{pattern}.duckdb" if not pattern.endswith(".duckdb") else pattern))
    if not files:
        print("no matching files")
        return 1
    for p in files:
        try:
            laps = [l for l in load_telemetry_file(p) if l.lap_time_s < 600]
        except Exception as e:
            print(f"{p.name}\n  LOAD FAILED: {e}")
            continue
        ref = reference_lap_time(laps)
        complete = sorted(l.lap_time_s for l in laps if l.is_complete)
        med = complete[len(complete) // 2] if complete else 0
        cells = []
        for l in laps:
            mark = "*" if not l.is_complete else ("j" if l.lap_time_s < 0.85 * med else "")
            if l.is_complete and ref is not None and l.lap_time_s == ref:
                mark += "R"
            m, s = divmod(l.lap_time_s, 60)
            cells.append(f"L{l.lap_number} {int(m)}:{s:06.3f}{mark}")
        car = laps[0].car_name if laps else "?"
        print(f"{p.name}\n  {car} | {len(laps)} laps | " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
