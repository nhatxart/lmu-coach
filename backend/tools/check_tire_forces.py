"""
Does a recording have usable tyre-force channels? (read-only)

    python -I backend/tools/check_tire_forces.py "data/sessions/<file>.duckdb"

LMU fills Tire Load / Tire Lateral Force for some cars and game versions
only (see DevNotes 10/12/2026). This prints channel rates, row counts,
value ranges, and whether summed lateral force lines up with lateral G
(lag 0 should be the best lag, r ~0.9+). Those files are the only ground
truth for validate_tire_model.py's test (a).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

CHANNELS = ("GPS Time", "Tire Load", "Tire Lateral Force", "Tire Longitudinal Force", "G Force Lat", "Slip Angle")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    con = duckdb.connect(str(Path(sys.argv[1])), read_only=True)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    rows = con.execute("SELECT channelName, frequency, unit FROM channelsList").fetchall()
    meta = {r[0]: (r[1], r[2]) for r in rows}
    for ch in CHANNELS:
        if ch not in tables:
            print(f"{ch:24s} missing")
            continue
        n = con.execute(f'SELECT count(*) FROM "{ch}"').fetchone()[0]
        freq, unit = meta.get(ch, ("?", "?"))
        print(f"{ch:24s} {n:>8} rows  {freq} Hz  unit={unit}")
    if "Tire Lateral Force" not in tables or meta.get("Tire Lateral Force", (0, ""))[1] == "unpopulated_by_lmu":
        print("\nno usable Tire Lateral Force")
        return 0

    def wheels(t):
        return np.array(con.execute(f'SELECT value1, value2, value3, value4 FROM "{t}"').fetchall(), float)

    fy, fz = wheels("Tire Lateral Force"), wheels("Tire Load")
    lat = np.array(con.execute('SELECT value FROM "G Force Lat"').fetchall(), float)[:, 0]
    n = min(len(fy), len(fz), len(lat))
    fy, fz, lat = fy[:n], fz[:n], lat[:n]
    print("\nFy p1/p50/p99 per wheel (N):\n", np.percentile(fy, [1, 50, 99], axis=0).round(0))
    print("Fz p1/p50/p99 per wheel (N):\n", np.percentile(fz, [1, 50, 99], axis=0).round(0))
    print(f"samples with any wheel Fz < 100 N: {(fz < 100).any(axis=1).mean():.1%} "
          f"(masked in validate_tire_model.py)")
    tot = fy.sum(axis=1)
    print("corr(sum Fy, lat G) by lag (samples):")
    for lag in (-50, -10, -1, 0, 1, 10, 50):
        a = tot[max(lag, 0): n + min(lag, 0)]
        b = lat[max(-lag, 0): n + min(-lag, 0)]
        print(f"  {lag:+4d}  {np.corrcoef(a, b)[0, 1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
