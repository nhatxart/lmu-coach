"""Builds small recorder-style .duckdb files for tests.

The game clock can skip ticks (the 10/14/2026 Fuji HF file lost ~18 s of
ticks in its first lap). Then row N of a 100 Hz table is no longer at
N / 100 s. To make any misalignment visible, every channel here stores its
own sample time as the value ("time tags"), so a correctly aligned reader
returns value == time at every point.
"""

from pathlib import Path

import duckdb

SLOW_PERIOD_S = 0.1
GPS_REF_LAT, GPS_REF_LON = 35.0, 139.0
LON_PER_S = 1e-5   # the car drives east, one step of longitude per second


def tick_times(start=10.0, end=130.0, drop_until=40.0):
    """Game clock: 20 ms steps (every other tick missing) until drop_until,
    then clean 10 ms steps. Integer arithmetic so times are exact."""
    out, ms = [], int(round(start * 1000))
    while ms <= int(round(end * 1000)):
        out.append(ms / 1000.0)
        ms += 20 if ms < int(round(drop_until * 1000)) else 10
    return out


def slow_rows(times):
    """recorder_hf.py's 10 Hz gate, replayed: index of each 10 Hz row."""
    nxt, idx = times[0], []
    for i, et in enumerate(times):
        if et >= nxt:
            nxt += SLOW_PERIOD_S
            if nxt <= et:
                nxt = et + SLOW_PERIOD_S
            idx.append(i)
    return idx


def write_session(path: Path, laps=((10.0, 0), (50.0, 1), (90.0, 2)),
                  lap_times=((50.0, 39.9937), (90.0, 40.0)),
                  recorded_by="lmu-coach-recorder hf1", times=None) -> Path:
    times = tick_times() if times is None else times
    slow_t = [times[i] for i in slow_rows(times)]
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE metadata (key VARCHAR, value VARCHAR)")
    con.executemany("INSERT INTO metadata VALUES (?, ?)", [
        ("TrackName", "Test Track"), ("CarName", "Test Car #1"),
        ("DriverName", "Tester"), ("RecordedBy", recorded_by),
    ])
    con.execute("CREATE TABLE channelsList (channelName VARCHAR, frequency DOUBLE, unit VARCHAR)")

    def scalar(name, values, hz):
        # CSV bulk load, like the recorder: executemany and list parameters
        # both cost seconds per channel in DuckDB.
        csv = path.with_name(path.name + ".csv")
        csv.write_text("value\n" + "\n".join(repr(float(v)) for v in values))
        con.execute(f'CREATE TABLE "{name}" AS SELECT value::DOUBLE AS value '
                    f"FROM read_csv('{csv.as_posix()}', header=true)")
        csv.unlink()
        con.execute("INSERT INTO channelsList VALUES (?, ?, '')", (name, hz))

    span = times[-1] - times[0]
    fast_hz = (len(times) - 1) / span          # measured, as the recorder writes it
    slow_hz = (len(slow_t) - 1) / span
    scalar("GPS Time", times, fast_hz)
    for ch in ("Ground Speed", "Brake Pos", "Throttle Pos", "G Force Lat",
               "G Force Long", "Steering Pos", "Yaw Rate"):
        scalar(ch, times, fast_hz)
    scalar("GPS Latitude", [GPS_REF_LAT] * len(slow_t), slow_hz)
    scalar("GPS Longitude", [GPS_REF_LON + t * LON_PER_S for t in slow_t], slow_hz)
    scalar("Path Lateral", slow_t, slow_hz)

    con.execute('CREATE TABLE "Lap" (ts DOUBLE, value INTEGER)')
    con.executemany('INSERT INTO "Lap" VALUES (?, ?)', list(laps))
    con.execute('CREATE TABLE "Lap Time" (ts DOUBLE, value DOUBLE)')
    con.executemany('INSERT INTO "Lap Time" VALUES (?, ?)', list(lap_times))
    con.close()
    return path
