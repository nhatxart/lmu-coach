"""A synthetic hilly, banked ring written as a recorder-HF1-shaped .duckdb.

Test fixture for the chase view (track_topography, the per-track build and
/trackmap). The ground is known exactly - `truth(s, lat)` - so a fit can be
scored against it instead of against another estimate.

Track: a stadium loop, two 400 m straights joined by semicircles of radius
200/pi m, 1200 m long, driven counter-clockwise in world x/z: every corner is
a left-hander (the loop centre lies on the left of travel, left = (-dz, dx)).
Height 8 sin(2 pi s / L); the second semicircle is banked 0.08 m/m rising to
the RIGHT of travel, which is the outside of that left turn.

Laps: the file starts AT the first crossing (no out-lap) and ends at the
crossing that closes the last lap, so load_telemetry_file returns exactly
len(lap_offsets) complete laps (the final Lap row has no data and is dropped).
Each lap drives at its `lap_offsets[i]` (lat, right-positive) and blends into
the next lap's offset over its last 60 m. The lap at offset 1.5 m drifts out
to lat 5.0 between s = 600 and 640 m (60 m cosine ramps either side) so its
right wheels (lat 5.8) run over the kerb (Surface Type 5 beyond lat 5.5).

Crossings sit on 10 Hz rows of the recorder's gate with Lap Dist exactly 0,
so telemetry's per-lap re-zero of Lap Dist is exact (it subtracts the first
10 Hz value after the wrap). Speed along the centre path is ~40 m/s; each lap
is stretched slightly so its crossings land on those 10 Hz ticks.

The car sits on the surface: its up axis is the surface normal, forward is
the direction of motion in the surface, so the ground under each wheel (the
reference point + fwd*(+-1.45) + left*(+-0.80), measured vertically down by
Ride Height and the 0.25 m CG allowance) is the surface, to first order.
Orientation is written rF2 style: local +x left, +y up, +z rear,
world = M @ local, table "Ori R<k><X|Y|Z>" = row k of M.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np

from synthetic_session import slow_rows

RING_LENGTH_M = 1200.0
STRAIGHT_M = 400.0
RADIUS_M = 200.0 / math.pi       # semicircle of exactly 200 m
HILL_M = 8.0
BANK = 0.08
BANK_RAMP_M = 20.0
CG_M, RIDE_M = 0.25, 0.05
HALF_TRACK_M = 0.80
KERB_LAT_M = 5.5
TRACK_EDGE_M = 6.0
SPEED_MS = 40.0
GPS_REF_LAT, GPS_REF_LON, R_DEG = 35.37, 138.92, 111320.0
T0_S = 10.0
_SEG_B = STRAIGHT_M                         # first semicircle starts
_SEG_C = STRAIGHT_M + math.pi * RADIUS_M    # back straight starts (600)
_SEG_D = 2 * STRAIGHT_M + math.pi * RADIUS_M  # second semicircle (1000)
L = RING_LENGTH_M


def true_height(s):
    return HILL_M * np.sin(2 * np.pi * np.asarray(s, dtype=float) / L)


def true_bank(s):
    s = np.mod(np.asarray(s, dtype=float), L)
    up = np.clip((s - _SEG_D) / BANK_RAMP_M, 0.0, 1.0)
    down = np.clip((L - s) / BANK_RAMP_M, 0.0, 1.0)
    return np.where(s >= _SEG_D, BANK * np.minimum(up, down), 0.0)


def truth(s, lat):
    """Ground height at lap distance s, lateral offset lat (right-positive)."""
    v = true_height(s) + true_bank(s) * np.asarray(lat, dtype=float)
    return float(v) if np.ndim(v) == 0 else v


def _centre(s):
    """Centre path position (x, z) and unit travel direction (dx, dz)."""
    s = np.mod(np.asarray(s, dtype=float), L)
    x, z = np.empty_like(s), np.empty_like(s)
    dx, dz = np.empty_like(s), np.empty_like(s)
    a = s < _SEG_B
    x[a], z[a], dx[a], dz[a] = s[a], 0.0, 1.0, 0.0
    b = (s >= _SEG_B) & (s < _SEG_C)
    th = -np.pi / 2 + (s[b] - _SEG_B) / RADIUS_M
    x[b], z[b] = STRAIGHT_M + RADIUS_M * np.cos(th), RADIUS_M + RADIUS_M * np.sin(th)
    dx[b], dz[b] = -np.sin(th), np.cos(th)
    c = (s >= _SEG_C) & (s < _SEG_D)
    x[c], z[c], dx[c], dz[c] = STRAIGHT_M - (s[c] - _SEG_C), 2 * RADIUS_M, -1.0, 0.0
    d = s >= _SEG_D
    th = np.pi / 2 + (s[d] - _SEG_D) / RADIUS_M
    x[d], z[d] = RADIUS_M * np.cos(th), RADIUS_M + RADIUS_M * np.sin(th)
    dx[d], dz[d] = -np.sin(th), np.cos(th)
    return x, z, dx, dz


def world(s, lat):
    """(n,3) world point (x, y, z) of the ground at (s, lat)."""
    s = np.asarray(s, dtype=float)
    lat = np.broadcast_to(np.asarray(lat, dtype=float), s.shape)
    x, z, dx, dz = _centre(s)
    # right of travel = -(left) = (dz, -dx)
    return np.stack([x + lat * dz, truth(s, lat) * np.ones_like(s), z - lat * dx], axis=1)


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def _lap_lat(offsets, i, s):
    """Lateral offset of lap i at lap distance s (0..L)."""
    off = offsets[i]
    lat = np.full_like(s, off)
    if off == 1.5:
        bump = np.where(s < 600.0, _smooth((s - 540.0) / 60.0),
                        1.0 - _smooth((s - 640.0) / 60.0))
        lat = lat + (5.0 - off) * bump
    if i + 1 < len(offsets):
        lat = lat + (offsets[i + 1] - off) * _smooth((s - (L - 60.0)) / 60.0)
    return lat


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def write_hilly_session(path: Path, lap_offsets=(0.0, 1.5, -1.5), with_pose=True,
                        with_surface=True, other_cars=0,
                        track_name="Synthetic Hill Ring") -> Path:
    path = Path(path)
    offsets = tuple(float(o) for o in lap_offsets)
    n_laps = len(offsets)
    ticks_per_lap = int(round(L / SPEED_MS * 100))
    ms0 = int(round(T0_S * 1000))
    all_t = [(ms0 + 10 * k) / 1000.0 for k in range((n_laps + 1) * ticks_per_lap)]
    gate = slow_rows(all_t)
    gate_arr = np.asarray(gate)
    # crossings on gate ticks, so Lap Dist is exactly 0 there
    cross = [int(gate_arr[np.argmin(np.abs(gate_arr - i * ticks_per_lap))])
             for i in range(n_laps + 1)]
    n = cross[-1] + 1
    t = np.asarray(all_t[:n])
    slow = [i for i in gate if i < n]

    # lap distance, lateral offset and their rates per tick
    s_lap = np.zeros(n)
    lat = np.zeros(n)
    dlat_ds = np.zeros(n)
    sdot = np.zeros(n)
    lap_idx = np.zeros(n, dtype=int)
    e = 1e-3
    for i in range(n_laps):
        k = np.arange(cross[i], cross[i + 1] + (1 if i == n_laps - 1 else 0))
        span = cross[i + 1] - cross[i]
        sl = L * (k - cross[i]) / span
        if i == n_laps - 1:
            sl[-1] = L   # closing crossing: geometry at L, Lap Dist written as 0
        s_lap[k], lap_idx[k], sdot[k] = sl, i, L / (span * 0.01)
        lat[k] = _lap_lat(offsets, i, sl)
        dlat_ds[k] = (_lap_lat(offsets, i, sl + e) - _lap_lat(offsets, i, sl - e)) / (2 * e)
    lap_dist = np.where(s_lap >= L, 0.0, s_lap)

    pos = world(s_lap, lat)
    t_s = (world(s_lap + e, lat) - world(s_lap - e, lat)) / (2 * e)
    t_l = (world(s_lap, lat + e) - world(s_lap, lat - e)) / (2 * e)
    vel_dir = t_s + t_l * dlat_ds[:, None]
    fwd = _unit(vel_dir)
    up = _unit(np.cross(t_s, t_l))
    left = np.cross(fwd, up)
    M = np.stack([left, up, -fwd], axis=2)          # columns: left, up, rear
    speed = np.linalg.norm(vel_dir, axis=1) * sdot
    heading = np.unwrap(np.arctan2(fwd[:, 2], fwd[:, 0]))
    yaw_rate = np.gradient(heading, t)

    wheel_lat = np.stack([lat - HALF_TRACK_M, lat + HALF_TRACK_M,
                          lat - HALF_TRACK_M, lat + HALF_TRACK_M], axis=1)
    surface = np.where(wheel_lat > KERB_LAT_M, 5.0, 0.0)

    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE metadata (key VARCHAR, value VARCHAR)")
    meta = [("TrackName", track_name), ("CarName", "Synthetic Car #1"),
            ("DriverName", "Tester"), ("SessionType", "Practice"),
            ("RecordedBy", "lmu-coach-recorder hf1")]
    if with_pose:
        meta.append(("PoseChannels", "1"))
    con.executemany("INSERT INTO metadata VALUES (?, ?)", meta)
    con.execute("CREATE TABLE channelsList (channelName VARCHAR, frequency DOUBLE, unit VARCHAR)")
    con.execute("CREATE TABLE eventsList (eventName VARCHAR, unit VARCHAR)")
    span_s = t[-1] - t[0]
    hz = (n - 1) / span_s
    shz = (len(slow) - 1) / span_s
    csv = path.with_name(path.name + ".csv")

    def table(name, cols_sql, cols, freq=None, unit=""):
        # CSV bulk load, like the recorder (see synthetic_session.write_session)
        rows = zip(*[np.asarray(c).tolist() for c in cols])
        # integral values as "3", not "3.0": DuckDB's CSV reader will not
        # cast "3.0" into the INTEGER columns of Other Cars
        csv.write_text("\n".join(",".join(str(int(v)) if float(v).is_integer()
                                          else repr(float(v)) for v in r)
                                 for r in rows) + "\n")
        con.execute(f'CREATE TABLE "{name}" ({cols_sql})')
        con.execute(f"COPY \"{name}\" FROM '{csv.as_posix()}' (FORMAT csv, HEADER false)")
        csv.unlink()
        if freq is not None:
            con.execute("INSERT INTO channelsList VALUES (?, ?, ?)", (name, freq, unit))

    def scalar(name, values, freq, unit=""):
        table(name, "value DOUBLE", [values], freq, unit)

    def wheel(name, arr, freq, unit=""):
        table(name, ", ".join(f"value{i} DOUBLE" for i in range(1, 5)),
              [arr[:, i] for i in range(4)], freq, unit)

    scalar("GPS Time", t, hz, "s")
    scalar("Pos X", pos[:, 0], hz, "m")
    if with_pose:
        scalar("Pos Y", pos[:, 1] + CG_M + RIDE_M, hz, "m")
    scalar("Pos Z", pos[:, 2], hz, "m")
    scalar("Ground Speed", speed * 3.6, hz, "km/h")
    scalar("Yaw Rate", yaw_rate, hz, "rad/s")
    scalar("Throttle Pos", np.full(n, 100.0), hz, "%")
    scalar("Brake Pos", np.zeros(n), hz, "%")
    scalar("Steering Pos", np.clip(yaw_rate * 20.0, -100, 100), hz, "%")
    scalar("G Force Lat", speed * yaw_rate / 9.81, hz, "G")
    scalar("G Force Long", np.zeros(n), hz, "G")
    if with_pose:
        for r in range(3):
            for c, a in enumerate("XYZ"):
                scalar(f"Ori R{r}{a}", M[:, r, c], hz)
        wheel("Ride Height", np.full((n, 4), RIDE_M), hz, "m")
    if with_surface:
        wheel("Surface Type", surface, hz, "0=dry 1=wet 2=grass 3=dirt 4=gravel 5=kerb 6=special")

    cos_lat = math.cos(math.radians(GPS_REF_LAT))
    scalar("GPS Latitude", GPS_REF_LAT + pos[slow, 2] / R_DEG, shz, "deg")
    scalar("GPS Longitude", GPS_REF_LON + pos[slow, 0] / (R_DEG * cos_lat), shz, "deg")
    scalar("Path Lateral", lat[slow], shz, "m")
    scalar("Track Edge", np.where(lat[slow] < 0, -TRACK_EDGE_M, TRACK_EDGE_M), shz, "m")
    scalar("Lap Dist", lap_dist[slow], shz, "m")

    if other_cars > 0:
        lats = (-2.5, 2.5, -1.0, 1.0, -4.0, 4.0)
        s_abs = lap_idx[slow] * L + s_lap[slow]
        cols = [[] for _ in range(8)]
        for j in range(other_cars):
            sj = np.mod(s_abs + L * (j + 1) / (other_cars + 1), L)
            lj = np.full_like(sj, lats[j % len(lats)])
            pj = world(sj, lj)
            for col, v in zip(cols, (t[slow], np.full_like(sj, j + 1), pj[:, 0],
                                     pj[:, 1] + CG_M + RIDE_M, pj[:, 2], sj, lj,
                                     np.zeros_like(sj))):
                col.append(v)
        table("Other Cars", "ts DOUBLE, slot INTEGER, pos_x DOUBLE, pos_y DOUBLE, "
              "pos_z DOUBLE, lap_dist DOUBLE, path_lateral DOUBLE, in_pits INTEGER",
              [np.concatenate(c) for c in cols], shz, "pose_rows")

    con.execute('CREATE TABLE "Lap" (ts DOUBLE, value INTEGER)')
    con.executemany('INSERT INTO "Lap" VALUES (?, ?)',
                    [(float(t[c]), i) for i, c in enumerate(cross)])
    con.execute('CREATE TABLE "Lap Time" (ts DOUBLE, value DOUBLE)')
    con.executemany('INSERT INTO "Lap Time" VALUES (?, ?)',
                    [(float(t[cross[i + 1]]), float(t[cross[i + 1]] - t[cross[i]]))
                     for i in range(n_laps)])
    con.executemany("INSERT INTO eventsList VALUES (?, ?)", [("Lap", ""), ("Lap Time", "s")])
    con.close()
    return path
