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

The surface frame at the car: up = surface normal, forward = direction of
motion within the surface. The chassis is that frame plus body attitude on
its springs: roll up to 1.5 deg right side down in the (left-hand) corners,
scaled by yaw rate, and a constant 0.5 deg nose up. Ori is the CHASSIS
orientation, and each wheel's Ride Height is what that attitude leaves
between its chassis corner (reference point + fwd*(+-1.45) + left*(+-0.80))
and the ground plane under it, so Pos Y + corner y - Ride Height - 0.25 is
the surface under that wheel, to first order. Ride heights therefore differ
per wheel (~0.02-0.08 m): a swapped wheel order or a dropped Ride Height
subtraction shows up as height and bank errors.
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
FRONT_M, REAR_M = 1.45, -1.45
BODY_ROLL_DEG = 1.5     # right side down at full cornering
BODY_PITCH_DEG = 0.5    # nose up
KERB_LAT_M = 5.5
TRACK_EDGE_M = 6.0
SPEED_MS = 40.0
# Per-wheel slip angles (write_hilly_session(slip_angles=True)): a car with
# no body slip (beta = 0) on the LMGT3 dims the UI falls back to for this
# unknown car (car_models.json: wheelbase 2.70, tracks 1.70 / 1.65, origin
# at mid-wheelbase). The front slip angle builds up linearly over the first
# SLIP_RAMP_M of each semicircle to SLIP_FRONT_DEG (positive = the tyre
# travels right of where it points: outward in these left-handers) and
# eases out over its last SLIP_RAMP_M; the rears carry exactly their travel
# angle (they do not steer). So the road-wheel angle the kinematic estimate
# should find is known: delta = front travel angle - front slip angle.
SLIP_FRONT_DEG = 2.5
SLIP_RAMP_M = 40.0
SLIP_A_M, SLIP_B_M = 1.35, 1.35
SLIP_HALF_TRACK_F, SLIP_HALF_TRACK_R = 0.85, 0.825
STEER_LOCK_DEG = 15.0   # Steering Pos +-1 = +-15 deg of road-wheel angle (right positive, as in LMU)
# Out-lap (write_hilly_session(out_lap=True)), modelled on COTA: lap 0 leaves
# the pits, `In Pits` = 1 from the lap start until the speed-limit line at
# PIT_FLAG_CLEAR_M, and until PIT_ROAD_END_M it drives a parallel pit-exit
# road PIT_ROAD_DROP_M below the track that merges by PIT_MERGE_M (cosine
# ramp). The game reports Lap Dist / Path Lateral for that road as if it
# were the track, so the file's Path Lateral is lap 0's own offset there.
PIT_FLAG_CLEAR_M = 120.0
PIT_ROAD_END_M, PIT_MERGE_M = 200.0, 300.0
PIT_ROAD_DROP_M = 2.5
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


def slip_ramp(s):
    """0..1: how far the front slip angle has built up at lap distance s
    (linear over SLIP_RAMP_M in and out of each semicircle, 1 between)."""
    s = np.mod(np.asarray(s, dtype=float), L)
    u = np.zeros_like(s)
    for a, b in ((_SEG_B, _SEG_C), (_SEG_D, L)):
        inside = (s >= a) & (s < b)
        u[inside] = np.minimum(np.clip((s[inside] - a) / SLIP_RAMP_M, 0, 1),
                               np.clip((b - s[inside]) / SLIP_RAMP_M, 0, 1))
    return u


def slip_truth(s, speed_ms, yaw_rate):
    """(alpha (n,4) deg FL FR RL RR, delta (n,) deg) for slip_angles=True.

    yaw_rate is the file's Yaw Rate (left-positive here: the ring is driven
    counter-clockwise in world x/z); the car frame below is right-positive.
    With beta = 0 a wheel at (along a, right w) travels at
    atan2(r*a, V - r*w) off the body axis (r right-positive). The rears
    point straight, so their slip angle is that angle; the fronts point at
    delta = (front travel angle at mid-track) - SLIP_FRONT_DEG * ramp, and
    each front wheel's slip angle is its own travel angle minus delta, so
    the mean of the two fronts' (travel - slip) is exactly delta."""
    v = np.asarray(speed_ms, dtype=float)
    r = -np.asarray(yaw_rate, dtype=float)
    th = lambda a, w: np.degrees(np.arctan2(r * a, v - r * w))  # noqa: E731
    delta = th(SLIP_A_M, 0.0) - SLIP_FRONT_DEG * slip_ramp(s)
    alpha = np.stack([th(SLIP_A_M, -SLIP_HALF_TRACK_F) - delta,
                      th(SLIP_A_M, SLIP_HALF_TRACK_F) - delta,
                      th(-SLIP_B_M, -SLIP_HALF_TRACK_R),
                      th(-SLIP_B_M, SLIP_HALF_TRACK_R)], axis=1)
    return alpha, delta


def pit_road_drop(s):
    """How far the pit-exit road of out_lap=True sits below the track at
    lap distance s (PIT_ROAD_DROP_M until PIT_ROAD_END_M, 0 from PIT_MERGE_M)."""
    s = np.asarray(s, dtype=float)
    return PIT_ROAD_DROP_M * (1.0 - _smooth((s - PIT_ROAD_END_M) / (PIT_MERGE_M - PIT_ROAD_END_M)))


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def write_hilly_session(path: Path, lap_offsets=(0.0, 1.5, -1.5), with_pose=True,
                        with_surface=True, other_cars=0,
                        track_name="Synthetic Hill Ring",
                        cross_offset_ticks=0, track_edge_m=TRACK_EDGE_M,
                        vehicle_class=None, with_edges=True,
                        scoring_hold=False, slip_angles=False, out_lap=False,
                        trailing_partial_m: float = 0.0,
                        lap_dist_scale: float = 1.0, speed_scale: float = 1.0,
                        speed_wave: float = 0.0, with_gps: bool = True,
                        with_lap_dist: bool = True,
                        gps_ref: tuple[float, float] = (GPS_REF_LAT, GPS_REF_LON)) -> Path:
    """cross_offset_ticks moves every crossing but the first off its 10 Hz row
    (-8: the first 10 Hz row after it reads ~3 m), like the real game, where
    the line falls anywhere between two rows. Default 0 keeps the crossings
    on rows with Lap Dist exactly 0.

    track_edge_m: the Track Edge half-width (the kerb stays at KERB_LAT_M).
    vehicle_class: written as metadata VehicleClass (None: no such key).
    with_edges=False: no Path Lateral / Track Edge tables (recorder v1).
    scoring_hold: Lap Dist and Path Lateral repeat the previous 10 Hz row on
    every other row, as in real HF1 files: the game updates them with its
    scoring data at 5 Hz, and the recorder writes them at 10 Hz.
    slip_angles: also write the recorder's per-wheel "Slip Angle" table
    (see SLIP_FRONT_DEG; `slip_truth` gives the values) and Steering Pos as
    the true road-wheel angle / STEER_LOCK_DEG, right positive like the
    game's (the default Steering Pos is yaw rate x 20, left positive).
    out_lap: lap 0 is an out-lap on a lower pit-exit road and the file gets
    an `In Pits` event table (see PIT_FLAG_CLEAR_M). The ground under lap 0
    there is truth(s, lat) - pit_road_drop(s), not truth(s, lat).
    trailing_partial_m: keep driving that many metres past the closing
    crossing, so the file ends mid-lap like a session left mid-lap: its last
    Lap row then opens one more (incomplete) lap with no Lap Time row.
    lap_dist_scale: multiplies the written Lap Dist values only (geometry,
    times and the other channels unchanged), to stand in for another layout
    of the same track name.
    speed_scale: multiplies SPEED_MS for this file (0.97: every lap ~3 %
    slower, the laps a lap comparison should show you gaining on).
    speed_wave: amplitude a of a speed variation along each lap, s = L *
    (u + a sin(2 pi u) / (2 pi)) at lap time fraction u: speed (1 + a cos
    2 pi u) times the mean, lap time unchanged. Against a constant-speed lap
    the time delta then rises in some parts of the lap and falls in others.
    with_gps=False: no GPS Latitude/Longitude tables (positions only as Pos
    X/Z, like recorder v1). with_lap_dist=False: no Lap Dist table."""
    path = Path(path)
    offsets = tuple(float(o) for o in lap_offsets)
    n_laps = len(offsets)
    ticks_per_lap = int(round(L / (SPEED_MS * float(speed_scale)) * 100))
    ms0 = int(round(T0_S * 1000))
    all_t = [(ms0 + 10 * k) / 1000.0 for k in range((n_laps + 1) * ticks_per_lap)]
    gate = slow_rows(all_t)
    gate_arr = np.asarray(gate)
    # crossings on gate ticks, so Lap Dist is exactly 0 there
    cross = [int(gate_arr[np.argmin(np.abs(gate_arr - i * ticks_per_lap))])
             for i in range(n_laps + 1)]
    cross = [cross[0]] + [c + int(cross_offset_ticks) for c in cross[1:]]
    n_tail = int(round(float(trailing_partial_m) / SPEED_MS * 100))
    n = cross[-1] + 1 + n_tail
    assert n <= len(all_t), "trailing_partial_m must be shorter than a lap"
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
        u = (k - cross[i]) / span
        a = float(speed_wave)
        sl = L * (u + a * np.sin(2 * np.pi * u) / (2 * np.pi))
        if i == n_laps - 1:
            sl[-1] = L   # closing crossing: geometry at L, Lap Dist written as 0
        s_lap[k], lap_idx[k] = sl, i
        sdot[k] = L / (span * 0.01) * (1.0 + a * np.cos(2 * np.pi * u))
        lat[k] = _lap_lat(offsets, i, sl)
        dlat_ds[k] = (_lap_lat(offsets, i, sl + e) - _lap_lat(offsets, i, sl - e)) / (2 * e)
    if n_tail:
        # past the closing crossing: lap n_laps at SPEED_MS, on the last
        # lap's line (its lat at s = 0 is where that lap ended)
        k = np.arange(cross[-1] + 1, n)
        sl = SPEED_MS * 0.01 * (k - cross[-1])
        s_lap[k], lap_idx[k], sdot[k] = sl, n_laps, SPEED_MS
        lat[k] = _lap_lat(offsets, n_laps - 1, sl)
        dlat_ds[k] = (_lap_lat(offsets, n_laps - 1, sl + e)
                      - _lap_lat(offsets, n_laps - 1, sl - e)) / (2 * e)
    lap_dist = np.where(s_lap >= L, 0.0, s_lap)

    pos = world(s_lap, lat)
    if out_lap:
        # the pit-exit road under lap 0: the car (and so its ground probes) is
        # lower; Ori and Ride Height stay as on the track (same pose, offset
        # down), which is all the topography fit sees of it
        pos[:, 1] -= np.where(lap_idx == 0, pit_road_drop(s_lap), 0.0)
    t_s = (world(s_lap + e, lat) - world(s_lap - e, lat)) / (2 * e)
    t_l = (world(s_lap, lat + e) - world(s_lap, lat - e)) / (2 * e)
    vel_dir = t_s + t_l * dlat_ds[:, None]
    fwd = _unit(vel_dir)
    up = _unit(np.cross(t_s, t_l))
    left = np.cross(fwd, up)
    speed = np.linalg.norm(vel_dir, axis=1) * sdot
    heading = np.unwrap(np.arctan2(fwd[:, 2], fwd[:, 0]))
    yaw_rate = np.gradient(heading, t)

    # Chassis = surface frame + body roll (right side down in the left-hand
    # corners, eased in over ~0.5 s) + constant nose-up pitch.
    corner = np.clip(yaw_rate / (SPEED_MS / RADIUS_M), -1.0, 1.0)
    corner = np.convolve(np.pad(corner, 25, mode="edge"), np.ones(51) / 51, mode="valid")
    roll = np.radians(BODY_ROLL_DEG) * corner[:, None]
    pitch = math.radians(BODY_PITCH_DEG)
    c_left = left * np.cos(roll) + up * np.sin(roll)        # left side up = right down
    c_up = up * np.cos(roll) - left * np.sin(roll)
    c_fwd = fwd * math.cos(pitch) + c_up * math.sin(pitch)  # nose up
    c_up = c_up * math.cos(pitch) - fwd * math.sin(pitch)
    M = np.stack([c_left, c_up, -c_fwd], axis=2)      # columns: left, up, rear

    # Ride height per wheel: chassis corner height above the ground plane
    # (normal `up`) straight below it. Pos Y puts the reference point
    # CG_M + RIDE_M above the ground, so a level chassis reads RIDE_M.
    ride = np.empty((n, 4))
    for j, (along, lfac) in enumerate(((FRONT_M, HALF_TRACK_M), (FRONT_M, -HALF_TRACK_M),
                                       (REAR_M, HALF_TRACK_M), (REAR_M, -HALF_TRACK_M))):
        o = c_fwd * along + c_left * lfac
        plane_dy = -(up[:, 0] * o[:, 0] + up[:, 2] * o[:, 2]) / up[:, 1]
        ride[:, j] = RIDE_M + o[:, 1] - plane_dy

    wheel_lat = np.stack([lat - HALF_TRACK_M, lat + HALF_TRACK_M,
                          lat - HALF_TRACK_M, lat + HALF_TRACK_M], axis=1)
    surface = np.where(wheel_lat > KERB_LAT_M, 5.0, 0.0)
    if out_lap:
        # the pit road reads as grass (any paint of its own would do): it must
        # not land on the track either
        surface[(lap_idx == 0) & (s_lap < PIT_ROAD_END_M)] = 2.0

    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE metadata (key VARCHAR, value VARCHAR)")
    meta = [("TrackName", track_name), ("CarName", "Synthetic Car #1"),
            ("DriverName", "Tester"), ("SessionType", "Practice"),
            ("RecordedBy", "lmu-coach-recorder hf1")]
    if with_pose:
        meta.append(("PoseChannels", "1"))
    if vehicle_class:
        meta.append(("VehicleClass", vehicle_class))
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
    if slip_angles:
        alpha_deg, delta_deg = slip_truth(s_lap, speed, yaw_rate)
        wheel("Slip Angle", np.radians(alpha_deg), hz, "rad")
        scalar("Steering Pos", np.clip(delta_deg / STEER_LOCK_DEG, -1, 1), hz, "")
    else:
        scalar("Steering Pos", np.clip(yaw_rate * 20.0, -100, 100), hz, "%")
    scalar("G Force Lat", speed * yaw_rate / 9.81, hz, "G")
    scalar("G Force Long", np.zeros(n), hz, "G")
    if with_pose:
        for r in range(3):
            for c, a in enumerate("XYZ"):
                scalar(f"Ori R{r}{a}", M[:, r, c], hz)
        wheel("Ride Height", ride, hz, "m")
    if with_surface:
        wheel("Surface Type", surface, hz, "0=dry 1=wet 2=grass 3=dirt 4=gravel 5=kerb 6=special")

    # gps_ref: where the game's GPS puts the world origin. LMU puts every
    # track near (35.37, 138.92) (Fuji), whatever its real place; another
    # gps_ref stands in for a file whose GPS is somewhere else.
    ref_lat, ref_lon = float(gps_ref[0]), float(gps_ref[1])
    cos_lat = math.cos(math.radians(ref_lat))
    if with_gps:
        scalar("GPS Latitude", ref_lat + pos[slow, 2] / R_DEG, shz, "deg")
        scalar("GPS Longitude", ref_lon + pos[slow, 0] / (R_DEG * cos_lat), shz, "deg")
    ld_slow, pl_slow = lap_dist[slow] * float(lap_dist_scale), lat[slow].copy()
    te_slow = np.where(lat[slow] < 0, -track_edge_m, track_edge_m)
    if scoring_hold:
        # the real half-width varies a little along the lap (so its values
        # repeat in pairs, like the other scoring channels); +-1 cm here
        te_slow = te_slow * (1.0 + 0.002 * np.sin(2 * np.pi * lap_dist[slow] / 50.0))
        ld_slow[1::2], pl_slow[1::2], te_slow[1::2] = ld_slow[0:-1:2], pl_slow[0:-1:2], te_slow[0:-1:2]
    if with_edges:
        scalar("Path Lateral", pl_slow, shz, "m")
        scalar("Track Edge", te_slow, shz, "m")
    if with_lap_dist:
        scalar("Lap Dist", ld_slow, shz, "m")

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
    events = [("Lap", ""), ("Lap Time", "s")]
    if out_lap:
        # as the recorder writes it: (ts, value) on change, first row at start
        clear = int(np.argmax((lap_idx == 0) & (s_lap >= PIT_FLAG_CLEAR_M)))
        con.execute('CREATE TABLE "In Pits" (ts DOUBLE, value DOUBLE)')
        con.executemany('INSERT INTO "In Pits" VALUES (?, ?)',
                        [(float(t[cross[0]]), 1.0), (float(t[clear]), 0.0)])
        events.append(("In Pits", ""))
    con.executemany("INSERT INTO eventsList VALUES (?, ?)", events)
    con.close()
    return path
