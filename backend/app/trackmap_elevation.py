"""
Elevation, attitude and the 3D surface ribbon of the /trackmap payload.

Kept out of main.get_trackmap (already huge): that function reads the raw
columns and slices them to the lap; everything here works on lap-length
numpy arrays and the per-track topography (track_topography).

Where the heights come from (`elevation.source`):
  recorded    this lap has Pos Y and a built topography matches its layout
  lap         Pos Y but no built topography: one is built from this file alone
  topography  no Pos Y (older recording), built topography matches
  none        neither: everything is flat, only kerb paint can show

Coordinates: see docs/superpowers/plans/2026-10-25-chase-view.md. `lat` is the
topography's frame (metres from the game's centre path, right of travel
positive) = lateral_sign * Path Lateral.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import track_topography as tt
from .telemetry import held_rows
from .vehicle_pose import ORI_CHANNELS, basis_vectors, detect_pose_basis, ori_matrices, pitch_roll_deg

SURFACE_COLS = 11
FLAT_NOTE = "Flat: no elevation recorded for this track yet"
LAP_NOTE = "Elevation from this session only"
PARTIAL_NOTE = "Partial lap: no topography match for this lap"
# Paint cells go out as the topography's own cells (see _lap_paint).
PAINT_CELL = {"along_m": tt.ROW_STEP_M, "across_m": tt.PAINT_CELL_M}
# Consecutive Lap Dist rows further apart than this are a jump (reset,
# teleport to the pits), not driven track: the rows between are not filled.
PAINT_GAP_MAX_M = 100.0

# (path, mtime_ns) -> topography built from that one file (None: nothing usable)
_LAP_CACHE: dict[tuple[str, int], dict | None] = {}
LAP_CACHE_MAX = 4


def lap_dist_rows(raw: list | np.ndarray, xp: np.ndarray) -> np.ndarray:
    """The game's Lap Dist at fractional source positions `xp`, wrap-safe.

    BUG FIX / PREVIOUSLY: the whole session was unwrapped by adding the
    pre-wrap sample at every start/finish wrap. That sample is up to one 10 Hz
    step (~4 m) short of the line, so the error added up lap after lap and
    later laps looked up heights, kerbs and labels metres away. Fix: use the
    game's value per sample (lookups take it modulo the layout length) and
    treat a wrap only where interpolation would cross it: between the last
    sample before it and the first after, interpolate towards the post-wrap
    value plus the pre-wrap one, so those rows stay near the line instead of
    landing at half a lap. The seam rows are off by the 10 Hz shortfall at
    most, and nothing accumulates.

    BUG FIX / PREVIOUSLY: real HF1 files write each 5 Hz scoring update of
    Lap Dist twice (telemetry.held_rows), and the repeat was interpolated as
    data: every other surface row looked its height up one update (~10 m)
    behind where it is, so the ribbon climbed in steps (pitch 0 / twice the
    slope, row by row, on the 10/27 Road Atlanta file) and a car resting on
    it would rock. Fix: interpolate between the rows where the game updated
    the value; the wrap rule below then runs on those rows.
    """
    a_all = np.asarray(raw, dtype=float)
    xp = np.asarray(xp, dtype=float)
    keep = np.nonzero(~held_rows(a_all))[0]
    if len(keep) == 0:
        return np.full(len(xp), np.nan)
    a = a_all[keep]
    xp = np.interp(xp, keep, np.arange(len(keep), dtype=float))   # position on the kept rows
    out = np.interp(xp, np.arange(len(a)), a)
    if len(a) < 2:
        return out
    lo = np.clip(np.floor(xp).astype(int), 0, len(a) - 2)
    seam = (a[lo + 1] - a[lo] < -500.0) & (xp > lo)
    if seam.any():
        frac = xp[seam] - lo[seam]
        # from a[lo] to a[lo+1] + a[lo] (the post-wrap value on the unwrapped axis)
        out[seam] = a[lo[seam]] + frac * a[lo[seam] + 1]
    return out


def source_positions(n_src: int, target: int, native_x: np.ndarray | None,
                     n_fast: int | None) -> np.ndarray:
    """Fractional source index of each of `target` rows: the same mapping as
    main._interp_to_length uses (game-time based for 100 Hz sources, else
    index fraction), so Lap Dist lines up with the other channels."""
    if n_src == target:
        return np.arange(target, dtype=float)
    if native_x is not None and target == len(native_x) and n_src == n_fast:
        return np.asarray(native_x, dtype=float)
    return np.linspace(0, n_src - 1, target)


def lap_topography(path: Path, track_name: str) -> dict | None:
    """Topography of this file alone (laps already driven in it), cached by
    (path, mtime). Used when no built topography matches the lap."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _LAP_CACHE:
        while len(_LAP_CACHE) >= LAP_CACHE_MAX:     # a topography is ~100 kB; keep a few
            _LAP_CACHE.pop(next(iter(_LAP_CACHE)))
        try:
            topo = tt.build_topography([path], tt.slug_for_track(track_name) or "", track_name)
        except Exception as e:                         # elevation is optional
            print(f"[trackmap] lap topography failed: {type(e).__name__}: {e}")
            topo = None
        _LAP_CACHE[key] = topo
    return _LAP_CACHE[key]


def attitude_deg(px: np.ndarray, pz: np.ndarray, speed_kmh: np.ndarray | None,
                 ori: dict[str, np.ndarray], t: np.ndarray | None,
                 f0: int, f1: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(pitch, roll) in degrees for every 100 Hz row, or None when the
    orientation matrix is missing or does not follow the direction of travel
    (vehicle_pose refuses rather than guesses).

    The matrix layout is detected on rows f0..f1 (the lap) from world
    velocity of the raw Pos X/Z; then applied to the whole session.
    """
    M = ori_matrices(ori)
    n = len(px)
    if M is None or len(M) != n or len(pz) != n or n < 3:
        return None
    sl = slice(max(0, f0), min(n, f1 + 1))
    tw = np.asarray(t, float)[sl] if t is not None and len(t) == n else np.arange(n)[sl] * 0.01
    if len(tw) < 3:
        return None
    with np.errstate(all="ignore"):
        vx, vz = np.gradient(px[sl], tw), np.gradient(pz[sl], tw)
    speed = (np.asarray(speed_kmh, float)[sl] / 3.6
             if speed_kmh is not None and len(speed_kmh) == n else np.hypot(vx, vz))
    basis = detect_pose_basis(M[sl], vx, vz, speed)
    if basis is None:
        return None
    fwd, left, _ = basis_vectors(M, basis)
    return pitch_roll_deg(fwd, left)


def _ribbon_xz(edge_idx, left_x, left_z, right_x, right_z):
    ei = np.asarray(edge_idx, dtype=int)
    t = np.linspace(0.0, 1.0, SURFACE_COLS)
    sx = (np.asarray(left_x, float)[ei][:, None] * (1 - t)[None, :]
          + np.asarray(right_x, float)[ei][:, None] * t[None, :]).ravel()
    sz = (np.asarray(left_z, float)[ei][:, None] * (1 - t)[None, :]
          + np.asarray(right_z, float)[ei][:, None] * t[None, :]).ravel()
    return sx, sz


def flat_payload(*, edge_idx, line_idx, n_labels: int, left_x, left_z, right_x, right_z,
                 note: str = FLAT_NOTE) -> dict:
    """The "none"-shaped payload with correct lengths: what /trackmap returns
    when the elevation step fails (it must never cost the map). Only the
    flat ribbon geometry is computed, from arrays that already worked."""
    R, C = len(edge_idx), SURFACE_COLS
    try:
        sx, sz = _ribbon_xz(edge_idx, left_x, left_z, right_x, right_z)
        sx, sz = np.round(sx, 2).tolist(), np.round(sz, 2).tolist()
    except Exception:
        sx = sz = [0.0] * (R * C)
    nl = len(line_idx)
    return {
        "elevation": {"source": "none", "orientation": "none", "datum_m": 0.0,
                      "note": note, "topography": None, "lateral_sign": 1},
        "y": [0.0] * nl, "pitch_deg": [0.0] * nl, "roll_deg": [0.0] * nl,
        "left_y": [0.0] * R, "right_y": [0.0] * R,
        "surface": {"rows": R, "cols": C, "x": sx, "y": [0.0] * (R * C),
                    "z": sz, "kind": [0] * (R * C), "s": [0.0] * R, "length_m": 0.0},
        "paint": [], "paint_cell": dict(PAINT_CELL),
        "label_y": [0.0] * n_labels,
    }


def _lap_paint(lay: dict | None, s_lap: np.ndarray | None, sign: int) -> list:
    """The layout's paint cells along the part of the track this lap drove,
    as [[s_m, g_m, kind], ...]: s = cell centre in lap distance (row*2 + 1),
    g = cell centre across in the RIBBON's frame (sign * topography lat, see
    elevation_payload), kind as in track_topography.PAINT_KINDS.

    BUG FIX / PREVIOUSLY: paint reached the chase view only as `surface.kind`,
    the cells re-sampled at the ribbon's 600 rows x 11 columns (rows ~6.8 m
    apart at Road Atlanta, ~22.7 m at Le Mans; columns 1.2-1.5 m). Kerbs
    snapped to row blocks, short ones vanished, and the car looked on or off
    a kerb by metres. Fix: send the 2 m x 0.5 m cells themselves; the UI
    places each at its true (s, g) on the ribbon.
    """
    if lay is None or not lay.get("paint") or s_lap is None:
        return []
    L = float(lay["length_m"])
    step = float(lay["row_step_m"])
    N = int(lay["rows"])
    cell = float(lay["paint_cell_m"])
    s = np.asarray(s_lap, dtype=float)
    s = np.mod(s[np.isfinite(s)], L)
    if len(s) == 0:
        return []
    covered = np.zeros(N, dtype=bool)
    covered[np.minimum((s // step).astype(int), N - 1)] = True
    # 10 Hz rows at speed are 4-9 m apart: fill the rows between two
    # consecutive samples (across the start/finish wrap too)
    d = np.mod(np.diff(s), L)
    for i in np.nonzero((d >= step) & (d < PAINT_GAP_MAX_M))[0]:
        covered[np.arange(int(s[i] // step), int((s[i] + d[i]) // step) + 1) % N] = True
    # one row either side: the wheels run 1.45 m ahead of / behind the
    # reference point whose Lap Dist this is
    covered = covered | np.roll(covered, 1) | np.roll(covered, -1)
    return [[round(r * step + step / 2, 2), round(sign * c * cell, 2), int(k)]
            for r, c, k in lay["paint"] if 0 <= int(r) < N and covered[int(r)]]


def _info(topo: dict, lay: dict) -> dict:
    src = lay.get("sources", {})
    return {"built": topo.get("built", ""), "files": int(src.get("files", 0)),
            "laps": int(src.get("laps", 0)), "coverage": float(lay.get("coverage", 0.0))}


def elevation_payload(*, path: Path, track_name: str, lap_len_m: float | None,
                      s_lap: np.ndarray | None, path_lat: np.ndarray,
                      ground_rec: np.ndarray | None, attitude: tuple | None,
                      edge_idx: list[int], track_edge: np.ndarray,
                      left_x: np.ndarray, left_z: np.ndarray,
                      right_x: np.ndarray, right_z: np.ndarray,
                      labels: list[tuple[int, float]], line_idx: list[int]) -> dict:
    """Everything the chase view needs, as the new /trackmap keys.

    s_lap          unwrapped Lap Dist on the lap's position rows (None: no Lap
                   Dist in the file, so no layout can be used)
    path_lat       Path Lateral per row (raw sign; the layout's lateral_sign
                   converts it to the right-positive frame)
    ground_rec     recorded ground height under the car per row (Pos Y - mean
                   ride height - cg), None without Pos Y
    attitude       (pitch_deg, roll_deg) per row from the orientation matrix
    edge_idx       row indices of the downsampled edge points (same as track.*)
    labels         [(row index, unused)] corner labels, for their heights
    """
    n = len(path_lat)
    built = None
    if s_lap is not None and track_name and lap_len_m:
        # lap_max_s: the furthest Lap Dist reached, so a partial lap can still
        # find its layout (track_topography.load_topography)
        s_fin = np.asarray(s_lap, float)
        s_fin = s_fin[np.isfinite(s_fin)]
        built = tt.load_topography(track_name, lap_len_m,
                                   lap_max_s=float(s_fin.max()) if len(s_fin) else None)

    chosen, source = None, "none"          # chosen = (topo, layout) that gives heights
    if s_lap is not None:
        if built and built[1].get("has_heights"):
            chosen, source = built, ("recorded" if ground_rec is not None else "topography")
        elif ground_rec is not None and track_name:
            lt = lap_topography(path, track_name)
            lay = tt.pick_layout(lt, lap_len_m, tolerance=None) if lt else None
            if lay and lay.get("has_heights"):
                chosen, source = (lt, lay), "lap"
    # kerbs (and the info block) may come from a built layout without heights
    shown = chosen or (built if built and s_lap is not None else None)

    datum = float(chosen[0].get("datum_m", 0.0)) if chosen else 0.0
    sign = int(shown[1].get("lateral_sign", {}).get("value", 1)) if shown else 1
    lay_h = chosen[1] if chosen else None
    lay_p = shown[1] if shown else None
    lat_car = sign * np.asarray(path_lat, float)

    # ── per-row height and attitude of the car ──
    if source in ("recorded", "lap"):
        y = np.asarray(ground_rec, float) - datum
    elif source == "topography":
        y = tt.lookup_height(lay_h, s_lap, lat_car) - datum
    else:
        y = np.zeros(n)

    if attitude is not None:
        pitch, roll, orientation = attitude[0], attitude[1], "recorded"
    elif lay_h is not None:
        grade, bank = tt.grade_bank(lay_h, s_lap)
        pitch, roll, orientation = np.degrees(np.arctan(grade)), -np.degrees(np.arctan(bank)), "surface"
    else:
        pitch, roll, orientation = np.zeros(n), np.zeros(n), "none"

    # ── edges and the surface ribbon (same rows as track.*) ──
    # Two lateral frames meet here. The ribbon's columns run from the left
    # edge to the right edge as main.py built them, and main.py treats Path
    # Lateral as right-positive: column offset g = -te .. +te in that frame.
    # The topography's lat is lateral_sign * Path Lateral (right of travel
    # when the sign detection is right). A ribbon point at g is where the car
    # would be drawn with Path Lateral = g, so its topography lat is sign * g,
    # the same mapping as lat_car above; a paint cell at topography lat sits at
    # g = sign * lat.
    # BUG FIX / PREVIOUSLY: the columns, the edge heights and the paint used
    # g directly as the topography lat while the car used sign * Path
    # Lateral, so with a detected sign of -1 the bank and the kerbs would have
    # been mirrored against the edges and the car.
    ei = np.asarray(edge_idx, dtype=int)
    te = np.asarray(track_edge, float)[ei]
    s_e = np.asarray(s_lap, float)[ei] if s_lap is not None else np.zeros(len(ei))
    R, C = len(ei), SURFACE_COLS
    t = np.linspace(0.0, 1.0, C)
    sr = np.repeat(s_e, C)
    g_r = (-te[:, None] + 2.0 * te[:, None] * t[None, :]).ravel()
    lat_r = sign * g_r
    if lay_h is not None:
        left_y = tt.lookup_height(lay_h, s_e, sign * -te) - datum
        right_y = tt.lookup_height(lay_h, s_e, sign * te) - datum
        sy = tt.lookup_height(lay_h, sr, lat_r) - datum
    else:
        left_y, right_y, sy = np.zeros(R), np.zeros(R), np.zeros(R * C)
    # `kind` (paint sampled at the columns) stays for older pages; the chase
    # view draws `paint` (true cells) when it is there
    kind = tt.paint_at(lay_p, sr, lat_r) if lay_p is not None else np.zeros(R * C, dtype=int)
    paint = _lap_paint(lay_p, s_lap, sign)
    # lap distance per ribbon row, as the topography indexes it (modulo the
    # layout length); the raw value without a layout
    s_rows = np.mod(s_e, float(lay_p["length_m"])) if lay_p is not None else s_e
    sx, sz = _ribbon_xz(edge_idx, left_x, left_z, right_x, right_z)

    # ── corner labels ──
    if lay_h is not None and labels:
        li = np.array([i for i, _ in labels], dtype=int)
        label_y = tt.lookup_height(lay_h, np.asarray(s_lap, float)[li], lat_car[li]) - datum
    else:
        label_y = np.zeros(len(labels))

    note = {"recorded": "", "topography": "", "lap": LAP_NOTE}.get(source, FLAT_NOTE)
    if source == "none" and shown is None and s_lap is not None and track_name \
            and (tt.load_track_topography(track_name) or {}).get("layouts"):
        # the track has a topography, this lap just fits none of its layouts:
        # "no elevation recorded for this track" would be false
        note = PARTIAL_NOTE
    if source == "none" and lay_p is not None and lay_p.get("paint"):
        note += " (kerbs from earlier laps)"

    li_ = np.asarray(line_idx, dtype=int)
    r3 = lambda a: np.round(np.asarray(a, float), 3).tolist()
    return {
        "elevation": {"source": source, "orientation": orientation,
                      "datum_m": round(datum, 3), "note": note,
                      "topography": _info(*shown) if shown else None,
                      # Path Lateral -> right-positive lat (track limits use it)
                      "lateral_sign": sign},
        "y": r3(np.asarray(y)[li_]),
        "pitch_deg": r3(np.asarray(pitch)[li_]),
        "roll_deg": r3(np.asarray(roll)[li_]),
        "left_y": r3(left_y), "right_y": r3(right_y),
        "surface": {"rows": R, "cols": C,
                    "x": np.round(sx, 2).tolist(), "y": r3(sy),
                    "z": np.round(sz, 2).tolist(), "kind": [int(k) for k in kind],
                    "s": np.round(s_rows, 2).tolist(),
                    # period of `s` (the layout length), 0 without a layout
                    "length_m": round(float(lay_p["length_m"]), 3) if lay_p is not None else 0.0},
        "paint": paint, "paint_cell": dict(PAINT_CELL),
        "label_y": r3(label_y),
    }
