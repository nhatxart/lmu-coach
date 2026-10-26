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
from .vehicle_pose import ORI_CHANNELS, basis_vectors, detect_pose_basis, ori_matrices, pitch_roll_deg

SURFACE_COLS = 11
FLAT_NOTE = "Flat: no elevation recorded for this track yet"
LAP_NOTE = "Elevation from this session only"

# (path, mtime_ns) -> topography built from that one file (None: nothing usable)
_LAP_CACHE: dict[tuple[str, int], dict | None] = {}


def unwrap_lap_dist(ld: list | np.ndarray) -> np.ndarray:
    """Lap Dist without the start/finish wrap: add the pre-wrap value after
    every big drop. Interpolating the raw value across a wrap would put a
    sample or two at half a lap (a height from the other side of the track);
    unwrapped first, the interpolation is smooth and lookups take it modulo
    the layout length."""
    a = np.asarray(ld, dtype=float)
    out = a.copy()
    for w in np.where(np.diff(a) < -500.0)[0]:
        out[w + 1:] += a[w]
    return out


def lap_topography(path: Path, track_name: str) -> dict | None:
    """Topography of this file alone (laps already driven in it), cached by
    (path, mtime). Used when no built topography matches the lap."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _LAP_CACHE:
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
        built = tt.load_topography(track_name, lap_len_m)

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
    ei = np.asarray(edge_idx, dtype=int)
    te = np.asarray(track_edge, float)[ei]
    s_e = np.asarray(s_lap, float)[ei] if s_lap is not None else np.zeros(len(ei))
    R, C = len(ei), SURFACE_COLS
    t = np.linspace(0.0, 1.0, C)
    sr = np.repeat(s_e, C)
    lat_r = (-te[:, None] + 2.0 * te[:, None] * t[None, :]).ravel()
    if lay_h is not None:
        left_y = tt.lookup_height(lay_h, s_e, -te) - datum
        right_y = tt.lookup_height(lay_h, s_e, te) - datum
        sy = tt.lookup_height(lay_h, sr, lat_r) - datum
    else:
        left_y, right_y, sy = np.zeros(R), np.zeros(R), np.zeros(R * C)
    kind = tt.paint_at(lay_p, sr, lat_r) if lay_p is not None else np.zeros(R * C, dtype=int)
    sx = (np.asarray(left_x, float)[ei][:, None] * (1 - t)[None, :]
          + np.asarray(right_x, float)[ei][:, None] * t[None, :]).ravel()
    sz = (np.asarray(left_z, float)[ei][:, None] * (1 - t)[None, :]
          + np.asarray(right_z, float)[ei][:, None] * t[None, :]).ravel()

    # ── corner labels ──
    if lay_h is not None and labels:
        li = np.array([i for i, _ in labels], dtype=int)
        label_y = tt.lookup_height(lay_h, np.asarray(s_lap, float)[li], lat_car[li]) - datum
    else:
        label_y = np.zeros(len(labels))

    note = {"recorded": "", "topography": "", "lap": LAP_NOTE}.get(source, FLAT_NOTE)
    if source == "none" and lay_p is not None and lay_p.get("paint"):
        note += " (kerbs from earlier laps)"

    li_ = np.asarray(line_idx, dtype=int)
    r3 = lambda a: np.round(np.asarray(a, float), 3).tolist()
    return {
        "elevation": {"source": source, "orientation": orientation,
                      "datum_m": round(datum, 3), "note": note,
                      "topography": _info(*shown) if shown else None},
        "y": r3(np.asarray(y)[li_]),
        "pitch_deg": r3(np.asarray(pitch)[li_]),
        "roll_deg": r3(np.asarray(roll)[li_]),
        "left_y": r3(left_y), "right_y": r3(right_y),
        "surface": {"rows": R, "cols": C,
                    "x": np.round(sx, 2).tolist(), "y": r3(sy),
                    "z": np.round(sz, 2).tolist(), "kind": [int(k) for k in kind]},
        "label_y": r3(label_y),
    }
