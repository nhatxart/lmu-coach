"""
Geometry-first corner detection: corners come from the TRACK, not from how one
driver happened to drive one lap.

corner_features.py used to find corners as "lateral G above 0.5 g in this lap".
That depends on the car and the driver: a hypercar takes Fuji's T2 or T7/T8 almost
flat and never reaches 0.5 g there, so those corners vanished (10 found, 16 on the
map). Here a corner is a stretch of road with real curvature:

    curvature k(s) = yaw rate / speed     (1/m, signed; falls back to lat G / v^2)

sampled every metre along the lap distance, so every lap of every car sees the same
road. An apex is a peak of |k|; a corner ends where |k| falls back to near zero or
changes sign. What the driver had to DO there (brake / lift / flat) is a separate
label, so flat-out kinks stay in the list but are not coached as braking zones.

Laps are then combined: apexes that appear in at least half of the laps at about the
same distance form the track's corner map (median position/extent), which is stable
across laps and cars. Official corner numbers are attached afterwards (track_db ref
points / a per-track file), by order.

Thresholds are engineering heuristics: tune against the track diagrams.
"""

from __future__ import annotations

import numpy as np

from .telemetry import LapTelemetry

GRID_M = 1.0
SMOOTH_M = 25                  # moving-average window on curvature
KAPPA_ENTER = 1 / 900.0        # (EN) an apex must reach radius <= 900 m ...
KAPPA_EXIT = 1 / 2500.0        # ... and a corner runs until radius > 2500 m
PEAK_PROMINENCE = 1 / 800.0    # (EN) separates two apexes inside one run
MIN_APEX_SEPARATION_M = 45.0
MIN_HEADING_DEG = 16.0         # (EN) tuned vs track_db ref points (COTA, Barcelona) and Wikipedia turn counts
MERGE_GAP_M = 0.0              # (EN) same-direction bits closer than this are one corner
MIN_SPEED_MS = 10.0            # curvature = yaw / v is meaningless when nearly stopped
CLUSTER_GAP_M = 45.0
MIN_LAP_FRACTION = 0.5         # a corner must show up in at least half of the laps ...
MIN_LAPS_SEEN = 2              # ... and in at least two when two or more laps are given
MIN_LAP_LENGTH_M = 1000.0

FLAT_MIN_THROTTLE_PCT = 90.0
BRAKE_ZONE_PCT = 20.0
BRAKE_LOOKBACK_M = 250.0


def _pct(a):
    a = np.nan_to_num(np.asarray(a, dtype=float))
    return a * 100.0 if a.size and np.max(np.abs(a)) <= 1.5 else a


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size < n:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def curvature_profile(lap: LapTelemetry, grid_m: float = GRID_M) -> dict | None:
    """The lap on a 1 m distance grid: signed curvature, speed, brake %, throttle %."""
    ch = lap.channels
    if "Lap Dist" not in ch or "Ground Speed" not in ch:
        return None
    d = np.maximum.accumulate(np.nan_to_num(ch["Lap Dist"]))
    keep = np.concatenate(([True], np.diff(d) > 1e-6))
    d = d[keep]
    if d.size < 20 or d[-1] < MIN_LAP_LENGTH_M:
        return None
    v = np.nan_to_num(ch["Ground Speed"])[keep] / 3.6
    if "Yaw Rate" in ch:
        k_raw = np.nan_to_num(ch["Yaw Rate"])[keep] / np.maximum(v, MIN_SPEED_MS)
        source = "yaw_rate"
    elif "G Force Lat" in ch:
        k_raw = np.nan_to_num(ch["G Force Lat"])[keep] * 9.81 / np.maximum(v, MIN_SPEED_MS) ** 2
        source = "lateral_g"
    else:
        return None
    k_raw = np.where(v >= MIN_SPEED_MS, k_raw, 0.0)
    s = np.arange(0.0, d[-1], grid_m)

    brake = _pct(ch.get("Brake Pos", np.zeros(len(keep))))[keep]
    thr = _pct(ch.get("Throttle Pos", np.zeros(len(keep))))[keep]
    return {
        "s": s,
        "k": _smooth(np.interp(s, d, k_raw), int(SMOOTH_M / grid_m)),
        "v_kmh": np.interp(s, d, v * 3.6),
        "brake": np.interp(s, d, brake),
        "thr": np.interp(s, d, thr),
        "length_m": float(d[-1]),
        "source": source,
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.size:
        return []
    m = np.concatenate(([False], mask.astype(bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def detect_apexes(prof: dict, grid_m: float = GRID_M) -> list[dict]:
    """Corners of ONE lap on the distance grid (see module docstring)."""
    from scipy.signal import find_peaks
    k, a = prof["k"], np.abs(prof["k"])
    out: list[dict] = []
    for r0, r1 in _runs(a > KAPPA_EXIT):
        sign = np.sign(k[r0:r1 + 1])
        cuts = [0] + (np.flatnonzero(np.diff(sign) != 0) + 1).tolist() + [r1 - r0 + 1]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            a0, a1 = r0 + c0, r0 + c1 - 1
            if a1 - a0 < 3:
                continue
            seg = a[a0:a1 + 1]
            pk, _ = find_peaks(seg, prominence=PEAK_PROMINENCE, distance=int(MIN_APEX_SEPARATION_M / grid_m))
            pk = [p for p in pk if seg[p] >= KAPPA_ENTER]
            if not pk:
                if seg.max() >= KAPPA_ENTER:
                    pk = [int(np.argmax(seg))]
                else:
                    continue
            # split between consecutive apexes at the curvature minimum between them
            edges = [0]
            for p0, p1 in zip(pk[:-1], pk[1:]):
                edges.append(p0 + int(np.argmin(seg[p0:p1 + 1])))
            edges.append(len(seg) - 1)
            for i, p in enumerate(pk):
                e0, e1 = a0 + edges[i], a0 + edges[i + 1]
                heading = float(np.degrees(np.sum(a[e0:e1 + 1]) * grid_m))
                if heading < MIN_HEADING_DEG:
                    continue
                apex = a0 + p
                lo = max(0, e0 - int(BRAKE_LOOKBACK_M / grid_m))
                zone_brake = float(np.max(prof["brake"][lo:apex + 1]))
                min_thr = float(np.min(prof["thr"][e0:e1 + 1]))
                out.append({
                    "apex_m": float(prof["s"][apex]),
                    "start_m": float(prof["s"][e0]),
                    "end_m": float(prof["s"][e1]),
                    "radius_m": float(1.0 / a[apex]),
                    "heading_deg": heading,
                    "sign": int(sign[c0 + p]) if len(sign) else 0,
                    "min_speed_kmh": float(np.min(prof["v_kmh"][e0:e1 + 1])),
                    "brake_peak_pct": zone_brake,
                    "min_throttle_pct": min_thr,
                    "demand": ("braking" if zone_brake >= BRAKE_ZONE_PCT
                               else "flat" if min_thr >= FLAT_MIN_THROTTLE_PCT else "lift"),
                })
    out.sort(key=lambda c: c["apex_m"])
    return merge_close(out)


def merge_close(corners: list[dict], gap_m: float | None = None) -> list[dict]:
    """Neighbouring same-direction bits of one bend (a long right-hander whose
    curvature wobbles) are one corner: merge when the gap between them is short."""
    gap_m = MERGE_GAP_M if gap_m is None else gap_m
    out: list[dict] = []
    for c in corners:
        p = out[-1] if out else None
        if p and p["sign"] == c["sign"] and c["start_m"] - p["end_m"] < gap_m:
            tight = c if c["radius_m"] < p["radius_m"] else p
            out[-1] = {
                **tight,
                "start_m": p["start_m"], "end_m": c["end_m"],
                "heading_deg": p["heading_deg"] + c["heading_deg"],
                "min_speed_kmh": min(p["min_speed_kmh"], c["min_speed_kmh"]),
                "brake_peak_pct": max(p["brake_peak_pct"], c["brake_peak_pct"]),
                "min_throttle_pct": min(p["min_throttle_pct"], c["min_throttle_pct"]),
                "demand": "braking" if "braking" in (p["demand"], c["demand"])
                else "lift" if "lift" in (p["demand"], c["demand"]) else "flat",
            }
        else:
            out.append(c)
    return out


def consensus_corners(per_lap: list[list[dict]], lengths: list[float]) -> list[dict]:
    """Apexes seen in >= half of the laps at about the same distance -> the track's
    corner map, numbered in lap order (C1..Cn) with median geometry."""
    # cluster each turning direction on its own: the two halves of a chicane are close
    # together but opposite, and must never be averaged into one corner
    clusters: list[list[tuple]] = []
    for sign in (1, -1):
        pts = sorted(((c["apex_m"], i, c) for i, lst in enumerate(per_lap) for c in lst if c["sign"] == sign),
                     key=lambda x: x[0])
        group: list[list[tuple]] = []
        for p in pts:
            if group and p[0] - group[-1][-1][0] <= CLUSTER_GAP_M:
                group[-1].append(p)
            else:
                group.append([p])
        clusters += group
    clusters.sort(key=lambda cl: float(np.median([p[0] for p in cl])))
    n = max(1, len(per_lap))
    result = []
    for cl in clusters:
        laps = {p[1] for p in cl}
        if len(laps) / n < MIN_LAP_FRACTION or (n >= 2 and len(laps) < MIN_LAPS_SEEN):
            continue
        cs = [p[2] for p in cl]
        med = lambda key: float(np.median([c[key] for c in cs]))
        demands = [c["demand"] for c in cs]
        result.append({
            "apex_m": round(med("apex_m"), 1),
            "start_m": round(med("start_m"), 1),
            "end_m": round(med("end_m"), 1),
            "radius_m": round(med("radius_m")),
            "heading_deg": round(med("heading_deg")),
            "min_speed_kmh": round(med("min_speed_kmh")),
            "brake_peak_pct": round(med("brake_peak_pct")),
            "demand": max(set(demands), key=demands.count),
            "sign": int(np.sign(np.sum([c["sign"] for c in cs])) or 1),
            "seen_in_laps": f"{len(laps)}/{n}",
        })
    for i, c in enumerate(result, 1):
        c["n"] = i
    return result


def detect_track_corners(laps: list[LapTelemetry]) -> dict:
    """Corner map of a track from several complete laps (any cars)."""
    profiles = [p for p in (curvature_profile(l) for l in laps) if p]
    if not profiles:
        return {"available": False, "reason": "no usable laps (need Lap Dist, speed, yaw or lateral G)"}
    per_lap = [detect_apexes(p) for p in profiles]
    corners = consensus_corners(per_lap, [p["length_m"] for p in profiles])
    return {
        "available": True,
        "laps_used": len(profiles),
        "length_m": round(float(np.median([p["length_m"] for p in profiles]))),
        "curvature_source": sorted({p["source"] for p in profiles}),
        "corner_count": len(corners),
        "by_demand": {d: sum(1 for c in corners if c["demand"] == d) for d in ("braking", "lift", "flat")},
        "corners": corners,
    }


def gather_profiles(files: list[str], expected_length_m: float | None = None,
                    max_files: int = 10, max_laps: int = 40) -> tuple[list[dict], list[str], str]:
    """Curvature profiles of the clean laps found in recording files of ONE track.

    Returns (profiles, file names used, kind): kind is 'full' (complete laps of the usual
    length, within 7 % of the best), 'partial' (only unfinished laps covering >= 90 % of
    expected_length_m: their distance origin is the pit exit, so positions are unreliable)
    or 'none'. Read-only on the recordings."""
    import os
    from .telemetry import load_telemetry_file
    full, partial, used = [], [], []
    for p in sorted(files, key=os.path.getsize, reverse=True)[:max_files]:
        try:
            laps = load_telemetry_file(p)
        except Exception:
            continue
        for lap in laps:
            if not 40 < lap.lap_time_s < 600:
                continue
            prof = curvature_profile(lap)
            if not prof:
                continue
            (full if lap.is_complete else partial).append((float(lap.lap_time_s), prof))
        used.append(os.path.basename(p))
    if full:
        lens = [c[1]["length_m"] for c in full]
        modal = float(max(lens, key=lambda x: sum(abs(y / x - 1) < 0.015 for y in lens)))
        good = [c for c in full if abs(c[1]["length_m"] / modal - 1) < 0.02]
        best = min(c[0] for c in good)
        good = sorted((c for c in good if c[0] <= best * 1.07), key=lambda c: c[0])[:max_laps]
        return [c[1] for c in good], used, "full"
    if expected_length_m:
        ok = sorted((c for c in partial if c[1]["length_m"] >= 0.9 * expected_length_m),
                    key=lambda c: -c[1]["length_m"])
        if ok:
            return [c[1] for c in ok[:max_laps]], used, "partial"
    return [], used, "none"
