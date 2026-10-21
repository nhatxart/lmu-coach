"""
Detect each track's corners from your recordings and compare with the official
corner count (read-only on data/sessions; writes data/track_corners only with --save).

    python -I backend/tools/detect_track_corners.py                  # every track, table only
    python -I backend/tools/detect_track_corners.py --track Fuji     # one track (name contains)
    python -I backend/tools/detect_track_corners.py --save           # also write data/track_corners/<track>.json
    python -I backend/tools/detect_track_corners.py --details        # list every detected corner

How it works: see backend/app/track_corners.py (curvature along lap distance).
Laps used per track: complete laps of the usual length, within 7 % of the best. Tracks that
only have partial laps fall back to laps covering >= 90 % of the official length and are
marked "partial". Where backend/app/track_db.py has one reference point per official turn
(COTA, Barcelona), detected apexes are also scored against them (precision / recall within
70 m).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402

from app import track_corners as tc  # noqa: E402
from app.telemetry import load_telemetry_file  # noqa: E402
from app.track_db import corner_index_for  # noqa: E402

# Official numbered turns and lap length, English Wikipedia infobox (10/19/2026).
# match = lower-case fragment of the file-name track name.
OFFICIAL = [
    {"match": "fuji", "turns": 16, "length_m": 4563},
    {"match": "road atlanta", "turns": 12, "length_m": 4088},
    {"match": "americas", "turns": 20, "length_m": 5513},
    {"match": "barcelona", "turns": 14, "length_m": 4657},
    {"match": "sarthe", "turns": 38, "length_m": 13626},
    {"match": "sebring", "turns": 17, "length_m": 6019},
    {"match": "algarve", "turns": 15, "length_m": 4653},
    {"match": "pace", "turns": 15, "length_m": 4309},        # Interlagos (file name has accents)
    {"match": "bahrain", "turns": 15, "length_m": 5412},
    {"match": "imola", "turns": 19, "length_m": 4909},
    {"match": "enzo e dino", "turns": 19, "length_m": 4909},
    {"match": "spa", "turns": 19, "length_m": 7004},
]
MAX_FILES, MAX_LAPS = 5, 8
SCORE_TOL_M = 70.0


def official_for(track: str) -> dict | None:
    t = track.lower()
    return next((o for o in OFFICIAL if o["match"] in t), None)


def slug(track: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", track.lower()).strip("_")


def score(apex: list[float], refs: list[float]) -> tuple[float, float]:
    used, tp = set(), 0
    for r in refs:
        cand = [(abs(a - r), i) for i, a in enumerate(apex) if i not in used and abs(a - r) <= SCORE_TOL_M]
        if cand:
            used.add(min(cand)[1])
            tp += 1
    return tp / max(1, len(apex)), tp / max(1, len(refs))


def pick_laps(files: list[str], off: dict | None) -> tuple[list, list[str], str]:
    """(profiles, file names used, 'full'|'partial'|'none')"""
    full, partial, used = [], [], []
    for p in sorted(files, key=os.path.getsize, reverse=True)[:MAX_FILES]:
        try:
            laps = load_telemetry_file(p)
        except Exception:
            continue
        for l in laps:
            if not 40 < l.lap_time_s < 600:
                continue
            prof = tc.curvature_profile(l)
            if not prof:
                continue
            item = (float(l.lap_time_s), prof, os.path.basename(p))
            (full if l.is_complete else partial).append(item)
        used.append(os.path.basename(p))
    if full:
        lens = [c[1]["length_m"] for c in full]
        modal = float(max(lens, key=lambda x: sum(abs(y / x - 1) < 0.015 for y in lens)))
        good = [c for c in full if abs(c[1]["length_m"] / modal - 1) < 0.02]
        best = min(c[0] for c in good)
        good = sorted((c for c in good if c[0] <= best * 1.07), key=lambda c: c[0])[:MAX_LAPS]
        return [c[1] for c in good], used, "full"
    if off:
        ok = [c for c in partial if c[1]["length_m"] >= 0.9 * off["length_m"]]
        ok.sort(key=lambda c: -c[1]["length_m"])
        if ok:
            return [c[1] for c in ok[:MAX_LAPS]], used, "partial"
    return [], used, "none"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", help="only tracks whose name contains this (case-insensitive)")
    ap.add_argument("--save", action="store_true", help="write data/track_corners/<track>.json")
    ap.add_argument("--details", action="store_true", help="print every detected corner")
    ap.add_argument("--sessions", default=str(REPO / "data" / "sessions"))
    a = ap.parse_args()

    by_track: dict[str, list[str]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(a.sessions, "*.duckdb"))):
        by_track[os.path.basename(p).split("-")[0].replace("_", " ")].append(p)

    print(f"{'track':34s} {'official':>8s} {'detected':>8s} {'laps':>4s}  {'brake/lift/flat':>15s}  vs track_db refs (P/R)   note")
    for track, files in sorted(by_track.items()):
        if a.track and a.track.lower() not in track.lower():
            continue
        off = official_for(track)
        profs, used, kind = pick_laps(files, off)
        name = track.encode("ascii", "replace").decode()[:34]
        if not profs:
            print(f"{name:34s} {off['turns'] if off else '?':>8} {'-':>8s} {0:>4d}  no complete lap (or no loadable file) in {len(files)} file(s)")
            continue
        per = [tc.detect_apexes(p) for p in profs]
        corners = tc.consensus_corners(per, [p["length_m"] for p in profs])
        by = {d: sum(c["demand"] == d for c in corners) for d in ("braking", "lift", "flat")}
        refs = [p["dist"] for p in corner_index_for(track, "") if "apex" in p["corner"].lower()]
        ref_txt = ""
        if len(refs) >= 10:
            pr, rc = score([c["apex_m"] for c in corners], refs)
            ref_txt = f"{pr:.2f}/{rc:.2f} ({len(refs)} refs)"
        note = ""
        if kind == "partial":
            cover = np.median([p["length_m"] for p in profs]) / off["length_m"] * 100
            note = f"PARTIAL laps, cover ~{cover:.0f}% of the track"
        elif len(profs) < 3:
            note = f"only {len(profs)} lap(s): corners may be missed"
        print(f"{name:34s} {off['turns'] if off else '?':>8} {len(corners):>8d} {len(profs):>4d}  "
              f"{by['braking']:>5d}/{by['lift']}/{by['flat']:<7d}  {ref_txt:22s}  {note}")
        if a.details:
            for c in corners:
                print(f"      {c['n']:>2d} apex {c['apex_m']:7.0f} m  {c['start_m']:6.0f}-{c['end_m']:<6.0f} R{c['radius_m']:>4d} m "
                      f"{c['heading_deg']:>4d} deg  min {c['min_speed_kmh']:>3d} km/h  {c['demand']:7s} seen {c['seen_in_laps']}")
        if a.save:
            out = REPO / "data" / "track_corners"
            out.mkdir(parents=True, exist_ok=True)
            res = {"track": track, "official_turns": off["turns"] if off else None, "laps_used": len(profs),
                   "lap_kind": kind, "length_m": round(float(np.median([p["length_m"] for p in profs]))),
                   "files": used, "verified": False, "corners": corners,
                   "note": "auto-detected from curvature; 'n' is order along the lap, not the official number"}
            (out / f"{slug(track)}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
