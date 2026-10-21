"""
Build the track reference (corners, straights, names) for LMU tracks. No LLM involved.

    python -I backend/tools/build_track_reference.py                 # every catalog track, cache only
    python -I backend/tools/build_track_reference.py --track fuji    # one track (slug, see track_catalog.py)
    python -I backend/tools/build_track_reference.py --fetch         # download OpenStreetMap/Wikipedia first
                                                                     # for tracks not cached yet
    python -I backend/tools/build_track_reference.py --refetch       # download again even if cached
    python -I backend/tools/build_track_reference.py --details       # print every corner

When to run it: when LMU adds a track (add one entry to backend/app/track_catalog.py, record 10+
clean laps, then run with --fetch), or after recording more laps (the alignment improves).

What it does per track: reads the cached OSM data (downloads once with --fetch), builds the
closed loops, extracts corners on the true centerline, aligns them with your recorded laps
(sequence alignment, see backend/app/track_reference.py), adds LMU/track_db names, and writes
data/track_reference/<slug>.json. Outside services are only used by --fetch/--refetch;
the app itself never calls them. Recordings are opened read-only.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402

from app import osm_source as osm_src  # noqa: E402
from app import track_corners as tc  # noqa: E402
from app import track_reference as tr  # noqa: E402
from app.track_catalog import CATALOG, find_entry  # noqa: E402
from app.track_centerline import loops_from_osm  # noqa: E402
from app.track_db import corner_index_for  # noqa: E402

SOURCES = REPO / "data" / "track_reference" / "sources"


def fetch(entry: dict, force: bool) -> dict | None:
    cached = osm_src.load_cached(SOURCES, entry["slug"])
    if cached and not force:
        if "circuits" not in cached:          # cache from before circuit relations were fetched
            try:
                lat, lon = cached["center"]
                cached["circuits"] = osm_src.fetch_circuits(lat, lon, cached.get("radius_m", osm_src.SEARCH_RADIUS_M))
                osm_src.save_cached(SOURCES, entry["slug"], cached)
            except osm_src.SourceError as e:
                print(f"   circuit relations not fetched: {e}")
        return cached
    try:
        lat, lon = entry.get("latlon") or osm_src.geocode(entry["query"])
        time.sleep(2)
        data = osm_src.fetch_osm(lat, lon, entry.get("radius_m", osm_src.SEARCH_RADIUS_M))
        time.sleep(2)
        data["wikipedia"] = osm_src.wikipedia_facts(entry["wiki"])
        data["query"] = entry["query"]
        osm_src.save_cached(SOURCES, entry["slug"], data)
        return data
    except osm_src.SourceError as e:
        print(f"   fetch failed: {e}")
        return cached


def recordings_by_slug(sessions: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(sessions, "*.duckdb"))):
        e = find_entry(os.path.basename(p).split("-")[0].replace("_", " "))
        if e:
            out[e["slug"]].append(p)
    return out


def build_one(entry: dict, files: list[str], do_fetch: bool, refetch: bool, details: bool) -> str:
    slug = entry["slug"]
    osm = fetch(entry, refetch) if (do_fetch or refetch or osm_src.load_cached(SOURCES, slug)) else None
    wiki = dict((osm or {}).get("wikipedia") or {})
    for k in ("turns", "length_m"):                       # official values the catalog supplies
        if not wiki.get(k) and entry.get(k):
            wiki[k] = entry[k]
    profs, used, kind = tc.gather_profiles(files, wiki.get("length_m"))
    if kind == "partial":
        profs = []                       # distances of unfinished laps do not start at the line
    rec_len = float(np.median([p["length_m"] for p in profs])) if profs else None
    det_all = None
    if profs:
        per_lap = [tc.detect_apexes(p) for p in profs]
        det_all = tc.consensus_corners(per_lap, [p["length_m"] for p in profs])
    db_points = tr.parse_track_db(corner_index_for(entry["registry"], "")) if entry.get("registry") else []
    sources = {"osm": "OpenStreetMap contributors (ODbL)" if osm else None, "wikipedia": wiki.get("title"),
               "lmu_track_db": bool(db_points), "recordings": [os.path.basename(f) for f in used]}

    loops, note = [], ""
    targets = [t for t in (wiki.get("length_m"), rec_len * 1.01 if rec_len else None) if t]
    if osm and targets:
        loops = loops_from_osm(osm, targets)[:4]
    if not loops:
        # no OpenStreetMap loop (or no OSM data): fall back to the recorded laps alone
        if det_all and len(profs) >= tr.MIN_LAPS_PROVISIONAL:
            layout = tr.build_recorded_layout(det_all, rec_len, len(profs), db_points, wiki)
            ref = {"schema": tr.SCHEMA, "slug": slug, "display_name": entry["registry"] or slug,
                   "built": __import__("datetime").date.today().isoformat(), "sources": sources,
                   "official": wiki, "layouts": [layout]}
            tr.save_reference(ref)
            return (f"{slug:12s} {layout['status']:11s} from recordings only (no OSM loop)  corners {len(layout['corners']):>2d} "
                    f"(official {wiki.get('turns', '?')})  laps {len(profs):>2d}"
                    + ("" if len(profs) >= tr.MIN_LAPS_VERIFIED else f"   needs {tr.MIN_LAPS_VERIFIED - len(profs)} more clean laps to verify")
                    + _details(layout, details))
        why = "no OpenStreetMap data (run with --fetch)" if not osm else (
            "no closed OSM loop of the right length" if targets else "no official length")
        return f"{slug:12s} nothing built: {why}, and fewer than {tr.MIN_LAPS_PROVISIONAL} clean recorded laps ({len(profs)})"

    per_layout = {}
    if det_all:
        for i, lp in enumerate(loops):
            if abs(lp.length_m - rec_len * 1.01) / lp.length_m < 0.05:
                per_layout[i] = (det_all, rec_len, len(profs))
    ref = tr.build_reference(slug, entry["registry"] or (osm or {}).get("query", slug), loops, per_layout, wiki,
                             db_points, sources)
    tr.save_reference(ref)
    best = max(ref["layouts"], key=lambda l: ({"verified": 2, "provisional": 1, "osm_only": 0}[l["status"]],
                                              -abs(l["length_m"] - (rec_len or l["length_m"]))))
    al = best.get("alignment") or {}
    return (f"{slug:12s} {best['status']:11s} layouts {len(ref['layouts'])}  corners {len(best['corners']):>2d} "
            f"(official {wiki.get('turns', '?')})  straights {len(best['straights'])}  laps {len(profs):>2d}"
            + (f"  matched {al['matched_pairs']}/{len(det_all)} det, pos err {al['median_position_error_m']} m, "
               f"scale {al['lap_scale']}" if al.get("matched_pairs") else "")
            + ("" if len(profs) >= tr.MIN_LAPS_VERIFIED else f"   needs {tr.MIN_LAPS_VERIFIED - len(profs)} more clean laps to verify")
            + _details(best, details))


def _details(layout: dict, details: bool) -> str:
    if not details:
        return ""
    lines = [""]
    for c in layout["corners"]:
        lm = c.get("lmu_apex_m")
        lines.append(f"      {c['order']:>2d} {c['display']:34s} {c['direction']:5s} {c['heading_deg']:>4d} deg R{c['radius_m']:>4d} "
                     f"lap {('%7.0f' % lm) if lm is not None else '      -'} m  {c['kind']}{'' if c.get('matched') else ' (unmatched)'}")
    for s in layout["straights"]:
        lines.append(f"      straight: {s['name']} {s['start_m']:.0f}-{s['end_m']:.0f} m")
    return "\n".join(lines)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", help="catalog slug (e.g. fuji, cota, sarthe); default: all")
    ap.add_argument("--fetch", action="store_true", help="download OSM/Wikipedia for tracks not cached yet")
    ap.add_argument("--refetch", action="store_true", help="download again even if cached")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--sessions", default=str(REPO / "data" / "sessions"))
    a = ap.parse_args()
    recs = recordings_by_slug(a.sessions)
    entries = [e for e in CATALOG if not a.track or e["slug"] == a.track]
    if not entries:
        print(f"unknown track '{a.track}'. Known: {', '.join(e['slug'] for e in CATALOG)}")
        return 1
    for e in entries:
        try:
            print(build_one(e, recs.get(e["slug"], []), a.fetch, a.refetch, a.details), flush=True)
        except Exception as ex:                       # one bad track must not stop the rest
            print(f"{e['slug']:12s} FAILED: {type(ex).__name__}: {ex}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
