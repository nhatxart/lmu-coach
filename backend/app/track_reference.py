"""
Track reference: the corners, straights and names of a circuit, extracted once from
OpenStreetMap (+ track_db names + our recorded laps) and cached on disk, so the coach
never needs an outside source at run time and never needs an LLM to build it.

Build (by hand, when a track is new):  backend/tools/BUILD_TRACK_REFERENCE.ps1
Use (at run time):                     load_reference(track_name) / locate(ref, lap_dist_m)

Pipeline per layout (see build_reference):
  1. OSM loop  -> reference corners on the TRUE centerline (track_centerline.py): apex,
     extent, radius, heading change, left/right, and the OSM name/number when volunteers
     mapped it ("Turn 7" ways at COTA).
  2. Recorded laps -> detected corners in LMU lap distance (track_corners.py).
  3. Sequence alignment of the two lists (order, direction, size; position only as a
     check) fixes where the start/finish line is, the left/right convention of the yaw
     channel, and the lap-distance of every reference corner as LMU measures it. Distance
     alone is never trusted: the car's odometer drifts from the centerline over a lap.
  4. Names: OSM first, then track_db (LMU's own corner names). Display format:
        "Coca-Cola Corner (3)"   "Esses (3, 4, 5, 6)"   "Porsche Curves"  (unnumbered)

Status of a layout: "verified" (>= 10 clean laps aligned), "provisional" (3-9 laps),
"osm_only" (no usable laps: corner order is known, start/finish is not).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np

from . import track_corners as tc
from .track_centerline import Loop, classify_name

SCHEMA = 1
MIN_LAPS_VERIFIED = 10
MIN_LAPS_PROVISIONAL = 3
def _reference_dir() -> Path:
    """TRACK_REFERENCE_DIR, else /data/track_reference (the Docker mount), else the repo's data/."""
    import os
    env = os.environ.get("TRACK_REFERENCE_DIR")
    if env:
        return Path(env)
    docker = Path("/data/track_reference")
    return docker if docker.exists() else Path(__file__).resolve().parents[2] / "data" / "track_reference"


REFERENCE_DIR = _reference_dir()

# alignment costs (see _align)
GAP_BASE, GAP_PER_DEG = 0.3, 0.01
MATCH_PER_LOG_RATIO = 0.8
NEAR_M = 60.0                    # an event this close to a corner's extent is "at" it


# --------------------------------------------------------------------------- reference corners

def _loop_profile(loop: Loop) -> dict:
    n = len(loop.s)
    return {"s": loop.s, "k": loop.k, "v_kmh": np.zeros(n), "brake": np.zeros(n), "thr": np.full(n, 100.0),
            "length_m": loop.length_m, "source": "osm"}


def corners_from_loop(loop: Loop) -> tuple[list[dict], list[dict]]:
    """(corners, straights) on the loop, ordered along the loop. Each corner:
    apex_m/start_m/end_m (metres along the loop), radius_m, heading_deg, direction
    ('left'|'right' in the loop's travel direction), and OSM name/number if mapped."""
    found = tc.detect_apexes(_loop_profile(loop))
    corners: list[dict] = []
    for c in found:
        corners.append({
            "apex_m": round(c["apex_m"], 1), "start_m": round(c["start_m"], 1), "end_m": round(c["end_m"], 1),
            "radius_m": round(c["radius_m"]), "heading_deg": round(c["heading_deg"]),
            "direction": "left" if c["sign"] > 0 else "right", "kind": "corner", "osm_names": [],
        })

    straights: list[dict] = []
    for seg in loop.segments:
        kind, number, suffix = classify_name(seg.name)
        s0, s1 = seg.s0 % loop.length_m, seg.s1 % loop.length_m
        if kind == "straight":
            straights.append({"name": seg.name, "start_m": round(s0, 1), "end_m": round(s1, 1)})
            continue
        span = (seg.s1 - seg.s0) % loop.length_m or loop.length_m
        if kind in ("none", "layout") or (kind == "named" and span > 0.25 * loop.length_m):
            continue
        # a named / numbered stretch: attach to the corners whose apex falls inside it, or
        # make a corner of its own (a numbered kink the detector's heading limit drops)
        inside = [c for c in corners if _in_span(c["apex_m"], s0, s1, loop.length_m)]
        for c in inside:
            c["osm_names"].append(seg.name)
            if kind == "turn":
                c["number"], c["suffix"] = int(number), suffix
        if not inside:
            idx = _span_indices(s0, s1, loop.length_m)
            k = loop.k[idx]
            a = idx[int(np.argmax(np.abs(k)))]
            corners.append({
                "apex_m": round(float(loop.s[a]), 1), "start_m": round(s0, 1), "end_m": round(s1, 1),
                "radius_m": round(float(1.0 / max(abs(k).max(), 1e-6))),
                "heading_deg": round(float(np.degrees(np.sum(np.abs(k))))),
                "direction": "left" if k[int(np.argmax(np.abs(k)))] > 0 else "right",
                "kind": "kink", "osm_names": [seg.name],
                **({"number": int(number), "suffix": suffix} if kind == "turn" else {}),
            })
    corners.sort(key=lambda c: c["apex_m"])
    corners = _merge_same_osm_turn(corners)
    for c in corners:
        names = [n for n in c.pop("osm_names") if classify_name(n)[0] != "turn"]
        if names:
            c["osm_name"] = names[0]
    return corners, straights


def _merge_same_osm_turn(corners: list[dict]) -> list[dict]:
    """One numbered OSM turn is one official corner even if its curvature has two peaks."""
    out: list[dict] = []
    for c in corners:
        key = (c.get("number"), c.get("suffix"))
        p = out[-1] if out else None
        if p and key[0] is not None and (p.get("number"), p.get("suffix")) == key:
            tight = c if c["radius_m"] < p["radius_m"] else p
            out[-1] = {**p, "apex_m": tight["apex_m"], "radius_m": tight["radius_m"],
                       "end_m": c["end_m"], "heading_deg": p["heading_deg"] + c["heading_deg"],
                       "osm_names": p["osm_names"] + [n for n in c["osm_names"] if n not in p["osm_names"]]}
        else:
            out.append(c)
    return out


def _in_span(x: float, s0: float, s1: float, L: float) -> bool:
    return s0 <= x <= s1 if s0 <= s1 else (x >= s0 or x <= s1)


def _span_indices(s0: float, s1: float, L: float) -> np.ndarray:
    a, b = int(round(s0)), int(round(s1))
    n = int(L)
    return np.arange(a, b + 1) % n if a <= b else np.concatenate([np.arange(a, n), np.arange(0, b + 1)])


# --------------------------------------------------------------------------- alignment

def _gap(h: float) -> float:
    return GAP_BASE + GAP_PER_DEG * h


def _dp(ref: list[dict], det: list[dict], flip: int, pos_cost=None) -> tuple[float, list[tuple[int, int]]]:
    """Global alignment of two corner sequences. Returns (cost, [(ref_i, det_j), ...])."""
    n, m = len(ref), len(det)
    INF = 1e9
    dp = np.full((n + 1, m + 1), INF)
    bk = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, how = INF, 0
            if i > 0 and dp[i - 1, j] + _gap(ref[i - 1]["heading_deg"]) < best:
                best, how = dp[i - 1, j] + _gap(ref[i - 1]["heading_deg"]), 1
            if j > 0 and dp[i, j - 1] + _gap(det[j - 1]["heading_deg"]) < best:
                best, how = dp[i, j - 1] + _gap(det[j - 1]["heading_deg"]), 2
            if i > 0 and j > 0:
                r, d = ref[i - 1], det[j - 1]
                same_dir = (1 if r["direction"] == "left" else -1) * flip == d["sign"]
                if same_dir:
                    cost = MATCH_PER_LOG_RATIO * abs(np.log(max(r["heading_deg"], 10) / max(d["heading_deg"], 10)))
                    if pos_cost:
                        cost += pos_cost(i - 1, j - 1)
                    if dp[i - 1, j - 1] + cost < best:
                        best, how = dp[i - 1, j - 1] + cost, 3
            dp[i, j], bk[i, j] = best, how
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        how = bk[i, j]
        if how == 3:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif how == 1:
            i -= 1
        else:
            j -= 1
    return float(dp[n, m]), pairs[::-1]


def _affine(ref_s: np.ndarray, det_s: np.ndarray, L: float) -> tuple[float, float, float] | None:
    """ref_pos = a + b * lap_dist (mod L), robustly. Returns (a, b, median abs residual m)."""
    if len(ref_s) < 3:
        return None
    # unwrap ref positions so they increase with lap distance
    r = np.unwrap(ref_s * (2 * np.pi / L)) * (L / (2 * np.pi))
    keep = np.ones(len(r), bool)
    a = b = 0.0
    for _ in range(3):
        if keep.sum() < 3:
            return None
        A = np.column_stack([np.ones(keep.sum()), det_s[keep]])
        (a, b), *_ = np.linalg.lstsq(A, r[keep], rcond=None)
        res = np.abs(r - (a + b * det_s))
        thr = max(60.0, 2.5 * float(np.median(res[keep])))
        keep = res <= thr
    return float(a), float(b), float(np.median(np.abs(r - (a + b * det_s))[keep]))


def align(ref: list[dict], L_ref: float, det: list[dict], L_det: float, directed: bool) -> dict | None:
    """Best (direction, mirror, rotation) of the reference corner list against the detected
    one. Rotation = which reference corner comes first after the start/finish line."""
    if len(ref) < 3 or len(det) < 3:
        return None
    best = None
    directions = [1] if directed else [1, -1]
    for direction in directions:
        base = ref if direction == 1 else [
            {**c, "direction": "left" if c["direction"] == "right" else "right",
             "apex_m": (L_ref - c["apex_m"]) % L_ref} for c in reversed(ref)]
        for flip in (1, -1):
            for r in range(len(base)):
                rot = base[r:] + base[:r]
                cost, pairs = _dp(rot, det, flip)
                if len(pairs) < 3:
                    continue
                ref_s = np.array([rot[i]["apex_m"] for i, _ in pairs])
                det_s = np.array([det[j]["apex_m"] for _, j in pairs])
                fit = _affine(ref_s, det_s, L_ref)
                if fit is None:
                    continue
                a, b, resid = fit
                # score: alignment cost, minus credit per matched pair, plus a position check
                score = cost - 0.25 * len(pairs) + 0.02 * resid + 8.0 * abs(b * L_det / L_ref - 1.0)
                if best is None or score < best["score"]:
                    best = {"score": score, "cost": cost, "direction": direction, "flip": flip, "rotation": r, "rot": rot,
                            "pairs": [(base[(r + i) % len(base)], det[j]) for i, j in pairs],
                            "pair_idx": [((r + i) % len(base), j) for i, j in pairs],
                            "a": a, "b": b, "resid_m": resid, "base": base}
    if best:
        for _ in range(2):
            best = _refine(best, det, L_ref) or best
    return best


def _refine(best: dict, det: list[dict], L_ref: float) -> dict | None:
    """Second pass: with a first fit (start/finish and scale), pair the corners again, now also
    penalising pairs whose position disagrees with the fit (distance never decides alone, it
    breaks ties between look-alike corners such as the two halves of a chicane)."""
    rot, a, b = best["rot"], best["a"], best["b"]
    a_mod = a % L_ref

    def pos_cost(i, j):
        u = (rot[i]["apex_m"] - a_mod) % L_ref
        return min(1.5, abs(u / b - det[j]["apex_m"]) * 0.012)

    cost, pairs = _dp(rot, det, best["flip"], pos_cost)
    if len(pairs) < 3:
        return None
    fit = _affine(np.array([rot[i]["apex_m"] for i, _ in pairs]), np.array([det[j]["apex_m"] for _, j in pairs]), L_ref)
    if fit is None:
        return None
    n = len(best["base"])
    r0 = best["rotation"]
    return {**best, "cost": cost, "a": fit[0], "b": fit[1], "resid_m": fit[2],
            "pairs": [(best["base"][(r0 + i) % n], det[j]) for i, j in pairs],
            "pair_idx": [((r0 + i) % n, j) for i, j in pairs]}


# --------------------------------------------------------------------------- names (track_db)

_TDB = re.compile(r"^T(\d+)(?:\s*[/&-]\s*T?(\d+))?\s*(Apex|Braking|Exit|Approach|Entry)?\s*(?:\((.*)\))?", re.I)


def parse_track_db(index: list[dict]) -> list[dict]:
    """LMU/track_db reference points -> [{dist, numbers, name, note}] for the 'Apex' points
    and for named points without a role (e.g. 'Tertre Rouge Apex')."""
    out = []
    for p in index:
        label = p["corner"]
        m = _TDB.match(label)
        if m:
            nums = [int(x) for x in (m.group(1), m.group(2)) if x]
            role = (m.group(3) or "").lower()
            inner = (m.group(4) or "").strip()
            if role not in ("apex", ""):
                continue
            name, _, note = inner.partition(" - ")
            out.append({"dist": float(p["dist"]), "numbers": nums, "name": name.strip() or None,
                        "note": note.strip() or None, "label": label})
        elif re.search(r"apex", label, re.I):
            name = re.sub(r"\s*apex.*$", "", label, flags=re.I).strip()
            out.append({"dist": float(p["dist"]), "numbers": [], "name": name or None, "note": None, "label": label})
    return out


# --------------------------------------------------------------------------- build

def _display(c: dict) -> str:
    nums = c.get("numbers") or ([c["number"]] if c.get("number") else [])
    nums_txt = ", ".join(f"{n}{c.get('suffix') or ''}" if len(nums) == 1 else str(n) for n in nums)
    name = c.get("name")
    if name and nums_txt:
        return f"{name} ({nums_txt})"
    if name:
        return name
    if nums_txt:
        return f"Turn {nums_txt}" if len(nums) == 1 else f"Turns {nums_txt}"
    return "Unnamed corner"


def build_layout(loop: Loop, det: list[dict] | None, det_len: float | None, laps_used: int,
                 db_points: list[dict], wiki: dict) -> dict:
    corners, straights = corners_from_loop(loop)
    for c in corners:
        if c.get("osm_name"):
            c["name"], c["name_source"] = c["osm_name"], "osm"
    status = "osm_only"
    alignment = None
    al = None
    if det and det_len and laps_used >= MIN_LAPS_PROVISIONAL:
        al = align(corners, loop.length_m, det, det_len, loop.directed)
    if al:
        status = "verified" if laps_used >= MIN_LAPS_VERIFIED else "provisional"
        L = loop.length_m
        base = al["base"]
        a, b = al["a"], al["b"]
        # reorder from the start/finish line; positions as LMU measures them
        matched = {}
        for (bi, dj) in al["pair_idx"]:
            matched.setdefault(bi, []).append(dj)
        ordered = []
        a_mod = a % L
        for k, bc in enumerate(base):
            c = dict(bc)
            c["_u"] = (c["apex_m"] - a_mod) % L          # reference metres after the start/finish line
            if k in matched:
                dets = [det[j] for j in matched[k]]
                c["lmu_apex_m"] = round(float(np.median([d["apex_m"] for d in dets])), 1)
                c["lmu_start_m"] = round(min(d["start_m"] for d in dets), 1)
                c["lmu_end_m"] = round(max(d["end_m"] for d in dets), 1)
                c["demand"] = dets[0]["demand"]
                c["matched"] = True
            else:
                c["matched"] = False
            ordered.append(c)
        # Corners the recordings did not see on their own (flat kinks, parts of a merged
        # bend): place them between their matched neighbours (piecewise linear), which
        # follows the odometer drift locally instead of with one global scale.
        known = sorted((c["_u"], c["lmu_apex_m"]) for c in ordered if c["matched"])
        ku, kl = np.array([k[0] for k in known]), np.array([k[1] for k in known])
        for c in ordered:
            if not c["matched"]:
                u = c["_u"]
                if u <= ku[0]:
                    pos = kl[0] - (ku[0] - u) / b
                elif u >= ku[-1]:
                    pos = kl[-1] + (u - ku[-1]) / b
                else:
                    pos = float(np.interp(u, ku, kl))
                c["lmu_apex_m"] = round(float(pos), 1)
        lap_len = float(det_len)
        for c in ordered:
            c.pop("_u", None)
            c["lmu_apex_m"] = round(float(c["lmu_apex_m"] % lap_len), 1)     # keep inside one lap
        ordered.sort(key=lambda c: c["lmu_apex_m"])                         # lap order from the line
        corners = ordered
        straights = _rebase(straights, a, b, L, al["direction"])
        alignment = {"laps_used": laps_used, "ref_corners": len(base), "detected": len(det),
                     "matched_pairs": len(al["pairs"]), "start_finish_ref_m": round(a % L, 1),
                     "lap_scale": round(b, 4), "median_position_error_m": round(al["resid_m"], 1),
                     "mirror": al["flip"] == -1, "direction": "osm" if al["direction"] == 1 else "reversed",
                     "alignment_cost": round(al["cost"], 2)}
        _attach_track_db(corners, db_points)
        _infer_numbers(corners, wiki.get("turns"))
    else:
        for c in corners:
            c["lmu_apex_m"] = None
            c["matched"] = False
        # start/finish unknown: list the corners from Turn 1 when OSM numbered them
        firsts = [i for i, c in enumerate(corners) if c.get("number") == 1]
        if firsts:
            corners = corners[firsts[0]:] + corners[:firsts[0]]
    for i, c in enumerate(corners, 1):
        c["order"] = i
    _group_complexes(corners)
    for c in corners:
        c["display"] = _display(c)
    net = float(np.sum(loop.k) * 1.0)
    return {
        "label": next((n for n in loop.names if classify_name(n)[0] == "layout"), None) or "Main circuit",
        "length_m": round(loop.length_m), "turning_direction": "counterclockwise" if net > 0 else "clockwise",
        "direction_from_osm": loop.directed, "status": status, "alignment": alignment,
        "official": {k: v for k, v in wiki.items() if k in ("turns", "length_m")},
        "corners": corners, "straights": straights,
        "complexes": [c for c in _complex_summary(corners)],
    }


def build_recorded_layout(det: list[dict], det_len: float, laps_used: int, db_points: list[dict], wiki: dict) -> dict:
    """A layout built from the recorded laps alone, for tracks whose OpenStreetMap data gives no
    closed loop (Le Mans: the public-road sections are not linked). Corners are the detected ones
    in LMU lap distance, so no alignment is needed; names/numbers come from track_db by
    position. Fewer details than an OSM layout (no true radii, no kinks the cars take flat)."""
    corners = []
    for d in det:
        corners.append({
            "kind": "corner", "direction": "left" if d["sign"] > 0 else "right", "heading_deg": d["heading_deg"],
            "radius_m": d["radius_m"], "apex_m": d["apex_m"], "start_m": d["start_m"], "end_m": d["end_m"],
            "lmu_apex_m": d["apex_m"], "lmu_start_m": d["start_m"], "lmu_end_m": d["end_m"],
            "demand": d["demand"], "matched": True, "name_source": None})
    _attach_track_db(corners, db_points)
    _infer_numbers(corners, wiki.get("turns"))
    for i, c in enumerate(corners, 1):
        c["order"] = i
    _group_complexes(corners)
    for c in corners:
        c["display"] = _display(c)
    return {
        "label": "Recorded laps (no usable OpenStreetMap loop)", "length_m": round(det_len),
        "turning_direction": None, "direction_from_osm": False,
        "status": "verified" if laps_used >= MIN_LAPS_VERIFIED else "provisional" if laps_used >= MIN_LAPS_PROVISIONAL else "osm_only",
        "alignment": {"laps_used": laps_used, "source": "recordings only"},
        "official": {k: v for k, v in wiki.items() if k in ("turns", "length_m")},
        "corners": corners, "straights": [], "complexes": _complex_summary(corners),
    }

def _rebase(straights, a, b, L, direction):
    return straights  # straights stay in loop metres (reference frame); kept for the coach's wording


def _attach_track_db(corners: list[dict], db: list[dict]) -> None:
    """Names/numbers from LMU's own reference points (track_db), by LMU lap distance.

    Each point is given to the nearest not-yet-named corner(s) within 150 m, matched
    corners first. A point with several numbers ('T9/T10 Apex (Dunlop Chicane)') takes that
    many adjacent corners (one number each); when fewer corners are found, the one corner
    carries all the numbers (a long bend the recording sees as one, like 100R = T4/T5)."""
    pos = [c for c in corners if c.get("lmu_apex_m") is not None]
    if not pos:
        return
    taken: set[int] = set()
    for p in sorted(db, key=lambda p: p["dist"]):
        near = sorted((c for c in pos if id(c) not in taken and abs(c["lmu_apex_m"] - p["dist"]) <= 150),
                      key=lambda c: (not c.get("matched"), abs(c["lmu_apex_m"] - p["dist"])))
        if not near:
            continue
        want = max(1, len(p["numbers"]))
        group = sorted(near[:want], key=lambda c: c["lmu_apex_m"])
        if len(group) > 1 and group[-1]["lmu_apex_m"] - group[0]["lmu_apex_m"] > 200:
            group = group[:1]            # not one complex: only the nearest corner is this point
        for k, c in enumerate(group):
            if p["name"]:
                if c.get("osm_name") and c["osm_name"] != p["name"]:
                    c.setdefault("aliases", []).append(c["osm_name"])    # keep OSM's wording
                c["name"], c["name_source"] = p["name"], "lmu"
            if p["numbers"]:
                if len(group) == len(p["numbers"]):
                    c["numbers"] = [p["numbers"][k]]
                else:
                    c["numbers"] = list(p["numbers"])
                c["number_source"] = "lmu"
            taken.add(id(c))


def _infer_numbers(corners: list[dict], turns: int | None) -> None:
    """Number the corners that have no number between two numbered ones, but only when the
    count fits exactly (gap of N-1 numbers, N-1 unnumbered corners in order). Also fills the
    ends of the lap from the official corner count. Marked number_source='inferred'."""
    def nums(c):
        return c.get("numbers") or ([c["number"]] if c.get("number") else [])
    n = len(corners)
    anchors = [i for i, c in enumerate(corners) if nums(c)]
    if not anchors:
        return
    spans = [(anchors[k], anchors[k + 1]) for k in range(len(anchors) - 1)]
    if turns:
        spans.append((anchors[-1], anchors[0] + n))          # wraps around the lap
    for a, b in spans:
        lo = max(nums(corners[a % n]))
        hi = min(nums(corners[b % n])) if b < n else (min(nums(corners[b % n])) + (turns or 0))
        between = [corners[i % n] for i in range(a + 1, b) if not nums(corners[i % n])]
        if between and hi - lo - 1 == len(between):
            for k, c in enumerate(between, 1):
                v = lo + k
                c["numbers"] = [((v - 1) % turns) + 1] if turns else [v]
                c["number_source"] = "inferred"

def _group_complexes(corners: list[dict]) -> None:
    """Consecutive corners that carry the same name form a complex (numbers comma-listed)."""
    i = 0
    while i < len(corners):
        j = i
        nm = corners[i].get("name")
        while nm and j + 1 < len(corners) and corners[j + 1].get("name") == nm:
            j += 1
        if j > i:
            nums = []
            for c in corners[i:j + 1]:
                nums += c.get("numbers") or ([c["number"]] if c.get("number") else [])
            for c in corners[i:j + 1]:
                c["complex"] = nm
                c["numbers"] = sorted(set(nums))
        i = j + 1


def _complex_summary(corners: list[dict]) -> list[dict]:
    out, seen = [], set()
    for c in corners:
        nm = c.get("complex")
        if nm and nm not in seen:
            seen.add(nm)
            parts = [x["order"] for x in corners if x.get("complex") == nm]
            out.append({"name": nm, "parts": parts, "numbers": c.get("numbers", [])})
    return out


def build_reference(slug: str, display_name: str, loops: list[Loop], per_layout_det: dict, wiki: dict,
                    db_points: list[dict], sources: dict) -> dict:
    """per_layout_det: {loop_index: (detected_corners, median_lap_length, laps_used)}"""
    layouts = []
    for i, loop in enumerate(loops):
        det, dl, n = per_layout_det.get(i, (None, None, 0))
        layouts.append(build_layout(loop, det, dl, n, db_points, wiki))
    return {"schema": SCHEMA, "slug": slug, "display_name": display_name, "built": date.today().isoformat(),
            "sources": sources, "official": wiki, "layouts": layouts}


# --------------------------------------------------------------------------- use at run time

def reference_path(slug: str) -> Path:
    return REFERENCE_DIR / f"{slug}.json"


def save_reference(ref: dict) -> Path:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    p = reference_path(ref["slug"])
    p.write_text(json.dumps(ref, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


_CACHE: dict[str, dict | None] = {}


def load_reference(track_name: str, lap_length_m: float | None = None) -> dict | None:
    """The best layout of this track's reference for an analysed lap, or None.
    Picks the aligned layout whose length is closest to the lap length when given."""
    from .track_catalog import find_entry
    entry = find_entry(track_name)
    if not entry:
        return None
    slug = entry["slug"]
    if slug not in _CACHE:
        p = reference_path(slug)
        _CACHE[slug] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    ref = _CACHE[slug]
    if not ref:
        return None
    usable = [l for l in ref["layouts"] if l["status"] in ("verified", "provisional")]
    if not usable:
        return None
    if lap_length_m:
        usable.sort(key=lambda l: abs(l["length_m"] - lap_length_m))
    return {"slug": slug, "display_name": ref["display_name"], "layout": usable[0]}


# Keyed by the sources file's path (so a test's TRACK_REFERENCE_DIR does not
# read another directory's answer), holding (mtime_ns, origin), like the
# topography loader. PREVIOUSLY a miss was cached for good: when
# BUILD_TRACK_REFERENCE wrote the sources file while the backend was up,
# that track stayed in the per-file frame until a restart. A missing file
# is not cached (one stat per call), so it is seen as soon as it appears.
_ORIGIN_CACHE: dict[str, tuple[int, tuple[float, float] | None]] = {}


def track_origin(track_name: str) -> tuple[float, float] | None:
    """The fixed (lat, lon) every recording of this track is projected
    around: sources/<slug>.json "center", the OSM query centre the build
    used. None when there is no such file or key, or it cannot be read.

    The slug is the catalog's, else the plain name with "_" for spaces
    ("Synthetic Hill Ring" -> synthetic_hill_ring), so a track outside the
    catalog shares a frame as soon as a sources file with a centre exists.
    The directory is resolved per call (TRACK_REFERENCE_DIR), like the
    topography files."""
    from .track_catalog import _plain, find_entry
    entry = find_entry(track_name)
    slug = entry["slug"] if entry else _plain(track_name).replace(" ", "_")
    if not slug:
        return None
    p = _reference_dir() / "sources" / f"{slug}.json"
    key = str(p)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        _ORIGIN_CACHE.pop(key, None)
        return None                   # no sources file (yet)
    hit = _ORIGIN_CACHE.get(key)
    if hit is None or hit[0] != mtime:
        origin = None
        try:
            c = json.loads(p.read_text(encoding="utf-8")).get("center")
            lat, lon = float(c[0]), float(c[1])
            if np.isfinite(lat) and np.isfinite(lon):
                origin = (lat, lon)
        except Exception:
            origin = None             # malformed, or no usable centre
        hit = (mtime, origin)
        _ORIGIN_CACHE[key] = hit
    return hit[1]


def locate_ex(layout: dict, dist_m: float) -> tuple[str, str] | None:
    """(label, place) for an event at LMU lap distance dist_m. `place` names the corner or the
    gap without the metre offset, so two events in the same corner compare equal.
    label e.g. 'Coca-Cola Corner (3)', 'Esses (3, 4, 5) (+25 m)', 'between X and Y'."""
    cs = [c for c in layout["corners"] if c.get("lmu_apex_m") is not None]
    if not cs:
        return None
    cs.sort(key=lambda c: c["lmu_apex_m"])
    for c in cs:
        s0 = c.get("lmu_start_m", c["lmu_apex_m"] - 20)
        s1 = c.get("lmu_end_m", c["lmu_apex_m"] + 20)
        if s0 - 15 <= dist_m <= s1 + 15:
            off = dist_m - c["lmu_apex_m"]
            return (c["display"] if abs(off) < 15 else f"{c['display']} ({off:+.0f} m)"), c["display"]
    before = [c for c in cs if c["lmu_apex_m"] <= dist_m]
    after = [c for c in cs if c["lmu_apex_m"] > dist_m]
    if before and after:
        gap = f"between {before[-1]['display']} and {after[0]['display']}"
        return gap, gap
    return None


def locate(layout: dict, dist_m: float) -> str | None:
    r = locate_ex(layout, dist_m)
    return r[0] if r else None


def annotate_report(report: dict, ref: dict) -> None:
    """Put the reference's names on a grip report: every event and analysed corner gets a
    'corner' name (and 'corner_end' when an event runs into the next corner), plus a short
    'track_reference' summary (layout, complexes, straights) the coach can quote."""
    layout = ref["layout"]
    cs = sorted((c for c in layout["corners"] if c.get("lmu_apex_m") is not None), key=lambda c: c["lmu_apex_m"])
    report["corner_names_available"] = True
    report["corner_index"] = [[round(c["lmu_apex_m"]), c["display"]] for c in cs]
    for key in ("abs_events", "tc_events", "understeer_candidates"):
        for ev in report.get(key, []):
            start, end = ev.get("lap_dist_start_m"), ev.get("lap_dist_end_m")
            a = locate_ex(layout, start) if start is not None else None
            if not a:
                continue
            ev["corner"] = a[0]
            b = locate_ex(layout, end) if end is not None else None
            if b and b[1] != a[1]:
                ev["corner_end"] = b[0]
    for cf in (report.get("corner_analysis") or {}).get("corners", []):
        if cf.get("mrp_dist_m") is not None:
            a = locate_ex(layout, cf["mrp_dist_m"])
            if a:
                cf["corner"] = a[0]
    for v in (report.get("slip_angle_analysis") or {}).values():
        if isinstance(v, dict) and v.get("lap_dist_m") is not None:
            a = locate_ex(layout, v["lap_dist_m"])
            if a:
                v["corner"] = a[0]
    report["track_reference"] = {
        "layout": layout.get("label"), "status": layout["status"],
        "official_corner_count": (layout.get("official") or {}).get("turns"),
        "complexes": [{"name": c["name"], "numbers": c["numbers"],
                       "parts": [cs_["display"] for cs_ in layout["corners"] if cs_.get("complex") == c["name"]]}
                      for c in layout.get("complexes", [])],
        "straights": [s["name"] for s in layout.get("straights", []) if s.get("name")],
    }