"""
Turn cached OpenStreetMap ways into closed circuit loops with true centerline geometry.

An OSM circuit is a pile of `highway=raceway` ways: the main loop, alternative layouts,
the pit lane, kart tracks, connectors. Here we
  1. keep only race-track ways (no karting, pit lane, drag strip),
  2. join them end to end into every closed loop (a layout is a closed loop),
  3. keep the loops whose length is plausible for the track (official length, or the
     length of our recorded laps),
  4. resample each loop on a 1 m grid and compute its signed curvature (+ = left turn),
  5. record which OSM name each stretch of the loop carries ("Turn 7", "Eau Rouge",
     "Kemmel Straight", "Grand Prix Circuit"...), so corner names/numbers mapped by
     OSM volunteers come along with their true extent.

Only this module understands OSM's quirks. Everything else sees a Loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

EARTH_M_PER_DEG = 111_320.0
GRID_M = 1.0
SMOOTH_M = 25
MAX_CYCLES = 4000
MAX_EDGES_PER_CYCLE = 60
CYCLE_SEARCH_SECONDS = 40.0      # hard limits so a tangle of layouts cannot hang the build
CYCLE_SEARCH_STEPS = 3_000_000

_EXCLUDE_NAME = re.compile(r"\bpit\s*-?lane|pitlane|pit\s*(exit|entry)|circuit\s*pit|drag|kart|paddock road|service", re.I)
_EXCLUDE_ALWAYS = re.compile(r"drag|kart", re.I)
LAYOUT_WORDS = re.compile(r"\b(circuit|layout|course|loop|track|configuration|grand prix|gp|national|international|"
                          r"endurance|classic|oval|outer|inner|short|long|school|club|historic)\b", re.I)
STRAIGHT_WORDS = re.compile(r"straight|ligne droite|rettifilo|rettilineo|gerade|recta|reta\b|rechte", re.I)
TURN_NAME = re.compile(r"^\s*(?:turn|corner|t|curva|virage|kurve|curve)\s*(\d+)\s*([a-z]?)\s*$", re.I)


@dataclass
class Segment:
    """A stretch of a loop that carries one OSM name (or none)."""
    name: str | None
    s0: float
    s1: float
    way_ids: list[int] = field(default_factory=list)


@dataclass
class Loop:
    way_ids: list[int]
    length_m: float
    s: np.ndarray                 # 1 m grid along the loop
    x: np.ndarray                 # local metres east
    y: np.ndarray                 # local metres north
    k: np.ndarray                 # signed curvature 1/m, + = left turn, smoothed
    segments: list[Segment]
    directed: bool                # True when OSM gave the direction of travel (oneway)
    names: list[str]              # distinct OSM names on this loop (for the layout label)
    lat0: float
    lon0: float
    layout_name: str | None = None  # name of the OSM circuit relation it came from, if any


def _xy(lat, lon, lat0, lon0):
    return ((np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * EARTH_M_PER_DEG,
            (np.asarray(lat) - lat0) * EARTH_M_PER_DEG)


def _key(p) -> tuple[int, int]:
    return (round(p["lat"] * 1e7), round(p["lon"] * 1e7))


def usable_ways(ways: list[dict], raceway_only: bool = True) -> list[dict]:
    """Ways that can be part of a race loop. Ways that belong to a circuit relation are
    accepted whatever their highway tag (Le Mans uses public roads)."""
    out = []
    for w in ways:
        t = w.get("tags", {})
        if len(w.get("geometry", [])) < 2 or not t.get("highway"):
            continue
        if raceway_only and t.get("highway") != "raceway":
            continue
        if t.get("sport") == "karting" or t.get("abandoned"):
            continue
        name = t.get("name", "") or ""
        # ways of an OSM circuit relation are trusted (they define the layout); only the
        # obviously-not-a-circuit ones are dropped there
        if (_EXCLUDE_NAME if raceway_only else _EXCLUDE_ALWAYS).search(name):
            continue
        out.append(w)
    return out


def _poly_len(pts: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def build_edges(ways: list[dict], lat0: float, lon0: float, raceway_only: bool = True) -> list[dict]:
    """Directed edges. Ways with oneway=yes run start->end (the real direction of travel);
    oneway=-1 runs end->start; anything else can be driven either way."""
    edges = []
    for w in usable_ways(ways, raceway_only):
        g = w["geometry"]
        x, y = _xy([p["lat"] for p in g], [p["lon"] for p in g], lat0, lon0)
        pts = np.column_stack([x, y])
        ow = w["tags"].get("oneway", "")
        fwd = {"id": w["id"], "name": w["tags"].get("name"), "a": _key(g[0]), "b": _key(g[-1]),
               "pts": pts, "len": _poly_len(pts), "directed": ow in ("yes", "-1", "true", "1")}
        rev = {**fwd, "a": fwd["b"], "b": fwd["a"], "pts": pts[::-1]}
        if ow in ("yes", "true", "1"):
            edges.append(fwd)
        elif ow == "-1":
            edges.append(rev)
        else:
            edges += [fwd, rev]
    return edges


def find_cycles(edges: list[dict], min_len: float, max_len: float) -> list[list[dict]]:
    """All simple closed walks (no node visited twice) whose length is within bounds, each
    reported once. Pruned so overlapping layouts do not explode: a cycle is only searched
    from its smallest node, and a branch is dropped when even the shortest way back to the
    start would exceed max_len."""
    import heapq
    import time as _time
    out_edges: dict = {}
    in_edges: dict = {}
    for e in edges:
        out_edges.setdefault(e["a"], []).append(e)
        in_edges.setdefault(e["b"], []).append(e)
    seen: set = set()
    cycles: list[list[dict]] = []
    deadline = _time.monotonic() + CYCLE_SEARCH_SECONDS
    budget = [CYCLE_SEARCH_STEPS]

    def dist_back(start) -> dict:
        """Shortest distance from every node to `start` (Dijkstra on reversed edges)."""
        dist = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, 1e18):
                continue
            for e in in_edges.get(n, []):
                nd = d + e["len"]
                if nd < dist.get(e["a"], 1e18) and nd <= max_len:
                    dist[e["a"]] = nd
                    heapq.heappush(heap, (nd, e["a"]))
        return dist

    def dfs(start, node, path, visited, length, back):
        budget[0] -= 1
        if budget[0] < 0 or len(cycles) >= MAX_CYCLES or _time.monotonic() > deadline:
            return
        for e in out_edges.get(node, []):
            nl = length + e["len"]
            if nl > max_len:
                continue
            if e["b"] == start:
                if nl >= min_len and len(path) + 1 <= MAX_EDGES_PER_CYCLE:
                    canon = frozenset((x["id"], x["a"]) for x in path + [e])
                    if canon not in seen:
                        seen.add(canon)
                        cycles.append(path + [e])
                continue
            if e["b"] in visited or e["b"] < start or len(path) >= MAX_EDGES_PER_CYCLE:
                continue
            if nl + back.get(e["b"], 1e18) > max_len:
                continue
            visited.add(e["b"])
            dfs(start, e["b"], path + [e], visited, nl, back)
            visited.discard(e["b"])

    for start in sorted(out_edges):
        dfs(start, start, [], {start}, 0.0, dist_back(start))
        if budget[0] < 0 or _time.monotonic() > deadline:
            break
    return cycles

def _periodic_curve(pts: np.ndarray, smooth_m: float = 1.0):
    """Closed smoothing spline through a loop of points; returns dense x, y, signed curvature."""
    from scipy.interpolate import splev, splprep
    keep = np.concatenate(([True], np.hypot(*np.diff(pts, axis=0).T) > 0.05))
    p = pts[keep]
    if np.hypot(*(p[0] - p[-1])) > 0.05:
        p = np.vstack([p, p[0]])
    else:
        p[-1] = p[0]
    tck, _ = splprep([p[:, 0], p[:, 1]], s=len(p) * smooth_m ** 2, per=True, k=3)
    u = np.linspace(0, 1, max(4000, int(len(p) * 40)), endpoint=False)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.maximum(dx * dx + dy * dy, 1e-12) ** 1.5
    return np.asarray(x), np.asarray(y), np.asarray(kappa)


def _smooth_circular(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return a
    k = np.ones(n) / n
    pad = np.concatenate([a[-n:], a, a[:n]])
    return np.convolve(pad, k, mode="same")[n:-n]


def make_loop(cycle: list[dict], lat0: float, lon0: float) -> Loop:
    pts = np.vstack([e["pts"][:-1] for e in cycle] + [cycle[-1]["pts"][-1:]])
    xs, ys, kap = _periodic_curve(pts)
    seg_len = np.hypot(np.diff(xs, append=xs[0]), np.diff(ys, append=ys[0]))
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    s = np.arange(0.0, total, GRID_M)
    x = np.interp(s, cum, np.concatenate([xs, xs[:1]]))
    y = np.interp(s, cum, np.concatenate([ys, ys[:1]]))
    k = _smooth_circular(np.interp(s, cum, np.concatenate([kap, kap[:1]])), int(SMOOTH_M / GRID_M))

    # where each edge starts along the smooth loop: nearest dense sample to its first vertex
    dense = np.column_stack([xs, ys])
    segs: list[Segment] = []
    bounds = []
    for e in cycle:
        i = int(np.argmin(np.hypot(dense[:, 0] - e["pts"][0, 0], dense[:, 1] - e["pts"][0, 1])))
        bounds.append(float(cum[i]))
    bounds.append(bounds[0] + total)
    for e, b0, b1 in zip(cycle, bounds[:-1], bounds[1:]):
        if segs and segs[-1].name == e["name"] and (e["name"] is not None):
            segs[-1].s1 = b1
            segs[-1].way_ids.append(e["id"])
        else:
            segs.append(Segment(e["name"], b0, b1, [e["id"]]))
    # The loop's start is arbitrary (an OSM way boundary). Re-origin it in the straightest
    # 100 m so no corner is ever cut by the seam; segment spans are shifted to match.
    win = 100
    kc = np.abs(k)
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([kc, kc[:win]]))])
    quiet = cs[win:win + len(kc)] - cs[:len(kc)]
    origin = (int(np.argmin(quiet)) + win // 2) % len(kc)
    shift = float(s[origin])
    k, x, y = np.roll(k, -origin), np.roll(x, -origin), np.roll(y, -origin)
    for sg in segs:
        ln = sg.s1 - sg.s0
        sg.s0 = (sg.s0 - shift) % total
        sg.s1 = sg.s0 + ln
    names = sorted({e["name"] for e in cycle if e["name"]})
    return Loop([e["id"] for e in cycle], total, s, x, y, k, segs,
                all(e["directed"] for e in cycle), names, lat0, lon0)


def classify_name(name: str | None) -> tuple[str, str | None, str | None]:
    """(kind, number, suffix): kind is 'turn' for 'Turn 7'/'Turn 10A', 'straight',
    'layout' for names that only label a layout, 'named' for anything else, 'none'."""
    if not name:
        return "none", None, None
    m = TURN_NAME.match(name)
    if m:
        return "turn", m.group(1), (m.group(2) or "").upper() or None
    if STRAIGHT_WORDS.search(name):
        return "straight", None, None
    if LAYOUT_WORDS.search(name) and len(name.split()) <= 5:
        return "layout", None, None
    return "named", None, None


def _loops_from_edges(edges, lat0, lon0, targets, tolerance, layout_name=None) -> list[tuple[float, Loop]]:
    if not edges:
        return []
    lo, hi = min(targets) * (1 - tolerance), max(targets) * (1 + tolerance)
    scored = []
    for c in find_cycles(edges, lo, hi):
        L = sum(e["len"] for e in c)
        d = min(abs(L - t) / t for t in targets)
        if d <= tolerance:
            scored.append((d, c))
    scored.sort(key=lambda x: x[0])
    out = []
    for d, c in scored[:12]:
        lp = make_loop(c, lat0, lon0)
        lp.layout_name = layout_name
        out.append((d, lp))
    return out


def loops_from_osm(osm: dict, target_lengths_m: list[float], tolerance: float = 0.07) -> list[Loop]:
    """Closed loops of the cached OSM data whose length is within `tolerance` of any target,
    best match first. Looked for in (a) all race-track ways, and (b) the ways of each OSM
    circuit relation (one per layout; also finds loops that use public roads, like Le Mans).
    The caller decides which loop is the layout (with recorded laps)."""
    if not target_lengths_m:
        return []
    lat0, lon0 = osm["center"]
    found: list[tuple[float, Loop]] = _loops_from_edges(
        build_edges(osm["ways"], lat0, lon0), lat0, lon0, target_lengths_m, tolerance)
    circ = osm.get("circuits") or {}
    by_id = {w["id"]: w for w in circ.get("ways", [])}
    for rel in circ.get("relations", []):
        ids = [m["ref"] for m in rel.get("members", []) if m.get("type") == "way" and m["ref"] in by_id]
        ways = [by_id[i] for i in ids]
        name = (rel.get("tags") or {}).get("name")
        found += _loops_from_edges(build_edges(ways, lat0, lon0, raceway_only=False), lat0, lon0,
                                   target_lengths_m, tolerance, layout_name=name)
    found.sort(key=lambda x: x[0])
    loops: list[Loop] = []
    for d, lp in found:
        ids = set(lp.way_ids)
        # the same layout drawn with a few different connector ways is one layout
        if any(abs(o.length_m - lp.length_m) / lp.length_m < 0.006
               and len(ids & set(o.way_ids)) / max(1, len(ids | set(o.way_ids))) > 0.8 for o in loops):
            continue
        loops.append(lp)
    return loops
