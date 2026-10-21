"""Track reference: OSM loop stitching, corner extraction and sequence alignment.

Run: uv run --no-project --with pytest --with httpx --with fastapi --with duckdb --with numpy --with scipy --with pydantic pytest backend/tests/test_track_reference.py
"""
import numpy as np

import app.track_reference as tr
from app.track_catalog import find_entry
from app.track_centerline import classify_name, loops_from_osm

LAT0, LON0 = 35.0, 139.0
M = 111_320.0


def to_ll(x, y):
    return {"lat": LAT0 + y / M, "lon": LON0 + x / (M * np.cos(np.radians(LAT0)))}


def arc(cx, cy, r, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a]


def line(p, q, n=12):
    return [(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t) for t in np.linspace(0, 1, n)]


def stadium_osm(oneway=True, names=True):
    """Counter-clockwise stadium: two 600 m straights, two 150 m radius hairpins (~1.8 km)
    plus a connector way that is NOT part of the loop (a pit lane)."""
    R, S = 150.0, 600.0
    pts_bottom = line((0, 0), (S, 0))
    pts_t1 = arc(S, R, R, -np.pi / 2, np.pi / 2)
    pts_top = line((S, 2 * R), (0, 2 * R))
    pts_t2 = arc(0, R, R, np.pi / 2, 3 * np.pi / 2)

    def way(i, pts, name, extra=None):
        t = {"highway": "raceway", "sport": "motor", **({"oneway": "yes"} if oneway else {}),
             **({"name": name} if (name and names) else {})}
        t.update(extra or {})
        return {"id": i, "tags": t, "geometry": [to_ll(*p) for p in pts]}

    ways = [way(1, pts_bottom, "Main Straight"), way(2, pts_t1, "Turn 1"),
            way(3, pts_top, None), way(4, pts_t2, "Turn 2"),
            way(5, line((100, 0), (100, -60)), "Pit Lane")]
    return {"center": [LAT0, LON0], "ways": ways, "named": []}


def test_name_classification():
    assert classify_name("Turn 7") == ("turn", "7", None)
    assert classify_name("Turn 10A") == ("turn", "10", "A")
    assert classify_name("Kemmel Straight")[0] == "straight"
    assert classify_name("Ligne droite des Hunaudières")[0] == "straight"
    assert classify_name("Grand Prix Circuit")[0] == "layout"
    assert classify_name("Eau Rouge")[0] == "named"
    assert classify_name(None)[0] == "none"


def test_catalog_finds_tracks_despite_accents_and_mojibake():
    assert find_entry("Circuit of the Americas")["slug"] == "cota"
    assert find_entry("Aut�dromo Jos� Carlos Pace")["slug"] == "interlagos"
    assert find_entry("Michelin Raceway Road Atlanta")["slug"] == "road_atlanta"
    assert find_entry("Circuit de Barcelona")["slug"] == "barcelona"
    assert find_entry("Nowhere Park") is None


def test_loop_is_stitched_and_pit_lane_ignored():
    loops = loops_from_osm(stadium_osm(), [2 * 600 + 2 * np.pi * 150])
    assert loops
    lp = loops[0]
    assert abs(lp.length_m - (1200 + 2 * np.pi * 150)) < 25
    assert 5 not in lp.way_ids and set(lp.way_ids) == {1, 2, 3, 4}
    assert lp.directed
    # counter-clockwise: net turning about +360 degrees (left turns positive)
    assert 340 < np.degrees(np.sum(lp.k)) < 380


def test_corners_straights_and_osm_numbers():
    lp = loops_from_osm(stadium_osm(), [2143])[0]
    corners, straights = tr.corners_from_loop(lp)
    assert len(corners) == 2
    assert all(c["direction"] == "left" and 160 < c["heading_deg"] < 200 for c in corners)
    assert sorted(c.get("number") for c in corners) == [1, 2]
    assert [s["name"] for s in straights] == ["Main Straight"]


def test_loop_origin_never_cuts_a_corner():
    lp = loops_from_osm(stadium_osm(), [2143])[0]
    # origin sits on a straight: curvature near s=0 is ~0
    assert abs(lp.k[0]) < 1e-3 and abs(lp.k[-1]) < 1e-3


# -------------------------------------------------------------------------- alignment

def ref_corners():
    return [
        {"apex_m": 300, "heading_deg": 90, "direction": "right"},
        {"apex_m": 700, "heading_deg": 40, "direction": "left"},
        {"apex_m": 1100, "heading_deg": 150, "direction": "left"},
        {"apex_m": 1600, "heading_deg": 60, "direction": "right"},
        {"apex_m": 2200, "heading_deg": 120, "direction": "right"},
        {"apex_m": 2800, "heading_deg": 30, "direction": "left"},
    ]


def detect_from(ref, start_idx, flip, scale=0.99, L=3000.0, drop=()):
    """What a recording would see: the corners in lap order from a start/finish placed just
    before ref corner start_idx, mirrored yaw sign (flip), odometer scale, some corners lost."""
    order = ref[start_idx:] + ref[:start_idx]
    s0 = ref[start_idx]["apex_m"] - 150
    out = []
    for c in order:
        if c["apex_m"] in drop:
            continue
        d = ((c["apex_m"] - s0) % L) * scale
        out.append({"apex_m": d, "start_m": d - 40, "end_m": d + 40, "heading_deg": c["heading_deg"] * 0.95,
                    "sign": (1 if c["direction"] == "left" else -1) * flip, "demand": "braking"})
    return out


def test_alignment_recovers_start_line_mirror_and_scale():
    ref = ref_corners()
    for start, flip in [(0, 1), (2, 1), (3, -1), (5, -1)]:
        det = detect_from(ref, start, flip)
        al = tr.align(ref, 3000.0, det, 3000.0 * 0.99, directed=True)
        assert al is not None
        assert al["flip"] == flip
        assert al["rotation"] == start
        assert len(al["pairs"]) == 6
        assert abs(al["b"] - 1 / 0.99) < 0.02            # ref metres per lap metre
        assert al["resid_m"] < 5


def test_alignment_survives_lost_corners_and_extra_detections():
    ref = ref_corners()
    det = detect_from(ref, 2, 1, drop=(700, 2800))
    det.insert(2, {"apex_m": 400.0, "start_m": 380, "end_m": 420, "heading_deg": 18, "sign": 1, "demand": "lift"})
    al = tr.align(ref, 3000.0, det, 2970.0, directed=True)
    assert al and al["flip"] == 1
    # rotation 1 and 2 are the same alignment when corner 700 is the one that was lost
    assert sorted(r["apex_m"] for r, _ in al["pairs"]) == [300, 1100, 1600, 2200]
    assert al["resid_m"] < 10


def test_undirected_loop_can_be_reversed():
    ref = ref_corners()
    rev = [{**c, "direction": "left" if c["direction"] == "right" else "right", "apex_m": 3000 - c["apex_m"]}
           for c in reversed(ref)]
    det = detect_from(rev, 1, 1)
    al = tr.align(ref, 3000.0, det, 2970.0, directed=False)
    assert al and al["direction"] == -1 and len(al["pairs"]) == 6


# -------------------------------------------------------------------------- names

def test_track_db_labels_are_parsed_and_display_format():
    pts = tr.parse_track_db([
        {"dist": 740, "corner": "T1 Apex (TGR Corner - Lowest 550m)"},
        {"dist": 450, "corner": "T1 Braking (Downhill)"},
        {"dist": 1727, "corner": "T4/T5 Apex (100R)"},
        {"dist": 2100, "corner": "Tertre Rouge Apex"},
    ])
    assert [(p["numbers"], p["name"]) for p in pts] == [([1], "TGR Corner"), ([4, 5], "100R"), ([], "Tertre Rouge")]
    assert tr._display({"name": "Coca-Cola Corner", "numbers": [3]}) == "Coca-Cola Corner (3)"
    assert tr._display({"name": "Esses", "numbers": [3, 4, 5, 6]}) == "Esses (3, 4, 5, 6)"
    assert tr._display({"name": "Porsche Curves"}) == "Porsche Curves"
    assert tr._display({"number": 7}) == "Turn 7"
    assert tr._display({"number": 10, "suffix": "A"}) == "Turn 10A"


def test_complex_grouping_comma_lists_numbers():
    cs = [{"name": "Esses", "numbers": [3]}, {"name": "Esses", "numbers": [4]}, {"name": "Esses", "numbers": [5]},
          {"name": "Hairpin", "numbers": [6]}]
    tr._group_complexes(cs)
    assert [c.get("complex") for c in cs] == ["Esses", "Esses", "Esses", None]
    assert cs[0]["numbers"] == [3, 4, 5]
    assert tr._display(cs[1]) == "Esses (3, 4, 5)"


def test_locate_names_the_corner_or_the_gap():
    layout = {"corners": [
        {"display": "Turn A", "lmu_apex_m": 500.0, "lmu_start_m": 450.0, "lmu_end_m": 560.0},
        {"display": "Turn B", "lmu_apex_m": 1500.0, "lmu_start_m": 1450.0, "lmu_end_m": 1560.0}]}
    assert tr.locate(layout, 505) == "Turn A"
    assert tr.locate(layout, 540) == "Turn A (+40 m)"
    assert tr.locate(layout, 1000) == "between Turn A and Turn B"
    assert tr.locate(layout, 100) is None


def _fake_ref():
    corners = [
        {"order": 1, "display": "Hairpin (1)", "lmu_apex_m": 500.0, "lmu_start_m": 450.0, "lmu_end_m": 560.0},
        {"order": 2, "display": "Esses (3, 4)", "complex": "Esses", "numbers": [3, 4], "lmu_apex_m": 1500.0,
         "lmu_start_m": 1450.0, "lmu_end_m": 1560.0},
        {"order": 3, "display": "Esses (3, 4)", "complex": "Esses", "numbers": [3, 4], "lmu_apex_m": 1650.0,
         "lmu_start_m": 1600.0, "lmu_end_m": 1700.0},
    ]
    return {"slug": "x", "display_name": "X", "layout": {
        "label": "Grand Prix", "status": "verified", "official": {"turns": 4}, "corners": corners,
        "straights": [{"name": "Back Straight", "start_m": 700, "end_m": 1300}],
        "complexes": [{"name": "Esses", "parts": [2, 3], "numbers": [3, 4]}]}}


def test_annotate_report_names_events_and_summarises_the_track():
    report = {
        "abs_events": [{"lap_dist_start_m": 505.0, "lap_dist_end_m": 520.0},
                       {"lap_dist_start_m": 1000.0, "lap_dist_end_m": 1655.0}],
        "tc_events": [], "understeer_candidates": [],
        "corner_analysis": {"corners": [{"mrp_dist_m": 1655.0}]},
        "slip_angle_analysis": {"peak_yaw_rate_deg_s": {"lap_dist_m": 1505.0}},
    }
    tr.annotate_report(report, _fake_ref())
    assert report["corner_names_available"] is True
    assert report["abs_events"][0]["corner"] == "Hairpin (1)"
    assert "corner_end" not in report["abs_events"][0]
    assert report["abs_events"][1]["corner"].startswith("between Hairpin (1) and Esses")
    assert report["abs_events"][1]["corner_end"].startswith("Esses (3, 4)")
    assert report["corner_analysis"]["corners"][0]["corner"] == "Esses (3, 4)"
    assert report["slip_angle_analysis"]["peak_yaw_rate_deg_s"]["corner"] == "Esses (3, 4)"
    s = report["track_reference"]
    assert s["layout"] == "Grand Prix" and s["official_corner_count"] == 4
    assert s["complexes"][0]["name"] == "Esses" and s["straights"] == ["Back Straight"]


def test_track_db_annotation_prefers_a_reference_and_falls_back(monkeypatch):
    import app.track_db as tdb
    monkeypatch.setattr("app.track_reference.load_reference", lambda name, length=None: _fake_ref())
    r = {"abs_events": [{"lap_dist_start_m": 505.0, "lap_dist_end_m": 510.0}], "tc_events": [], "understeer_candidates": []}
    tdb.annotate_report_with_corners(r, "Whatever", "", lap_length_m=3000)
    assert r["abs_events"][0]["corner"] == "Hairpin (1)" and "track_reference" in r
    monkeypatch.setattr("app.track_reference.load_reference", lambda name, length=None: None)
    r2 = {"abs_events": [], "tc_events": [], "understeer_candidates": []}
    tdb.annotate_report_with_corners(r2, "Fuji Speedway", "", lap_length_m=4500)
    assert r2["corner_names_available"] is True and "track_reference" not in r2


def test_recorded_only_layout_uses_lap_distance_and_track_db_names():
    det = [
        {"apex_m": 600.0, "start_m": 560, "end_m": 640, "heading_deg": 90, "sign": 1, "radius_m": 60, "demand": "braking"},
        {"apex_m": 1500.0, "start_m": 1450, "end_m": 1560, "heading_deg": 150, "sign": -1, "radius_m": 40, "demand": "braking"},
        {"apex_m": 2300.0, "start_m": 2250, "end_m": 2350, "heading_deg": 60, "sign": 1, "radius_m": 120, "demand": "lift"},
    ]
    db = tr.parse_track_db([{"dist": 1490, "corner": "T2 Apex (Hairpin - Lowest 10m)"}])
    lay = tr.build_recorded_layout(det, 3000.0, 12, db, {"turns": 3})
    assert lay["status"] == "verified" and lay["alignment"]["source"] == "recordings only"
    assert [c["display"] for c in lay["corners"]] == ["Unnamed corner", "Hairpin (2)", "Unnamed corner"] or \
        lay["corners"][1]["display"] == "Hairpin (2)"
    assert lay["corners"][1]["direction"] == "right" and lay["corners"][0]["direction"] == "left"
    assert tr.locate(lay, 1505) == "Hairpin (2)"
    assert tr.build_recorded_layout(det, 3000.0, 4, db, {})["status"] == "provisional"
    assert tr.build_recorded_layout(det, 3000.0, 2, db, {})["status"] == "osm_only"
