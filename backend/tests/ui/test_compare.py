"""Lap comparison (plan 2026-11-04-lap-comparison): lap list flags, BEST,
the Compare tab (ghosts on the map, time delta, same class and layout only)."""

import math
import re

import pytest

pytest.importorskip("playwright.sync_api")

from serve_for_ui import (CMP_A, CMP_B, CMP_HYPER, CMP_KEPT, CMP_LAYOUT, CMP_NOLD, CMP_V1,
                          CMP_WAVE, HILLY, PARTIAL)

TEAL, CORAL = "rgb(45, 212, 167)", "rgb(255, 107, 107)"
LOADED = "() => S.mapData && document.getElementById('map-overlay').classList.contains('hidden')"


def test_best_chip_skips_partial_lap(page, server):
    page.goto(server + "/")
    page.locator(f'.session-item[title="{PARTIAL}"]').click()
    page.wait_for_function("() => S.mapData && document.getElementById('map-overlay').classList.contains('hidden')")
    laps = page.evaluate("S.laps")
    assert len(laps) == 4 and laps[-1]["real"] is False
    best = page.locator(".lap-chip.best")
    assert best.count() == 1
    assert best.get_attribute("data-lap") != str(laps[-1]["lap_number"])
    fastest_real = min((l for l in laps if l["real"]), key=lambda l: l["lap_time_s"])
    assert best.get_attribute("data-lap") == str(fastest_real["lap_number"])
    assert page.evaluate("S.lap") == fastest_real["lap_number"]
    assert page.errors == []


# ── Compare tab helpers ──────────────────────────────────────────────────────

def _open(page, server, file, lap=None, goto=True):
    if goto:
        page.goto(server + "/")
    page.locator(f'.session-item[title="{file}"]').click()
    # selectSession auto-loads the best lap; let that finish first so its
    # late response cannot overwrite the lap clicked next.
    page.wait_for_function(f"() => S.file === {file!r} && ({LOADED})()")
    if lap is not None and page.evaluate("S.lap") != lap:
        page.locator(f'.lap-chip[data-lap="{lap}"]').click()
        page.wait_for_function(f"() => S.lap === {lap} && ({LOADED})()")


def _compare_tab(page):
    page.locator("#tab-compare").click()
    page.wait_for_function("() => document.querySelectorAll('#cmp-file option').length > 0")


def _add(page, file, lap, src="sessions"):
    return page.evaluate("([s, f, l]) => CMP.add(s, f, l)", [src, file, lap])


def _scrub(page, v):
    page.evaluate("v => { setScrub(v); onScrub(String(v)); }", v)


def _ghost_rows(page):
    return page.locator("#cmp-list .cmp-row[data-ghost]")


# ── Picker ───────────────────────────────────────────────────────────────────

def test_picker_offers_same_class_and_layout_only(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    opts = page.eval_on_selector_all("#cmp-file option", "os => os.map(o => [o.value, o.textContent])")
    values = [v for v, _ in opts]
    assert values[0] == f"sessions|{CMP_A}" and opts[0][1] == "This session"
    assert f"sessions|{CMP_B}" in values
    kept = [t for v, t in opts if v == f"reference_laps|{CMP_KEPT}"]
    assert kept and "kept" in kept[0]
    for f in (CMP_HYPER, CMP_LAYOUT, CMP_V1):
        assert f"sessions|{f}" not in values
    head = page.inner_text("#cmp-filter")
    assert head.startswith("LMGT3 only") and re.search(rf"{len(values)} files on this track", head)
    # this session: its other complete laps, not yours
    laps = page.eval_on_selector_all("#cmp-lap option", "os => os.map(o => o.value)")
    assert "1" not in laps and set(laps) == {"0", "2"}
    # another file: its complete laps, the real best marked
    page.select_option("#cmp-file", f"sessions|{CMP_B}")
    page.wait_for_function("() => [...document.querySelectorAll('#cmp-lap option')].some(o => o.textContent.includes('★'))")
    assert page.errors == []


# ── Adding ghosts, delta ─────────────────────────────────────────────────────

def test_add_ghost_and_delta_sign(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    r = _add(page, CMP_B, 0)
    assert r["ok"] is True, r
    assert page.evaluate("CMP.ghosts.length") == 1
    _scrub(page, 500)
    d = page.evaluate("CMP.deltaAt(CMP.ghosts[0])")
    assert d < -0.3                       # ~3 % slower over half a 30 s lap
    cell = page.locator("#cmp-list .cmp-d[data-g]").first
    txt = cell.inner_text()
    assert txt.startswith("−") and abs(float(txt[1:]) - abs(d)) < 0.0015
    assert cell.evaluate("e => getComputedStyle(e).color") == TEAL
    assert "10 Hz" in (cell.get_attribute("title") or "")
    pill = page.locator("#cmp-pill")
    assert pill.is_visible() and pill.get_attribute("aria-hidden") == "true"
    assert txt in pill.inner_text()
    assert page.errors == []


def test_same_file_ghost_and_best_buttons(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    page.get_by_role("button", name="Best lap of this session").click()
    page.wait_for_function("() => CMP.ghosts.length === 1")
    g = page.evaluate("({file: CMP.ghosts[0].file, lap: CMP.ghosts[0].lap})")
    assert g["file"] == CMP_A and g["lap"] != 1
    urls = []
    page.on("request", lambda req: urls.append(req.url))
    page.get_by_role("button", name="Best other file").click()
    page.wait_for_function("() => CMP.ghosts.length === 2")
    g2 = page.evaluate("({file: CMP.ghosts[1].file, src: CMP.ghosts[1].src})")
    assert g2["file"] in (CMP_B, CMP_KEPT)
    maps = [u for u in urls if "/trackmap" in u]
    assert maps and all(g2["file"] in u for u in maps), maps
    assert page.errors == []


def test_four_ghost_limit(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    for f, lap in ((CMP_A, 0), (CMP_A, 2), (CMP_B, 0), (CMP_B, 1)):
        assert _add(page, f, lap)["ok"] is True
    r = _add(page, CMP_WAVE, 0)
    assert r["ok"] is False and r.get("reason")
    assert page.evaluate("CMP.ghosts.length") == 4
    assert _ghost_rows(page).count() == 4
    assert page.inner_text("#cmp-status") == r["reason"]
    assert page.errors == []


def test_refusals_have_reasons(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    hyper = _add(page, CMP_HYPER, 0)
    layout = _add(page, CMP_LAYOUT, 0)
    v1 = _add(page, CMP_V1, 0)
    assert hyper["ok"] is False and "Hypercar" in hyper["reason"] and "LMGT3" in hyper["reason"]
    assert layout["ok"] is False and "layout" in layout["reason"]
    assert v1["ok"] is False and "GPS" in v1["reason"]
    assert page.evaluate("CMP.ghosts.length") == 0
    assert page.locator("#cmp-status").get_attribute("role") == "status"
    assert page.errors == []


def test_late_ghost_response_is_dropped(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    held = []

    def hold(route):
        if CMP_B in route.request.url:
            held.append(route)          # released below, after the main lap changed
        else:
            route.continue_()
    page.route(re.compile(r".*/trackmap.*"), hold)
    page.evaluate(f"() => {{ window._late = CMP.add('sessions', {CMP_B!r}, 0); }}")
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(50)
    assert held, "the ghost's /trackmap request never went out"
    _open(page, server, HILLY, goto=False)       # unknown class, same track
    page.wait_for_timeout(1500)
    held[0].continue_()
    r = page.evaluate("window._late")
    page.wait_for_timeout(300)
    assert r["ok"] is False
    assert page.evaluate("CMP.ghosts.length") == 0
    assert _ghost_rows(page).count() == 0
    assert page.errors == []


# ── Alignment ────────────────────────────────────────────────────────────────

POSE = """() => { const g = CMP.ghosts[0], p = CMP.ghostPose(g);
  return {x: p.x, z: p.z, yaw: p.yaw, f: p.f, fMe: p.fMe, n: g.d.driven_line.x.length}; }"""


def test_same_time_holds_at_finish(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    assert _add(page, CMP_B, 0)["ok"] is True
    for mode in ("time", "dist"):
        page.evaluate(f"CMP.setAlign('{mode}')")
        for v in range(0, 1001, 50):
            _scrub(page, v)
            p = page.evaluate(POSE)
            assert all(math.isfinite(p[k]) for k in ("x", "z", "yaw", "f", "fMe")), (mode, v, p)
    page.evaluate("CMP.setAlign('time')")
    _scrub(page, 1000)
    p = page.evaluate(POSE)
    assert p["f"] < p["n"] - 1 - 5                 # the slower ghost has not finished yet
    page.evaluate("CMP.setAlign('dist')")
    p = page.evaluate(POSE)
    assert p["f"] > p["n"] - 1.5                   # alongside you: at its line
    assert page.errors == []


ALONG = """() => { const dl = S.mapData.driven_line, n = dl.x.length, f = S.playhead * (n - 1);
  const p = CMP.ghostPose(CMP.ghosts[0]);
  const [gx, gz] = toScreen(p.x, p.z), [mx, mz] = toScreen(lerpAt(dl.x, f), lerpAt(dl.z, f));
  const h = angleAt(dl.heading, f);
  return (gx - mx) * Math.cos(h) + (gz - mz) * Math.sin(h); }"""


def test_alignment_switch_moves_ghost(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    assert _add(page, CMP_B, 0)["ok"] is True
    _scrub(page, 500)
    page.locator("#cmp-al-dist").click()
    assert page.evaluate("CMP.align") == "dist"
    assert abs(page.evaluate(ALONG)) < 2.0
    page.locator("#cmp-al-time").click()
    assert page.evaluate("CMP.align") == "time"
    assert page.evaluate(ALONG) < -5.0             # slower ghost: behind you
    # radio group: arrow keys move and select
    page.locator("#cmp-al-time").focus()
    page.keyboard.press("ArrowRight")
    assert page.evaluate("CMP.align") == "dist"
    assert page.locator("#cmp-al-dist").get_attribute("aria-checked") == "true"
    assert page.errors == []


def test_ghost_lap_dist_wrapped_at_start(page, server):
    """A lap whose first point still reads the previous lap's distance (the
    game's wrap falls after it, seen on Fuji 10/15) must not read as one
    long stop at the line: that gave a +43 s delta on real data."""
    _open(page, server, CMP_A, 1)
    _compare_tab(page)

    def wrap_first(route):
        if CMP_B not in route.request.url:
            route.continue_()
            return
        resp = route.fetch()
        j = resp.json()
        ld = j["driven_line"]["lap_dist"]
        ld[0] = j["lap_len_m"] + 2.0          # just before the line, previous lap
        route.fulfill(response=resp, json=j)
    page.route(re.compile(r".*/trackmap.*"), wrap_first)
    assert _add(page, CMP_B, 0)["ok"] is True
    _scrub(page, 500)
    d = page.evaluate("CMP.deltaAt(CMP.ghosts[0])")
    assert -1.0 < d < -0.3, d                 # ~3 % slower over half a lap
    page.evaluate("CMP.setAlign('dist')")
    assert abs(page.evaluate(ALONG)) < 2.0
    assert page.errors == []


# ── Map overlay ──────────────────────────────────────────────────────────────

def test_delta_overlay_option(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    opt = page.locator('#heatmap-mode option[value="delta"]')
    assert opt.get_attribute("disabled") is not None
    assert _add(page, CMP_WAVE, 0)["ok"] is True
    assert opt.get_attribute("disabled") is None
    page.select_option("#heatmap-mode", "delta")
    cols = page.evaluate("() => { const n = S.mapData.driven_line.x.length, o = [];"
                         " for (let i = 0; i < n - 1; i++) o.push(lineColorAt(i)); return o; }")
    assert any(c.startswith("rgba(45,212,167") for c in cols)
    assert any(c.startswith("rgba(255,107,107") for c in cols)
    assert page.inner_text("#leg-lo") == "Gaining" and page.inner_text("#leg-hi") == "Losing"
    page.get_by_role("button", name="Remove ghost 1").click()
    assert page.evaluate("CMP.ghosts.length") == 0
    assert page.eval_on_selector("#heatmap-mode", "e => e.value") == "throttle_brake"
    assert opt.get_attribute("disabled") is not None
    assert page.inner_text("#leg-lo") == "Brake"
    assert page.errors == []


NON_GREY = """() => { const n = S.mapData.driven_line.x.length; let c = 0;
  for (let i = 0; i < n - 1; i++) { const s = lineColorAt(i);
    if (s.startsWith('rgba(45,212,167') || s.startsWith('rgba(255,107,107')) c++; }
  return c / (n - 1); }"""


def test_delta_overlay_ignores_a_single_spike(page, server):
    """One jump in the ghost's game clock (a skipped tick: 0.5 s in one
    segment, ~100x the wave's largest slope) must not set the overlay's
    scale. Normalised to the largest slope it greys out every other
    segment; to the 95th percentile the speed wave still colours most of
    the lap (only segments near the wave's zero crossings stay grey)."""
    _open(page, server, CMP_A, 1)
    _compare_tab(page)

    def step(route):
        if CMP_WAVE not in route.request.url:
            route.continue_()
            return
        resp = route.fetch()
        j = resp.json()
        t = j["driven_line"]["t"]
        k = len(t) // 3
        j["driven_line"]["t"] = t[:k] + [v + 0.5 for v in t[k:]]
        route.fulfill(response=resp, json=j)
    page.route(re.compile(r".*/trackmap.*"), step)
    assert _add(page, CMP_WAVE, 0)["ok"] is True
    page.select_option("#heatmap-mode", "delta")
    share = page.evaluate(NON_GREY)
    assert share > 0.6, share
    assert page.errors == []


# ── Main lap changes ─────────────────────────────────────────────────────────

def test_main_lap_change_rules(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    assert _add(page, CMP_B, 0)["ok"] is True
    assert _add(page, CMP_A, 0)["ok"] is True
    page.locator('.lap-chip[data-lap="2"]').click()
    page.wait_for_function(f"() => S.lap === 2 && ({LOADED})()")
    page.wait_for_function("() => CMP.ghosts.length === 2")
    # your new lap is a ghost: that ghost goes, the other is re-aligned
    page.locator('.lap-chip[data-lap="0"]').click()
    page.wait_for_function(f"() => S.lap === 0 && ({LOADED})()")
    page.wait_for_function("() => CMP.ghosts.length === 1")
    g = page.evaluate("({file: CMP.ghosts[0].file, n: CMP.ghosts[0].delta.length})")
    assert g["file"] == CMP_B and g["n"] == page.evaluate("S.mapData.driven_line.x.length")
    _open(page, server, HILLY, goto=False)       # class unknown: the LMGT3 ghosts go
    page.wait_for_function("() => CMP.ghosts.length === 0")
    assert page.errors == []


# ── Accuracy: shared frame, lap distance, coordinate system ──────────────────

def test_no_shared_frame_is_approximate(page, server_no_origin):
    _open(page, server_no_origin, CMP_A, 1)
    assert page.evaluate("S.mapData.coordinate_origin.kind") == "file"
    _compare_tab(page)
    assert _add(page, CMP_B, 0)["ok"] is True
    row = _ghost_rows(page).first
    assert "approximate" in row.inner_text()
    assert "no shared frame" in (row.locator(".cmp-approx").get_attribute("title") or "")
    lat = page.evaluate("() => { const g = CMP.ghosts[0]; return g.lat[Math.round(CMP.meAtDist(600))]; }")
    assert abs(lat) < 3.5
    assert page.errors == []


def test_coordinate_system_mismatch_refused(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    values = page.eval_on_selector_all("#cmp-file option", "os => os.map(o => o.value)")
    assert f"sessions|{CMP_V1}" not in values
    r = _add(page, CMP_V1, 0)
    assert r["ok"] is False and "GPS" in r["reason"]
    # the other way round: your file has no GPS
    _open(page, server, CMP_V1, 1, goto=False)
    r = _add(page, CMP_A, 0)
    assert r["ok"] is False and "Different position data" in r["reason"]
    assert page.evaluate("CMP.ghosts.length") == 0
    assert page.errors == []


def test_no_lap_dist_falls_back_and_says_approximate(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    r = _add(page, CMP_NOLD, 0)
    assert r["ok"] is True, r
    assert page.evaluate("CMP.ghosts[0].approx") is True
    row = _ghost_rows(page).first
    assert "approximate" in row.inner_text()
    assert "no lap distance" in (row.locator(".cmp-approx").get_attribute("title") or "")
    _scrub(page, 500)
    assert page.evaluate("CMP.deltaAt(CMP.ghosts[0])") < 0       # CMP_NOLD is 4 % slower
    assert page.errors == []


# ── Tab: keyboard and a narrow panel ─────────────────────────────────────────

def test_compare_tab_keyboard_and_narrow_panel(page, server):
    _open(page, server, CMP_A, 1)
    page.locator("#tab-record").click()
    page.locator("#tab-record").focus()
    page.keyboard.press("ArrowRight")
    assert page.locator("#tab-compare").get_attribute("aria-selected") == "true"
    assert page.locator("#body-compare").is_visible()
    page.keyboard.press("Home")
    assert page.locator("#tab-balance").get_attribute("aria-selected") == "true"
    page.keyboard.press("End")
    assert page.locator("#tab-compare").get_attribute("aria-selected") == "true"
    page.evaluate("PanelSize.setWidth(300)")
    tab = page.locator("#tab-compare").bounding_box()
    btn = page.locator("#panel-float-btn").bounding_box()
    assert tab["width"] >= 30 and tab["x"] + tab["width"] <= btn["x"] + 0.5
    assert page.errors == []


# ── Traces over lap distance: synced zoom, pan, click to scrub ───────────────

TRACES = ["cmp-delta", "cmp-spd", "cmp-thr", "cmp-brk", "cmp-str", "cmp-lat"]
# Your lap distance at the playhead (CMP_A has the game's lap distance).
PLAY_DIST = """() => { const dl = S.mapData.driven_line, n = dl.x.length;
  return lerpAt(dl.lap_dist, clamp(S.playhead * (n - 1), 0, n - 1)); }"""


def _frames(page, k=2):
    """Let k animation frames pass (trace redraws are rAF-throttled)."""
    page.evaluate("k => new Promise(r => { const f = i => i ? requestAnimationFrame(() => f(i - 1)) : r(); f(k); })", k)


def _with_ghost(page, server):
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    assert _add(page, CMP_B, 0)["ok"] is True
    _frames(page)
    v = page.evaluate("CMP.view")
    assert v["s0"] == 0 and v["s1"] > 100          # whole lap to begin with
    return v["s1"]


def _view(page):
    v = page.evaluate("CMP.view")
    return v["s0"], v["s1"]


def _hover(page, sel, fx=0.5):
    loc = page.locator(sel)
    loc.scroll_into_view_if_needed()
    b = loc.bounding_box()
    page.mouse.move(b["x"] + b["width"] * fx, b["y"] + b["height"] / 2)
    return b


def _drawn_windows(page):
    return page.evaluate("ids => ids.map(id => document.getElementById(id).dataset.view || '')", TRACES)


def test_wheel_zoom_syncs_all_traces(page, server):
    L = _with_ghost(page, server)
    assert page.locator("#h-cmp-delta").get_attribute("title") == "Estimate from 10 Hz position and game lap distance"
    assert page.inner_text("#cmp-range") == ""            # whole lap: no range label
    _hover(page, "#cmp-delta")
    page.mouse.wheel(0, -100)
    _frames(page)
    s0, s1 = _view(page)
    assert abs((s1 - s0) - 0.8 * L) < 1e-6 and abs(s0 - 0.1 * L) < 0.02 * L   # x0.8 around the cursor
    want = f"{s0:.1f},{s1:.1f}"
    assert _drawn_windows(page) == [want] * len(TRACES)   # every graph drawn with the same window
    assert page.inner_text("#cmp-range") == f"{round(s0)}–{round(s1)} m"
    page.mouse.wheel(0, 100)                               # x1.25 back out
    _frames(page)
    s0, s1 = _view(page)
    assert abs(s0) < 1e-6 and abs(s1 - L) < 1e-6
    assert page.errors == []


def test_drag_pans_and_dblclick_resets(page, server):
    L = _with_ghost(page, server)
    page.evaluate("CMP.setView(100, 300)")
    _frames(page)
    play0 = page.evaluate("S.playhead")
    b = _hover(page, "#cmp-spd")
    x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    page.mouse.down()
    page.mouse.move(x - 40, y, steps=4)
    page.mouse.move(x - 100, y, steps=4)
    page.mouse.up()
    _frames(page)
    s0, s1 = _view(page)
    assert abs((s1 - s0) - 200) < 1e-6                     # the span is kept
    assert abs(s0 - (100 + 100 / b["width"] * 200)) < 3    # dragged left = later in the lap
    assert page.evaluate("S.playhead") == play0            # a drag is not a click
    assert _drawn_windows(page) == [f"{s0:.1f},{s1:.1f}"] * len(TRACES)
    page.mouse.dblclick(x, y)
    _frames(page)
    s0, s1 = _view(page)
    assert s0 == 0 and abs(s1 - L) < 1e-6
    assert page.inner_text("#cmp-range") == ""
    assert page.errors == []


def test_click_scrubs_to_distance(page, server):
    L = _with_ghost(page, server)
    b = _hover(page, "#cmp-thr")
    page.mouse.click(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    assert abs(page.evaluate(PLAY_DIST) - L / 2) < 0.02 * L
    assert page.evaluate("+document.getElementById('pb-scrub').value") == round(page.evaluate("S.playhead") * 1000)
    # zoomed: a click lands on the distance under the cursor, not on the lap fraction
    page.evaluate("CMP.setView(200, 400)")
    page.mouse.click(b["x"] + b["width"] * 0.25, b["y"] + b["height"] / 2)
    assert abs(page.evaluate(PLAY_DIST) - 250) < 0.02 * L
    assert page.errors == []


def test_ctrl_wheel_still_scales_panel(page, server):
    _with_ghost(page, server)
    v0 = _view(page)
    _hover(page, "#cmp-spd")
    page.keyboard.down("Control")
    page.mouse.wheel(0, -100)
    page.mouse.wheel(0, -100)
    page.keyboard.up("Control")
    page.wait_for_timeout(50)
    assert page.inner_text("#panel-scale-val") == "110%"
    assert _view(page) == v0
    assert page.errors == []


def test_trace_keyboard_zoom(page, server):
    L = _with_ghost(page, server)
    for id_ in TRACES:
        c = page.locator("#" + id_)
        assert c.get_attribute("tabindex") == "0" and c.get_attribute("aria-label")
    page.locator("#cmp-str").focus()
    page.keyboard.press("+")
    s0, s1 = _view(page)
    assert abs((s1 - s0) - 0.8 * L) < 1e-6 and abs((s0 + s1) / 2 - L / 2) < 1e-6   # about the centre
    page.keyboard.press("-")
    s0, s1 = _view(page)
    assert abs(s0) < 1e-6 and abs(s1 - L) < 1e-6
    page.keyboard.press("+")
    page.keyboard.press("+")
    a0, a1 = _view(page)
    page.keyboard.press("ArrowRight")
    s0, s1 = _view(page)
    assert abs(s0 - (a0 + 0.1 * (a1 - a0))) < 1e-6 and abs((s1 - s0) - (a1 - a0)) < 1e-6
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    s0, _ = _view(page)
    assert abs(s0 - (a0 - 0.1 * (a1 - a0))) < 1e-6
    page.keyboard.press("Enter")                           # the playhead to the window's centre
    s0, s1 = _view(page)
    assert abs(page.evaluate(PLAY_DIST) - (s0 + s1) / 2) < 0.02 * L
    page.keyboard.press("0")
    s0, s1 = _view(page)
    assert s0 == 0 and abs(s1 - L) < 1e-6
    assert page.errors == []


def test_min_span_25m(page, server):
    L = _with_ghost(page, server)
    page.evaluate("CMP.setView(100, 101)")
    s0, s1 = _view(page)
    assert abs((s1 - s0) - 25) < 1e-6 and 75 <= s0 <= 100
    page.evaluate("CMP.setView(-50, 10 * " + str(L) + ")")   # clamped to the lap
    s0, s1 = _view(page)
    assert s0 == 0 and abs(s1 - L) < 1e-6
    _hover(page, "#cmp-lat")
    for _ in range(40):
        page.mouse.wheel(0, -100)
    _frames(page)
    s0, s1 = _view(page)
    assert abs((s1 - s0) - 25) < 1e-6
    assert page.errors == []


def _dip_at_driven_fraction(dl, frac=0.4):
    """Index of the point at fraction `frac` of the distance driven, with the
    speed there halved: a feature at one place on the track in every lap."""
    cum = [0.0]
    for i in range(1, len(dl["x"])):
        cum.append(cum[-1] + math.hypot(dl["x"][i] - dl["x"][i - 1], dl["z"][i] - dl["z"][i - 1]))
    k = min(range(len(cum)), key=lambda i: abs(cum[i] - frac * cum[-1]))
    dl["speed"][k] *= 0.5
    return k


def test_ghost_without_lap_dist_on_your_axis(page, server):
    """Your lap has the game's lap distance, the ghost (CMP_NOLD) has not, so
    attach aligns on driven distance. The graphs' x axis is your game lap
    distance (sPrim). Every ghost series must be mapped onto that axis, not
    drawn at its driven distance. Your lap distance is warped here (+60 m
    sin(pi s / L), same ends) so the two bases differ by up to 60 m."""
    def route(r):
        url = r.request.url
        if CMP_A not in url and CMP_NOLD not in url:
            r.continue_()
            return
        resp = r.fetch()
        j = resp.json()
        dl = j["driven_line"]
        _dip_at_driven_fraction(dl)
        if CMP_A in url:
            L = j["lap_len_m"]
            dl["lap_dist"] = [s + 60 * math.sin(math.pi * s / L) for s in dl["lap_dist"]]
        r.fulfill(response=resp, json=j)
    page.route(re.compile(r".*/trackmap.*"), route)
    _open(page, server, CMP_A, 1)
    _compare_tab(page)
    assert _add(page, CMP_NOLD, 0)["ok"] is True
    assert page.evaluate("CMP.ghosts[0].approx") is True
    _scrub(page, 400)
    sprim = page.evaluate("CMP._plotted('cmp-spd').series.at(-1).x")      # yours: drawn last
    for id_ in ("cmp-delta", "cmp-lat", "cmp-spd", "cmp-thr"):
        p = page.evaluate(f"CMP._plotted('{id_}')")
        ghost = p["series"][0]
        assert abs(ghost["dotS"] - p["sp"]) < 1e-6, (id_, ghost["dotS"], p["sp"])   # dot on the playhead
        if id_ in ("cmp-delta", "cmp-lat"):
            assert ghost["x"] == sprim                     # values at your points: your axis
    # the speed dip (one place on the track) lands where yours does. Tolerance
    # 6 m: one 10 Hz point is ~5 m apart here, plus the small difference in
    # driven length between the two lines (both scaled to the lap).
    spd = page.evaluate("""() => { const p = CMP._plotted('cmp-spd');
      return p.series.map(s => { let k = 0; for (let i = 1; i < s.v.length; i++) if (s.v[i] < s.v[k]) k = i;
        return s.x[k]; }); }""")
    assert abs(spd[0] - spd[-1]) < 6, spd
    assert page.errors == []


def test_traces_follow_panel_scale(page, server):
    _with_ghost(page, server)
    h1 = page.evaluate("document.getElementById('cmp-delta').getBoundingClientRect().height")
    assert 118 <= h1 <= 124                               # 120 panel units (+ border)
    page.evaluate("PanelSize.setScale(2)")
    _frames(page)
    h2 = page.evaluate("document.getElementById('cmp-delta').getBoundingClientRect().height")
    assert abs(h2 / h1 - 2.0) < 0.05
    assert page.errors == []
