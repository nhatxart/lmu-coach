"""Chase view: vendored three.js, the Top-down/Chase switch, the 3D scene,
the no-WebGL fallback and the flat-track note (Task 6); the smoothed chase
camera, scrub snaps, the animated transition and chase zoom (Task 7)."""

import struct
import urllib.request
import zlib

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

from serve_for_ui import HILLY, LEGACY

THREE_URL = "/static/vendor/three/three.module.js"


def _png_rgb(data: bytes) -> np.ndarray:
    """Decode a Playwright PNG (8-bit RGB/RGBA, not interlaced) to HxWx3,
    so the test needs no imaging library."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    w = h = ctype = 0
    while pos < len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and ctype in (2, 6)
        elif typ == b"IDAT":
            idat += body
        pos += 12 + n
    bpp = 3 if ctype == 2 else 4
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * bpp)
    out = np.zeros((h, w * bpp), dtype=np.int32)
    prev = np.zeros(w * bpp, dtype=np.int32)
    for y in range(h):
        f, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        cur = np.zeros_like(prev)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 255
        else:                                   # 1 Sub, 3 Average, 4 Paeth need the left pixel
            for x in range(w * bpp):
                a = cur[x - bpp] if x >= bpp else 0
                b, c = prev[x], (prev[x - bpp] if x >= bpp else 0)
                if f == 1:
                    p = a
                elif f == 3:
                    p = (a + b) >> 1
                else:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[x] = (line[x] + p) & 255
        out[y], prev = cur, cur
    return out.reshape(h, w, bpp)[:, :, :3]


def _open_session(page, server, file, lap=None):
    page.goto(server + "/")
    page.locator(f'.session-item[title="{file}"]').click()
    # selectSession auto-loads the best lap; let that finish first so its
    # late response cannot overwrite the lap clicked next.
    page.wait_for_function("() => S.mapData && document.getElementById('map-overlay').classList.contains('hidden')")
    if lap is not None:
        page.locator(f'.lap-chip[data-lap="{lap}"]').click()
        page.wait_for_function(f"() => S.lap === {lap} && S.mapData && "
                               "document.getElementById('map-overlay').classList.contains('hidden')")


def _wait_built(page):
    page.wait_for_function("() => window.Chase && window.Chase.state().built", timeout=30000)


def _radio(page, name):
    return page.get_by_role("radio", name=name, exact=True)


def test_vendor_served(server):
    with urllib.request.urlopen(server + THREE_URL) as r:
        assert r.status == 200
        assert "javascript" in r.headers["content-type"]


def test_top_down_does_not_load_three(page, server, shot_dir):
    vendor = []
    page.on("request", lambda r: vendor.append(r.url) if "/static/vendor/" in r.url else None)
    _open_session(page, server, HILLY, lap=1)
    page.wait_for_timeout(500)
    if shot_dir:
        page.screenshot(path=str(shot_dir / "top-down.png"))
    assert vendor == []
    assert page.errors == []


def test_switch_to_chase_renders(page, server, shot_dir):
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Chase").click()
    _wait_built(page)
    page.wait_for_timeout(300)
    assert page.evaluate("Chase.state().view") == "chase"
    assert _radio(page, "Chase").get_attribute("aria-checked") == "true"
    # Sample the upper middle of the stage: the HUDs (bottom corners), the
    # zoom column (right) and the note pill (top left) sit elsewhere, so
    # contrast here can only come from the 3D scene.
    box = page.locator("#chase-canvas").bounding_box()
    clip = {"x": box["x"] + box["width"] * 0.2, "y": box["y"] + box["height"] * 0.12,
            "width": box["width"] * 0.45, "height": box["height"] * 0.38}
    img = _png_rgb(page.screenshot(clip=clip))
    if shot_dir:
        page.screenshot(path=str(shot_dir / "chase.png"))
    assert img.std() > 8, img.std()
    assert page.errors == []


def test_chase_falls_back_without_webgl(page, server):
    page.add_init_script("""
      const orig = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (kind, ...rest) {
        return /webgl/i.test(kind) ? null : orig.call(this, kind, ...rest);
      };""")
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Chase").click()
    page.wait_for_function("() => document.getElementById('view-top').getAttribute('aria-checked') === 'true'"
                           " && window.Chase && window.Chase.state().failed")
    msg = page.locator("#chase-msg")
    assert msg.is_visible()
    assert "needs WebGL" in msg.inner_text()
    assert page.evaluate("S.view") == "top"


def test_build_failure_falls_back_once(page, server):
    """A malformed surface must cost the chase view once (message, back to
    top-down), not throw on every frame."""
    def corrupt(route):
        resp = route.fetch()
        body = resp.json()
        body["surface"]["x"] = None
        route.fulfill(response=resp, json=body)
    page.route("**/trackmap*", corrupt)
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Chase").click()
    page.wait_for_function("() => document.getElementById('view-top').getAttribute('aria-checked') === 'true'"
                           " && document.getElementById('chase-msg').classList.contains('on')")
    assert "Showing top-down" in page.locator("#chase-msg").inner_text()
    for v in (100, 300, 500, 700):              # frames after the failure
        page.evaluate("v => onScrub(v)", v)
    page.wait_for_timeout(200)
    assert page.evaluate("Chase.state().built") is False
    assert len(page.errors) <= 1, page.errors
    # the user's choice survives a failure of this one lap
    assert page.evaluate("localStorage.getItem('lmu-coach.view')") == "chase"


def test_paint_cells_drawn_at_true_size(page, server, shot_dir):
    """Important 1: the kerb is drawn from /trackmap `paint` (2 m x 0.5 m
    cells) as one merged mesh, not from the 11-column `surface.kind` grid."""
    _open_session(page, server, HILLY, lap=1)
    n_paint = page.evaluate("S.mapData.paint.length")
    assert n_paint > 0
    _radio(page, "Chase").click()
    _wait_built(page)
    st = page.evaluate("Chase.state()")
    assert st["paintCells"] == n_paint, st
    assert st["paintVerts"] == 6 * n_paint, st
    page.wait_for_function("() => Chase.state().mix === 1", timeout=10000)
    page.evaluate("onScrub(Math.round(1000 * 585 / 1200))")      # just before the kerb (s 587-653 m)
    page.wait_for_function("() => !Chase.state().settling", timeout=5000)
    if shot_dir:
        page.screenshot(path=str(shot_dir / "kerb-cells.png"))
    assert page.errors == []


def test_old_payload_without_paint_uses_surface_kind(page, server):
    """A payload from before `paint` existed still builds (kerbs from the
    column grid, no paint mesh)."""
    def strip(route):
        resp = route.fetch()
        body = resp.json()
        body.pop("paint", None)
        body.pop("paint_cell", None)
        body["surface"].pop("s", None)
        route.fulfill(response=resp, json=body)
    page.route("**/trackmap*", strip)
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Chase").click()
    _wait_built(page)
    assert page.evaluate("Chase.state().paintCells") == 0
    assert page.errors == []


def test_persisted_chase_view_on_reload(page, server):
    page.add_init_script("localStorage.setItem('lmu-coach.view', 'chase')")
    _open_session(page, server, HILLY, lap=1)
    _wait_built(page)
    assert page.evaluate("Chase.state().mix") == 1
    assert _radio(page, "Chase").get_attribute("aria-checked") == "true"
    assert page.errors == []


def test_fallback_keeps_saved_preference(page, server):
    page.add_init_script("""
      localStorage.setItem('lmu-coach.view', 'chase');
      const orig = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (kind, ...rest) {
        return /webgl/i.test(kind) ? null : orig.call(this, kind, ...rest);
      };""")
    page.goto(server + "/")
    page.wait_for_function("() => window.Chase && Chase.state().failed"
                           " && document.getElementById('view-top').getAttribute('aria-checked') === 'true'")
    assert page.evaluate("localStorage.getItem('lmu-coach.view')") == "chase"


def test_context_loss_falls_back(page, server):
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Chase").click()
    _wait_built(page)
    page.evaluate("document.getElementById('chase-canvas').getContext('webgl2')"
                  ".getExtension('WEBGL_lose_context').loseContext()")
    page.wait_for_function("() => document.getElementById('view-top').getAttribute('aria-checked') === 'true'"
                           " && document.getElementById('chase-msg').classList.contains('on')")
    assert page.evaluate("S.view") == "top"


def test_legacy_session_chase_shows_flat_note(page, server):
    _open_session(page, server, LEGACY)
    _radio(page, "Chase").click()
    _wait_built(page)
    note = page.locator("#chase-note")
    assert note.is_visible()
    assert "Flat" in note.inner_text()
    assert page.errors == []


def test_view_switch_keyboard(page, server):
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Top-down").focus()
    page.keyboard.press("ArrowRight")
    assert _radio(page, "Chase").get_attribute("aria-checked") == "true"
    _wait_built(page)
    page.keyboard.press("v")
    assert _radio(page, "Top-down").get_attribute("aria-checked") == "true"
    assert page.evaluate("S.view") == "top"
    assert page.evaluate("localStorage.getItem('lmu-coach.view')") == "top"


# ── Task 7: chase camera, transition, zoom ─────────────────────────────────

def _cam_to_car(page):
    """Horizontal camera-to-car distance (m), the smoothed course's error
    against its target (rad, shortest arc) and the chase distance D."""
    return page.evaluate("""() => {
      const s = Chase.state(), dx = s.car.x - s.cam.x, dz = s.car.z - s.cam.z;
      const e = s.courseTarget - s.course;
      return {d: Math.hypot(dx, dz), err: Math.abs(Math.atan2(Math.sin(e), Math.cos(e))), D: s.dist};
    }""")


def _to_chase(page):
    _radio(page, "Chase").click()
    _wait_built(page)
    # mix can read 1 a frame before GSAP's onComplete clears the tween; until
    # then render() leaves the frames to the tween (no settle loop), which
    # made test_settle_loop_stops_when_paused flaky. Wait for both.
    page.wait_for_function("() => Chase.state().mix === 1 && !Chase.state().tweening", timeout=10000)


def test_transition_animates(page, server, shot_dir):
    _open_session(page, server, HILLY, lap=1)
    _radio(page, "Chase").click()
    _wait_built(page)
    page.wait_for_timeout(300)
    mix = page.evaluate("Chase.state().mix")
    if shot_dir:
        page.screenshot(path=str(shot_dir / "transition-mid.png"))
    assert 0 < mix < 1, mix
    page.wait_for_timeout(1500)
    assert page.evaluate("Chase.state().mix") == 1
    c = _cam_to_car(page)
    assert c["d"] < 15, c
    assert page.errors == []


def test_reduced_motion_is_instant(browser, server):
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, reduced_motion="reduce")
    try:
        page = ctx.new_page()
        _open_session(page, server, HILLY, lap=1)
        _radio(page, "Chase").click()
        _wait_built(page)
        mix = page.evaluate("() => new Promise(r => requestAnimationFrame(() => r(Chase.state().mix)))")
        assert mix == 1
    finally:
        ctx.close()


def test_switch_back_restores_top_down(page, server):
    _open_session(page, server, HILLY, lap=1)
    page.evaluate("zoomStep(1)")                 # a non-default 2D framing to restore
    page.evaluate("S.panX = 37; S.panY = -21; drawMap()")
    before = page.evaluate("[S.zoom, S.panX, S.panY]")
    _to_chase(page)
    page.locator("#map-canvas").hover()
    page.mouse.wheel(0, 300)                      # chase zoom: must not touch the 2D framing
    _radio(page, "Top-down").click()
    page.wait_for_timeout(1200)
    assert page.evaluate("Chase.state().mix") == 0
    op = page.evaluate("getComputedStyle(document.getElementById('chase-canvas')).opacity")
    assert float(op) == 0
    assert page.evaluate("[S.zoom, S.panX, S.panY]") == before
    assert float(page.locator("#zoom-slider").input_value()) == pytest.approx(before[0])
    assert page.errors == []


def test_scrub_jump_snaps_camera(page, server):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.evaluate("onScrub(100)")
    page.evaluate("onScrub(600)")                # front straight -> back straight: 180 deg
    c = _cam_to_car(page)
    assert abs(c["d"] - c["D"]) < 1, c
    assert c["err"] < 0.01, c                    # no swing across the track
    assert page.errors == []


def test_lap_change_in_chase_view(page, server):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.evaluate("onScrub(500)")
    page.locator('.lap-chip[data-lap="2"]').click()
    page.wait_for_function("() => S.lap === 2 && S.mapData && "
                           "document.getElementById('map-overlay').classList.contains('hidden')")
    page.wait_for_timeout(200)
    assert page.evaluate("Chase.state().built") is True
    c = _cam_to_car(page)
    assert c["d"] < 15 and c["err"] < 0.01, c
    assert page.errors == []


def test_chase_zoom_independent(page, server):
    _open_session(page, server, HILLY, lap=1)
    z0 = page.evaluate("[S.zoom, S.panX, S.panY]")
    _to_chase(page)
    plus = page.get_by_role("button", name="Zoom in")
    plus.click()
    plus.click()
    d = page.evaluate("S.chaseDist")
    assert d < 7.5
    assert _cam_to_car(page)["D"] == pytest.approx(d)
    page.locator("#map-canvas").hover()
    page.mouse.wheel(0, -300)
    assert page.evaluate("S.chaseDist") < d
    page.get_by_role("button", name="Zoom to fit").click()
    assert page.evaluate("S.chaseDist") == 7.5
    _radio(page, "Top-down").click()
    assert page.evaluate("[S.zoom, S.panX, S.panY]") == z0


def test_settle_loop_stops_when_paused(page, server):
    """Paused, the camera settles and then stops asking for frames: no
    perpetual requestAnimationFrame."""
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.evaluate("onScrub(330)")
    # < 2 % jump into the corner: smoothed, not snapped. Read `settling` in
    # the same task as the scrub: under full-suite load the settle loop
    # could finish before a separate evaluate round trip got there.
    assert page.evaluate("onScrub(345), Chase.state().settling") is True
    page.wait_for_function("() => !Chase.state().settling", timeout=5000)
    n = page.evaluate("Chase.state().renders")
    page.wait_for_timeout(500)
    assert page.evaluate("Chase.state().renders") == n
    assert _cam_to_car(page)["err"] < 0.001


# ── Round 2, Task 2: velocity-locked camera + View options menu ────────────

OPTS_KEY = "lmu-coach.chase.options"


def _open_options(page):
    btn = page.get_by_role("button", name="View options")
    btn.click()
    assert btn.get_attribute("aria-expanded") == "true"
    return btn


def _set_camera(page, label):
    """Pick a Camera option through the menu, as a user would, then close it."""
    btn = _open_options(page)
    page.get_by_role("radio", name=label, exact=True).check()
    page.keyboard.press("Escape")
    assert btn.get_attribute("aria-expanded") == "false"


def _render_at(page, f):
    """Put the playhead at fractional point f and draw one chase frame."""
    page.evaluate("f => { const n = S.mapData.driven_line.x.length;"
                  " S.playhead = f / (n - 1); Chase.render(); }", f)


def _arrow_screen_angle(page):
    """Angle (deg) of the projected velocity arrow off screen-vertical
    (0 = pointing straight up the screen)."""
    a = page.evaluate("Chase.state().arrowScreen")
    assert a is not None
    dx, dy = a["x1"] - a["x0"], a["y1"] - a["y0"]
    assert np.hypot(dx, dy) > 20, a                 # long enough to measure
    return float(np.degrees(np.arctan2(dx, -dy)))


def test_view_options_menu_keyboard_and_aria(page, server):
    _open_session(page, server, HILLY, lap=1)
    btn = page.get_by_role("button", name="View options")
    assert not btn.is_visible()                     # top-down: the menu is chase-only
    body_box = "() => { const r = document.querySelector('.map-body').getBoundingClientRect(); return [r.width, r.height]; }"
    box_top = page.evaluate(body_box)
    _to_chase(page)
    assert btn.is_visible()
    # showing the button must not reflow the stage (that would re-fit the
    # 2D map under the camera transition)
    assert page.evaluate(body_box) == box_top
    assert btn.get_attribute("aria-haspopup") == "dialog"
    btn.click()
    dlg = page.get_by_role("dialog", name="View options")
    assert dlg.is_visible()
    group = page.get_by_role("group", name="Camera")
    assert group.is_visible()
    # focus moves into the menu, onto the checked option
    assert page.evaluate("document.activeElement.name") == "chase-camera"
    assert page.get_by_role("radio", name="Smooth", exact=True).is_checked()
    page.keyboard.press("Escape")
    assert not dlg.is_visible()
    assert page.evaluate("document.activeElement === document.querySelector('#view-opts-btn')")
    assert page.errors == []


def test_locked_camera_arrow_vertical(page, server, shot_dir):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    _set_camera(page, "Locked to velocity")
    assert page.evaluate("Chase.state().camera") == "locked"
    n = page.evaluate("S.mapData.driven_line.x.length")
    rng = np.random.default_rng(7)
    worst = 0.0
    for f in rng.uniform(2, n - 3, 20):
        _render_at(page, float(f))
        worst = max(worst, abs(_arrow_screen_angle(page)))
    assert worst < 0.5, worst
    if shot_dir:
        page.screenshot(path=str(shot_dir / "locked.png"))
    assert page.errors == []


def test_smooth_camera_arrow_lags_in_corner(page, server):
    """Control for the test above: in Smooth mode the projected arrow does
    turn off vertical in a corner (otherwise the vertical test proves
    nothing about the projection)."""
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.evaluate("onScrub(330)")
    page.evaluate("onScrub(345)")                   # small step into the corner: smoothed
    assert abs(_arrow_screen_angle(page)) > 0.5


def test_locked_heading_wrap_has_no_jump(page, server):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    _set_camera(page, "Locked to velocity")
    hd = page.evaluate("S.mapData.driven_line.heading")
    wraps = [i for i in range(len(hd) - 1) if abs(hd[i + 1] - hd[i]) > np.pi]
    assert wraps, "synthetic lap should cross +-pi"
    i = wraps[0]
    prev = None
    for f in np.arange(max(0, i - 3), min(len(hd) - 1, i + 4), 0.05):
        _render_at(page, float(f))
        c = page.evaluate("Chase.state().course")
        assert c is not None and np.isfinite(c)
        if prev is not None:
            d = np.degrees(abs(np.arctan2(np.sin(c - prev), np.cos(c - prev))))
            assert d < 5, (f, d)
        prev = c
    assert page.errors == []


def test_locked_holds_course_at_standstill(page, server):
    """Review focus 3: where the car stands still the backend heading is
    atan2(0, 0) = 0 (a flip); Locked must hold the last real course, with
    no NaN."""
    a, b = 150, 160                                 # of the lap's 301 points
    def stall(route):
        resp = route.fetch()
        body = resp.json()
        dl = body["driven_line"]
        for k in range(a, b):                       # parked: same position, speed 0, heading 0
            dl["x"][k], dl["z"][k] = dl["x"][a], dl["z"][a]
            dl["heading"][k] = 0.0
            if dl.get("speed"):
                dl["speed"][k] = 0.0
        route.fulfill(response=resp, json=body)
    page.route("**/trackmap*", stall)
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    _set_camera(page, "Locked to velocity")
    assert page.evaluate(f"S.mapData.driven_line.heading[{a + 3}]") == 0
    h_before = page.evaluate(f"S.mapData.driven_line.heading[{a - 1}]")
    out = []
    for f in np.arange(a - 4, b + 4, 0.25):
        _render_at(page, float(f))
        out.append(page.evaluate("Chase.state().course"))
    assert all(c is not None and np.isfinite(c) for c in out)
    steps = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(out)))))
    assert steps.max() < 5, steps.max()
    mid = out[len(out) // 2]                        # f ~ 155, parked
    assert np.degrees(abs(np.angle(np.exp(1j * (mid - h_before))))) < 10
    assert page.errors == []


def test_switch_camera_mid_playback(page, server):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.evaluate("onScrub(200)")
    page.evaluate("togglePlay()")
    for label in ("Locked to velocity", "Smooth", "Locked to velocity", "Smooth"):
        _set_camera(page, label)
        page.wait_for_timeout(150)
    assert page.evaluate("S.playing") is True
    page.evaluate("togglePlay()")
    st = page.evaluate("Chase.state()")
    assert st["camera"] == "smooth" and np.isfinite(st["course"])
    assert page.errors == []


def test_camera_option_persists(page, server):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    _set_camera(page, "Locked to velocity")
    saved = page.evaluate(f"JSON.parse(localStorage.getItem('{OPTS_KEY}'))")
    # the other keys are reserved for the later options, with their defaults
    assert saved == {"camera": "locked", "car": "chevron", "limits": False, "slip": False}
    page.reload()
    page.locator(f'.session-item[title="{HILLY}"]').click()
    page.wait_for_function("() => S.mapData && document.getElementById('map-overlay').classList.contains('hidden')")
    _wait_built(page)
    assert page.evaluate("Chase.state().camera") == "locked"
    _open_options(page)
    assert page.get_by_role("radio", name="Locked to velocity", exact=True).is_checked()


def test_bad_stored_options_fall_back(page, server):
    page.add_init_script(f"localStorage.setItem('{OPTS_KEY}', '{{\"camera\": \"spin\", \"limits\": 3')")
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    assert page.evaluate("Chase.state().camera") == "smooth"
    assert page.errors == []


def test_top_down_unchanged_by_camera_option(page, server):
    """The 2D map keeps its own course function: Locked changes nothing in
    top-down (pixel-identical canvas)."""
    _open_session(page, server, HILLY, lap=1)
    page.evaluate("onScrub(345)")
    page.mouse.move(5, 5)                           # no hover over the map

    def round_trip(label):
        # Both shots go through the same chase round trip: a visit to the
        # chase view itself changes a few 2D pixels (seen with Smooth too),
        # which is not what this test is about.
        _to_chase(page)
        _set_camera(page, label)
        _radio(page, "Top-down").click()
        page.wait_for_function("() => Chase.state().mix === 0", timeout=10000)
        page.evaluate("drawMap()")
        return page.locator("#map-canvas").screenshot()

    before = round_trip("Smooth")
    after = round_trip("Locked to velocity")
    assert page.evaluate("S.chaseOpts.camera") == "locked"
    assert np.array_equal(_png_rgb(before), _png_rgb(after))


# ── Round 2, Task 3: surface contact, car model, track limits ─────────────

from serve_for_ui import LIMITS  # noqa: E402

# Lap 1 of the synthetic ring at fractions of the lap: the crest (s = 300 m,
# highest point), the downhill back straight, the kerb run-out (s = 620 m),
# the banked left-hander (s = 1100 m) and its entry ramp.
PLACES = (0.12, 0.25, 0.4, 0.52, 0.85, 0.92, 0.97)


def _set_car(page, label):
    btn = _open_options(page)
    page.get_by_role("radio", name=label, exact=True).check()
    page.keyboard.press("Escape")
    assert btn.get_attribute("aria-expanded") == "false"


def _contact_errors(page, n):
    worst, seen = 0.0, 0
    for frac in PLACES:
        _render_at(page, frac * (n - 1))
        cs = page.evaluate("Chase.state().contacts")
        assert cs, "no contacts reported"
        seen += 1
        worst = max(worst, max(abs(c["y"] - c["surface"]) for c in cs))
    return worst, seen


def test_chevron_rests_on_surface(page, server, shot_dir):
    """The chevron's three corners sit on the rendered surface (within 3 cm)
    on the crest, the slopes and the banked corner: height, pitch and roll
    come from the surface under them, not from the line and the recorded
    attitude (which sank one side into the ground on real recordings)."""
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    n = page.evaluate("S.mapData.driven_line.x.length")
    worst, seen = _contact_errors(page, n)
    assert seen == len(PLACES)
    assert worst < 0.03, worst
    # the bank is really there: the chevron rolls in the banked corner
    _render_at(page, 0.92 * (n - 1))
    assert abs(page.evaluate("Chase.state().placement.roll")) > np.radians(2)
    assert page.evaluate("Chase.state().marker") == "chevron"
    if shot_dir:
        page.screenshot(path=str(shot_dir / "chevron-banked.png"))
    assert page.errors == []


def test_car_mode_rests_on_surface_and_disposes(page, server, shot_dir):
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    g0 = page.evaluate("Chase.state().geometries")
    _set_car(page, "Car")
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    st = page.evaluate("Chase.state()")
    assert st["carModel"]["cls"] == "LMGT3" and st["carModel"]["source"] == "fallback"
    assert len(st["contacts"]) == 4
    n = page.evaluate("S.mapData.driven_line.x.length")
    worst, _ = _contact_errors(page, n)
    assert worst < 0.03, worst
    if shot_dir:
        _render_at(page, 0.92 * (n - 1))
        page.screenshot(path=str(shot_dir / "car-banked.png"))
    g1 = page.evaluate("Chase.state().geometries")
    assert g1 > g0
    # the fallback class is said in the menu
    _open_options(page)
    assert "drawn as LMGT3" in page.locator("#view-opts-car-note").inner_text()
    page.keyboard.press("Escape")
    # chevron <-> car: the car's geometry is freed every time, no leak
    for _ in range(3):
        _set_car(page, "Chevron")
        assert page.evaluate("Chase.state().marker") == "chevron"
        assert page.evaluate("Chase.state().geometries") == g0
        _set_car(page, "Car")
        page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
        assert page.evaluate("Chase.state().geometries") == g1
    st = page.evaluate("Chase.state()")
    assert st["carBuilds"] == 4 and st["carDisposals"] == 3
    # a lap change rebuilds it for the new payload (old one disposed)
    page.locator('.lap-chip[data-lap="2"]').click()
    page.wait_for_function("() => S.lap === 2 && S.mapData && Chase.state().built && "
                           "document.getElementById('map-overlay').classList.contains('hidden')")
    _render_at(page, 10)
    st = page.evaluate("Chase.state()")
    assert st["marker"] == "car" and st["carDisposals"] == 4 and st["geometries"] <= g1 + 50
    assert page.evaluate("JSON.parse(localStorage.getItem('lmu-coach.chase.options')).car") == "car"
    assert page.errors == []


def test_track_limits_tint_in_excursion(page, server, shot_dir):
    _open_session(page, server, LIMITS, lap=1)
    _to_chase(page)
    _set_car(page, "Car")
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    n = page.evaluate("S.mapData.driven_line.x.length")
    # off: nothing is tinted even in the excursion
    _render_at(page, 0.517 * (n - 1))
    assert page.evaluate("Chase.state().limitsMask") == 0
    btn = _open_options(page)
    box = page.get_by_role("checkbox", name="Show wheels past the edge")
    assert box.is_enabled()
    box.check()
    page.keyboard.press("Escape")
    assert btn.get_attribute("aria-expanded") == "false"
    masks = {}
    for frac in np.linspace(0.02, 0.98, 49):
        _render_at(page, float(frac) * (n - 1))
        masks[round(float(frac), 3)] = page.evaluate("Chase.state().limitsMask")
    out = {k: v for k, v in masks.items() if v}
    assert out, masks
    assert all(v & 0b0101 == 0 for v in out.values()), out          # right wheels only
    assert min(out) > 0.45 and max(out) < 0.58, out                 # s ~ 540-700 m of 1200
    _render_at(page, 0.517 * (n - 1))                                # s = 620: both right wheels
    st = page.evaluate("Chase.state()")
    assert st["limitsMask"] == 0b1010 and st["limitFrames"] > 0
    colours = st["wheelColours"]
    assert colours[1] == colours[3] != colours[0] == colours[2], colours
    if shot_dir:
        page.screenshot(path=str(shot_dir / "limits-kerb.png"))
    assert page.errors == []


def test_track_limits_unavailable_disables_toggle(page, server):
    def strip(route):
        r = route.fetch()
        d = r.json()
        d["track_limits"] = {"available": False, "note": "No Track Edge / Path Lateral in this file: track limits unavailable."}
        for k in ("lat", "edge_half", "wheels_out"):
            d["driven_line"][k] = []
        route.fulfill(response=r, json=d)
    page.route("**/trackmap*", strip)
    page.add_init_script("localStorage.setItem('lmu-coach.chase.options', JSON.stringify({camera:'smooth',car:'car',limits:true,slip:false}))")
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    _open_options(page)
    box = page.get_by_role("checkbox", name="Show wheels past the edge")
    assert box.is_disabled()
    assert "unavailable" in page.locator("#view-opts-limits-note").inner_text()
    page.keyboard.press("Escape")
    n = page.evaluate("S.mapData.driven_line.x.length")
    for frac in (0.2, 0.5, 0.8):
        _render_at(page, frac * (n - 1))
    assert page.evaluate("Chase.state().limitFrames") == 0
    assert page.errors == []


def test_placement_is_fast(page, server):
    """Resting the car on the surface costs well under 1 ms a frame."""
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    _set_car(page, "Car")
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    ms = page.evaluate("Chase.benchPlacement(2000)")
    assert ms < 0.2, ms


def test_broken_car_models_fall_back_to_chevron(page, server):
    """A car_models.json that cannot build a car (no classes) must fail once:
    message, chevron, the Car choice not shown as active, no error per frame."""
    page.route("**/static/car_models.json", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"version": 1, "classes": {}}'))
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.evaluate("setChaseOpt('car', 'car')")
    page.wait_for_function("() => Chase.state().carFailed", timeout=10000)
    n = page.evaluate("S.mapData.driven_line.x.length")
    for frac in (0.2, 0.4, 0.6, 0.8):
        _render_at(page, frac * (n - 1))
    st = page.evaluate("Chase.state()")
    assert st["marker"] == "chevron" and st["carBuilds"] == 0
    assert "Car model" in page.locator("#chase-msg").inner_text()
    _open_options(page)
    assert page.get_by_role("radio", name="Chevron", exact=True).is_checked()
    page.keyboard.press("Escape")
    assert len(page.errors) <= 1, page.errors


def test_marker_yaw_is_course_plus_body_offset(page, server):
    """The drawn body's forward axis, projected on the ground, points along
    course + bodyOff (the 2D chevron's angle) for both markers, also where
    the plane fit tilts it (hill, banked corner)."""
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    n = page.evaluate("S.mapData.driven_line.x.length")
    for label in ("Chevron", "Car"):
        _set_car(page, label)
        if label == "Car":
            page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
        for frac in PLACES:
            f = frac * (n - 1)
            _render_at(page, f)
            d = page.evaluate("""f => { const m = carMarkerAt(f), y = Chase.state().forwardYaw;
                const e = y - (m.ang + m.bodyOff); return Math.abs(Math.atan2(Math.sin(e), Math.cos(e))); }""", f)
            assert np.degrees(d) < 0.5, (label, frac, np.degrees(d))
    assert page.errors == []


# ── Round 2, Task 4: tyre slip arrows ──────────────────────────────────────

from serve_for_ui import SLIP  # noqa: E402
from synthetic_track import (RADIUS_M, SLIP_A_M, SLIP_B_M, SLIP_FRONT_DEG,  # noqa: E402
                             SLIP_HALF_TRACK_F, SLIP_HALF_TRACK_R, SPEED_MS)

SLIP_LABEL = "Show tyre slip arrows"
MID_CORNER = 0.42          # s ~ 504 m: middle of the first semicircle, slip fully built up


def _slip_truth_deg():
    """Known values on SLIP's constant-radius corner (synthetic_track): the
    car frame is right-positive, the ring's corners are left-handers."""
    r = -SPEED_MS / RADIUS_M
    th = lambda a, w: float(np.degrees(np.arctan2(r * a, SPEED_MS - r * w)))  # noqa: E731
    delta = th(SLIP_A_M, 0.0) - SLIP_FRONT_DEG
    return {"delta": delta,
            "fl": th(SLIP_A_M, -SLIP_HALF_TRACK_F) - delta, "fr": th(SLIP_A_M, SLIP_HALF_TRACK_F) - delta,
            "rl": th(-SLIP_B_M, -SLIP_HALF_TRACK_R), "rr": th(-SLIP_B_M, SLIP_HALF_TRACK_R)}


def _set_slip(page, on):
    btn = _open_options(page)
    box = page.get_by_role("checkbox", name=SLIP_LABEL)
    box.check() if on else box.uncheck()
    page.keyboard.press("Escape")
    assert btn.get_attribute("aria-expanded") == "false"


def _wrap_deg(a):
    return float((a + 180.0) % 360.0 - 180.0)


def _check_arrows(page, gain):
    """Each arrow's drawn direction against the body: rear = alpha x gain,
    front = delta + alpha x gain; and its world yaw = the body's + that."""
    want = _slip_truth_deg()
    st = page.evaluate("Chase.state()")
    sl = st["slip"]
    assert sl["arrows"] and [a["wheel"] for a in sl["arrows"]] == ["fl", "fr", "rl", "rr"]
    body = np.degrees(st["forwardYaw"])
    for a in sl["arrows"]:
        k = a["wheel"]
        assert abs(a["alphaDeg"] - want[k]) < 0.1, (k, a, want)
        rel = (want["delta"] if k in ("fl", "fr") else 0.0) + want[k] * gain
        assert abs(a["relDeg"] - rel) < 0.2, (k, a["relDeg"], rel)
        assert abs(_wrap_deg(np.degrees(a["worldYaw"]) - body - rel)) < 0.3, (k, a, body, rel)
    return st


def test_tyre_slip_arrows_at_known_angles(page, server, shot_dir):
    """Constant-radius corner with known slip angles: the arrows point at
    alpha x the slip-angle gain off their wheel's heading, the fronts also
    steered by the kinematic delta (yaw-rate sign from the correlation with
    Steering Pos: -1 on this file, whose Yaw Rate is left-positive)."""
    _open_session(page, server, SLIP, lap=0)
    _to_chase(page)
    page.evaluate("setSlipGain('1')")
    _set_slip(page, True)
    n = page.evaluate("S.mapData.driven_line.x.length")
    _render_at(page, MID_CORNER * (n - 1))
    st = _check_arrows(page, 1)
    steer = st["slip"]["steer"]
    assert steer["enabled"] and steer["sign"] == -1 and steer["corr"] > 0.99, steer
    assert abs(steer["deg"] - _slip_truth_deg()["delta"]) < 0.1, steer
    assert st["slip"]["wheelSteerDeg"] is None              # chevron: no wheels to steer
    page.evaluate("setSlipGain('5')")
    _render_at(page, MID_CORNER * (n - 1))
    _check_arrows(page, 5)
    # colour by |alpha| like the tyre HUD dial (white -> amber by 8 deg):
    # the fronts (2.5 deg) carry less blue than the rears (1.2 deg)
    cols = {a["wheel"]: a["colour"] for a in page.evaluate("Chase.state().slip.arrows")}
    assert int(cols["fl"][5:7], 16) < int(cols["rl"][5:7], 16), cols
    if shot_dir:
        page.screenshot(path=str(shot_dir / "slip-chevron.png"))
    # car mode: the same angles at the model's wheels, front wheels steered by delta
    _set_car(page, "Car")
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    _render_at(page, MID_CORNER * (n - 1))
    st = _check_arrows(page, 5)
    assert abs(st["slip"]["wheelSteerDeg"] - _slip_truth_deg()["delta"]) < 0.1, st["slip"]
    if shot_dir:
        page.screenshot(path=str(shot_dir / "slip-car.png"))
    # on a straight: no slip, arrows straight ahead, wheels straight
    _render_at(page, 0.15 * (n - 1))
    st = page.evaluate("Chase.state().slip")
    assert all(abs(a["relDeg"]) < 0.2 for a in st["arrows"]), st["arrows"]
    assert abs(st["wheelSteerDeg"]) < 0.2
    assert page.errors == []


def test_tyre_slip_toggle_disposes(page, server):
    _open_session(page, server, SLIP, lap=0)
    _to_chase(page)
    n = page.evaluate("S.mapData.driven_line.x.length")
    _render_at(page, MID_CORNER * (n - 1))
    g0 = page.evaluate("Chase.state().geometries")
    assert page.evaluate("Chase.state().slip.arrows") is None      # default off
    _set_slip(page, True)
    g1 = page.evaluate("Chase.state().geometries")
    assert g1 > g0
    # many frames, one build: nothing is created per frame
    for i in range(30):
        _render_at(page, (MID_CORNER + i * 0.005) * (n - 1))
    st = page.evaluate("Chase.state()")
    assert st["slip"]["builds"] == 1 and st["geometries"] == g1
    for _ in range(3):
        _set_slip(page, False)
        st = page.evaluate("Chase.state()")
        assert st["slip"]["arrows"] is None and st["geometries"] == g0
        _set_slip(page, True)
        assert page.evaluate("Chase.state().geometries") == g1
    assert page.evaluate("JSON.parse(localStorage.getItem('lmu-coach.chase.options')).slip") is True
    # chevron <-> car: rebuilt at the other marker's wheels (old one freed)
    b = page.evaluate("Chase.state().slip")
    _set_car(page, "Car")
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    _render_at(page, MID_CORNER * (n - 1))
    b2 = page.evaluate("Chase.state().slip")
    assert b2["builds"] == b["builds"] + 1 and b2["disposals"] == b["disposals"] + 1
    # wheels straight again once the option is off
    _set_slip(page, False)
    assert page.evaluate("Chase.state().slip.wheelSteerDeg") == 0
    _set_slip(page, True)
    # a lap change frees them; they come back for the new lap
    d0 = page.evaluate("Chase.state().slip.disposals")
    page.locator('.lap-chip[data-lap="1"]').click()
    page.wait_for_function("() => S.lap === 1 && S.mapData && Chase.state().built && "
                           "document.getElementById('map-overlay').classList.contains('hidden')")
    _render_at(page, 10)
    st = page.evaluate("Chase.state().slip")
    assert st["disposals"] > d0 and st["arrows"] is not None
    assert page.errors == []


def test_tyre_slip_unavailable_without_slip_angles(page, server):
    """HILLY has no Slip Angle table (like native LMU exports): the toggle is
    disabled with a note, a stored 'on' draws nothing, the car still renders."""
    page.add_init_script("localStorage.setItem('lmu-coach.chase.options', JSON.stringify({camera:'smooth',car:'car',limits:false,slip:true}))")
    _open_session(page, server, HILLY, lap=1)
    _to_chase(page)
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    _open_options(page)
    box = page.get_by_role("checkbox", name=SLIP_LABEL)
    assert box.is_disabled()
    assert "slip angle" in page.locator("#view-opts-slip-note").inner_text().lower()
    page.keyboard.press("Escape")
    n = page.evaluate("S.mapData.driven_line.x.length")
    for frac in (0.2, 0.42, 0.8):
        _render_at(page, frac * (n - 1))
        st = page.evaluate("Chase.state().slip")
        assert st["available"] is False and st["arrows"] is None and st["builds"] == 0
        assert st["wheelSteerDeg"] == 0
    assert page.errors == []


def test_tyre_slip_steer_off_when_uncorrelated(page, server):
    """Steering Pos that does not match the kinematic estimate (here: noise)
    leaves the front wheels straight and says why; the arrows still draw."""
    def scramble(route):
        r = route.fetch()
        d = r.json()
        rng = np.random.default_rng(3)
        d["driven_line"]["steering"] = rng.normal(0, 0.2, len(d["driven_line"]["steering"])).tolist()
        route.fulfill(response=r, json=d)
    page.route("**/trackmap*", scramble)
    page.add_init_script("localStorage.setItem('lmu-coach.chase.options', JSON.stringify({camera:'smooth',car:'car',limits:false,slip:true}))")
    _open_session(page, server, SLIP, lap=0)
    _to_chase(page)
    page.wait_for_function("() => Chase.state().marker === 'car'", timeout=10000)
    n = page.evaluate("S.mapData.driven_line.x.length")
    _render_at(page, MID_CORNER * (n - 1))
    st = page.evaluate("Chase.state().slip")
    assert st["steer"]["enabled"] is False and abs(st["steer"]["corr"]) < 0.6
    assert "correlat" in st["steer"]["reason"]
    assert st["wheelSteerDeg"] == 0 and st["arrows"] is not None
    # fronts then point at alpha x gain like the rears (no delta)
    gain = page.evaluate("S.slipGain")
    for a in st["arrows"]:
        assert abs(a["relDeg"] - a["alphaDeg"] * gain) < 0.05, a
    _open_options(page)
    assert "not steered" in page.locator("#view-opts-slip-note").inner_text()
    page.keyboard.press("Escape")
    assert page.errors == []
