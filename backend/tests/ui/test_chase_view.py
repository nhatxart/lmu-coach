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
    page.wait_for_function("() => Chase.state().mix === 1", timeout=10000)


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
    page.evaluate("onScrub(345)")                # < 2 % jump into the corner: smoothed, not snapped
    assert page.evaluate("Chase.state().settling") is True
    page.wait_for_function("() => !Chase.state().settling", timeout=5000)
    n = page.evaluate("Chase.state().renders")
    page.wait_for_timeout(500)
    assert page.evaluate("Chase.state().renders") == n
    assert _cam_to_car(page)["err"] < 0.001
