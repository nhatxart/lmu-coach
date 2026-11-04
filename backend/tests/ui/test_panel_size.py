"""Telemetry panel sizing: the Size control (content scale, separate from the
panel's size), the docked width handle and the floating frame's edge grips
(plan 2026-11-04-resizable-telemetry-panel)."""

import pytest

pytest.importorskip("playwright.sync_api")

from serve_for_ui import HILLY


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


def _scale(page):
    return float(page.evaluate(
        "getComputedStyle(document.getElementById('right-panel')).getPropertyValue('--panel-scale')") or 1)


# ── Task 1: Size control ────────────────────────────────────────────────────

def test_size_buttons_scale_contents(page, server):
    _open_session(page, server, HILLY)
    page.get_by_role("button", name="Larger text and graphs").click()
    page.get_by_role("button", name="Larger text and graphs").click()
    assert abs(_scale(page) - 1.2) < 1e-6
    assert page.inner_text("#panel-scale-val") == "120%"
    page.get_by_role("button", name="Reset panel size").click()
    assert abs(_scale(page) - 1.0) < 1e-6
    assert page.errors == []


def test_scale_limits(page, server):
    _open_session(page, server, HILLY)
    for _ in range(20):
        page.get_by_role("button", name="Smaller text and graphs").click()
    assert abs(_scale(page) - 0.5) < 1e-6
    for _ in range(30):
        page.get_by_role("button", name="Larger text and graphs").click()
    assert abs(_scale(page) - 2.0) < 1e-6


def test_ctrl_wheel_scales(page, server):
    _open_session(page, server, HILLY)
    page.hover("#body-balance")
    page.keyboard.down("Control")
    page.mouse.wheel(0, -100)
    page.mouse.wheel(0, -100)
    page.keyboard.up("Control")
    page.wait_for_timeout(50)
    assert abs(_scale(page) - 1.1) < 1e-6


def test_scale_persists(page, server):
    _open_session(page, server, HILLY)
    page.get_by_role("button", name="Larger text and graphs").click()
    page.reload()
    page.wait_for_function("() => window.PanelSize")
    assert abs(_scale(page) - 1.1) < 1e-6


def test_bad_stored_values_fall_back(page, server):
    page.goto(server + "/")
    for v in ["abc", "7", "-1"]:
        page.evaluate(f"localStorage.setItem('lmu-coach.panel.scale', {v!r}); "
                      f"localStorage.setItem('lmu-coach.panel.w', {v!r})")
        page.reload()
        page.wait_for_function("() => window.PanelSize")
        assert abs(_scale(page) - 1.0) < 1e-6
        assert page.evaluate("document.getElementById('right-panel').style.width") == ""
    assert page.errors == []


def test_scale_redraws_traces(page, server):
    _open_session(page, server, HILLY)
    w0 = page.evaluate("document.getElementById('spd-canvas').width")
    page.get_by_role("button", name="Larger text and graphs").click()
    w1 = page.evaluate("document.getElementById('spd-canvas').width")
    assert w1 != w0      # bitmap re-sized to the new on-screen width, not stretched


# ── Task 2: docked width handle ─────────────────────────────────────────────

def _panel_w(page):
    return page.evaluate("Math.round(document.getElementById('right-panel').getBoundingClientRect().width)")


def _drag(page, sel, dx, dy=0):
    b = page.locator(sel).bounding_box()
    x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=8)
    page.mouse.up()


def test_docked_drag_resizes_and_persists(page, server):
    _open_session(page, server, HILLY)
    w0 = _panel_w(page)
    map0 = page.evaluate("document.getElementById('map-canvas').getBoundingClientRect().width")
    _drag(page, "#panel-splitter", -200)
    assert abs(_panel_w(page) - (w0 + 200)) <= 2
    assert page.evaluate("document.getElementById('map-canvas').getBoundingClientRect().width") < map0 - 150
    page.reload()
    page.wait_for_function("() => window.PanelSize")
    assert abs(_panel_w(page) - (w0 + 200)) <= 2
    page.get_by_role("button", name="Reset panel size").click()
    assert _panel_w(page) == w0


def test_docked_width_limits(page, server):
    _open_session(page, server, HILLY)
    _drag(page, "#panel-splitter", 2000)
    assert _panel_w(page) == 300
    _drag(page, "#panel-splitter", -3000)
    assert _panel_w(page) == min(1200, 1600 - 520)


def test_docked_keyboard_and_reset(page, server):
    _open_session(page, server, HILLY)
    w0 = _panel_w(page)
    page.focus("#panel-splitter")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("Shift+ArrowLeft")
    assert _panel_w(page) == w0 + 80
    assert int(page.get_attribute("#panel-splitter", "aria-valuenow")) == w0 + 80
    page.dblclick("#panel-splitter")
    assert _panel_w(page) == w0
    assert page.evaluate("localStorage.getItem('lmu-coach.panel.w')") is None


def test_docked_width_clamped_to_window(page, server):
    _open_session(page, server, HILLY)
    _drag(page, "#panel-splitter", -3000)            # 1080 at 1600 wide
    page.set_viewport_size({"width": 1200, "height": 1000})
    page.wait_for_timeout(100)
    assert _panel_w(page) <= 1200 - 520


def test_stacked_layout_ignores_width(page, server):
    page.goto(server + "/")
    page.evaluate("localStorage.setItem('lmu-coach.panel.w','700')")
    page.set_viewport_size({"width": 800, "height": 1000})
    page.reload()
    page.wait_for_function("() => window.PanelSize")
    assert page.evaluate("document.getElementById('right-panel').style.width") == ""
    assert not page.is_visible("#panel-splitter")
    assert page.evaluate("localStorage.getItem('lmu-coach.panel.w')") == "700"
