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
