"""Ideal lap in the Compare tab (design 2026-11-10-ideal-lap-design.md §7, §12
Q1-Q6/Q11, §14/§15 flags): the button adds the ideal lap as a ghost of your own
line plus a per-corner table; the headline is always a range; a refusal is a
message, never a number; low confidence is marked; a lap change drops a
pending request; everything works from the keyboard.

The server runs with UI_IDEAL_FAKE=1 (serve_for_ui.install_fake_ideal_store):
a known car per file instead of a fit; the solver, corners and the trace on
the driven line are the real code."""

import re
from urllib.parse import quote

import pytest

pytest.importorskip("playwright.sync_api")

from serve_for_ui import (CMP_B, IDEAL_DELAY_S, IDEAL_LOW, IDEAL_OK, IDEAL_REFUSAL, IDEAL_REFUSED,
                          IDEAL_SLOW)

LOADED = "() => S.mapData && document.getElementById('map-overlay').classList.contains('hidden')"
IDEAL_COL = "#FDE047"
HAS_IDEAL = "() => CMP.ghosts.some(g => g.ideal)"


def _open(page, base, file, lap=None):
    page.goto(base + "/")
    page.locator(f'.session-item[title="{file}"]').click()
    page.wait_for_function(f"() => S.file === {file!r} && ({LOADED})()")
    if lap is not None and page.evaluate("S.lap") != lap:
        page.locator(f'.lap-chip[data-lap="{lap}"]').click()
        page.wait_for_function(f"() => S.lap === {lap} && ({LOADED})()")
    page.locator("#tab-compare").click()
    page.wait_for_function("() => document.querySelectorAll('#cmp-file option').length > 0")


def _payload(page, file, lap):
    return page.evaluate("async ([f, l]) => (await fetch(`/sessions/sessions/${encodeURIComponent(f)}/ideal-lap"
                         "?lap_number=${l}`)).json()", [file, lap])


def _add_ideal(page, timeout=60000):
    page.locator("#cmp-ideal-btn").click()
    page.wait_for_function(HAS_IDEAL, timeout=timeout)


def test_button_adds_the_ideal_ghost_and_the_corner_table(page, ideal_server):
    _open(page, ideal_server, IDEAL_OK, 1)
    btn = page.locator("#cmp-ideal-btn")
    assert btn.inner_text().strip() == "Ideal lap (this car, this line)"
    _add_ideal(page)
    d = _payload(page, IDEAL_OK, 1)
    assert d["status"] == "ok"
    g = page.evaluate("(() => { const g = CMP.ghosts.find(g => g.ideal); return {col: g.col, lap: g.lap, "
                      "lapTime: g.lapTime, n: g.L.n, src: g.src, file: g.file}; })()")
    # the lap's own line (Q2) with the ideal's clock: same points as your lap
    assert g["col"] == IDEAL_COL and g["lap"] == 1 and g["file"] == IDEAL_OK
    assert g["n"] == page.evaluate("S.mapData.driven_line.x.length")
    assert g["lapTime"] == pytest.approx(d["ideal"]["time_s"], abs=1e-3)
    # the list names it, and it does not use one of the four ghost slots
    row = page.locator('#cmp-list .cmp-row[data-ideal="1"]')
    assert row.count() == 1 and "Ideal" in row.inner_text()
    assert row.locator("button").get_attribute("aria-label") == "Remove the ideal lap"
    # speed trace: a series in the ideal's colour equal to the trace's speeds
    page.evaluate("CMP.onPlayhead()")
    spd = page.evaluate("CMP._plotted('cmp-spd')")
    ideal = [s for s in spd["series"] if s["col"] == IDEAL_COL]
    assert len(ideal) == 1 and ideal[0]["v"] == pytest.approx(d["trace"]["v_kmh"], abs=0.01)
    # live delta at the end of the lap = the official lap time minus the ideal's
    page.evaluate("() => { setScrub(1000); onScrub('1000'); }")
    dl = page.evaluate("CMP.deltaAt(CMP.ghosts.find(g => g.ideal))")
    assert dl == pytest.approx(d["lap_time_s"] - d["ideal"]["time_s"], abs=0.02)
    # the table: one row per corner and straight, corners first column a button
    rows = page.locator("#ideal-table tbody tr")
    assert rows.count() == len(d["corners"])
    corner = page.locator("#ideal-table tbody tr[data-kind='corner']").first
    c0 = next(c for c in d["corners"] if c["kind"] == "corner")
    txt = corner.inner_text()
    assert c0["name"] in txt
    assert f"{c0['gain_min_s']:.2f}" in txt.replace("−", "-") and f"{c0['gain_s']:.2f}" in txt.replace("−", "-")
    lim = {"front": "Front", "rear": "Rear", "brake": "Brakes", "traction": "Traction", "power": "Power",
           "measured": "Your limit", "driven": "Your lap"}[c0["binding"]]
    assert lim in txt
    assert f"{round(100 * c0['u_front_ref'])} %" in txt and f"{round(100 * c0['u_rear_ref'])} %" in txt
    # Q5: each tyre's load share on the row
    fl = round(100 * c0["load_share"]["front"][0])
    assert re.search(rf"FL {fl}\b", txt)
    # no public times here (no public_laptimes.json): the line is hidden (Q11)
    assert page.locator("#ideal-public").is_hidden()
    assert page.errors == []


def test_headline_is_always_the_range_with_the_ceiling(page, ideal_server):
    _open(page, ideal_server, IDEAL_OK, 1)
    _add_ideal(page)
    d = _payload(page, IDEAL_OK, 1)
    head = page.inner_text("#ideal-head").replace("−", "-")
    i = d["ideal"]
    assert f"at least -{i['gain_min_s']:.2f} s" in head.lower() and f"expected -{i['gain_s']:.2f} s" in head.lower()
    ceil = page.inner_text("#ideal-ceiling").replace("−", "-")
    assert f"Setup potential: up to -{d['ceiling']['gain_s']:.2f} s" in ceil
    assert page.locator("#ideal-conf").get_attribute("data-conf") == "medium"
    assert "Low confidence" not in page.inner_text("#cmp-ideal")
    assert page.errors == []


def test_a_refusal_shows_its_reason_and_no_number(page, ideal_server):
    _open(page, ideal_server, IDEAL_REFUSED, 1)
    page.locator("#cmp-ideal-btn").click()
    page.wait_for_function("() => document.getElementById('ideal-msg').textContent.includes('needed')")
    msg = page.inner_text("#ideal-msg")
    assert IDEAL_REFUSAL in msg
    assert page.evaluate(HAS_IDEAL) is False
    assert page.locator("#ideal-head").is_hidden() and page.locator("#ideal-table").count() == 0
    assert not re.search(r"\d+\.\d+ s", page.inner_text("#cmp-ideal"))
    assert page.locator("#ideal-msg").get_attribute("role") == "status"
    assert page.errors == []


def test_low_confidence_is_marked(page, ideal_server):
    _open(page, ideal_server, IDEAL_LOW, 1)
    _add_ideal(page)
    conf = page.locator("#ideal-conf")
    assert conf.get_attribute("data-conf") == "low" and "Low confidence" in conf.inner_text()
    card = page.inner_text("#cmp-ideal")
    assert "estimate from other tracks" in card.lower()
    assert "setup differs" in card.lower() and "camber_f" in card
    # Q11: the public line when the server has public times
    pub = page.locator("#ideal-public")
    assert pub.is_visible() and "public best" in pub.inner_text() and "Midpack" in pub.inner_text()
    assert page.errors == []


def test_a_lap_change_drops_the_pending_request(page, ideal_server):
    _open(page, ideal_server, IDEAL_SLOW, 1)
    page.locator("#cmp-ideal-btn").click()
    # progress while it runs, and the UI stays usable
    page.wait_for_function("() => document.getElementById('cmp-ideal-btn').getAttribute('aria-busy') === 'true'")
    assert "Computing" in page.inner_text("#ideal-msg")
    assert page.locator("#ideal-cancel").is_visible()
    page.locator('.lap-chip[data-lap="0"]').click()
    page.wait_for_function(f"() => S.lap === 0 && ({LOADED})()")
    page.wait_for_timeout(int((IDEAL_DELAY_S + 1.5) * 1000))
    assert page.evaluate(HAS_IDEAL) is False
    assert page.locator("#cmp-ideal-btn").get_attribute("aria-busy") != "true"
    assert page.locator("#ideal-head").is_hidden()
    assert page.errors == []


def test_a_lap_change_removes_the_ideal_of_the_old_lap(page, ideal_server):
    _open(page, ideal_server, IDEAL_OK, 1)
    _add_ideal(page)
    page.evaluate("([s, f, l]) => CMP.add(s, f, l)", ["sessions", CMP_B, 0])
    page.wait_for_function("() => CMP.ghosts.length === 2")
    page.locator('.lap-chip[data-lap="2"]').click()
    page.wait_for_function(f"() => S.lap === 2 && ({LOADED})()")
    page.wait_for_function("() => CMP.ghosts.length === 1")
    assert page.evaluate(HAS_IDEAL) is False             # it belonged to L1's line
    assert page.locator("#ideal-table").count() == 0 and page.locator("#ideal-head").is_hidden()
    assert page.errors == []


def test_cancel_and_remove(page, ideal_server):
    _open(page, ideal_server, IDEAL_SLOW, 1)
    page.locator("#cmp-ideal-btn").click()
    page.wait_for_function("() => document.getElementById('cmp-ideal-btn').getAttribute('aria-busy') === 'true'")
    page.locator("#ideal-cancel").click()
    assert page.locator("#cmp-ideal-btn").get_attribute("aria-busy") != "true"
    page.wait_for_timeout(int((IDEAL_DELAY_S + 1.5) * 1000))
    assert page.evaluate(HAS_IDEAL) is False
    # a full one, then removed from the list: ghost and table go
    _open(page, ideal_server, IDEAL_OK, 1)
    _add_ideal(page)
    page.locator('#cmp-list .cmp-row[data-ideal="1"] button').click()
    assert page.evaluate(HAS_IDEAL) is False and page.locator("#ideal-table").count() == 0
    assert page.errors == []


def test_keyboard(page, ideal_server):
    _open(page, ideal_server, IDEAL_OK, 1)
    page.focus("#cmp-ideal-btn")
    page.keyboard.press("Enter")
    page.wait_for_function(HAS_IDEAL, timeout=60000)
    # each corner's name is a button: Enter zooms every trace to that corner
    b = page.locator("#ideal-table tbody tr[data-kind='corner'] button").first
    b.focus()
    assert page.evaluate("document.activeElement.closest('#ideal-table') !== null")
    page.keyboard.press("Enter")
    s0, s1 = float(b.get_attribute("data-s0")), float(b.get_attribute("data-s1"))
    v = page.evaluate("CMP.view")
    assert v["s0"] <= s0 + 1 and v["s1"] >= s1 - 1 and v["s1"] - v["s0"] < page.evaluate("S.mapData.lap_len_m || 1e9")
    # the table is a real table with headers
    assert page.locator("#ideal-table th[scope='col']").count() >= 4
    assert page.locator("#ideal-table caption").count() == 1
    # the button says it is on screen (pressed again: no second request)
    page.focus("#cmp-ideal-btn")
    page.keyboard.press("Enter")
    assert "already on screen" in page.inner_text("#cmp-status")
    assert page.evaluate("CMP.ghosts.filter(g => g.ideal).length") == 1
    assert page.errors == []
