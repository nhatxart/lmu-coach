"""Keep laps (plan 2026-11-08-keep-laps, Task 6; spec 4.1 / 4.4a): keep mode
on the lap bar, the keep dialog and its results, the separate "Delete
recording..." button with its warning, the Kept laps group and KEPT chips.

Every test runs on its own backend over WRITABLE tmp folders (keep_server):
keeping writes files, a delete removes one."""

import json

import pytest

pytest.importorskip("playwright.sync_api")

from serve_for_ui import KEEP_NATIVE, KEEP_ORIG

LOADED = "() => S.mapData && document.getElementById('map-overlay').classList.contains('hidden')"
STAMP = "2026-11-12T10_00_00Z"
OTHER_TEXT = ("Its other {n} laps and their data for future grip-profile rebuilds are deleted "
              "with it. This cannot be undone.")


def _lap_text(t):
    ms = int(round(t * 1000))
    return f"{ms // 60000}m{ms % 60000 // 1000:02d}.{ms % 1000:03d}"


def _kept_name(page, lap):
    t = page.evaluate(f"S.laps.find(l => l.lap_number === {lap}).lap_time_s")
    return f"Synthetic Hill Ring-Synthetic Car _1 L{lap} {_lap_text(t)}-{STAMP}.duckdb"


def _open(page, base, file, goto=True):
    if goto:
        page.goto(base + "/")
    page.locator(f'.session-item[title="{file}"]').click()
    page.wait_for_function(f"() => S.file === {file!r} && ({LOADED})()")


def _keep_ready(page):
    """The Keep button has its state (it waits for the session index)."""
    page.wait_for_function("() => { const b = document.getElementById('keep-btn');"
                           " return b && b.dataset.state && b.dataset.state !== 'pending'; }")


def _record(page):
    """POST bodies to /keep and /delete, in order."""
    sent = []

    def on_req(r):
        if r.method == "POST" and (r.url.endswith("/keep") or r.url.endswith("/delete")):
            sent.append((r.url.rsplit("/", 1)[1], json.loads(r.post_data or "null")))
    page.on("request", on_req)
    return sent


def _keep_via_ui(page, laps):
    _keep_ready(page)
    page.locator("#keep-btn").click()
    for n in laps:
        page.locator(f'.lap-chip[data-lap="{n}"]').click()
    page.locator("#keep-btn").click()
    page.wait_for_function("() => document.getElementById('keep-dlg').open")


def _wait_results(page):
    page.wait_for_function("() => document.querySelector('#kd-done')", timeout=120000)


def _active_id(page):
    return page.evaluate("document.activeElement && document.activeElement.id")


def test_keep_two_laps_flow(page, keep_server):
    k = keep_server
    sent = _record(page)
    _open(page, k.base, KEEP_ORIG)
    _keep_ready(page)
    lap_before = page.evaluate("S.lap")
    btn = page.locator("#keep-btn")
    assert btn.inner_text() == "Keep laps…" and btn.is_enabled()
    btn.click()
    chips = page.locator(".lap-chip")
    assert chips.count() == 3
    for i in range(3):
        assert chips.nth(i).get_attribute("aria-pressed") == "false"
    assert btn.inner_text() == "Cancel"
    page.locator('.lap-chip[data-lap="0"]').click()
    assert btn.inner_text() == "Keep 1 lap"
    page.locator('.lap-chip[data-lap="2"]').click()
    assert page.locator('.lap-chip[data-lap="2"]').get_attribute("aria-pressed") == "true"
    assert btn.inner_text() == "Keep 2 laps"
    assert page.evaluate("S.lap") == lap_before          # in keep mode a chip click selects, not loads
    names = [_kept_name(page, 0), _kept_name(page, 2)]
    btn.click()
    page.wait_for_function("() => document.getElementById('keep-dlg').open")
    body = page.inner_text("#kd-body")
    assert "The recording is not changed" in body
    assert page.eval_on_selector_all(".kd-lap .kd-name", "es => es.map(e => e.textContent)") == names
    assert sent == []                                   # nothing before the dialog's button
    assert page.locator("#kd-go").inner_text() == "Keep 2 laps"
    page.locator("#kd-go").click()
    _wait_results(page)
    assert sent == [("keep", {"laps": [0, 2]})]
    res = page.locator(".kd-res")
    assert res.count() == 2
    for i, n in enumerate((0, 2)):
        t = res.nth(i)
        assert t.get_attribute("data-status") == "kept"
        assert f"L{n}" in t.inner_text() and "checked" in t.inner_text()
    for n in names:
        assert (k.kept / n).is_file()
    # all checked: a separate Delete button, but Done is the default
    assert page.locator("#kd-delete").inner_text() == "Delete recording…"
    assert _active_id(page) == "kd-done"
    page.locator("#kd-done").click()
    assert not page.evaluate("document.getElementById('keep-dlg').open")
    assert [x[0] for x in sent] == ["keep"]
    # refreshed: the kept laps are listed, the chips say KEPT, the file stays open
    page.wait_for_function("() => document.querySelectorAll('.lap-chip .kept-tag').length === 2")
    assert page.evaluate("S.file") == KEEP_ORIG
    for n in names:
        assert page.locator(f'.session-item[title="{n}"]').count() == 1
    assert (k.sessions / KEEP_ORIG).is_file()
    assert page.locator("#keep-btn").inner_text() == "Keep laps…"
    assert page.errors == []


def test_delete_needs_button_and_warning(page, keep_server):
    k = keep_server
    sent = _record(page)
    _open(page, k.base, KEEP_ORIG)
    _keep_via_ui(page, [1])
    page.locator("#kd-go").click()
    _wait_results(page)
    name = _kept_name(page, 1)
    page.locator("#kd-delete").click()
    warn = page.inner_text("#kd-body")
    assert KEEP_ORIG in warn
    assert OTHER_TEXT.format(n=2) in warn
    assert _active_id(page) == "kd-cancel"              # the default is Cancel
    assert page.locator("#kd-delete-go").inner_text() == "Delete permanently"
    assert [x[0] for x in sent] == ["keep"]
    # Esc at the warning: closes, sends nothing
    page.keyboard.press("Escape")
    assert not page.evaluate("document.getElementById('keep-dlg').open")
    assert [x[0] for x in sent] == ["keep"] and (k.sessions / KEEP_ORIG).is_file()

    # again: the lap is now already kept (marked), checked again, not rewritten
    page.wait_for_function("() => document.querySelectorAll('.lap-chip .kept-tag').length === 1")
    _keep_via_ui(page, [1])
    assert "already kept" in page.inner_text(".kd-lap")
    page.locator("#kd-go").click()
    _wait_results(page)
    assert page.locator(".kd-res").get_attribute("data-status") == "already_kept"
    # Cancel at the warning: back to the results, sends nothing
    page.locator("#kd-delete").click()
    page.locator("#kd-cancel").click()
    assert page.evaluate("document.getElementById('keep-dlg').open")
    assert page.locator("#kd-done").count() == 1
    assert [x[0] for x in sent] == ["keep", "keep"] and (k.sessions / KEEP_ORIG).is_file()
    # Delete recording..., then Delete permanently: the one request
    page.locator("#kd-delete").click()
    page.locator("#kd-delete-go").click()
    page.wait_for_function(f"() => S.file === {name!r} && S.source === 'reference_laps' && ({LOADED})()",
                           timeout=60000)
    assert sent[-1] == ("delete", {"confirm": KEEP_ORIG, "kept_files": [name]})
    assert [x[0] for x in sent] == ["keep", "keep", "delete"]
    assert not (k.sessions / KEEP_ORIG).exists() and (k.kept / name).is_file()
    assert not page.evaluate("document.getElementById('keep-dlg').open")
    assert page.locator(f'.session-item[title="{KEEP_ORIG}"]').count() == 0
    assert page.locator(f'.session-item[title="{name}"].active').count() == 1
    assert page.errors == []


def test_no_delete_button_when_a_check_failed(page, keep_server):
    k = keep_server
    sent = _record(page)
    _open(page, k.base, KEEP_ORIG)
    n0, n1 = _kept_name(page, 0), _kept_name(page, 1)
    body = {"ok": True, "all_checked": False, "results": [
        {"lap": 0, "kept_file": n0, "status": "kept", "lap_time_original": 1.0,
         "lap_time_kept": 1.0, "error": None},
        {"lap": 1, "kept_file": n1, "status": "failed", "lap_time_original": 1.0,
         "lap_time_kept": 1.5, "error": "the kept file failed its check and was removed: lap time differs"}]}
    page.route("**/keep", lambda r: r.fulfill(status=200, content_type="application/json",
                                              body=json.dumps(body)))
    _keep_via_ui(page, [0, 1])
    page.locator("#kd-go").click()
    _wait_results(page)
    assert page.locator('.kd-res[data-status="failed"]').inner_text().count("lap time differs") == 1
    assert page.locator('.kd-res[data-status="kept"]').count() == 1
    assert page.locator("#kd-delete").count() == 0
    assert _active_id(page) == "kd-done"
    page.locator("#kd-done").click()

    # a refusal (409) shows inline as an alert, and no Delete either
    page.unroute("**/keep")
    page.route("**/keep", lambda r: r.fulfill(status=409, content_type="application/json", body=json.dumps(
        {"ok": False, "results": [], "error": "The kept-laps folder is read-only: rebuild with the new docker-compose.yml"})))
    _keep_via_ui(page, [2])
    page.locator("#kd-go").click()
    _wait_results(page)
    alert = page.locator('#keep-dlg [role="alert"]')
    assert "read-only" in alert.inner_text()
    assert page.locator("#kd-delete").count() == 0
    assert [x[0] for x in sent] == ["keep", "keep"]
    assert (k.sessions / KEEP_ORIG).is_file()
    # Chromium logs the routed 409 as a failed resource load; nothing else
    assert all("status of 409" in e for e in page.errors)


def test_keep_disabled_for_kept_and_native_files(page, keep_server):
    k = keep_server
    r = k.keep([1])
    assert r["all_checked"]
    kept = r["results"][0]["kept_file"]
    page.goto(k.base + "/")
    _open(page, k.base, KEEP_NATIVE, goto=False)
    _keep_ready(page)
    btn = page.locator("#keep-btn")
    assert btn.is_disabled() and "lmu-coach recorder" in btn.get_attribute("title")
    _open(page, k.base, kept, goto=False)
    _keep_ready(page)
    assert btn.is_disabled() and "kept lap" in btn.get_attribute("title")
    _open(page, k.base, KEEP_ORIG, goto=False)
    _keep_ready(page)
    assert btn.is_enabled() and not btn.get_attribute("title")
    assert page.errors == []


def test_kept_group_and_tags(page, keep_server):
    k = keep_server
    kept = k.keep([1])["results"][0]["kept_file"]
    page.goto(k.base + "/")
    page.wait_for_selector(".session-item")
    groups = page.eval_on_selector_all(".session-group", "es => es.map(e => e.textContent)")
    assert groups == ["My sessions", "Kept laps"]
    item = page.locator(f'.session-item[title="{kept}"]')
    assert item.locator(".s-tag").inner_text().lower() == "kept"
    assert page.locator(f'.session-item[title="{KEEP_ORIG}"] .s-tag').count() == 0
    _open(page, k.base, KEEP_ORIG, goto=False)
    page.wait_for_function("() => document.querySelector('.lap-chip .kept-tag')")
    tags = page.eval_on_selector_all(".lap-chip", "es => es.map(e => [e.dataset.lap, "
                                     "(e.querySelector('.kept-tag') || {}).textContent || ''])")
    assert tags == [["0", ""], ["1", "KEPT"], ["2", ""]]
    assert page.errors == []
