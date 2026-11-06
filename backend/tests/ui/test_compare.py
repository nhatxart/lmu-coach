"""Lap comparison (plan 2026-11-04-lap-comparison): lap list flags, BEST."""

import pytest

pytest.importorskip("playwright.sync_api")

from serve_for_ui import PARTIAL


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
