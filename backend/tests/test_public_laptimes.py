"""Public lap times (design §14 R9): the import tool's parsing on hand-written
fixtures in the sheet's format (never the live sheet) and the offline matching
rules of app.public_laptimes."""
import importlib.util
import json
import socket
from pathlib import Path

import pytest

from app import public_laptimes as PL
from app.ideal_lap import public_block

FIX = Path(__file__).parent / "fixtures" / "public_laptimes"
TOOL = Path(__file__).resolve().parents[1] / "tools" / "import_public_laptimes.py"


def _tool():
    spec = importlib.util.spec_from_file_location("import_public_laptimes", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def data():
    t = _tool()
    tabs = {"COTA": (FIX / "cota.csv").read_text(encoding="utf-8"),
            "Fuji (chicane)": (FIX / "fuji_chicane.csv").read_text(encoding="utf-8"),
            "Fuji (classic)": (FIX / "fuji_classic.csv").read_text(encoding="utf-8")}
    return t.build(tabs, (FIX / "laptimes.csv").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test tried to go online")
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_parse_layout_tab(data):
    cars = data["layouts"]["COTA"]["cars"]
    assert {"class": "LMGT3", "car": "Porsche 992", "patch": "1.4", "time_s": 125.226} in cars
    assert {"class": "LMGT3", "car": "Porsche 992", "patch": "1.24", "time_s": 125.675} in cars
    assert {"class": "LMH", "car": "Ferrari 499P", "patch": "1.42", "time_s": 111.98} in cars
    assert all(c["time_s"] < 600 for c in cars)                     # padding rows (1440:00.000) skipped
    assert len(cars) == 9


def test_parse_laptimes_tab(data):
    assert data["sheet_updated"] == "2026.11.16. 10:44 CET"
    assert data["layouts"]["COTA"]["class_ref"] == {"LMGT3": {"time_s": 125.33, "patch": "1.4+"},
                                                     "LMH": {"time_s": 111.41, "patch": "1.4+"}}
    assert data["layouts"]["COTA"]["slug"] == "cota" and data["layouts"]["Fuji (chicane)"]["layout_len_m"] == 4527.0


def test_tab_list_from_pubhtml():
    html = ('items.push({name: "Laptimes", pageUrl: "x", gid: "1766901750",initialSheet: (1)});'
            'items.push({name: "Fuji (chicane)", pageUrl: "y", gid: "1467811719",initialSheet: (0)});')
    assert _tool().parse_tabs(html) == {"Laptimes": "1766901750", "Fuji (chicane)": "1467811719"}


def test_match_exact_older_and_class_only(data):
    m = PL.match("cota", 5490.9, "Ferrari 499P", "Hypercar", "14200", data=data)
    assert (m["car_best_s"], m["patch"], m["patch_match"], m["class_ref_s"]) == (111.98, "1.42", "exact", 111.41)
    m = PL.match("cota", 5490.9, "Porsche 911 GT3 R LMGT3", "LMGT3", "14200", data=data)
    assert (m["car"], m["car_best_s"], m["patch"], m["patch_match"]) == ("Porsche 992", 125.226, "1.4", "older")
    m = PL.match("cota", 5490.9, "Ford Mustang LMGT3", "LMGT3", "14200", data=data)
    assert (m["car_best_s"], m["patch"]) == (125.127, "1.4")         # best of the newest patch <= 1.42
    m = PL.match("cota", 5490.9, "Unknown Car GT3", "LMGT3", "14200", data=data)
    assert m["car_best_s"] is None and m["class_ref_s"] == 125.33
    m = PL.match("cota", 5490.9, "Porsche 911 GT3 R LMGT3", "LMGT3", "11000", data=data)
    assert m["patch_match"] == "newer" and m["patch"] == "1.4"        # every patch is newer than ours


def test_fuji_needs_the_length(data):
    assert PL.match("fuji", None, "BMW M4 LMGT3", "LMGT3", "14200", data=data) is None
    assert PL.match("fuji", 4400.0, "BMW M4 LMGT3", "LMGT3", "14200", data=data) is None
    m = PL.match("fuji", 4527.3, "BMW M4 LMGT3", "LMGT3", "14200", data=data)
    assert m["tab"] == "Fuji (chicane)" and m["car_best_s"] == 99.7 and m["patch_match"] == "exact"


def test_patch_and_band():
    assert PL.patch_of("14200") == "1.42" and PL.patch_of(None) is None
    assert PL.patch_key("1.4 ") == (1, 40) < PL.patch_key("1.42") and PL.patch_key("0.5") == (0, 50)
    assert [PL.band(p) for p in (100.2, 101.0, 102.9, 105.3, 106.2, 108.0)] == [
        "Alien", "Competitive", "Good", "Midpack", "Tail-ender", "Offline"]


def test_load_is_offline_and_missing_file_is_none(tmp_path, data):
    assert PL.load(tmp_path / "none.json") is None
    p = tmp_path / "public_laptimes.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert PL.load(p)["layouts"]["COTA"]["slug"] == "cota"


def test_public_block_flags_an_ideal_faster_than_the_public_best(monkeypatch, data):
    monkeypatch.setattr(PL, "load", lambda path=None: data)
    pub, chk = public_block("Porsche 911 GT3 R LMGT3", "LMGT3", "cota", 5490.9, "14200", 131.86, 130.87, 129.9)
    assert chk == "ok" and pub["driven_pct"] == pytest.approx(105.30, abs=0.01) and pub["band"] == "Midpack"
    pub, chk = public_block("Porsche 911 GT3 R LMGT3", "LMGT3", "cota", 5490.9, "14200", 126.5, 125.5, 124.0)
    assert chk == "faster_than_public_best"
    assert public_block("Porsche 911 GT3 R LMGT3", "LMGT3", "fuji", None, "14200", 100, 99, 98) == (None, None)
