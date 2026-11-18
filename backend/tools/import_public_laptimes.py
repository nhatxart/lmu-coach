"""
Import Ohne Speed's public LMU lap-time sheet (design 2026-11-10-ideal-lap-design.md
§14 R9) into data/track_reference/public_laptimes.json. The ONLY code of the app
that goes online, and only when you run it: plain GETs of the sheet's published
pages (pubhtml for the tab list, then one CSV per tab), generic User-Agent,
nothing of ours sent. The backend reads the JSON offline (public_laptimes.py).

    python -I backend/tools/import_public_laptimes.py                 # writes the JSON
    python -I backend/tools/import_public_laptimes.py --dry-run       # fetch and print only
    python -I backend/tools/import_public_laptimes.py --out C:\\temp\\pl.json

Tabs: `Laptimes` (per class and track: the class reference "Class avgW" = 100 %,
its patch, the band limits) and one tab per layout (rows: patch, class, car,
time; the best known hotlap per car and patch, older patches further down).

Tab -> our track_catalog slug and, for a slug with several tabs, the recorded
layout length (session_index layout_len_m) that identifies it. No length = no
match: fill a length here when a recording of that layout appears.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHEET = ("https://docs.google.com/spreadsheets/d/e/2PACX-1vTN03UvJDm99byA6vQPZHKOCYVvfxLu1zkJAzdaKyROykzEKY2"
         "-Xl1rl1q5znZEf36m88dxMKsY2eaO")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# tab -> (slug, layout_len_m or None). A slug with one tab matches without a length.
TABS: dict = {
    "Bahrain (wec)": ("bahrain", None), "Bahrain (endurance)": ("bahrain", None),
    "Bahrain (outer)": ("bahrain", None), "Bahrain (paddock)": ("bahrain", None),
    "Barcelona": ("barcelona", None),
    "Circuit de la Sarthe": ("sarthe", None),                  # recordings read 13,618 m; which tab is
    "Circuit de la Sarthe (straight)": ("sarthe", None),       # that? not settled: no match until it is
    "COTA": ("cota", 5490.5),                                  # recordings 5,489-5,494 m (GP layout)
    "COTA (national)": ("cota", None),
    "Daytona": ("daytona", None),
    "Fuji (chicane)": ("fuji", 4527.0),                        # user, design §12 Q12: our Fuji sessions
    "Fuji (classic)": ("fuji", None),
    "Imola": ("imola", None), "Interlagos": ("interlagos", None), "Laguna Seca": ("laguna_seca", None),
    "Long Beach": ("long_beach", None), "Monza": ("monza", None), "Monza (curvagrande)": ("monza", None),
    "Paul Ricard": ("paul_ricard", None), "Paul Ricard (1A)": ("paul_ricard", None),
    "Paul Ricard (1A v2)": ("paul_ricard", None), "Paul Ricard (1A v2 short)": ("paul_ricard", None),
    "Paul Ricard (3A)": ("paul_ricard", None),
    "Portimao": ("algarve", None), "Qatar": ("lusail", None), "Qatar (short)": ("lusail", None),
    "Road Atlanta": ("road_atlanta", None),
    "Sebring": ("sebring", 5812.9),                            # recordings 5,813 m (the school layout is ~3 km)
    "Sebring (school)": ("sebring", None),
    "Silverstone (GP)": ("silverstone", None), "Silverstone (National)": ("silverstone", None),
    "Silverstone (International)": ("silverstone", None),
    "Spa": ("spa", None),
}
CLASSES = ("LMH", "LMGT3", "LMP2wec", "LMP2elms", "LMP3", "GTE")
_TIME = re.compile(r"^(\d+):(\d{2})\.(\d{1,3})$")


def parse_time(s: str) -> float | None:
    m = _TIME.match((s or "").strip())
    if not m:
        return None
    t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3).ljust(3, "0")) / 1000
    return t if 0 < t < 3600 else None


def parse_tabs(pubhtml: str) -> dict:
    """{tab name: gid} from the pubhtml page's sheet list."""
    out = {}
    for name, gid in re.findall(r'items\.push\(\{name: "(.*?)",.*?gid: "(\d+)"', pubhtml):
        out[name.encode().decode("unicode_escape").strip()] = gid
    return out


def parse_layout_csv(text: str) -> list:
    """Rows after the 'Patch,Class,,,Car,,,,Time' header: (patch, class, car,
    time) with column 9 the clean time; padding rows and junk are skipped."""
    rows = list(csv.reader(io.StringIO(text)))
    start = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Patch" and len(r) > 4
                  and r[1].strip() == "Class"), None)
    if start is None:
        return []
    out = []
    for r in rows[start + 1:]:
        if len(r) < 10:
            continue
        patch, cls, car = r[0].strip(), r[1].strip(), r[4].strip()
        t = parse_time(r[9])
        if not patch or cls not in CLASSES or not car or t is None:
            continue
        out.append({"class": cls, "car": car, "patch": patch, "time_s": round(t, 3)})
    return out


def parse_laptimes_csv(text: str) -> tuple[dict, str | None]:
    """({tab: {class: {time_s, patch}}}, 'Last updated' text). Class blocks
    follow each other; a row is <tab+class>, tab, patch, Class avgW, bands..."""
    rows = list(csv.reader(io.StringIO(text)))
    updated = None
    for r in rows[:5]:
        for c in r:
            if "Last updated" in c:
                updated = c.split(":", 1)[1].strip() if ":" in c else c.strip()
    out: dict = {}
    for r in rows:
        if len(r) < 4 or not r[1].strip() or r[1].strip() == "Track":
            continue
        tab = r[1].strip()
        cls = r[0].strip()[len(tab):] if r[0].strip().startswith(tab) else None
        t = parse_time(r[3])
        if cls in CLASSES and t is not None:
            out.setdefault(tab, {})[cls] = {"time_s": round(t, 3), "patch": r[2].strip()}
    return out, updated


def build(tab_csvs: dict, laptimes_csv: str, source_url: str = SHEET + "/pubhtml") -> dict:
    """The JSON of public_laptimes.py from the fetched texts (no network)."""
    refs, updated = parse_laptimes_csv(laptimes_csv)
    layouts = {}
    for tab, text in tab_csvs.items():
        if tab not in TABS:
            continue
        slug, length = TABS[tab]
        layouts[tab] = {"slug": slug, "layout_len_m": length, "class_ref": refs.get(tab, {}),
                        "cars": parse_layout_csv(text)}
    return {"source_url": source_url, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sheet_updated": updated, "layouts": layouts}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "track_reference" / "public_laptimes.json")
    ap.add_argument("--dry-run", action="store_true", help="fetch and print; write nothing")
    a = ap.parse_args()
    print(f"GET {SHEET}/pubhtml")
    gids = parse_tabs(_get(SHEET + "/pubhtml"))
    if "Laptimes" not in gids:
        print("no 'Laptimes' tab found: the sheet changed; nothing written")
        return 1
    lap_csv = _get(f"{SHEET}/pub?output=csv&gid={gids['Laptimes']}")
    tab_csvs = {}
    for tab in TABS:
        if tab in gids:
            tab_csvs[tab] = _get(f"{SHEET}/pub?output=csv&gid={gids[tab]}")
        else:
            print(f"  tab not in the sheet: {tab}")
    data = build(tab_csvs, lap_csv)
    print(f"sheet 'Last updated': {data['sheet_updated']}")
    for tab, l in data["layouts"].items():
        print(f"  {tab:34s} {l['slug']:12s} len {l['layout_len_m'] or '-':>7} | {len(l['cars']):3d} car rows | "
              f"class refs {sorted(l['class_ref'])}")
    unknown = sorted(set(gids) - set(TABS) - {"Laptimes"})
    print(f"  tabs not imported: {unknown}")
    if a.dry_run:
        print("DRY RUN - nothing written")
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
