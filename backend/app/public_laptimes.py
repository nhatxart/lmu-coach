"""Public lap times (design 2026-11-10-ideal-lap-design.md §14 R9): Ohne Speed's
LMU lap-time sheet, imported by hand with backend/tools/import_public_laptimes.py
into <track_reference>/public_laptimes.json. Never fetched at request time; no
file = no public line, nothing else changes. No network here.

JSON: {source_url, fetched_at, sheet_updated, layouts: {"<tab>": {slug,
layout_len_m | null, class_ref: {"<class>": {time_s, patch}}, cars: [{class,
car, patch, time_s}]}}}. Sheet classes: LMH, LMGT3, LMP2wec, LMP2elms, LMP3, GTE.

Matching (refuse rather than guess):
- track: the layouts with our slug; one -> it; several -> the one whose
  layout_len_m is within 0.5 % of the file's; no length -> no match (Fuji's
  two layouts differ by one chicane);
- class: Hypercar -> LMH, LMGT3 -> LMGT3, LMP2 -> LMP2wec (LMP2elms when the
  model says ELMS), LMP3 -> LMP3, GTE -> GTE;
- car: car_identity model -> sheet car by an alias table, else by normalised
  name; unmatched -> class reference only;
- patch: GameVersion 14200 -> "1.42" (v // 10000, (v // 100) % 100; checked
  only on 14200, the one version in the recordings so far); the car's row for
  that patch, else the newest patch <= ours ("older"), else the newest ("newer").
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LAYOUT_LEN_TOL = 0.005
CLASS_MAP = {"hypercar": "LMH", "lmh": "LMH", "lmgt3": "LMGT3", "gt3": "LMGT3", "lmp2": "LMP2wec",
             "lmp3": "LMP3", "gte": "GTE"}
# car_identity model (normalised) -> sheet car (normalised)
CAR_ALIASES = {
    "porsche 911": "porsche 992",
    "bmw m4": "bmw m4",
    "ford mustang": "ford mustang",
    "lexus rc f": "lexus rc f",
    "mercedes-amg": "mercedes-amg",
    "aston martin vantage": "aston martin v8 vantage evo",
    "aston martin valkyrie": "aston martin valkyrie",
    "aston martin valkyrie amr-lmh": "aston martin valkyrie",
    "ferrari 296": "ferrari 296",
    "mclaren 720s": "mclaren 720s evo",
    "corvette z06": "corvette z06",
    "lamborghini huracan": "lamborghini huracan evo ii",
}
# Upper limits of the sheet's bands, in % of the reference (the "~100%"
# column is 100.5 %, "101%" is 101.5 %, ...); above the last: Offline.
BANDS = ((100.5, "Alien"), (101.5, "Competitive"), (103.5, "Good"), (105.5, "Midpack"),
         (106.5, "Tail-ender"))

_CACHE: dict = {}


def default_path() -> Path:
    from .track_reference import _reference_dir
    return _reference_dir() / "public_laptimes.json"


def load(path: str | Path | None = None) -> dict | None:
    """The imported sheet, cached by mtime; None without a file."""
    p = Path(path) if path else default_path()
    try:
        mt = p.stat().st_mtime_ns
    except OSError:
        return None
    hit = _CACHE.get(str(p))
    if hit is None or hit[0] != mt:
        hit = (mt, json.loads(p.read_text(encoding="utf-8")))
        _CACHE[str(p)] = hit
    return hit[1]


def normalise_car(name: str) -> str:
    s = re.sub(r"\(v[\d.]*\)", "", (name or "").lower())
    s = re.sub(r"#\d+.*$", "", s)
    for drop in ("lmgt3", "gt3 r", "lmh", "lmdh"):
        s = s.replace(drop, " ")
    return re.sub(r"\s+", " ", s).strip()


def sheet_class(car_class: str, car_model: str = "") -> str | None:
    c = CLASS_MAP.get((car_class or "").strip().lower())
    if c == "LMP2wec" and "elms" in (car_model or "").lower():
        return "LMP2elms"
    return c


def patch_of(game_version) -> str | None:
    """GameVersion 14200 -> '1.42' (unverified beyond 14200)."""
    try:
        v = int(str(game_version).strip())
    except (TypeError, ValueError):
        return None
    return f"{v // 10000}.{(v // 100) % 100:02d}"


def patch_key(p: str) -> tuple:
    """'1.4' -> (1, 40), '1.42' -> (1, 42), '0.5' -> (0, 50), '1.0' -> (1, 0)."""
    p = str(p).strip().rstrip("+")
    major, _, minor = p.partition(".")
    minor = (minor + "0")[:2] if minor else "00"
    return int(major), int(minor)


def band(pct: float) -> str:
    for lim, name in BANDS:
        if pct <= lim:
            return name
    return "Offline"


def _layout_for(data: dict, slug: str | None, layout_len_m: float | None):
    cands = [(tab, l) for tab, l in data.get("layouts", {}).items() if l.get("slug") == slug]
    if not slug or not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    if not layout_len_m:
        return None
    near = [(abs(l["layout_len_m"] - layout_len_m), tab, l) for tab, l in cands
            if l.get("layout_len_m") and abs(l["layout_len_m"] - layout_len_m) <= LAYOUT_LEN_TOL * l["layout_len_m"]]
    if not near:
        return None
    _, tab, l = min(near, key=lambda x: x[0])
    return tab, l


def _car_rows(rows: list, car_model: str) -> list:
    want = normalise_car(car_model)
    for k, v in CAR_ALIASES.items():
        if want.startswith(k):
            want = v
            break
    exact = [r for r in rows if normalise_car(r["car"]) == want]
    return exact or [r for r in rows if normalise_car(r["car"]).startswith(want) or want.startswith(normalise_car(r["car"]))]


def match(slug: str | None, layout_len_m: float | None, car_model: str, car_class: str,
          game_version, data: dict | None = None) -> dict | None:
    """{tab, car, car_best_s, class_ref_s, class_ref_patch, patch, patch_match}
    or None (no file, no layout match, no class)."""
    data = data if data is not None else load()
    if not data:
        return None
    hit = _layout_for(data, slug, layout_len_m)
    cls = sheet_class(car_class, car_model)
    if not hit or not cls:
        return None
    tab, lay = hit
    ref = (lay.get("class_ref") or {}).get(cls) or {}
    rows = _car_rows([r for r in lay.get("cars", []) if r.get("class") == cls], car_model)
    ours = patch_of(game_version)
    car_best, patch, how, car = None, None, None, None
    if rows:
        by_patch: dict = {}
        for r in rows:
            k = patch_key(r["patch"])
            if k not in by_patch or r["time_s"] < by_patch[k]["time_s"]:
                by_patch[k] = r
        keys = sorted(by_patch)
        if ours and patch_key(ours) in by_patch:
            k, how = patch_key(ours), "exact"
        elif ours and any(x <= patch_key(ours) for x in keys):
            k, how = max(x for x in keys if x <= patch_key(ours)), "older"
        else:
            k, how = keys[-1], "newer"
        r = by_patch[k]
        car_best, patch, car = r["time_s"], r["patch"], r["car"]
    if car_best is None and not ref.get("time_s"):
        return None
    return {"tab": tab, "car": car, "car_best_s": car_best, "class_ref_s": ref.get("time_s"),
            "class_ref_patch": ref.get("patch"), "patch": patch, "patch_match": how,
            "sheet_updated": data.get("sheet_updated")}
