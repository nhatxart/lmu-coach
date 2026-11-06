"""Car identity for the tyre model: model, class and game version of a file.

The grip profile used to be keyed by the livery name (`CarName`, e.g. "Akkodis
ASP Team 2026 #78:WEC"), which changes with the season/team and lumps nothing
together. HF recordings carry `VehicleModel`/`VehicleClass`/`GameVersion` in
metadata; older files only have the livery, so a small table maps those.
"""

import re
from dataclasses import dataclass

# (team substring, car number) -> (model, class). Number match is exact on the
# digits after '#', so "#7" (Toyota) never matches "#007" (Aston Martin).
LIVERY_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    ("Team WRT", "69"): ("BMW M4 LMGT3", "LMGT3"),
    ("Manthey", "91"): ("Porsche 911 GT3 R LMGT3", "LMGT3"),
    ("Manthey", "92"): ("Porsche 911 GT3 R LMGT3", "LMGT3"),
    ("Proton", "88"): ("Ford Mustang LMGT3", "LMGT3"),
    ("Akkodis", "78"): ("Lexus RC F LMGT3", "LMGT3"),
    ("Iron Lynx", "61"): ("Mercedes-AMG LMGT3", "LMGT3"),
    ("Heart of Racing", "23"): ("Aston Martin Vantage LMGT3", "LMGT3"),
    ("Aston Martin THOR", "007"): ("Aston Martin Valkyrie", "Hypercar"),
    ("Cadillac", "12"): ("Cadillac V-Series.R", "Hypercar"),
    ("Ferrari AF Corse", "50"): ("Ferrari 499P", "Hypercar"),
    ("Genesis", "17"): ("Genesis GMR-001", "Hypercar"),
    ("Toyota", "7"): ("Toyota TR010", "Hypercar"),
}

_CLASS_ALIASES = {
    "hypercar": "Hypercar", "lmh": "Hypercar", "lmdh": "Hypercar",
    # BUG FIX / PREVIOUSLY "Hyper" stayed "Hyper": the Ferrari (COTA) and BMW
    # (Daytona) recordings write VehicleClass "Hyper", so they were a class of
    # their own (lap comparison refused them against other Hypercars).
    "hyper": "Hypercar",
    "gt3": "LMGT3", "lmgt3": "LMGT3",
}


@dataclass(frozen=True)
class CarIdentity:
    model: str
    car_class: str
    game_version: str


def _normalise_class(raw: str) -> str:
    return _CLASS_ALIASES.get(raw.strip().lower(), raw.strip())


def _livery_lookup(car_name: str) -> tuple[str, str] | None:
    m = re.search(r"#(\d+)", car_name)
    if not m:
        return None
    number = m.group(1)
    for (team, num), value in LIVERY_TABLE.items():
        if num == number and team in car_name:
            return value
    return None


def identify_car(meta: dict[str, str], car_name: str) -> CarIdentity:
    livery = _livery_lookup(car_name)

    model = meta.get("VehicleModel") or (livery[0] if livery else car_name)

    if meta.get("VehicleClass"):
        car_class = _normalise_class(meta["VehicleClass"])
    elif model.endswith("LMGT3"):
        car_class = "LMGT3"
    elif livery:
        car_class = livery[1]
    else:
        car_class = "unknown"

    return CarIdentity(model, car_class, meta.get("GameVersion") or "unknown")
