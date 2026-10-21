"""
The tracks LMU ships, and how to find each one on OpenStreetMap / Wikipedia.

Used only by backend/tools/build_track_reference.py. To add a new track, append one entry
(slug, the name as LMU writes it, an OSM search query, Wikipedia page titles) and run:

    .\\backend\\tools\\BUILD_TRACK_REFERENCE.ps1 --track <slug> --fetch

Optional per-track overrides: "latlon" (skip geocoding), "radius_m" (search radius, default 3500 m),
"turns" / "length_m" (official values to use when Wikipedia has no infobox for the track).
`registry` is the key in track_db.TRACK_REGISTRY (None when track_db does not know the
track yet). `match` lists lower-case fragments of the track name as it appears in
recording file names / metadata, so recordings find their reference.
"""

from __future__ import annotations

import re
import unicodedata

CATALOG: list[dict] = [
    {"slug": "bahrain", "registry": "Bahrain International Circuit", "query": "Bahrain International Circuit",
     "wiki": ["Bahrain International Circuit"], "match": ["bahrain"]},
    {"slug": "sarthe", "registry": "Circuit de La Sarthe", "query": "Circuit des 24 Heures du Mans",
     "wiki": ["Circuit de la Sarthe"], "match": ["sarthe", "le mans"],
     "latlon": (47.9505, 0.2100), "radius_m": 9000},
    {"slug": "paul_ricard", "registry": "Paul Ricard", "query": "Circuit Paul Ricard",
     "wiki": ["Circuit Paul Ricard"], "match": ["ricard"]},
    {"slug": "cota", "registry": "Circuit of the Americas", "query": "Circuit of the Americas",
     "wiki": ["Circuit of the Americas"], "match": ["americas"]},
    {"slug": "fuji", "registry": "Fuji Speedway", "query": "Fuji Speedway",
     "wiki": ["Fuji Speedway"], "match": ["fuji"]},
    {"slug": "imola", "registry": "Autodromo Internazionale Enzo e Dino Ferrari",
     "query": "Autodromo Enzo e Dino Ferrari", "wiki": ["Imola Circuit"], "match": ["imola", "enzo e dino"]},
    {"slug": "interlagos", "registry": "Autodromo Jose Carlos Pace", "query": "Autódromo José Carlos Pace",
     "wiki": ["Autódromo José Carlos Pace"], "match": ["interlagos", "pace"]},
    {"slug": "lusail", "registry": "Lusail International Circuit", "query": "Lusail International Circuit",
     "wiki": ["Lusail International Circuit"], "match": ["lusail"]},
    {"slug": "monza", "registry": "Autodromo Nazionale Monza", "query": "Autodromo Nazionale Monza",
     "wiki": ["Monza Circuit"], "match": ["monza"]},
    {"slug": "algarve", "registry": "Algarve International Circuit", "query": "Autódromo Internacional do Algarve",
     "wiki": ["Algarve International Circuit"], "match": ["algarve"]},
    {"slug": "sebring", "registry": "Sebring International Raceway", "query": "Sebring International Raceway",
     "wiki": ["Sebring International Raceway"], "match": ["sebring"]},
    {"slug": "silverstone", "registry": "Silverstone", "query": "Silverstone Circuit",
     "wiki": ["Silverstone Circuit"], "match": ["silverstone"]},
    {"slug": "spa", "registry": "Circuit de Spa-Francorchamps", "query": "Circuit de Spa-Francorchamps",
     "wiki": ["Circuit de Spa-Francorchamps"], "match": ["spa-francorchamps", "spa francorchamps"],
     "radius_m": 5000},
    {"slug": "barcelona", "registry": "Circuit de Barcelona", "query": "Circuit de Barcelona-Catalunya",
     "wiki": ["Circuit de Barcelona-Catalunya"], "match": ["barcelona", "catalunya"]},
    {"slug": "laguna_seca", "registry": "WeatherTech Raceway Laguna Seca", "query": "WeatherTech Raceway Laguna Seca",
     "wiki": ["WeatherTech Raceway Laguna Seca"], "match": ["laguna"]},
    {"slug": "daytona", "registry": "Daytona International Speedway", "query": "Daytona International Speedway",
     "wiki": ["Daytona International Speedway"], "match": ["daytona"], "latlon": (29.1852, -81.0705), "radius_m": 3000},
    {"slug": "road_atlanta", "registry": None, "query": "Michelin Raceway Road Atlanta",
     "wiki": ["Road Atlanta"], "match": ["road atlanta"], "latlon": (34.1494, -83.8139)},
    {"slug": "long_beach", "registry": None, "query": "Long Beach Convention Center",
     "wiki": ["Grand Prix of Long Beach"], "match": ["long beach"], "turns": 11, "length_m": 3167},
]


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def find_entry(track_name: str) -> dict | None:
    """Catalog entry for a track name as written in recordings (accents, underscores and
    mojibake tolerated: the file for Interlagos has a damaged 'Aut?dromo Jos? Carlos Pace')."""
    t = _plain(track_name)
    for e in CATALOG:
        if any(m in t for m in map(_plain, e["match"])):
            return e
    return None


def get(slug: str) -> dict | None:
    return next((e for e in CATALOG if e["slug"] == slug), None)
