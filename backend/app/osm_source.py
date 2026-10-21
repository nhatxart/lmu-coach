"""
Fetch a circuit's geometry and names from OpenStreetMap (and its official length /
turn count from Wikipedia) ONCE, and cache the raw answers on disk.

Everything downstream (track_reference.py) reads the cache only, so the app never
depends on an outside service at run time. This module is only used by
backend/tools/build_track_reference.py, which you run by hand when a new track comes
out (or to rebuild one).

Data (c) OpenStreetMap contributors, ODbL: https://www.openstreetmap.org/copyright
Be polite: one request per track, identify ourselves, cache everything, back off.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

USER_AGENT = "lmu-coach-track-reference/1.0 (local tool; github.com/lmu-coach)"
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
NOMINATIM = "https://nominatim.openstreetmap.org/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARCH_RADIUS_M = 3500
REQUEST_PAUSE_S = 2.0


class SourceError(Exception):
    """A source could not be reached or returned nothing usable; str(e) is the advice."""


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True)


def geocode(query: str) -> tuple[float, float]:
    """(lat, lon) of a place by name (Nominatim)."""
    with _client() as c:
        r = c.get(NOMINATIM, params={"q": query, "format": "json", "limit": 1})
    if r.status_code != 200 or not r.json():
        raise SourceError(f"Could not find '{query}' on OpenStreetMap (HTTP {r.status_code}).")
    j = r.json()[0]
    return float(j["lat"]), float(j["lon"])


def overpass(query: str, tries_per_mirror: int = 2) -> dict:
    """Run an Overpass query, trying every mirror with exponential backoff."""
    last = "no mirror answered"
    wait = 4.0
    with _client() as c:
        for _ in range(tries_per_mirror):
            for url in OVERPASS_MIRRORS:
                try:
                    r = c.post(url, data={"data": query})
                    if r.status_code == 200 and r.text.lstrip().startswith("{"):
                        return r.json()
                    last = f"{url}: HTTP {r.status_code}"
                except httpx.HTTPError as e:
                    last = f"{url}: {type(e).__name__}"
                time.sleep(REQUEST_PAUSE_S)
            time.sleep(wait)
            wait *= 2
    raise SourceError(f"OpenStreetMap (Overpass) is not answering ({last}). Try again later.")


def fetch_osm(lat: float, lon: float, radius_m: int = SEARCH_RADIUS_M) -> dict:
    """Race-track ways (with geometry) and every named point/way/relation tagged as part
    of a circuit near (lat, lon). Corner names live in several tag styles, so the second
    query is deliberately broad; track_reference.py filters it."""
    ways = overpass(
        f'[out:json][timeout:90];(way["highway"="raceway"](around:{radius_m},{lat},{lon});'
        f'way["sport"~"motor|motor_racing|karting",i]["highway"](around:{radius_m},{lat},{lon}););out geom tags;'
    )
    time.sleep(REQUEST_PAUSE_S)
    names = overpass(
        f'[out:json][timeout:90];('
        f'node["name"]["motorsport"](around:{radius_m},{lat},{lon});'
        f'node["name"]["sport"~"motor",i](around:{radius_m},{lat},{lon});'
        f'node["name"]["highway"~"raceway|turning_circle|milestone"](around:{radius_m},{lat},{lon});'
        f'node["name"]["ref"](around:{radius_m},{lat},{lon});'
        f'way["name"]["motorsport"](around:{radius_m},{lat},{lon});'
        f');out geom tags;'
    )
    out = {"center": [lat, lon], "radius_m": radius_m, "ways": ways.get("elements", []),
           "named": [n for n in names.get("elements", []) if n.get("type") != "relation"],
           "attribution": "OpenStreetMap contributors, ODbL"}
    time.sleep(REQUEST_PAUSE_S)
    out["circuits"] = fetch_circuits(lat, lon, radius_m)
    return out


def fetch_circuits(lat: float, lon: float, radius_m: int = SEARCH_RADIUS_M) -> dict:
    """The circuit relations (type=circuit: one per layout) near a point, and the ways that
    belong to them. Public-road sections of a circuit (Le Mans) are not tagged raceway, so
    this is the only way to get their loop; the relation also says which ways form a layout."""
    rel = overpass(
        f'[out:json][timeout:90];relation["type"="circuit"](around:{radius_m},{lat},{lon});out;'
    )
    ids = [r["id"] for r in rel.get("elements", [])]
    ways: list = []
    if ids:
        time.sleep(REQUEST_PAUSE_S)
        sel = "".join(f"relation({i});" for i in ids)
        got = overpass(f"[out:json][timeout:120];({sel})->.c;way(r.c);out geom tags;")
        ways = got.get("elements", [])
    return {"relations": rel.get("elements", []), "ways": ways}


def wikipedia_facts(title_candidates: list[str]) -> dict:
    """Official lap length (m) and number of turns from the Wikipedia infobox; {} if none."""
    with _client() as c:
        for title in title_candidates:
            r = c.get(WIKI_API, params={
                "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                "format": "json", "formatversion": 2, "redirects": 1, "titles": title})
            time.sleep(REQUEST_PAUSE_S)
            try:
                page = r.json()["query"]["pages"][0]
                text = page["revisions"][0]["slots"]["main"]["content"]
            except (KeyError, IndexError, ValueError):
                continue
            facts: dict = {"title": page.get("title", title)}
            m = re.search(r"\|\s*turns\s*=\s*(\d+)", text)
            if m:
                facts["turns"] = int(m.group(1))
            m = re.search(r"\|\s*length_km\s*=\s*([\d.]+)", text)
            if m:
                facts["length_m"] = round(float(m.group(1)) * 1000)
            if "turns" in facts or "length_m" in facts:
                return facts
    return {}


def cache_path(cache_dir: Path, slug: str) -> Path:
    return cache_dir / f"{slug}.json"


def load_cached(cache_dir: Path, slug: str) -> dict | None:
    p = cache_path(cache_dir, slug)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_cached(cache_dir: Path, slug: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path(cache_dir, slug).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
