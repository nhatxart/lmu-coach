# Track reference

One JSON file per circuit (`<slug>.json`): its corners, straights and names, as the coach uses them.
Built by `backend/tools/BUILD_TRACK_REFERENCE.ps1`. No LLM is involved at any step, and the app
never calls an outside service: it only reads these files (the Docker backend mounts this folder
read-only; see `TRACK_REFERENCE_PATH` in `docker-compose.yml`).

## What is in a file

- `layouts[]`: every closed loop found for the track that has a plausible length (a layout is a loop).
  - `status`: `verified` (aligned with 10+ clean recorded laps), `provisional` (3-9 laps) or
    `osm_only` (no usable laps: the corner order is known, the start/finish line is not, and the
    coach ignores the layout).
  - `corners[]`: `display` (`Coca-Cola Corner (3)`, `Esses (3, 4, 5)`, `Turn 7`, `Porsche Curves`),
    `name`, `numbers`, `complex`, `direction`, `heading_deg`, `radius_m`, `kind` (`corner` / `kink`),
    `apex_m` (metres along the OSM centerline), `lmu_apex_m` / `lmu_start_m` / `lmu_end_m` (lap distance
    as LMU records it; what the coach uses), `matched` (seen in the recordings or placed between
    neighbours), `name_source` (`lmu` track_db, `osm`), `number_source` (`lmu`, `inferred`).
  - `straights[]`, `complexes[]`, `alignment` (laps used, matched pairs, start/finish offset, lap
    scale, median position error, whether the yaw sign was mirrored).
- `sources`, `official` (Wikipedia turn count and length), `built`.

## How it is built

1. OpenStreetMap geometry (`sources/<slug>.json`, downloaded once by `--fetch`): race-track ways and
   circuit relations are stitched into closed loops; corners come from the curvature of the true
   centerline; names/numbers that OSM volunteers mapped ("Turn 7") come along.
2. Recorded laps (`data/sessions`, read-only): the same detector finds the corners in LMU lap distance.
3. The two lists are aligned by sequence (order, direction, size), not by metres, because a car's
   odometer drifts from the centerline over a lap. This fixes the start/finish line, the yaw sign and
   every corner's LMU lap distance.
4. LMU's own names (`track_db.py`) win over OSM names; OSM names are kept as aliases. Numbers that no
   source gives are inferred only when the count fits exactly (`number_source: inferred`).

Add a new track: one entry in `backend/app/track_catalog.py`, record 10+ clean laps, then
`.\backend\tools\BUILD_TRACK_REFERENCE.ps1 --track <slug> --fetch`.

## Attribution

Geometry and names in `sources/` are (c) OpenStreetMap contributors, available under the Open
Database License (https://www.openstreetmap.org/copyright). Official corner counts/lengths come from
English Wikipedia infoboxes. If you redistribute these files, keep this notice.
