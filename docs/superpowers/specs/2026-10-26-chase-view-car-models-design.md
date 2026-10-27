# Chase view, round 2: velocity-locked camera, car models, track limits, tyre slip - design

Date: 2026-10-26. Builds on `2026-10-25-chase-view-design.md` (merged to master at f25bc3e). Approved by the user in chat ("Let's do it").

## Intent

Make the chase view a better rotation and track-limits instrument:
1. A camera that can lock to the velocity vector, so the velocity arrow always points straight up the screen and the car's angle off vertical *is* the body slip angle.
2. A representative car (GT3 / Hypercar) instead of the chevron, sized for its class, so you can see where the four tyres are relative to the track edge: track limits.
3. Per-wheel tyre slip, animated like the slip-angle gauges.

All of it toggleable; the chevron stays as an option.

## Decisions made with the user

- **Velocity lock uses the data we already have.** No new recorded channels or payload points for the camera: the heading is interpolated more smoothly in the UI (it is a display concern).
- **Car models with wings** (rear wing with endplates and mounts, front splitter; GT3 and Hypercar shapes).
- **Preview before implementation:** the models are exported as STL files the user opens in a 3D-printer slicer and approves before they go into the UI.
- **Chevron remains selectable.** Every feature is a toggle.

## Builder's calls (mine, reviewable)

- **Velocity lock:** a Camera setting **Smooth** (today: course smoothed with τ = 0.35 s) / **Locked to velocity**. Locked = the course is interpolated with a cubic (Catmull-Rom on the unwrapped heading) at the playhead's fractional index, and the *same* course drives the camera yaw and the velocity arrow, with no time lag, so the arrow is exactly vertical. The backend heading is already smoothed over ±0.2 s; the cubic removes the kinks between samples. If it still twitches on real laps, a short τ (≤ 0.08 s) can be added, but camera and arrow then share it.
- **One source of truth for car dimensions:** `backend/app/static/car_models.json` (per class: length, width, height, wheelbase, front/rear track, front/rear overhang, tyre diameter and widths, wing span/chord/height, splitter). Read by the UI, by the STL exporter, and by the backend for track limits. Values are representative public figures, not a specific car. LMP2/LMP3 map to the Hypercar shape with their own dimensions when we record them.
- **Models are procedural** (three.js primitives, extrusions), built by `backend/app/static/car_models.js`, an ES module shared by the UI and the exporter. No downloaded or branded meshes, so it works offline and has no licensing questions. This is a second exception to "UI code stays in index.html", because the exporter has to reuse the exact same geometry.
- **Model frame:** nose +X, up +Y, right +Z (the chase-view car frame), origin on the ground at mid-wheelbase, the four wheels as separate child objects (FL, FR, RL, RR) so they can steer and be tinted.
- **STL previews:** binary STL in millimetres, one file per class at 1:1 and one at 1:43 (about 107 mm long, printable), written by `backend/tools/EXPORT_CAR_MODELS.ps1` into `docs/car_models/` (committed so they can be opened directly).
- **Track limits:** the backend computes, per driven-line sample, each wheel's lateral position (car lateral offset ± half track, rotated by β relative to the path) against the track half-width, and returns which wheels are fully beyond the edge (`driven_line.wheels_out`, 4-bit mask). A wheel turns red when out; the whole car outline turns red when all four are out (LMU's rule: the white line counts as track). The track edge comes from LMU's Track Edge (car's side only, assumed symmetric), so it is approximate and labelled as such.
- **Tyre slip arrows:** at each contact patch, an arrow showing where the tyre is travelling against where it points; angle = the recorded per-wheel slip angle × the existing slip-angle exaggeration. Front wheels are steered by a kinematic estimate (bicycle model from β, yaw rate, speed and the front slip angle). The estimate is checked against `Steering Pos` on real laps (correlation), which also fixes the yaw-rate sign. Unavailable (greyed, with a note) on files without per-wheel slip angles (native LMU exports).
- **Controls:** a small **View options** menu in the map toolbar, shown in chase view: Camera (Smooth / Locked to velocity), Car (Chevron / Car), Track limits (on/off), Tyre slip (on/off). Persisted in localStorage.

## Non-goals

Scrubber marks / per-lap counts for track-limit excursions (later), brand-specific liveries or meshes, suspension animation.
