# Chase view (3D "swingman" replay) and track topography - design

Date: 2026-10-25. Status: approved to build by the user ("use your best judgement, start building, we'll review after").

## Intent

The top-down replay shows where the car went but not how it rotated or how it used the track. The user wants a second perspective of the same playback: a camera behind and above the car (like LMU's swingman cam) on a 3D track with real elevation, edges, kerbs and corner labels, with the car drawn as the existing chevron. The purpose is coaching, not spectacle: **seeing rotation through corners and where the wheels hit the lines**. Accuracy of rotation and position beats looks.

## Decisions made with the user

- **Perspective, not a mode.** Same scrubber, play/pause, lap chips, HUD. A smooth animated transition from the top-down view into the chase view and back.
- **Camera locked to the direction of travel (course), smoothed - not to the car's heading.** Rotation (body slip) then reads as the chevron turning away from the camera's line of sight.
- **Elevation matters** (it changes car behaviour). The game exposes `mPos.y` (height) and `mOri` (orientation matrix) at ~100 Hz; the recorder currently drops both.
- **Topography is built per track like the corner reference** (`BUILD_TRACK_REFERENCE.ps1`, `data/track_reference/`), from every recording of that track, so old recordings render in 3D too once any new recording of that track exists. Build it "as best we can" from every source the game gives.

## Sources of topography (all from the game; no external terrain data in v1)

1. Car pose: `Pos Y` + orientation (`Ori Row0..2`) -> height, grade, roll along the line.
2. Four wheel ground probes: chassis corner height (pose + assumed wheel positions) minus that wheel's `Ride Height` -> ground height under each tyre; left/right difference gives banking with suspension roll removed.
3. Accumulation over all laps/files of the track in a (lap distance x lateral offset) grid; each lap drives a different line, so coverage grows.
4. Other cars (scoring `mPos`, `mLapDist`, `mPathLateral` per vehicle, 10 Hz gate): more lines per session. Unverified in game; lower weight.
5. `Surface Type` per wheel (already recorded since ~10/19): kerb / grass / gravel paint where wheels touched them. Works for old files.

Rejected: extracting the game's track meshes (packed/encrypted, licensing); SRTM-class terrain (~30 m, misleading). National LiDAR remains a possible later source for tracks with no new recordings.

## Builder's calls (mine, reviewable)

- Libraries: **Three.js 0.186.1** (ES module) and **GSAP 3.15.0**, vendored under `backend/app/static/vendor/` with licences so the UI stays offline and build-free. Loaded lazily on first switch to the chase view. This is the one exception to "UI stays in index.html"; the chase code itself is inline in `index.html`.
- Topography lives in a separate file per track, `data/track_reference/<slug>.topography.json`, built by the same tool (flag `--topography-only` for a quick rebuild). Rows every 2 m of lap distance, a plane per row (height at the centre path + bank slope), sparse surface-paint cells 2 m x 0.5 m.
- Orientation convention is **detected from data** (forward axis vs. velocity on straights), not assumed, because the `mOri` layout is not documented in the repo. Refuse rather than guess (same philosophy as body slip).
- Wheel geometry assumption: half-track 0.80 m, wheelbase 2.9 m (front +1.45 m, rear -1.45 m from the reference point), CG-to-ground allowance 0.25 m. Constant per car, cancels in banking.
- Lap with its own `Pos Y` but no built topography: a topography is built on the fly from that file alone. No elevation anywhere: flat track plus a visible "Flat - no elevation recorded for this track yet" note.
- Chase camera: 7.5 m behind, 2.6 m above the car, looking 12 m ahead; course direction smoothed with a 0.35 s time constant; snaps on scrub jumps; never below ground + 1 m. Zoom controls chase distance (4-30 m) in this view; Fit resets it.
- Transition: 1.1 s `power2.inOut` (GSAP), starting from a top-down camera that matches the current 2D framing exactly, quaternion-slerped into the chase pose; instant under `prefers-reduced-motion`.
- Visual language follows `DESIGN.md` (pit-wall console, `--map-bg` fog, Fira fonts, canvas colours as JS literals in sync with tokens). Driven line drawn on the surface in the active Overlay colouring. Chevron is the 2D chevron extruded (car-sized: 4.7 m x 2.0 m), with the same US/OS colours, β exaggeration (`slip-gain`) and β label.

## Goals / non-goals / anti-goals

- Goals: correct rotation and position relative to edges/kerbs; real elevation where recorded; works for old recordings via topography; no regressions in the 2D view.
- Non-goals: car 3D model, scenery, ghost/reference-lap comparison, other cars rendered in the replay, VR.
- Anti-goals: invented terrain presented as measured; a camera that hides rotation (heading-locked); motion sickness (no shake, smoothing, reduced-motion respected); slowing the 2D view (three.js loads only when used).

## Open, needs the user in game

- Verify `Pos Y` / `Ori` values and that other-car positions are populated (DevNotes TODO).
- `Path Lateral` sign (positive = right) is unverified; the builder checks it against world positions and logs the result.
