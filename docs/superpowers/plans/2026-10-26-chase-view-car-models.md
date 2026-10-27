# Chase View Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Velocity-locked chase camera, procedural GT3/Hypercar car models (previewed as STL first), track-limits highlight and per-wheel tyre slip arrows, all toggleable.

**Architecture:** Car dimensions live in one JSON (`static/car_models.json`) read by the UI, the STL exporter and the backend. `static/car_models.js` builds the models from three.js primitives; a Node exporter writes STL previews from the same code. The backend adds the car class/dimensions and a per-sample wheels-out mask to `/trackmap`. The UI adds a View options menu and the new render paths inside the existing chase module.

**Tech Stack:** three.js 0.186.1 (vendored), Node 24 (exporter only), Python 3.12 backend (pytest), Playwright UI tests.

**Spec:** `docs/superpowers/specs/2026-10-26-chase-view-car-models-design.md` (and the round-1 spec `2026-10-25-chase-view-design.md`).

## Global Constraints

- PowerShell: chain with `;`, never `&&`. Never modify `.env`, `lmu-coach.config.json`, `data/sessions/` (read-only queries fine; real recordings at `C:\Users\hoang\Projects\lmu-coach\data\sessions`).
- No new backend runtime deps; no new vendored libraries (three 0.186.1 and GSAP 3.15.0 are already in `backend/app/static/vendor/`, as `three/three.module.js` + `three/three.core.js`).
- UI code stays in `backend/app/static/index.html` except `static/car_models.js` and `static/car_models.json` (shared with the exporter, per spec).
- Chase-view frame: Three `(X, Y, Z) = (map x, height, map z)`; car model nose +X, up +Y, right +Z; `rotation.set(roll, -(course + bodyOff), pitch, 'YZX')`; wheel order FL, FR, RL, RR. Slip angle sign: positive = tyre travelling right of where it points (`/trackmap` `slip_angle`). Lateral offsets right-positive.
- Canvas/WebGL colours as JS literals synced with `:root` (DESIGN.md); WCAG AA; keyboard operable; sentence case; `prefers-reduced-motion` respected.
- Comment the why. Commit with `git -c user.name=hoang -c user.email=hoangnhatvn2002@gmail.com commit -F <msgfile>`, message ending in ONE paragraph with both trailers:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_019PwkBbwnBPirt9qzDWznVm`.
- Tests: backend `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic python -m pytest backend/tests -v`; UI same plus `--with "uvicorn[standard]" --with playwright` on `backend/tests/ui`; recorder untouched.

## Review Focus

1. Files without per-wheel slip angles (native exports) or without Track Edge: car still renders; track-limit and tyre-slip toggles disabled with a note; no errors. (Tasks 3, 4)
2. Unknown car class: falls back to the GT3 dimensions and says so (`car.class_source`). (Task 3)
3. Locked camera at low speed / pit lane / spins: no flipping or NaN (heading interpolation over the ±π wrap). (Task 2)
4. Toggling options mid-transition or mid-playback: no rebuild storms, no stacked tweens. (Tasks 2-4)
5. STL output is a valid binary STL (triangle count matches file size, normals finite) and opens in a slicer at the stated scale. (Task 1)

---

### Task 1: Car dimensions, procedural models, STL previews (USER CHECKPOINT after this task)

**Files:** Create `backend/app/static/car_models.json`, `backend/app/static/car_models.js`, `backend/tools/export_car_models.mjs`, `backend/tools/EXPORT_CAR_MODELS.ps1`, `docs/car_models/*.stl` (generated), `backend/tests/test_car_models.py`; modify `backend/tools/README.md`.

**Interfaces (produces):**
- `car_models.json`: `{"version": 1, "classes": {"LMGT3": {...}, "Hypercar": {...}}, "aliases": {"LMP2": "Hypercar", "LMP3": "Hypercar"}}`. Per class (metres): `length, width, height, wheelbase, track_front, track_rear, overhang_front, tyre_diameter, tyre_width_front, tyre_width_rear, ride_height, wing: {span, chord, height_top, setback, endplate_height, mount: "swan_neck"|"pylon"}, splitter: {depth, width}, fin: bool`. Representative values: LMGT3 ≈ 4.60 × 2.04 × 1.20, wheelbase 2.70, tracks 1.70/1.65, tyres 0.71 Ø, 0.31/0.33 wide, swan-neck wing span ≈ 1.80; Hypercar ≈ 5.00 × 2.00 × 1.06, wheelbase 3.15, tracks 1.65/1.62, tyres 0.71 Ø, 0.29/0.34 wide, full-width wing with large endplates, shark fin. Document sources/assumptions in a `"notes"` field.
- `car_models.js` (ES module, no DOM): `export function buildCar(THREE, cls, dims) -> { group, body, wheels: {fl, fr, rl, rr}, outline }` and `export function classDims(json, carClass) -> {cls, dims, source: "exact"|"alias"|"fallback"}` (unknown → LMGT3, `"fallback"`). Frame: nose +X, up +Y, right +Z; origin on the ground at mid-wheelbase; wheels are separate `Object3D`s positioned at their hubs (rotation about their own Y steers them). Body: side-profile extrusion (low nose, cabin/greenhouse, rear deck) with plan-view taper; rear wing with endplates and mounts; front splitter; GT3: swan-neck mounts; Hypercar: pylons + shark fin. Low poly (< 5k triangles per car). `outline`: an `EdgesGeometry`-based line object for the track-limit tint.
- `export_car_models.mjs`: `node backend/tools/export_car_models.mjs [outDir]` imports `../app/static/vendor/three/three.module.js` and `../app/static/car_models.js`, reads the JSON, builds each class, bakes world matrices, writes binary STL in millimetres: `<class>_1to1_mm.stl` and `<class>_1to43_mm.stl` (Y-up model converted to Z-up for slicers, wheels on the bed at z = 0). Prints triangles and bounding box per file.
- `EXPORT_CAR_MODELS.ps1`: runs the exporter into `docs/car_models/`.

- [ ] Step 1: failing test `backend/tests/test_car_models.py`: runs the exporter into `tmp_path` (subprocess `node`; skip if node missing) and asserts for each class/scale: file size == 84 + 50·n_triangles; all normals/vertices finite; bounding box (1:1) within 2 % of the JSON length/width/height (height includes the wing); wheels touch z = 0 (min z ≈ 0); 1:43 box = 1:1 / 43. Also asserts the JSON has both classes and every key listed above.
- [ ] Step 2: run → FAIL. Step 3: implement. Step 4: run → PASS; open each STL in a quick render (e.g. a Playwright page with three's STLLoader-free BufferGeometry from the same builder, screenshot front/side/top/3-quarter into `.superpowers/sdd/2026-10-26-chase-view-car-models/models/`) and look at them; fix proportions that look wrong.
- [ ] Step 5: run the exporter into `docs/car_models/`, commit everything.

**Checkpoint:** controller shows the user the STL paths + screenshots and waits for approval/changes before Task 3.

---

### Task 2: Velocity-locked camera + View options menu

**Files:** `backend/app/static/index.html`, `backend/tests/ui/test_chase_view.py`.

- View options menu: a button in `.map-toolbar` (visible in chase view) opening a small popover (proper ARIA: button `aria-expanded`, `role="menu"`/radiogroups or a `dialog` with labelled fieldsets; Esc closes; focus returns). Settings stored in `localStorage['lmu-coach.chase.options']` as `{camera: "smooth"|"locked", car: "chevron"|"car", limits: bool, slip: bool}`; this task wires `camera` only and renders the other controls disabled ("coming soon" is not shown; they are simply absent until their task).
- Locked mode: `courseAt(f)` = Catmull-Rom interpolation on the unwrapped `driven_line.heading` around index `f` (handles the ±π wrap), used for BOTH camera yaw and the velocity arrow/chevron course; no exponential smoothing on yaw (height smoothing unchanged). Smooth mode unchanged.
- Tests: in locked mode, at 20 random playhead positions the projected velocity arrow is vertical on screen (|angle| < 0.5°); switching Smooth↔Locked mid-playback produces no console errors; heading wrap across ±π (synthetic) gives no jump > 5° between consecutive frames; setting persists across reload.

---

### Task 3: Car model in the chase view + track limits

**Files:** `backend/app/trackmap_elevation.py` (or a new `backend/app/track_limits.py`), `backend/app/main.py`, `backend/app/static/index.html`, tests.

- Backend payload: `car: {class, source: "exact"|"alias"|"fallback", dims}` (from `car_identity` + `car_models.json`); `driven_line.lat` (right-positive, topography convention), `driven_line.edge_half` (track half-width), `driven_line.wheels_out` (int mask bit0 FL, bit1 FR, bit2 RL, bit3 RR: wheel's inner tyre edge beyond the edge line), `track_limits: {available: bool, note: str}` (false without Track Edge / Path Lateral). Wheel lateral = lat + R(β)·(wheel offset), β from `body_slip_deg` when present else 0.
- UI: Car option renders `buildCar` (lazy `import('/static/car_models.js')`), same placement/attitude as the chevron, body semi-transparent like the chevron; keep velocity arrow and β pill. Track-limits toggle: wheels tinted red when their bit is set; outline red when all four. Chevron mode keeps working.
- Tests: backend: synthetic lap with the kerb excursion → wheels_out bits set only for right wheels in that window; legacy/no-Track-Edge → `available: false`; unknown class → fallback. UI: Car mode builds without errors, track-limit tint appears in the excursion window (state counter), chevron↔car switch disposes geometry.

---

### Task 4: Tyre slip arrows

**Files:** `backend/app/static/index.html` (and backend only if a field is missing), tests.

- Per wheel: arrow at the contact patch, direction = wheel heading + α·gain (α from `slip_angle.<wheel>`, gain = existing slip-angle exaggeration), colour by |α| like the gauge. Front wheels steered by δ estimated kinematically (bicycle model from β, yaw rate, speed, front slip angle, wheelbase split); yaw-rate sign chosen by the best correlation of δ with `driven_line.steering` over the lap; if correlation < 0.6 the front wheels are not steered (note in `Chase.state()`).
- Unavailable without slip angles (toggle disabled + note).
- Tests: synthetic constant-radius corner with known α gives arrows at the right angles (state values); real-file check (read-only) reports the δ–steering correlation in the report; toggle on/off disposes arrows; no per-frame allocation.
