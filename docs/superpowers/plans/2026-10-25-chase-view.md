# Chase View + Track Topography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 3D chase-camera ("swingman") perspective of the existing lap replay, on a track surface with real elevation, banking, edges, kerbs and corner labels, fed by new recorder pose channels and a per-track topography built offline.

**Architecture:** The recorder adds `Pos Y`, the raw orientation matrix and other cars' positions. A pure-numpy module (`track_topography.py`) turns laps into ground probes and fits a per-track grid (2 m rows along lap distance, a plane across each row, sparse kerb/grass/gravel paint), built by `BUILD_TRACK_REFERENCE.ps1` into `data/track_reference/<slug>.topography.json`. `/trackmap` attaches heights, pitch/roll and a 3D surface ribbon to its payload. `index.html` lazily loads vendored Three.js + GSAP and renders the chase view in a second canvas that shares the playback state, with an animated camera transition from the top-down view.

**Tech Stack:** Python 3.13 recorder (unittest, uv), Python 3.12 FastAPI backend (pytest, numpy), vanilla JS single-file UI, Three.js 0.186.1 (ES module), GSAP 3.15.0 (UMD), Playwright (Python) for the UI smoke test.

**Spec:** `docs/superpowers/specs/2026-10-25-chase-view-design.md`

## Global Constraints

- PowerShell on Windows: chain with `;`, never `&&`.
- Never modify `.env`, `lmu-coach.config.json`, or anything in `data/sessions/` (read-only queries on real recordings in `C:\Users\hoang\Projects\lmu-coach\data\sessions` are fine).
- Backend runs on Python 3.12 with `backend/requirements.txt` (numpy 2.1.3, scipy 1.14.1, duckdb 1.1.3); no new backend runtime dependencies.
- Recorder: keep `RECORDER_ID = "lmu-coach-recorder hf1"` unchanged; keep CSV + `COPY` bulk writes; keep NORMAL priority behaviour untouched.
- UI code stays inside `backend/app/static/index.html`. The only new static files are vendored libraries under `backend/app/static/vendor/` (three: `three.module.min.js` + `three.core.min.js` + `LICENSE`; gsap: `gsap.min.js` + `LICENSE`), pinned to three **0.186.1** and gsap **3.15.0**.
- Three.js and GSAP load lazily (dynamic `import()` / injected `<script>`) only when the chase view is first opened. The 2D view must not load them.
- Canvas/WebGL colours are JS literals kept in sync with `:root` tokens (`DESIGN.md`). Fonts: Fira Sans UI, Fira Code numbers. WCAG AA, visible focus, keyboard-operable controls, `prefers-reduced-motion` respected.
- Code style: comment the *why* of non-obvious choices ("BUG FIX / PREVIOUSLY ... Fix:" habit for fixes). Match surrounding idiom.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_019PwkBbwnBPirt9qzDWznVm`
- Test commands:
  - Recorder: `cd recorder; uv run python -m unittest test_recorder -v`
  - Backend (repo root): `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic python -m pytest backend/tests -v`
  - UI (repo root): `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic --with "uvicorn[standard]" --with playwright python -m pytest backend/tests/ui -v` (first run: `uv run --no-project --with playwright python -m playwright install chromium`)

### Coordinate conventions (binding for every task)

- **LMU world:** `mPos` metres, y up. The recorder's pseudo-GPS is `lat = 35.37 + z/111320`, `lon = 138.92 + x/(111320*cos(35.37°))`.
- **Map frame (existing `/trackmap`):** `x = world x - c`, `z = -(world z - c)` (latitude negated). Canvas draws x right, z down. This frame is verified non-mirrored (Road Atlanta T1 is a right-hander).
- **Three.js frame:** `(X, Y, Z) = (map x, height, map z)`. Seen from +Y with `camera.up = (0,0,-1)`, this is exactly the canvas picture (right-handed; x right, z down-screen). Do not negate anything else.
- **Course/heading angle `a`** (`driven_line.heading`): canvas radians, 0 = +x, positive toward +z. Direction in Three = `(cos a, 0, sin a)`. A model whose nose points +X gets `rotation.y = -a`.
- **Car attitude:** Euler order `'YZX'`: `rotation.y = -(course + bodyOff)`, `rotation.z = pitch` (+ = nose up), `rotation.x = roll` (+ = right side down; the car's right side is +Z when facing +X).
- **Lateral offset `lat`:** metres from the game's centre path, **positive = right of travel** (the convention `/trackmap` already uses for `Path Lateral`; `detect_lateral_sign` checks it per build and records the evidence).
- **Bank `bank`:** dh/dlat in m per m, + = surface rises to the right. Surface-derived roll = `-atan(bank)`; surface-derived pitch = `atan(dh/ds)`.
- **Left of travel in world (x, z):** for a world travel direction `(dx, dz)`, left = `(-dz, dx)` (derived from the non-mirrored map frame).
- **Wheel order:** value1..4 = FL, FR, RL, RR. **Surface Type codes:** 0 dry, 1 wet, 2 grass, 3 dirt, 4 gravel, 5 kerb (rumble strip), 6 special.

## Review Focus

1. **Old recordings (no `Pos Y`, no `Ride Height`) and native LMU files** must still open in both views: `/trackmap` returns `elevation.source == "none"` or `"topography"`, all new arrays present with the right lengths, no exception. Pinned in Task 5 (`test_trackmap_legacy_file_flat`) and Task 6 (UI test on a legacy session).
2. **Orientation convention unknown/broken data** (identity matrix, zeros, a transposed matrix): `detect_pose_basis` must either find the right axes or return `None`, never silently produce garbage pitch/roll. Pinned in Task 2 (`test_detect_transposed`, `test_detect_refuses_garbage`).
3. **Lap-distance wrap and layout mismatch:** rows at s≈0/L must blend across the start/finish line; a lap whose length differs >3 % from every layout gets no topography lookup. Pinned in Task 3 (`test_lookup_wraps_start_finish`) and Task 4 (`test_load_topography_length_mismatch`).
4. **WebGL unavailable or library load failure:** switching to Chase must show a readable message and fall back to Top-down, not leave a blank stage. Pinned in Task 6 (`test_chase_falls_back_without_webgl`).
5. **Lap change / scrub jump while in chase view or mid-transition:** scene rebuilds for the new lap, camera snaps instead of swinging across the track, no stacked tweens. Pinned in Task 7 (`test_lap_change_in_chase_view`, `test_scrub_jump_snaps_camera`).

---

### Task 1: Recorder records pose: `Pos Y`, orientation matrix, other cars

**Files:**
- Modify: `recorder/recorder.py` (`SessionBuffer.add_sample` ~710-851, `write_session` ~856-1118, the main loop call site ~1469-1557, `selftest` channel list ~1683)
- Test: `recorder/test_recorder.py`

**Interfaces:**
- Produces (DuckDB tables in every new recording, read by Tasks 2-5):
  - `Pos Y` — scalar `(value DOUBLE)`, 100 Hz (same rate/rows as `Pos X`), unit `m`. Never flagged unpopulated (0 is a real height).
  - `Ori R0X, Ori R0Y, Ori R0Z, Ori R1X, Ori R1Y, Ori R1Z, Ori R2X, Ori R2Y, Ori R2Z` — nine scalars, 100 Hz, unit `''`, raw `telem.mOri[r].x/.y/.z`. Raw values, raw signs (same reasoning as the existing raw extras: a wrong axis guess must not be baked into files).
  - `Other Cars` — table `(ts DOUBLE, slot INTEGER, pos_x DOUBLE, pos_y DOUBLE, pos_z DOUBLE, lap_dist DOUBLE, path_lateral DOUBLE, in_pits INTEGER)`, one row per non-player vehicle per 10 Hz gate tick, from `data.scoring.vehScoringInfo[i]` (`mID`, `mPos`, `mLapDist`, `mPathLateral`, `mInPits`), `ts` = the player's `mElapsedTime` of that tick. `channelsList` row `('Other Cars', <shz>, 'pose_rows')`.
  - metadata key `PoseChannels` = `"1"`.
- `add_sample(...)` gains a keyword argument `others: list | None = None` (list of scoring structs for non-player vehicles, evaluated only on 10 Hz gate ticks). The main loop passes `[scor[i] for i in range(num_v) if i != si]`.

- [ ] **Step 1: Write failing tests** in `test_recorder.py`:
  - extend `fake_telem` with `mOri=[_vec(1,0,0), _vec(0,1,0), _vec(0,0,1)]` and `mPos=_vec(pos_x, 12.5, 5.0)`;
  - add `"Pos Y"` and the nine `Ori R..` names to `NEW_SCALAR` so `test_new_channels_recorded` covers them;
  - `test_write_session_roundtrip`: assert `rows("Pos Y") == 300`, all `Pos Y` values `== 12.5`, `freqs["Ori R2Z"] == freqs["Pos X"]`, `meta["PoseChannels"] == "1"`;
  - new `test_other_cars_written`: buffer fed with `others=[fake_scor(mID=7, mPos=_vec(1.0, 2.0, 3.0), mLapDist=55.0, mPathLateral=-1.2, mInPits=0)]` on gated samples → table `Other Cars` has rows with `slot == 7`, `pos_y == 2.0`, `lap_dist == 55.0`, `path_lateral == -1.2`, and row count equals the number of 10 Hz gate rows (`rows("Lap Dist")`);
  - new `test_other_cars_absent_is_empty_table`: no `others` → `Other Cars` exists with 0 rows (a spool/selftest buffer without the channel must still write).
- [ ] **Step 2:** Run `cd recorder; uv run python -m unittest test_recorder -v` → the new assertions FAIL.
- [ ] **Step 3: Implement.** Append `Pos Y` next to `Pos X/Z`; append the nine Ori values in the raw-extras block; write them with `_write_scalar(..., hz, unit)` (no `detect_empty`). Buffer `Other Cars` rows in the 10 Hz gate branch; write via a new `_write_rows(ch, columns_sql, rows, freq, unit)` helper using the same CSV + `COPY` path as `_bulk`. Add `("PoseChannels","1")` to `buf.meta` on the first sample. Update the main loop call and `selftest`'s channel list. Make `fake_scor` accept the new fields.
- [ ] **Step 4:** Run the recorder tests → all PASS (whole file, not just the new tests).
- [ ] **Step 5: Commit** `Recorder: record Pos Y, raw orientation matrix and other cars' poses (for track topography)`.

---

### Task 2: Backend pose channels and orientation-convention detection

**Files:**
- Create: `backend/app/vehicle_pose.py`
- Modify: `backend/app/telemetry.py` (`CONTINUOUS_CHANNELS`, `WHEEL_CHANNELS`)
- Test: `backend/tests/test_vehicle_pose.py`

**Interfaces:**
- Consumes: Task 1 table names.
- Produces:
  - `ORI_CHANNELS: tuple[str, ...]` = the nine `Ori R..` names in row-major order.
  - `ori_matrices(rows: dict[str, np.ndarray]) -> np.ndarray | None` — `(n,3,3)` from the nine arrays (`M[:, r, c]`), `None` if any is missing/empty/length-mismatched.
  - `@dataclass(frozen=True) PoseBasis: axis_kind: str ("row"|"col"); fwd: tuple[int,int] (index, sign); up: tuple[int,int]; left: tuple[int,int]; fwd_alignment: float; up_mean_y: float`
  - `detect_pose_basis(M: np.ndarray, vel_x: np.ndarray, vel_z: np.ndarray, speed_ms: np.ndarray) -> PoseBasis | None` — world-frame velocity (from `Pos X/Z` differences). Candidate axes: each of the 3 rows and 3 columns of M, each with sign ±1. **fwd** = candidate maximising the mean cosine between its horizontal (x,z) part and the velocity direction over samples with speed > 15 m/s; **up** = among the remaining axes of the same kind, the one with the largest mean y component, sign making it positive; **left** = the remaining axis, sign chosen so its horizontal part aligns with the world left `(-dz, dx)` of travel. Refuse (`None`) if fewer than 50 fast samples, `fwd_alignment < 0.95`, `up_mean_y < 0.9`, or left alignment mean < 0.9.
  - `basis_vectors(M: np.ndarray, basis: PoseBasis) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(fwd, left, up)`, each `(n,3)` unit vectors in world.
  - `pitch_roll_deg(fwd: np.ndarray, left: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — pitch = `degrees(arcsin(fwd_y))` (+ nose up); roll = `degrees(arcsin(left_y))` (+ = right side down, i.e. left side up).
- `telemetry.py`: add `"Pos X", "Pos Y", "Pos Z"` and the nine Ori names to `CONTINUOUS_CHANNELS`; add `"Ride Height"` to `WHEEL_CHANNELS`. (Missing tables are already skipped.)

- [ ] **Step 1: Write failing tests** (`test_vehicle_pose.py`), with a helper that builds a car driving a circle at 30 m/s for 20 s at 100 Hz with true pitch 2° and roll -3°, giving world fwd/left/up vectors and positions:
  - `test_detect_rows_convention`: rF2 style (local +x left, +y up, +z rear; `world = M @ local`, so M's **columns** are the world vectors left, up, rear = -fwd) → `detect_pose_basis` returns a basis; `pitch_roll_deg` recovers 2.0 and -3.0 within 0.05°.
  - `test_detect_transposed`: same data with M transposed → still recovers 2.0 / -3.0 within 0.05°.
  - `test_detect_forward_positive_z`: local forward +z variant → recovers.
  - `test_detect_refuses_garbage`: identity matrices for all samples → `None`; all-zero → `None`; fewer than 50 fast samples → `None`.
  - `test_ori_matrices_missing_channel`: one Ori array missing → `None`.
  - `test_telemetry_loads_pose_channels`: write a tiny DuckDB with `Pos Y`, the nine Ori tables and `Ride Height` (value1..4) using the pattern in `synthetic_session.py`, load with `telemetry.load_telemetry_file(path, grid_hz=50)`, assert `"Pos Y"`, `"Ori R1Y"`, `"Ride Height FL"` in `lap.channels`.
- [ ] **Step 2:** Run `... python -m pytest backend/tests/test_vehicle_pose.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement `vehicle_pose.py` and the telemetry list changes.
- [ ] **Step 4:** Run the whole backend suite → PASS (no regressions from the new channel names).
- [ ] **Step 5: Commit** `Backend: read pose channels; detect the orientation-matrix convention from data`.

---

### Task 3: Track topography core (probes, grid fit, paint, lookup) + synthetic hilly track

**Files:**
- Create: `backend/app/track_topography.py`
- Create: `backend/tests/synthetic_track.py`
- Test: `backend/tests/test_track_topography.py`

**Interfaces:**
- Consumes: `vehicle_pose` (Task 2), `telemetry.LapTelemetry` (`lap.channels[name]` numpy arrays on the lap grid; `Lap Dist` unwrapped and re-zeroed per lap).
- Produces (`track_topography.py`):
  - Constants: `ROW_STEP_M = 2.0`, `PAINT_CELL_M = 0.5`, `PAINT_KINDS = {2, 3, 4, 5}`, `MIN_SPEED_MS = 8.0`.
  - `@dataclass(frozen=True) WheelGeometry: half_track_m=0.80, front_m=1.45, rear_m=-1.45, cg_height_m=0.25`; `DEFAULT_GEOMETRY`.
  - `@dataclass ProbeSet: s: np.ndarray; lat: np.ndarray; y: np.ndarray; w: np.ndarray; paint_s: np.ndarray; paint_lat: np.ndarray; paint_kind: np.ndarray` (all 1-D; `s` = game lap distance in metres, `lat` right-positive, `y` absolute LMU ground height, `w` weight).
  - `detect_lateral_sign(path_lat: np.ndarray, track_edge: np.ndarray, surface: dict[str, np.ndarray]) -> tuple[int, float, str]` — `(sign, confidence, evidence)`; uses samples where `|path_lat| > 0.6*|track_edge|`; counts non-dry contacts on the FR/RR pair vs FL/RL pair when `path_lat > 0` (and the mirror for `< 0`). `sign = +1` if right-side wheels dominate on the positive side (positive = right), `-1` if left dominates; inconclusive (< 20 contacts or ratio < 2:1) → `(+1, 0.0, "inconclusive: default positive=right")`.
  - `lap_probes(lap: LapTelemetry, lap_dist_offset: float = 0.0, lateral_sign: int = 1, geom: WheelGeometry = DEFAULT_GEOMETRY, basis: PoseBasis | None = None) -> ProbeSet` — samples with speed > `MIN_SPEED_MS`. Heights only if `Pos Y` present: with a basis and `Ride Height` → four wheel probes: wheel world offset = `fwd*front_or_rear + left*(±half_track)`; ground y = `pos_y + offset_y - ride_height_w - cg_height`; probe `s = lap_dist + (front_m|rear_m)`, `lat = lateral_sign*path_lat + (−half_track for left wheels, +half_track for right)`; weight 1.0. Without basis or ride height → one centre probe `y = pos_y - mean(ride_height or 0) - cg_height`, weight 0.5. Paint from `Surface Type FL..RR` (kind in `PAINT_KINDS`) at the same wheel `(s, lat)` even when there is no height data.
  - `other_car_probes(rows: np.ndarray, length_m: float, lateral_sign: int = 1) -> ProbeSet` — rows `(ts, slot, pos_x, pos_y, pos_z, lap_dist, path_lateral, in_pits)`; drops in-pits rows; `y = pos_y - 0.30`, weight 0.3; no paint.
  - `build_layout_topography(probes: list[ProbeSet], length_m: float) -> dict | None` — algorithm below; `None` if there are no height probes **and** no paint.
  - `lookup_height(layout: dict, s: np.ndarray, lat: np.ndarray) -> np.ndarray` — `h(s) + bank(s)*lat`, with `h`/`bank` linearly interpolated between row centres, periodic in `s` (mod `length_m`).
  - `grade_bank(layout: dict, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(dh/ds, bank)` at `s`.
  - `paint_at(layout: dict, s: np.ndarray, lat: np.ndarray) -> np.ndarray[int]` — kind code of the cell, 0 if none.
- Layout dict (schema, also the JSON layout entry in Task 4): `{"length_m", "row_step_m": 2.0, "rows": N, "h": [N], "bank": [N], "n": [N], "lat_min": [N], "lat_max": [N], "paint_cell_m": 0.5, "paint": [[row, lat_cell, kind], ...], "has_heights": bool, "coverage": float}`. Floats rounded to 3 decimals for size.
- Fit algorithm for `build_layout_topography` (decided here; implement exactly):
  1. `N = ceil(length_m / ROW_STEP_M)`; row of a probe = `floor((s mod L)/ROW_STEP_M)`.
  2. For each row r, take probes in rows `r-2..r+2` (periodic). If total weight ≥ 3: weighted least squares `y = h + bank*lat` when the weighted std of `lat` ≥ 0.5 m, else `bank = NaN` and `h` = weighted mean at `lat` 0 (correct with the neighbouring bank later). Reject probes with |residual| > 0.15 m and refit once (kerb strikes/airborne).
  3. Rows without data: periodic linear interpolation of `h` and `bank` from rows that have them (`bank` NaN → interpolated, all-NaN → 0).
  4. Smooth along s, periodic: Gaussian σ = 1.5 rows for `h`, σ = 3 rows for `bank` (`scipy.ndimage.gaussian_filter1d(mode="wrap")`).
  5. `n[r]` = raw probe count in row r; `lat_min/lat_max` from those probes (0 when empty); `coverage` = fraction of rows with `n > 0`.
  6. Paint: per `(row, round(lat/PAINT_CELL_M))` count kinds; keep cells with ≥ 2 hits, majority kind.
- `synthetic_track.py` (test fixture, also used by Tasks 4-7):
  - `RING_LENGTH_M` ≈ 1200: stadium loop (two 400 m straights joined by semicircles of radius 63.66 m), counter-clockwise in world x/z.
  - `true_height(s) = 8.0*sin(2*pi*s/L)`; `true_bank(s)` = 0.08 inside the second semicircle (rising to the right, i.e. outside of a left turn), 0 elsewhere (with 20 m linear ramps).
  - `write_hilly_session(path: Path, lap_offsets=(0.0, 1.5, -1.5), with_pose=True, with_surface=True, other_cars=0, track_name="Synthetic Hill Ring") -> Path` — writes a recorder-HF1-shaped DuckDB: `metadata` (TrackName, CarName, RecordedBy `lmu-coach-recorder hf1`, PoseChannels when with_pose), `channelsList`, 100 Hz `GPS Time`, `Pos X/Y/Z`, the nine Ori (rF2 style: local +x left, +y up, +z rear; `world = M @ local`, i.e. M's columns are the world left/up/rear vectors; `Ori RkX/Y/Z` = row k), `Ride Height` (0.05 m each wheel), `Surface Type` (5 for wheels whose `lat` > 5.5 m, else 0), `Ground Speed` (km/h, 40 m/s), `Yaw Rate`, `Throttle/Brake/Steering Pos`, `G Force Lat/Long`; 10 Hz `GPS Latitude/Longitude` (recorder pseudo-GPS constants), `Lap Dist`, `Path Lateral` (lap offset, right-positive), `Track Edge` (+6.0 signed like Path Lateral); `Lap`/`Lap Time` events; `Other Cars` rows when `other_cars > 0` (each car at a different lateral offset). Each lap drives at its `lap_offsets[i]`. Ground under the car is `true_height(s) + true_bank(s)*lat`; `Pos Y` = ground + 0.25 + 0.05 (cg + ride height). `with_pose=False` omits `Pos Y`, Ori and `Ride Height` (a legacy recording).
  - `truth(s, lat) -> float` helper exported for assertions.

- [ ] **Step 1: Write failing tests** (`test_track_topography.py`):
  - `test_flat_ring_fits_zero`: probes from a ring with height 0 → all `h` within 0.01, `bank` within 0.005.
  - `test_hill_and_bank_recovered`: `write_hilly_session` (3 laps, offsets 0/1.5/-1.5) → load laps → `lap_probes` with the detected basis → `build_layout_topography` → for s on a 5 m grid and lat in {-3, 0, 3}: `|lookup_height - truth| < 0.08`; mean bank over the middle of the second semicircle within 0.01 of 0.08.
  - `test_paint_kerb_cells`: kerb paint exists only at `lat > 5` cells.
  - `test_lookup_wraps_start_finish`: `lookup_height` at s = L-0.5 and s = 0.5 differ by < 0.05 and both match truth within 0.08.
  - `test_detect_lateral_sign`: synthetic surface where the right wheels touch kerb at positive lat → `(+1, conf>0.5, ...)`; mirrored data → `-1`; no contacts → `(+1, 0.0, "inconclusive...")`.
  - `test_legacy_lap_gives_paint_only`: `with_pose=False` → `has_heights` False, paint present.
  - `test_other_car_probes_weighted`: rows in pits are dropped; weights 0.3.
- [ ] **Step 2:** Run the new test file → FAIL.
- [ ] **Step 3:** Implement `synthetic_track.py` and `track_topography.py`.
- [ ] **Step 4:** Run the whole backend suite → PASS.
- [ ] **Step 5: Commit** `Track topography: ground probes from pose + ride height, per-track grid fit, kerb paint`.

---

### Task 4: Build topography per track with BUILD_TRACK_REFERENCE; runtime loader

**Files:**
- Modify: `backend/app/track_topography.py` (file IO section)
- Modify: `backend/tools/build_track_reference.py`, `backend/tools/BUILD_TRACK_REFERENCE.ps1` (usage comment), `backend/tools/README.md`
- Test: `backend/tests/test_topography_build.py`

**Interfaces:**
- Consumes: Task 3 functions; `track_reference._reference_dir()`; `track_catalog.find_entry`; `telemetry.load_telemetry_file`.
- Produces:
  - `topography_path(slug: str) -> Path` = `_reference_dir() / f"{slug}.topography.json"`.
  - `slug_for_track(track_name: str) -> str | None` (via `track_catalog.find_entry`; tests monkeypatch this).
  - **Lap distance must be the game's raw `Lap Dist`** (what `/trackmap` looks up with). `load_telemetry_file` re-zeroes `Lap Dist` per lap (`telemetry.py` ~509); pass the subtracted value back as `lap_probes(..., lap_dist_offset=...)` (read how it is computed and expose it, e.g. store it on `LapTelemetry` as `lap_dist_zero`), otherwise topography and the replay disagree by up to one 10 Hz step (~6 m).
  - `build_topography(files: list[Path], slug: str, display_name: str) -> dict | None` — loads every file with `load_telemetry_file(path, grid_hz=50.0)`; keeps laps of 40-600 s; groups laps into layouts by lap length (laps within 3 % of each other share a layout; `length_m` = median lap length); detects the lateral sign over all laps of a layout (`detect_lateral_sign`), detects the pose basis per file (needs world velocity: from `Pos X/Z` channels' differences / dt), collects `lap_probes` + `other_car_probes` (from the `Other Cars` table read directly with duckdb, read-only), fits each layout. Returns `{"schema": 1, "slug", "display_name", "built": ISO-8601 UTC, "datum_m": min h over layouts with heights (0.0 if none), "layouts": [layout + {"lateral_sign": {"value","confidence","evidence"}, "pose": {"files_with_pose": k, "files_refused": j}, "sources": {"files", "laps", "other_car_samples"}}]}`. Prints one line per file (`pose: rows/col …, fwd alignment 0.99` or `pose refused: <reason>`).
  - `save_topography(topo: dict) -> Path`; `load_topography(track_name: str, lap_length_m: float | None) -> tuple[dict, dict] | None` — returns `(topo, layout)`; picks the layout whose `length_m` is within 3 % of `lap_length_m` (closest), `None` if none or no file; caches by `(path, mtime)`.
  - CLI: `build_track_reference.py` builds topography for every catalog track that has recordings after its reference (unless `--no-topography`); new `--topography-only` skips OSM/corner work. Output line: `topography: <slug> layouts=1 coverage=0.93 files=4 (2 with pose) laps=17`.

- [ ] **Step 1: Write failing tests** (`test_topography_build.py`), using `write_hilly_session` into `tmp_path` and `monkeypatch.setenv("TRACK_REFERENCE_DIR", tmp_path)` plus `monkeypatch.setattr(track_topography, "slug_for_track", lambda n: "synthetic-ring")`:
  - `test_build_and_load_roundtrip`: two files (one with pose, one legacy) → `build_topography` → `save_topography` → `load_topography("Synthetic Hill Ring", RING_LENGTH_M)` returns a layout with `has_heights`, coverage > 0.9, `sources.files == 2`; `lookup_height` matches truth within 0.08.
  - `test_load_topography_length_mismatch`: lap length 1.1 × ring → `None`.
  - `test_build_without_pose_files_is_paint_only`: only legacy files → layout `has_heights` False, `datum_m == 0.0`.
  - `test_other_cars_add_coverage`: a file with `other_cars=3` and a single lap at offset 0 → `lat_max - lat_min` wider than without other cars.
  - `test_cli_topography_only` (subprocess or direct `main([...])` call with `--topography-only --sessions <tmp>`): writes the file and prints the `topography:` line.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; document `--topography-only` / `--no-topography` in the `.ps1` header comment and `backend/tools/README.md`.
- [ ] **Step 4:** Run the whole backend suite → PASS. Then run the real build read-only against the real recordings: `.\backend\tools\BUILD_TRACK_REFERENCE.ps1 --topography-only --sessions C:\Users\hoang\Projects\lmu-coach\data\sessions` (writes only into this worktree's `data/track_reference/`). Expected: every track prints a `topography:` line with `(0 with pose)` and paint-only layouts (no real recording has `Pos Y` yet). Record the output in the report.
- [ ] **Step 5: Commit** (code + the generated `data/track_reference/*.topography.json` files) `Build per-track topography with BUILD_TRACK_REFERENCE; runtime loader`.

---

### Task 5: `/trackmap` returns elevation, attitude and a 3D surface

**Files:**
- Modify: `backend/app/main.py` (`get_trackmap`, ~416-984)
- Test: `backend/tests/test_trackmap_elevation.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces (new payload keys; all existing keys unchanged):
  - `elevation`: `{"source": "recorded"|"topography"|"lap"|"none", "orientation": "recorded"|"surface"|"none", "datum_m": float, "note": str, "topography": {"built": str, "files": int, "laps": int, "coverage": float} | null}`.
    - `recorded`: this lap has `Pos Y` and a built topography matched; `lap`: has `Pos Y`, no built topography → topography built on the fly from this file only (cache by `(path, mtime)`); `topography`: no `Pos Y`, built topography matched; `none`: neither (all heights 0).
    - `note` (shown by the UI): `""` for recorded/topography; `"Elevation from this session only"` for lap; `"Flat: no elevation recorded for this track yet"` for none. Append `" (kerbs from earlier laps)"` when source is none but paint exists.
  - `driven_line.y` (len = len(`driven_line.x`)): ground height under the car minus `datum_m`. Recorded/lap: `pos_y - mean(ride height) - 0.25 - datum`; topography: `lookup_height(s, path_lat) - datum`; none: 0.
  - `driven_line.pitch_deg`, `driven_line.roll_deg` (same length): from `vehicle_pose` when a basis is detected (`orientation: "recorded"`), else from `grade_bank` at the car (`"surface"`), else zeros (`"none"`).
  - `track.left_y`, `track.right_y` (same length as `track.left_x`): surface height at lat = -te / +te (minus datum; 0 when none).
  - `surface`: `{"rows": R, "cols": 11, "x": [R*11], "y": [...], "z": [...], "kind": [...]}` row-major; row r = the r-th downsampled edge sample (same `_ds(…, 600)` indices as `track.*`), col c at `t = c/10` from left edge to right edge: `x,z = lerp(left, right, t)`, `lat = -te + 2*te*t`, `y = lookup_height(s, lat) - datum` (0 when none), `kind = paint_at(s, lat)` (0 when none).
  - `corner_labels[i].y`: surface height at that label (0 when none).
  - `s` for lookups = the game's raw `Lap Dist` interpolated onto the position rows (same `_interp_to_length` path as Path Lateral), taken modulo the layout's `length_m`. Lap length for layout matching = `max(Lap Dist) - min(Lap Dist)` over the lap slice.
- Keep the new work in helper functions (e.g. `_elevation_payload(...)`) rather than growing `get_trackmap` inline; reuse the existing `_col`, `_cols4`, `_interp_to_length`, `_ds`, `line_idx`.

- [ ] **Step 1: Write failing tests** (`test_trackmap_elevation.py`), pattern of `test_time_alignment.py` (`monkeypatch.setattr(main, "OWN_SESSIONS_DIR", dir)`, `TestClient(main.app).get(.../trackmap, params={"lap_number": 1})`), with `TRACK_REFERENCE_DIR` → `tmp_path` and `slug_for_track` monkeypatched:
  - `test_trackmap_lap_source`: hilly session, no built topography → `elevation.source == "lap"`, `orientation == "recorded"`, `driven_line.y` range ≈ 16 m (± 0.5), `len(y) == len(x)`, `surface.rows*surface.cols == len(surface.x)`, `track.left_y` same length as `track.left_x`, pitch_deg range within ±3° of the true grade angle.
  - `test_trackmap_recorded_source`: after `build_topography`/`save_topography` → `"recorded"`, `elevation.topography.files == 1`.
  - `test_trackmap_topography_source_for_legacy`: build from a pose file, then request a `with_pose=False` file of the same ring → `"topography"`, `orientation == "surface"`, heights match truth − datum within 0.1 at 20 random driven-line samples.
  - `test_trackmap_legacy_file_flat`: legacy file, no topography → `"none"`, note `"Flat: no elevation recorded for this track yet"`, all y zeros, arrays have the right lengths, existing keys unchanged.
  - `test_surface_kind_has_kerb`: hilly session → some `surface.kind == 5` on the right-hand columns only.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the whole backend suite → PASS (including `test_trackmap_real_file.py`, which uses real recordings when present). Also call `/trackmap` through `TestClient` for one real file from `C:\Users\hoang\Projects\lmu-coach\data\sessions` (read-only; e.g. the newest Fuji file) and note `elevation` and payload size in the report (target: payload growth < 400 KB).
- [ ] **Step 5: Commit** `/trackmap: elevation, attitude and 3D surface ribbon for the chase view`.

---

### Task 6: UI — vendored libraries, view switch, 3D scene (static frame)

**Files:**
- Create: `backend/app/static/vendor/three/three.module.min.js`, `backend/app/static/vendor/three/three.core.min.js`, `backend/app/static/vendor/three/LICENSE` (from npm `three@0.186.1`, `build/`)
- Create: `backend/app/static/vendor/gsap/gsap.min.js`, `backend/app/static/vendor/gsap/LICENSE` (npm `gsap@3.15.0`, `dist/gsap.min.js`)
- Modify: `backend/app/static/index.html`
- Create: `backend/tests/ui/serve_for_ui.py`, `backend/tests/ui/conftest.py`, `backend/tests/ui/test_chase_view.py`

**Interfaces:**
- Consumes: Task 5 payload; existing globals in `index.html`: `S` (state), `drawMap`, `loadMap`, `lerpAt`, `angleAt`, the overlay colour helpers (`colTB/colBal/colYaw/colScalar`), `S.slipGain`, `fontData`.
- Produces (used by Task 7):
  - `S.view`: `'top'|'chase'` (persisted in `localStorage['lmu-coach.view']`).
  - A toolbar control in `.map-toolbar` before `.legend`: a two-option segmented radiogroup "View": **Top-down** / **Chase** (`role="radiogroup"`, arrow-key navigation, 44 px targets, visible focus), plus keyboard shortcut `V` toggling when focus is not in a text field.
  - `<canvas id="chase-canvas">` inside `.map-body`, sibling after `#map-canvas`, `position:absolute; inset:0; opacity:0; pointer-events:none` (z below the HUDs), and DOM layers `#chase-labels` (corner pills + β pill) and `#chase-note` (elevation note pill, top-left).
  - `window.Chase` object (inline `<script type="module">` at the end of `index.html`): `ensure(): Promise<boolean>` (lazy-load three + gsap, create renderer; false + message on failure), `build(mapData)` (dispose old meshes, build scene), `render()` (draw current playhead), `setOverlayColors()`, `resize()`, `state()` (debug: `{ready, built, mix, view, cam:{x,y,z}}`), and `lineColorAt(i)` shared helper extracted from `drawMap`'s per-segment colour logic (2D uses it too, so both views colour the line identically).
  - Scene, all in the Three frame from Global Constraints: surface `Mesh` from `surface` (indexed grid, `MeshLambertMaterial({vertexColors:true})`; colours road `#151D31`, kerb alternating `#D9464E`/`#E6E9F0` every 2 m of lap distance, grass `#1E3527`, dirt `#3A3226`, gravel `#4A4232`); edge lines (`LineBasicMaterial`, white, opacity 0.35) at `left_y/right_y + 0.02`; transverse distance marks every 50 m (opacity 0.06); driven line as a 0.35 m-wide ribbon at `y + 0.03` with per-vertex colours from `lineColorAt`; the car as the 2D chevron outline extruded (`ExtrudeGeometry`, 4.7 m long × 2.0 m wide × 0.45 m, nose +X, same proportions as `drawChevronArrow`) plus a white velocity arrow along the course; `HemisphereLight('#9FB3D9', '#0A0F1C', 1.1)` + `DirectionalLight('#ffffff', 1.2)` at (-0.5, 1, 0.3); background and `Fog('#060912', 150, 900)`. Car attitude per Global Constraints, `bodyOff` exactly as 2D (`clamp(beta*S.slipGain,-60,60)` deg), colour exactly as the 2D US/OS colour logic.
  - Integration: `drawMap()` — when `S.view==='chase'` and the transition is finished, call `Chase.render()` and skip the 2D draw; `loadMap()` end → `Chase.build(S.mapData)` if the chase view is active (else mark dirty); `onModeChange` → `Chase.setOverlayColors()`; `setSlipGain` → render; the `ResizeObserver` → `Chase.resize()`. Map hover tooltip and right-drag pan are disabled in the chase view; hold-to-pause still works.
  - In this task, switching to Chase may cut directly (no animation yet) to a fixed chase pose; Task 7 adds the camera behaviour and the transition.
  - Failure path: `ensure()` false (no WebGL / import error) → inline message in `#map-overlay` style: "Chase view needs WebGL, which isn't available here. Showing top-down." and the radio returns to Top-down.
- `backend/tests/ui/serve_for_ui.py`: `python serve_for_ui.py <sessions_dir> <port> [<track_reference_dir>]` — sets `app.main.OWN_SESSIONS_DIR`, optional `TRACK_REFERENCE_DIR`, runs `uvicorn.run(app.main.app, port=port)`.
- `conftest.py`: session fixture that writes a hilly session and a legacy session (`synthetic_track.write_hilly_session`) to a tmp dir, starts the server on a free port as a subprocess, yields the base URL, kills it; a `page` fixture (sync Playwright, Chromium with args `["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]`) that records console errors; skip the whole module if Playwright/Chromium is unavailable.

- [ ] **Step 1: Vendor the libraries.** `npm pack three@0.186.1 gsap@3.15.0` in `$env:TEMP\opencode\vendor-<random>` (its own empty directory), extract, copy only the listed files + licences. Verify `three.module.min.js` imports `./three.core.min.js` and nothing else external.
- [ ] **Step 2: Write failing UI tests** (`test_chase_view.py`):
  - `test_vendor_served`: `GET /static/vendor/three/three.module.min.js` 200 with a JavaScript content type.
  - `test_top_down_does_not_load_three`: open `/`, select the hilly session + lap 1, assert no request to `/static/vendor/` was made.
  - `test_switch_to_chase_renders`: click "Chase" → wait for `window.Chase.state().built` → screenshot of `#chase-canvas` region has pixel standard deviation > 8 (not blank) → no console errors.
  - `test_chase_falls_back_without_webgl`: a context with `--disable-webgl`/`--disable-3d-apis` (or `HTMLCanvasElement.prototype.getContext` stubbed to return null for webgl via `add_init_script`) → after clicking Chase, Top-down is checked again and the message is visible.
  - `test_legacy_session_chase_shows_flat_note`: legacy session → Chase → `#chase-note` text contains "Flat".
  - `test_view_switch_keyboard`: focus the radiogroup, ArrowRight → Chase checked; `V` toggles back.
- [ ] **Step 3:** Run the UI tests → FAIL.
- [ ] **Step 4:** Implement the HTML/CSS/JS.
- [ ] **Step 5:** Run the UI tests → PASS; run the backend suite → PASS. Capture screenshots (top-down, chase) into the report directory (not committed).
- [ ] **Step 6: Commit** `UI: chase view scene (Three.js, lazy-loaded) behind a Top-down/Chase view switch`.

---

### Task 7: UI — chase camera, animated transition, zoom, polish, docs

**Files:**
- Modify: `backend/app/static/index.html`
- Modify: `backend/tests/ui/test_chase_view.py`
- Modify: `DevNotes.md`, `docs/HANDOFF.md`, `CLAUDE.md` (layout line for `track_topography.py`, `vehicle_pose.py`, vendor dir)

**Interfaces:**
- Consumes: Task 6 `window.Chase`, `S.view`.
- Produces:
  - **Chase camera** (each `render()`): car point `C` = lerp of `driven_line.x/y/z` at `dotF` (same fractional index the 2D map uses). Course direction = `angleAt(heading, dotF)`, then exponentially smoothed in time on the circle (shortest arc) with τ = 0.35 s, `alpha = 1 - exp(-dt/τ)`, `dt` clamped to 0.1 s. Camera position = `C - dir*D + up*2.6` where `D` = chase distance (default 7.5 m); camera height smoothed with τ = 0.5 s and kept ≥ ground under the camera + 1.0 m (ground = `driven_line.y` at the point `D` metres back along the line by cumulative distance). Look-at = `C + dir*12 + up*0.8`. FOV 55°, near 0.1, far 2000.
  - **Snap rule:** if the playhead moved by more than 2 % of the lap since the last render (scrub, lap chip, hover jump) or the lap changed, reset the smoothed course/height to their targets (no swing across the track).
  - **Settling while paused:** after a pause or a smoothing change, keep rendering via rAF until the course and height are within 0.001 rad / 0.005 m of target, then stop.
  - **Transition** (`setView('chase'|'top')`), GSAP `power2.inOut`, 1.1 s (top→chase) and 0.8 s (chase→top), driving `mix` 0..1:
    - Top-down pose `P0` exactly matches the current 2D framing: centre `((-S.panX)/sc + S.fitCxW, y_ref, (-S.panY)/sc + S.fitCzW)` with `sc = S.fitSc*S.zoom`, height `d = (H/sc/2)/tan(fov/2)` above `y_ref` (= car ground height), looking straight down with `up = (0,0,-1)`.
    - Chase pose `P1` as above. Position: horizontal `lerp(P0,P1,mix)`, height above the car interpolated in log space (`exp(lerp(ln h0, ln h1, mix))`); orientation `Quaternion.slerp(q0, q1, mix)`.
    - `#chase-canvas` opacity: 0→1 over the first 15 % (top→chase); 1→0 over the last 15 % (chase→top). The 2D canvas keeps drawing underneath until it is covered.
    - A new `setView` call kills a running tween and starts from the current `mix` (no stacked tweens).
    - `prefers-reduced-motion: reduce` → duration 0 (instant cut).
  - **Zoom in chase view:** wheel, ± buttons and the slider change `S.chaseDist` (4-30 m; slider maps `zoom` 0.5-10 → `D = clamp(7.5/zoom, 4, 30)`), stored separately from the 2D zoom/pan, which are restored on return. Fit resets `D` to 7.5.
  - **Labels:** corner pills (same look as the 2D pills: amber text on `rgba(7,11,20,0.85)`, Fira Code) projected each render; hidden when behind the camera or > 600 m away; β pill near the car with the same text format as 2D (`β +1.2°  ×5`).
- Docs: `DevNotes.md` entry under `### 10/25/2026`: what was added, the in-game TODOs (verify `Pos Y`/Ori values and the detected convention printed by the build tool; verify `Other Cars` rows are populated in a race/AI session; `Path Lateral` sign evidence), rebuild reminder (`docker compose down; docker compose up -d --build`, then record a few laps and run `BUILD_TRACK_REFERENCE.ps1 --topography-only`). HANDOFF §4.6 schema gets the new tables; §9 file list gets the new files.

- [ ] **Step 1: Write failing UI tests** (append to `test_chase_view.py`):
  - `test_transition_animates`: click Chase; at ~0.3 s `0 < Chase.state().mix < 1`; after 1.5 s `mix == 1` and the camera is within 15 m of the car.
  - `test_reduced_motion_is_instant`: context with `reduced_motion="reduce"` → `mix == 1` on the next frame after clicking Chase.
  - `test_switch_back_restores_top_down`: Chase then Top-down → after 1.2 s `mix == 0`, `#chase-canvas` opacity 0, 2D zoom/pan equal to before.
  - `test_scrub_jump_snaps_camera`: in chase view, set the scrubber to 0.1 then 0.6 → after one render, camera-to-car horizontal distance within 1 m of `D` (no swing).
  - `test_lap_change_in_chase_view`: click another lap chip while in chase view → `Chase.state().built` stays true, no console errors, camera near the new car position.
  - `test_chase_zoom_independent`: in chase view press + twice → `S.chaseDist` < 7.5 while `S.zoom` (2D) unchanged after returning.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the camera, transition, zoom and labels.
- [ ] **Step 4:** Run the UI tests and the backend suite → PASS. Capture screenshots of: mid-transition, chase on a straight, chase on the banked corner (hilly synthetic), a legacy real recording in chase (via a session dir pointing at a copy-free read-only path: `serve_for_ui.py C:\Users\hoang\Projects\lmu-coach\data\sessions`). Look at them and fix anything visibly wrong (z-fighting, line under the surface, labels misplaced).
- [ ] **Step 5:** Update `DevNotes.md`, `docs/HANDOFF.md`, `CLAUDE.md`.
- [ ] **Step 6: Commit** `UI: chase camera follows the smoothed direction of travel; animated top-down <-> chase transition`.
