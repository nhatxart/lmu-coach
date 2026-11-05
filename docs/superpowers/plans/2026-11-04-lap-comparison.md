# Lap Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare your lap with up to 4 other laps (same file or other files, same track layout and car class): ghost cars on the map and in the chase view, a time-delta trace, overlaid input traces with synced zoom, and the line coloured by gain/loss.

**Architecture:** Backend: a lap index per file (lap times, complete/real flags) shared by `/laps` and a new metadata-only `/sessions/index` (track, class, model, layout length, laps; cached by mtime); `/trackmap` projects GPS around a fixed per-track origin (the track reference's `sources/<slug>.json` `center`) and adds per-point game lap distance and lap time. Frontend (`index.html`): a `Compare` module (`window.CMP`) loads ghost laps through `/trackmap`, aligns them by game lap distance, and feeds a Compare tab, map/chase drawing hooks and a traces block.

**Tech Stack:** Python 3.12 FastAPI + DuckDB (backend, pytest); vanilla JS/canvas + three.js 0.186.1 (vendored) in the single-file UI; Playwright browser tests.

**Spec:** `docs/superpowers/specs/2026-11-03-keep-laps-and-lap-comparison-design.md` §5 (and §4.5, the partial-lap "best" bug, which comparison's best-lap helpers depend on). Visual reference only: branch `spike/compare-mockup` commit `b83a068` (search `const CMP=` and `ghostSync` in its `index.html`, `sessions_index` in its `main.py`). Do not merge or copy it wholesale; it has none of the alignment, layout, race and test work below.

## Global Constraints

- PowerShell: chain with `;`, never `&&`. Never modify `.env`, `lmu-coach.config.json`, `data/sessions/` (read-only queries are fine; real recordings at `C:\Users\hoang\Projects\lmu-coach\data\sessions`).
- UI changes stay inside `backend/app/static/index.html`. No new libraries, no new runtime deps. Canvas colour literals stay in sync with `:root` tokens (DESIGN.md); add tokens `--ghost-1:#F472B6; --ghost-2:#A3E635; --ghost-3:#FB923C; --ghost-4:#A78BFA; --you:#E8ECF6`. WCAG AA, keyboard operable, sentence case.
- Backend rule (HANDOFF §8): full-length first, slice last; any new per-point array goes through the same `line_idx` sampling as `driven_line.x`.
- `/trackmap` must stay backward compatible: only add keys. Every new step is fail-closed like the elevation block (an error costs the new keys, never the response).
- Comparison is refused, not hidden behind a toggle, for: another car class; another layout; another track; a different coordinate system (`gps_projected` vs `world_metres`). Each refusal shows its reason in the Compare tab.
- Up to 4 ghosts. Sign convention everywhere: delta = your time - ghost time at the same lap distance; `+` = you are slower (coral `--c-brk`), `-` = faster (teal `--c-thr`). Line offset: metres, `+` = ghost to the right of your direction of travel.
- Scale-aware canvases: draw in panel units and size the bitmap with `panelScale()` exactly as `redrawTraceWithDot` does (it follows the Size setting).
- Panel Ctrl+scroll (Size) keeps working over the Compare traces: the traces handle only wheel events WITHOUT `ctrlKey`.
- Comment the why in the file's style (`PREVIOUSLY … Fix:` for changed behaviour). Any module the code reaches through `window.X` is assigned to `window` explicitly.
- Commit: `git -c user.name=hoang -c user.email=hoangnhatvn2002@gmail.com commit -F <msgfile>` (write the message file UTF-8 without BOM), message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests (repo root): backend `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic python -m pytest backend/tests -q --ignore=backend/tests/ui`; UI `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic --with "uvicorn[standard]" --with playwright python -m pytest backend/tests/ui -q`.

## Review Focus

1. Your lap changes while a ghost is still loading: the late response must not attach to the new lap (or a ghost of another track/class). (Task 4 test `test_late_ghost_response_is_dropped`)
2. Same-time mode past a lap's end: the slower lap's ghost/you hold at the finish line; no extrapolation, no NaN. (Task 4 test `test_same_time_holds_at_finish`)
3. Files whose positions are in different coordinate systems (`world_metres` recorder-v1 vs GPS) or have no `Lap Dist`: refused with a reason, or aligned by driven distance with "approximate" shown; never silently misdrawn. (Task 4 tests `test_coordinate_system_mismatch_refused`, `test_no_lap_dist_falls_back_and_says_approximate`)
4. A corrupt or half-written `.duckdb` in either folder: `/sessions/index` lists it with an error and keeps going; the picker skips it. (Task 2 test `test_index_survives_unreadable_file`)
5. Ghost lap with skipped game ticks: alignment uses the payload's per-point time, not a uniform split of the lap time. (Task 3 test `test_driven_line_time_follows_game_clock`)

---

### Task 1: Lap index, complete/real laps, and the "best lap" fix

**Files:**
- Modify: `backend/app/telemetry.py` (new `LapInfo`, `lap_index`, `real_lap_flags`; `reference_lap_time` uses `real_lap_flags`)
- Modify: `backend/app/main.py` (`/sessions/{source}/{file}/laps` built from `lap_index`, adds `complete`, `real`)
- Modify: `backend/app/static/index.html` (`selectSession` auto-pick and `renderLapBar` BEST use `real`)
- Modify: `backend/tests/synthetic_track.py` (`write_hilly_session(..., trailing_partial_m: float = 0.0)`: keep driving that many metres past the closing crossing, so the file ends mid-lap with one more `Lap` row, like a session left mid-lap)
- Create: `backend/tests/test_lap_index.py`; Modify: `backend/tests/ui/conftest.py`, `backend/tests/ui/serve_for_ui.py` (new file name `PARTIAL`), Create: `backend/tests/ui/test_compare.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) LapInfo(lap_number: int, lap_time_s: float, complete: bool, real: bool)`; `lap_index(con: duckdb.DuckDBPyConnection) -> list[LapInfo]` (same lap numbers and complete-lap times as `load_telemetry_file`, official 1 ms time when within `OFFICIAL_LAP_TOLERANCE_S`; the open last lap ends at the last `GPS Time` value; laps >= 600 s dropped like `/laps` today); `real_lap_flags(times: list[float], complete: list[bool]) -> list[bool]` (complete, < 600 s and >= 0.85 x median of the complete laps' times; the rule `reference_lap_time` already uses). `/laps` items: existing keys plus `complete: bool`, `real: bool`.

- [ ] **Step 1: Failing tests** in `backend/tests/test_lap_index.py`:
  - `test_lap_index_matches_load_telemetry_file`: synthetic file (3 laps); `[(l.lap_number, round(l.lap_time_s,3), l.complete)]` from `lap_index` equals the same from `load_telemetry_file` for every complete lap.
  - `test_trailing_partial_lap_is_not_real`: `write_hilly_session(..., trailing_partial_m=300)` → last item `complete is False`, `real is False`, its time ≈ 300 / 40 s (± 0.2); the full laps `real is True`.
  - `test_real_lap_flags_rule`: `real_lap_flags([100,101,36,175], [True,True,True,True]) == [True,True,False,True]`; incomplete → False.
  - `test_reference_lap_time_unchanged`: `reference_lap_time` on the synthetic laps equals its value before the refactor (compute it with the old rule inline in the test).
  - `test_laps_endpoint_has_flags`: `/sessions/sessions/<f>/laps` items have `complete`, `real`; partial file's last lap `real == False`.
  UI (`test_compare.py`): `test_best_chip_skips_partial_lap`: open `PARTIAL`; the `.lap-chip.best` is not the last lap, and the auto-selected `S.lap` is the fastest real lap.
- [ ] **Step 2: Run** backend file and the UI test → FAIL (no `lap_index`, no flags, BEST on the partial lap).
- [ ] **Step 3: Implement** per Interfaces. `/laps` stops loading the whole file (comment: PREVIOUSLY `/laps` ran `load_telemetry_file`, seconds per big file; the lap list needs only the Lap / Lap Time tables). UI: BEST and the auto-pick = fastest `real` lap, else fastest `complete`, else the first lap (comment the 10/19 Fuji L13 36 s case).
- [ ] **Step 4: Run** both suites → PASS.
- [ ] **Step 5: Commit** "Laps: complete/real flags; BEST skips partial laps".

---

### Task 2: `/sessions/index` (metadata-only, cached)

**Files:**
- Create: `backend/app/session_index.py`
- Modify: `backend/app/main.py` (route `GET /sessions/index`)
- Modify: `backend/tests/synthetic_track.py` (`lap_dist_scale: float = 1.0`: multiplies the written `Lap Dist` values only, to stand in for another layout of the same track name)
- Create: `backend/tests/test_session_index.py`

**Interfaces:**
- Consumes: `telemetry.lap_index`, `car_identity.identify_car`, `track_catalog.find_entry`.
- Produces: `file_entry(path: Path, source: str) -> dict` and `build_index(folders: list[tuple[str, Path]]) -> list[dict]`. Entry keys: `source`, `file`, `track` (metadata `TrackName`), `track_slug` (catalog slug or `None`), `car_model`, `car_class` (`identify_car`; `"unknown"` when unknown), `layout_len_m` (median over complete laps of the largest raw `Lap Dist` inside the lap's time window, 1 decimal; `None` without a `Lap Dist` table), `coordinate_system` (`"gps_projected"` if the file has GPS Latitude/Longitude tables, else `"world_metres"`), `laps` (`[{lap_number, lap_time_s, complete, real}]` from `lap_index`), and `error` (string) only when the file could not be read (then `laps: []`). Cache: module dict keyed by absolute path, valid while `(st_mtime_ns, st_size)` is unchanged. Route returns `{"files": build_index([("sessions", OWN_SESSIONS_DIR), ("reference_laps", REFERENCE_LAPS_DIR)])}`, files in the same order as `/sessions` lists them; read-only connections only.

- [ ] **Step 1: Failing tests** (`test_session_index.py`, `TestClient` with `OWN_SESSIONS_DIR` / `REFERENCE_LAPS_DIR` monkeypatched to tmp dirs):
  - `test_index_entry_fields`: synthetic file with `vehicle_class="LMGT3"` → `car_class == "LMGT3"`, `coordinate_system == "gps_projected"`, `layout_len_m == pytest.approx(1200, abs=15)`, laps equal `/laps`.
  - `test_index_layout_length_follows_lap_dist`: `lap_dist_scale=0.9` → `layout_len_m ≈ 1080 (± 15)`.
  - `test_index_lists_both_folders`: one file in each folder → two entries with the right `source`.
  - `test_index_cache_reuses_and_invalidates`: count `duckdb.connect` calls (monkeypatch wrapper): second request opens nothing; after touching the file's mtime, it opens that file again.
  - `test_index_survives_unreadable_file`: a `.duckdb` file of random bytes → entry with `error` and `laps == []`; the good file's entry is complete; status 200.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** the backend suite → PASS. Also time it once on the real folder through `serve_for_ui.py` (report first and cached request times in the commit message; first request should be a few seconds at most for ~55 files, cached well under 100 ms; if the first is far slower, note it in DevNotes, do not optimise beyond the plan).
- [ ] **Step 5: Commit** "Sessions index: track, class, layout length and laps per file".

---

### Task 3: One coordinate frame per track; lap distance and time per point

**Files:**
- Modify: `backend/app/track_reference.py` (new `track_origin`)
- Modify: `backend/app/main.py` (`/trackmap` projection origin; new keys)
- Create: `backend/tests/test_trackmap_compare.py`

**Interfaces:**
- Produces: `track_origin(track_name: str) -> tuple[float, float] | None`: `REFERENCE_DIR / "sources" / f"{slug}.json"` → `center` `[lat, lon]`, cached per slug, `None` when the file or key is missing or unreadable. Slug: `track_catalog.find_entry(name)["slug"]`, else the name made plain (`track_catalog._plain`, spaces → `_`, e.g. "Synthetic Hill Ring" → `synthetic_hill_ring`), so a track outside the catalog gets a shared frame as soon as a `sources/<slug>.json` with a `center` exists. `/trackmap` new keys: `coordinate_origin: {"kind": "track"|"file"|"world", "lat": float|None, "lon": float|None}` (`track` = fixed origin used; `file` = this file's bbox centre as before; `world` = no GPS, `Pos X/Z`), `lap_len_m: float|None` (the existing local `lap_len_m`, game lap distance span), `driven_line.lap_dist` (game lap distance per line point, from the existing `s_lap` at `line_idx`, metres, 2 decimals; `[]` when `s_lap` is None) and `driven_line.t` (seconds from the lap's first position row, per line point, from the game clock `t_pos` when available, else `lap_time * i / (n-1)`; 3 decimals).

- [ ] **Step 1: Failing tests** (`test_trackmap_compare.py`; pattern of `test_trackmap_elevation.py`, `TRACK_REFERENCE_DIR` to a tmp dir):
  - `test_track_origin_reads_sources_center`: write `<tmp>/sources/synthetic_hill_ring.json` `{"center":[35.37,138.92]}` → "Synthetic Hill Ring" (not in the catalog) gives `(35.37, 138.92)`; a catalog track (`"Fuji Speedway"`) reads `sources/fuji.json` (write one in tmp); missing file → `None`; malformed JSON → `None`.
  - `test_trackmap_uses_track_origin`: with that origin, `coordinate_origin.kind == "track"`, and the first driven-line point's `x, z` equal the projection of its GPS row around (35.37, 138.92) (read the file's first lap GPS rows; tolerance 0.05 m).
  - `test_trackmap_without_origin_unchanged`: no sources file → `kind == "file"` and `x, z` identical (to 1e-9) to a call on the base commit's formula (bbox centre; compute in the test).
  - `test_two_files_share_the_frame`: two synthetic files whose GPS bboxes differ (second one `trailing_partial_m=300` plus `lap_offsets=(3.0,)`), same origin → the point nearest lap distance 600 m on each lap is within 3.5 m of each other (offset 3 m) instead of offset by their bbox-centre difference; with no origin the test records that difference is non-zero (guards the test itself).
  - `test_driven_line_lap_dist_and_time`: `lap_dist` and `t` have the line's length; `lap_dist` rises from ~0 to ~1200 (allow the wrap at the end); `t[-1] ≈ lap_time_s (± 0.05)`; `lap_len_m ≈ 1200`.
  - `test_driven_line_time_follows_game_clock`: a synthetic file with a 2 s gap of skipped ticks inside lap 1 (write it by deleting 200 consecutive 100 Hz rows and 20 slow rows from a copy: helper in the test) → `t` jumps by ~2 s at that point instead of being uniform.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (comment why: PREVIOUSLY each file was projected around its own session's bounding box, so the same corner sat metres apart between files; a fixed per-track origin puts every file of a track in one frame). **Step 4: Run** the whole backend suite (incl. `test_trackmap_real_file.py`: real recordings now use the track origin; update only assertions that encoded absolute coordinates, with a comment) → PASS. Then the UI suite → PASS (the chase view and map read coordinates from the same payload, so nothing else may change).
- [ ] **Step 5: Commit** "Trackmap: one frame per track; lap distance and game time per point".

---

### Task 4: Compare tab, ghosts on the map, delta

**Files:**
- Modify: `backend/app/static/index.html` (tab, panel body, CSS, `Compare` module, hooks in `drawMap`, `lineColorAt`, `updateLegend`, `updateTraceDots`, `loadMap`, `switchTab`, the tab keyboard list)
- Modify: `backend/tests/ui/conftest.py`, `backend/tests/ui/serve_for_ui.py` (new fixture files below), `backend/tests/ui/test_compare.py`

The UI fixture's track reference dir gets `sources/synthetic_hill_ring.json` with `{"center": [35.37, 138.92]}` (the generator's GPS reference), so the fixtures share the track frame like real catalog tracks. New UI fixture files (all "Synthetic Hill Ring", in the sessions folder unless noted): `CMP_A` (`vehicle_class="LMGT3"`, `lap_offsets=(0.0, 1.0, -1.0)`), `CMP_B` (`"LMGT3"`, `lap_offsets=(2.0, 2.0)`, `T0` shifted so laps are slower: pass a `speed_scale=0.97` param, add it to `write_hilly_session` if missing: scales `SPEED_MS` for that file), `CMP_HYPER` (`"Hypercar"`), `CMP_LAYOUT` (`"LMGT3"`, `lap_dist_scale=0.9`), `CMP_KEPT` (copy of `CMP_B` in the reference-laps folder; `serve_for_ui.py` gains an optional 4th argument: reference laps dir), `CMP_V1` (`"LMGT3"`, positions only as `Pos X/Z`: no GPS tables; add `with_gps=True` param).

**Interfaces:**
- Consumes: `/sessions/index` (Task 2), `/trackmap` `coordinate_origin`, `lap_len_m`, `driven_line.lap_dist`, `driven_line.t` (Task 3), `panelScale()`, `drawMap`, `toScreen`, `lerpAt`, `angleAt`.
- Produces: `window.CMP` = `{ ghosts: Ghost[] (read-only view), align: 'time'|'dist', add(src, file, lap): Promise<{ok: boolean, reason?: string}>, remove(id), setAlign(mode), ghostPose(g): {x, z, yaw, f, fMe}, meAtDist(s): number /* fractional index of your lap at lap distance s */, deltaAt(g): number, onMainLap(), onPlayhead(), drawLines2D(ctx), drawMarkers2D(ctx), deltaColAt(i): string }`. `Ghost = {id, src, file, lap, lapTime, col, cls, model, d /* trackmap payload */, s: Float64Array /* lap distance per point */, t: Float64Array, delta: Float64Array /* per YOUR point */, lat: Float64Array /* per YOUR point */, approx: boolean}`. Hooks call `window.CMP` only if present.
- Behaviour (spec §5.1-5.3):
  - Picker options = index entries with: same `track_slug` (else same `track`), same `car_class` (if yours is `unknown`: only `unknown`, labelled "class unknown"), `layout_len_m` within 1 % of yours when both known, same `coordinate_system` (`gps_projected` or `world_metres`; the frame itself is checked after loading, below). "This session" first, then by date; kept laps tagged "kept". Lap select: that file's complete laps, `real` best marked ★; partial laps not offered. Header text: `"<class> only · N files on this track"`.
  - Quick buttons: "Best lap of this session" (fastest real lap other than yours) and "Best other file" (fastest real lap across the other offered files, from the index only: no trackmap loads to search).
  - `add` re-checks after loading the payload. Frame: both `track` with the same `lat/lon`, or the same file, or both `world` (game world metres are one frame) → exact; both `file` (a track with no `sources` centre) → allowed, the ghost shifted by the mean position difference at 200 matching lap distances and marked "approximate"; `world` against GPS → refused. Layout by `lap_len_m` within 1 % (fallback: driven length within 3 %). Refusals return `{ok:false, reason}` shown under the picker: "<class> lap: only <your class> laps can be compared", "Another layout (<a> m vs your <b> m)", "Different position data (this file has no GPS)".
  - Alignment: by `driven_line.lap_dist` when both laps have it; otherwise by distance driven scaled to the lap, and the ghost row shows "approximate". Delta at your point i = `t_me[i] - t_ghost(s_me[i])` (interpolated on the ghost's `s`). Same time: ghost index where `t_ghost = t_me(playhead)`, held at its last point once past its lap time (you likewise in the delta/pill). Same distance: ghost index at `s_me(playhead)`.
  - Line offset as the mockup: nearest ghost point within ±20 points of the same distance, signed by `(-sin h, cos h)` of your heading.
  - Race guard: every `add` and `onMainLap` bumps a generation counter; a payload that arrives for an older generation is dropped. Main-lap change: other track → all ghosts go; other class → that class's ghosts go; a ghost equal to your new lap goes; the rest are re-aligned.
  - Map: ghost lines dashed 1.6 px in their colour under your line; markers (filled chevron at 55 % + outline) under your chevron, labelled `G1..G4`; delta pill top centre of `.map-body` (`top:58px`, never over the chase view's notices), `aria-hidden="true"` (values are in the list).
  - Overlay option "Time vs ghost 1" in `#heatmap-mode`, `disabled` while there is no ghost (falls back to Throttle / Brake when the last ghost goes); colour by the slope of ghost 1's delta per segment, normalised to the lap's largest slope, grey below 8 %; legend Gaining (teal) / Losing (coral). Calls `Chase.setOverlayColors()` when ghosts change.
  - Estimates say so (spec §5.3): the live delta values and the delta trace heading carry a `title` "Estimate from 10 Hz position and game lap distance"; "approximate" ghosts add "(no shared frame / no lap distance)".
  - Compare tab (4th tab, keyboard list updated): "Laps on screen" list (You + ghosts: colour swatch, `G<n> · L<lap> · <time>`, model · date or "this session", live delta, remove button with `aria-label="Remove ghost <n>"`), picker, quick buttons, a status line (`role="status"`) for refusals and searches, and the Same time / Same distance radio group (`role="radiogroup"`, arrow keys).

- [ ] **Step 1: Failing tests** (`test_compare.py`): 
  - `test_picker_offers_same_class_and_layout_only`: open `CMP_A` lap 1 → file options include this session, `CMP_B`, `CMP_KEPT` (tagged kept); not `CMP_HYPER`, `CMP_LAYOUT`, `CMP_V1`.
  - `test_add_ghost_and_delta_sign`: add `CMP_B` best lap (slower car) → `CMP.ghosts.length == 1`; at playhead 0.5 `CMP.deltaAt(g) < 0` (you are faster) and the list shows a teal value starting with `−`; the pill shows the same value.
  - `test_same_file_ghost_and_best_buttons`: "Best lap of this session" adds a lap of `CMP_A` that is not yours; "Best other file" adds `CMP_B`'s or `CMP_KEPT`'s fastest real lap, and the network log shows no `/trackmap` request for files other than the one added.
  - `test_four_ghost_limit`: a 5th add returns `{ok:false}` with a reason; list still 4.
  - `test_refusals_have_reasons`: `CMP.add` on `CMP_HYPER`, `CMP_LAYOUT`, `CMP_V1` laps → `ok:false`, reasons mention the class / "layout" / "GPS".
  - `test_late_ghost_response_is_dropped`: delay `/trackmap` for `CMP_B` (Playwright `route` with a 1.5 s delay), call `CMP.add`, immediately open `HILLY` (another class) → after the delay `CMP.ghosts.length == 0`, no errors.
  - `test_same_time_holds_at_finish`: ghost slower than you; at playhead 1.0 in Same time the ghost pose is finite and `f < n-1`; in Same distance at 1.0 it is at its last point; no NaN in `ghostPose`.
  - `test_alignment_switch_moves_ghost`: Same distance puts the ghost marker within 2 px of your chevron's lap distance projection on the line; Same time does not (slower ghost behind).
  - `test_delta_overlay_option`: disabled with no ghost; enabled after adding; selecting it colours segments teal and coral (sample `lineColorAt` for both); removing the last ghost resets the select to Throttle / Brake.
  - `test_main_lap_change_rules`: ghosts survive a change to another lap of `CMP_A`; opening `HILLY` (unknown class, same track) removes them.
  - `test_no_shared_frame_is_approximate`: with the fixture's `sources` centre removed for one test (a second server fixture `server_no_origin` with an empty reference dir), adding `CMP_B` works, the row shows "approximate", and the ghost line lies within 3.5 m of yours at lap distance 600 m.
  - `test_coordinate_system_mismatch_refused` (from the list above) and `test_no_lap_dist_falls_back_and_says_approximate`: a fixture lap without `Lap Dist` (add `with_lap_dist=True` param to the generator; file `CMP_NOLD`, LMGT3) adds with the row showing "approximate".
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** the UI suite → PASS; backend suite → PASS.
- [ ] **Step 5: Commit** "Compare tab: ghosts on the map, time delta, same class and layout only".

---

### Task 5: Overlaid traces with synced zoom

**Files:**
- Modify: `backend/app/static/index.html` (Compare tab traces block; `CMP` drawing and zoom)
- Modify: `backend/tests/ui/test_compare.py`

**Interfaces:**
- Consumes: `CMP` from Task 4, `panelScale()`, `onScrub`, `setScrub`.
- Produces: `CMP.view: {s0: number, s1: number}` (metres of YOUR lap distance; whole lap = `{0, lapLen}`), `CMP.setView(s0, s1)`. Canvases `#cmp-delta` (120), `#cmp-spd` (100), `#cmp-thr` (64), `#cmp-brk` (64), `#cmp-str` (76), `#cmp-lat` (76) (heights in panel units), each `tabindex="0"` with an `aria-label`.
- Behaviour: x = your lap distance; every lap plotted at its own `lap_dist` (aligned). You in `--you` 1.6 px, ghosts in their colours. Y fits the visible window (throttle/brake fixed −4..104 %, steering and delta and offset symmetric about 0). Playhead line + a dot per lap at your distance. Visible range label "`<s0>–<s1> m`" when zoomed; a hint line "Scroll to zoom all graphs · drag to pan · double-click for the whole lap · click to move the playhead". Wheel without Ctrl zooms around the cursor (×0.8 / ×1.25), minimum span 25 m; drag pans; double-click resets; click without drag (< 4 px) scrubs to that distance (your lap); keyboard on a focused trace: `+`/`-` zoom about the centre, Left/Right pan 10 %, `0` resets, Enter scrubs to the window centre. The view is shared by all six and kept across ghost changes, reset on a main-lap change to another file. Redraw is rAF-throttled during wheel/drag.
- [ ] **Step 1: Failing tests:** `test_wheel_zoom_syncs_all_traces` (wheel over `#cmp-delta` → `CMP.view` span < lap length and every canvas redrawn with the same window: compare a range label text read via `CMP.view`), `test_drag_pans_and_dblclick_resets`, `test_click_scrubs_to_distance` (click at 50 % of a full-lap view → `S.playhead` such that your lap distance at the playhead ≈ lapLen/2 ± 2 %), `test_ctrl_wheel_still_scales_panel` (Ctrl+wheel over `#cmp-spd` changes the Size, not `CMP.view`), `test_trace_keyboard_zoom`, `test_min_span_25m`, `test_traces_follow_panel_scale` (on-screen height of `#cmp-delta` doubles at 200 %).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** UI + backend suites → PASS.
- [ ] **Step 5: Commit** "Compare: overlaid traces with synced zoom, pan and click-to-scrub".

---

### Task 6: Ghost cars in the chase view; docs

**Files:**
- Modify: `backend/app/static/index.html` (chase module: ghost objects, `render()` hook)
- Modify: `backend/tests/ui/test_compare.py`
- Modify: `DevNotes.md` (`### 11/4/2026` or the day's heading, newest last), `docs/HANDOFF.md` (§2 endpoint table: `/sessions/index`; §4.3: the per-track origin; §9 lines for `session_index.py`, `index.html`)

**Interfaces:**
- Consumes: `CMP.ghosts`, `CMP.ghostPose(g)`, chase internals (`ensureCarModels`, `carModels`, `placeOnSurface`, `surfRowAt`, `surfH`, `surf`, `scene`, `T`).
- Produces: `Chase.state().ghosts: number` (built ghost objects, for tests).
- Behaviour: one ghost per `CMP` ghost: the car model of the ghost payload's class (`mod.buildCar` with `classDims(json, ghost.d.car.requested||class)`), every material cloned and tinted to the ghost colour, opacity 0.35, `depthWrite:false`; if car models are unavailable or fail, a see-through box `CAR_LEN × 1.1 × CAR_W` in the ghost colour with an edge outline. Placed every frame like your car: `placeOnSurface` with the ghost's tyre contacts, the surface search seeded from YOUR point at the same lap distance (`surfRow = surfRowAt(CMP.meAtDist(s_ghost))`), `surfRow` restored afterwards. Without a surface: your lap's `y` at the same distance, attitude flat. The ghost's driven line as a `T.Line` on the surface (+0.08 m) in its colour. Built only when the ghost set, the payload or the surface changes; disposed (geometry + materials) on change and in `disposeWorld`. The pill and the list are the readout (no 3D labels).
- [ ] **Step 1: Failing tests:** `test_chase_builds_one_ghost_per_lap` (two ghosts → `Chase.state().ghosts == 2`; remove one → 1; no console errors), `test_chase_ghost_rests_on_surface` (`HILLY`-like hilly fixture of class LMGT3: add a same-class ghost, switch to Chase, at 10 playhead positions the ghost group's y is within 0.05 m of `surfH` under it: expose a test hook `Chase.ghostProbe(i) -> {y, ground}`), `test_chase_ghost_falls_back_to_box` (block `car_models.js` with a route returning 404 → ghosts still built, as boxes, a single message), `test_chase_ghost_disposed_on_lap_change` (geometry count in `renderer.info.memory.geometries` returns to the pre-ghost value after removing all ghosts).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** UI + backend suites → PASS.
- [ ] **Step 5: Manual look** on real recordings (backend from this branch via `serve_for_ui.py` on `C:\Users\hoang\Projects\lmu-coach\data\sessions` and `data\track_reference`): Fuji 10/19 WRT lap 5 vs lap 9 and vs the Porsche file's best (`Fuji Speedway-The Bend Manthey 2026 _92_LM-2026-10-19T22_49_27Z.duckdb`); screenshots of top-down (pill, lines, markers, delta overlay), the zoomed traces, and chase (ghost cars on the surface). Check the two files now overlay without the mockup's offset correction (lines within ~1 m on the straights). Record the numbers in DevNotes; fix what is visibly wrong.
- [ ] **Step 6: DevNotes + HANDOFF**, then **commit** "Compare: ghost cars in the chase view; notes".
