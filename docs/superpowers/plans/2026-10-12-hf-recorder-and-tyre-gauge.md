# HF Test Recorder + Tyre Gauge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental 1000 Hz recorder that captures the extra channels a physics-based tyre model needs, and rework the tyre gauges to show slip-angle direction, plus smooth chevron playback and hold-to-pause.

**Architecture:** `recorder/recorder_hf.py` is a copy of `recorder.py` that is edited freely; the working recorder is not touched. The backend gains one per-wheel channel in `/trackmap`. All UI work stays in `static/index.html`.

**Tech Stack:** Python 3.13 + uv (recorder), stdlib `unittest`; FastAPI/numpy/scipy (backend), pytest in an ephemeral `uv run --no-project --with` env; vanilla JS canvas.

**Spec:** the "Design" section below (agreed in chat on 2026-10-12; there is no separate spec file).

## Design

### Recorder (`recorder_hf.py`)
- Polls at `--poll-hz` (default 1000) with a sleep+spin pacer; Windows `time.sleep(0.001)` really takes ~1.5 ms.
- Stores one sample per game tick (when `mElapsedTime` changes). `--every-poll` stores every poll instead.
- Counts polls, unique ticks, the histogram of clock steps, and polls where car data changed but the clock did not. Printed at each save and written to `metadata`.
- `--probe SECONDS` runs only that measurement, records nothing, and exits. This answers "does LMU publish faster than 100 Hz".
- 10 Hz channels are gated by game time (every 0.1 s), not by sample count.
- `channelsList.frequency` is the measured rate (`(n-1)/span`), not the nominal one.
- `Lap Time` uses `mLastLapTime` (1 ms precision) when it arrives within 3 s of the crossing and agrees with the clock difference within 0.05 s; otherwise the clock difference is kept.
- New channels (units; `W` = per wheel `value1..4`):
  - Tick rate scalar: `Poll Time` s, `Local Vel X/Y/Z` m/s, `Yaw Accel` rad/s², `Front Downforce` N, `Rear Downforce` N, `Drag` N, `Front Ride Height` m, `Rear Ride Height` m, `Steering Shaft Torque` Nm.
  - Tick rate W: `Susp Force` N, `Ride Height` m, `Camber` rad, `Toe` rad, `Lateral Patch Vel`, `Longitudinal Patch Vel`, `Lateral Ground Vel`, `Longitudinal Ground Vel` m/s, `Tire Deflection` m, `Surface Type` (0-6).
  - 10 Hz W: `TyresWear` (0-1), `TyresCarcassTemp` C, `TyresTempLeft`, `TyresTempCenter`, `TyresTempRight` C, `Brake Temp` C.
  - `metadata`: `RecordedBy=lmu-coach-recorder hf1`, `PollHz`, `EffectiveHz`, `Polls`, `UniqueTicks`, `ClockStepHistogram`, `DataChangedWithoutTick`, `TyreRadiusCm`, `TyreCompoundFront`, `TyreCompoundRear`, `TyreOptimalTempC`, `VehicleModel`.
- New all-zero channels are marked `unpopulated_by_lmu`, as the existing wheel channels are.
- Own stop file `.stop_requested_hf`; launched by `START_RECORDER_HF.ps1`. Not wired into `recorder_control.py`.

### Backend
- `/trackmap` returns `slip_angle: {fl, fr, rl, rr}` in degrees: 8 Hz low-pass at native rate, interpolated to `full_len`, sliced last, zero below 20 km/h, clipped to ±45°.
- Sign: positive = the tyre is travelling to the right of where it points (clockwise on screen). The raw sign convention is measured from a real lap and pinned in `SLIP_ANGLE_SCREEN_SIGN`.
- Lists are empty for files with no `Slip Angle` table (native LMU).

### UI
- Tyre gauge: the suspension bar is removed. The bar now shows slip-ratio magnitude (full at 0.5, blue slip, red lockup). The ring becomes a direction dial: top = where the tyre points; an arc runs from the top to a small square at `slip angle × 9` (10° of slip = 90° on the ring, capped at ±135°). Centre shows slip angle in degrees with the slip ratio beneath it.
- Suspension stays available in the HUD (`F Susp`, `R Susp`), the Axle Balance graph and the Corner Suspension card.
- Chevron: position, heading, β and slip delta are interpolated at the fractional playhead; heading along the shortest arc.
- Hold-to-pause: holding the left button on the map freezes playback; release anywhere resumes. Right button still pans.

## Global Constraints

- Do not modify `recorder/recorder.py`, `recorder/recorder_control.py`, `.env`, `lmu-coach.config.json` or anything in `data/sessions/`.
- PowerShell: chain with `;`, never `&&`.
- The repo is not under git: there are no commit steps.
- New per-sample backend channels: interpolate to `full_len` first, slice `[s_idx:e_idx+1]` last.
- `_lowpass_filter(..., sample_hz=100.0)` runs on native-rate arrays before slicing; do not change it.
- Canvas colours are literals in JS, kept in sync with the `:root` tokens.
- Non-obvious changes get a "why" comment in the house style.
- Log the work in `DevNotes.md` under `### 10/12/2026`.

## Review Focus

- Clock stalls (pause, loading): the pacer must not burst-poll to catch up, and the 10 Hz gate must not emit a burst.
- `mLastLapTime` already updated before the lap number changes, or never updated (invalid lap): the diffed time must survive.
- Recorder started mid-lap: the official time of a lap that was only partly recorded must not be written.
- Slip angle near standstill reaches ±90°: the gauge must read 0, not peg.
- Files without slip angle: the gauge must still show slip ratio and throw no JS errors.

---

### Task 1: HF recorder core (pacer, stats, tick gate, probe)

**Files:**
- Create: `recorder/recorder_hf.py` (copy of `recorder/recorder.py`), `recorder/test_recorder_hf.py`

**Interfaces:**
- Produces: `Pacer(interval_s)` with `.wait()`; `PollStats()` with `.record(et: float, data_changed: bool)`, `.polls`, `.ticks`, `.data_changed_without_tick`, `.histogram() -> dict[str,int]` (keys like `"10.0ms"`), `.summary(wall_s: float) -> list[str]`; `_effective_hz(times: list[float], fallback: float) -> float`; `run_probe(seconds: float, poll_hz: float) -> None`.

- [ ] Step 1: `Copy-Item recorder.py recorder_hf.py`.
- [ ] Step 2: write failing tests:
  - `test_pacer_holds_rate`: 200 `wait()` calls at 0.001 s take 0.17–0.26 s.
  - `test_pacer_does_not_burst_after_stall`: after a 0.05 s sleep, the next 5 waits take ≥ 0.004 s.
  - `test_pollstats_counts`: ets `[1.00,1.00,1.01,1.01,1.03]` → `polls==5`, `ticks==3`, histogram `{"10.0ms":1,"20.0ms":1}`.
  - `test_pollstats_data_change_without_tick`: `record(1.0,False); record(1.0,True)` → `data_changed_without_tick==1`.
  - `test_effective_hz`: 101 times spaced 0.01 → 100.0 (±1e-6); fewer than 2 samples → fallback.
- [ ] Step 3: run `uv run --frozen --no-sync python -m unittest test_recorder_hf -v`; expect failures (names undefined).
- [ ] Step 4: implement the four names; set `DEFAULT_POLL_HZ = 1000.0`, `STOP_FILE = .stop_requested_hf`; in the main loop cache the player scoring/telemetry indices and rescan only when they stop matching, replace `time.sleep(interval)` in the sampling path with `pacer.wait()`, gate `add_sample` on a new `mElapsedTime` unless `--every-poll`, feed `PollStats`, print `summary()` at each flush; add `--probe` and `--every-poll` arguments.
- [ ] Step 5: run the tests; expect all pass.

### Task 2: HF recorder channels, lap time, writer

**Files:**
- Modify: `recorder/recorder_hf.py`, `recorder/test_recorder_hf.py`
- Create: `recorder/START_RECORDER_HF.ps1`

**Interfaces:**
- Consumes: `PollStats`, `_effective_hz` from Task 1.
- Produces: `SessionBuffer.add_sample(telem, wheels, player_scor, current_et, poll_time=0.0)`; `SessionBuffer.update_official_lap_time(last_lap_time: float, current_et: float) -> None`; `SessionBuffer.meta: dict[str,str]`.

- [ ] Step 1: write failing tests using `types.SimpleNamespace` fakes for `telem`, `wheels`, `player_scor`:
  - `test_new_channels_recorded`: one `add_sample` → every channel named in the Design has one row; wheel rows are 4-tuples.
  - `test_ten_hz_gate_is_time_based`: 100 samples at 0.01 s → `len(s["Lap Dist"])` is 10 or 11; a 5 s jump in `et` adds exactly one row.
  - `test_official_lap_time_applied`: crossing at diff 72.46, then `update_official_lap_time(72.4583, et+0.2)` → last `Lap Time` value `72.4583`.
  - `test_official_lap_time_rejected_on_mismatch`: official `95.0` vs diff `72.46` → value stays `72.46` (±1e-9).
  - `test_official_lap_time_expires`: update arriving 3.5 s after the crossing → unchanged.
  - `test_write_session_roundtrip`: write to a temp dir, reopen read-only → new tables exist with the buffered row counts; an all-zero new channel has unit `unpopulated_by_lmu`; `GPS Time` frequency equals `_effective_hz`; `metadata` has `RecordedBy = lmu-coach-recorder hf1`.
- [ ] Step 2: run; expect failures.
- [ ] Step 3: implement. `write_session` reads `getattr(buf, "meta", {})` so `--recover` and `--selftest` namespaces still work; add `"meta"` to `SPOOL_FIELDS`.
- [ ] Step 4: run the tests; expect all pass. Run `uv run --frozen --no-sync python recorder_hf.py --selftest --selftest-minutes 1`; expect `PASS`.
- [ ] Step 5: write `START_RECORDER_HF.ps1` (`cd $PSScriptRoot`, `uv sync`, `uv run recorder_hf.py @args`).

### Task 3: Backend per-wheel slip angle

**Files:**
- Modify: `backend/app/main.py` (helper beside `slip_delta`, response dict)
- Create: `backend/tests/test_slip_angle.py`

**Interfaces:**
- Produces: `_wheel_slip_angles_deg(sa_raw: list, speed_kmh_full: list, s_idx: int, e_idx: int, full_len: int) -> dict[str, list[float]]` with keys `fl fr rl rr`; response key `slip_angle` with the same keys, downsampled like `slip`.

- [ ] Step 1: measure the raw sign on a real lap (script in `%TEMP%\lmu_scan`): correlate mean front `Slip Angle` with the screen heading rate from `/trackmap`. Set `SLIP_ANGLE_SCREEN_SIGN` so a right-hand corner gives a negative value.
- [ ] Step 2: write failing tests: constant `0.05` rad at 100 km/h → every value `SLIP_ANGLE_SCREEN_SIGN * 2.865` (±0.01); speed 10 km/h → all `0.0`; raw `1.5` rad → clipped to ±45; output length `e_idx - s_idx + 1`; empty `sa_raw` → four empty lists.
- [ ] Step 3: run `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy --with scipy --with httpx --with pydantic python -m pytest backend/tests -v` from the repo root; expect failures.
- [ ] Step 4: implement and add to the response; run; expect pass.
- [ ] Step 5: smoke test through `TestClient` on a real recorder file: `slip_angle.fl` has the same length as `slip.fl`, and the right-hand-corner sign holds.

### Task 4: Tyre gauge

**Files:**
- Modify: `backend/app/static/index.html` (`drawTireHUD`, tyre legend, aria labels)

- [ ] Step 1: rewrite `drawTireHUD(head)` per the Design; constants `SLIP_ANGLE_RING_GAIN = 9`, `RING_CAP_DEG = 135`.
- [ ] Step 2: add a "slip angle" entry to `.tire-legend`; update the canvas `aria-label`s.
- [ ] Step 3: verify in the browser against a local server on port 8090: a recorder file shows marker and both numbers; a file without slip angle shows the bar and slip ratio only; the snapshot reports no page errors.

### Task 5: Smooth chevron and hold-to-pause

**Files:**
- Modify: `backend/app/static/index.html` (`drawMap` marker block, `rafLoop`, map mouse handlers)

**Interfaces:**
- Produces: `lerpAt(arr, f)`, `angleAt(arr, f)`; `S.holdPause: boolean`.

- [ ] Step 1: add the two helpers; use them for the playhead marker (hover keeps the exact index).
- [ ] Step 2: `mousedown` button 0 on the map while playing sets `S.holdPause`; `window` `mouseup` clears it and resets `S.lastTs`; `rafLoop` skips the advance while held.
- [ ] Step 3: verify in the browser: playback advances, freezes while held, resumes on release; no page errors.

### Task 6: Notes

- [ ] Add a `### 10/12/2026` entry to `DevNotes.md` and update `docs/HANDOFF.md` §4.1 (tyre force channels are populated in the 10/2–10/4 files) and §9 (new files).
