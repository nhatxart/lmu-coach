# lmu-coach — Handoff (chat → Claude Code CLI)

Snapshot date: 2026-10-12. Last shipped zip in the chat: `lmu-coach-bodyslip-v1.zip`.
This repo snapshot already contains every change made in the chat (layered in order, see §9).

---

## 1. Goal and what the app does

lmu-coach is a **local** Le Mans Ultimate (LMU) telemetry-coaching stack for one sim racer (Thomas, Windows 11, Ryzen 9800X3D, RX 7900 XTX).

1. A **recorder** on the Windows host reads LMU shared memory at ~100 Hz and writes one `.duckdb` file per session, schema-compatible with LMU's own native export.
2. A **FastAPI backend** (Docker) reads those files and serves: a session/lap list, a trackmap + driven-line + per-sample channel payload, and an `/analyze` endpoint that builds a compact grip report and asks a local LLM (llama.cpp) for coaching text.
3. A **single-file vanilla JS/canvas web UI** (`backend/app/static/index.html`) shows the track map with a playback scrubber, driver-input HUD, traction circle, tyre gauges, telemetry graphs (ground speed, ABS/TC, yaw rate, axle balance, corner suspension), a Coach tab (LLM), a Record tab (start/stop recorder + live log) and a Settings bubble (config import/export).
4. Per-car JSON "grip profiles" accumulate across every analyzed lap and include a probabilistic tyre model (slip-angle peak / efficient range / confidence).

---

## 2. Architecture and data flow

```
 LMU (game) ──shared memory──▶ recorder.py ──writes──▶ data/sessions/*.duckdb
                                   ▲ spawned/stopped by                │ (bind-mounted :ro)
                                   │                                   ▼
 browser ──HTTP :8420──▶ recorder_control.py (host agent)     backend (Docker, FastAPI)
    │                      │  config file, LLM health+launch        │  :8080 on host → :8000 in container
    │                      └──▶ launches lmu-coach.ps1 ──▶ llama-server (native Windows, :8000 per user's compose)
    └──HTTP :8080──▶ backend ──HTTP──▶ http://host.docker.internal:<port>/v1  (OpenAI-compatible chat)
```

| Component | Where | Port | Notes |
|---|---|---|---|
| `recorder/recorder.py` | Windows host (uv venv) | – | pyLMUSharedMemory; writes DuckDB; stop via file `recorder/.stop_requested` |
| `recorder/recorder_control.py` | Windows host | **8420** (`LMU_COACH_AGENT_PORT`) | FastAPI: `/ping /status /start /stop /logs/stream(SSE) /config(GET,PUT) /paths/check(GET,POST) /llm/status /llm/ensure` |
| `backend/app/*` | Docker `lmu-coach-backend` | **8080→8000** | `/ /health /sessions /sessions/{source}/{file}/laps /sessions/{source}/{file}/trackmap?lap_number= /analyze /analyze/followup` |
| llama-server | native Windows | user's compose says **8000** (`OLLAMA_BASE_URL=http://host.docker.internal:8000/v1`); `lmu-coach.config*.json` defaults say **8001** — see §6 open question | model path is the GGUF path used as the OpenAI `model` string |
| Web UI | served by backend `/` | 8080 | no build step, one HTML file, Fira fonts bundled |

Mounts (docker-compose.yml): `${LMU_TELEMETRY_PATH}`→`/data/sessions:ro` (source `sessions`), `${REFERENCE_LAPS_PATH:-./data/reference_laps}`→`/data/reference_laps:ro` (source `reference`), `${GRIP_PROFILES_PATH:-./data/grip_profiles}`→`/data/grip_profiles` (rw).
Env fallbacks: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_S` (older names `OLLAMA_BASE_URL` / `OLLAMA_MODEL` are still read by `coach.py`; the user's own compose still uses those).
**The Settings panel sends the LLM URL/model/timeout with each request**, so compose env vars are fallbacks only.

Host-agent security (`GuardMiddleware`): Host header must be localhost/127.0.0.1; `Origin` must be in `host_agent.allowed_origins`; CORS only for those origins. Reason: a web page must not be able to launch programs on the host.

Config file: `lmu-coach.config.json` (path from env `LMU_COACH_CONFIG`, default `<recorder>/../lmu-coach.config.json`). Sections: `host_agent`, `llm` (auto_start, launcher_path, health_url, backend_base_url, model, timeouts, show_window), `recorder.output_dir`, `stack_paths`. Import/export from the Settings bubble; UI also caches in `localStorage['lmu-coach.config.v1']`.

Analyze flow: UI "Analyze" → (if auto_start) host agent `/llm/ensure` → polls `health_url`, runs launcher `C:\llm\ai-infra\launchers\profiles\lmu-coach.ps1` if no server → UI POST `/analyze` `{file, source, lap_number, llm_*}` → `telemetry.load_telemetry_file` → `grip_analysis.build_grip_report` + `tire_model` + `vehicle_dynamics` summary → `grip_profile.update_profile_with_lap` → `coach.get_coaching_analysis` (system prompt + compact JSON, **not** raw telemetry) → text back.

---

## 3. Key design decisions and WHY

| Decision | Why / rejected alternative |
|---|---|
| **Host agent (`recorder_control.py`) instead of the backend launching things** | The Docker container cannot reach Windows shared memory or start host programs (it was confused for a while: "container vs container"). Rejected: running recorder/LLM from the backend. |
| **Stop recorder via a stop-file, not signals** | On Windows `Popen.terminate()` is a hard kill; signal handlers never run. Agent writes `.stop_requested`, recorder polls it each loop, saves, exits. Also deferred stop: agent waits for the save instead of force-killing after 10 s (that force-kill destroyed the write → 12 KB `.tmp`). |
| **CSV + `COPY` bulk load, not `executemany`/list params** | On the user's PC DuckDB inserts cost ≈0.36 ms/value; a session of 150k+ values took minutes (`--selftest` showed 50–240 s per channel). CSV `COPY` is orders faster. |
| **Write to `.<name>.duckdb.tmp`, atomic rename; plus `.spool` pickle safety copy; recover both at startup** | Ctrl+C / crash must not lose a session. Spool is deleted after a successful write. |
| **Keep recorder output schema-compatible with LMU native DuckDB** (tables `Lap`, `eventsList`, `channelsList`, `GPS Time`, …) | The community viewer https://lamerman.github.io/telemetry-visualizer-for-lmu/ failed with `Catalog Error: Table with name Lap does not exist!` / `eventsList does not exist`. Also lets backend treat native and recorder files uniformly. |
| **GPS Time = `telem.mElapsedTime`, not `scoringInfo.mCurrentET`** | `mCurrentET` refreshes ~5 Hz → lap times quantised to ~0.1 s. `mElapsedTime` is 100 Hz and is the same clock LMU's native export uses. |
| **Original `recorder.py`/`START_RECORDER.ps1` flow was kept** (user: "the original worked just fine") | Don't rewrite what works; only add session-end detection, truncation fixes, saving. Standalone use still works; the web Record tab is optional. |
| **llama.cpp (llama-server) instead of Ollama** | User decision; runs natively on Windows on the 7900 XTX. Qwen3.6-27B UD-Q4_K_XL, moving to Qwen3.8-27B. Backend speaks plain OpenAI `/v1/chat/completions`. |
| **One self-contained `index.html`, vanilla JS + canvas, no framework/build** | Served from the FastAPI static dir, zero toolchain, easy to ship as a zip. Fonts are bundled (offline). |
| **Slip-angle-relative grip proxy, not Fx/Fy/F_available** | LMU exposes no tyre forces/loads/grip fraction (zero in recorder, absent in native). GU = lap's 95th-percentile \|slip angle\| ÷ learned peak slip angle (cap ≈1.5). 95th percentile because raw max was dominated by single-tick outliers (13 of 1680 samples >10° → everything pinned at the cap). |
| **Probabilistic, accumulating tyre model in the per-car profile JSON** | Peak seen in one lap is a hypothesis. Confidence = `1-exp(-n/4000)`, never 1.0. The LLM prompt is told the confidence. A lap is scored against the model *before* its own samples are folded in. |
| **Low-pass 8 Hz, 2nd-order Butterworth (filtfilt) on Yaw Rate and slip-angle delta, applied at native 100 Hz BEFORE slicing to the lap** | FFT of yaw/slip showed real signal <1 Hz with a thin tail >3–5 Hz (physics jitter). Filtering before slicing avoids edge artefacts at start/finish. |
| **Everything is interpolated to `full_len` first, sliced `[s_idx:e_idx+1]` last** | Position can be 10 Hz (3924 samples/405 s) while channels are 100 Hz (39239). Elementwise math across mismatched lengths caused `IndexError` in `_slip_ratio_col`. |
| **Mirroring fix = negate exactly ONE axis** | See §4. The earlier "negate both" was a 180° rotation, not a mirror. |
| **Dynamic chart ranges (min/max ±5 % margin) for all graphs except brake, throttle, steering** | Axle balance lives in negative values (rake) and a fixed range hid it. |
| **Body slip β is reconstructed (yaw integral − course), calibrated on straights; refuses rather than guesses** | Earlier approach (integrating lateral velocity from lateral G) was tried and was unusable (see `vehicle_dynamics.py` docstring). Detail in §4. Gate is median/fraction based, not std based, because a std gate wrongly rejected a lap containing a real spin. |
| **On-screen β exaggeration (×1/×3/×5/×10, default ×5, cap ±60°) with true value in label** | Real β is ~1–7°, invisible at true scale. |
| **Settings JSON + host agent owns config** | For distribution to other users; docker-compose lines can be generated from Settings > Stack paths. |
| Design system: Fira Sans + Fira Code, Lucide sprite, CSS tokens, WCAG AA, container queries | Full UI redesign requested ("retain all info/functions, only layout/visuals change"). See `DESIGN.md`. |

---

## 4. Domain knowledge (not obvious from code)

### 4.1 Shared memory / recorder quirks
- Library: `recorder/pyLMUSharedMemory/` (vendored). Player found by `mIsPlayer` in scoring; telemetry slot mapped by `mID`.
- **Actual poll rate ≈ 96.5 Hz**, not 100. Timestamps come from `mElapsedTime`, so dt is *not* 0.01; `compute_body_slip_deg` derives dt from the lap's real time span.
- `scoringInfo.mCurrentET` ≈ 5 Hz (don't use as master clock). Used only for the `ts` of `Lap`/`Lap Time` events.
- **CORRECTED 2026-10-12:** the next bullet is out of date. Tire Load / Lateral Force / Longitudinal Force / Grip Fraction are populated in the six recordings of 10/2–10/4 (Aston #007, Cadillac #12; load up to ~15–21 kN) and empty in every earlier file. Car class vs game update is not yet separated. See DevNotes 10/12/2026. **CORRECTED AGAIN 2026-10-18:** the zero-vs-populated split is by CLASS, not by date. Hypercars (Aston #007, Cadillac #12, Ferrari 499P, Genesis GMR-001) record real Tire Load / Lateral / Longitudinal Force / Grip Fraction; every LMGT3 (BMW M4, Porsche 911 GT3 R, Ford Mustang, and by team name the Lexus) is all zero, in every file. The Iron Lynx #61 (Mercedes GT3) files from 9/19 only looked populated: recorder v1 wrote zero tables without the unpopulated_by_lmu flag. LMP2/LMP3 not yet recorded. The cause is on the game side (the library maps the fields for all cars; TinyPedal and lmuFFB report the same); GT3 load comes from Susp Force instead. See DevNotes 10/18/2026.
- Zero-filled by LMU for every car tested (up to 9/30): **Tire Load, Tire Lateral Force, Tire Longitudinal Force, Grip Fraction**. Recorder detects all-zero wheel channels and marks `channelsList` unit `unpopulated_by_lmu` instead of faking data. Native LMU files simply don't have these tables.
- Recorder derives: `Slip Angle` per wheel = `atan(mLateralGroundVel/mLongitudinalGroundVel)` (**radians**, value1..4 = FL, FR, RL, RR); `Yaw Rate` = `mLocalRot.y` (rad/s).
- Freeze watchdog (`ConnectionState`): `mElapsedTime` stops for >2 s (`FREEZE_TIMEOUT_S`), garage entry, vehicle-slot change, or elapsed time going backwards → flush current session and start a new buffer. Sessions < 200 samples are discarded.
- Rates: 100 Hz position/speed/inputs/G/yaw/wheel/slip/susp/RPM; 10 Hz `Path Lateral`, `Track Edge`, `Lap Dist`, tyre thermals; event tables (`Gear`, `ABS`, `TC`, `In Pits`, `Lap`, `Lap Time`) on change/crossing.
- `recorder.py` default output: `<repo>/data/sessions` anchored to `__file__` (a relative path once wrote files somewhere unexpected).

### 4.2 Channel meanings and units
| Channel (table) | Unit / meaning |
|---|---|
| `GPS Time` | session seconds, 100 Hz master clock (`mElapsedTime`) |
| `GPS Latitude/Longitude` (native and recorder v2+) | 10 Hz → backend uses `pos_hz=10`; files without GPS tables (recorder v1) fall back to `Pos X/Z` @100 Hz |
| `Ground Speed` | km/h (fallback: GPS Speed in m/s ×3.6) |
| `Yaw Rate` | rad/s (recorder only; **absent in native files**) |
| `Slip Angle` | rad per wheel (recorder only); backend converts to deg for display/tyre model |
| `Wheel Speed` | recorder: rad/s; native: ≈ m/s. Backend heuristic: `max>50` ⇒ rad/s, `R_TYRE=0.33 m`. Slip ratio = (ws−v)/v where v>2 m/s |
| `Susp Pos` | metres of deflection (load-transfer proxy); axle balance = front − rear mean; rake makes it negative |
| `G Force Lat / Long / Vert` | g. **Long is positive under braking**; the frontend negates it for display (traction-circle "brake" is up) |
| `TyresPressure` kPa, `TyresRubberTemp` °C (mean surface) | recorder tables named exactly this (older code used non-existent `Tire Pressure/Temp`) |
| `ABS`, `TC` | sparse (ts,value) step events, forward-filled onto the grid via `_event_active_col` |
| `Lap` | (ts, lap_number) at each crossing; `Lap Time` | (ts, duration of the lap that just ended) |
| `metadata` (key,value; has TrackName, car, setup JSON), `channelsList`, `eventsList` | needed for the external viewer |

### 4.3 Track map, GPS, mirroring
- Projection (`main.py ~l.365-383`): `pos_x = (lon−c_lon)·R·cos(c_lat)`, `pos_z = −(lat−c_lat)·R`, `R = 111 320 m/deg`. **Only latitude is negated.** History: user saw a left-turning map vs a right-turn in telemetry ("Road Atlanta is the other way"). The first "fix" negated both axes — verified identical curvature sign to negating neither (a 180° rotation). Single-axis flip reproduces Road Atlanta T1 as a right-hander.
- `heading = atan2(dz, dx)` in **canvas** angles (0 = right, clockwise positive since canvas y is down). Smoothing half-window `W_HEAD = max(2, round(0.2·pos_hz))` (was fixed 6, which at 10 Hz was 1.2 s each side).
- Track edges come from LMU scoring (`Path Lateral`, `Track Edge`); corner labels from `track_db.py` matched by `Lap Dist`.
- Lap selection: auto-picks the fastest complete lap if `lap_number` omitted. Laps are cut by `Lap` event timestamps against `GPS Time`. Partial last lap shows (e.g. lap 4, 16 s).

### 4.4 Tyre model / grip math (`tire_model.py`, `grip_profile.py`)
**Tyre model v3 (2026-10-19; replaces the v2 histogram model described in older notes).**
- Data: each lap is reduced to per-axle binned statistics (`tire_bins.LapBins`: slip, load ratio, temperature, pressure, wear, camber cells; proxy mu = lat G / load ratio, plus the game's true mu where the car records tyre forces). Files are loaded at `tire_model.TIRE_GRID_HZ` = 50 Hz for the tyre steps; the coaching report stays at 20 Hz.
- Profile (`grip_profile.py`, `TIRE_MODEL_VERSION = 3`): keyed by car **model** (`car_identity.identify_car`, e.g. `BMW M4 LMGT3.json`), not livery. `tire_files["<file>"]` holds game version and track once per file and the per-lap `LapBins`; `tire_fit_cache` caches the fit (sha1 fingerprint of lap keys, measure, prior). v1/v2 profiles lose their tyre data on load (baselines kept) and are rebuilt by `backend/tools/ingest_sessions.py`.
- Fit (`tire_mf.py`, `tire_class.py`): factorised Magic Formula, only the load factor enabled (`ENABLED_FACTORS`; the others did not generalise on held-out data); bootstrap over laps gives `peak_slip_ci90` and a confidence based on the interval, not the sample count. Newest game version with ≥ 3 laps is the fit set. A class prior (`CLASS_PRIOR_WEIGHT` 0.3) is used only for cars with < 3 laps. No fit → `binned_v2_fallback` (confidence ≤ 0.3, no interval).
- Output per axle (`build_axle_tire_model`): all v2 keys (`peak`, `efficient_range`, `peak_observed`, `peak_grip`, `available_grip`, `max_observed_slip_deg`, `confidence`, `sample_count`, `grip_measure`, `tyre_temperature`) plus `model`, `peak_slip_ci90`, `peak_grip_ci90`, `factors`, `factor_effects`, `prior_used`, `version_note`, `coverage`. `efficient_range` is clipped to the slip angles the driver reached. `peak_status` is `observed` | `plateau_unconfirmed` (fitted peak inside the driven range, no fall-off: peak = the fit, GU not an upper bound) | `rising_at_edge` (peak = largest slip driven, a lower bound; GU is an upper bound). GU also works from the binned fallback peak, in which case CE uses the v2 envelope (`ce_method` `v2_envelope`). CE is the lap p90 mu divided by the state-aware peak mu (capped at the largest slip driven), so it is not comparable with v2-era CE values. The older-version prior branch in `_resolve_prior` is dormant by design (PRIOR_MAX_LAPS == MIN_VERSION_LAPS).
- **GU** = p95(|slip|)/fitted peak (≈1.5 cap). **CE** (front) = the lap's p90 mu ÷ the fitted peak mu at each sample's own load (`ce_method` "state_aware_mf", else "v2_envelope"). Both are `None` until a fit exists.
- Scorecard (DevNotes 10/19/2026): held-out R2 of the Magic Formula beats the slip-binned curve on every axle that could be scored; weak spots are Porsche rear, Ferrari rear (true mu) and Cadillac rear (peak at the 15° bound, never reached). Largest refit 4.4 s (Porsche, 2.3 MB profile).
- Traction circle in the UI plots `G Force Lat` vs `Long` with a moving dot and gradient tail.
- Assumptions to keep in mind: lat G / load ratio is a proxy for tyre mu (reads ~5-15 % high against the game's mu on Hypercars); GT3 cars have no Tire Load and use Susp Force; slip angle comes from ground-velocity ratio (noisy at low speed).

### 4.5 Body slip β (new this session; `vehicle_dynamics.compute_body_slip_deg`)
- β = ψ − course, ψ = ∫ yaw rate (trapezoid, real dt), course = unwrapped velocity heading from position.
- Sign and offset of ψ are unknown → try ±1, offset = circular mean of (course − sgn·ψ) on straights (|yaw|<0.03 rad/s, v>50 km/h); pick sign with the smaller median |residual|.
- **Refuses (returns `[]`)** if <40 straight samples, <100 fast samples, yaw std <1e-3, median |residual|>4°, or >25 % of fast samples >20°. β=0 below 25 km/h; 3-point moving average; output degrees, positive = body rotated clockwise on screen relative to velocity.
- Validation: real lap 2 — straights median |β| 0.14°, β follows turn direction; regression against recorded tyre slip angles R² 0.95 rear / 0.97 front; synthetic car with known β (±6°) RMS error 0.09°, max 0.21°, corr 1.0 in both mirrored-frame signs; refusal cases (no yaw, all-zero, noise, no straights) all return `[]`.
- Real laps (Road Atlanta recorder file): lap 0 max 3.9°, lap 1 max 175.8° (a real spin), lap 2 max 11.8°, lap 3 max 4.5°, lap 4 `[]`. Native files `[]`.
- Limits: it is an *estimate*; 10 Hz position limits accuracy at low speed/kinks.

### 4.6 DuckDB schema
Recorder file tables: `metadata`, `channelsList`, `eventsList`, `GPS Time`, `Pos X`, `Pos Z`, `Ground Speed`, `Yaw Rate`, `Slip Angle` (value1..4), `Wheel Speed`, `Susp Pos`, `G Force Lat/Long/Vert`, `Throttle Pos`, `Brake Pos`, `Steering Pos`, `Lap Dist`, `Path Lateral`, `Track Edge`, `TyresPressure`, `TyresRubberTemp`, `Gear`, `In Pits`, `ABS`, `TC`, `Lap` (ts,value), `Lap Time` (ts,value). Scalar tables are `(value)` per sample with frequency in `channelsList`; wheel tables `(value1..value4)` = FL, FR, RL, RR; event tables `(ts, value)`.
Known zeroed/unpopulated: Tire Load, Tire Lateral Force, Tire Longitudinal Force, Grip Fraction (see 4.1). Native files lack Slip Angle, Yaw Rate and those four.

---

## 5. Current state

### Works (verified by the user or by tests in chat)
- Recorder v4 → full-session DuckDB; readable by backend and the external viewer; saving works after the CSV/COPY + stop-file fix (**user: "IT WORKS NOW!"**, after remembering to `docker compose down` between versions).
- Lap selector chips with lap times; trackmap orientation fixed; persistent chevron when paused; 8 Hz filter; dynamic ranges; Ground Speed and ABS/TC graphs; balance fix; HUD/fonts/floating dockable telemetry panel; zoom bar; grip utilization + traction circle + tyre model; Settings bubble with JSON import/export + LLM auto-start; full UI redesign (user: "This works!").
- Body slip + degree conversion + new chevron (backend verified through the real `/trackmap` endpoint on laps 0–4 and a native file; page renders with zero JS errors; screenshots viewed for ×5 and true scale).

### Tyre model v3 (branch `tyre-model-v3`, 2026-10-19)
- Implemented, covered by the backend suite and re-ingested into `data/grip_profiles` of the worktree (models: Valkyrie, Cadillac, Ferrari, Genesis, BMW, Porsche, Mustang, Lexus, Mercedes; old livery files moved to `data/grip_profiles/backups/`). Real `/analyze` smoke test on the Fuji WRT file with a stub LLM; **not tried in the browser or with the real local LLM**.
- To use it: merge, copy the re-ingested profiles over `data/grip_profiles`, `docker compose down; docker compose up -d --build`.
- Open: Cadillac rear / Ferrari rear / Porsche rear peaks and held-out R2 are weak (see DevNotes scorecard); the coach prompt for the new fields has not been judged against a real model answer.

### Half-done / not verified
- **Not user-tested yet:** `lmu-coach-bodyslip-v1` (β chevron, `slip_delta_deg`, `slip-gain` select). User must rebuild: `docker compose down; docker compose up -d --build`.
- jsdom/axe regression suites were **not** re-run after the chevron change (the old `ui_test.js` / `ui_test2.js` lived in `/tmp` and are gone). Previous axe result: 0 violations.
- Recorder auto-stop on leaving a session / auto-start on joining is implemented via `ConnectionState` + the Record tab, but was not re-verified in-game in this chat beyond saving working.
- Screenshots D (×10) and E (toolbar) from the earlier attempt were never viewed; ×10 was rendered in the later pass but not inspected.
- Unanswered older questions: Unsloth quants, and a Windows 11 + 7900 XTX ROCm install/migrate/remove-Ollama guide (user later said llama.cpp is "Tested", so low priority — ask before doing).

### Known bugs / quirks (exact symptoms)
1. **Lap Time event off-by-one in `telemetry._read_lap_time_events`** (cross-check only, cosmetic): backend log on the recorder file shows
   `[telemetry] WARNING: lap 0 duration mismatch — diffed=122.600s vs Lap Time event=0.000s (tolerance 0.25s). Possible out-of-order lap or mid-lap session reset.` then `lap 1 … diffed=98.200s vs Lap Time event=122.600s`, `lap 2 … diffed=84.000s vs … event=98.200s`, `lap 4 … diffed=16.400s vs … event=83.800s`. Each event holds the duration of the lap that *just ended* but is keyed to the *next* lap row. Displayed lap times (diffed) are correct (L0 2:02.600, L1 1:38.200, L2 1:24.000, L3 1:23.800, L4 0:16.400).
2. Body slip is `[]` for native LMU files, partial laps and laps whose straights/yaw don't calibrate — UI intentionally falls back to heading-only chevron (no white velocity line, no β label).
3. Config default port mismatch: `lmu-coach.config*.json` use `http://127.0.0.1:8001/health` / `…:8001/v1` but the user's compose points at `:8000`. Not a code bug; see §6.
4. Historical errors already fixed (don't regress): `NameError: os` after the atomic-write change; `IndexError` in `_slip_ratio_col` (3924 vs 39239); 12 KB `.tmp` with a `DeprecationWarning` flood after "Stop recording"; `"No complete laps in this file"` before lap handling was fixed.
5. No automated tests existed at handoff. Since 2026-10-12: `recorder/test_recorder_hf.py` (unittest) and `backend/tests/` (pytest; run with the `uv run --no-project --with ...` command in `test_slip_angle.py`'s docstring). Everything older is still untested.

---

## 6. Prioritized next steps and open questions

1. **Have the user rebuild and eyeball the β chevron** on real laps in-game data (esp. a lap with sustained oversteer). Tune `SD_THRESH=0.12°`, colour scale `/1.7°`, default gain ×5 if needed.
2. **Add a real test suite** (pytest for `compute_body_slip_deg` synthetic car + refusal cases, `_lowpass_filter`, projection/heading, lap slicing; Playwright/jsdom smoke test + axe for the UI). Synthetic-car recipe: 100 Hz, 55 m/s, left arc/right arc with β up to ±6°, 10 Hz position, test both yaw sign conventions.
3. Fix `_read_lap_time_events` keying (event i belongs to the lap *ending* at crossing i, i.e. lap number of the previous Lap row; first event may be 0.000).
4. Reconcile ports/defaults: pick one llama-server port (user's actual: 8000) and update `lmu-coach.config*.json`, `docker-compose.yml` fallback, and the launcher health URL together. **Ask the user which is canonical.**
5. In-game re-verification of recorder session exit/join auto-stop and the Record tab log stream.
6. Optional: persist β exaggeration per user config; pass `body_slip_deg` stats (e.g. per-corner peak β, rotation vs. apex) into the grip report for the LLM.
7. Native-file support for β is impossible without yaw rate (could approximate from lateral G/speed, but that approach was previously rejected as unusable — do not silently reintroduce).
8. Ask the user whether they still want the Unsloth answer / ROCm migration guide.

Open questions: canonical llama-server port; whether Qwen3.8 model path in `lmu-coach.config.json` is final; whether distribution to other users is imminent (affects packaging, `.env` handling, README).

---

## 7. User preferences and corrections

- Windows 11 + PowerShell; **PowerShell has no `&&`** (use `;`/separate lines). Uses `uv` for the recorder venv. Runs Docker Compose himself ("the user does their own docker compose rebuilds"). Reminder worth giving after backend changes: `docker compose down` then `up -d --build` (stale container once made a new version look broken).
- Wants **zipped deliverables of only changed files, laid out to unzip over the project**, with a short explanation of *what* changed and *where*. For small fixes he sometimes says "I can do it myself" — then just show where/how to edit.
- Wants honesty about verification: say what was tested vs assumed, and flag estimated vs measured quantities.
- Retain information/readouts/graphs/functions during visual redesigns; only layout/visuals change.
- Don't rewrite things that work: "The original implementation of recorder.py … worked just fine" — extend, don't replace.
- Corrections he made: the **"mobile + dockable frame" = telemetry panel (ground speed, yaw, balance, stats)**, while the **"50 % bigger" frame = brake/throttle/steering HUD**; fonts for readouts 20 % bigger; zoom +/- buttons were too far from the bar; chevron must persist when paused; map mirrored/Road Atlanta direction; lap times rounded to 0.1 s; OS/US readout was 100 % OS on a file where it shouldn't be (he asked where US/Neu/OS values come from — they come from the front-minus-rear mean slip-angle delta); "Just respond got it" for FYI messages (keep replies minimal when told).
- Likes step-by-step guides for infra work (Ollama→llama.cpp style).
- Commit trailers (when committing): `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_019PwkBbwnBPirt9qzDWznVm`.
- Style seen in code: heavily commented "BUG FIX / PREVIOUSLY … Fix:" blocks explaining *why*; keep that habit for non-obvious changes.

---

## 8. Odd-but-intentional things in the code

- `_lowpass_filter(..., sample_hz=100.0)` is hard-coded to 100 even though `full_len` may be 10 Hz: it runs on the **raw 100 Hz** arrays *before* interpolation/slicing. Don't "fix" it to `pos_hz`.
- `yaw_raw_full` (unfiltered) feeds `compute_body_slip_deg` (needs the integral); `yaw` (filtered, lap-sliced) is only used for straight detection.
- `W_HEAD` scales with `pos_hz` (≈0.2 s each side) so heading smoothing is time-based.
- Heading uses screen-Z (already negated) → canvas angles; don't negate again (double negation = rotation, not mirror).
- Frontend negates `G Force Long` for display; backend leaves it positive-under-braking.
- `0.12°` / `1.7°` thresholds for OS/US colour replaced `0.002 rad` / `0.03 rad` — same numbers in degrees (0.002 rad = 0.115°, 0.03 rad = 1.72°).
- Slip-gain exaggeration applies only to the **drawn** chevron; the label always shows the true β. Cap ±60° is only a drawing guard (lap 1's spin reaches ±176° in the data).
- GU uses the 95th percentile on purpose (not max). The 1.5 cap is for display sanity.
- Grip profile file names contain spaces/underscores from the car name (`Akkodis ASP Team 2026 _78_WEC.json`).
- `recorder.py` has `--selftest` and `_probe_methods()` that time DuckDB insert methods — kept so a slow-insert regression on the user's PC can be re-measured.
- `.stop_requested`, `.<name>.duckdb.tmp`, `.<name>.spool(.part)` are dotfiles in the output dir by design; they are recovered at startup, never shown in the UI list.
- Canvas drawing colours are literals in JS and must be kept in sync with CSS tokens in `:root` (see `DESIGN.md`).
- `lmu-coach.config.json` in the repo contains the user's real launcher path and the Qwen3.8 model path (not the example file).
- Backend filter/slice rule: *full-length first, slice last* — any new per-sample channel must be interpolated with `_interp_to_length(raw, full_len)` before `[s_idx:e_idx+1]`.

---

## 9. Files created / changed in the chat (one line each)

Backend (`backend/`)
- `app/main.py` — FastAPI app; `/trackmap` builds projection, heading, driven-line channels, filters, balance, slip ratio, ABS/TC, `slip_delta_deg`, `body_slip_deg`.
- `app/telemetry.py` — loads laps from DuckDB (recorder + native), lap boundaries, channel lists (fixed pressure/temp table names, added `Susp Pos`).
- `app/vehicle_dynamics.py` — slip/yaw summaries; new `compute_body_slip_deg`.
- `app/tire_model.py` — probabilistic per-axle slip-angle model, temp/pressure/load relationships, grip utilization + cornering efficiency.
- `app/grip_profile.py` — per-car JSON profile accumulation incl. tyre model.
- `app/grip_analysis.py` — ABS/TC events, wheel slip, combined G, understeer signatures, report builder.
- `app/coach.py` — LLM client (OpenAI-compatible; llama.cpp), system prompt, follow-up messages, `LLMError/LLMUnavailable`.
- `app/track_db.py` — track/layout registry + corner reference points.
- `app/static/index.html` — whole UI (redesigned): map, HUD, traction circle, graphs, Coach/Record/Settings, lap chips, β chevron, slip-gain selector.
- `app/static/fonts/*` + `LICENSES.txt` — bundled Fira Sans/Code (OFL).
- `requirements.txt`, `Dockerfile` — python:3.12-slim, uvicorn on 8000.

Recorder (`recorder/`)
- `recorder.py` — v4 recorder: full-session DuckDB, mElapsedTime clock, CSV+COPY writes, .tmp/.spool recovery, stop-file, selftest.
- `recorder_control.py` — host agent (port 8420): recorder subprocess, SSE logs, config, path checks, LLM health/auto-start, GuardMiddleware.
- `START_RECORDER.ps1` — `uv sync; uv run recorder.py` (original standalone launch).
- `START_RECORDER_CONTROL.ps1` — launches the host agent.
- `recorder_hf.py` — (added 2026-10-12 in Claude Code) experimental clone of `recorder.py`: 1000 Hz polling, one sample per game tick, `--probe`, extra tyre-model channels, official lap times. Not wired into the host agent. Not yet run in-game.
- `START_RECORDER_HF.ps1`, `test_recorder_hf.py` — its launcher and tests (`uv run python -m unittest test_recorder_hf`).
- `pyproject.toml`, `uv.lock`, `pyLMUSharedMemory/*` — deps and vendored shared-memory bindings (base upload; not edited by us).

Root
- `docker-compose.yml` — backend service, mounts, LLM env fallbacks.
- `.env` / `.env.example` — `LMU_TELEMETRY_PATH` etc.
- `lmu-coach.config.json` / `lmu-coach.config.example.json` — stack config (user's real vs template).
- `DESIGN.md` — design tokens/layout/accessibility reference for the redesign.
- `DevNotes.md` — user's own dev log (not ours; contains the 9/18–9/19 notes on zeroed tables and the viewer error).
- `data/grip_profiles/*.json` — accumulated profiles (Akkodis ASP #78, Proton #88).
- `docs/HANDOFF.md` — this file.

Delivered zips in the chat (date order, layer with `unzip -o`): `lmu-coach-updated`, `recorder-fixed`, `backend-fixed`, `fixes-round3`, `lowpass-filter-fix`, `dynamic-range-fix`, `speed-abs-tc-graphs`, `balance-fix-and-ui-changes`, `tire-model-grip-utilization-traction-circle`, `recorder-overhaul`, `recorder-save-fix`, `-v2`, `-v3`, `recorder-py-v3.1`, `lmu-coach-settings-v1`, `lmu-coach-redesign-v1`, `lmu-coach-bodyslip-v1`. The repo snapshot delivered with this handoff already has all of them applied.

Test data used (not in repo): recorder file `Michelin_Raceway_Road_Atlanta-Akkodis_ASP_Team_2026__78_WEC-2026-09-26T06_11_13Z.duckdb` (laps 0–4; lap 1 contains a spin; lap 2 used for validation) and native LMU file `Michelin_Raceway_Road_Atlanta_P_2026-09-26T06_18_07Z.duckdb`.

---

## 10. Quick start for the new agent

```powershell
# repo root
copy .env.example .env        # set LMU_TELEMETRY_PATH to the recorder output dir or LMU's Telemetry folder
docker compose down; docker compose up -d --build   # UI at http://localhost:8080
cd recorder; .\START_RECORDER_CONTROL.ps1           # host agent on :8420 (Record tab, LLM auto-start)
# or standalone: .\START_RECORDER.ps1
```

Backend smoke test without Docker (Linux/macOS dev): `pip install fastapi duckdb numpy scipy httpx uvicorn pydantic`, then monkeypatch `app.main.OWN_SESSIONS_DIR` to a folder of `.duckdb` files and use `fastapi.testclient.TestClient` against `/sessions/sessions/<file>/trackmap?lap_number=2`.
