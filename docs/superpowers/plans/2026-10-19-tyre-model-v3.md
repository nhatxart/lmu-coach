# Tyre Model v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v2 binned slip-angle curve with a factorized, class-pooled Magic Formula tyre model fitted on per-lap binned statistics, with bootstrap confidence, production-wired into `/analyze`.

**Architecture:** Per-lap steady-state samples are reduced to sparse binned sufficient statistics (`tire_bins.py`), fitted with a factorized Magic Formula (`tire_mf.py`, scipy least_squares) using a class/older-version prior (`tire_class.py`). `grip_profile.py` stores bins + a cached fit per car model; `tire_model.py` turns the fit into the existing v2 output shape plus new fields.

**Tech Stack:** Python 3.12, numpy 2.1.3, scipy 1.14.1, duckdb 1.1.3, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-19-tyre-model-v3.md`

## Global Constraints

- No new entries in `backend/requirements.txt`. Backend runs Python 3.12 in Docker; recorder Python 3.13.
- Never write to `data/sessions/*.duckdb` (open read-only). Real recordings live at `C:\Users\hoang\Projects\lmu-coach\data\sessions` (the worktree has no `data/sessions`); tools take `--sessions-dir`.
- PowerShell: chain with `;`, never `&&`.
- Backend tests: from the worktree root run
  `uv run --no-project --with pytest --with fastapi --with duckdb==1.1.3 --with numpy==2.1.3 --with scipy==1.14.1 --with httpx --with pydantic python -m pytest backend/tests -q`
  Recorder tests: `cd recorder; uv run python -m unittest test_recorder_hf`
- Every v2 key in `build_axle_tire_model()` output and `compute_grip_utilization_for_lap()` output keeps its name and meaning.
- Comment style: explain WHY for non-obvious choices ("PREVIOUSLY ... Fix:" blocks), as in the existing files.
- Commits: `git -c user.name=hoang -c user.email=hoangnhatvn2002@gmail.com commit ...` with trailers `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_019PwkBbwnBPirt9qzDWznVm`.
- Log findings in `DevNotes.md` under `### 10/19/2026` (append bullets; heading already exists).

## Review Focus

- A car with no load source (Porsche #91 Barcelona): MF must fit on `lateral_g` only, never mix measures; output says so. -> Task 5 test `test_fit_measure_never_mixed`.
- Native LMU files / laps without `Slip Angle`: zero bins, no exception anywhere in ingest or `/analyze`. -> Task 9 test `test_native_lap_adds_nothing`.
- A profile whose data spans one temperature/pressure (cold Fuji): those factors reported `"not_identifiable"` and fixed neutral; prediction outside the data range does not explode. -> Task 5 test `test_unidentifiable_factor_fixed_neutral`.
- Loading a v2 (or v1) profile JSON: no crash; tyre data reset; ABS/TC baselines kept. -> Task 9 test `test_v2_profile_loads_and_resets_tyre_data`.
- Degenerate data (all |slip| < 2 deg, or < MIN cells): fit returns None -> axle uses `binned_v2_fallback`, `/analyze` still answers. -> Task 6 test `test_degenerate_falls_back`.

---

### Task 1: Recorder metadata (class, game version, temps)

**Files:**
- Modify: `recorder/recorder_hf.py` (session meta in `add_sample` ~l.469-477, metadata table ~l.657-670)
- Test: `recorder/test_recorder_hf.py`

**Interfaces:**
- Produces: metadata keys `VehicleClass` (str, from the player's scoring `mVehicleClass`, decoded), `GameVersion` (str of `generic.gameVersion`), `TrackTempC`, `AmbientTempC` (str, 1 decimal, from `scoring.scoringInfo.mTrackTemp/mAmbientTemp` at first sample). Missing source -> key written with value `""`.

- [ ] **Step 1:** Write `test_metadata_has_class_version_and_temps`: drive a fake session through the existing fakes, flush, open the DuckDB, assert `metadata` has the 4 keys with the fake values (e.g. `VehicleClass == "LMGT3"`, `GameVersion == "1234"`, `TrackTempC == "31.5"`). Extend the fakes with `generic.gameVersion` and scoring temps.
- [ ] **Step 2:** Run recorder tests; expect the new test to FAIL.
- [ ] **Step 3:** Capture the values where the buffer gets its car/track names (the same place `player_scor` is available), store on the buffer, write them in the metadata `executemany`. Use the existing `_safe_decode`.
- [ ] **Step 4:** Run recorder tests; all pass (22 existing + new).
- [ ] **Step 5:** Commit `recorder_hf: record vehicle class, game version, track/ambient temp`.

### Task 2: Car identity + new telemetry channels

**Files:**
- Create: `backend/app/car_identity.py`
- Modify: `backend/app/telemetry.py` (`WHEEL_CHANNELS` l.65-77, wheel resampling)
- Test: `backend/tests/test_car_identity.py`, `backend/tests/test_tire_model.py` (channel test)

**Interfaces:**
- Produces: `@dataclass(frozen=True) class CarIdentity: model: str; car_class: str; game_version: str`
- Produces: `identify_car(meta: dict[str, str], car_name: str) -> CarIdentity`
  - model: `meta["VehicleModel"]` if non-empty, else `LIVERY_TABLE` lookup by `#<number>` + team substring of `car_name`, else `car_name`.
  - car_class: `meta["VehicleClass"]` if non-empty; else `"LMGT3"` if model ends with `"LMGT3"`; else LIVERY_TABLE class; else `"unknown"`. Normalise "Hypercar"/"HYPERCAR"/"LMH"/"LMDh" -> `"Hypercar"`, "GT3"/"LMGT3" -> `"LMGT3"`.
  - game_version: `meta.get("GameVersion") or "unknown"`.
- `LIVERY_TABLE` (team substring, number) -> (model, class), exact values:
  `("Team WRT", "69") -> ("BMW M4 LMGT3", "LMGT3")`, `("Manthey", "91") -> ("Porsche 911 GT3 R LMGT3", "LMGT3")`, `("Manthey", "92") -> ("Porsche 911 GT3 R LMGT3", "LMGT3")`, `("Proton", "88") -> ("Ford Mustang LMGT3", "LMGT3")`, `("Akkodis", "78") -> ("Lexus RC F LMGT3", "LMGT3")`, `("Iron Lynx", "61") -> ("Mercedes-AMG LMGT3", "LMGT3")`, `("Heart of Racing", "23") -> ("Aston Martin Vantage LMGT3", "LMGT3")`, `("Aston Martin THOR", "007") -> ("Aston Martin Valkyrie", "Hypercar")`, `("Cadillac", "12") -> ("Cadillac V-Series.R", "Hypercar")`, `("Ferrari AF Corse", "50") -> ("Ferrari 499P", "Hypercar")`, `("Genesis", "17") -> ("Genesis GMR-001", "Hypercar")`, `("Toyota", "7") -> ("Toyota TR010", "Hypercar")`.
  Number match is exact on the digits after `#` (so `#7` does not match `#007`).
- Produces: `WHEEL_CHANNELS` gains `"Camber"` (rad), `"TyresWear"` (fraction), `"TyresCarcassTemp"` (C), `"Surface Type"` (code), `"Tire Lateral Force"` (N). `"Surface Type"` is resampled with previous-sample hold (categorical), never linear interpolation; values stay integers.

- [ ] **Step 1:** Tests: `test_identify_from_vehicle_model` (meta VehicleModel "BMW M4 LMGT3" -> class LMGT3, version "unknown"), `test_identify_from_livery_table` (car_name "Aston Martin THOR Team 2026 #007:WEC", meta {} -> Valkyrie/Hypercar), `test_number_match_exact` ("Toyota Racing 2026 #7:WEC" -> TR010, not Valkyrie), `test_unknown_car` -> model == car_name, class "unknown", `test_vehicle_class_meta_wins_and_normalises` (meta VehicleClass "HYPERCAR" -> "Hypercar"), `test_surface_type_not_interpolated` (synthetic channel 0,0,5,5 at 10 Hz onto a 100 Hz grid -> only values {0,5}).
- [ ] **Step 2:** Run; FAIL.
- [ ] **Step 3:** Implement `car_identity.py`; add channels; add categorical hold for `Surface Type` in the resampling path.
- [ ] **Step 4:** Run full backend suite; all pass.
- [ ] **Step 5:** Commit `Car identity (model/class/version) and HF tyre channels in telemetry`.

### Task 3: v3 per-sample extraction

**Files:**
- Modify: `backend/app/tire_model.py` (`_lap_axle_samples`, `_OBS_FIELDS` users)
- Test: `backend/tests/test_tire_model.py`

**Interfaces:**
- Produces: `_lap_axle_samples(lap, calibration=None, load_ref=None)` per-axle dict gains keys (numpy arrays, NaN where unknown): `"carcass_delta_c"` (axle mean carcass temp minus optimal), `"wear"` (axle mean of `TyresWear`), `"camber_deg"` (axle mean of |Camber| in degrees), `"long_g"`, `"speed_kmh"`, `"true_mu"` (sum of |Tire Lateral Force| over the axle's two wheels / sum of their Tire Load; NaN where Tire Load sum < 30 % of the lap median axle load, or channels absent). `"mask"` additionally requires both wheels' Surface Type in {0, 1} (dry, wet) when the channel exists.
- Existing keys and `extract_axle_observations_from_lap` behaviour unchanged (v2 path stays working until Task 9).

- [ ] **Step 1:** Tests with the existing synthetic `_lap()` helper extended: `test_kerb_samples_masked` (Surface Type 5 on FL for 50 samples -> those samples False in front mask, rear unaffected), `test_true_mu_from_forces` (Fy 6000/6000 N, Fz 4000/4000 N -> true_mu 1.5), `test_true_mu_nan_without_forces`, `test_new_state_fields_present` (camber -0.05 rad both wheels -> camber_deg ≈ 2.865; wear 0.9 -> 0.9).
- [ ] **Step 2:** Run; FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Full backend suite passes.
- [ ] **Step 5:** Commit `Tyre samples: carcass temp, wear, camber, true mu, kerb filter`.

### Task 4: Per-lap binned sufficient statistics

**Files:**
- Create: `backend/app/tire_bins.py`
- Test: `backend/tests/test_tire_bins.py`

**Interfaces:**
- Consumes: Task 3 per-axle sample dict.
- Produces:
  - Constants: `STATE_VARS = ("load_ratio", "temp_delta_c", "pressure_kpa", "wear", "camber_deg", "carcass_delta_c", "long_g")`; bin widths `BIN = {"slip": 0.4, "load_ratio": 0.1, "temp_delta_c": 5.0, "pressure_kpa": 5.0, "wear": 0.05, "camber_deg": 0.5}`; slip range 0-15 deg. Cell key = tuple of bin indices for slip + the first 5 STATE_VARS (carcass and long_g are carried, not binned); NaN state -> index `-999`.
  - `@dataclass class Cell`: `n: int`, `sum_mu`, `sumsq_mu`, `sum_slip`, `n_true`, `sum_true`, `sumsq_true`, and per STATE_VAR `sum_<var>` + `n_<var>` (finite count).
  - `@dataclass class LapBins`: `measure: str` (`"mu_load_normalized"` or `"lateral_g"`), `cells: dict[tuple, Cell]`; `to_dict() -> dict` (list of rows, JSON-safe), `from_dict(d) -> LapBins`, `merge(others: list[LapBins]) -> LapBins` (staticmethod; raises `ValueError` on mixed measures), `n_samples() -> int`.
  - `bin_axle_samples(s: dict, mask: np.ndarray) -> LapBins`: grip = |lat G| / max(load_ratio, 0.2) where load_ratio finite (measure mu_load_normalized) else |lat G| (lateral_g). Mixed availability inside one lap: if >= 50 % of masked samples have load_ratio, keep only those (mu); else use lateral_g for all.
  - `cell_arrays(bins: LapBins) -> dict[str, np.ndarray]`: per-cell `n`, `mu_mean`, `mu_var`, `slip`, each state var mean (NaN when `n_<var>`==0), `true_n`, `true_mean`.
  - `coverage(bins: LapBins) -> dict`: counts by (load_ratio bin, temp_delta bin) for cells with slip >= 1.5 deg, plus the list of slip bins reached with >= 20 samples.

- [ ] **Step 1:** Tests: `test_round_trip_preserves_cells`, `test_merge_sums_counts` (two laps, same cell, n adds, means weighted), `test_merge_rejects_mixed_measures`, `test_nan_state_goes_to_unknown_bin`, `test_cell_means_are_state_means_not_centres` (load_ratio 1.03 and 1.07 in one cell -> mean 1.05), `test_true_mu_accumulated_only_where_finite`, `test_json_size_reasonable` (one synthetic 10k-sample lap serialises < 200 kB).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (vectorised: `np.unique` on stacked index rows + `np.add.at`). **Step 4:** Suite passes.
- [ ] **Step 5:** Commit `Per-lap binned tyre statistics (tire_bins)`.

### Task 5: Factorized Magic Formula fit

**Files:**
- Create: `backend/app/tire_mf.py`
- Test: `backend/tests/test_tire_mf.py` (+ synthetic generator in `backend/tests/synthetic_tyre.py`)

**Interfaces:**
- Consumes: `cell_arrays` (Task 4).
- Produces:
  - `ALL_FACTORS = ("load", "temp", "pressure", "wear", "camber")`; `ENABLED_FACTORS: tuple[str, ...] = ALL_FACTORS` (Task 7 may narrow it).
  - Model (state relative to the fit's reference state `s0`):
    `D = D0 * (1 + kDL*(L-L0)) * T(td) * (1 + kDP*dp) * (1 - kDW*(w-w0)) * (1 + kDC*(c-c0))`,
    `T(td) = exp(-((td-T0)/wl)^2)` for td<T0 else `exp(-((td-T0)/wh)^2)`,
    `B = B0 * (1 + kBL*(L-L0)) * (1 + kBP*dp)`; `C`, `E` shared;
    `dp` = pressure minus the count-weighted linear fit of pressure on temp_delta (pressure residual given temperature). The fit's coefficients are stored in `params` as `pt_slope`, `pt_intercept` so `predict()` can compute `dp` for any new state (raw `pressure_kpa` + `temp_delta_c` in, never `dp`).
  - Bounds: `D0 (0.1, 5)`, `B0 (0.05, 3)` per deg, `C (1.0, 2.0)`, `E (-3, 1)`, `kDL (-1, 0.5)`, `kBL (-1, 1)`, `T0 (-30, 30)`, `wl, wh (5, 80)`, `kDP, kBP (-0.05, 0.05)` per kPa, `kDW (0, 2)`, `kDC (-0.2, 0.2)` per deg.
  - Identifiability gate (count-weighted, cells with slip >= 1.5 deg): load needs std(L) >= 0.05; temp needs p95-p05 of td >= 10 C; pressure needs p95-p05 of dp >= 5 kPa; wear needs p95-p05 >= 0.02; camber needs p95-p05 >= 0.3 deg. Failing factors are fixed neutral (k=0; temp: T(td)=1) and listed in `factors_status[name] = "not_identifiable"`; others `"fitted"`, factors not in ENABLED_FACTORS `"disabled"`.
  - `@dataclass class MFFit`: `params: dict[str, float]`, `ref_state: dict[str, float]`, `factors_status: dict[str, str]`, `measure: str`, `n_cells: int`, `n_samples: int`, `rmse: float`, `peak_slip_deg: float`, `peak_mu: float`, `peak_observed: bool`, `max_observed_slip_deg: float`; `to_dict()/from_dict()`.
  - `fit_mf(bins: LapBins, target: str = "mu", prior: MFFit | None = None, prior_weight: float = 0.0, factors: tuple[str, ...] | None = None) -> MFFit | None` — `target="true"` fits `true_mean` weighted by `true_n`. Weighted residuals `sqrt(n)*(mu_mean - pred)`, `loss="soft_l1"`, `f_scale` = 0.1. Prior adds residuals `sqrt(prior_weight * n_total / n_params) * (p - p_prior) / scale_p` (scale_p = parameter bound width / 10). Returns None if < 30 cells with n >= 5, or slip span (max well-sampled slip) < 2.5 deg, or optimizer status <= 0.
  - `predict(fit: MFFit, slip_deg: np.ndarray, state: dict[str, np.ndarray | float] | None = None) -> np.ndarray` (state defaults to ref_state).
  - `peak_at(fit, state=None) -> tuple[float, float]` (slip_deg, mu) by grid search 0-15 deg step 0.01.
  - `ref_state` = count-weighted medians over cells with slip >= 1.5 deg. `peak_observed` = some bin with >= 20 samples exists at slip >= peak_slip + 0.4 and its mean is <= 0.97 * peak_mu at ref state (use model-normalised: observed mean / predict at that cell's state <= 0.97 * peak).

- [ ] **Step 1:** `synthetic_tyre.py`: `make_bins(n_laps=6, true_params=..., factors=..., noise=0.03, seed=0) -> list[LapBins]` drawing slip 0-7 deg, load 0.9-1.6, temp_delta -20..+10, pressure tied to temp + independent ±6 kPa, from a known MF.
- [ ] **Step 2:** Tests: `test_recovers_known_params` (D0 within 3 %, peak slip within 0.2 deg), `test_unidentifiable_factor_fixed_neutral` (all temp_delta = -30 ± 2 -> temp "not_identifiable", predict at td=+40 equals predict at ref), `test_fit_measure_never_mixed` (lateral_g bins -> fit.measure "lateral_g"), `test_returns_none_on_few_cells`, `test_prior_pulls_toward_prior` (1 lap data + prior with peak 5 deg, weight 10 -> peak closer to 5 than weight 0), `test_true_target` (bins with true mu = 0.9 * proxy -> fit_mf(target="true").peak_mu ≈ 0.9 * proxy fit), `test_round_trip`.
- [ ] **Step 3:** FAIL. **Step 4:** Implement. **Step 5:** Suite passes.
- [ ] **Step 6:** Commit `Factorized Magic Formula fit (tire_mf)`.

### Task 6: Bootstrap confidence + runtime budget

**Files:**
- Modify: `backend/app/tire_mf.py`
- Test: `backend/tests/test_tire_mf.py`

**Interfaces:**
- Produces: `bootstrap_fit(lap_bins: list[LapBins], n_boot: int = 50, seed: int = 0, **fit_kw) -> dict` returning `{"fit": MFFit | None, "peak_slip_ci90": [lo, hi] | None, "peak_mu_ci90": [lo, hi] | None, "n_boot_ok": int, "confidence": float}`. Resamples LAPS with replacement, warm-starts each resample from the full fit's params. Single-lap input: resample cells instead and cap confidence at 0.3.
- `confidence = (1 - exp(-n_laps / 8)) * clip(1 - (hi - lo) / peak_slip, 0, 1) * (1.0 if peak_observed else 0.6) * (n_boot_ok / n_boot)`, rounded to 3, capped at 0.95; 0.0 when fit is None.

- [ ] **Step 1:** Tests: `test_ci_contains_truth` (synthetic 8 laps), `test_confidence_lower_when_peak_unobserved` (same data truncated at slip 3 deg vs full), `test_confidence_capped`, `test_degenerate_falls_back` (all slip < 2 -> fit None, confidence 0.0, no exception), `test_runtime_budget` (25 synthetic laps × ~400 cells, both axles sequentially, n_boot=50 -> < 20 s on CI; print the time).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Suite passes.
- [ ] **Step 5:** Commit `Bootstrap-over-laps confidence for the MF fit`.

### Task 7: Hypercar factor experiment (tool + real-data run)

**Files:**
- Create: `backend/tools/mf_experiment.py`, `backend/tools/MF_EXPERIMENT.ps1`
- Modify: `backend/tools/README.md`, `DevNotes.md`

**Interfaces:**
- Consumes: telemetry, car_identity, `_lap_axle_samples`, `file_load_reference`, `bin_axle_samples`, `fit_mf`, `predict`.
- Tool: `python -I backend/tools/mf_experiment.py --sessions-dir <dir> [--class Hypercar] [--files GLOB]`. Reads files read-only, groups laps by car model (class filter), bins per lap (same lap selection as `ingest_tire_laps`: 115 % of reference lap). For each car with true mu: leave-one-FILE-out (leave-one-LAP-out when the car has one file); for each model variant compute held-out weighted R² and RMSE on cornering cells (slip >= 1.5 deg) against TRUE mu:
  `binned_v2` (v2 `_fit_curve`-style bin means as predictor), `mf_slip_only` (factors=()), `+load`, `+load+temp`, `+load+temp+pressure`, `+...+wear`, `+...+camber`, and `mf_all_on_proxy` (fit on proxy mu, scored on true mu).
- Prints a markdown table per car and axle.

- [ ] **Step 1:** Implement the tool (unit test: `backend/tests/test_mf_experiment.py::test_runs_on_synthetic_dir` builds a tiny synthetic DuckDB with the recorder schema via `synthetic_session.py` helpers if available, otherwise test the scoring function `heldout_r2(fit, bins)` on synthetic bins).
- [ ] **Step 2:** Run on real data: `--sessions-dir C:\Users\hoang\Projects\lmu-coach\data\sessions --class Hypercar`. Paste the tables into the report file and into `DevNotes.md` under 10/19.
- [ ] **Step 3:** Commit `MF factor experiment tool + Hypercar results`.
- [ ] **PM gate (controller):** set `ENABLED_FACTORS` from the results: keep a factor if adding it raises held-out R² by >= 0.01 on at least half of the car-axle rows where it was identifiable, else disable. Ledger the ruling; the change goes into Task 8's dispatch.

### Task 8: Class pooling + GT3 collapse tool

**Files:**
- Create: `backend/app/tire_class.py`, `backend/tools/class_collapse.py`, `backend/tools/CLASS_COLLAPSE.ps1`
- Modify: `backend/app/tire_mf.py` (`ENABLED_FACTORS` per PM gate), `backend/tools/README.md`, `DevNotes.md`
- Test: `backend/tests/test_tire_class.py`

**Interfaces:**
- Produces: `CLASS_PRIOR_WEIGHT: float` (default 1.0; tool picks), `fit_class_prior(car_bins: dict[str, list[LapBins]], measure: str) -> MFFit | None` — pooled fit over all cars of a class (each car's laps merged, cars weighted equally by scaling counts to the median car's n), `fit_car_with_prior(lap_bins: list[LapBins], prior: MFFit | None, weight: float = CLASS_PRIOR_WEIGHT, n_boot: int = 50) -> dict` (same shape as `bootstrap_fit`, plus `"prior_used": str` one of `"class"`, `"older_version"`, `"none"`).
- `class_collapse.py --sessions-dir <dir> --class LMGT3`: per car: standalone MF peak slip / peak mu at the class reference state, offset vs class prior; held-out (leave-one-lap-out) R² on proxy mu for weights {0, 0.3, 1, 3, 10}; prints table and the weight with the best mean held-out R².

- [ ] **Step 1:** Tests: `test_class_prior_between_cars` (two synthetic cars with D0 1.4 and 1.6 -> prior D0 in (1.4, 1.6)), `test_equal_car_weighting` (car A 10× more samples does not dominate: prior D0 within 0.05 of midpoint), `test_small_car_borrows_from_prior` (1 lap car, prior from 3 others: held-out error lower with weight 1 than 0), `test_prior_used_label`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Suite passes.
- [ ] **Step 5:** Run tool on real data `--class LMGT3` and `--class Hypercar`; record tables in report + DevNotes; note that GT3 cars were recorded on different tracks (track confounds car).
- [ ] **Step 6:** Commit `Class-pooled MF prior + collapse tool`.
- [ ] **PM gate (controller):** set `CLASS_PRIOR_WEIGHT` to the tool's best weight (ledger ruling); carried into Task 9.

### Task 9: Profile v3 (storage, versioning, cache)

**Files:**
- Modify: `backend/app/grip_profile.py`, `backend/app/tire_model.py` (remove raw-obs storage use from the profile path), `backend/tests/test_tire_model.py`, `backend/tests/test_corners_and_ref_lap.py` (monkeypatch targets if renamed)
- Test: `backend/tests/test_grip_profile_v3.py`

**Interfaces:**
- `TIRE_MODEL_VERSION = 3`. Profile file key = `identify_car(...).model` (`load_profile(model)`, `_profile_path` unchanged sanitising). New fields: `car_class: str`, `tire_lap_bins: dict[str, dict]` (`"<file>#<lap>" -> {"game_version", "track", "load_source", "front": LapBins.to_dict(), "rear": ...}`), `tire_fit_cache: dict` (`{"fingerprint": str, "front": {...bootstrap result as dict...}, "rear": {...}}`). `front_axle_obs/rear_axle_obs` are dropped from v3 JSON. `tire_model_laps` and `tire_model_load_sources` kept (derived from `tire_lap_bins`).
- `ingest_tire_laps(laps, file_key) -> list[int]` same contract; bins per lap via Task 3/4; then `refit_tyre_model(class_prior: MFFit | None = None)`.
- Version selection in `refit_tyre_model`: newest `game_version` (lexicographic numeric-aware; "unknown" oldest) with >= 3 laps is the fit set; laps of older versions form a prior fit (`prior_used="older_version"`, weight `CLASS_PRIOR_WEIGHT/2`) when they have >= 3 laps, else class prior, else none. Fewer than 3 laps in every version -> fit on all laps, `version_note="mixed_versions"`.
- Fingerprint = sha1 of sorted lap keys in the fit set + prior fit params rounded 4 dp; cache reused when equal.
- `class_prior_for(car_class: str, exclude_model: str) -> MFFit | None` in grip_profile: loads every profile JSON in PROFILES_DIR with the same `car_class` (excluding this model), `fit_class_prior` on their bins. Cached per process by (class, set of (model, fingerprint)).
- `from_dict`: version != 3 -> drop all tyre fields (obs, bins, laps, sources, cache), keep baselines; `car_class` default `"unknown"`.

- [ ] **Step 1:** Tests: `test_v2_profile_loads_and_resets_tyre_data`, `test_ingest_dedups_and_bins` (2 laps -> 2 keys; re-ingest adds none), `test_native_lap_adds_nothing` (lap without Slip Angle -> no key, no exception), `test_cache_reused_when_unchanged` (monkeypatch `bootstrap_fit` counter: second `compute_baselines` call does not refit), `test_newest_version_selected_with_older_prior`, `test_mixed_versions_note`, `test_profile_keyed_by_model` (two liveries of "Aston Martin Valkyrie" land in one file).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Full suite passes (update v2-specific tests that assert raw obs storage; keep their intent).
- [ ] **Step 5:** Commit `Grip profile v3: per-lap bins, model key, versioned cached MF fit`.

### Task 10: v3 outputs, /analyze, coach prompt

**Files:**
- Modify: `backend/app/tire_model.py` (`build_axle_tire_model`, `compute_grip_utilization_for_lap`, `build_temp_pressure_load_relationships`), `backend/app/grip_profile.py` (`compute_baselines`), `backend/app/main.py` (~l.279-311), `backend/app/coach.py` (tyre-model paragraph ~l.93-114), module docstring "Model v3" section
- Test: `backend/tests/test_tire_model.py`, `backend/tests/test_coach_prompt.py`

**Interfaces:**
- `build_axle_tire_model(cached: dict | None, lap_bins: list[LapBins]) -> dict` returns all v2 keys (`peak`, `efficient_range` = slip range where predict >= 0.95*peak_mu at ref state, `peak_observed`, `peak_grip` = peak_mu, `available_grip` = predict at peak at the p90 load/temp state of the data, `max_observed_slip_deg`, `confidence`, `sample_count`, `grip_measure`, `tyre_temperature`) plus `model` (`"magic_formula_v3"` | `"binned_v2_fallback"`), `peak_slip_ci90`, `peak_grip_ci90`, `factors` (factors_status), `factor_effects` (per fitted factor: mu at peak for 3 representative values, e.g. temp_delta -20/0/+10 C), `prior_used`, `version_note`, `game_version`, `coverage` (from Task 4 `coverage` of merged bins: under-covered = (load, temp) cells with < 50 cornering samples among those within the data's p05-p95 box).
- `compute_grip_utilization_for_lap(lap, front_model, rear_model, calibration=None, load_ref=None, front_fit: MFFit | None = None)`: GU unchanged; when `front_fit` given, CE = lap p90 grip / mean over the lap's cornering samples of `peak_at(front_fit, sample_state)[1]` (state-aware ceiling); else v2 CE. Adds `"ce_method": "state_aware_mf" | "v2_envelope"`.
- `build_temp_pressure_load_relationships` replaced in baselines by `factor_effects`; keep the key `front_temp_pressure_load` / `rear_temp_pressure_load` populated with `{"available": True, "factor_effects": ..., "note": ...}` so the prompt key still exists.
- `main.py`: identity via `identify_car(lap.meta, lap.car_name)`; `load_profile(identity.model)`; pre-update scoring uses the cached fit (no refit before scoring). Profile save stays after the LLM answer.
- `coach.py`: replace the tyre-model paragraph: explain model (MF, factors, CI, prior_used, coverage); confidence < 0.5 = tentative; peak_slip_ci90 width; factor_effects are model estimates within the observed range only; never present them as setup targets; coverage tells the driver what data would firm the model up (one sentence max, under Confidence).

- [ ] **Step 1:** Tests: `test_v3_output_has_all_v2_keys`, `test_fallback_when_no_fit` (`model == "binned_v2_fallback"`), `test_ce_state_aware_used_with_fit`, `test_prompt_mentions_magic_formula_and_ci` (SYSTEM_PROMPT contains "peak_slip_ci90" and "factor_effects"), `/analyze` smoke via TestClient with a monkeypatched LLM on a synthetic session (existing pattern in tests) returns 200 and `grip_utilization` present.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Full suite passes.
- [ ] **Step 5:** Commit `Tyre model v3 outputs wired into /analyze and the coach prompt`.

### Task 11: Tools, re-ingest, scorecard, docs

**Files:**
- Modify: `backend/tools/ingest_sessions.py` (profile key = model via identify_car; `--migrate` merges ABS/TC baselines of livery-named profiles into the model profile and moves old livery files into the backup dir), `backend/tools/validate_tire_model.py` (adds MF rows: held-out R² of MF vs v2 binned on proxy and, where available, true mu; confidence and CI per axle), `backend/tools/README.md`, `DevNotes.md`, `docs/HANDOFF.md` (§4.4 tyre model, §5 state), `data/grip_profiles/*.json` (worktree copies)

- [ ] **Step 1:** Implement tool changes; unit test `test_migrate_merges_baselines` (two livery profiles -> one model profile with concatenated capped baselines, `lap_count` summed).
- [ ] **Step 2:** Dry run, then real run against the worktree profiles: `python -I backend/tools/ingest_sessions.py --sessions-dir C:\Users\hoang\Projects\lmu-coach\data\sessions --profiles-dir data/grip_profiles --month 2026-10 --tyre-only --migrate --save` (backups automatic). Also September files: `--month 2026-09 --tyre-only --save`.
- [ ] **Step 3:** Run `validate_tire_model.py` for WRT, Cadillac, Aston, Ferrari, Genesis, Porsche; record the v3 scorecard table (v2 vs v3 held-out R², CI, confidence) in DevNotes 10/19. Measure refit time of the largest profile (budget 10 s).
- [ ] **Step 4:** Full backend suite + recorder suite pass; `docker compose build` succeeds (if Docker is available; otherwise note it).
- [ ] **Step 5:** Commit `Tyre model v3: tools, re-ingested profiles, scorecard, docs`.
