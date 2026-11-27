# Ideal driver inputs, phase 1-lite: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ideal lap's speed profile on your own line into ideal brake %, throttle % and a steering reference per point, with per-corner input lines ("Brake 8 m later…"). It is returned as an `inputs` block of `GET /sessions/{source}/{file}/ideal-lap`.

**Architecture:** A new flat module `backend/app/ideal_inputs.py` (pure numpy, no FastAPI) fits three inverse maps per recording (brake, throttle, steering) from the file's qualifying laps. It applies them to the ideal's state on the 2 m line grid, checks that they reproduce your lap, and extracts per-corner events. `ideal_lap.analyse_lap` calls it after the solve, with no extra solve. The UI is NOT in this plan: spec §8 step 9 is gated on I1-I5 passing on the real files.

**Tech Stack:** Python 3.12 (Docker) / numpy, pytest. Tests run with the `uv run --no-project --with pytest --with fastapi --with duckdb --with numpy …` command in `backend/tests/test_slip_angle.py`'s docstring, from `backend/`.

**Spec:** `docs/superpowers/specs/2026-11-24-ideal-inputs-1lite-design.md` (user answers 11/24: Q1 steering = reference only plus a corrections flag; Q2 traction-limited = 100 % with a TC band; Q4 one pedal per metre; Q6 thresholds kept). Background: `docs/superpowers/specs/2026-11-10-ideal-lap-design.md`. Reference probes (read-only, the numbers to match): `docs/superpowers/spikes/ideal-lap/inp_p1_maps.py`, `inp_p2_thr_steer.py`, `inp_p3_delta.py`.

## Global Constraints

- Signals are on the 50 Hz grid (`ideal_state.GRID_HZ`), smoothed 0.2 s (`SMOOTH_N = 10`). Tyre-side longitudinal `gxt = gx + grade + car.drag(v)`, engine-side `gxa = gx + grade`.
- Maps are fitted per FILE on its qualifying laps (`ideal_state.qualifying_laps`). With fewer than 3 qualifying laps, the maps are pooled over the fit group's files and flagged `scope: "group"`.
- Never an exception and never a number from a failed map. Each refusal is per input: that input is `null` with a reason string, and the other inputs are still returned.
- Steering is a reference trace only, with no steering advice (Q1). Traction-bound points get 100 % throttle with `tc: true` (Q2). There is one pedal per point and no overlap (Q4).
- The per-corner line thresholds (§4) are exactly: brake point 4 m, peak brake 5 % pts, release 6 m, throttle pick-up / full 0.10 s or 5 m, corner throttle amount 15 % pts, corrections flagged in the worst 20 % of your laps.
- Reproduce limits (§3.1): brake ≤ 3.5 % pts, throttle ≤ 15 % pts (drive points), steering ≤ 10 % of lock (car class `Hypercar`) / ≤ 13 % (every other class).
- Read-only on `data/sessions/*.duckdb`. Tool caches go to `%TEMP%`, never under `data/`.
- Code habits: "PREVIOUSLY … Fix:" comments for non-obvious changes, PowerShell `;` not `&&`. Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A spin or off on the analysed lap.** `/ideal-lap` accepts any complete lap, not only qualifying ones. Expected: the steering reproduce check fails and steering is null with a reason; brake and throttle are unaffected; no 500. Test in Task 7.
2. **Steering Pos scaled ±100 instead of ±1** (the synthetic ring writes % form; older files may too). Expected: `steer_rms_pct_lock` and the % of lock numbers are unchanged, because lock is the file's own p99.5. Test in Task 4.
3. **A corner window across the start/finish line** (`s0 > s1` on Lap Dist). Expected: events are found on both sides of the line, with no negative or lap-length deltas. Test in Task 6.
4. **A GT3 where the ideal needs more decel than the pedal map gives at the cap** (ABS-limited). Expected: the pedal equals the cap, `abs_ideal` is true there, and the brake point still comes out. Test in Task 5.
5. **A corner the other laps of the file never cover with steering data** (no samples in the window). Expected: `corrections` is `{you: x, laps_median: null, flagged: false}`, never NaN in the JSON. Test in Task 6.

---

### Task 1: The reduction keeps steering, yaw, signed axle slip and Lap Dist

**Files:**
- Modify: `backend/app/ideal_state.py` (`State`, `lap_raw`, `lap_state`, `REDUCTION_VERSION`)
- Modify: `backend/tests/synthetic_ideal.py` (`_telemetry` returns `steer=None`)
- Test: `backend/tests/test_ideal_inputs_state.py`

**Interfaces:**
- Produces: `State.steer: ndarray | None` (Steering Pos, unsmoothed, the recorded unit), `State.yaw: ndarray | None` (rad/s, smoothed 0.2 s), `State.alpha: ndarray | None` (shape (2, n): signed mean slip angle of the front and rear axle in rad, smoothed 0.2 s), `State.ld: ndarray | None` (`ideal_solver.unwrap_lap_dist` of the raw Lap Dist). `lap_raw(...)["steer"]`. `REDUCTION_VERSION = 3`.

- [ ] **Step 1: Write the failing tests**

```python
def test_lap_state_carries_the_input_channels():
    raw = <lap_raw-shaped dict with slip rows (+0.02, +0.04, -0.01, -0.03) rad, yaw, steer, lapdist wrapping at 1000>
    st = S.lap_state(raw)
    assert np.allclose(st.alpha[:, 50], [0.03, -0.02], atol=1e-9)      # signed, rad, axle means
    assert np.array_equal(st.steer, raw["steer"])                        # unsmoothed, like thr/brk
    assert st.yaw is not None and (np.diff(st.ld) >= -1e-9).all()        # unwrapped
    assert (st.slip >= 0).all()                                          # the old |deg| field is unchanged

def test_missing_channels_are_none_and_survive_the_npz(tmp_path):
    # raw without yaw / slip / steer / lapdist -> all four None; _save_npz/_load_npz round trip keeps arrays and Nones
def test_reduction_version_is_3():
    assert S.REDUCTION_VERSION == 3
```

- [ ] **Step 2: Run them and check that they fail**: `… pytest tests/test_ideal_inputs_state.py -v` should FAIL (no `alpha` field).
- [ ] **Step 3: Implement.** Add `steer=_one(lap, "Steering Pos", None)` to `lap_raw`. Add the four `State` fields with default `None`. Build them in `lap_state` (alpha from the raw signed `slip` rows: `(slip[0]+slip[1])/2`, `(slip[2]+slip[3])/2`, each smoothed `SMOOTH_N`). Bump `REDUCTION_VERSION` to 3 with a comment (`# 3: + steer, yaw, signed axle slip, Lap Dist (ideal inputs)`). `_load_npz` already maps missing keys to None. Check that no existing caller passes `State` positionally beyond `gf`.
- [ ] **Step 4: Run the new test and the whole `tests/test_ideal_*.py` suite**: everything should PASS.
- [ ] **Step 5: Commit** `Ideal inputs: the reduction keeps steering, yaw, signed slip, Lap Dist`.

### Task 2: Brake map, and synthetic telemetry with known input maps

**Files:**
- Create: `backend/app/ideal_inputs.py`
- Modify: `backend/tests/synthetic_ideal.py` (add `TRUE_INPUTS`, `with_inputs`)
- Test: `backend/tests/test_ideal_inputs_maps.py`

**Interfaces:**
- Consumes: `State` (Task 1), `CarModel.drag(v)`, `CarModel.power(v, gy)`.
- Produces:
  - `@dataclass BrakeMap(a, e0, e1, r2, rms_pct, n, abs_share: list[float] (5 bins 0-20…80-100 %), cap, scope)` with `to_dict()` / `from_dict()`, `decel(brk, v) -> ndarray` and `pedal(neg_gxt, v) -> ndarray` (unclipped).
  - `fit_brake_map(states: list[State], car, scope: str = "file") -> BrakeMap | str` (a str is the refusal reason).
  - `synthetic_ideal.TRUE_INPUTS: dict` and `with_inputs(raw: dict, car, rng, noise: bool = True) -> dict`. The returned raw has `brk`, `thr`, `steer`, `yaw` and signed `slip` from the true maps.

- [ ] **Step 1: Write the synthetic truth.** `TRUE_INPUTS = {"brake": {"a": 1.80, "e1": 0.0015, "e0": 0.10}, "coast": [−0.05, 0.0, −4.0e-5, −0.02] (over [1, v, v², gy²]), "table": [0.05, 0.15, 0.26, 0.37, 0.48, 0.58, 0.68, 0.77, 0.86, 1.0], "steer": {"a": 9.0, "b": −2.8, "c0": 0.0}, "us": [0.012, 0.004]}`. `with_inputs` derives the mode per sample as spec §2.4, sets `brk = 100·(−gxt − e1 v − e0)/a` clipped 0..100 when braking, `thr = F⁻¹((gxa − coast)/(power − coast))` when driving (100 where the binding code is power/traction), and 0 otherwise. It sets `yaw = G·gy/v`, front slip `= +US(gy)/2`, rear `= −US(gy)/2` (all four wheels set), and `steer = a·yaw/v + b·US(gy) + c0`. With `noise`, it adds N(0, 3 % pts) to pedals (clipped) and N(0, 0.02) to steer. Docstring: these laps are for the input maps only. Their slip is not the knee model's, so they are not for axle fits.
- [ ] **Step 2: Write the failing tests**

```python
def test_T12_brake_map_recovers_the_truth():        # 6 perfect laps (true_car("B"), line()), with_inputs, noise
    bm = I.fit_brake_map(states, car)
    for k in ("a", "e0", "e1"):
        assert getattr(bm, k) == pytest.approx(TRUE_INPUTS["brake"][k], rel=0.02)
    assert bm.r2 > 0.95 and bm.cap == pytest.approx(100, abs=1) and len(bm.abs_share) == 5

def test_brake_map_refuses_without_braking():       # brk all 0 -> str containing "braking samples"
def test_brake_map_round_trips_to_dict():
```

- [ ] **Step 3: Run them and check that they fail** (no module).
- [ ] **Step 4: Implement `fit_brake_map`.** Use spec §1.1/§2.1 least squares on the 0.2 s smoothed pedal, mask `brk > 3, thr < 5, abs < 0.5, |gy| < 0.4`, lag 0, and refuse with fewer than 200 samples (`"fewer than 200 braking samples"`). The served coefficients are fitted on ALL laps. `r2` / `rms_pct` are held out: fit on the even-index laps, score on the odd ones, `None` with one lap (**plan's call**: the spec stores the held-out score but does not say which coefficients are served). `cap` = p99 of the pedal where `brk > 3`. `abs_share` = share of `abs > 0.5` per 20 % pedal bin on `brk > 3, thr < 5`. Module constants: `BRAKE_MIN_N = 200`, `INPUTS_VERSION = 1`.
- [ ] **Step 5: Run the tests**: they should PASS.
- [ ] **Step 6: Commit** `Ideal inputs: brake map (T12 brake)`.

### Task 3: Throttle map

**Files:** Modify `backend/app/ideal_inputs.py`. Test `backend/tests/test_ideal_inputs_maps.py`.

**Interfaces:**
- Produces: `@dataclass ThrottleMap(table: list[float] (10, at pedal centres 5…95), coast: list[float] (4, over [1, v, v², gy²]), lag_s, r2, rms_pct_part, n_bins, scope)` with `to_dict` / `from_dict`. Methods: `coast_g(v, gy)`, `fraction(gxa, v, gy, pw)` (= `(gxa − coast)/max(pw − coast, 0.05)`), `pedal(frac) -> %` (inverse, 100 at or above `table[-1]`), `fraction_of(pedal)`. `fit_throttle_map(states, car, scope="file") -> ThrottleMap | str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_T12_throttle_table_recovers_the_truth():
    tm = I.fit_throttle_map(states, car)
    assert np.allclose(tm.table, TRUE_INPUTS["table"], atol=0.03)
    assert all(np.diff(tm.table) > 0)                    # strictly increasing: the inverse is defined
    assert 0.0 <= tm.lag_s <= 0.10

def test_throttle_inverse_round_trips():                 # pedal(fraction_of(p)) == p for p in 5..95
def test_throttle_map_refuses_with_too_few_bins():       # thr only 0 / 100 -> str containing "populated bins"
```

- [ ] **Step 2: Run them and check that they fail.**
- [ ] **Step 3: Implement** as spec §1.2/§2.2. Samples: `brk < 1, tc < 0.05, v > 8`. Coast is least squares on `thr < 2`. The table holds the median of the fraction per 10 % bin with `thr > 2` and at least 20 samples per bin. Refuse with fewer than 5 populated bins (`"throttle table has fewer than 5 populated bins"`); empty bins are interpolated, and the table is made monotone with `np.maximum.accumulate` plus `1e-4·arange` so it is strictly increasing. Search the lag over 0…5 samples (0…0.10 s) on the even laps by gxa R2. The engine curve is `car.power(v, gy)` with no `p_off` (the fit laps are not the line). Held-out like Task 2.
- [ ] **Step 4: Run the tests**: they should PASS.
- [ ] **Step 5: Commit** `Ideal inputs: throttle map (T12 throttle)`.

### Task 4: Steering map

**Files:** Modify `backend/app/ideal_inputs.py`. Test `backend/tests/test_ideal_inputs_maps.py`.

**Interfaces:**
- Produces: `@dataclass SteerMap(a, b, c0, us: list[float] (2), lock_p995, lag_s, r2_measured, r2_us, n, scope)` with `to_dict` / `from_dict`. Methods: `us_rad(gy)` (= `u1 gy + u2 gy|gy|`) and `steer(k, gy)` (= `a k + b us_rad(gy) + c0`). `fit_steer_map(states, scope="file") -> SteerMap | str`. Constant `STEER_R2_MIN = 0.95`.

- [ ] **Step 1: Write the failing tests**

```python
def test_T12_steer_map_recovers_the_truth():
    sm = I.fit_steer_map(states)
    assert sm.a == pytest.approx(9.0, rel=0.03) and sm.b == pytest.approx(-2.8, rel=0.03)
    assert sm.r2_measured >= 0.95

def test_steer_map_refuses_without_yaw_or_slip():       # yaw None -> str containing "Yaw Rate"
def test_steer_map_refuses_below_r2_095():               # steer = pure noise -> str containing "R2"
def test_steering_in_percent_units_gives_the_same_percent_of_lock():   # Review Focus 2
    # states with steer x 100: a, b, c0, lock scale by 100; (steer(k, gy) / lock) unchanged to 1e-9
```

- [ ] **Step 2: Run them and check that they fail.**
- [ ] **Step 3: Implement** as spec §1.3/§2.3. Samples: `v > 12, |gy| > 0.05`. Use `k = yaw / max(v, 5)` and `da = alpha[0] − alpha[1]`, with steer smoothed 0.2 s. Search the lag over 0…15 samples (0…0.3 s) on the even laps. `r2_measured` = held-out R2 of `a k + b da + c0`. The US map is least squares of `da` on `[gy, gy|gy|]`, and `r2_us` = held-out R2 of `steer(k, gy)` refitted with the US map (the served `a, b, c0` are the US-map refit, as in `inp_p2` line 109). `lock_p995` = p99.5 of |steer| over all samples. Refuse when `r2_measured < 0.95` (`"steering map R2 %.3f < 0.95"`).
- [ ] **Step 4: Run the tests**: they should PASS.
- [ ] **Step 5: Commit** `Ideal inputs: steering map (T12 steering)`.

### Task 5: Inputs on the line, reproduce your lap (T13, T14)

**Files:** Modify `backend/app/ideal_inputs.py`. Test `backend/tests/test_ideal_inputs_line.py`.

**Interfaces:**
- Consumes: Tasks 2-4 maps. `ln` from `ideal_solver.line_of` (keys `s, ks, v, grade, thr, brk, tc, ds`, optionally `p_off`) and the `raw` of the same lap (`ideal_state.lap_raw`).
- Produces:
  - `@dataclass InputMaps(brake: BrakeMap | None, throttle: ThrottleMap | None, steer: SteerMap | None, refused: dict[str, str])`.
  - `driven_on_line(ln, raw) -> dict` with `brk_ref, thr_ref, steer_ref, k_line` on the grid: your inputs smoothed 0.2 s at 50 Hz, then put on the grid by `raw["s"]`; `k_line` = smoothed yaw / max(v, 5).
  - `inputs_on_line(car, ln, v, gxt, code, maps, ref, absolute=False) -> dict` with `brk, thr, steer, mode (−1/0/1), abs, tc` on the grid. `v`/`gxt` are the profile (the ideal's, or `ln["v"]` and its kinematic gxt). `code` is the binding code per point (`object` array).
  - `reproduce(car, ln, gxt_ref, maps, ref, car_class) -> dict` with `{brake_rms, throttle_rms, steer_rms_pct_lock, pass: {brake, throttle, steer}}`.
  - Constants `LIMITS = {"brake": 3.5, "throttle": 15.0, "steer_hypercar": 10.0, "steer_other": 13.0}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_T13_reproduce_on_perfect_laps():         # maps fitted on 6 noisy laps; one lap's own line
    r = I.reproduce(car, ln, gxt_ref, maps, ref, "LMGT3")
    assert r["brake_rms"] <= 1.5 and r["throttle_rms"] <= 4.0 and r["steer_rms_pct_lock"] <= 3.0
    assert all(r["pass"].values())

def test_T14_ideal_equals_truth():
    # true car solved on line() without noise -> TRUE-map pedals (with_inputs, noise=False) vs
    # inputs_on_line(fitted maps, sol v, sol gxt, sol binding)
    assert rms(brk_fit - brk_true) <= 3.0 and rms_drive(thr_fit - thr_true) <= 6.0
    assert all(abs(d) <= 4.0 for d in brake_point_deltas_m)          # first brk >= 10 % per corner
    assert all(abs(d) <= 0.10 for d in throttle_pickup_deltas_s)     # first thr >= 20 % after vmin

def test_power_and_traction_codes_are_flat():      # code "traction" -> thr 100, tc True; "power" -> 100, tc False
def test_driven_points_return_your_pedal():        # code "driven"/"measured" with v = ln["v"] -> thr == thr_ref
def test_abs_cap():                                # Review Focus 4: -gxt beyond pedal(cap) -> brk == cap, abs True
def test_one_pedal_per_point():                    # never brk > 0 and thr > 0 at one point
```

- [ ] **Step 2: Run them and check that they fail.**
- [ ] **Step 3: Implement** spec §2.1-§2.4:
  - The mode comes from `gxt` and the maps.
  - Brake is absolute, clipped to the cap; `abs` is true where the unclipped pedal is above the cap.
  - Throttle depends on the code. `power` gives 100; `traction` gives 100 with `tc`. Otherwise use the delta form `thr_ref + [F⁻¹(state) − F⁻¹(state_ref)]` (F⁻¹ in % pedal) where the driven point is in drive mode, else the absolute `F⁻¹(state)`, clipped 0..100.
  - The engine curve on the line is `car.power(v, gy) + ln.get("p_off", 0)`; `gy = ks v²/G`.
  - Steering is the absolute `steer(k_line, gy)`.
  - Lags: the value at s is the map at `s + lag_s·v(s)` (closed lap: wrap with `np.interp(..., period=L)`).
  - Pedal smoothing is 0.2 s in time at the profile's speed: resample to 50 Hz along `t = cumsum(ds/v)`, `smooth(…, SMOOTH_N)`, back.
  - `absolute=True` skips the delta and the code rules: that is `reproduce`'s mode, and with `p_off` a flat-out point inverts to 100 by itself.
  - The reproduce RMS is taken over the whole lap for brake, over driven drive points (`thr_ref > 2, brk_ref < 1`) for throttle, and over `v > 12` in % of `lock_p995` for steering.
- [ ] **Step 4: Run the tests**: they should PASS.
- [ ] **Step 5: Commit** `Ideal inputs: inputs on the line, reproduce (T13, T14)`.

### Task 6: Per-corner events, lines and steering corrections

**Files:** Modify `backend/app/ideal_inputs.py`. Test `backend/tests/test_ideal_inputs_corners.py`.

**Interfaces:**
- Consumes: grid traces from Task 5, `ln["lap_dist_mono"]`, lap length `L`, and the corner rows of `analyse_lap` (`kind == "corner"`, keys `n, name, s0, s1, gain_s, where`). `ideal_lap._window_mask(ld, s0, s1, L)` is passed in as `mask_fn` (no import cycle).
- Produces:
  - `corrections_rms(steer, ld, s0, s1, L, lock, mask_fn) -> float | None` (one lap at 50 Hz: RMS of steer smoothed 0.2 s minus steer smoothed 1.0 s inside the window, in % of `lock`; None with fewer than 10 samples in the window).
  - `corner_events(ld, L, corners, you: dict, ideal: dict, v_ref, v_i, corr: dict[n -> (you, others: list[float])], mask_fn) -> list[dict]`, rows shaped as spec §5 `corners[]`.
  - Constants `TH = {"brake_point_m": 4, "brake_peak_pct": 5, "release_m": 6, "throttle_s": 0.10, "throttle_m": 5, "corner_throttle_pct": 15, "corr_quantile": 0.8}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_brake_8_m_later_gives_the_line():       # synthetic corner: ideal brake starts 8 m after yours
    row = I.corner_events(...)[0]
    assert row["brake_point_m"]["delta"] == pytest.approx(8, abs=0.5)
    assert row["lines"][0].startswith("Brake 8 m later")
def test_3_m_gives_no_line():                     # below the 4 m threshold -> no brake text
def test_throttle_line_in_seconds():              # pick-up 0.2 s earlier -> "Throttle 0.2 s earlier"
def test_at_most_two_lines_largest_first():
def test_corner_across_the_line():                # Review Focus 3: s0 = L - 60, s1 = 80; deltas finite, |delta| < 50 m
def test_corrections_without_other_laps():        # Review Focus 5: others == [] -> laps_median None, flagged False
def test_corrections_flagged_in_worst_20pct():    # you above the 0.8 quantile of 5 laps -> flagged and the corrections line
```

- [ ] **Step 2: Run them and check that they fail.**
- [ ] **Step 3: Implement** spec §4. Events per corner, for both you and the ideal:
  - brake point = first brk ≥ 10 % in the window;
  - peak brake;
  - release = last brk ≥ 5 % before the window's v minimum;
  - throttle on = first thr ≥ 20 % after the minimum;
  - full throttle = first thr ≥ 95 %.

  `delta_s` = `delta_m / v_i` at the event. The exact text, filled in:
  - `"Brake {d:.0f} m {later|earlier}, {p:.0f} % {less|more} pressure, release {r:.0f} m {later|earlier}"` (only the clauses above their thresholds);
  - `"Throttle {t:.1f} s {earlier|later}, flat {m:.0f} m {earlier|later}"`;
  - `"Hold more speed at the apex: less brake, earlier part throttle"` (the ideal's mode at its v minimum is drive and yours is brake);
  - `"More steering corrections than on your other laps here"`.

  Keep at most two lines, ordered by the corner's `where` (entry text before exit text when `where == "entry"`, the other way round for `"exit"`). A missing event is `None`, never NaN.
- [ ] **Step 4: Run the tests**: they should PASS.
- [ ] **Step 5: Commit** `Ideal inputs: per-corner events and lines`.

### Task 7: Wire into the ideal lap and the endpoint

**Files:**
- Modify: `backend/app/ideal_inputs.py` (`fit_input_maps`, `inputs_block`), `backend/app/ideal_lap.py` (`analyse_lap`, `build_ideal_lap`, `_refusal`, cache key), `backend/app/ideal_store.py` (`IdealStore.group_reductions`)
- Test: `backend/tests/test_ideal_inputs_endpoint.py`; extend `backend/tests/test_ideal_endpoint.py` (`TOP` gains `"inputs"`)

**Interfaces:**
- Produces:
  - `fit_input_maps(own: FileReduction, group: list[FileReduction], car) -> InputMaps` (file scope with ≥ 3 qualifying laps, else pooled over `group` with scope `"group"`; cached in a module `LRU(32)` keyed on `(file, mtime, size, INPUTS_VERSION, scope files)`).
  - `inputs_block(car, ln, raw, v_i, gxt_i, code, maps, corners, L, car_class, points_at, nearest, mask_fn, others) -> dict` returns exactly the spec §5 `inputs` schema. `status` is `ok` (all three traces), `partial` (some), or `insufficient_data` (neither brake nor throttle). The `note` is the sentence below.
  - `IdealStore.group_reductions(path) -> list[FileReduction]` (the `group_for(path).paths` reductions; `[]` on a Refusal).
  - `analyse_lap(..., input_maps: InputMaps | None = None, input_others: list[State] | None = None)` adds `res["inputs"]` (`None` when `input_maps` is None).
  - `build_ideal_lap` fits the maps with `store._reduction(path)` and `getattr(store, "group_reductions", lambda p: [])(path)` and adds `INPUTS_VERSION` to the `_CACHE` key; `_refusal` sets `"inputs": None`.
  - Note text: `"Ideal inputs on your own line: the brake and throttle that would drive the ideal lap's speed, from maps fitted on your laps of this recording; steering is a reference only."`

- [ ] **Step 1: Write the failing tests**

```python
def test_inputs_block_schema_and_lengths(session, monkeypatch):    # FakeStore whose _reduction returns a real reduction of the ring
    d = _get(session, "ideal-lap", lap=1)["inputs"]
    assert set(d) == {"status", "reason", "maps", "reproduce", "trace", "corners", "note"}
    n = len(_get(session, "trackmap", lap=1)["driven_line"]["lap_dist"])
    assert all(v is None or len(v) == n for v in d["trace"].values())
def test_refused_ideal_lap_has_inputs_none():
def test_v4_file_without_steering_is_partial():                     # yaw None -> steer null with reason, status partial
def test_brake_stuck_at_zero_is_refused_per_input():               # brake null, throttle still there
def test_spin_lap_never_500():                                      # Review Focus 1: steer pinned at +-1 for 3 s on the analysed lap
def test_json_has_no_nan():                                         # json.dumps(d, allow_nan=False) succeeds
```

The ring (all throttle, no brake) cannot pass the brake map. The schema test may therefore assert brake null with a reason and check lengths only on the non-null traces. The fully populated path is T13/T14 plus the real files (Task 8).
- [ ] **Step 2: Run them and check that they fail.**
- [ ] **Step 3: Implement.** In `analyse_lap`, after the corner rows: `gxt_i = _kinematic_gxt(car, ln, v_i)`, then `inputs_block(...)` with the trace's `at` / `near` helpers so the traces sit on `/trackmap`'s driven-line points (pedals are interpolated; `mode/abs/tc` use the nearest point). `input_others` are the file's qualifying `State`s (their `steer, v, ld`), used for corrections. Every exception inside `inputs_block` becomes `{"status": "insufficient_data", "reason": "ideal inputs failed: <type>: <msg>"}` and never fails the ideal lap. A reproduce failure sets that input's `*_ideal` trace to None and `refused[input] = "map does not reproduce your lap: <input> RMS %.1f"`. Its map and RMS are still returned.
- [ ] **Step 4: Run the new tests and the whole backend suite** (`… pytest tests -q`): everything should PASS, with no regressions in `test_ideal_*.py`.
- [ ] **Step 5: Measure the run time added (I8)** on one real HF1 file. `build_ideal_lap` with and without `input_maps` (second call, maps cached) should differ by ≤ 0.5 s. Print both.
- [ ] **Step 6: Commit** `Ideal inputs: the inputs block of /ideal-lap`.

### Task 8: Validation report (I1-I8) on the real files, and docs

**Files:**
- Modify: `backend/tools/ideal_lap_report.py` (`--inputs`), `backend/tools/IDEAL_LAP_REPORT.ps1` (passes args through, check), `backend/tools/README.md`
- Create: `backend/tools/IDEAL_INPUTS_RUN_ALL.ps1` (the spec §7 file list, one `--fit X --sim X --holdout odd --inputs` per file)
- Modify: `DevNotes.md` (`### 11/27/2026` or the ship date), `docs/HANDOFF.md` (§2 `/ideal-lap` row: the `inputs` block; §9 `ideal_inputs.py` line)
- Test: `backend/tests/test_ideal_report_rows.py` (row keys of `--inputs` on the synthetic laps)

**Interfaces:**
- Consumes: `fit_brake_map`, `fit_throttle_map`, `fit_steer_map`, `inputs_on_line`, `reproduce`, `corner_events`, `analyse_lap(input_maps=…)`.
- Produces: per pair a JSON / printed row `I1…I8`, as spec §7. I1-I3 are the held-out map scores. I4 is reproduce per test lap. I5 is the "another lap" test, ported from `inp_p3_delta.py`: brake ABSOLUTE, throttle DELTA, steering ABSOLUTE, with RMS where the two laps differ by > 2 km/h. I6 is the spread of the ideal's brake point and throttle pick-up per corner over the 3 test laps. I7 counts violations (ideal brake > cap, throttle outside 0..100, ideal brake point before yours where the ideal's entry is faster). I8 is the added seconds.

- [ ] **Step 1: Write the failing row test.** `--inputs` on the synthetic set returns the keys `i1…i8`, each with its threshold and a `pass` bool.
- [ ] **Step 2: Implement it and run the test**: it should PASS.
- [ ] **Step 3: Real-file checks against the spec's probes**, read-only, with `--cache` in `%TEMP%`:
  - Porsche COTA `11-09T04_29_26`, BMW M4 Fuji `10-14`, Ferrari COTA `10-27` and `10-30`.
  - Expected: brake `a`, `e1`, `e0` and held-out R2 within 0.01 of the §1.1 table; throttle table within 0.03 per bin of §1.2; steering `a`, `b` within 3 % of §1.3.
  - A miss is reported with both numbers, not tuned.
- [ ] **Step 4: Run `IDEAL_INPUTS_RUN_ALL.ps1`** on every spec §7 file and save the output to `%TEMP%\ideal_inputs_report.txt`. Report the gate:
  - I1-I5 for brake and throttle on every pair, and I3-I5 for steering on every pair that has steering;
  - I6-I8 are reported only.

  A failure is a finding for the conceptual owner (Opus), not a threshold to tune.
- [ ] **Step 5: Docs.**
  - DevNotes: what shipped, the I-table per pair (measured), the open items (UI step 9 gated on this report; the new brake-bias / brake-pressure channels are not used yet).
  - HANDOFF §2 and §9 lines.
  - tools README entry.
- [ ] **Step 6: Commit** `Ideal inputs: validation report I1-I8, docs`.

---

## Not in this plan

- **The UI** (spec §6 and step 9): a separate plan once the Task 8 gate passes.
- **Using the recorder's new `Rear Brake Bias` / `Brake Pressure` / `Steering Wheel Range` channels** to carry maps across setups. There are no in-game recordings with them yet.
- **The HUD pedal marker** (Q5: later).
