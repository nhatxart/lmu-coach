# Ideal driver inputs, phase 1-lite: design (11/24/2026)

Status: design for review. Nothing built. Not committed. Conceptual owner: Opus 5.5 (ideal-lap
design analyst). Decisions that are mine are marked **[Opus]**; questions only the user can answer
are in §10 with a recommendation each.

Builds on the ideal lap (`2026-11-10-ideal-lap-design.md`, through §15.z2): a speed profile on the
DRIVEN line (2 m grid) with a tyre-side longitudinal demand `gxt` and a binding code per point
(`front`, `rear`, `brake`, `traction`, `power`, `measured`, `driven`). Phase 1-lite turns that
profile into brake %, throttle % and steering % per point. It uses inverse maps fitted from the
recordings and no optimal-control solve. Out of scope: a dynamic OCP on the line, a free line, GPU.

Probes (spike folder `docs/superpowers/spikes/ideal-lap/`, read-only on recordings, this
worktree's `ideal_*` modules):
- `inp_p1_maps.py`: brake and steering maps, even laps fit, odd laps scored; lag search.
- `inp_p2_thr_steer.py`: throttle over all lateral g; steering in bicycle form with slip angles.
- `inp_p3_delta.py`: absolute vs delta maps, scored as "predict lap j's inputs from lap j's
  state (absolute) or from lap i's inputs plus the map difference (delta)", over all ordered
  pairs of held-out laps.

Files used: Porsche 911 GT3 R COTA `11-09T04_29_26` (20 laps), BMW M4 GT3 Fuji `10-14` (15 laps),
Ferrari 499P COTA `10-27` (11 laps) and `10-30` (6 laps).

---

## 1. What the recordings say (measured)

Channels: `Brake Pos`, `Throttle Pos` (0-100 %, the game's UNFILTERED pedal, before TC/ABS),
`Steering Pos` (−1..1, unfiltered), `ABS`, `TC`, `Yaw Rate`, `Slip Angle` ×4. Recorder HF1 files
only for steering (yaw and slips). NOT recorded: brake bias (`mRearBrakeBias`), steering-wheel
range (`mPhysicalSteeringWheelRange`), brake pressure (`mBrakePressure`). The library has all three
(`lmu_data.py:47, 146, 149`); see §10 Q3.

### 1.1 Brake map

`-gxt = a · brk/100 + e1 · v + e0`, fitted on braking samples (brk > 3 %, throttle < 5 %, ABS
off); gxt = gx + grade + drag (the same tyre-side quantity the solver uses).

| car | a (g per 100 %) | e1 (g per m/s), e0 | lag | held-out R2 (straight, ABS off) | RMS (g / brake-% pts) | ABS active by pedal 0-20 / ... / 80-100 % | speed effect at > 50 % pedal |
|---|---|---|---|---|---|---|---|
| Porsche GT3 R | 1.70 | 0.0022, 0.12 | 0.00 s | 0.957 | 0.050 / 2.9 | 2 / 2 / 3 / 7 / 8 % | 0.0000 g per m/s |
| BMW M4 GT3 | 1.76 | 0.0013, 0.25 | 0.00 s | 0.975 | 0.043 / 2.4 | 5 / 15 / 16 / 16 / 38 % | −0.0007 |
| Ferrari 499P 10-30 | 1.96 | 0.0017, 0.07 | 0.00 s | 0.978 | 0.048 / 2.5 | 0 % everywhere | +0.0005 |
| Ferrari 499P 10-27 | 1.99 | 0.0019, 0.03 | 0.00 s | 0.993 | 0.031 / 1.6 | 0 % everywhere | +0.0001 |

- Linear in the pedal and independent of speed at high pedal (downforce does not change the
  pedal-to-decel map below the limit; the limit itself is the ideal lap's `kb · S`).
- No measurable lag at 50 Hz with 0.2 s smoothing.
- Pedal shown: p99 100 %, the peak pedal of a lap's brake zones has median 100 % in every file.
- Hypercars: ABS never active in these files. GT3: ABS 2-38 % at high pedal (BMW most).
- Front share of the braking force (Ferrari, forces recorded): median 0.48-0.49, p10/p90
  0.31-0.36 / 0.55. It moves a lot within a stop (regen, load transfer). Not used by the map; reported.
- "Another lap" test (`inp_p3`, held-out lap pairs): brake RMS 1.8-2.1 % pts over the whole lap
  (2.1-2.5 where the two laps' speeds differ by > 2 km/h); the trivial "same as your lap" 7.2-8.4.

### 1.2 Throttle map

`gxa = coast(v, gy) + f(thr) · (power(v, gy) − coast(v, gy))`, `gxa` = gx + grade (the engine
curve's own quantity), `coast` fitted on throttle < 2 %, f per 10 % pedal bin (monotone).
Samples: brake off, TC off, v > 8.

| car | f(5, 15, ..., 95 %) | pedal use while driving 0-5 / 5-95 / 95-100 % | held-out gxa R2 / RMS | part throttle RMS | pedal from gxa (inverse), part throttle | lag |
|---|---|---|---|---|---|---|
| Porsche | 0.04 0.11 0.22 0.33 0.45 0.54 0.65 0.75 0.83 1.00 | 1 / 16 / 83 % | 0.955 / 0.039 g | 0.055 g | RMS 8 % pts, bias −1 | 0.04 s |
| BMW M4 | −0.01 0.07 0.13 0.30 0.40 0.50 0.62 0.71 0.79 0.94 | 0 / 12 / 88 % | 0.919 / 0.053 g | 0.058 g | 8 % pts, 0 | 0.04 s |
| Ferrari 10-30 | 0.04 0.17 0.30 0.41 0.51 0.60 0.70 0.75 0.82 1.01 | 1 / 16 / 83 % | 0.942 / 0.056 g | 0.053 g | 6 % pts, +1 | 0.02 s |
| Ferrari 10-27 | 0.06 0.20 0.32 0.40 0.51 0.60 0.72 0.80 0.88 1.00 | 0 / 20 / 80 % | 0.940 / 0.060 g | 0.063 g | 7 % pts, −1 | 0.06 s |

- The pedal is close to linear in drive fraction, with a soft top (last 10 % of travel = last
  ~15 % of drive). Part throttle is where the corners are (it was absent when only |gy| < 0.4
  was used: 0-1 % of the samples).
- TC: active on 8-14 % of drive samples; the driver's pedal while TC is active has p50 97-99 %
  (p10 15-68 %). The drivers stay flat and let TC cut.
- "Another lap" test: throttle RMS over the whole lap ABSOLUTE 10.5 / 12.0 / 49.0 % pts
  (Porsche / Ferrari / BMW), DELTA 9.3 / 9.4 / 11.5, "same as your lap" 21.8 / 16.5 / 18.4. Where
  the laps differ: ABSOLUTE 11.0-12.7, DELTA 10.5-12.6, same 19.8-28.1. BMW's 49 % absolute is a
  full-throttle misread on the straights: the inverse gives < 100 % where gxa sits just under the
  median engine curve. The delta form and the A1 power rule (§2.2) both remove it.

### 1.3 Steering map

Bicycle form `steer = a · k + b · (α_f − α_r) + c0`, `k` = yaw rate / v (path curvature),
α = mean slip angle of the axle (rad, signed). With MEASURED slip angles:

| car | a | b | implied wheelbase a/b | road-wheel lock 1/|b| | held-out R2 / RMS (% of p95 |steer|) | lag |
|---|---|---|---|---|---|---|
| Porsche | 8.11 | −2.82 | 2.87 m | 20.3° | 0.980 / 7.0 % | 0.12 s |
| BMW M4 | 9.09 | −2.53 | 3.58 m | 22.6° | 0.987 / 5.3 % | 0.10-0.12 s |
| Ferrari 10-30 | 11.33 | −3.13 | 3.62 m | 18.3° | 0.981 / 6.3 % | 0.10 s |
| Ferrari 10-27 | 11.54 | −3.08 | 3.75 m | 18.6° | 0.981 / 6.6 % | 0.10-0.12 s |

- The bicycle form holds (R2 0.98-0.99). The implied wheelbases are of the right order but read
  15-25 % long against the real cars (about 2.5-3.2 m). The map is empirical: its a and b are fitted,
  not taken from geometry.
- Steering ratio/lock differs by car (a 8.1 / 9.1 / 11.3-11.5) and is a setup item in LMU →
  the map is fitted **per file** (like the load model).
- The ideal lap has no slip angles. Their difference must come from an understeer map
  `α_f − α_r = u1 gy + u2 gy|gy|`. That map is weak: R2 0.30-0.54 on the fit laps. Steering from
  it gives held-out R2 0.86-0.95, RMS 10.6-18.2 % of p95 |steer|, curvature alone 13.4-22.0 %.
- Driver corrections (steering smoothed 0.2 s minus 1.0 s): RMS 0.021-0.036 = 6-8 % of p95.
- "Another lap" test: steering RMS ABSOLUTE 5.7-10.5 % of full lock (the p99.5 of |steer|),
  DELTA 6.1-11.7, "same as your lap" 8.2-13.8.
- Size of the ideal's steering change ON THE SAME LINE: only the understeer term moves (k is the
  line's). Example, Porsche: gy 1.6 → 1.7 g changes α_f − α_r by 0.014 · (2.89 − 2.56) = 0.0046 rad,
  steering by 0.013 = 2.6 % of lock. That is below the 6-10 % accuracy.

## 2. The maps and the ideal inputs **[Opus]**

All maps are fitted per FILE on the file's qualifying laps (pedal maps depend on brake bias and
pressure setup, steering on the lock; one file = one setup, ideal-lap §4). Fewer than 3 laps:
the group's pooled maps, flagged `inputs_map: "group"`.

### 2.1 Brake
- Map: as §1.1, least squares on brk > 3 %, throttle < 5 %, ABS off, |gy| < 0.4. Store
  `a, e0, e1, r2_heldout` (fit even laps / score odd laps), `n`, `abs_share_by_pedal`.
- Ideal pedal at a point with ideal mode `brake` (§2.4): `brk_I = clip(100 · (−gxt_I − e1 v_I − e0) / a,
  0, brk_cap)`, `brk_cap` = p99 of the file's pedal (100 % in every file measured). Where the
  ideal's decel needs more than the pedal map gives at `brk_cap` (the tyre limit with ABS, GT3), the
  pedal is `brk_cap` and the point is marked `abs: true`.
- Absolute map, not delta: absolute and delta score the same (1.8-2.1 vs 1.8-2.0 % pts), and
  absolute has no channel-switching cases. Lag 0.

### 2.2 Throttle
- Map: as §1.2, the ideal lap's engine curve `power(v, gy)` with A1's measured offset `p_off(s)`
  on the reference line (so full throttle reproduces the driven straight exactly), coast fitted on
  throttle < 2 %, f table per 10 % (monotone, ≥ 20 samples per bin, empty bins interpolated).
- Ideal pedal by binding code:
  - `power` → **100 %**. A1 makes power-bound mean the measured flat-out acceleration.
  - `traction` → **100 % with `tc: true`**. The driver's own TC data: pedal p50 97-99 % while TC
    cuts. LMU's TC is the device that holds the traction limit, so the ideal is "flat and let TC
    work" (see §10 Q2).
  - otherwise in drive mode → **delta form**: `thr_I = clip(thr_D + 100·[F⁻¹(state_I) − F⁻¹(state_D)], 0, 100)`,
    where F⁻¹ inverts the table at the state's drive fraction. Where the driven point is not in
    drive mode, the absolute `F⁻¹(state_I)` is used.
  - Reason for delta: 9.3-11.5 vs 10.5-49 % pts (§1.2); it also returns exactly your pedal
    where the ideal is your lap (`driven`, `measured`).
- Lag: the ideal pedal is shifted 0.04 s earlier than the state it produces (measured 0.02-0.06 s).

### 2.3 Steering
- Map: §1.3 bicycle form with the understeer map, per file; requires Yaw Rate and Slip Angle
  (HF1) and held-out R2 (measured slips) ≥ 0.95, else steering is omitted (`steer: null`, reason).
- Ideal steering = the absolute map at the ideal state, `steer_I = a k(s) + b US(gy_I) + c0`, shown
  as a smooth reference, shifted 0.10 s earlier (measured lag).
- **No steering ADVICE in phase 1-lite.** On the driven line the ideal changes steering only by the
  understeer term (2-3 % of lock per 0.1 g), below the map's 6-10 % accuracy. What steering CAN
  say is how much you corrected: `corrections` = RMS of (your steering 0.2 s − your steering 1.0 s)
  per corner, compared with the same corner on your other laps of the file (median, p90). It is
  shown only when the corner is in the worst 20 % of your laps there.

### 2.4 Mode per point and trail braking
- Mode of the ideal per point from its `gxt_I` and the maps: `brake` if `−gxt_I > e1 v + e0 + 0.05 g`;
  `drive` if `gxa_I > coast + 0.03 g`; else `coast` (both pedals 0).
- One pedal per point. The quasi-steady solver has no trail-braking state of its own: where it
  brakes in the corner (binding `brake` with |gy| > 0.4), the brake pedal follows `gxt_I` there
  (that IS the trail-brake release the model wants). Overlap of both pedals is not produced.
- Pedal traces are smoothed with the same 0.2 s window as yours (in time at the ideal speed) so the
  overlay compares like with like.

## 3. Tests before ideal inputs are trusted **[Opus]**

### 3.1 "Reproduce your lap" (per file, every request, cached)
Feed the DRIVEN profile (`v_ref` on the 2 m line, its kinematic `gxt`, `gy = ks v²`, k from the
line) through the ABSOLUTE maps (delta would reproduce it trivially). Compare with your recorded
inputs (smoothed 0.2 s) over the lap. Limits (measured "another lap" ABSOLUTE values, which also
include line differences, with margin):

| input | Hypercar limit | LMGT3 limit | measured (ABSOLUTE, held-out lap pairs) |
|---|---|---|---|
| brake (% pts RMS) | ≤ 3.5 | ≤ 3.5 | 1.8-2.1 |
| throttle (% pts RMS, drive points) | ≤ 15 | ≤ 15 | 10.5-12.0 (BMW 49 before the §2.2 power rule; the builder re-measures) |
| steering (% of lock RMS) | ≤ 10 | ≤ 13 | Ferrari 6.9, BMW 5.7, Porsche 10.5 |

Fail → that input's ideal trace is not returned (`refused: "map does not reproduce your lap: X"`),
the others still are. Stored per input: the RMS, so the UI can show it.

### 3.2 Synthetic perfect driver (T12-T14)
`tests/synthetic_ideal.py` gains known input maps: `brk = 100 (−gxt − e1 v − e0)/a`, a throttle
table over the true engine curve, steering `a k + b (α_f − α_r)` with α from the true car
(understeer from axle usage). Telemetry carries those pedals and the steering, plus 3 % pedal noise
and 0.02 rad steering noise.
- T12 map recovery: a, e0, e1 within 2 %; f table within 0.03 per bin; steering a, b within 3 %.
- T13 reproduce: brake ≤ 1.5 % pts, throttle ≤ 4 % pts, steering ≤ 3 % of lock (noise level).
- T14 ideal = truth: solve the true car on the line (no noise) → its pedals from the TRUE maps vs the
  app's ideal inputs from the FITTED maps: brake ≤ 3 % pts, throttle ≤ 6 % pts; brake point
  (first ≥ 10 %) within 4 m, throttle pick-up (first ≥ 20 % after the minimum speed) within 0.1 s.

## 4. Per-corner summary **[Opus]**

Events per corner window (the ideal lap's §5.6 windows), for you and for the ideal on the same
Lap Dist axis:
- brake point = first brk ≥ 10 % in the entry; peak brake %; release = last brk ≥ 5 % before the
  minimum speed;
- throttle pick-up = first thr ≥ 20 % after the minimum speed; full throttle = first thr ≥ 95 %;
- steering `corrections` (§2.3).

A line is shown only above the resolution of the method:

| item | resolution (threshold to show) | basis |
|---|---|---|
| brake point | 4 m | 2 m grid + 0 lag + brake map 2 % pts |
| peak brake | 5 % pts | brake RMS 1.8-2.9 |
| release point | 6 m | grid + map |
| throttle pick-up / full | 0.10 s or 5 m | pedal lag 0.02-0.06 s, delta throttle 9-12 % pts |
| throttle amount in the corner (mean over min-speed → full) | 15 % pts | throttle RMS 9-12 |
| steering corrections | worst 20 % of your laps at that corner | §2.3 |

Text, at most two lines per corner, from the largest time item first (the ideal lap's `where`:
entry / mid / exit):
- "Brake 8 m later, 15 % less pressure, release 6 m later" (entry)
- "Throttle 0.2 s earlier, flat 12 m earlier" (exit)
- "Hold more speed at the apex: less brake, earlier part throttle" (mid, when the mode at the
  ideal's minimum speed is drive and yours is brake)
- "More steering corrections than on your other laps here" (when flagged).
Each line carries the corner's `gain_s` so the UI orders by time.

## 5. API **[Opus]**

The inputs come from the same line and solve as the ideal lap and cost no extra solve. They are a
block of the existing `GET /sessions/{source}/{file}/ideal-lap?lap_number=N` response (same cache
key + `INPUTS_VERSION`), `inputs` = null when the ideal lap is refused:

    "inputs": {
      "status": "ok" | "partial" | "insufficient_data", "reason": str | null,
      "maps": {"brake":    {a, e0, e1, r2, n, abs_share, cap, scope: "file"|"group"},
               "throttle": {table[10], coast[4], lag_s, r2, scope},
               "steer":    {a, b, us[2], c0, lock_p995, lag_s, r2_measured, r2_us, scope} | null},
      "reproduce": {"brake_rms": x, "throttle_rms": x, "steer_rms_pct_lock": x | null, "pass": {brake, throttle, steer}},
      "trace": {brk_ref[], brk_ideal[], thr_ref[], thr_ideal[], steer_ref[], steer_ideal[],
                mode_ideal[], abs_ideal[], tc_ideal[]}        # at the /trackmap driven_line points, like ideal.trace
      "corners": [{n, brake_point_m: {you, ideal, delta}, brake_peak_pct: {...}, release_m: {...},
                   throttle_on: {you_m, ideal_m, delta_m, delta_s}, full_throttle: {...},
                   corrections: {you, laps_median, flagged}, lines: [str, str]}],
      "note": "one sentence: what the ideal inputs are and their limits"
    }

Refusals per input (never an exception, never a number from a failed map): no Brake Pos variation
(< 200 braking samples) → brake null; throttle table with < 5 populated bins → throttle null; no
Yaw Rate / Slip Angle or steering R2 < 0.95 → steer null; reproduce fail (§3.1) → that input's
ideal trace null, its map and RMS still returned. Native LMU files are already refused by the ideal
lap.

## 6. UI sketch (words) **[Opus]**

Compare tab, when the "Ideal lap" ghost is active:
- An "Inputs" strip under the six overlaid traces, same Lap Dist axis and synced zoom: three thin
  traces with brake (red) and throttle (green) together, and steering below. Yours is solid and the
  ideal dashed, in the ghost's colour. The ideal's ABS points are a small tick on the brake trace
  and its TC points a band on the throttle trace.
- The steering trace is labelled "reference" (no advice; §2.3).
- The per-corner table gains an "Inputs" column with the ≤ 2 lines of §4. Hover on a line
  highlights the event on the strip (both markers: you, ideal).
- An input whose reproduce test failed shows "not shown: the model does not reproduce your
  throttle on this lap (RMS 18 %)" instead of the trace.
- The live HUD's pedal bars can show the ideal pedal as a thin marker at the scrubber position
  (phase 1-lite: optional, §10 Q5).

## 7. Validation protocol (I-rows, like ideal-lap §9) **[Opus]**

Files: every HF1 pair of the ideal-lap §9 list with ≥ 3 qualifying laps (Porsche COTA, Barcelona,
Road Atlanta, Fuji, Sebring; BMW M4 Fuji, Imola; Mustang COTA; Ferrari COTA, Road Atlanta; BMW
Hybrid Daytona; Genesis, Valkyrie Sarthe). The v4 files (no yaw or slips) are tested for brake and
throttle only.

| # | Check | Threshold | Measured in probes |
|---|---|---|---|
| I1 | Brake map held-out (odd laps): R2, RMS % pts on ABS-off samples | R2 ≥ 0.93, RMS ≤ 3.5 | 0.957-0.993, 1.6-2.9 |
| I2 | Throttle map held-out: gxa R2; inverse pedal RMS on part throttle | R2 ≥ 0.90; ≤ 10 % pts | 0.919-0.955; 6-8 |
| I3 | Steering map held-out with measured slips: R2 | ≥ 0.95 | 0.980-0.987 |
| I4 | Reproduce your lap (§3.1), per input | §3.1 table | §3.1 |
| I5 | Another lap (`inp_p3` method in the tool): brake ABSOLUTE, throttle DELTA, steering ABSOLUTE, RMS where the laps differ | brake ≤ 3.5, throttle ≤ 14, steering ≤ 10 / 13 % of lock (Hyp / GT3) | 2.1-2.5, 10.5-12.6, 7.9-13.0 |
| I6 | Stability: the ideal's brake point and throttle pick-up per corner over the 3 reference laps of a file | median spread ≤ 6 m / 0.15 s | not measured |
| I7 | Plausibility: ideal brake ≤ cap, throttle 0-100, ideal brake point never before yours where the ideal's entry is faster | 0 violations | by construction; checked |
| I8 | Run time added to `/ideal-lap` | ≤ 0.5 s | not measured (maps are least squares) |

Gate for the UI: I1-I5 on every listed pair (brake and throttle), I3-I5 steering on every pair
that has steering; I6-I8 reported. A failure is a finding for the conceptual owner, not a
threshold to tune.

## 8. Builder steps (each ends with its tests green)

Flat module `backend/app/ideal_inputs.py` (pure numpy, no FastAPI): `fit_brake_map`,
`fit_throttle_map`, `fit_steer_map` (each returns a dataclass with `to_dict` / `from_dict` and a
held-out score); `inputs_on_line(car, ln, sol, maps, raw_ref) -> trace`; `reproduce(maps, ln,
raw_ref) -> rms`; `corner_events(trace, ld, corners) -> corners`.
1. `ideal_state.lap_raw` gains `steer` (Steering Pos, unfiltered) and `yaw` is already there;
   bump `REDUCTION_VERSION`; the per-file cache keeps steering.
2. `fit_brake_map` + T12 (brake part); real check: `inp_p1` numbers for Porsche/Ferrari within 0.01 in R2.
3. `fit_throttle_map` (with A1's `p_off` on the reference line) + T12 (throttle); real check: `inp_p2`
   tables within 0.03 per bin.
4. `fit_steer_map` (bicycle + understeer map, lag search 0-0.3 s) + T12 (steering); real check:
   `inp_p2` a, b within 3 %.
5. `inputs_on_line` (modes, brake absolute, throttle by binding code + delta, steering reference,
   lags, smoothing) + `reproduce` + T13, T14.
6. `corner_events` and the text lines (§4 thresholds) + unit tests on a synthetic corner (brake 8 m
   later → the line says 8 m; a 3 m difference → no line).
7. `ideal_lap.build_ideal_lap` adds the `inputs` block; T9 (endpoint) asserts its schema and that the
   traces have the driven line's length; refusal tests per input (no steering in a v4 file; a file
   with the brake pedal stuck at 0).
8. `backend/tools/ideal_lap_report.py --inputs`: rows I1-I8; `run_all.ps1` pairs; hand over to Sonnet.
9. UI (index.html, Compare tab) only after I1-I5 pass and §10 Q1-Q2 are answered.
Keep the code base's habits ("PREVIOUSLY / Fix" comments, DevNotes entry, HANDOFF §4 note when it ships).

## 9. Why not more (scope guard) **[Opus]**

- No steering advice: the measured accuracy (6-10 % of lock) is above the size of the change the
  same-line ideal implies (≈ 2-3 % per 0.1 g). A free-line phase is where steering advice belongs.
- No pedal overlap or trail-brake rotation: the quasi-steady solver has no yaw state; phase 2 (a
  dynamic OCP) can add it.
- Brake bias and steering lock are not recorded, so maps are per file. They cannot be compared
  across setups until the recorder stores them (§10 Q3).

## 10. Questions for the user (my recommendation after each)

1. **Steering:** on your own line the ideal lap changes your steering by less than the model can
   measure (about 2-3 % of lock vs an accuracy of 6-10 %). *Recommended:* show the ideal steering
   as a smooth reference trace with no advice, plus a per-corner flag when you corrected more than
   on your other laps there.
2. **Where the ideal lap is traction-limited** (slow-corner exits), should the ideal throttle be
   "flat, TC holds the limit" (100 % with a TC band) or the pedal that would just avoid TC?
   *Recommended:* 100 % with the TC band. Your own data shows you already stay at 97-99 % while TC
   cuts, and that is what LMU's TC is for. Tell me if you run TC off or low in some cars.
3. **Recorder additions:** brake bias (`mRearBrakeBias`), steering-wheel range
   (`mPhysicalSteeringWheelRange`) and brake pressure (`mBrakePressure`) are available in the game
   but not recorded. *Recommended:* yes, a small recorder change. The maps could then carry across
   setups and sessions, and the ideal could say "brake bias" where it matters. New recordings only.
4. **One pedal at a time:** phase 1-lite shows either brake or throttle per metre (no overlap) and
   treats trail braking as the brake fading with the model's deceleration. *Recommended:* accept for
   1-lite; overlap and rotation need the dynamic model (phase 2).
5. **HUD marker:** also show the ideal pedal as a thin marker on the live pedal bars while
   scrubbing? *Recommended:* yes, after the Compare-tab version has been used for a while.
6. **Thresholds of the per-corner lines** (§4: 4 m brake point, 5 % pressure, 0.1 s throttle):
   *Recommended:* keep these (they are the method's resolution). Say if you want fewer lines (e.g.
   only items worth > 0.05 s).


**Answered 11/24/2026 (user accepted every recommendation):** Q1 steering = reference trace only, no advice (corrections flag instead); Q2 traction-limited exits = flat throttle with the TC band; Q3 yes, add brake bias, steering range and brake pressure to the recorder; Q4 one pedal per metre in this phase; Q5 HUD marker later; Q6 thresholds kept (brake point 4 m, pressure 5 points, throttle 0.1 s).
