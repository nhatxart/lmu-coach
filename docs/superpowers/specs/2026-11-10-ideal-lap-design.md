# Ideal lap: design (11/10/2026)

Status: design for review. Nothing built in the app. Not committed.
Follows `2026-11-09-theoretical-best-lap-feasibility.md` and `2026-11-10-ideal-lap-strategy.md`
(measured facts and the prototype live there; this document does not repeat them).
Roles: Fable wrote this (decisions, questions, instructions). Opus 5.5 builds, Sonnet validates
(§9), Opus reviews. Come back to Fable only for a conceptual problem (§11 says what counts).

Every decision that is mine and not the user's is marked **[Fable]**. Things only the user can
decide are in §12 with a recommendation; §12 lists what the build may assume until answered.

Probe scripts behind the numbers here: `docs/superpowers/spikes/ideal-lap/fable_p1.py` ..
`fable_p4.py` (read-only on recordings; they patch `il_core.py` in memory, the file is untouched).

---

## 1. Intent

Given a lap the user drove, show the lap this car could have done **on the same line** if every
corner, braking zone and exit used the grip the car's tyres have, with tyre load (aero, weight
transfer, crests) deciding how much grip each axle has at each point. Works for every class from
the channels that class records: tyre forces where the game gives them (Hypercar), suspension
force and slip angles where it does not (LMGT3).

The output is a second speed trace over the same line, a lap time with an honest range, and per
corner: time available, what limits the car there, and how much of each axle's grip was used.

## 2. Goals and non-goals

Goals
- One algorithm for all classes; the class only changes which channels feed the fit.
- Built on the car's own recordings; no hand-entered car data.
- Never shows an "ideal" that is slower than the lap it is compared with.
- Says how sure it is: a range on the lap time, and a status when the data is too thin.
- Refuses (no ideal lap, with the reason) instead of guessing when the data is not there.

Non-goals (v1)
- No line optimisation. The line is the driven line of the reference lap.
- No setup advice. One number ("setup potential", §5.3) is shown; nothing more.
- No class prior: a car with no usable recordings gets no ideal lap **[Fable]** (loads and aero
  differ too much between cars of a class, and there is no evidence yet that anything transfers).
- No transient model (yaw inertia, damper states), no tyre temperature or wear state.
- No per-tyre force model: capacity is per tyre, usage is per axle (§5.5).
- Not fed to the LLM coach yet (§12 Q7).

## 3. What the probes settled (11/10/2026)

All on COTA unless said: Porsche 911 GT3 R file `...92_LM-2026-11-09T04_29_26Z` (the "11/8" file,
20 laps), Ferrari 499P `...2026-10-30T03_57_04Z` (6 laps) and `...2026-10-27T23_21_49Z` (11 laps),
Porsche Road Atlanta `...2026-10-26T04_07_00Z` (10 laps).

**F1. The balanced-mode error was an inconsistent load model, not a moving balance.**
The prototype fitted axle grip against the MEASURED load ratio and then solved with the MODELLED
one. The two differ in hard cornering: Porsche front axle mean load ratio measured 1.17-1.18,
modelled 1.10-1.11 (120-180 km/h); in the esses the front Susp Force reads +0.15 and the rear
-0.16 against the linear model (a Susp Force effect of cornering itself, probably suspension
jacking; whatever it is, the linear model had no term for it). Result: the driver exceeded the
"front limit" on 41 % of hard-cornering samples (p97 of usage 1.17) while the rear never passed
0.93 (`fable_p1.py`).
Fix, two parts (`fable_p2.py`, Porsche L9, q 0.75, same file):

| Variant | Balanced slower than driven on | worst | front usage > 1 | rear is the limiting axle on |
|---|---|---|---|---|
| prototype | 23.6 % of lap | 27 km/h | 41 % | 0 % |
| + grip fitted on the modelled load | 9.7 % | 20 km/h | 12 % | 16 % |
| + a `|gy|` term per wheel in the load model | 13.9 % | 18 km/h | 27 % | 0 % |
| both | 8.1 % | 15 km/h | 10 % | 12 % |

Ferrari: 14.0 % / 13 km/h to 12.0 % / 13 km/h (it was less wrong: Tire Load moves less in corners).

**F2. The axle balance does not move with speed or throttle.** Ferrari, front share of lateral
tyre force in hard cornering: median 0.470-0.495 in every speed band from 50 to 260 km/h
(p10-p90 0.44-0.50); 0.467 at full throttle, 0.495 on the brakes, 0.476 coasting. The share is
set by where the centre of gravity is, as moment balance says. So there is no "balance envelope
by speed" to measure: each axle must carry its fixed share, and the car is limited by whichever
axle runs out first. **Balanced mode is the physical model. Ceiling mode (both axles at their
limit at once) is a car with a different setup.**

**F3. What is left is noise at the level of this model, about 6 % in g.** On saturated samples
the ratio observed / predicted lateral g has a log sd of 0.053-0.071 (four files). It is not
place: a per-25 m correction learned on even laps does not hold on odd laps (sd 0.064 to 0.060
Porsche COTA, 0.055 to 0.045 Ferrari 10/30, 0.053 to 0.048 Ferrari 10/27, 0.066 to 0.062 with a
WORSE p97 on Road Atlanta). It is not banking: correlation of the per-place residual with
bank from the topography -0.14..+0.05 (`fable_p3.py`). **So: no location map, no bank term.**

**F4. The lap time moves 0.44 s per 1 % of grip level at COTA**, so the calibration quantile is a
2.4 s knob (Porsche balanced gain: q75 +1.97 s, q90 +3.26 s, q97 +4.41 s). "Never slower than the
driver" can only be bought with a high quantile, which invents seconds. The headline therefore
cannot be a bare model time at an arbitrary quantile (§5.4).

**F5. The envelope ("never slower than the reference lap") is 2.3x less sensitive to the level.**
At the median calibration (q 0.5), level scaled 0.98 / 1.00 / 1.02 (`fable_p4.py`):

| Lap (same-file fit) | plain model gain | envelope gain |
|---|---|---|
| Porsche L9 131.86 | -0.63 / +0.34 / +1.27 s | +0.93 / +1.32 / +1.83 s |
| Porsche L4 132.05 | -0.99 / +0.04 / +0.95 s | +1.03 / +1.39 / +1.79 s |
| Ferrari L11 117.85 | -0.72 / +0.32 / +1.32 s | +1.01 / +1.43 / +1.95 s |
| Ferrari L3 117.97 | -0.54 / +0.51 / +1.50 s | +1.18 / +1.56 / +2.07 s |

The envelope ideal is also stable across reference laps: Porsche 130.54 / 130.66 / 130.58 s from
laps of 131.86 / 132.05 / 132.70 (spread 0.09 %); Ferrari 116.42 / 116.41 / 116.87 (0.4 %).

**F6. The model holds across sessions on one track, not across tracks.**
Ferrari fitted on 10/27, run on 10/30: envelope gain +1.55 vs +1.43 s same-file (0.1 % of the lap).
Porsche fitted on Road Atlanta, run on COTA: +2.66 vs +1.32 s (1.0 %); fitted on COTA, run on Road
Atlanta: the model is under the driven speed on 63-76 % of the lap (22-34 % same-track), front
mu1 1.15 (COTA) vs 1.27 (Road Atlanta). Different setup and different track grip; not separable
with today's data. **So the model is fitted per car model AND track.**

**F7. Knee and Grip Fraction agree where both exist.** Ferrari: the slip-angle knee set (the GT3
method) reads 1.005-1.025 of the level fitted on Grip Fraction < 0.3 samples. With the modelled
load the Porsche rear knee set doubled (612 to 1,284 samples, knee 2.5-3.0 deg).
Tyre model v3 for the Porsche says front peak 3.98 deg (knee here 4.0: agrees), rear 3.91 deg
(knee here 2.5-3.0: disagrees; the validator reports this per car, §9 V7).

**F8. Drag without forces was never measured.** The joint drag/braking fit returned 0 on Road
Atlanta and 4.5-6e-5 on COTA, and its "coasting bound" was the hard-coded default both times
(fewer than 50 coasting samples). Replaced in §4.6.

## 4. The car model

Units as in the prototype: accelerations in g; `gx` kinematic longitudinal (+ speeding up, =
−`G Force Long`); `gy` lateral as recorded; `gz` vertical (crest −, compression +); `grade` =
dh/ds; `v` in m/s; tyre longitudinal demand `gxt = gx + grade + drag(v)`. Wheels FL, FR, RL, RR.
Signals used in fits are smoothed 0.2 s (10 samples at 50 Hz, edge-padded), as in `il_core.lap_state`.
Files are loaded at 50 Hz (`tire_model.TIRE_GRID_HZ`).

Two parts with different scope **[Fable]**:
- **Load model: per file.** A recorder file is one garage stint, so one setup (wing, ride height).
- **Grip, braking, traction, drag, engine: per fit group** = (car model, track slug, layout).

### 4.1 Load model (per file)

For each wheel i, least squares over the file's qualifying laps (§6.2):

    L_i = c0_i · (1 + a_i·v² + b_i·gx + c_i·gy + d_i·gz + e_i·|gy|)
    lam_i(v, gx, gy, gz) = max(L_i / c0_i, 0.02)

`L` = `Tire Load` when the file has it for all laps, else `Susp Force`. `e_i·|gy|` is new (F1).
Only the ratio `lam` is ever used (Susp Force is not wheel load). Store `c0`, the 5 coefficients
per wheel, `load_source`, and the fit R² per wheel on odd laps when fitted on even laps.
A file with fewer than 2 qualifying laps borrows the group's pooled load model (all the group's
qualifying laps in one fit) and says so (`load_model: "group"`).

**Rule that fixes F1, and binds every later change: every fit in §4.2-4.7 uses `lam` from this
model evaluated at the sample's own (v, gx, gy, gz), never the measured load.** The solver can
only know the modelled load, so the grip numbers must be fitted against the same quantity.

### 4.2 Lateral capacity per axle

    mu_j(lam, v) = mu1_j · max(1 − k_j·(lam − 1) − c_j·(v − 40)/40, 0.3)        j = front, rear
    A_j = ½ · Σ_{i in axle j} mu_j(lam_i, v) · lam_i

`A_j` is "the lateral g of the car when axle j is saturated" (it already contains the axle's
share of the lateral force, F2). Bounds `k ∈ [0, 0.15]`, `c ∈ [0, 0.45]`; fit as in
`il_core.fit_car` step 3 (least squares on the saturated set, trying the pinned corners).

Candidate samples (`base`): `|gxt| < 0.2`, `|gy| > 0.45·p99(|gy|)`, `v > 15`, `|yaw accel| < 0.4`.
Saturated set of an axle:
- **forces recorded** (Grip Fraction present): `base` and mean Grip Fraction of the axle < 0.3
  (the game's Grip Fraction is LOW when the tyre is saturated).
- **no forces**: the knee. `r = |gy| / mean modelled lam of the axle`; median of `r` in 0.5 deg bins
  of the axle's mean |slip angle| (bins with ≥ 60 samples). Knee **(round 1, §13 R3; was "first
  bin whose median reaches 97 % of the highest", which jumped a whole bin on a flat plateau and
  moved the front capacity 4-5 %)**: smooth the bin medians with a 3-bin running median;
  plateau = median of the three highest smoothed values; knee = the slip angle, linearly
  interpolated between bin centres, where the smoothed curve first reaches 97 % of the plateau.
  Set = `base` and knee ≤ slip ≤ knee + 4 deg.

Which terms are fitted **(round 1, §13 R2)**: `k` and `c` of an axle are fitted only if its
saturated set has ≥ 250 samples, a speed span (p90 − p10 of `v`) ≥ 15 m/s and a load span
(p90 − p10 of the axle-mean modelled `lam`) ≥ 0.15. Otherwise they are borrowed from the other
axle when that one is fitted (only `mu1` is then fitted), else the defaults `k = 0.03, c = 0.10`.
Stored as `terms: fitted | borrowed | default`, with `v_sat: [p02, p98]` of the set's speed.

Calibration **[Fable]**: scale `mu1_j` so that the MEDIAN of observed / predicted on the set is 1
(`q = 0.5`). Past the knee a tyre holds its peak force, so a saturated sample IS the limit plus
noise, and the median is the unbiased level. The prototype's 0.75 bought fewer "slower than
driven" points with invented grip (F4); §5.4 handles those points honestly instead.

Axle status:
- `observed`: ≥ 250 samples in the set (5 s).
- `lower_bound`: fewer. Then fit on the top 20 % of `r` among `base` samples; `k`, `c` taken from
  the other axle (both axles `lower_bound`: the defaults above). The car has not been seen to reach this axle's limit; the capacity is what was
  shown. (Mirrors tyre v3's `rising_at_edge`.) Expected for the rear of an understeering GT3.
- Store per axle: `mu1, k, c, n_sat, saturation` (`grip_fraction` | `knee`), `knee_deg`, status,
  `ratio_sd` (log sd of observed / predicted on the set), and tyre v3's `peak_slip_deg` for the same
  car next to the knee, with `knee_vs_v3: "agree" | "differ"` (differ = more than 1.5 deg apart).
  **[Fable]** v3's peak is a cross-check, not the source: v3 measures mu against the measured load,
  this model must use the modelled one.

### 4.3 Axle share

`s_f` = median of front lateral force / total on samples `|gy| > 1.0, |gx + grade| < 0.2,
|yaw accel| < 0.15` when lateral forces are recorded; else the static front share of the load
source (`Σ c0 front / Σ c0`), flagged `estimate` (Susp Force share is not weight share: Ferrari
0.369 vs 0.432). `s_r = 1 − s_f`. Used for braking, traction and the ceiling only; the balanced
cornering limit does not need it.

### 4.4 Braking

    |gxt| ≤ kb · (s_f·√(A_f² − gy²) + s_r·√(A_r² − gy²))

Limit samples, first rule that gives ≥ 100: `|gy| < 0.3` and brake > 40 % and (front Grip
Fraction < 0.3 | ABS active); else `|gy| < 0.3` and brake > 90 %; else brake > 85 %. `kb` = median
of the ratio. Store `kb`, the rule used, `n`. Fewer than 100 samples by every rule: `kb = 1.0`
flagged `assumed`.

### 4.5 Traction

    gxt ≤ kt · s_r · √(A_r² − gy²)

Limit samples: TC active, throttle > 30 %, `v < 45`, `gxt > 0.2`, `|gy| < 0.8·A_r`; if fewer than
100: throttle > 95 %, `v < 40`, same rest. `kt` = median. Fewer than 30: `kt = 1.0` flagged.
(Rear drive only. The Hypercar front motor above 190 km/h is inside the engine curve, where
traction does not bind.)

### 4.6 Drag

- Forces recorded: as the prototype (`D = (ΣFx − m·g·(gx + grade)) / (m·g)` against `1, v²`).
- No forces **[Fable]**: coast-down if the group has ≥ 150 samples (3 s) with throttle < 2 %,
  brake < 1 %, `|gy| < 0.3`, `v > 40`: `d2 = median(−(gx + grade) / v²)`, `d0 = 0`, flagged
  `coast_down` (it contains engine braking, so it is an upper bound; cap at 7e-5). Otherwise a
  prior, flagged `prior`: `d0 = 0.015`, `d2 = 4.8e-5` for LMGT3, `4.0e-5` for every other class
  (Ferrari measured: 0.054 g + 3.98e-5·v²). The joint fit with braking is dropped (F8).
  Drag only splits `gxt` between tyres and air; the engine curve below is kinematic and
  contains it. §9 V8 checks that the ideal time moves less than 0.2 s for ±30 % of `d2`.

### 4.7 Engine

Full throttle (> 98 %, TC off), `|gy| < 0.25`: median of `gx + grade` per 2.5 m/s speed bin
(≥ 25 samples), smoothed, extended as the prototype does. Cornering cost `cc` = median of
(curve − (gx + grade)) / gy² on full throttle with `|gy| > 0.8`, clipped to [0, 0.08].
`power(v, gy) = curve(v) − cc·gy²`.

### 4.8 What is dropped from the prototype
`q` as a parameter (fixed median everywhere), the joint drag fit, fitting on measured `lam`.

## 5. The solver and the ideal lap

### 5.1 The line
From the reference lap, on a 2 m grid of its own distance `s` (integrated speed), as
`il_core.line_of`: path curvature `ks = g·gy/v²` (smoothed 0.3 s), vertical curvature
`kv = g·gz/v²` (0.5 s), grade (0.5 s), and the driven speed `v_ref`.
Grade source in this order: `Pos Y` of the lap; else the track topography
(`track_topography.grade_bank` at the lap's Lap Dist); else zero with the note
`grade: none` (old v4 files on tracks without topography).

### 5.2 Balanced solve (the model)
As `il_core.solve(mode="balanced")`, unchanged in structure:
- feasible(v, gxt, gy, gz): `|gy| ≤ A_f` and `|gy| ≤ A_r` (with `lam` at `gx = gxt − drag(v)`), and
  `gxt ≤ kt·s_r·√(A_r² − gy²)` for drive, `−gxt ≤ kb·(s_f·√(A_f² − gy²) + s_r·√(A_r² − gy²))` for braking.
- corner limit `vc(s)` = highest speed at which some `gxt` is feasible (the prototype's fix: a
  front-limited car corners harder while slowing).
- forward pass with `min(power − grade, best feasible gxt − drag − grade)`, backward pass with the
  strongest feasible braking, twice round the closed lap. `gz` clipped to [−0.9, 1.5].
- **(round 1, §13 R4)** `vc` is bisected between the 0.5 m/s grid speeds (the builder's change,
  accepted), and the strongest feasible drive / braking demand is bisected between the last
  feasible value of the 0.025 g grid and the next (4 halvings), so the solver uses the fitted
  `kb`, `kt` and not 1-2 % less.
Result `v_M(s)` and, new, a **binding code** per grid point: `front`, `rear` (corner limit, which
axle), `brake`, `traction`, `power`, and `measured` (the point is held by the saturation clamp, §5.4).

### 5.3 Ceiling solve (setup potential)
`|gy| ≤ s_f·A_f + s_r·A_r` instead of the two axle conditions; rest the same. One time only.
Meaning shown to the user: "with a setup that brings both axles to the limit together, up to
−X s". It is an upper bound on what balance changes could give, not a lap this car can do (F2).

### 5.4 The ideal lap = envelope **[Fable]** (user to confirm, §12 Q1)

    v_I(s) = max(v_M(s), v_ref(s))          T_I = Σ ds / v_I

Why this is a legitimate lap and not a trick: the model's state is (s, v). Where the two speed
traces cross they are in the same state, so following one before the crossing and the other after
is a lap in which every piece is either allowed by the model or was actually driven. The
reference lap is proof of what was possible where the model says less (the model is ±6 % in g, F3).

**Saturation clamp (round 1, §13 R1; user question §12 Q10).** Where the reference lap's own
tyres were measured saturated, the limit is the speed driven there, not the model's:

    sat_ref (per 50 Hz sample of the reference lap) =
          |gy| > 0.45·p99(|gy|) of the lap   and   v > 15   and   |gxt| < 0.5·|gy|
      and ( any axle: mean Grip Fraction < 0.3                       when recorded
            else any axle: knee_j ≤ mean |slip angle| ≤ knee_j + 4 deg )
    runs shorter than 0.2 s are dropped; put on the 2 m grid (a grid point is clamped when more
    than half of its samples are)
    at clamped grid points:  vc(s) = v_ref(s)      (replaces the model's corner limit, up or down)

The forward and backward passes then run as before from those speeds, at every level (a measured
point does not scale with the grip level), and the envelope is taken as above. Clamped points
carry the binding code `measured`. `|gxt| < 0.5·|gy|` keeps the clamp to lateral-dominant
states: a tyre saturated by wheelspin or by braking says nothing about the corner speed.
The ceiling (§5.3) is NOT clamped: it describes another setup. `model_only` and V5 stay unclamped.
Why: at a saturated point `v_M / v_ref = √(A / |gy|)` is the model's own residual, so without
the clamp the envelope keeps the model's noise at half of the at-the-limit points (0.10-0.45 s
per COTA lap directly, 0.18-0.94 s with its knock-on; §13 R1).

Three levels: solve §5.2 with `mu1` of both axles × (1 − β), × 1, × (1 + β) and take the envelope
of each:
- `gain_min` (level 1 − β): "at least". Gains that survive a grip haircut.
- `gain` (level 1): "expected". The ghost uses this profile.
- `gain_max` (level 1 + β).

β **[Fable]**, to be confirmed by §9 V4: 0.02 for a track-scope fit with ≥ 6 laps and both axles
`observed`; 0.03 for any other track-scope fit **and for every knee-based (no-force) fit (round
1, §13 R3: a knee-based level is good to about ±4 %)**; 0.05 for a car-scope fit (§6.3). (Level
shift between sessions measured so far: 1.8 % Ferrari; between tracks 10 %.)

Also returned, not headlined: `model_only_time` = Σ ds / v_M at level 1 (the unbiased model
time; it can be slower than the driven lap), and the share of the lap where `v_M < v_ref − 2 km/h`.

Known bias **(round 1)**: the clamp removes the model's noise where the tyres were measured
saturated in cornering. What is left (braking and traction zones, corners the driver did not
saturate) is measured by the synthetic test (§8 T6) and stated in the UI note; `gain_min` is
the conservative figure. Also returned: `clamped_share` (share of the lap held by the clamp).

Lap time shown: `official lap time − (T_ref_grid − T_I)`, so the grid's own small timing error
cancels.

### 5.5 Grip used (per corner and per point)
For a state (v, gxt, gy, gz), per axle j:

    u_j = √( gy² + (gxt_j / κ)² ) / A_j
    braking:  gxt_j = gxt (both axles, by construction of §4.4), κ = kb
    driving:  rear gxt_r = gxt / s_r, κ = kt;  front gxt_f = 0

Computed for the ideal profile (from the solver's own gxt) and for the driven lap (from the
recorded state). Per tyre the model gives the capacity share `mu·lam_i` (which tyre carries the
axle), not a usage: usage is per axle **[Fable]** (§12 Q5). Hypercars could later show the
game's own per-tyre figure for the driven lap.

### 5.6 Corners
Corner windows from the track reference (`track_corners` / the same windows the Compare tab and
`corner_features` use, by Lap Dist). Per corner: entry/exit Lap Dist, time driven, time ideal,
`gain`, `gain_min`, min speed driven / ideal, the binding code at the ideal's minimum speed, max
`u_front` / `u_rear` driven, verdict:
- `at_limit`: `gain_min ≤ 0.02 s`;
- `time_available`: otherwise; with where: `entry` (braking zone), `mid` (min speed), `exit`
  by which third of the window holds most of the gain.
**(round 1)** Also per corner: `measured_share` (share of the window held by the clamp) and
`extrapolated: true` when the ideal's minimum speed is more than 5 m/s above the binding axle's
`v_sat[1]` (the model has never seen that axle saturated that fast; §13 R2).
Time outside corner windows goes to `straights` rows between corners.

## 6. Data, fit groups, storage

### 6.1 Channels per class

| Need | Hypercar (forces) | LMGT3 and any car without forces |
|---|---|---|
| Load ratio | Tire Load | Susp Force |
| Saturation | Grip Fraction < 0.3 | slip-angle knee (Slip Angle: recorder files only) |
| Axle share | lateral force share | static share of Susp Force (estimate) |
| Braking limit | front Grip Fraction, else ABS / brake % | ABS, else brake % |
| Drag | longitudinal forces | coast-down, else class prior |
| Always | Ground Speed, G Force Lat/Long/Vert, Throttle, Brake, ABS, TC, Yaw Accel, Lap Dist; Pos Y or topography for grade |

`telemetry.WHEEL_CHANNELS` must gain `Tire Longitudinal Force` and `Grip Fraction` (the prototype
appends them at run time). Native LMU files (no Slip Angle, no Yaw Rate, no forces): refused,
`insufficient_data: no slip angles`. LMP2/LMP3: no recordings yet; they take whichever column
their channels fit, with no code of their own.

### 6.2 Qualifying laps
Complete, real (the `session_index` flags), not a pit lap, lap time ≤ 1.05 × the file's best.
Per fit group at most the 40 most recent (cost). The reference lap itself may be in the fit.

### 6.3 Fit groups and fallbacks **[Fable]** (F6)
- **track scope**: (car model from `car_identity.identify_car`, track slug, layout within 3 % of
  length). Needs ≥ 3 qualifying laps. Game version: as tyre v3 (newest version with ≥ 3 laps, else
  all with the note `mixed_versions`).
- **car scope** (fallback): all tracks of the car model pooled; needs ≥ 6 laps; result flagged
  `transferred`, β = 0.05, and the UI says "estimate from other tracks".
- else `insufficient_data` with what is missing ("3 clean laps on this track needed, have 1").

Confidence shown: `high` (track scope, ≥ 6 laps, both axles observed, V1-style self-check passes),
`medium` (other track scope), `low` (car scope, or an `assumed`/`prior` part that binds on more
than 10 % of the lap).
Self-check stored with every fit: fit on even laps, median and sd of log(observed / predicted) on
the saturated samples of odd laps (`selfcheck_shift`, `selfcheck_sd`).

### 6.4 Storage **[Fable]**
Under the existing writable profiles mount, no compose change:
`data/grip_profiles/ideal_lap/<car model>.json`:

    {"version": 1, "car_model", "car_class",
     "groups": {"<slug>|<layout_len_m>": {scope, fingerprint, game_version, files[], n_laps,
                 axles{front{...}, rear{...}}, s_front, kappa_brake{...}, kappa_trac{...}, drag{...},
                 engine{power_v[], power_g[], corner_drag}, beta, confidence, selfcheck{...}, notes[]}},
     "car_scope": {...same shape...},
     "files": {"<file>": {mtime, size, load_model{static[4], coef[4][5], source, r2[4]}, laps[]}}}

Separate from the tyre-v3 profile on purpose: different fit, different version, and the v3 file
is already 2.3 MB for the Porsche. `fingerprint` = sha1 of the sorted `file#lap` keys + file
mtime/size + `version`, as `grip_profile._keys_fingerprint`; a group is refitted when it changes.
Fitting needs the raw 50 Hz laps of the group's files: keep a per-file reduction cache
(`data/grip_profiles/ideal_lap/cache/<sha>.npz`: the smoothed state arrays of the qualifying
laps: v, gx, gy, gz, grade, thr, brk, abs, tc, yaw accel, load source ×4, slip ×4, and fx, fy,
grip fraction, load ×4 when recorded), rebuildable, keyed by file mtime + size + version.
`backend/tools/ingest_sessions` gets an `--ideal-lap` pass that builds the caches and fits every
group, so the first request in the UI is not a multi-minute load of 6 files.
Relation to tyre v3: reads `tire_fit_cache[axle].fit.peak_slip_deg` for the cross-check only.
Nothing in the tyre-v3 profile or `/analyze` changes.

## 7. API

`GET /sessions/{source}/{file}/ideal-lap?lap_number=N` (N omitted = fastest complete lap, as `/trackmap`).

    {"status": "ok" | "insufficient_data", "reason": str | null,
     "car_model", "car_class", "track", "lap_number", "lap_time_s",
     "fit": {scope, n_laps, n_files, load_source, confidence, beta, notes[],
             axles: {front: {mu1, k, c, status, terms, v_sat[2], saturation, knee_deg, n_sat, ratio_sd, v3_peak_slip_deg, knee_vs_v3}, rear: {...}},
             s_front, kappa_brake, kappa_trac, drag: {d0, d2, source}, selfcheck: {shift, sd}},
     "ideal":   {time_s, gain_s, gain_min_s, gain_max_s, clamped_share},
     "ceiling": {time_s, gain_s},
     "model_only": {time_s, under_driven_share},
     "trace": {lap_dist[], v_kmh[], v_ref_kmh[], t[], binding[], saturated_ref[], u_front[], u_rear[], u_front_ref[], u_rear_ref[]},
     "corners": [{n, name, s0, s1, t_ref, t_ideal, gain_s, gain_min_s, vmin_ref_kmh, vmin_ideal_kmh,
                  binding, u_front_ref, u_rear_ref, verdict, where, measured_share, extrapolated}],
     "note": "one sentence for the UI: what this is and its range"}

`trace` is given at the same points as `/trackmap`'s `driven_line` for that lap (interpolate the
2 m grid onto the points' lap distance), with `t` = ideal game-clock seconds from the start line,
ending at `ideal.time_s`. The ghost is then the lap's own line with another `t` and speed: the
Compare tab needs no new geometry, and its finish anchor works because `t` ends at the lap time.
Cache the response by (file mtime + size, lap, group fingerprint). Target: under 10 s warm
(4 solves: three levels + ceiling); the builder measures the prototype's solve time first and
vectorises the two passes only if needed.

## 8. Building instructions (Opus 5.5)

Flat modules in `backend/app/`, matching the code base. Port from `il_core.py` / `il_data.py`,
do not import the spike. Pure functions on numpy arrays; no FastAPI in the first four.

| Module | Holds | Interface |
|---|---|---|
| `ideal_state.py` | lap to smoothed state arrays; qualifying laps; per-file reduction cache | `lap_state(lap) -> State`; `qualifying_laps(laps) -> list`; `load_reduction(path) -> FileReduction` |
| `ideal_car.py` | §4: `fit_load_model`, `fit_axle`, `fit_braking`, `fit_traction`, `fit_drag`, `fit_engine`; `CarModel` dataclass (`to_dict` / `from_dict`), `lam()`, `axles()`, `drag()`, `power()`, `scaled(level)` | `fit_group(reductions, car_class, v3_peaks) -> GroupFit` |
| `ideal_solver.py` | §5.1-5.5: `line_of`, `solve(car, line, mode) -> {v, vc, binding, gxt}`, `envelope`, `usage` | no I/O |
| `ideal_store.py` | §6.3-6.4: groups, fingerprints, JSON profile, fallbacks, confidence, β | `car_for_lap(file, lap) -> (CarModel, fit_info) | Refusal` |
| `ideal_lap.py` | orchestration: three levels + ceiling, corners (§5.6), response dict | `build_ideal_lap(path, lap_number) -> dict` |
| `main.py` | the endpoint only | |
| `backend/tools/ideal_lap_report.py` + `IDEAL_LAP_REPORT.ps1` | the validation CLI of §9: `--fit f1,f2 --sim f3 [--holdout odd|file] [--levels] [--json out]`, prints one line per lap with every §9 metric | |

Order (each step ends with its tests green; stop and report if a number in §3 cannot be
reproduced within the tolerance given):
1. `ideal_state` + `ideal_car` load model. Check: Porsche COTA front axle mean `lam` in hard
   cornering, modelled vs measured, within 2 % (was 6-7 %); held-out R² per wheel ≥ 0.93.
2. `ideal_car` rest + `ideal_solver` balanced/ceiling. Check against `fable_p2.py --variants both`
   at q 0.75 through a test-only quantile argument: Porsche L9 balanced 129.90 ± 0.15 s, Ferrari
   L11 115.75 ± 0.15 s. Then switch to the median for good.
3. Envelope, levels, usage, binding codes. Check against `fable_p4.py` (table in F5) ± 0.1 s.
4. `ideal_store` (groups, per-file load model, fallbacks, refusal), ingest pass.
5. `ideal_lap` + endpoint + `ideal_lap_report` tool. Hand over to Sonnet for §9.
6. UI only after §9 passes and the user has answered §12 Q1-Q3.

Tests (`backend/tests/test_ideal_*.py`, pytest, synthetic data only, no recordings):
- T1 load model recovers known coefficients incl. the `|gy|` term (noise 3 %): each within 5 %.
- T2 synthetic car with known `mu1, k, c` per axle, saturated samples flagged by Grip Fraction and,
  separately, by a synthetic slip angle with a knee: `mu1` recovered within 2 % both ways; an axle
  with 100 saturated samples comes back `lower_bound`.
- T3 solver on a circle of constant radius: speed = closed form `√(A·g·R)` within 0.5 %; on a
  straight between two hairpins: braking point and exit match hand integration within 2 m.
- T4 envelope: never below `v_ref`; equals `v_M` where `v_M ≥ v_ref`; `gain_min ≤ gain ≤ gain_max`.
- T5 ceiling ≥ balanced speed everywhere; equal when `A_f = A_r`.
- T6 **perfect driver (rewritten in round 1, §13 R1; the first version put 5 % noise on gy,
  which also bends the line: no recording does that)**. Build telemetry by solving a known car
  on the synthetic line of `tests/synthetic_ideal.py`, where the TRUE capacity of both axles is
  multiplied by a noise field over lap distance: a place part (same on every lap, log sd 0.035,
  correlation length 50 m) times a lap part (new on every lap, log sd 0.035, 50 m); together
  about 5 %, split as F3 found. The perfect driver drives each lap at that lap's capacity. gy,
  speed and the line are exact; Grip Fraction is low (and the slip angle past the knee) where
  the usage against the NOISY capacity is ≥ 0.995. 6 laps, refit, solve on lap 3's line with
  the clamp. Cars A and B of the first version, seeds 2026 / 7 / 99, and once without noise.
  Assert per run: phantom `gain` ≤ 0.5 % of the lap (the §11 trigger); `model_only` (unclamped)
  within ±0.5 %; `kb`, `kt` within 1 % of the true values. Without noise: `gain` ≤ 0.1 %,
  `model_only` within ±0.3 %. Report `gain_min` / `gain` / `gain_max` and `clamped_share` per run;
  targets, not asserts: `gain` ≤ 0.3 %, `gain_min` ≤ 0.15 %. The largest `gain` goes into the
  UI note ("up to X s of this can be model noise").
- T7 refusals: no slip angles and no Grip Fraction; fewer than 3 laps; native file. Each returns
  `insufficient_data` with a reason, never an exception and never a number.
- T8 store: fingerprint change refits; a file with one lap borrows the group load model;
  JSON round trip.
- T9 endpoint on a synthetic session (`tests/synthetic_track.py`): schema, `t` ends at
  `ideal.time_s`, `trace` length equals the `/trackmap` driven line's.

Keep the code base's habits: "PREVIOUSLY ... / Fix:" comments for F1 and F8, a DevNotes entry
under `### 11/10/2026`, HANDOFF §4 gets a short "Ideal lap" subsection when it ships.
Recordings stay read-only; nothing writes into `data/sessions`.

## 9. Validation protocol (Sonnet)

Run with `ideal_lap_report`. A "pair" = (fit set, test laps). Held-out always: the test laps are
never in the fit. Test laps = the 3 fastest qualifying laps of the test set.

Splits
- **odd/even** (one file): fit even laps, test odd.
- **cross-session** (same car, same track, different file): fit on one, test on the other, both ways.
- **cross-track** (car scope): reported, not gating.

Files (names in `data/sessions`; times are UTC, so "11/8" is `2026-11-09T04`)
- Hypercar: Ferrari COTA 10/27 ↔ 10/30 (cross-session); Ferrari Road Atlanta 10/18 (odd/even);
  Genesis Sarthe 10/18 ↔ 11/9; Valkyrie Sarthe 10/21 `03_04_37` (odd/even); BMW M Hybrid Daytona
  10/29 (odd/even); Aston 10/2-10/4 and Cadillac 10/2 Road Atlanta (v4, no Pos Y: grade from
  topography; cross-session).
- LMGT3: Porsche COTA `11-09T04_29_26` (odd/even); Porsche Road Atlanta 10/26 ↔ 10/27; Porsche
  Barcelona 10/18 (#91) ↔ 10/26 `03_24_16` ↔ `03_47_50`; Porsche Fuji 10/19 `22_49_27`, Sebring
  10/15 (odd/even); BMW M4 Fuji 10/14 ↔ 10/15 ↔ 10/19; BMW M4 Imola 10/22 (odd/even); Mustang
  COTA 10/15 (odd/even); Lexus Road Atlanta 9/25-9/26 (expected: `insufficient_data`, which is a pass).
- Cross-track: Porsche COTA ↔ Road Atlanta ↔ Barcelona; BMW M4 Fuji → Imola; Ferrari COTA → Road Atlanta.

Checks (thresholds are from the probes with margin; a fail is a finding, not a reason to tune)

| # | Check | Threshold | Measured in probes |
|---|---|---|---|
| V1 | Load model held-out R² per wheel; axle-mean `lam` bias in hard cornering | ≥ 0.93; ≤ 2 % | 0.93-0.99; was 6-7 % before F1 |
| V2 | Level: median of log(observed / predicted) on the held-out saturated samples | within ±0.02 odd/even, ±0.03 cross-session | 1.8 % Ferrari cross-session |
| V3 | Spread: log sd of the same | ≤ 0.075 | 0.053-0.071 |
| V4 | β: share of pairs with V2 inside the β of §5.4 | ≥ 80 %; else report the p80 as the new β | not measured |
| V5 | Structure: at level × 1.10, plain model under driven by > 2 km/h | on ≤ 4 % of the lap, nowhere > 10 km/h | 1.1-3.4 %, 5-9 km/h |
| V6 | Stability (round 1): UNCLAMPED envelope time over the 3 reference laps of a file (the model must not depend on the lap). The clamped ideal is reported next to it, not gated: it belongs to its lap | spread ≤ 0.5 % of the lap; clamped: report (expected ≤ 0.8 %) | unclamped 0.15 % Porsche, 0.4 % Ferrari; clamped 0.49 % / 0.37 % |
| V7 | Cross-session: ideal time fitted without the test file vs with it | ≤ 0.4 % of the lap apart | 0.1 % Ferrari |
| V8 | Drag sensitivity (no-force cars): `d2` ± 30 % | ideal time moves ≤ 0.2 s | not measured |
| V9 | Plausibility: `gain` for laps within 1 % of the file's best | 0 to 3 % of the lap; `gain_min` > 0 | 1.0-1.6 % |
| V10 | Axles: status per axle, knee vs tyre-v3 peak, share of the lap each binding code holds | report only | Porsche rear knee 2.5-3.0 vs v3 3.9 |
| V11 | Cross-track pairs: V2, V5, V7 | report only (expected to fail on GT3) | Porsche COTA→Road Atlanta: under driven on 63-76 % |
| V12 | Run time per request, warm | ≤ 10 s | 1.5-1.7 s per solve (builder) |
| V13 | (round 1) Clamp and extrapolation: `clamped_share`; gain with and without the clamp; share of `gain` in corners flagged `extrapolated`; `terms` per axle; knee (continuous) per file of a group | report only; come back if more than half of the gain of a class sits in `extrapolated` corners | clamp removes 0.18-0.94 s per COTA lap |

Report per pair one row with V1-V12, then a per-class summary. Gate for the UI step: V1-V3 and
V5-V7 pass on every Hypercar pair and every LMGT3 pair that has ≥ 6 laps; anything else is
listed with its `confidence`. The user's draft acceptance test ("less than 2 % of the lap more
than 0.5 km/h under the driven speed") holds by construction for the envelope (0 %), so it no
longer tests anything; V2, V5, V6 and V7 are what can actually fail (§12 Q9).

## 10. Rule for adding terms later
A new term in the load or grip model (yaw acceleration shifting the axle shares, bank, tyre
temperature, a location map) is kept only if it lowers V3 (held-out log sd) by ≥ 10 % relative on
at least two cars of different classes. Bank and the location map already failed this (F3).
Same spirit as `tire_mf.ENABLED_FACTORS`.

## 11. When to come back to Fable
- V2 or V5 fails on a whole class (the model is structurally wrong there, as F1 was).
- A class shows up whose channels fit neither column of §6.1.
- V6 fails (the ideal depends on which lap gave the line more than 0.5 %): the line handling or
  the envelope is wrong.
- The phantom gain of T6 exceeds 0.5 % of the lap. (Round 1: it did, 0.26-0.72 %; answered in
  §13 with the saturation clamp and a rewritten T6. The trigger stands for the new T6; the next
  candidate would be the same clamp on braking and traction zones.)
Not for: thresholds missed by a little on one file, performance, UI layout, bugs.

## 13. Build round 1 decisions (11/12/2026)

Answers to `.superpowers/sdd/2026-11-11-ideal-lap/batch-a-report.md` (Q-A, Q-B, Q-C). All
decisions here are **[Fable]**. The sections they change (§4.2, §5.2, §5.4, §5.6, §7, §8 T6,
§9, §11) are edited in place and marked "(round 1)". Probe: `fable_p5.py` (spike folder of the
worktree; env `FABLE_GATE=latdom` is the rule adopted). Car in the probe = spike variant "both"
at the median, same-file fit, COTA Porsche `11-09T04_29_26` and Ferrari `10-30`.

### R1. Q-A: the envelope design changes (saturation clamp); T6's noise moves to the capacity

The builder is right twice: T6 put the noise on the wrong signal, and the bias is real anyway.
- Wrong signal: F3's 6 % is the limit moving (place, state, transients), not an error of gy.
  gy is kinematic and exact (`G Force Lat` = −0.98·v·yaw rate), and it builds the line. Noise on
  gy bends the line, which no recording does.
- Real anyway: at a saturated sample in pure cornering `v_M / v_ref = √(A / |gy|)`, which is
  F3's residual itself. So wherever the driver is at the limit, the model beats him at about
  half of the points, and the envelope keeps exactly those.

Measured on real laps (`fable_p5.py`, design fit order, envelope gain at level 0.98 / 1.00 / 1.02):

| Lap | measured saturated | unclamped envelope | of it ON saturated points | with the clamp |
|---|---|---|---|---|
| Porsche L9 131.86 | 27.6 % of lap | +1.00 / +1.42 / +1.97 s | 0.28 s | +0.76 / +0.99 / +1.24 s |
| Porsche L4 132.05 | 25.4 % | +1.06 / +1.45 / +1.88 | 0.10 | +0.99 / +1.27 / +1.56 |
| Porsche L6 132.70 | 27.2 % | +1.74 / +2.22 / +2.75 | 0.45 | +1.06 / +1.28 / +1.53 |
| Ferrari L11 117.85 | 25.9 % | +1.01 / +1.43 / +1.95 | 0.20 | +0.65 / +0.85 / +1.06 |
| Ferrari L3 117.97 | 23.3 % | +1.18 / +1.56 / +2.07 | 0.15 | +1.00 / +1.25 / +1.52 |
| Ferrari L8 118.06 | 23.8 % | +0.80 / +1.19 / +1.69 | 0.16 | +0.65 / +0.91 / +1.14 |

0.10-0.45 s of the "expected" gain (0.08-0.34 % of the lap) sat on points where the tyres were
measured at their limit: that part is model noise by definition, and it is the size T6 reported.
With its knock-on into the braking and exit either side, the clamp removes 0.18-0.94 s.

**Decision: where the reference lap's tyres were measured saturated, the corner limit is the
speed that was driven there, not the model's (§5.4, "saturation clamp").** The measurement
replaces the model where there is one; the model speaks only where the driver was not at the
limit, which is the question the user is asking. Side effects, all wanted:
- sensitivity to the grip level halves again: ±0.25 s for ±2 % (was ±0.45 s; plain model ±1.0 s);
- the knee question (Q-C) stops mattering for the headline: knee 4.0 vs 3.0 deg moves the clamped
  gain by 0.00 / 0.01 / 0.03 s on the three Porsche laps (unclamped: 0.06-0.10 s; plain: 0.22-0.27 s);
- "at least" becomes a defensible name for `gain_min`.
Cost: the ideal now depends more on the reference lap (a lap saturated at a lower speed in a
corner is held to that speed there). Clamped ideal over the three reference laps: Porsche
130.87 / 130.78 / 131.42 (0.49 %), Ferrari 117.00 / 116.72 / 117.15 (0.37 %); unclamped it was
0.15 % / 0.4 %. That is consistent with §12 Q2 ("the ideal lap belongs to that lap"); V6 is
re-defined (§9).
The headline numbers get smaller (Porsche L9: "at least −0.76 s, expected −0.99 s" instead of
−1.00 / −1.42), and the definition gains a clause: **new user question §12 Q10**.

Not clamped in this round: braking and traction (ABS / TC active zones). T6 (capacity noise)
measures what is left; if the phantom is still above 0.5 % of the lap, come back (§11).

### R2. Q-B: speed and load terms only where the saturated set spans them

Real sets are narrow too: Porsche front saturated at 21-44 m/s (p10-p90), rear 20-40; Ferrari
front 22-50, rear 20-39 (median 25, n 264, fitted c 0.194). Above about 50 m/s every capacity
in the solver is an extrapolation of `c`.
- An axle's `k` and `c` are **fitted** only if its saturated set has ≥ 250 samples, a speed span
  (p90 − p10) ≥ 15 m/s and a modelled-load span (p90 − p10 of the axle-mean lam) ≥ 0.15.
- Otherwise **borrowed** from the other axle if that one is fitted (then only `mu1` is fitted,
  as `lower_bound` already does), else **default** `k = 0.03, c = 0.10` (middle of what has been
  fitted so far: k 0-0.08, c 0.05-0.19), confidence `low`. This also covers the builder's
  uncovered case "both axles lower_bound".
- Store per axle `terms: fitted | borrowed | default` and `v_sat: [p02, p98]` of the set.
- A corner whose ideal minimum speed is above the binding axle's `v_sat[1] + 5 m/s` gets
  `extrapolated: true` in the corners table (§5.6); the validator reports how much of the gain
  sits in such corners (V13). Totals are not changed by the flag in this round.

### R3. Q-C: keep the design order, drop F5 as a 0.1 s reference, make the knee continuous

- The order stays: drag is part of what the tyres must supply, so `gxt` with drag is the right
  gate. The spike's order was an accident of the dropped joint fit.
- The knee jump is real, not a re-parametrisation: front capacity at three reference states moved
  +3.9 / +4.5 / +4.8 % (1.232 → 1.280 at 30 m/s, lam 1.05; 1.359 → 1.420 at 45, 1.2; 1.527 → 1.601
  at 60, 1.4) for a one-bin jump of the knee. So a knee-based level is good to about ±4 %:
  **β = 0.03 at least for every knee-based (no-force) fit** (§5.4), and the knee rule becomes
  continuous (§4.2) so that it cannot jump a whole bin on a flat plateau.
- F5 was measured with the spike order and without the clamp; it is history. The step-3
  reference is now the builder's own numbers after this fix round (table to be added below by
  the builder, "R5"), checked against R1's clamped column within ±0.15 s as a sanity bound only
  (the probe clamps with `min(vc, v_ref)`, the design with `vc = v_ref`, so the app may read a
  little higher).

### R4. Smaller items from the report
- `vc` bisection: accepted, it is the more exact solver. §9 thresholds stay.
- Demand grid (kb / kt read 1.5-2.5 % low): refine the strongest feasible braking and drive
  demand by bisection between the last feasible grid value and the next (4 halvings, 0.0016 g).
  T6 then expects kb, kt within 1 % of the true values.
- `S > 0.05` guard in braking rule 3: accepted.
- `load_reduction` choosing the topography from a possibly partial `laps[0]`: fix in step 4 as
  the builder proposes, and run the npz round trip once on a real file there.
- 180-260 km/h load bias of +3.0 %: noted, inside the clamp's reach where it matters; V1 reports
  it per speed band from now on.

### Builder's fix round (in this order)
1. Knee rule and term identifiability (§4.2, R2, R3). Tests: T2 gains a case "axle saturated only
   at 20-30 m/s gets the other axle's k, c" and a case "flat plateau: the knee moves by less than
   0.3 deg when 5 % of the samples are removed at random".
2. Demand bisection (R4).
3. Saturation clamp (§5.4): `saturated_ref(state, fit) -> bool[]`, `line_of(..., sat_ref)`,
   `solve` honours `line["v_clamp"]`, binding code `measured`. T4 gains: clamped points have
   `v_I == v_ref`; a lap with no saturated point gives the old envelope.
4. T6 as rewritten in §8; remove the xfail. Report the four numbers per seed.
5. Re-run `build_check_step3.py` with the clamp on and add the table "R5" here.
Step 4 of §8 (store, ingest, `load_reduction`) does not touch any of this and can run in
parallel. Step 5 (`ideal_lap`, endpoint, report tool) consumes the clamp and the new fields:
start it after item 3 above.

## 12. Questions for the user (most important first; my recommendation after each)

**Answered 11/11/2026: the user accepted every recommendation below (Q1-Q9).** The build
follows them; step 6 (UI) still waits for §9 to pass.

1. **What is "the ideal lap" you see?** Your words ("every corner uses the maximum grip of each
   tyre") describe the ceiling: all four tyres at their limit at once. The data says this car
   cannot do that: the front carries 47 % of the cornering force at every speed, so one axle runs
   out first. *Recommended:* the ghost and the headline are the **reachable** lap (limiting axle at
   its limit, never slower than your own lap), shown as "at least −0.9 s, expected −1.3 s"; the
   ceiling is one extra number, "setup potential: up to −1.7 s".
2. **Which line?** *Recommended:* the line of the lap you have selected (default: your fastest);
   the ideal lap belongs to that lap. Later, cheaply: run it on your 3 fastest laps and say which
   line is worth most.
3. **Where in the app?** *Recommended:* the Compare tab, a button "Ideal lap (this car, this line)"
   that adds it as a ghost (speed trace, live delta, chase-view ghost for free), plus a per-corner
   table: time available (at least / expected), what limits there (front / rear / brakes /
   traction / power), front and rear grip used by you.
4. **A car on a new track:** the model does not carry between tracks (Porsche COTA to Road
   Atlanta is off by 10 % in grip). *Recommended:* no ideal lap until there are 3 clean laps on
   that track; before that, an estimate from other tracks clearly marked "low confidence".
5. **Per tyre or per axle?** The model knows each tyre's load and capacity, but how the two tyres
   of an axle share the force is not modelled. *Recommended:* grip used per axle in v1, with
   each tyre's load share shown; true per-tyre usage for Hypercars (the game records it) later.
6. **Show the range?** *Recommended:* yes, always "at least / expected"; a bare number would be
   wrong by about ±0.5 s on a 2-minute lap.
7. **Give it to the LLM coach** (per-corner time available and the limiting axle in the report)?
   *Recommended:* yes, as a second step once the numbers have passed validation.
8. **LMP2 / LMP3:** nothing recorded yet. *Recommended:* nothing to build; they work once
   recorded, like a GT3 or like a Hypercar depending on which channels the game fills.
9. **The acceptance test:** "never meaningfully slower than a real lap" is now true by
   construction, so it cannot catch a bad model. *Recommended:* accept the checks of §9 instead
   (level, structure, stability across laps and sessions).

**Open, added 11/12/2026 (build round 1, §13 R1). Not yet answered; the build assumes the
recommendation.**

10. **Where your tyres were already at their limit, should the ideal lap simply equal your own
    speed there?** The first build showed that the tyre model is only good to about ±6 %, so in
    corners where you were at the limit it "found" time about half of the time, and that time
    was noise (0.1-0.45 s per COTA lap directly, up to 0.9 s with the braking and exit around
    it). *Recommended:* yes. The game tells us when a tyre is saturated (Grip Fraction, or the
    slip angle past its peak); there, the ideal lap uses the speed you drove, and the model only
    speaks where you were not at the limit. The numbers get smaller and more trustworthy:
    Porsche COTA best lap "at least −0.76 s, expected −0.99 s" instead of −1.00 / −1.42 s;
    Ferrari "at least −0.65 s, expected −0.85 s" instead of −1.01 / −1.43 s. Consequence to
    know: the ideal lap is tied more closely to the lap it is built from (a lap where you slid
    through a corner slowly is held to that speed in that corner), so the three best Porsche
    laps give ideals within 0.5 % of each other instead of 0.15 %.
    **Answered 11/12/2026: yes**, with the user's note: "this is multivariable. Even if speed was
    the same, if loading of each axles or setup (spring rate, ARB, etc) differ then speed wouldn't
    mean much. How to get around this? A good resource is also
    https://docs.google.com/spreadsheets/d/e/2PACX-1vTN03UvJDm99byA6vQPZHKOCYVvfxLu1zkJAzdaKyROykzEKY2-Xl1rl1q5znZEf36m88dxMKsY2eaO/pubhtml,
    where public laptimes are held." (Open, for the design author.)
