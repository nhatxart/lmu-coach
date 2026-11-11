# Ideal lap from tyre and load data: strategy (11/10/2026)

Status: strategy and a validated-in-part prototype. Not designed for the app yet,
nothing built in the app. Follows `2026-11-09-theoretical-best-lap-feasibility.md`.

## What the user asked for
"An algorithm that can take in analyzed class/car data and return an ideal lap",
for **all classes**, built from tyre data so that every corner uses the maximum
grip of each tyre, combined with tyre load and suspension load.

## Model roles (user, 11/10/2026)
Fable 5.2: questions and building instructions; Opus 5.5: building and
implementing; Sonnet: validation; Opus: review. Loop back to Fable only for
conceptual problems, not for debugging.

## Prototype
`docs/superpowers/spikes/ideal-lap/` (throwaway, read-only on recordings; run from
that folder with `uv run --no-project --with duckdb --with numpy --with scipy python <script> ...`;
paths default to this repo's `backend`, `data/sessions`, `data/track_reference`,
override with env `IL_BACKEND`, `IL_SESSIONS`, `IL_REF`; per-file cache in `%TEMP%\il_cache`).
- `il_data.py`: loads a recording at 50 Hz into per-lap arrays (also `Tire Longitudinal Force`,
  `Grip Fraction`, which the backend does not load today), `real_laps` = within 5 % of the best.
- `il_core.py`: `fit_car(files) -> Car`, `line_of(lap)`, `solve(car, line, mode)`, `compare(line, sol)`.
- `s3_run.py`: fit on file(s), solve on the best laps, print the validation line.
  `python s3_run.py [-q 0.75] [--laps N] fitfile[,fitfile2] [simfile]`.
- `s1_*`, `s2_explore.py`, `s4_speed.py`, `s5_mu.py`: the measurements below.

## Measured facts (COTA: Ferrari 499P 10/30, 6 laps; Porsche 911 GT3 R 11/8, 20 laps)
Channel conventions (both cars):
- `G Force Long` is **kinematic**: d(speed)/dt matches it (corr 0.9995, slope 0.995) and the
  hill does not appear in it. So the tyres' longitudinal force, in g, is
  `gxt = gx + grade + drag(v)` (our `gx` = −`G Force Long`, + = speeding up).
- `G Force Vert` is the path's vertical acceleration: it matches v·dθ/dt of the height
  profile (slope 0.967, corr 0.91). Crest negative, compression positive.
- `G Force Lat` = −0.98·v·yaw rate/g: kinematic too.
- `Pos X/Y/Z` speed equals `Ground Speed` (ratio 1.0000); grade from `Pos Y` is −12..+14 % at COTA.

Loads (`lam` = wheel load / its static value):
- `load = c0 + c1·v² + c2·gx + c3·gy + c4·gz` per wheel, signals smoothed 0.2 s, fitted on
  even laps, scored on odd laps: Ferrari Tire Load R² 0.971–0.991 (4–6 % rmse), Susp Force
  0.977–0.992; Porsche Susp Force 0.930–0.976 (5–10 %). Extra terms (gx·v², |gy|·v², gy·gx, gy²)
  add at most 0.02 R². Without gz it was 0.64–0.87.
- Susp Force is not wheel load (Ferrari: 2,342 kg static vs 1,219 kg from Tire Load), so the
  model uses only ratios to the static value; absolute load is never needed.
- Ferrari aero: front share of downforce 0.46; lam at 200 km/h front 1.63, rear 1.56.
  Porsche: 0.27; front 1.21, rear 1.52.

Tyres (Ferrari, from the game's forces):
- Sum of lateral forces vs lateral G: corr 0.9998, mass 1,156 kg. Sum of long forces vs gx: corr 0.995.
- Drag = 615 N + 0.451·v² N (0.25 g at 250 km/h), residual sd 0.038 g.
- **Grip Fraction is LOW when the tyre is saturated** (mu 1.40–1.55 at < 0.25, about 0.5 above 0.8).
- Saturated lateral mu falls with **speed** much more than with load: per tyre
  mu ≈ 1.44 − 0.02..0.05·(lam−1) − 0.11..0.28·(v−40 m/s)/40. Load alone explains less.
- Saturated mu by direction: lateral 1.40–1.48, braking 1.14–1.26, traction (rear) 1.50–1.56.
- Quasi-steady corners: front share of lateral force 0.474 (static weight 0.433).
- Braking: front share of brake force 0.54 (= front share of load then, 0.55).
- Hybrid: front share of drive force at full throttle about 0.41 above 190 km/h, about 0 below.
- Brake > 60 %: 57 s over 13 laps, ABS never active; TC active 73 s.

Without forces (GT3), the axle proxy r = |gy| / mean lam of the axle against the axle's
mean slip angle rises and then flattens: Porsche front flat from about 4°, rear from about 3°.
On the Ferrari the same proxy finds the same level as the true axle mu (front 1.39–1.41 vs 1.39–1.44).

## The algorithm (prototype, `il_core.py`)
Car model (fitted per car model from its recordings, all JSON-able numbers):
1. Load ratios per wheel: `lam_i = 1 + a_i·v² + b_i·gx + c_i·gy + d_i·gz` (from Tire Load, else Susp Force).
2. Lateral capacity per axle, in g: `A_axle = ½ Σ_tyres mu1·(1 − k·(lam−1) − c·(v−40)/40)·lam`,
   `k ∈ [0, 0.15]`, `c ∈ [0, 0.45]`, fitted on saturated samples (Grip Fraction < 0.3 when recorded,
   else the slip-angle knee + 4°), then scaled to the q-quantile (0.75) of the observed/predicted ratio.
3. Axle share `s_front`: from lateral forces when recorded, else static weight distribution.
4. Braking: `|gxt| ≤ kb·(s_f·√(Af²−gy²) + s_r·√(Ar²−gy²))`; kb from samples at the limit
   (front grip fraction < 0.3, else ABS active, else brake > 90 %).
5. Traction: `gxt ≤ kt·s_r·√(Ar²−gy²)`; kt from TC active (else full throttle < 144 km/h).
6. Drag: from forces; without forces jointly with kb, bounded by coasting deceleration.
7. Engine: full-throttle flat-road acceleration by speed (q-quantile), minus a cornering cost `cc·gy²`.

Solver (quasi-steady, fixed driven line, 2 m steps): curvature = gy/v², vertical curvature = gz/v²,
grade from `Pos Y`; corner limit = highest speed at which some longitudinal demand is feasible;
forward pass (acceleration = min(engine, traction/grip)), backward pass (braking), twice round the
closed lap. Two modes: **balanced** (|gy| ≤ Af and ≤ Ar) and **ceiling** (|gy| ≤ s_f·Af + s_r·Ar).

## Results so far (best two laps of each file, fitted on the same file)

| Car | Lap | Driven | Ceiling | Balanced | Balanced slower than driven on | worst |
|---|---|---|---|---|---|---|
| Ferrari 499P | L11 | 117.85 | 115.12 (−2.73) | 116.78 (−1.07) | 14.0 % of lap | 13 km/h |
| Ferrari 499P | L3 | 117.97 | 114.98 (−3.00) | 116.51 (−1.47) | 11.0 % | 10 km/h |
| Porsche GT3 R | L9 | 131.86 | 128.38 (−3.50) | 132.15 (+0.27) | 23.6 % | 27 km/h |
| Porsche GT3 R | L4 | 132.05 | 128.92 (−3.09) | 132.61 (+0.60) | 32.5 % | 25 km/h |

Ceiling mode is slower than driven on only 2.6–6.6 % of the lap, worst 9–11 km/h.

## Acceptance test
An ideal lap must not be meaningfully slower than any real lap on the same line:
target less than 2 % of the lap more than 0.5 km/h under the driven speed, none more than 5 km/h.
Also: fitted on some files, tested on others (held out); every class and track with data.

## Open problems (conceptual: for Fable)
1. **Balanced mode under-predicts**, worst on GT3 in the COTA esses (1,000–1,500 m) and where the
   driver is at full throttle while cornering. Candidates: the axle share is fixed (a real car's
   balance moves with speed, aero, throttle and steering); the knee-based saturation set for GT3
   catches the rear only in 612 samples; transient direction changes are not quasi-steady.
2. **Ceiling vs balanced**: which one is "the ideal lap" the user sees, or both, or a balance that
   the car has been seen to reach (a measured balance envelope by speed)?
3. **GT3 without forces**: the rear knee, braking from ABS (7 s on the Porsche), drag from coasting
   (the joint fit returned zero drag): how robust per car model and with few laps?
4. **Line**: the driven line of which lap; the ideal lap keeps it.
5. Where it lives in the app (Compare ghost, per-corner table, per-tyre "grip used").
