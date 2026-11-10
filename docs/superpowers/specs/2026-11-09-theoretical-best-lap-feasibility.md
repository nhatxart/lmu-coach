# Theoretical best lap from tyre and load data: feasibility (11/9/2026)

Status: feasibility check only. No design approved, nothing built in the app.
Throwaway script: `docs/superpowers/spikes/2026-11-09-theoretical-lap-feasibility.py`
(`uv run --no-project --with duckdb --with numpy --with scipy python <script> backend <file.duckdb>`; read-only).

## What the user asked for
A simulated theoretical best lap built purely from tyre data: every corner uses
the maximum grip of each tyre, combined with tyre load and suspension load.

## Data used
COTA, both at 50 Hz, laps within 5 % of the file's best:
- Hypercar: Ferrari 499P, 10/30 file, 6 laps, best 1:57.851.
- LMGT3: Porsche 911 GT3 R, 11/8 file, 20 laps, best 2:11.860.

## Findings (measured)
1. **Wheel load is predictable from the lap.** `load = c0 + c1 v² + c2 longG + c3 latG + c4 vertG + c5 vertG·v²`,
   signals smoothed 0.2 s, fitted on even laps and scored on odd laps:
   Ferrari Tire Load R² 0.971-0.991 (rmse 4-6 % of mean); Porsche Susp Force R² 0.930-0.976 (5-10 %).
   Without the vertical-G terms it was R² 0.64-0.87 (about 20 %): COTA's crests and compressions
   (vertical G -0.37..+0.42 on the lap) matter. The track topography (`*.topography.json`) can supply this.
2. **The game's tyre forces are consistent.** Ferrari: the four lateral forces against lateral G give
   correlation 1.000 and a mass of 1,156 kg (static loads: 1,219 kg).
3. **Grip falls with load on every tyre** (p98 of |Fy|/Fz in the loaded half): fronts about 1.52 -> 1.40
   over 5-7 kN, rears about 1.50 -> 1.15-1.25 over 6-8 kN.
4. **Suspension force is not wheel load.** On the Ferrari, Susp Force sums to 2,342 kg static against 1,219 kg
   from Tire Load, and the front share differs (0.369 vs 0.432). The Porsche's Susp Force gives 1,795 kg.
   So a GT3's absolute loads need a per-axle scale from somewhere else (car mass and weight distribution).
5. **Braking data is thin on a Hypercar file** (26.5 s of brake > 80 % in 6 laps, nothing below 160 km/h);
   plenty on the GT3 file (89 s). `Tire Longitudinal Force` is not among the loaded channels (check `telemetry.py`).
6. **Prototype lap (point mass on the best lap's own line, curvature = yaw rate / speed, 2 m steps):**

   | Model | Ferrari (driven 117.86 s) | Porsche (driven 131.88 s) |
   |---|---|---|
   | Proven grip (p95 G by speed, from the laps) | 114.67 s (-3.19 s) | 128.89 s (-3.00 s) |
   | Per-tyre grip, axles balanced, flat | 117.60 s (-0.26 s) | not possible (no tyre forces) |
   | Per-tyre grip, axles balanced, with hills | 118.18 s (+0.31 s) | |
   | Per-tyre grip, all four at max, with hills | 116.42 s (-1.44 s) | |

   The per-tyre models land within about 1 % of the driven lap, but they are SLOWER than the driver on
   28-41 % of the lap (median 186 km/h, 1.3 g lateral demand, mostly not braking). So they under-estimate
   grip in fast combined corner-exit zones. A correct limit model may never be slower than a driven lap:
   that is the calibration test.

## What the prototype gets wrong (estimates of cause, not measured)
- Friction ellipse and one traction limit (1.0 g) for the whole speed range: with downforce the real
  traction limit grows with speed.
- Peak mu from a p98 of all samples, not from samples at the slip peak (the tyre-model v3 fit knows the peak).
- Axle balance from the static front share only; the aero balance moves it with speed.
- No banking; vertical G taken from one lap instead of the topography.
- Point mass: no yaw inertia, so direction changes (the esses, 1,500 m) are too easy.
- The "proven grip" envelope is too generous (3 s): a p95 by speed bin mixes compressions and banking into flat corners.

## Verdict
- **Hypercar: feasible.** Loads, forces and load sensitivity are all there. Expect about 1 % (roughly 1 s)
  after calibration, to be shown with a confidence note.
- **LMGT3: feasible only as an estimate.** No tyre forces and no absolute loads. Options: anchor total grip
  to what the car has shown and use Susp Force only for how load is shared between the tyres.
- The line is the driven line; optimising the line is a separate, much larger project.

## Open design questions (for brainstorming with the user)
- Which lap supplies the line: the selected lap, or each kept/best lap?
- Show both numbers (all four at max = ceiling; balanced = reachable) or one?
- Where it lives: a ghost in the Compare tab (speed trace, time delta, chase ghost) seems the natural home.
- Per-tyre "grip used" readout per corner.
- GT3: accept an estimate with a label, or Hypercar only at first?
