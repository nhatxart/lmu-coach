# Tyre model v3 — spec

Date: 2026-10-19. Owner: PM/CTO (Opus). Scope: plan steps 1-6 from the 10/19 review. Live predictor is OUT of scope.

## Problem (from DevNotes 10/18 scorecard)
- v2 is a binned slip-angle curve per car. Slip angle explains 0-34 % of sample-level grip; the rest is load, temperature, pressure, wear, camber, combined slip.
- Peaks are mostly never observed; confidence `1-exp(-n/4000)` is overstated (WRT 0.979 with an unobserved peak, 90 % CI 3.8-5.8 deg).
- Profiles are keyed by livery ("..._LM" vs "..._WEC" are separate); no class, no game version; raw samples capped FIFO at 20k/axle.
- Hypercars expose real tyre forces (Tire Load, Tire Lateral Force); LMGT3 do not (all zero) -> GT3 must run on the lat-G / load-ratio proxy.

## Goals
1. **Data hygiene.** Recorder stores vehicle class, game version, track/ambient temp. Backend reads Camber, TyresWear, TyresCarcassTemp, Surface Type, Tire Lateral Force. Car identity (model, class) resolved for every file, incl. old livery-only files. Kerb/off-track samples excluded.
2. **Honest confidence** from bootstrap over laps + whether the peak was observed.
3. **Factorized Magic Formula** per axle: `mu = D*sin(C*atan(B*a - E*(B*a - atan(B*a))))`, with D and B scaled by state factors (load ratio, temperature vs optimal, pressure residual given temperature, wear, camber). Factors that the data cannot identify are fixed neutral and reported as such.
4. **Hypercar experiment**: which factors improve held-out prediction of the game's TRUE mu (sum|Fy|/sum Fz). Only factors that earn their place go to production.
5. **Class pooling**: class-level prior + per-car fit pulled toward it (penalized least squares = approximate partial pooling). GT3 collapse check tool.
6. **Binned per-lap sufficient statistics** replace the raw FIFO store; doubles as a coverage map.
7. **Production v3** in `tire_model.py` / `grip_profile.py` / `/analyze` / coach prompt, keeping every v2 output key so UI/prompt keep working.

## Decisions (binding)
- Fitting: scipy `least_squares` (no NumPyro/JAX). Accuracy via per-cell state means (not bin centres), soft_l1 loss, count weights, bootstrap over LAPS.
- Profile key = car MODEL (e.g. "BMW M4 LMGT3"), not livery. Class from `VehicleClass` metadata, else `VehicleModel` suffix, else a livery table.
- Game version: laps tagged; production fit uses the newest version with >= 3 laps; older-version fit of the same car (weight halved) is the prior, else the class prior. "unknown" sorts oldest.
- Steady-state filter stays for the MF fit (|long G| < 0.25 g, |yaw accel| < 1 rad/s^2, > 40 km/h). Long G is stored per cell for later combined-slip work but not modelled now.
- Fits are computed at ingest and cached in the profile (fingerprint of lap keys + prior); `compute_baselines()` never fits.
- Runtime budget: refit of the largest profile (WRT, ~25 laps, both axles, 50 bootstrap resamples) <= 10 s on the user's PC.
- When MF fails (too few cells, no convergence, degenerate slip range) the axle falls back to the v2 binned curve and says `"model": "binned_v2_fallback"`. `/analyze` never raises because of the tyre model.
- `data/sessions/*.duckdb` read-only, always. Profiles are backed up before any write by a tool.

## Non-goals
Live prediction; combined-slip (friction ellipse) modelling; thermal dynamics; per-wheel (vs per-axle) models; NumPyro; changing `recorder.py` (non-HF).
