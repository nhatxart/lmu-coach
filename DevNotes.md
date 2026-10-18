# Dev Notes LMU-Coach

### 9/18/2026
- LMU and pyLMUSharedMemory work
- Recorder was able to create DuckDB that contains good data
    - Zeros tables: grip fraction, lap, tire lateral force, tire load, tire longitudinal force
- Error when loading into https://lamerman.github.io/telemetry-visualizer-for-lmu/:
    - Catalog Error: Table with name eventsList does not exist! Did you mean "channelsList"? LINE 1: SELECT eventName, unit FROM "telemetry"."main"."eventsList" ORDER BY eventName ^

### 9/19/2026
- Added track map and car position based on GPS coordinates - track edges extracted using LMU's scoring system.
- Added scrubber for playback with moving window of brake, throttle, and steering input traces.
    - Also displays live speed, yaw rate, and front/rear suspension load (in mm, will have to clarify unit or data meaning)
- Need to add moving dot to yaw rate and suspension delta in accordance to scrubber on the right side. Other static data should be for statistical analysis (mean, st.dev, range)
- Car balance, yaw rate, and lateral G need more aggressive, non-linear color scaling (mostly gray and only show color when 25% of max value reached to emphasize visuals)
- IMPORTANT: track map is mirrored. Need to flip again (include images).
- Make zoom focal point the cursor, bound the generated track map in a box where the width and height are 10% bigger than the size of the track, but keep the bound box in the middle of the display panel. Also, add a "zoom to fit" button (home icon) below the zoom bar for resetting.
- Make UI and text 30% bigger for readability (rough estimate)
### 10/12/2026
- Moved development from the claude.ai chat to Claude Code (CLAUDE.md, docs/HANDOFF.md, START_CLAUDE.ps1).
- FINDING: Tire Load / Tire Lateral Force / Tire Longitudinal Force / Grip Fraction ARE populated in the six newest recordings (10/2-10/4, Aston #007 and Cadillac #12; load up to ~15-21 kN). Still empty in every earlier file (#78, #88, #92, #61). Not yet known whether it is the car class or a game update - a GT3 session recorded now would settle it.
- FINDING: recorder.py's game clock only ever steps in multiples of 0.01 s. On the 10/4 Aston LM file ~16% of samples repeat the previous tick and ~20% skip one (avg 96 Hz), and lap times are quantised to 0.01 s.
- Added recorder\recorder_hf.py (experimental clone; recorder.py untouched) + START_RECORDER_HF.ps1 + test_recorder_hf.py:
    - polls at 1000 Hz, stores one sample per game tick, measures what the game really publishes
    - `.\START_RECORDER_HF.ps1 --probe 10` while driving answers the 1000 Hz question without recording
    - extra channels for a force-based tyre model (local velocity, susp force, camber, toe, patch/ground velocities, tyre deflection, surface type, wear, carcass + 3-zone temps, downforce, ride heights)
    - Lap Time from LMU's own mLastLapTime (1 ms) when it agrees with the clock; one row per completed lap. The unfinished final lap stays in the Lap table so the last completed lap has an end boundary
    - 500 Hz and above keeps one CPU core fully busy (measured 99% at 1000 Hz, 19% at 100 Hz). Use `--poll-hz 200` for everyday recording
    - safety copies are `.spool_hf`, converted with `recorder_hf.py --recover` (never with recorder.py)
    - NOT yet run in-game. Tested with fakes only (22 tests).
- /trackmap now returns per-wheel signed slip angle (`slip_angle`). Sign measured on 3 files: right-hand corners are negative (tyre travels left of where it points).
- Tyre gauges: ring is now a slip-angle direction dial (10 deg slip = quarter turn), bar is slip-ratio magnitude. Suspension bar removed (still in F/R Susp readouts, Axle Balance graph, Corner Suspension card).
- Chevron is interpolated between track points (a lap is only ~700 points because position is 10 Hz). Hold left mouse on the map to pause playback, release to resume.
- TODO: the F Susp HUD readout shows FL only, while R Susp averages RL+RR.
- TODO: /laps returns 500 on a file with no `metadata` table (Imola #92, 9/28). Several older files list 0 laps.
- TODO: backend does not read the HF channels yet; body slip could come straight from Local Vel X/Z. Tyre model redesign (Pacejka fit, load sensitivity, friction ellipse) needs its own spec.
- TODO: recorder.py drops the last completed lap's Lap Time row at save (fixed in the HF clone only).

### 10/14/2026
- First in-game HF recording (Fuji, WRT #69 BMW M4 LMGT3, 19 laps, 208,566 ticks, 99 Hz published at 1000 Hz polling). Saved and committed fine.
- ROOT CAUSE (map/telemetry misalignment, wrong S/F label, tangent chevron): the game clock skipped ~18 s of ticks in lap 0 (1,792 steps of 20 ms). 100 Hz rows are one per tick, 10 Hz rows are gated by game time, and the backend mapped one onto the other by index fraction. Measured: /trackmap lap start was 167 rows (16.7 s) early on lap 1, 78 rows (7.8 s) on lap 11, 20 rows on lap 17. Body slip was refused on every lap for the same reason.
    - Fix: telemetry.session_time_axes - 100 Hz time = GPS Time, 10 Hz time = recorder_hf's gate replayed over GPS Time (reproduces all 21,036 rows exactly). Used by /trackmap (lap cut, resampling, slip angles, body slip) and by telemetry.load_telemetry_file (coach/grip analysis).
    - After: lap start within 0-14 m of the S/F reference on laps 2/6/11/17; body slip back on every lap.
    - Also fixed by this: recorder v4 files that start GPS Time at the session clock (e.g. 1041.4 s) were sliced outside the data in telemetry.py, which assumed row N = N / Hz from 0.
- Body slip now comes from Local Vel X/Z when present: beta = atan2(vx, -vz). Agrees with the yaw-integral method at r = 0.90-0.94, 0.3 deg RMS (laps 2, 6, 11). Yaw-integral stays as the fallback.
- Lap times: the backend now shows LMU's official 1 ms time (Lap Time table) when it agrees with the clock difference within 50 ms. Lap Time rows are matched by timestamp now, which also fixes recorder.py files (placeholder row -> off-by-one "duration mismatch" warnings).
- Only 10 of 18 laps got an official time (missing: 1, 6, 10, 11, 13, 14, 16). Cause unknown; recorder_hf now logs why for each lap (unchanged / arrived late / disagreed). Check the log of the next session.
- US/Neu/OS bars: now front-minus-rear tyre slip angle INTO the turn (sign from yaw rate), cornering only (>40 km/h, |yaw|>0.05 rad/s, |lat G|>0.3 g, at least one axle slipping the way the car rotates), neutral band +-0.25 deg. Chevron colour uses the same delta and band. Previously the bars were the sign of the susp F-R delta over the whole lap (rake/pitch). Fuji lap 11: US 64 / Neu 21 / OS 15 % of cornering samples (68 % of the lap).
- Scrubbing while playing now holds playback until release (not verified with a real drag).
- TODO: the unfinished final lap is listed with its partial time (L18 1:41.030 on this file) and gets the BEST badge / is picked as "fastest" by /analyze.
- Guard for the time-axis fix: backend/tests/test_hf_recorder_file.py runs recorder_hf's real 10 Hz gate over skipped/odd/paused ticks and requires the backend replay to pick the same rows (fails if either side changes). For HF files whose 10 Hz rows don't replay, telemetry.py now prints a WARNING instead of drifting silently.
- Tyre model v2 (tire_model.py, grip_profile.py, telemetry.py, coach.py prompt). Checked on the 10/14 Fuji WRT #69 file (8 laps):
    - /analyze now folds EVERY lap of the file within 115 % of its fastest into the tyre model, once per lap (`tire_model_laps` keys "<file>#<lap>"). Before: one lap per /analyze.
    - Steady-state samples only (|long G| < 0.25 g, |Yaw Accel| < 1 rad/s^2): 4,888 of ~16.5k samples per axle over 8 laps.
    - Load: grip is now mu = |lat G| / load_ratio, load_ratio = axle load / the file's low-speed straight-line load. Sources: Tire Load > Susp Force (mSuspForce, pushrod N, recorder_hf only) > Susp Pos via a per-car deflection->force fit (front r 0.93, rear 0.94 on Fuji) > none (falls back to raw lat G).
    - Flying laps at Fuji have NO low-speed straight samples (straights are 240+ km/h); the reference comes from the out lap / pit lane of the same file. A file without any slow straight running gets no load ratio.
    - Peak fix: rear "peak" was 4.2 deg from a 5-sample last bin. Now bins need 20 samples to be the peak and `peak_observed` says whether the curve fell off past it.
    - Fuji result: load-normalized curves never fall off (front still rising at 5.8 deg, rear at 3.8 deg) -> peak_observed false, GU flagged as upper bound. The raw-G drop-off past ~3.5 deg was mostly slow corners having less downforce.
    - Fuji tyres ran 32 C below the 92 C optimum (2 % of front samples within +-10 C) -> model flags a cold-tyre curve.
    - CE = lap p90 grip / model p90 envelope near the peak (was lap p95 / bin mean, so ~1.08 on an ordinary lap). Best Fuji lap: CE 0.93, GU F 0.93 / R 0.89 (upper bounds).
    - Old (v1) tyre observations are discarded on load (tire_model_version). ABS/TC/slip baselines are kept. Profiles are now saved without indent.
    - telemetry.py now skips tables marked unpopulated_by_lmu (they were read anyway because of the default-frequency fallback).
    - Tests: backend/tests/test_tire_model.py (13).
- TODO (tyre model): profile is per car only, not per compound/track; CE is per lap, not per corner; analyzing a second lap of an already-ingested file scores it against a model that contains it; recorder.py (non-HF) does not record Susp Force.

### 10/18/2026
- Bulk grip-profile update from all 13 October sessions (no LLM): script in C:\Users\hoang\scratch\gripbatch\batch.py (outside the repo), pre-run profile backup in scratch\gripbatch\backup. Per file: update_from_grip_report for every complete lap + ingest_tire_laps. Ledger scratch\gripbatch\ledger.json stops a re-run double-counting the ABS/TC/slip baselines (update_from_grip_report has no dedup; the tyre model dedups by "<file>#<lap>").
- Result (tyre-model laps / samples per axle): WRT #69 23 / 14,410; Cadillac #12 14 / 5,217; Aston #007 WEC 10 / 3,491; Aston #007 LM 4 / 1,342; Manthey #92 5 / 3,171; Proton #88 3 / 2,040. NEW profiles: Manthey #91 (Barcelona) 5 / 3,412; Genesis #17 (Sarthe) 2 / 2,212; Ferrari #50 (Road Atlanta) 6 / 1,502.
- BUG (confirmed, not fixed in backend): the LAST lap of every file has no closing Lap event (telemetry.py ~l.461, end_ts=None -> ends at the last sample), so it is always partial (1.4 s - 114 s seen). A dry run through the unmodified path made that partial lap the "fastest", and ingest_tire_laps' 115 % cutoff then rejected every real lap (Ferrari/Sebring/Proton/Barcelona/Sarthe got ONE partial lap each). Same root cause as the "BEST badge on the unfinished final lap" TODO. The script drops the last lap and any lap < 85 % of the file median (mid-lap join, e.g. Barcelona L19 71.5 s, Sebring L0 48 s). /analyze should do the same.
- Old v1 profiles (Aston LM, Manthey #92, WRT; no tire_model_version) lost their v1 tyre observations on load, by design (tire_model_version). ABS/TC baselines kept.
- Tyre-model sample counts are still small for the new cars (<= 3.4k); confidence stays low until more laps.
- FIX: telemetry.LapTelemetry.is_complete (False for the last lap, which has no closing Lap event) + telemetry.reference_lap_time() = fastest complete lap ignoring laps < 85 % of the median. Used by ingest_tire_laps (115 % cutoff; partial laps are now ingested, not used as the reference) and by /analyze's default "fastest lap" pick. The UI BEST badge still uses its own min() - not changed.
- Second ingest pass (scratch\gripbatch\batch2.py, backup in scratch\gripbatch\backup2): +1-2 partial/join laps per file, tyre model only. WRT #69 now 25 laps / 15,369 samples per axle.
- NEW: corner names for the coach. track_db.annotate_report_with_corners() adds "corner" / "corner_end" (e.g. "T3 Apex (Coca-Cola Corner) (+25 m)" or "between X and Y", 60 m snap) to ABS/TC/understeer events and slip peaks, plus corner_index and corner_names_available. Prompt rule 4 now says to use those fields, and to use metres and never invent names when the track has no index (Road Atlanta is NOT in the registry). ref_points distances are unverified (Fuji has T16 at 3680 m on a 4563 m lap). Tests: backend/tests/test_corners_and_ref_lap.py.

#### Tyre model scorecard (10/18/2026, model v2) - re-run: scratch\gripbatch\validate.py <repo> "<car>"
Tests: (a) proxy mu (lat G / load_ratio) vs the game's own mu = sum|Tire Lateral Force| / sum Tire Load (only the 10/2-10/4 files have forces; axle load < 30 % of median masked - 0 N Tire Load over kerbs made the first attempt read r 0.02); (b) predict held-out samples' grip from the curve fitted on the other laps / files, cornering samples (slip > 1.5 deg) only, R2 vs the held-out mean; (c) 200x bootstrap over laps; (d) Spearman of per-lap CE (scored against a model WITHOUT that lap) vs lap time, per file.

| Car | laps / samples | (a) proxy vs game mu: r, curve-shape r, peak game vs proxy | (b) cornering R2 / RMSE F, R | (c) peak 90 % CI F, R | (d) CE vs lap time |
|---|---|---|---|---|---|
| WRT #69 M4 GT3 (Fuji) | 25 / 15,369 | not testable (no forces) | 0.34 / 0.12, 0.05 / 0.15 | 5.4-6.2 (never observed), 3.8-5.8 (never observed) | rho -0.15 (p 0.57, n 16); +0.00 (n 7) |
| Cadillac #12 (Road Atlanta) | 14 / 5,217 | F 0.97, 0.999, 3.8 vs 4.6; R 0.97, 0.995, 3.8 vs 3.8 | 0.03 / 0.34, ~0 / 0.28 | 3.4-4.6 (obs 77 %), 2.6-4.6 (19 %) | rho -0.69 (p 0.06, n 8); +0.09 (n 6) |
| Aston #007 WEC (Road Atlanta) | 11 / 3,491 | F 0.98, 0.999, 4.2 vs 4.2; R 0.98, 0.94, 3.4 vs 3.4 (rear proxy +14 % high) | 0.06-0.09 / 0.28, ~0 / 0.29 | 4.2-4.6 (11 %), 3.0-3.8 (16 %) | too few laps per file |

Findings:
- The grip PROXY is right: on both cars with game forces it tracks the game's mu at r 0.97, the binned curve shapes match (r 0.94-0.999) and peak slip angles agree within one 0.4 deg bin except Cadillac front (0.8 deg). Rear proxy reads ~3-14 % high.
- The CURVE is an average, not a predictor: inside the cornering range slip angle explains 0-34 % of sample-to-sample grip (front better than rear). Grip at a given slip angle varies a lot with speed, load, temperature and combined slip. So GU ("where on the curve") is defensible; reading the curve as "grip you would get at X deg" is not, at the sample level.
- PEAKS ARE MOSTLY NOT FOUND: WRT never got past the peak in any bootstrap (and ran 33 C below optimal tyre temp - a cold-tyre curve); Aston only in 11-16 % of resamples. The peak is a lower bound most of the time, as the model already says.
- CONFIDENCE IS OVERSTATED: WRT shows 0.979 while the rear peak's 90 % CI is 3.8-5.8 deg and it was never observed. 1-exp(-n/4000) only counts samples. TODO: base confidence on bootstrap spread + peak_observed.
- CE does not predict lap time yet (one file at rho -0.69, p 0.06; the rest ~0). Laps within one file vary by ~1 % so this needs more laps before it means anything.
- Theoretical lap time from this model: not worth building until peaks are observed and confidence is fixed.
- Tools moved into the repo: backend/tools/ (ingest_sessions, validate_tire_model, list_laps, check_tire_forces + .ps1 wrappers, README.md). Scorecard re-run: backend\tools\VALIDATE_TIRE_MODEL.ps1 "<car>" (replaces scratch\gripbatch\validate.py above). Ingest ledger now lives at data/grip_profiles/.ingest_ledger.json (seeded with the 13 October files), backups under data/grip_profiles/backups/.

#### Tyre forces by class, and the Susp Force pilot (10/18/2026)
- Tire Load / Tire Lateral Force / Tire Longitudinal Force / Grip Fraction: real values for Hypercars (Aston #007, Cadillac #12, Ferrari 499P, Genesis GMR-001); all zero for every LMGT3 recorded (BMW M4, Porsche 911 GT3 R, Mustang; Lexus by team name; ~20 files). Same recorder (hf1) and tyre compound on 10/18: Porsche GT3 03:03 UTC zero, Genesis 04:39 and Ferrari 04:57 populated. So it is the game, not the recorder or pyLMUSharedMemory (the struct maps the fields for every car).
- Iron Lynx #61 (Mercedes GT3, 9/19) is NOT an exception: recorder v1 wrote all-zero tables without the unpopulated_by_lmu flag, which made a flag-only scan read "ok". Values are 0 in all 14,345 / 10,612 rows. A channel check must look at the values (check_tire_forces.py does for one file).
- Others report the same: TinyPedal v2.34.0 changelog ("tyre load ... may not be available from certain vehicles in game API (such as LMGT3)"; it falls back to suspension load) and the lmuFFB app (LMU 1.2 returns 0 for tire load and grip fraction). LMU 1.2 notes: new shared memory header + "Updated tyre model for LMP2, LMP3 and GT3 classes" - the class split matches, cause unconfirmed (Studio 397 issue; no public statement found). LMP2/LMP3 expected to be zero too - untested, no files.
- PILOT, Susp Force as the load source (backend\tools\COMPARE_LOAD_SOURCES.ps1; Genesis 4 laps, Ferrari 9 laps, steady-state samples vs the game's own mu = sum|Fy| / sum Fz):
  - Susp Force tracks Tire Load (r 0.97-0.98 per axle) but is 1.5-2.3x bigger (pushrod, not wheel load), so only the ratio to a low-speed reference is usable - which is how the model uses it.
  - Cornering samples (|slip| > 1.5 deg), r with the game's mu: raw lat G 0.01-0.32; lat G / load_ratio from Tire Load 0.85/0.93 (Genesis F/R) and 0.87/0.94 (Ferrari); from Susp Force 0.74/0.82 and 0.79/0.88. Load normalisation is the big win; Susp Force gives up ~0.07-0.11 of r against the true load.
  - Peak slip: game mu 3.8 F / 2.6 R (both files, never observed). Susp Force proxy 3.8 / 2.6 (Genesis), 4.2 / 2.6 (Ferrari): within one 0.4 deg bin except Ferrari front by one bin.
  - Available grip (p90): game 1.41-1.51, proxy (either load source) 1.6-1.8: the proxy reads 5-15 % high, with Tire Load too, so that bias is the lat-G stand-in, not the Susp Force substitution. Compare proxy values between laps/cars, don't read them as absolute mu.
  - Not a GT3 validation: Hypercar suspension/pushrod geometry and downforce differ, 2 cars, 13 laps. A GT3 run cannot be checked against game forces at all (no data). Next: repeat on LMP2/LMP3 if those turn out to have both channels.
- TODO: make check_tire_forces.py scan all files by values and print the car/class table; draft a report for the Studio 397 forum / TinyPedal issue tracker (not sent).
