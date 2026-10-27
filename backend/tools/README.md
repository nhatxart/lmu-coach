# backend/tools

Command-line tools that run the backend's own code on your recordings. You don't need Docker or the LLM for any of them.
Each `.ps1` wrapper can be run from any folder. It calls `uv` with the backend's pinned `requirements.txt`, so there is no venv to manage.
The first run downloads the packages, and later runs use uv's cache.

| Tool | Writes? | What it's for |
|---|---|---|
| `INGEST_SESSIONS.ps1` | grip profiles (only with `--save`) | Folds many sessions into the per-car grip profiles at once |
| `VALIDATE_TIRE_MODEL.ps1` | nothing | Checks how accurate one car's tyre model is and prints a scorecard row |
| `LIST_LAPS.ps1` | nothing | Shows each file's laps as the backend sees them |
| `DETECT_TRACK_CORNERS.ps1` | corner maps (only with `--save`) | Finds each track's corners from curvature and compares them with the official turn count |
| `BUILD_TRACK_REFERENCE.ps1` | `data/track_reference/*.json`, `*.topography.json` | Builds each track's reference (corners, complexes, names) from OpenStreetMap + LMU names + your laps; `--fetch` downloads OSM once for a new track. Also builds the track's topography (height, banking, kerb/grass/gravel paint for the 3D chase view) from the recordings; `--topography-only` does just that, `--no-topography` skips it. Height needs recordings made with the recorder from 10/25/2026 on (pose channels); older ones give paint only |
| `CHECK_TIRE_FORCES.ps1` | nothing | Checks whether a recording has usable tyre-force channels |
| `COMPARE_LOAD_SOURCES.ps1` | nothing | Pilot: does Susp Force give the same grip numbers as the game's Tire Load? (needs a car that has both) |
| `MF_EXPERIMENT.ps1` | nothing | Tyre model v3 experiment: which state factors (load, temp, pressure, wear, camber) improve held-out prediction of the game's true mu? |
| `CLASS_COLLAPSE.ps1` | nothing | Tyre model v3 experiment: how much should a car lean on the pooled curve of the other cars of its class (works for LMGT3, which has no tyre forces)? |
| `EXPORT_CAR_MODELS.ps1` | `docs/car_models/*.stl` | Writes the chase view's GT3 and Hypercar models as STL files (1:1 and 1:43, millimetres) to check in a 3D-printer slicer. Needs Node.js, not uv |

None of these tools ever modify `data/sessions/*.duckdb`. They open it read-only.

---

## INGEST_SESSIONS.ps1 — bulk grip-profile update

This makes the same profile update that pressing **Analyze** makes, without the LLM, for every matching file.

- **Tyre model:** every lap of every file goes in, including the unfinished last lap. Laps slower than 115 % of the file's fastest real lap are skipped (out-laps and in-laps). The profile remembers `<file>#<lap>`, so a re-run never adds a lap twice.
- **ABS / TC / slip baselines:** these take real laps only. A real lap is complete and at least 85 % of the file's median lap time. That rules out the unfinished last lap and a mid-lap join, which would skew peak G and slip. The baseline step has no dedup of its own, so finished files are recorded in `data/grip_profiles/.ingest_ledger.json` and skipped on the next run.

The tool picks files by the UTC stamp in the file name (`...-2026-10-18T04_57_04Z.duckdb`).

```powershell
cd backend\tools
.\INGEST_SESSIONS.ps1 --month 2026-10                       # dry run: shows what would change
.\INGEST_SESSIONS.ps1 --month 2026-10 --save                # write it
.\INGEST_SESSIONS.ps1 --since 2026-10-14 --until 2026-10-18 --save
.\INGEST_SESSIONS.ps1 --files "Fuji*" --save                # glob on the file name
.\INGEST_SESSIONS.ps1 --files "Fuji*" --tyre-only --save    # tyre model only, ledger untouched
.\INGEST_SESSIONS.ps1 --month 2026-10 --tyre-only --migrate --save   # also fold old livery-named profiles into the model profile
```

- It's a **dry run unless you pass `--save`**. Check the output first.
- **Backups:** `--save` first copies every profile and the ledger to `data/grip_profiles/backups/<timestamp>/`. To undo a run, copy that folder's files back.
- **Stop the backend first** (`docker compose down`). It reads and writes the same profile files. Restart it afterwards (`docker compose up -d --build`).
- **Profile key (tyre model v3):** profiles are named after the car **model** (`BMW M4 LMGT3.json`), found the same way `/analyze` finds it, so every livery of a model shares one file. A car without a profile gets a new one.
- **Tyre grid:** the tyre bins are built from files loaded at 50 Hz (`tire_model.TIRE_GRID_HZ`), exactly like `/analyze`. The ABS/TC baselines keep the 20 Hz coaching load.
- **`--migrate`:** folds the old livery-named profiles (`Team WRT 2026 _69_LP.json`) into their model profile. The ABS/TC/slip lists are concatenated (newest 200 kept) and `lap_count` is summed. Which model a livery belongs to comes from the metadata of the files in `--sessions-dir`. With `--save` the livery files are **moved** into the backup folder, never deleted. Run it once after upgrading to v3, together with `--tyre-only` if you do not want to touch the ledger.
- **`--label-only`:** does nothing since v3. The load source of each lap (`Tire Load`, `Susp Force`, `Susp Pos (calibrated...)` or `none`) is stored at ingest and shown to the LLM as `load_normalization`.
- **Old profiles:** a profile from tyre model v1 or v2 loses its tyre samples when loaded, because v3 stores per-lap bins instead of raw samples. This is by design. Its baselines are kept. Re-ingest the recordings to rebuild the tyre model.
- The profile summary at the end prints, per axle, the model used, peak slip, its 90 % interval, confidence and prior, and how long the fit took.

Typical output per file:

```
== Michelin Raceway Road Atlanta-Ferrari AF Corse 2026 _50_WEC-2026-10-18T04_57_04Z.duckdb
  car=Ferrari AF Corse 2026 #50:WEC (NEW profile) track=Michelin Raceway Road Atlanta
  baselines: +8 laps (not real laps: [8])
  tyre model: +laps [1, 2, 3, 4, 5, 7, 8] +samples/axle 1549
```

---

## VALIDATE_TIRE_MODEL.ps1 — how good is the tyre model?

This is read-only. It re-extracts the laps a car's tyre model was built from, then runs four tests:

| Test | Question | How to read it |
|---|---|---|
| **(a) ground truth** | Does our grip proxy (lat G ÷ load ratio) match the game's own grip, Σ\|Tire Lateral Force\| ÷ Σ Tire Load? | `r` near 1 and `curve-shape r` near 1 mean the proxy is right. The peak from the game's data should match the peak from the proxy. This only works for files that recorded tyre forces (so far the 10/2–10/4 Aston #007 and Cadillac #12 files). Otherwise it says "not testable". |
| **(b) held-out prediction** | Does a curve fitted on the other laps (or files) predict a new lap's grip from its slip angle? | It uses cornering samples only (slip > 1.5°). R² is the share of grip variation the slip angle explains, and RMSE is in grip units. Low R² means the curve is a good average but a poor sample-by-sample predictor. |
| **(c) bootstrap** | How much does the peak move if the laps are resampled? | It gives a 90 % interval for the peak and for the available grip. "peak observed in X %" near 0 means you've never driven past the peak, so the peak is only a lower bound. |
| **(d) CE vs lap time** | Do laps with higher cornering efficiency come out faster? | Spearman ρ per file. Negative means higher CE goes with a faster lap. A file needs ≥ 5 real laps. |

```powershell
cd backend\tools
.\VALIDATE_TIRE_MODEL.ps1 WRT                 # any unique part of the profile file name
.\VALIDATE_TIRE_MODEL.ps1 007_WEC --boot 100  # fewer bootstrap resamples = faster (default 200)
```

**(e) v3 model (50 Hz grid):** since tyre model v3 the validator also prints, from the profile's stored bins alone (no session files needed), the held-out R2 of the Magic Formula against the old slip-binned curve, per axle, on the proxy mu and, for cars that record tyre forces, on the game's true mu. It also prints the peak's 90 % interval, the confidence and the prior used, and the time of a full refit. Tests (a) to (d) keep the old 20 Hz grid so their rows stay comparable with earlier scorecards; the final output has one v2 row (20 Hz) and two v3 rows (50 Hz, front and rear). Profiles are named after the car model now, so use e.g. `BMW`, `Cadillac`, `Valkyrie`, `Ferrari`, `Genesis` or `Porsche`.

If the name matches several profiles (e.g. `Manthey`), the tool lists them. Use a longer piece, e.g. `_92_WEC`.
The last line is a markdown row. Paste it into the **Tyre model scorecard** table in `DevNotes.md` under a new dated heading, so accuracy can be compared over time. Run it after each ingest batch, or after any change to `tire_model.py`.

A WRT #69 run takes about 1–2 minutes, and most of that is (c) and (d). Both refit the model many times.

---

## LIST_LAPS.ps1 — laps per file

```powershell
.\LIST_LAPS.ps1                  # every file
.\LIST_LAPS.ps1 "*2026-10-18*"   # glob on the file name
```

```
Michelin Raceway Road Atlanta-Ferrari AF Corse 2026 _50_WEC-2026-10-18T04_57_04Z.duckdb
  Ferrari AF Corse 2026 #50:WEC | 9 laps | L0 2:01.463  L1 1:13.058 ... L7 1:12.885R  L8 0:14.080*
```

- `*` marks the unfinished last lap. It has no closing Lap event, so its time is wherever the recording stopped.
- `j` marks a lap shorter than 85 % of the file's median, usually a mid-lap join.
- `R` marks the reference lap: the fastest real lap. `/analyze` uses it by default, and the tyre model's 115 % cutoff is based on it.

---

## CHECK_TIRE_FORCES.ps1 — does this file have tyre forces?

```powershell
.\CHECK_TIRE_FORCES.ps1 "..\..\data\sessions\<file>.duckdb"
```

The output covers:
- The rate and row count of each force channel.
- Fy and Fz percentiles per wheel.
- The share of samples where a wheel's load is near 0 N (kerbs, wheel lift). The validator masks those.
- How well summed lateral force matches lateral G at different time lags. Lag 0 should be best, with r ≈ 0.9 or more.

A file that passes can serve as ground truth for test (a).

---

## COMPARE_LOAD_SOURCES.ps1 — can Susp Force replace Tire Load?

LMU writes zero Tire Load / Tire Lateral Force for the GT3 cars (BMW M4, Porsche, Mustang, ...), so their tyre model takes load from Susp Force (pushrod load). That substitute can only be tested on a car that has **both** channels plus the game's own lateral force. So far that means the 10/18 Hypercar files.

`powershell
.\COMPARE_LOAD_SOURCES.ps1 "*Genesis*" "*Ferrari AF Corse*"    # quote the globs
`

Per axle it prints the load correlation and the grip proxy against the game's mu for three load choices: none (raw lat G), Tire Load, and Susp Force. Use it again when you record an LMP2 or LMP3: if that class has both channels, you get a second check. A GT3 file is skipped because it has no true Tire Load.

---

## MF_EXPERIMENT.ps1 — which tyre-model factors earn their place?

Tyre model v3 fits a Magic Formula whose peak and stiffness are scaled by load, tyre temperature, pressure, wear and camber. This tool tests, on recordings that carry the game's own tyre forces (true mu), whether each factor makes the model predict **unseen** data better. It bins the laps like the profile ingest does, then leaves one file out (one lap out, for a car with a single file), fits on the rest and scores the held-out cornering cells (slip of 1.5 degrees or more) against the true mu.

```powershell
.\MF_EXPERIMENT.ps1 --sessions-dir C:\path\to\data\sessions --class Hypercar
.\MF_EXPERIMENT.ps1 --sessions-dir C:\path\to\data\sessions --files "*Valkyrie*"
```

It prints one table per car and axle: held-out R², the change from the previous factor, RMSE, and the factor status of the training fits (`fit` = fitted, `NI` = not identifiable, `AB` = at a bound). `NI` means the data could not test that factor; it is not the same as "fitted but no gain". It runs in seconds and needs recordings with Tire Load and Tire Lateral Force (Hypercars today).

---

## CLASS_COLLAPSE.ps1 - how much should a car borrow from its class?

Cars of one class (all LMGT3, all Hypercars) share tyre physics, so a car with few laps can lean on a Magic Formula pooled over the *other* cars of its class (the "class prior"). This tool measures how much leaning helps. The target is the **proxy mu** (lateral G / load), because LMGT3 cars record no tyre forces.

```powershell
.\CLASS_COLLAPSE.ps1 --sessions-dir C:\path\to\data\sessions --class LMGT3
.\CLASS_COLLAPSE.ps1 --sessions-dir C:\path\to\data\sessions --class Hypercar
```

Per axle and car it prints the car's own fitted peak slip and peak mu (at the class reference state, with the offset vs the class prior; `*` = the peak is beyond that car's data), then leave-one-lap-out held-out R2 for prior weights 0, 0.3, 1, 3 and 10. The prior for a car is always built from the other cars only. The line `BEST WEIGHT` is the weight with the highest mean R2 over the scored cars. A car with one lap cannot be scored (no training lap) and is shown as `n/a`; laps without a load channel (plain lateral G) cannot be pooled and are listed as left out.

**Read it with care:** the cars of a class were recorded on different tracks, so the track is mixed up with the car, and a lap held out from the same file as its training laps is an easy test for weight 0. Runs in about half a minute.

---

## EXPORT_CAR_MODELS.ps1 - car models as STL

The chase view can draw a representative car instead of the chevron, so you can see where the four tyres are against the track edge. The models are built from `backend/app/static/car_models.js`, sized by `backend/app/static/car_models.json` (one entry per class: length, width, height, wheelbase, tracks, overhang, tyre size, ride height, wing, splitter, fin; the `notes` say where the numbers come from). This tool runs the same builder in Node and writes binary STL files you can open in any slicer:

```powershell
cd backend\tools
.\EXPORT_CAR_MODELS.ps1                    # into docs\car_models
.\EXPORT_CAR_MODELS.ps1 C:\temp\cars       # somewhere else
```

- One file per class and scale: `LMGT3_1to1_mm.stl`, `LMGT3_1to43_mm.stl`, `Hypercar_1to1_mm.stl`, `Hypercar_1to43_mm.stl`. Units are millimetres, Z is up, the nose points along +X and the wheels stand on z = 0.
- It prints the triangle count and the bounding box of each file. At 1:1 the box is the JSON's length x width x height (the height includes the wing); at 1:43 it is the same divided by 43 (about 107 mm long for the GT3).
- The 1:43 files are previews, not print-tuned: the wing endplates, splitter and fin are well under 0.5 mm thick at that scale, so a 0.4 mm nozzle may drop or thicken them.
- LMP2 and LMP3 use the Hypercar shape until they have their own entry; any other class is drawn as the GT3.
- Edit the JSON, run the tool again and reload the slicer to compare.

---

## Running the Python directly

The wrappers only add the `uv` line, so you can also run the Python directly from the repo root:

```powershell
uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\validate_tire_model.py WRT
```

`ingest_sessions.py` and `validate_tire_model.py` take `--help`. `check_tire_forces.py` prints its usage when started without a file. `list_laps.py` takes one optional glob.
