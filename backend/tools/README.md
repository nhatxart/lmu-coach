# backend/tools

Command-line tools that run the backend's own code on your recordings. You don't need Docker or the LLM for any of them.
Each `.ps1` wrapper can be run from any folder. It calls `uv` with the backend's pinned `requirements.txt`, so there is no venv to manage.
The first run downloads the packages, and later runs use uv's cache.

| Tool | Writes? | What it's for |
|---|---|---|
| `INGEST_SESSIONS.ps1` | grip profiles (only with `--save`) | Folds many sessions into the per-car grip profiles at once |
| `VALIDATE_TIRE_MODEL.ps1` | nothing | Checks how accurate one car's tyre model is and prints a scorecard row |
| `LIST_LAPS.ps1` | nothing | Shows each file's laps as the backend sees them |
| `CHECK_TIRE_FORCES.ps1` | nothing | Checks whether a recording has usable tyre-force channels |
| `COMPARE_LOAD_SOURCES.ps1` | nothing | Pilot: does Susp Force give the same grip numbers as the game's Tire Load? (needs a car that has both) |

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
.\INGEST_SESSIONS.ps1 --month 2026-10 --label-only --save   # only record where each lap's load came from; adds no samples
```

- It's a **dry run unless you pass `--save`**. Check the output first.
- **Backups:** `--save` first copies every profile and the ledger to `data/grip_profiles/backups/<timestamp>/`. To undo a run, copy that folder's files back.
- **Stop the backend first** (`docker compose down`). It reads and writes the same profile files. Restart it afterwards (`docker compose up -d --build`).
- **New cars:** a car without a profile gets a new one, named after the car, just as `/analyze` would name it.
- **Load source labels:** each tyre-model lap is tagged with where its load came from (`Tire Load`, `Susp Force`, `Susp Pos (calibrated...)` or `none`). It is shown in the baselines the LLM sees as `load_normalization`. `--label-only` fills the tag in for laps ingested before it existed.
- **Old profiles:** a profile from tyre model v1 (no `tire_model_version`) loses its v1 tyre samples when loaded. This is by design. Its baselines are kept.

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

## Running the Python directly

The wrappers only add the `uv` line, so you can also run the Python directly from the repo root:

```powershell
uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\validate_tire_model.py WRT
```

`ingest_sessions.py` and `validate_tire_model.py` take `--help`. `check_tire_forces.py` prints its usage when started without a file. `list_laps.py` takes one optional glob.
