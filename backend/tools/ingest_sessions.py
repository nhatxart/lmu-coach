"""
Bulk-fold recorded sessions into the per-car grip profiles, without the LLM.

The same profile updates /analyze makes, for many files at once:
  - tyre model: ingest_tire_laps over every lap of each file (deduplicated by
    "<file>#<lap>" inside the profile, so re-runs add nothing)
  - ABS/TC/slip baselines: update_from_grip_report once per real lap. That
    call has no dedup of its own, so files done here are recorded in
    <profiles>/.ingest_ledger.json and skipped next time.

Baselines use real laps only: complete, and >= 85 % of the file's median
(the unfinished last lap and a mid-lap join would skew peak G / slip). The
tyre model takes those laps too (see telemetry.reference_lap_time).

Dry run unless --save. With --save the profiles are copied to
<profiles>/backups/<timestamp>/ first.

    python -I backend/tools/ingest_sessions.py --month 2026-10
    python -I backend/tools/ingest_sessions.py --since 2026-10-14 --save
    python -I backend/tools/ingest_sessions.py --files "Fuji*.duckdb" --tyre-only --save
    python -I backend/tools/ingest_sessions.py --month 2026-10 --label-only --save

--label-only is a no-op since profile v3 (the load source is stored at ingest).

Profiles are keyed by the car MODEL (car_identity.identify_car, the same helper
/analyze uses), so every livery shares one file. Tyre bins are built from files
loaded at tire_model.TIRE_GRID_HZ, like /analyze; the ABS/TC baselines use the
20 Hz coaching load (the default grid), also like /analyze.

--migrate folds the old livery-named profiles (e.g. "Team WRT 2026 _69_LP.json")
into their model profile: the ABS/TC/slip baseline lists are concatenated (capped
at MAX_SAMPLES, newest kept) and lap_count is summed. With --save the livery files
are moved into the backup directory (never deleted). Which model a livery belongs
to comes from the CarName/VehicleModel metadata of the files in --sessions-dir,
falling back to the livery table.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app import grip_profile as gp  # noqa: E402
from app.car_identity import identify_car  # noqa: E402
from app.grip_analysis import build_grip_report  # noqa: E402
from app.telemetry import anonymize_metadata, get_car_setup_json, load_telemetry_file  # noqa: E402
from app.tire_model import TIRE_GRID_HZ  # noqa: E402

STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})T[\d_]+Z")

_BASELINE_LISTS = ("abs_lockup_durations_s", "tc_active_fractions", "peak_combined_g_values",
                   "peak_front_slip_angle_values", "peak_rear_slip_angle_values",
                   "max_wheel_slip_ratio_values")


def _peak_str(m) -> str:
    """Peak with a marker: '>=' = never observed (lower bound); notes a fit at the grid edge."""
    s = f"{m['peak']}" if m.get("peak_observed") else f">={m['peak']}"
    if m.get("peak_at_grid_bound"):
        s += f" (raw fit at grid bound {m.get('peak_slip_extrapolated')}, not quoted)"
    return s


def merge_baselines(dst: gp.GripProfile, srcs: list) -> None:
    """Fold the ABS/TC/slip baselines of livery profiles into the model profile `dst`.

    WHY: the baselines were learned per livery name; a model's liveries drive the same
    car, so the history belongs together. Lists are concatenated (dst first, then each
    src in the order given; so livery lists follow the model list and the order is not
    chronological) and capped to the newest MAX_SAMPLES; lap_count is summed.
    Tyre data is not touched (v2 livery profiles carry none after loading).
    """
    for src in srcs:
        for name in _BASELINE_LISTS:
            setattr(dst, name, (getattr(dst, name) + getattr(src, name))[-dst.MAX_SAMPLES:])
        dst.lap_count += src.lap_count
        if dst.car_class == "unknown":
            dst.car_class = src.car_class
        # a calibration the model profile lacks is kept; an existing one wins
        for k, v in src.deflection_calibration.items():
            dst.deflection_calibration.setdefault(k, v)


def livery_models(sessions_dir: Path, profiles_dir: Path) -> dict:
    """{profile file stem of a livery: model} for the livery-named profiles in profiles_dir.

    Sources: CarName/VehicleModel metadata of every recording in sessions_dir (so the
    mapping does not depend on which files this run ingests), then the livery table for
    stems like "Aston Martin THOR Team 2026 _007_LM" (the sanitised "#007:LM").
    """
    stems = {p.stem for p in profiles_dir.glob("*.json") if not p.name.startswith(".")}
    out: dict = {}
    for path in sorted(sessions_dir.glob("*.duckdb")):
        try:
            meta = anonymize_metadata(path)
        except Exception:
            continue
        car = meta.get("CarName")
        if not car:
            continue
        stem = gp._profile_path(car).stem
        model = identify_car(meta, car).model
        if stem in stems and stem != gp._profile_path(model).stem:
            out[stem] = model
    for stem in stems - set(out):
        model = identify_car({}, re.sub(r"_(\d+)_", r"#\1:", stem)).model
        # identify_car returns its input when the livery table has no entry: not a mapping
        if model != re.sub(r"_(\d+)_", r"#\1:", stem):
            out[stem] = model
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", type=Path, default=REPO / "data" / "sessions")
    ap.add_argument("--profiles-dir", type=Path, default=REPO / "data" / "grip_profiles")
    ap.add_argument("--month", help="YYYY-MM, from the UTC stamp in the file name")
    ap.add_argument("--since", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--until", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--files", help="glob on the file name, e.g. \"Fuji*\"")
    ap.add_argument("--tyre-only", action="store_true",
                    help="only the tyre model; skip ABS/TC/slip baselines and the ledger")
    ap.add_argument("--label-only", action="store_true",
                    help="only backfill the per-lap load-source labels; add no samples or baselines")
    ap.add_argument("--migrate", action="store_true",
                    help="merge the baselines of livery-named profiles into their model profile; "
                         "with --save the livery files move to the backup directory")
    ap.add_argument("--save", action="store_true", help="write profiles (default: dry run)")
    args = ap.parse_args()

    gp.PROFILES_DIR = args.profiles_dir
    ledger_path = args.profiles_dir / ".ingest_ledger.json"
    ledger = set(json.loads(ledger_path.read_text())) if ledger_path.exists() else set()

    files = []
    for p in args.sessions_dir.glob("*.duckdb"):
        m = STAMP.search(p.name)
        day = m.group(1) if m else ""
        if args.month and not day.startswith(args.month):
            continue
        if args.since and day < args.since or args.until and day > args.until:
            continue
        if args.files and not fnmatch.fnmatch(p.name, args.files):
            continue
        files.append((m.group(0) if m else p.name, p))
    files.sort()
    if not files:
        print("no matching files")
        return 1

    profiles: dict[str, gp.GripProfile] = {}
    new_cars = set()
    for _, path in files:
        print(f"\n== {path.name}")
        try:
            # the tyre bins are built at the tyre grid, as /analyze does
            laps = load_telemetry_file(path, grid_hz=TIRE_GRID_HZ)
        except Exception as e:  # one bad file should not stop the batch
            print(f"  LOAD FAILED: {e}")
            continue
        valid = [l for l in laps if l.lap_time_s < 600]
        if not valid:
            print("  no laps, skipped")
            continue
        # PREVIOUSLY keyed by valid[0].car_name (the livery), while /analyze keys by the car
        # model, so the tool and the app built separate files for one car. Fix: the same
        # helper /analyze uses (load_profile_for_lap).
        prof = gp.load_profile_for_lap(valid[0])
        car = prof.car_name
        if car not in profiles:
            if not gp._profile_path(car).exists():
                new_cars.add(car)
            profiles[car] = prof
        prof = profiles[car]
        print(f"  car={car} (livery {valid[0].car_name}) "
              f"{'(NEW profile)' if car in new_cars else '(extending)'} track={valid[0].track_name}")

        if args.label_only:
            # v3 stores each lap's load source at ingest, so there is nothing to backfill.
            print("  --label-only: load sources are recorded at ingest in profile v3, skipped")
            continue

        if not args.tyre_only:
            if path.name in ledger:
                print("  baselines: already in ledger, skipped")
            else:
                # the coaching report /analyze builds is on the default 20 Hz load, not the
                # 50 Hz tyre grid, so the baselines are fed from a 20 Hz load of the file
                coaching = [l for l in load_telemetry_file(path) if l.lap_time_s < 600]
                complete = sorted(l.lap_time_s for l in coaching if l.is_complete)
                med = complete[len(complete) // 2] if complete else 0
                real = [l for l in coaching if l.is_complete and l.lap_time_s >= 0.85 * med]
                setup = get_car_setup_json(path)
                done = 0
                for lap in real:
                    try:
                        prof.update_from_grip_report(build_grip_report(lap, car_setup_json=setup))
                        done += 1
                    except Exception as e:
                        print(f"  lap {lap.lap_number}: report failed: {e}")
                skipped = sorted(set(l.lap_number for l in coaching) - set(l.lap_number for l in real))
                print(f"  baselines: +{done} laps (not real laps: {skipped or 'none'})")
                ledger.add(path.name)

        n0 = prof.tire_sample_count()
        try:
            added = prof.ingest_tire_laps(laps, file_key=path.name)
        except Exception as e:
            print(f"  tyre model FAILED: {e}")
            added = []
        print(f"  tyre model: +laps {added or 'none'} +samples/axle {prof.tire_sample_count() - n0}")

    moved: list = []
    if args.migrate:
        print("\n== migrate livery profiles")
        for stem, model in sorted(livery_models(args.sessions_dir, args.profiles_dir).items()):
            path = args.profiles_dir / f"{stem}.json"
            livery = gp.GripProfile.from_dict(json.loads(path.read_text()))
            target = profiles.get(model) or gp.load_profile(model)
            before = target.lap_count
            merge_baselines(target, [livery])
            profiles[model] = target
            moved.append(path)
            print(f"  {stem} -> {model}: baseline laps {before} + {livery.lap_count} = {target.lap_count}")
        if not moved:
            print("  no livery-named profiles to migrate")

    print("\n== profiles")
    for car, prof in profiles.items():
        t0 = time.perf_counter()
        b = prof.compute_baselines()
        dt = time.perf_counter() - t0
        f, r = b["front_tire_model"], b["rear_tire_model"]

        def show(m):
            return (f"{m['model']} peak {_peak_str(m)} CI90 {m['peak_slip_ci90']} "
                    f"conf {m['confidence']} prior {m['prior_used']} ({m['grip_measure']})")

        print(f"  {car}: baseline laps {prof.lap_count}, tyre laps {len(prof.tire_model_laps)}, "
              f"samples/axle {f['sample_count']}, fit+summary {dt:.2f} s\n"
              f"    F {show(f)}\n    R {show(r)}")

    if not args.save:
        print("\nDRY RUN - nothing saved (add --save)")
        return 0
    backup = args.profiles_dir / "backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True, exist_ok=True)
    for p in args.profiles_dir.glob("*.json"):
        shutil.copy2(p, backup / p.name)
    if ledger_path.exists():
        shutil.copy2(ledger_path, backup / ledger_path.name)
    for prof in profiles.values():
        gp.save_profile(prof)
    for path in moved:
        # already copied above; the move only removes it from the live directory
        path.replace(backup / path.name)  # os.replace: shutil.move fails on an existing target on Windows
    if not args.tyre_only and not args.label_only:
        ledger_path.write_text(json.dumps(sorted(ledger), indent=1))
    print(f"\nSAVED {len(profiles)} profiles (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

