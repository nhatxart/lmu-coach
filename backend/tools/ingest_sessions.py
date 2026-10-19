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

--label-only adds nothing to any profile: it only fills in where each tyre-model
lap's load came from (Tire Load / Susp Force / none) for laps ingested before
that was recorded. Needs the session files; laps whose file is gone stay
"not recorded".
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app import grip_profile as gp  # noqa: E402
from app.grip_analysis import build_grip_report  # noqa: E402
from app.telemetry import get_car_setup_json, load_telemetry_file  # noqa: E402
from app.tire_model import TIRE_GRID_HZ  # noqa: E402

STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})T[\d_]+Z")


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
        car = valid[0].car_name
        if car not in profiles:
            if not gp._profile_path(car).exists():
                new_cars.add(car)
            profiles[car] = gp.load_profile(car)
        prof = profiles[car]
        print(f"  car={car} {'(NEW profile)' if car in new_cars else '(extending)'} track={valid[0].track_name}")

        if args.label_only:
            # v3 stores each lap's load source at ingest, so there is nothing to backfill.
            print("  --label-only: load sources are recorded at ingest in profile v3, skipped")
            continue

        if not args.tyre_only:
            if path.name in ledger:
                print("  baselines: already in ledger, skipped")
            else:
                complete = sorted(l.lap_time_s for l in valid if l.is_complete)
                med = complete[len(complete) // 2] if complete else 0
                real = [l for l in valid if l.is_complete and l.lap_time_s >= 0.85 * med]
                setup = get_car_setup_json(path)
                done = 0
                for lap in real:
                    try:
                        prof.update_from_grip_report(build_grip_report(lap, car_setup_json=setup))
                        done += 1
                    except Exception as e:
                        print(f"  lap {lap.lap_number}: report failed: {e}")
                skipped = sorted(set(l.lap_number for l in valid) - set(l.lap_number for l in real))
                print(f"  baselines: +{done} laps (not real laps: {skipped or 'none'})")
                ledger.add(path.name)

        n0 = prof.tire_sample_count()
        try:
            added = prof.ingest_tire_laps(laps, file_key=path.name)
        except Exception as e:
            print(f"  tyre model FAILED: {e}")
            added = []
        print(f"  tyre model: +laps {added or 'none'} +samples/axle {prof.tire_sample_count() - n0}")

    print("\n== profiles")
    for car, prof in profiles.items():
        b = prof.compute_baselines()
        f, r = b["front_tire_model"], b["rear_tire_model"]
        print(f"  {car}: baseline laps {prof.lap_count}, tyre laps {len(prof.tire_model_laps)}, "
              f"samples/axle {f['sample_count']}, peak F {f['peak']} (observed {f['peak_observed']}) "
              f"R {r['peak']} (observed {r['peak_observed']})")

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
    if not args.tyre_only and not args.label_only:
        ledger_path.write_text(json.dumps(sorted(ledger), indent=1))
    print(f"\nSAVED {len(profiles)} profiles (backup: {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
