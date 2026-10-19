"""
Hypercar factor experiment for tyre model v3 (read-only).

    python -I backend/tools/mf_experiment.py --sessions-dir <dir> [--class Hypercar] [--files GLOB]

Question: which state factors (load, temperature, pressure, wear, camber) of the
factorized Magic Formula actually improve the prediction of the game's own tyre mu
(sum|Fy| / sum Fz, "true mu") on data the fit has not seen?

For every car model of the chosen class that has true mu, laps are binned exactly as
the profile ingest does (valid = lap_time < 600 s, within 115 % of the reference lap,
deflection calibration from the first lap with Susp Force). Then leave-one-FILE-out
(leave-one-LAP-out when the car has a single file): fit on the other folds, score on
the held-out one. Score = count-weighted R^2 / RMSE over held-out cells with
slip >= 1.5 deg and true_n >= 3, target = the cells' true mean weighted by true_n, R^2
against the held-out weighted mean (summed per fold, so a session offset between files
is not counted as explained variance).

Variants: binned_v2 / binned_v2_proxy (slip-bin means of the training data, trained on
true / proxy mu, interpolated by slip), mf_slip_only, +load, +load+temp, ... and
mf_all_on_proxy (all factors, fitted on the proxy mu, scored on true mu; it measures the\nproxy-vs-game scale mismatch, not shape quality).\nCaveat: for a single-file car (leave-one-lap-out) the held-out lap still contributes to the\nfile-level load reference (input normalisation only, not target leakage).
The status column says what the factor gate decided in the training fits
(fit = fitted, NI = not identifiable, AB = at bound), so "not identifiable" is not
mistaken for "fitted but no gain".
"""
import argparse
import fnmatch
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "backend"))
from app import telemetry, tire_mf, tire_model as tm  # noqa: E402
from app.car_identity import identify_car  # noqa: E402
from app.tire_bins import LapBins, bin_axle_samples, cell_arrays  # noqa: E402

HELDOUT_MIN_SLIP_DEG = 1.5
HELDOUT_MIN_TRUE_N = 3
TIRE_LAP_MAX_RATIO = 1.15  # same as grip_profile.TIRE_LAP_MAX_RATIO
STATE_KEYS = ("load_ratio", "temp_delta_c", "pressure_kpa", "wear", "camber_deg")

# (name, factors, fitted target). None factors = not a Magic Formula fit.
MF_VARIANTS = [
    ("mf_slip_only", (), "true"),
    ("+load", ("load",), "true"),
    ("+load+temp", ("load", "temp"), "true"),
    ("+load+temp+pressure", ("load", "temp", "pressure"), "true"),
    ("+...+wear", ("load", "temp", "pressure", "wear"), "true"),
    ("+...+camber", tire_mf.ALL_FACTORS, "true"),
    ("mf_all_on_proxy", tire_mf.ALL_FACTORS, "mu"),
]
VARIANT_NAMES = ["binned_v2", "binned_v2_proxy"] + [v[0] for v in MF_VARIANTS]


# ------------------------------------------------------------------ scoring ------------

def _heldout_cells(bins: LapBins):
    a = cell_arrays(bins)
    ok = ((a["slip"] >= HELDOUT_MIN_SLIP_DEG) & (a["true_n"] >= HELDOUT_MIN_TRUE_N)
          & np.isfinite(a["true_mean"]))
    return {k: v[ok] for k, v in a.items()}, ok


def _sums(pred, y, w):
    """(sum w, sum w*err^2, sum w*(y-ybar)^2) of one fold; the building blocks of R^2."""
    sw = float(w.sum())
    ybar = float(np.sum(w * y) / sw)
    return sw, float(np.sum(w * (y - pred) ** 2)), float(np.sum(w * (y - ybar) ** 2))


def heldout_r2(fit, bins: LapBins, predictor=None) -> dict | None:
    """Weighted R^2 and RMSE of `fit` (an MFFit) or `predictor(slip) -> mu` on the
    held-out cornering cells of `bins` against the game's true mu. None if no cell
    qualifies."""
    a, ok = _heldout_cells(bins)
    if not ok.any():
        return None
    w, y = a["true_n"], a["true_mean"]
    if predictor is not None:
        pred = predictor(a["slip"])
    else:
        pred = tire_mf.predict(fit, a["slip"], {k: a[k] for k in STATE_KEYS})
    sw, sse, sst = _sums(pred, y, w)
    return {"r2": 1.0 - sse / sst if sst > 0 else float("nan"),
            "rmse": float(np.sqrt(sse / sw)), "n_cells": int(ok.sum()), "sw": sw,
            "sse": sse, "sst": sst}


def binned_predictor(bins: LapBins, target: str):
    """v2-style predictor: count-weighted mean of mu per 0.4 deg slip bin of the training
    data, linearly interpolated between bin centres, nearest bin outside the range."""
    a = cell_arrays(bins)
    if target == "true":
        y, n = a["true_mean"], a["true_n"]
    else:
        y, n = a["mu_mean"], a["n"]
    ok = np.isfinite(y) & np.isfinite(a["slip"]) & (n > 0)
    y, n, slip = y[ok], n[ok], a["slip"][ok]
    idx = np.floor(slip / 0.4).astype(int)
    centres, means = [], []
    for b in np.unique(idx):
        m = idx == b
        if n[m].sum() < tm.MIN_BIN_SAMPLES:
            continue
        centres.append(np.average(slip[m], weights=n[m]))
        means.append(np.average(y[m], weights=n[m]))
    if len(centres) < 2:
        return None
    centres, means = np.array(centres), np.array(means)
    return lambda s: np.interp(np.asarray(s, float), centres, means)


def _status_str(statuses: list) -> str:
    """Per factor the gate outcome over the training fits, e.g. 'load:fit temp:NI'."""
    short = {"fitted": "fit", "not_identifiable": "NI", "at_bound": "AB"}
    out = []
    for f in tire_mf.ALL_FACTORS:
        c = Counter(short.get(s.get(f), "?") for s in statuses if s.get(f) != "disabled")
        if c:
            out.append(f"{f}:" + ("/".join(f"{k}x{v}" for k, v in c.items())
                                  if len(c) > 1 else next(iter(c))))
    return " ".join(out) or "-"


def score_axle(units: list) -> dict:
    """units: [(fold_id, LapBins)]. Leave-one-fold-out over fold_id.

    Folds without scorable held-out cells are counted (`no_cells`), not silently
    dropped. Every variant's headline R2/RMSE/cells is pooled over the COMMON folds -
    those where ALL variants produced a fit - so the variants and the dR2 column are
    compared on identical held-out data. `r2_all` keeps the pool over every fold that
    variant could score. Returns {variant: {...}} plus n_folds / no_cells / n_scored /
    n_common.
    """
    fold_ids = sorted({f for f, _ in units})
    per_fold, statuses, no_cells = [], {v: [] for v in VARIANT_NAMES}, 0
    for held in fold_ids:
        test = LapBins.merge([b for f, b in units if f == held])
        train_laps = [b for f, b in units if f != held]
        if not train_laps:
            no_cells += 1
            continue
        if not _heldout_cells(test)[1].any():
            no_cells += 1  # nothing to score on this fold, for any variant
            continue
        train = LapBins.merge(train_laps)
        row = {}
        for name, target in (("binned_v2", "true"), ("binned_v2_proxy", "mu")):
            pred = binned_predictor(train, target)
            row[name] = heldout_r2(None, test, pred) if pred else None
        for name, factors, target in MF_VARIANTS:
            fit = tire_mf.fit_mf(train, target=target, factors=factors)
            row[name] = heldout_r2(fit, test) if fit else None
            if fit:
                statuses[name].append(fit.factors_status)
        per_fold.append(row)
    common = [r for r in per_fold if all(r[v] is not None for v in VARIANT_NAMES)]

    def pool(rows):
        sw = sum(r["sw"] for r in rows); sse = sum(r["sse"] for r in rows)
        sst = sum(r["sst"] for r in rows)
        return (1.0 - sse / sst if sst > 0 else float("nan"),
                float(np.sqrt(sse / sw)) if sw > 0 else float("nan"),
                sum(r["n_cells"] for r in rows))

    out = {"n_folds": len(fold_ids), "no_cells": no_cells, "n_scored": len(per_fold),
           "n_common": len(common)}
    for name in VARIANT_NAMES:
        ok = [r[name] for r in per_fold if r[name] is not None]
        r2, rmse, cells = pool([r[name] for r in common])
        out[name] = {
            "r2": r2, "rmse": rmse, "n_cells": cells,
            "r2_all": pool(ok)[0], "folds": len(ok), "skipped": len(per_fold) - len(ok),
            "status": _status_str(statuses[name]),
        }
    return out


def format_table(title: str, res: dict) -> str:
    lines = [f"#### {title}",
             f"folds: {res['n_folds']} total, {res['no_cells']} without scorable held-out "
             f"cells, {res['n_scored']} scored, {res['n_common']} common (every variant fitted)",
             "",
             "R2 / dR2 / RMSE / cells are pooled over the common folds; R2(all) over every fold "
             "the variant could score. mf_all_on_proxy fits the proxy mu and is scored on true "
             "mu, so it measures the proxy-vs-game scale mismatch, not shape quality.",
             "",
             "| variant | R2 | dR2 vs prev MF | RMSE | cells | R2(all) | folds ok/skipped "
             "| factor status |",
             "|---|---|---|---|---|---|---|---|"]
    prev = None
    for name in VARIANT_NAMES:
        d = res[name]
        is_mf = name not in ("binned_v2", "binned_v2_proxy")
        delta = ""
        if is_mf and name != "mf_all_on_proxy":
            delta = "" if prev is None else f"{d['r2'] - prev:+.3f}"
            prev = d["r2"]
        lines.append(f"| {name} | {d['r2']:.3f} | {delta} | {d['rmse']:.3f} | {d['n_cells']} "
                     f"| {d['r2_all']:.3f} | {d['folds']}/{d['skipped']} | {d['status']} |")
    return "\n".join(lines)

# ------------------------------------------------------------------ data ---------------

def file_lap_bins(path) -> tuple:
    """(CarIdentity, {axle: [(lap_number, LapBins)]}) for one recording,
    with the same lap selection as GripProfile.ingest_tire_laps. Laps without
    any true-mu sample are kept out (nothing to score or fit on the true target)."""
    laps = telemetry.load_telemetry_file(path, grid_hz=tm.TIRE_GRID_HZ)
    ident = identify_car(laps[0].meta, laps[0].car_name) if laps else None
    valid = [l for l in laps if l.lap_time_s < 600]
    out = {"front": [], "rear": []}
    if not valid:
        return ident, out
    fastest = telemetry.reference_lap_time(valid) or min(l.lap_time_s for l in valid)
    calibration = {}
    for lap in valid:
        cal = tm.fit_deflection_calibration(lap)
        if cal:
            calibration = cal
            break
    load_ref = tm.file_load_reference(valid, calibration)
    for lap in valid:
        if lap.lap_time_s > fastest * TIRE_LAP_MAX_RATIO:
            continue
        samples = tm._lap_axle_samples(lap, calibration, load_ref)
        if samples is None:
            continue
        for axle, s in samples.items():
            bins = bin_axle_samples(s, s["mask"])
            if any(c.n_true for c in bins.cells.values()):
                out[axle].append((lap.lap_number, bins))
    return ident, out


def _peek_identity(path):
    meta = telemetry.anonymize_metadata(path)
    return identify_car(meta, meta.get("CarName", "unknown"))


def collect(sessions_dir: Path, car_class: str, pattern: str | None) -> dict:
    """{model: {axle: [(file name, lap number, LapBins)]}} for the class."""
    cars: dict = {}
    for path in sorted(sessions_dir.glob("*.duckdb")):
        if path.name.startswith("."):
            continue
        if pattern and not fnmatch.fnmatch(path.name, pattern):
            continue
        try:
            if car_class and _peek_identity(path).car_class != car_class:
                continue
            ident, per_axle = file_lap_bins(path)
            if ident is None:
                print(f"skip {path.name}: no laps", file=sys.stderr)
                continue
        except Exception as e:  # unreadable/partial recording: say so, go on
            print(f"skip {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for axle, lst in per_axle.items():
            for lap_no, bins in lst:
                cars.setdefault(ident.model, {"front": [], "rear": []})[axle].append(
                    (path.name, lap_no, bins))
        print(f"loaded {path.name}: {ident.model} "
              f"{len(per_axle['front'])} front / {len(per_axle['rear'])} rear laps with true mu",
              file=sys.stderr)
    return cars


def run(sessions_dir: Path, car_class: str = "Hypercar", pattern: str | None = None) -> str:
    t0 = time.time()
    cars = collect(sessions_dir, car_class, pattern)
    parts = []
    for model in sorted(cars):
        for axle in ("front", "rear"):
            rows = cars[model][axle]
            files = sorted({f for f, _, _ in rows})
            if not rows:
                continue
            by_file = len(files) > 1
            units = [((f if by_file else f"{f}#{lap}"), b) for f, lap, b in rows]
            res = score_axle(units)
            fmt = "leave-one-file-out" if by_file else "leave-one-lap-out"
            parts.append(format_table(
                f"{model} - {axle} axle, {fmt}, {len(files)} file(s), {len(rows)} laps", res))
    if not parts:
        parts.append("no recordings with true tyre forces matched")
    parts.append(f"wall time: {time.time() - t0:.0f} s")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", required=True, type=Path)
    ap.add_argument("--class", dest="car_class", default="Hypercar")
    ap.add_argument("--files", default=None, help="glob on file names, e.g. '*Valkyrie*'")
    args = ap.parse_args()
    print(run(args.sessions_dir, args.car_class, args.files))


if __name__ == "__main__":
    main()
