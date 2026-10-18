"""
Read-only accuracy check of one car's tyre model (writes nothing).

    python -I backend/tools/validate_tire_model.py "WRT"            # car-name substring
    python -I backend/tools/validate_tire_model.py "Cadillac" --boot 100

Uses exactly the laps the profile's tyre model contains (tire_model_laps),
re-extracted from data/sessions the same way ingest_tire_laps does.

(a) GROUND TRUTH: where the file recorded Tire Lateral Force + Tire Load
    (10/2-10/4 Aston #007 / Cadillac #12 so far), the game's own axle
    mu = sum|Fy| / sum Fz against the model's proxy (lat G / load_ratio).
    Samples with axle load < 30 % of its median are masked: Tire Load hits
    0 N over kerbs and Fy/0 wrecked the first attempt (r 0.02).
(b) HELD-OUT PREDICTION: the binned curve fitted on all other laps (or
    files) predicts each held-out sample's grip from its slip angle.
    Cornering samples only (slip > 1.5 deg) - including straights makes R2
    look great for free, since grip ~ 0 at slip ~ 0. R2 is vs the held-out
    mean.
(c) BOOTSTRAP: refit on laps resampled with replacement; spread of the peak
    and available grip, and how often the peak counts as observed.
(d) CE vs LAP TIME: each real lap's cornering efficiency, scored against a
    model WITHOUT that lap, Spearman-correlated with lap time per file.

Ends with a markdown row for the scorecard in DevNotes.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app import grip_profile as gp, telemetry, tire_model as tm  # noqa: E402

CORNER_MIN_SLIP_DEG = 1.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("car", help="substring of the profile file name, e.g. WRT, Cadillac, 007_WEC")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples (default 200)")
    ap.add_argument("--sessions-dir", type=Path, default=REPO / "data" / "sessions")
    ap.add_argument("--profiles-dir", type=Path, default=REPO / "data" / "grip_profiles")
    args = ap.parse_args()

    # (a) needs the force channel, which the backend does not normally load
    if "Tire Lateral Force" not in telemetry.WHEEL_CHANNELS:
        telemetry.WHEEL_CHANNELS.append("Tire Lateral Force")

    matches = [p for p in args.profiles_dir.glob("*.json")
               if not p.name.startswith(".") and args.car.lower() in p.name.lower()]
    if len(matches) != 1:
        print(f"'{args.car}' matches {len(matches)} profiles: {[p.name for p in matches]}")
        return 1
    prof = gp.GripProfile.from_dict(json.loads(matches[0].read_text()))
    cal = prof.deflection_calibration
    keys = set(prof.tire_model_laps)
    files = sorted({k.rsplit("#", 1)[0] for k in keys})
    print(f"# {prof.car_name}: {len(keys)} model laps in {len(files)} files\n")

    per_lap, info, ref_time, file_ref = {}, {}, {}, {}
    for f in files:
        path = args.sessions_dir / f
        if not path.exists():
            print(f"  missing session file, skipped: {f}")
            continue
        laps = [l for l in telemetry.load_telemetry_file(path) if l.lap_time_s < 600]
        ref_time[f] = telemetry.reference_lap_time(laps)
        # whole-file load reference, like ingest_tire_laps (only the out lap
        # has slow straight-line running at most tracks)
        file_ref[f] = tm.file_load_reference(laps, cal)
        for lap in laps:
            key = f"{f}#{lap.lap_number}"
            if key not in keys:
                continue
            s = tm._lap_axle_samples(lap, cal, file_ref[f])
            if s is None:
                continue
            per_lap[key] = {ax: _axle_arrays(lap, s[ax], wheels)
                            for ax, wheels in (("front", ("FL", "FR")), ("rear", ("RL", "RR")))}
            info[key] = (f, lap)
    if not per_lap:
        print("no usable laps")
        return 1
    print(f"laps with samples: {len(per_lap)}; samples/axle: "
          f"{sum(len(e['front']['slip_angle_deg']) for e in per_lap.values())}")

    full = {ax: _model(per_lap, list(per_lap), ax) for ax in ("front", "rear")}
    for ax, m in full.items():
        print(f"model {ax}: peak {m['peak']} deg observed={m['peak_observed']} range {m['efficient_range']} "
              f"avail {m['available_grip']} {m['grip_measure']} confidence {m['confidence']} "
              f"tyre temp vs optimal {m['tyre_temperature'].get('median_delta_c')} C")
    row = {}

    # ── (a) ──
    print("\n(a) GROUND TRUTH vs the game's tyre forces")
    a_cells = []
    for ax in ("front", "rear"):
        arr = _concat(per_lap, [k for k in per_lap if "mu_true" in per_lap[k][ax]], ax)
        if not arr:
            print("  no Tire Lateral Force / Tire Load in these files - not testable")
            a_cells = ["not testable (no forces)"]
            break
        proxy, ok = _grip(arr, full[ax]["grip_measure"])
        t = arr["mu_true"]
        ok &= np.isfinite(t)
        r = np.corrcoef(proxy[ok], t[ok])[0, 1]
        rho = spearmanr(proxy[ok], t[ok])[0]
        ratio = np.median(proxy[ok] / np.maximum(t[ok], 1e-3))
        _, st = _curve(arr["slip_angle_deg"][ok], t[ok])
        _, sp = _curve(arr["slip_angle_deg"][ok], proxy[ok])
        n = min(len(st), len(sp))
        shape = np.corrcoef(st[:n], sp[:n])[0, 1]
        ft = tm._fit_curve(arr["slip_angle_deg"][ok], t[ok])
        fp = tm._fit_curve(arr["slip_angle_deg"][ok], proxy[ok])
        pt, pp = (ft or {}).get("peak"), (fp or {}).get("peak")
        print(f"  {ax}: n={ok.sum()} r={r:.3f} spearman={rho:.3f} median proxy/game={ratio:.3f} "
              f"curve-shape r={shape:.3f}; peak game {pt} vs proxy {pp}")
        a_cells.append(f"{ax[0].upper()} {r:.2f}, {shape:.3f}, {pt} vs {pp} (proxy/game {ratio:.2f})")
    row["a"] = "; ".join(a_cells)

    # ── (b) ──
    print(f"\n(b) HELD-OUT PREDICTION, cornering samples (slip > {CORNER_MIN_SLIP_DEG} deg)")
    splits = [("leave-one-lap-out", [[k] for k in per_lap])]
    if len(files) > 1:
        splits.append(("leave-one-file-out", [[k for k in per_lap if info[k][0] == f] for f in files]))
    for name, groups in splits:
        for ax in ("front", "rear"):
            rmse, bias, r2, n = _heldout(per_lap, groups, ax, full[ax]["grip_measure"])
            print(f"  {name:19s} {ax}: R2 {r2:.3f} RMSE {rmse:.3f} bias {bias:+.3f} (n={n})")
            if name == "leave-one-lap-out":
                row[f"b_{ax}"] = f"{r2:.2f} / {rmse:.2f}"

    # ── (c) ──
    print(f"\n(c) BOOTSTRAP over laps ({args.boot} resamples)")
    rng = np.random.default_rng(0)
    ks = list(per_lap)
    c_cells = []
    for ax in ("front", "rear"):
        peaks, avail, observed = [], [], []
        for _ in range(args.boot):
            m = _model(per_lap, list(rng.choice(ks, len(ks), replace=True)), ax)
            if m["peak"] is not None:
                peaks.append(m["peak"]); avail.append(m["available_grip"]); observed.append(m["peak_observed"])
        if not peaks:
            print(f"  {ax}: no peak in any resample")
            c_cells.append(f"{ax[0].upper()} none")
            continue
        p = np.array(peaks)
        lo, hi = np.percentile(p, 5), np.percentile(p, 95)
        print(f"  {ax}: peak median {np.median(p):.2f} deg, 90% [{lo:.2f}, {hi:.2f}], sd {p.std():.2f}; "
              f"available grip 90% [{np.percentile(avail, 5):.3f}, {np.percentile(avail, 95):.3f}]; "
              f"peak observed in {np.mean(observed):.0%}")
        c_cells.append(f"{ax[0].upper()} {lo:.1f}-{hi:.1f} (obs {np.mean(observed):.0%})")
    row["c"] = ", ".join(c_cells)

    # ── (d) ──
    print("\n(d) CORNERING EFFICIENCY vs LAP TIME (each lap vs a model without it)")
    d_cells = []
    for f in files:
        rows = []
        for k, (kf, lap) in info.items():
            if kf != f or not lap.is_complete or ref_time.get(f) is None:
                continue
            if not (0.85 * ref_time[f] <= lap.lap_time_s <= 1.15 * ref_time[f]):
                continue
            others = [x for x in per_lap if x != k]
            gu = tm.compute_grip_utilization_for_lap(
                lap, _model(per_lap, others, "front"), _model(per_lap, others, "rear"),
                calibration=cal, load_ref=file_ref[f])
            if gu and gu["cornering_efficiency"] is not None:
                rows.append((lap.lap_time_s, gu["cornering_efficiency"], gu["front_grip_utilization"]))
        if len(rows) < 5:
            print(f"  {f[:64]}: {len(rows)} laps - too few")
            continue
        t, ce, guf = (np.array(c) for c in zip(*rows))
        rho, p = spearmanr(ce, t)
        rho2, p2 = spearmanr(guf, t)
        print(f"  {f[:64]}: n={len(rows)} CE {ce.min():.3f}-{ce.max():.3f} "
              f"rho(CE, time) {rho:+.2f} (p {p:.2f}); rho(front GU, time) {rho2:+.2f} (p {p2:.2f})")
        d_cells.append(f"rho {rho:+.2f} (p {p:.2f}, n {len(rows)})")
    print("  negative rho = higher value, faster lap")
    row["d"] = "; ".join(d_cells) or "too few laps per file"

    print("\nScorecard row (DevNotes.md):")
    print(f"| {prof.car_name} | {len(per_lap)} / {len(_concat(per_lap, ks, 'front')['slip_angle_deg'])} "
          f"| {row['a']} | {row['b_front']}, {row['b_rear']} | {row['c']} | {row['d']} |")
    return 0


# ── helpers ──────────────────────────────────────────────────────────────────

def _axle_arrays(lap, s: dict, wheels) -> dict:
    m = s["mask"]
    a = {n: (np.asarray(s[n])[m] if s[n] is not None else np.full(int(m.sum()), np.nan))
         for n in tm._OBS_FIELDS}
    fy = [lap.channels.get(f"Tire Lateral Force {w}") for w in wheels]
    fz = [lap.channels.get(f"Tire Load {w}") for w in wheels]
    if all(x is not None for x in fy + fz) and np.nanmax(np.abs(fy[0])) > 1:
        fz_ax = fz[0] + fz[1]
        mu = (np.abs(fy[0]) + np.abs(fy[1])) / np.maximum(fz_ax, 1.0)
        mu[fz_ax < 0.3 * np.nanmedian(fz_ax)] = np.nan
        a["mu_true"] = mu[m]
    return a


def _concat(per_lap, ks, axle) -> dict:
    out = defaultdict(list)
    for k in ks:
        for n, v in per_lap[k][axle].items():
            out[n].append(v)
    return {n: np.concatenate(v) for n, v in out.items()} if out else {}


def _grip(arr, measure):
    if measure == "mu_load_normalized":
        lr = arr["load_ratio"]
        ok = np.isfinite(lr)
        return arr["lateral_g"] / np.maximum(np.where(ok, lr, 1.0), 0.2), ok
    return arr["lateral_g"], np.ones(len(arr["lateral_g"]), bool)


def _curve(sa, g):
    """Same binning + smoothing as tire_model._fit_curve, but returns the curve."""
    sa = np.abs(sa)
    ok = np.isfinite(sa) & np.isfinite(g) & (sa < 20)
    sa, g = sa[ok], g[ok]
    edges = np.arange(0.0, min(sa.max(), 15.0) + tm.SLIP_ANGLE_BIN_DEG, tm.SLIP_ANGLE_BIN_DEG)
    idx = np.digitize(sa, edges)
    c, mu = [], []
    for b in range(1, len(edges)):
        mk = idx == b
        if mk.sum() >= tm.MIN_BIN_SAMPLES:
            c.append((edges[b - 1] + edges[b]) / 2)
            mu.append(g[mk].mean())
    mu = np.array(mu)
    return np.array(c), np.convolve(np.concatenate([[mu[0]], mu, [mu[-1]]]), [.25, .5, .25], "valid")


def _model(per_lap, ks, axle) -> dict:
    arr = _concat(per_lap, ks, axle)
    obs = tm.AxleSlipObservations()
    obs.add_samples(**{n: arr[n] for n in tm._OBS_FIELDS})
    return tm.build_axle_tire_model(obs)


def _heldout(per_lap, groups, ax, measure):
    errs, ys = [], []
    for test in groups:
        test_set = set(test)
        tr = _concat(per_lap, [k for k in per_lap if k not in test_set], ax)
        te = _concat(per_lap, test, ax)
        if not te or not len(te["slip_angle_deg"]):
            continue
        gtr, oktr = _grip(tr, measure)
        gte, okte = _grip(te, measure)
        c, s = _curve(tr["slip_angle_deg"][oktr], gtr[oktr])
        sa, y = np.abs(te["slip_angle_deg"][okte]), gte[okte]
        # inside the fitted range only: np.interp would clamp spins at 15+ deg
        # to the last bin and fake a bias
        keep = np.isfinite(sa) & np.isfinite(y) & (sa >= c[0]) & (sa <= c[-1]) & (sa > CORNER_MIN_SLIP_DEG)
        errs.append(y[keep] - np.interp(sa[keep], c, s))
        ys.append(y[keep])
    e, y = np.concatenate(errs), np.concatenate(ys)
    return (float(np.sqrt(np.mean(e ** 2))), float(np.mean(e)),
            float(1 - np.mean(e ** 2) / np.mean((y - y.mean()) ** 2)), len(e))


if __name__ == "__main__":
    sys.exit(main())
