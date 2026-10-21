"""
Class-collapse experiment for tyre model v3 (read-only).

    python -I backend/tools/class_collapse.py --sessions-dir <dir> --class LMGT3

Question: how much should a car lean on the pooled curve of the OTHER cars of its class?
LMGT3 cars record no tyre forces, so the target is the PROXY mu (lateral G / load).

Per axle and car:
  - standalone MF fit of all its laps: peak slip / peak mu at the class reference state
    (the pooled prior's state), and the offset vs the class prior's peak;
  - leave-one-lap-out held-out R^2 on the proxy mu for prior weights
    {0, 0.3, 1, 3, 10}. The prior is fit_class_prior of the OTHER cars only - never the
    held-out car. Score = count-weighted R^2 over held-out cells with slip >= 1.5 deg
    and n >= 3, summed per fold (as in mf_experiment) and pooled over the folds where
    every weight produced a fit, so the weights are compared on identical data.
A car with a single lap has no training lap in a leave-one-out fold and is listed as
not scorable. The best weight is the one with the highest mean R^2 over scored cars.

CAVEAT: the cars of a class were recorded on different tracks, so the track is
confounded with the car - an offset vs the class prior can be the track.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_experiment as mfx  # noqa: E402  (file loading, lap selection, held-out scoring)

tire_mf = mfx.tire_mf
from app.tire_bins import LapBins  # noqa: E402
from app.tire_class import fit_class_prior  # noqa: E402

WEIGHTS = (0.0, 0.3, 1.0, 3.0, 10.0)


def score_car(laps: list, prior) -> dict:
    """Leave-one-lap-out held-out proxy-mu scores of one car for every weight.

    Returns {"folds": scored folds, "weights": {w: {"r2", "rmse"}}} - r2 is nan when no
    fold could be scored for every weight (single-lap car, thin data, no prior).
    """
    folds = []
    for i, held in enumerate(laps):
        train_laps = [l for j, l in enumerate(laps) if j != i]
        if not train_laps or not mfx._heldout_cells(held, "mu")[1].any():
            continue
        train = LapBins.merge(train_laps)
        row = {}
        for w in WEIGHTS:
            fit = tire_mf.fit_mf(train, prior=prior if w > 0 else None, prior_weight=w)
            row[w] = mfx.heldout_r2(fit, held, target="mu") if fit else None
        folds.append(row)
    common = [r for r in folds if all(r[w] is not None for w in WEIGHTS)]
    out = {"folds": len(common), "weights": {}}
    for w in WEIGHTS:
        sw = sum(r[w]["sw"] for r in common)
        sse = sum(r[w]["sse"] for r in common)
        sst = sum(r[w]["sst"] for r in common)
        out["weights"][w] = {
            "r2": 1.0 - sse / sst if common and sst > 0 else float("nan"),
            "rmse": float(np.sqrt(sse / sw)) if common and sw > 0 else float("nan")}
    return out


def class_measure(car_laps: dict) -> str:
    """The measure most laps of the class carry; laps of another measure are not mixed in."""
    c = Counter(l.measure for laps in car_laps.values() for l in laps)
    return c.most_common(1)[0][0]


def analyse_axle(car_laps: dict) -> dict:
    """{car: [LapBins]} of one axle -> {"measure", "prior", "cars": {car: {...}}, "best"}."""
    measure = class_measure(car_laps)
    # Laps of another measure (no load -> lateral_g) cannot be mixed in; say so per car.
    dropped = {c: sum(1 for l in laps if l.measure != measure) for c, laps in car_laps.items()}
    dropped = {c: n for c, n in dropped.items() if n}
    car_laps = {c: [l for l in laps if l.measure == measure] for c, laps in car_laps.items()}
    car_laps = {c: laps for c, laps in car_laps.items() if laps}
    class_prior = fit_class_prior(car_laps, measure)
    rows = {}
    for car, laps in car_laps.items():
        others = {c: l for c, l in car_laps.items() if c != car}
        prior = fit_class_prior(others, measure) if others else None
        own = tire_mf.fit_mf(LapBins.merge(laps))
        row = {"laps": len(laps), "samples": sum(l.n_samples() for l in laps),
               "own": None, "score": score_car(laps, prior) if prior else None}
        if own is not None and class_prior is not None:
            s, mu = tire_mf.peak_at(own, class_prior.ref_state)
            ps, pmu = tire_mf.peak_at(class_prior)
            row["own"] = {"peak_slip": s, "peak_mu": mu, "d_slip": s - ps, "d_mu": mu - pmu,
                          "observed": own.peak_observed}
        rows[car] = row
    best = None
    scored = [r["score"] for r in rows.values() if r["score"] and r["score"]["folds"]]
    if scored:
        means = {w: float(np.nanmean([s["weights"][w]["r2"] for s in scored])) for w in WEIGHTS}
        best = {"means": means, "weight": max(means, key=means.get), "n_cars": len(scored)}
    return {"measure": measure, "prior": class_prior, "cars": rows, "best": best,
            "dropped": dropped}


def format_axle(car_class: str, axle: str, res: dict) -> str:
    prior = res["prior"]
    lines = [f"#### {car_class} - {axle} axle, proxy mu ({res['measure']})", ""]
    if prior is None:
        return "\n".join(lines + ["class prior: not fittable"])
    ps, pmu = tire_mf.peak_at(prior)
    lines += [f"class prior (all cars pooled, equal car weighting): peak {ps:.2f} deg / "
              f"mu {pmu:.2f}, {prior.n_samples} weighted samples, peak observed: "
              f"{prior.peak_observed}", "",
              "| car | laps | samples | own peak slip | own peak mu | d slip vs prior "
              "| d mu vs prior | folds | " + " | ".join(f"R2 w={w:g}" for w in WEIGHTS) + " |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(WEIGHTS)]
    for car in sorted(res["cars"]):
        r = res["cars"][car]
        o = r["own"]
        own = ("n/a | n/a | n/a | n/a" if o is None else
               f"{o['peak_slip']:.2f}{'' if o['observed'] else '*'} | {o['peak_mu']:.2f} "
               f"| {o['d_slip']:+.2f} | {o['d_mu']:+.2f}")
        sc = r["score"]
        if sc and sc["folds"]:
            cols = " | ".join(f"{sc['weights'][w]['r2']:.3f}" for w in WEIGHTS)
            folds = str(sc["folds"])
        else:
            cols = " | ".join("n/a" for _ in WEIGHTS)
            folds = "0 (single lap or no prior)" if r["laps"] < 2 else "0"
        lines.append(f"| {car} | {r['laps']} | {r['samples']} | {own} | {folds} | {cols} |")
    lines.append("")
    lines.append("(* = peak not observed in that car's data, i.e. extrapolated)")
    if res["dropped"]:
        lines.append("laps left out (other measure, e.g. no load channel): "
                     + ", ".join(f"{c}: {n}" for c, n in sorted(res["dropped"].items())))
    b = res["best"]
    if b:
        lines += ["", f"mean held-out R2 over {b['n_cars']} scored car(s): "
                  + ", ".join(f"w={w:g}: {m:.3f}" for w, m in b["means"].items()),
                  f"best weight ({axle}): {b['weight']:g}"]
    else:
        lines += ["", f"best weight ({axle}): not determinable (no scorable car)"]
    return "\n".join(lines)


def run(sessions_dir: Path, car_class: str) -> str:
    t0 = time.time()
    cars = mfx.collect(sessions_dir, car_class, None, require_true=False)
    parts, per_axle_best = [], {}
    for axle in ("front", "rear"):
        car_laps = {m: [b for _, _, b in v[axle]] for m, v in cars.items() if v[axle]}
        if not car_laps:
            continue
        res = analyse_axle(car_laps)
        parts.append(format_axle(car_class, axle, res))
        if res["best"]:
            per_axle_best[axle] = res["best"]
    if not parts:
        parts.append(f"no {car_class} recordings with usable laps")
    if per_axle_best:
        # Pool both axles: every scored (car, axle) counts once.
        tot = {w: sum(b["means"][w] * b["n_cars"] for b in per_axle_best.values())
               / sum(b["n_cars"] for b in per_axle_best.values()) for w in WEIGHTS}
        parts.append("overall mean held-out R2 (front+rear): "
                     + ", ".join(f"w={w:g}: {m:.3f}" for w, m in tot.items())
                     + f"\nBEST WEIGHT: {max(tot, key=tot.get):g}")
    parts.append(f"wall time: {time.time() - t0:.0f} s")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", required=True, type=Path)
    ap.add_argument("--class", dest="car_class", default="LMGT3")
    args = ap.parse_args()
    print(run(args.sessions_dir, args.car_class))


if __name__ == "__main__":
    main()
