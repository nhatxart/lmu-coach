"""Class-level prior for the tyre model (v3): cars of one class share tyre physics.

A car with one or two laps cannot pin a Magic Formula by itself. The other cars of its
class (e.g. every LMGT3) have a similar curve shape, so a pooled fit over THEIR laps is
a sensible prior that the car's own data then moves it away from.
"""

import numpy as np

from .tire_bins import _CELL_FIELDS, Cell, LapBins
from .tire_mf import bootstrap_fit, fit_mf

# Strength of the pull toward the class prior (units of fit_mf's prior_weight).
# Evidence (class_collapse.py, leave-one-lap-out, DevNotes 10/19): the best weight was 0.
# LMGT3 mean held-out R2 0.531 at weight 0 vs 0.522 at 0.3; Hypercar 0.332 vs 0.263; the
# Mustang at 4 laps got worse with a prior. A prior is therefore kept ONLY for cars with
# fewer than PRIOR_MAX_LAPS laps, where same-file leave-one-lap-out cannot measure its
# benefit (there is nothing to hold out), at a deliberately weak weight.
CLASS_PRIOR_WEIGHT = 0.3

# At this many laps or more the car's own data is trusted alone and no prior (class or
# older-version) is applied.
PRIOR_MAX_LAPS = 3

PRIOR_KINDS = ("class", "older_version")


def _car_laps(laps, measure):
    """Laps of one car that carry `measure` (an empty lap has no measure and is skipped)."""
    return [l for l in laps if l.cells and l.measure == measure]


def _weighted_pool(per_car: list, weights: list) -> LapBins:
    """Sum of several LapBins, each scaled by a factor.

    WHY a copy with float cells instead of LapBins.merge: the scale factors are
    fractional (median n / car n), merge only adds, and the inputs must stay untouched.
    Cell fields are plain sums, so scaling a cell scales its weight in the fit exactly.
    """
    out = LapBins(per_car[0].measure)
    for bins, w in zip(per_car, weights):
        for key, c in bins.cells.items():
            t = out.cells.setdefault(key, Cell())
            for f in _CELL_FIELDS:
                setattr(t, f, getattr(t, f) + w * getattr(c, f))
    return out


def fit_class_prior(car_bins: dict, measure: str):
    """Pooled fit over all cars of a class: {car: [LapBins per lap]} -> MFFit | None.

    Cars are weighted equally: each car's counts are scaled by (median car total n) /
    (that car's total n) before pooling, so a car with ten times the samples does not
    dominate the class curve. Cars with no lap of `measure` are left out.
    """
    merged = {}
    for car, laps in car_bins.items():
        usable = _car_laps(laps, measure)
        if usable:
            merged[car] = LapBins.merge(usable)
    if not merged:
        return None
    totals = {car: b.n_samples() for car, b in merged.items()}
    median_n = float(np.median(list(totals.values())))
    cars = sorted(merged)
    pooled = _weighted_pool([merged[c] for c in cars],
                            [median_n / totals[c] for c in cars])
    return fit_mf(pooled)


def fit_car_with_prior(lap_bins: list, prior, weight: float = CLASS_PRIOR_WEIGHT,
                       n_boot: int = 50, prior_kind: str = "class") -> dict:
    """bootstrap_fit of one car's laps pulled toward `prior`.

    Same shape as bootstrap_fit plus "prior_used": `prior_kind` ("class", or
    "older_version" when the prior is the car's own fit from an earlier game version) if
    the prior actually entered the fit, else "none" - no prior, zero weight, or one
    fit_mf ignored as incompatible (measure/target mismatch), or the car already has
    PRIOR_MAX_LAPS laps or more (Ruling 11b: the prior did not help cars with data).
    """
    if prior_kind not in PRIOR_KINDS:
        raise ValueError(f"prior_kind must be one of {PRIOR_KINDS}")
    n_laps = sum(1 for l in lap_bins if l.cells)
    usable = prior is not None and weight > 0 and n_laps < PRIOR_MAX_LAPS
    out = bootstrap_fit(lap_bins, n_boot=n_boot, prior=prior if usable else None,
                        prior_weight=weight if usable else 0.0)
    fit = out["fit"]
    out["prior_used"] = (prior_kind if usable and fit is not None
                         and not any(n.startswith("prior ignored") for n in fit.notes)
                         else "none")
    return out
