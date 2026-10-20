"""
Probabilistic tire model, grip utilization, and cornering efficiency.

── Why this can't be the literal Fx/Fy force-based formula ──────────────────
The originally-proposed grip utilization formula is:

    GU_i = sqrt(Fx_i^2 + Fy_i^2) / F_available_i

This requires actual per-tire force channels (Fx, Fy) and a known available-
grip ceiling. LMU does NOT expose these for most cars: Tire Lateral Force /
Tire Longitudinal Force / Grip Fraction come back from LMU's mmap as flat
zero (the recorder marks them "unpopulated_by_lmu" rather than write fake
data), and LMU's own native .duckdb export has no such tables at all.
Tire Load is populated for some cars (Aston #007, Cadillac #12 in the 10/2-
10/4 recordings) and zero for others (BMW M4 LMGT3, 10/14).

── What we build instead ─────────────────────────────────────────────────────
Grip utilization here is a SLIP-ANGLE-RELATIVE proxy, not a force-ratio.
Tire lateral force is a function of slip angle that rises, peaks, and falls
off (the classic "tire slip curve") — a tire generating force near its OWN
empirically observed peak-force slip angle is near its own grip ceiling. We
recover that curve directly from real driving data by binning slip angle
against a grip measure, not from a textbook constant or a spec sheet.

── Probabilistic, not deterministic ──────────────────────────────────────────
The model accumulates per-car, per-axle observations in the JSON profile
grip_profile.py maintains; peak/efficient_range/confidence/sample_count are
recomputed from the full accumulated set on every read. Confidence starts
low and rises with sample_count; callers (coach.py's LLM prompt) are told the
confidence and whether the peak was actually observed.

── Model v2 (10/14/2026) — what changed and WHY ──────────────────────────────
Checked against 7 laps of the 10/14 Fuji BMW M4 LMGT3 file:
1. STEADY-STATE ONLY. v1 binned every sample above 40 km/h, so trail-braking,
   throttle-on exits (combined slip uses friction for Fx, lowering lateral G
   at a given slip angle), kerb strikes and yaw transients all smeared the
   curve. v2 keeps samples with |G long| < 0.25 g and |yaw accel| < 1 rad/s²
   (when recorded). The front curve sharpened from a broad 1.52 g plateau
   to a 1.62 g peak at 3-3.5°.
2. LOAD-NORMALIZED GRIP. G against slip angle mixes slow corners (little
   downforce, big slip angles) with fast corners (high load, small slip
   angles), so the "drop-off" past the peak was partly just slow corners.
   v2 stores each axle's vertical load relative to that file's own
   low-speed straight-line load (load_ratio) and fits mu = |lat G| /
   load_ratio. Units, motion ratio and car mass cancel in the ratio; what
   remains is: pushrod force excludes unsprung weight (a few % bias), and
   lateral G stands in for axle lateral force (exact in steady state with
   the static weight split — see _axle_load — which is why the steady
   filter matters). Load sources, best first: Tire Load, Susp Force,
   Susp Pos via a per-car deflection->force calibration learned from files
   that had Susp Force (deflection alone has an unknown zero-load offset,
   plus ARB/bump-stop/heave effects, so "deflection × constant spring rate"
   is not proportional to load without that calibration). Without any of
   these the fit falls back to raw lateral G and says so.
3. PEAK MUST BE OBSERVED. v1 took the argmax of the binned means, with the
   edge bins left unsmoothed: the WRT rear "peak" was the last bin, 5
   samples. v2 only lets bins with >= MIN_BIN_COUNT samples compete and
   reports peak_observed = False when the curve never falls >= 3% past its
   best bin. On the Fuji data the load-normalized front curve is still
   rising at 5.5° — the driver has not been past the tire's peak — and the
   model now says that instead of inventing one.
4. TYRE TEMPERATURE vs OPTIMAL. recorder_hf writes TyreOptimalTempC; every
   sample stores its temp minus optimal. The Fuji fronts ran 59-73 °C vs
   92 °C optimal, so that whole curve is a cold-tyre curve. When enough
   in-window samples exist the fit uses only those.
5. CORNERING EFFICIENCY uses matching statistics: lap p90 grip ÷ the model's
   p90 envelope in the efficient range (v1 divided a lap p95 by a bin MEAN,
   so an ordinary lap read ~1.08).
Old (v1) observations are discarded on load — see grip_profile.py.

── Model v3 (10/19/2026) — Magic Formula fit on state cells ─────────────────
The coach's tyre model is now a Magic Formula curve (tire_mf.py) fitted by
grip_profile.py to per-lap sufficient-statistic bins (tire_bins.py), with
bootstrap 90 % CIs over laps and a class prior (tire_class.py) when the car
has few laps. build_axle_tire_model(cached, lap_bins) keeps every v2 key and
adds model, peak_slip_ci90, peak_grip_ci90, factors, factor_effects,
prior_used, version_note, game_version and coverage. Only the load factor is
enabled; factor_effects reports peak mu at p10/median/p90 of the driver's own
data, never an extrapolation. When no MF fits, the v2 histogram curve is used
on the stored cells with confidence capped (model = "binned_v2_fallback").
Cornering efficiency divides by the MF peak at each sample's own load/temp
state when a fit is given (ce_method = "state_aware_mf"), else by the v2
envelope. v2's build_axle_tire_model(obs) is now build_v2_axle_model(obs)
(validate_tire_model.py compares against it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .telemetry import LapTelemetry, WHEEL_LABELS
from .tire_bins import BIN, UNKNOWN_BIN, LapBins, cell_arrays
from .tire_bins import coverage as tb_coverage
from .tire_mf import PEAK_GRID, STATE_KEY, MFFit, peak_at, predict

# Minimum speed to trust slip-angle/G readings for tire-model purposes —
# below this, steering noise and low-speed maneuvering (pit lane, hairpin
# creep) dominate and would corrupt the learned curve.
MIN_SPEED_KMH_FOR_MODEL = 40.0

# Steady-state cornering gate (see module docstring, point 1).
STEADY_MAX_LONG_G = 0.25
STEADY_MAX_YAW_ACCEL = 1.0  # rad/s^2; only applied when Yaw Accel was recorded

# Reference load window: straight-line, low-speed (little downforce) samples
# that define load_ratio = 1.0 for one lap.
REF_LOAD_SPEED_KMH = (40.0, 120.0)
REF_LOAD_MAX_G = 0.1
REF_LOAD_MIN_SAMPLES = 20

# Time grid the tyre model loads recordings at. PREVIOUSLY the 20 Hz coaching grid
# left only ~240 steady samples per lap on HF files, too few to populate the state
# cells. Bins are sufficient statistics and the bootstrap resamples whole laps, so
# the autocorrelated 50 Hz samples add information without faking independence.
TIRE_GRID_HZ = 50.0

# Bin width for the slip-angle -> grip histogram, degrees.
SLIP_ANGLE_BIN_DEG = 0.4
# A bin needs this many samples to count toward the curve at all, and
# MIN_BIN_COUNT to be eligible as the peak or as evidence of fall-off.
MIN_BIN_SAMPLES = 5
MIN_BIN_COUNT = 20
# Past-peak fall-off required before we call the peak "observed".
PEAK_FALLOFF_FRACTION = 0.97

# A model needs at least this many total observations before we report a
# peak/efficient_range at all — below this, we say so explicitly.
MIN_SAMPLES_FOR_MODEL = 300

# Confidence scale: sample_count needed to reach ~63% (1 - 1/e) of maximum
# confidence. Exponential saturation, never 1.0.
CONFIDENCE_SATURATION_SAMPLES = 4000.0

# |tyre temp - optimal| within this counts as "in the working window".
TEMP_WINDOW_C = 10.0

# Upper envelope percentile for "available grip" and for the lap's own
# achieved grip (same statistic on both sides of the CE ratio).
ENVELOPE_PERCENTILE = 90

_OBS_FIELDS = ("slip_angle_deg", "lateral_g", "tire_temp_c", "tire_pressure_kpa",
               "susp_deflection_m", "load_ratio", "temp_delta_c")


@dataclass
class AxleSlipObservations:
    """Raw per-sample observations accumulated for one axle (parallel lists)."""
    slip_angle_deg: list[float] = field(default_factory=list)
    lateral_g: list[float] = field(default_factory=list)  # |lat G|
    tire_temp_c: list[float] = field(default_factory=list)
    tire_pressure_kpa: list[float] = field(default_factory=list)
    susp_deflection_m: list[float] = field(default_factory=list)
    # axle vertical load ÷ that lap's low-speed straight-line axle load;
    # NaN when no load source was available
    load_ratio: list[float] = field(default_factory=list)
    # tyre temp minus the file's TyreOptimalTempC; NaN when unknown
    temp_delta_c: list[float] = field(default_factory=list)

    MAX_SAMPLES: int = 20000  # keep the most recent; see add_samples
    # Where THIS extraction's load came from ("Tire Load", "Susp Force",
    # "Susp Pos (calibrated to Susp Force)" or "none (...)"). Not a per-sample
    # list, never stored: grip_profile keeps it per lap (tire_model_load_sources)
    # so the model can say whether its mu rests on the real tyre load or on
    # the Susp Force stand-in LMU's GT3s force on us (Tire Load is all zero there).
    load_source: str | None = None

    def to_dict(self) -> dict:
        # rounded so a 20k-sample profile stays a few MB, not tens
        def _r(v):
            return [None if not np.isfinite(x) else round(float(x), 4) for x in v]
        return {name: _r(getattr(self, name)) for name in _OBS_FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "AxleSlipObservations":
        n = len(d.get("slip_angle_deg", []))
        kw = {}
        for name in _OBS_FIELDS:
            v = d.get(name) or []
            v = [float("nan") if x is None else float(x) for x in v]
            # older profiles lack the newer fields: pad so lists stay parallel
            kw[name] = v if len(v) == n else [float("nan")] * n
        return cls(**kw)

    def __len__(self) -> int:
        return len(self.slip_angle_deg)

    def add_samples(self, **arrays: np.ndarray | None) -> None:
        n_new = len(arrays["slip_angle_deg"])
        for name in _OBS_FIELDS:
            v = arrays.get(name)
            getattr(self, name).extend(
                np.asarray(v, dtype=float).tolist() if v is not None else [float("nan")] * n_new
            )
        # Cap: keep the most recent MAX_SAMPLES observations, so the model
        # slowly forgets old setups in favour of current driving. After a
        # big setup/compound change, deleting the profile is more honest
        # than blending regimes.
        for name in _OBS_FIELDS:
            lst = getattr(self, name)
            if len(lst) > self.MAX_SAMPLES:
                setattr(self, name, lst[-self.MAX_SAMPLES:])

    def extend(self, other: "AxleSlipObservations") -> None:
        self.add_samples(**{name: np.asarray(getattr(other, name)) for name in _OBS_FIELDS})

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name), dtype=float) for name in _OBS_FIELDS}


def _confidence_from_sample_count(n: int) -> float:
    """Exponential-saturation confidence in [0, 1); never exactly 1.0."""
    if n <= 0:
        return 0.0
    return round(float(1.0 - np.exp(-n / CONFIDENCE_SATURATION_SAMPLES)), 3)


def grip_values(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, str, np.ndarray]:
    """(per-sample grip, measure name, mask of samples that have it).
    mu = |lat G| / load_ratio on the samples that have a load ratio, if
    there are enough of them; otherwise raw |lat G| on all samples. Never
    mixed: a curve half in mu and half in G would be meaningless."""
    lr = arr["load_ratio"]
    has_load = np.isfinite(lr)
    if has_load.sum() >= MIN_SAMPLES_FOR_MODEL:
        return arr["lateral_g"] / np.maximum(np.where(has_load, lr, 1.0), 0.2), "mu_load_normalized", has_load
    return arr["lateral_g"], "lateral_g", np.ones(len(lr), dtype=bool)


def _fit_curve(slip_angle_deg: np.ndarray, grip: np.ndarray) -> dict | None:
    """
    Bins |slip angle| against grip and finds the peak. Returns None when
    there isn't enough data, else a dict with peak / efficient_range /
    peak_grip (smoothed bin mean at the peak) / available_grip (p90
    envelope in the efficient range) / peak_observed.

    BUG FIX (v1 rear peak 4.2° from a 5-sample last bin): edge bins used to
    skip smoothing and any bin with >= 5 samples could win, so a thin tail
    bin decided the peak. Now smoothing replicates the edges, only bins with
    MIN_BIN_COUNT samples can be the peak, and peak_observed says whether
    the curve actually fell off past it.
    """
    sa = np.abs(np.asarray(slip_angle_deg, dtype=float))
    g = np.asarray(grip, dtype=float)
    valid = np.isfinite(sa) & np.isfinite(g) & (sa < 20)
    sa, g = sa[valid], g[valid]
    if len(sa) < MIN_SAMPLES_FOR_MODEL:
        return None

    max_sa = min(float(sa.max()), 15.0)
    edges = np.arange(0.0, max_sa + SLIP_ANGLE_BIN_DEG, SLIP_ANGLE_BIN_DEG)
    if len(edges) < 4:
        return None
    idx = np.digitize(sa, edges)

    centers, means, counts, members = [], [], [], []
    for b in range(1, len(edges)):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt < MIN_BIN_SAMPLES:
            continue
        centers.append((edges[b - 1] + edges[b]) / 2)
        means.append(float(g[mask].mean()))
        counts.append(cnt)
        members.append(mask)
    if len(centers) < 4:
        return None
    centers, means, counts = np.array(centers), np.array(means), np.array(counts)

    padded = np.concatenate([[means[0]], means, [means[-1]]])
    smoothed = np.convolve(padded, [0.25, 0.5, 0.25], mode="valid")

    eligible = counts >= MIN_BIN_COUNT
    if not eligible.any():
        return None
    peak_idx = int(np.argmax(np.where(eligible, smoothed, -np.inf)))
    peak_value = float(smoothed[peak_idx])
    if peak_value <= 0:
        return None

    after = np.arange(len(smoothed)) > peak_idx
    peak_observed = bool(np.any(after & eligible & (smoothed < peak_value * PEAK_FALLOFF_FRACTION)))

    threshold = peak_value * 0.95
    lo = peak_idx
    while lo > 0 and smoothed[lo - 1] >= threshold:
        lo -= 1
    hi = peak_idx
    while hi < len(smoothed) - 1 and smoothed[hi + 1] >= threshold:
        hi += 1
    eff_lo, eff_hi = float(centers[lo]), float(centers[hi])
    if eff_hi <= eff_lo:
        eff_hi = eff_lo + SLIP_ANGLE_BIN_DEG

    in_range = np.zeros(len(sa), dtype=bool)
    for m in members[lo:hi + 1]:
        in_range |= m
    available = float(np.percentile(g[in_range], ENVELOPE_PERCENTILE))

    return {
        "peak": round(float(centers[peak_idx]), 2),
        "efficient_range": [round(eff_lo, 2), round(eff_hi, 2)],
        "peak_grip": round(peak_value, 3),
        "available_grip": round(available, 3),
        "peak_observed": peak_observed,
        "max_observed_slip_deg": round(float(centers[eligible][-1]), 2),
    }


def _temp_selection(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Mask of samples to fit on, plus a summary of tyre temp vs optimal."""
    td = arr["temp_delta_c"]
    n = len(td)
    known = np.isfinite(td)
    if not known.any():
        return np.ones(n, dtype=bool), {"known": False}
    in_win = known & (np.abs(td) <= TEMP_WINDOW_C)
    summary = {
        "known": True,
        "median_delta_c": round(float(np.median(td[known])), 1),
        "fraction_in_window": round(float(in_win.sum() / known.sum()), 3),
        "window_c": TEMP_WINDOW_C,
    }
    if in_win.sum() >= MIN_SAMPLES_FOR_MODEL:
        summary["fit_on"] = "in-window samples only"
        return in_win, summary
    summary["fit_on"] = "all samples (too few in the temperature window)"
    if summary["median_delta_c"] < -TEMP_WINDOW_C:
        summary["warning"] = "tyres mostly below optimal temperature: this is a cold-tyre curve"
    elif summary["median_delta_c"] > TEMP_WINDOW_C:
        summary["warning"] = "tyres mostly above optimal temperature: this is an overheated-tyre curve"
    return np.ones(n, dtype=bool), summary


def build_v2_axle_model(obs: AxleSlipObservations) -> dict:
    """
    The v2 binned model of one axle from RAW samples, e.g.
        {"peak": 3.2, "efficient_range": [2.8, 4.0], "peak_observed": true,
         "confidence": 0.71, "sample_count": 4888,
         "grip_measure": "mu_load_normalized", "peak_grip": 1.52,
         "available_grip": 1.78, "tyre_temperature": {...}}
    peak/efficient_range/peak_grip/available_grip are null when there isn't
    enough data. peak_observed False means the curve was still rising at
    the largest well-sampled slip angle: "peak" is then a LOWER BOUND.

    PREVIOUSLY this was build_axle_tire_model and the profile fed it. The profile now
    stores per-lap bins and the Magic Formula fit (build_axle_tire_model below); this
    stays for validate_tire_model.py, which re-extracts raw samples to hold laps out.
    """
    arr = obs.arrays()
    grip, measure, has_grip = grip_values(arr)
    arr = {k: v[has_grip] for k, v in arr.items()}
    grip = grip[has_grip]
    sel, temp_summary = _temp_selection(arr)
    fit = _fit_curve(arr["slip_angle_deg"][sel], grip[sel])
    out = {
        "peak": None, "efficient_range": None, "peak_observed": False,
        "peak_grip": None, "available_grip": None, "max_observed_slip_deg": None,
    }
    if fit:
        out.update(fit)
    out.update({
        "confidence": _confidence_from_sample_count(len(obs)),
        "sample_count": len(obs),
        "grip_measure": measure,
        "tyre_temperature": temp_summary,
    })
    return out


# ── Model v3 outputs: the cached Magic Formula fit -> the dict the coach reads ──────

MODEL_MF = "magic_formula_v3"
MODEL_FALLBACK = "binned_v2_fallback"
# v3's confidence is a bootstrap-over-laps number. The fallback is a plain slip histogram
# with no interval at all, so its (sample-count based) confidence is capped below the
# coach's "tentative" line of 0.5 rather than claiming more than it knows.
FALLBACK_MAX_CONFIDENCE = 0.3
# A (load, temperature) region with fewer cornering samples than this does not pin the
# curve there.
UNDER_COVERED_N = 50
COVERAGE_LISTED = 4
CORNERING_SLIP_DEG = 1.5   # same line coverage() and the MF reference state use


def _wq(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, v))


def _merge_usable(lap_bins: list, measure: str | None) -> LapBins | None:
    """One LapBins of the laps that share a measure (the cached one, else the first's)."""
    usable = [b for b in lap_bins if b.cells]
    if not usable:
        return None
    measure = measure or usable[0].measure
    usable = [b for b in usable if b.measure == measure]
    return LapBins.merge(usable) if usable else None


def _tyre_temperature(bins: LapBins | None) -> dict:
    """Tyre temperature vs the car's optimum over the stored cells.

    Kept from v2 because the coach still names a cold-tyre curve as a root cause. v3 fits
    all temperatures (temperature is not a fitted factor), so there is no in-window
    selection any more, only the description."""
    if bins is None:
        return {"known": False}
    a = cell_arrays(bins)
    known = np.isfinite(a["temp_delta_c"]) & (a["n"] > 0)
    if not known.any():
        return {"known": False}
    td, w = a["temp_delta_c"][known], a["n"][known]
    median = _wq(td, w, 0.5)
    out = {
        "known": True,
        "median_delta_c": round(median, 1),
        "fraction_in_window": round(float(w[np.abs(td) <= TEMP_WINDOW_C].sum() / w.sum()), 3),
        "window_c": TEMP_WINDOW_C,
        "fit_on": "all samples",
    }
    if median < -TEMP_WINDOW_C:
        out["warning"] = "tyres mostly below optimal temperature: this is a cold-tyre curve"
    elif median > TEMP_WINDOW_C:
        out["warning"] = "tyres mostly above optimal temperature: this is an overheated-tyre curve"
    return out


def coverage_summary(bins: LapBins) -> dict:
    """Which (load, temperature) regions of the driver's own data are thinly sampled.

    Built on tire_bins.coverage: cornering samples (slip >= 1.5 deg) per (load_ratio bin,
    temp_delta bin). The region that counts is the box between the data's weighted p05
    and p95 on each known dimension; inside it a cell with fewer than UNDER_COVERED_N
    samples (including empty ones) is under-covered. A dimension with no known values at
    all (no load source / no optimal temperature) is left out instead of reported empty."""
    raw = tb_coverage(bins)["counts"]
    use_load = any(k[0] != UNKNOWN_BIN for k in raw)
    use_temp = any(k[1] != UNKNOWN_BIN for k in raw)
    if not raw or not (use_load or use_temp):
        return {"available": False, "reason": "no cornering samples with a known load or temperature"}
    counts: dict = {}
    for (lb, tb_), n in raw.items():
        if (use_load and lb == UNKNOWN_BIN) or (use_temp and tb_ == UNKNOWN_BIN):
            continue
        key = (lb if use_load else None, tb_ if use_temp else None)
        counts[key] = counts.get(key, 0) + n
    if not counts:
        return {"available": False, "reason": "no cornering samples with a known load or temperature"}
    keys = list(counts)
    w = np.array([counts[k] for k in keys], dtype=float)

    def span(i):
        # discrete weighted quantile (smallest bin reaching the cumulative share): an
        # interpolated one lands between two bins and can cut a thin outer bin out of the box
        v = np.array([k[i] for k in keys])
        o = np.argsort(v)
        cum = np.cumsum(w[o])

        def pick(q):
            return int(v[o][min(int(np.searchsorted(cum, q * cum[-1])), len(v) - 1)])
        return pick(0.05), pick(0.95)

    l_lo, l_hi = span(0) if use_load else (None, None)
    t_lo, t_hi = span(1) if use_temp else (None, None)
    loads = range(l_lo, l_hi + 1) if use_load else [None]
    temps = range(t_lo, t_hi + 1) if use_temp else [None]
    under = []
    for lb in loads:
        for tb_ in temps:
            n = counts.get((lb, tb_), 0)
            if n < UNDER_COVERED_N:
                cell = {"samples": int(n)}
                if use_load:
                    cell["load_ratio"] = [round(lb * BIN["load_ratio"], 2),
                                          round((lb + 1) * BIN["load_ratio"], 2)]
                if use_temp:
                    cell["temp_delta_c"] = [round(tb_ * BIN["temp_delta_c"], 1),
                                            round((tb_ + 1) * BIN["temp_delta_c"], 1)]
                under.append(cell)
    under.sort(key=lambda c: c["samples"])
    out = {
        "available": True,
        "box": {**({"load_ratio": [round(l_lo * BIN["load_ratio"], 2),
                                   round((l_hi + 1) * BIN["load_ratio"], 2)]} if use_load else {}),
                **({"temp_delta_c": [round(t_lo * BIN["temp_delta_c"], 1),
                                     round((t_hi + 1) * BIN["temp_delta_c"], 1)]} if use_temp else {})},
        "min_samples": UNDER_COVERED_N,
        "n_cells": len(loads) * len(temps),
        "n_under_covered": len(under),
        "under_covered": under[:COVERAGE_LISTED],
    }
    if under:
        parts = []
        for c in under[:2]:
            where = " / ".join(
                f"{k} {c[k][0]}..{c[k][1]}" for k in ("load_ratio", "temp_delta_c") if k in c)
            parts.append(f"{where} ({c['samples']} samples)")
        out["summary"] = (f"{len(under)} of {out['n_cells']} regions inside the data's range have "
                          f"under {UNDER_COVERED_N} cornering samples, e.g. " + "; ".join(parts))
    else:
        out["summary"] = "every region inside the data's range has enough cornering samples"
    return out


def _state_p(a: dict, var: str, q: float) -> float | None:
    """Count-weighted quantile of one state variable over the cornering cells."""
    ok = np.isfinite(a[var]) & (a["n"] > 0) & (a["slip"] >= CORNERING_SLIP_DEG)
    if not ok.any():
        return None
    return _wq(a[var][ok], a["n"][ok], q)


def _factor_effects(fit: MFFit, bins: LapBins) -> dict:
    """Peak mu at three representative values of every FITTED factor.

    The values are the p10 / median / p90 of that state variable in the driver's own
    cornering data, so the effect is only ever stated inside the observed range, never
    extrapolated. Factors that are disabled or not identifiable are not listed (their
    status is in `factors`). Everything else is held at the fit's reference state."""
    a = cell_arrays(bins)
    out = {}
    for factor, status in fit.factors_status.items():
        if status not in ("fitted", "at_bound"):
            continue
        var = STATE_KEY[factor]
        qs = [_state_p(a, var, q) for q in (0.1, 0.5, 0.9)]
        if any(v is None for v in qs) or qs[0] == qs[2]:
            continue
        peaks = [peak_at(fit, {var: v}) for v in qs]
        out[factor] = {
            "variable": var, "status": status,
            "values": [round(v, 3) for v in qs],
            "peak_mu": [round(p[1], 3) for p in peaks],
            "peak_slip_deg": [round(p[0], 2) for p in peaks],
            "range": "p10 / median / p90 of this driver's cornering data",
        }
    return out


def _available_state(fit: MFFit, a: dict) -> dict:
    """The p90 load / temperature state of the data (fitted factors only)."""
    state = {}
    for factor in ("load", "temp"):
        if fit.factors_status.get(factor) in ("fitted", "at_bound"):
            v = _state_p(a, STATE_KEY[factor], 0.9)
            if v is not None:
                state[STATE_KEY[factor]] = v
    return state


def _binned_v2_fallback(bins: LapBins) -> dict | None:
    """v2's slip histogram curve from the stored bins, for when no MF could be fitted.

    Each cell contributes its mean mu `n` times (the bins keep sums, not samples), so the
    fit sees the same counts per slip bin as the raw data would, at cell-mean resolution."""
    a = cell_arrays(bins)
    ok = (a["n"] > 0) & np.isfinite(a["mu_mean"]) & np.isfinite(a["slip"])
    if not ok.any():
        return None
    reps = a["n"][ok].astype(int)
    return _fit_curve(np.repeat(a["slip"][ok], reps), np.repeat(a["mu_mean"][ok], reps))


def build_axle_tire_model(cached: dict | None, lap_bins: list) -> dict:
    """
    One axle's model for the coach, from the profile's cached fit entry and the stored
    LapBins of the laps that fit used.

    `cached` is one axle of GripProfile.tire_fit_cache (fit dict + bootstrap CIs +
    confidence + prior_used + version_note + fit_game_version), or None. Every v2 key keeps
    its name and meaning (peak, efficient_range, peak_observed, peak_grip, available_grip,
    max_observed_slip_deg, confidence, sample_count, grip_measure, tyre_temperature); v3
    adds model, peak_slip_ci90, peak_grip_ci90, factors, factor_effects, prior_used,
    version_note, game_version, coverage.

    model == "magic_formula_v3" when the fit exists. Otherwise "binned_v2_fallback": the v2
    histogram curve from the bins (or all-null when there are no bins), with no interval.
    """
    cached = cached or {}
    fit = MFFit.from_dict(cached["fit"]) if cached.get("fit") else None
    bins = _merge_usable(lap_bins, fit.measure if fit else cached.get("measure"))
    n_samples = bins.n_samples() if bins is not None else 0
    out = {
        "model": MODEL_FALLBACK,
        "peak": None, "efficient_range": None, "efficient_range_note": None,
        "peak_slip_extrapolated": None, "peak_grip_extrapolated": None,
        "peak_at_grid_bound": False,
        "peak_observed": False, "peak_status": None,
        "peak_grip": None, "available_grip": None, "max_observed_slip_deg": None,
        "confidence": 0.0, "sample_count": n_samples,
        "grip_measure": bins.measure if bins is not None else cached.get("measure"),
        "tyre_temperature": _tyre_temperature(bins),
        "peak_slip_ci90": None, "peak_grip_ci90": None,
        "factors": {}, "factor_effects": {},
        "prior_used": cached.get("prior_used", "none"),
        "version_note": cached.get("version_note"),
        "game_version": cached.get("fit_game_version"),
        "n_fit_laps": cached.get("n_fit_laps", sum(1 for b in lap_bins if b.cells)),
        "coverage": coverage_summary(bins) if bins is not None else {
            "available": False, "reason": "no stored laps"},
    }
    if fit is None:
        curve = _binned_v2_fallback(bins) if bins is not None else None
        if curve:
            out.update(curve)
            # v2 histogram: observed, or a lower bound at the edge of the data
            if curve.get("peak") is not None:
                out["peak_status"] = "observed" if curve.get("peak_observed") else "rising_at_edge"
            out["confidence"] = min(_confidence_from_sample_count(n_samples),
                                    FALLBACK_MAX_CONFIDENCE)
        return out

    # Efficient range = the slip span around the peak within 5 % of the peak mu, at the
    # fit's reference state.
    mu = predict(fit, PEAK_GRID)
    ip = int(np.argmax(mu))
    lo = hi = ip
    while lo > 0 and mu[lo - 1] >= 0.95 * mu[ip]:
        lo -= 1
    while hi < len(mu) - 1 and mu[hi + 1] >= 0.95 * mu[ip]:
        hi += 1
    a = cell_arrays(bins)
    # PREVIOUSLY the range and available_grip came straight off the fitted curve, so a
    # curve still rising at the data's edge gave an efficient_range out to 15 deg and an
    # available_grip at a slip the driver never reached (Fuji front: [2.79, 15.0]). Fix:
    # like v2, keep both inside the observed span [lowest well-sampled slip, max observed].
    max_obs = float(fit.max_observed_slip_deg)
    per_slip: dict = {}
    for s_, n_ in zip(a["slip"], a["n"]):
        if np.isfinite(s_):
            per_slip[round(float(s_), 3)] = per_slip.get(round(float(s_), 3), 0) + int(n_)
    sampled = [s_ for s_, n_ in sorted(per_slip.items()) if n_ >= MIN_BIN_SAMPLES]
    lo_deg = max(float(PEAK_GRID[lo]), sampled[0] if sampled else 0.0)
    hi_deg = min(float(PEAK_GRID[hi]), max_obs)
    lo_deg = min(lo_deg, hi_deg)
    # peak_status: "observed" (curve fell off past the peak), "plateau_unconfirmed" (fitted
    # peak is INSIDE the data but no fall-off was seen: BMW M4 front fitted 5.34 deg, driven
    # to 7.4), "rising_at_edge" (fitted peak at/after the largest slip driven).
    if fit.peak_observed:
        peak_status = "observed"
    elif fit.peak_slip_deg < max_obs:
        peak_status = "plateau_unconfirmed"
    else:
        peak_status = "rising_at_edge"
    range_note = None
    if float(PEAK_GRID[hi]) > max_obs:
        # PREVIOUSLY the note always said "still rising", also when the peak WAS observed and
        # the curve merely plateaus past the data. Fix: only rising_at_edge says "rising".
        if peak_status == "rising_at_edge":
            range_note = (f"upper end is the largest slip angle driven ({max_obs:.2f} deg), not a "
                          "measured edge: the curve is still rising there")
        else:
            range_note = (f"upper end is the largest slip angle driven ({max_obs:.2f} deg), not a "
                          "measured edge: the curve is flat (within 5 % of the peak) there and "
                          "may stay flat beyond it")
    state = _available_state(fit, a)
    avail_slip = min(fit.peak_slip_deg, max_obs)
    # PREVIOUSLY `peak` / `peak_grip` were the raw fit even when the peak was never observed
    # (Cadillac rear: 15.0 deg = the PEAK_GRID edge, data only to 6.2), so GU = p95 / peak
    # was meaningless. Fix (v2 meaning): an unobserved peak is reported as the largest slip
    # angle driven, a LOWER bound inside the data, with the mu the curve predicts there; the
    # raw extrapolation moves to *_extrapolated. GU = p95 / peak is then an upper bound on
    # the true utilisation, which is what *_utilization_is_upper_bound says.
    # PREVIOUSLY this also applied to a peak fitted INSIDE the data without a fall-off
    # (plateau_unconfirmed), replacing a sensible fit with max_observed. Fix: only
    # rising_at_edge is a lower bound; a plateau keeps the fitted peak.
    at_bound = bool(fit.peak_slip_deg >= float(PEAK_GRID[-1]) - 1e-6)
    if peak_status != "rising_at_edge":
        peak_deg, peak_mu = fit.peak_slip_deg, fit.peak_mu
    else:
        peak_deg = min(fit.peak_slip_deg, max_obs)
        peak_mu = float(predict(fit, np.array([peak_deg]))[0])
    out.update({
        "model": MODEL_MF,
        "peak": round(peak_deg, 2),
        "peak_slip_extrapolated": round(fit.peak_slip_deg, 2),
        "peak_grip_extrapolated": round(fit.peak_mu, 3),
        "peak_at_grid_bound": at_bound,
        "efficient_range": [round(lo_deg, 2), round(hi_deg, 2)],
        "efficient_range_note": range_note,
        "peak_observed": fit.peak_observed,
        "peak_status": peak_status,
        "peak_grip": round(peak_mu, 3),
        # mu at the p90 load / temperature state of the data, at a slip the driver reached
        "available_grip": round(float(predict(fit, np.array([avail_slip]), state)[0]), 3),
        "max_observed_slip_deg": round(fit.max_observed_slip_deg, 2),
        "grip_measure": fit.measure,
        "confidence": cached.get("confidence", 0.0),
        "peak_slip_ci90": cached.get("peak_slip_ci90"),
        "peak_grip_ci90": cached.get("peak_mu_ci90"),
        "factors": dict(fit.factors_status),
        "factor_effects": _factor_effects(fit, bins),
    })
    return out


# ── Per-lap sample extraction ─────────────────────────────────────────────────

def _pair_mean(lap: LapTelemetry, ch: str, wheels: tuple[str, str]) -> np.ndarray | None:
    a, b = lap.channels.get(f"{ch} {wheels[0]}"), lap.channels.get(f"{ch} {wheels[1]}")
    if a is None or b is None:
        return None
    return (a + b) / 2


def _axle_load(
    lap: LapTelemetry, wheels: tuple[str, str], calibration: dict | None,
) -> tuple[np.ndarray | None, str | None]:
    """
    Axle vertical load (mean of the two wheels, N or calibrated N), the
    numerator of load_ratio = load ÷ the FILE's low-speed straight-line
    load (file_load_reference).

    Why a ratio: lateral G is (axle lateral force) ÷ (that axle's static
    share of car weight) in steady state — Fy_f = m·ay·b/L and the front's
    static load is m·g·b/L — so |lat G| ÷ (dynamic axle load ÷ static axle
    load) is the axle's force per unit of actual vertical load. The
    reference window has little downforce, so the ratio carries downforce
    and longitudinal transfer. Units, motion ratio and mass cancel.

    calibration: {"slope", "intercept"} mapping summed axle Susp Pos (m) to
    summed Susp Force (N), learned from files that had both (see
    fit_deflection_calibration). Without it, deflection is NOT used for
    load: its zero-load offset is unknown, so deflection ratios understate
    load changes.
    """
    for ch in ("Tire Load", "Susp Force"):
        v = _pair_mean(lap, ch, wheels)
        if v is not None and np.isfinite(v).all() and np.nanmax(np.abs(v)) > 1.0:
            source = ch
            break
    else:
        v = _pair_mean(lap, "Susp Pos", wheels)
        if v is None or not calibration:
            return None, None
        v = calibration["slope"] * v + calibration["intercept"]
        source = "Susp Pos (calibrated to Susp Force)"
    return v, source


def _ref_mask(lap: LapTelemetry) -> np.ndarray | None:
    speed = lap.channels.get("Ground Speed")
    lat_g = lap.channels.get("G Force Lat")
    if speed is None or lat_g is None:
        return None
    m = (speed >= REF_LOAD_SPEED_KMH[0]) & (speed <= REF_LOAD_SPEED_KMH[1]) & (np.abs(lat_g) < REF_LOAD_MAX_G)
    long_g = lap.channels.get("G Force Long")
    if long_g is not None:
        m &= np.abs(long_g) < REF_LOAD_MAX_G
    return m


def file_load_reference(
    laps: list[LapTelemetry], calibration: dict[str, dict] | None = None,
) -> dict[str, float]:
    """
    Per-axle reference load (load_ratio = 1.0): median axle load over every
    low-speed, straight-line, no-G sample in the whole file.

    Whole file, not per lap: on the 10/14 Fuji file the flying laps have
    zero such samples — every straight is flat out at 240+ km/h, where
    downforce already adds ~40% load. Only the out lap / pit lane has them
    (69 samples). Missing axle = no usable reference (no load source or
    too few low-speed samples): those samples get no load_ratio.
    """
    out = {}
    for axle, wheels in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        vals = []
        for lap in laps:
            m = _ref_mask(lap)
            v, _ = _axle_load(lap, wheels, (calibration or {}).get(axle))
            if m is not None and v is not None:
                vals.append(v[m])
        vals = np.concatenate(vals) if vals else np.array([])
        if len(vals) >= REF_LOAD_MIN_SAMPLES and np.median(vals) > 0:
            out[axle] = float(np.median(vals))
    return out


def fit_deflection_calibration(lap: LapTelemetry) -> dict[str, dict]:
    """Per-axle linear fit (axle mean Susp Pos -> axle mean Susp Force) from
    a lap that has both, or {} when it lacks them. Stored in the car profile
    so laps without Susp Force (older recordings, native files with Susp
    Pos) can still be load-normalized."""
    out = {}
    speed = lap.channels.get("Ground Speed")
    if speed is None:
        return out
    for axle, wheels in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        f = _pair_mean(lap, "Susp Force", wheels)
        d = _pair_mean(lap, "Susp Pos", wheels)
        if f is None or d is None:
            continue
        m = (speed > MIN_SPEED_KMH_FOR_MODEL) & np.isfinite(f) & np.isfinite(d)
        if m.sum() < 200 or np.std(d[m]) < 1e-4:
            continue
        slope, intercept = np.polyfit(d[m], f[m], 1)
        r = float(np.corrcoef(d[m], f[m])[0, 1])
        if slope > 0 and r > 0.8:
            out[axle] = {"slope": float(slope), "intercept": float(intercept),
                         "r": round(r, 3), "n": int(m.sum())}
    return out


def _lap_axle_samples(
    lap: LapTelemetry, calibration: dict[str, dict] | None = None,
    load_ref: dict[str, float] | None = None,
) -> dict[str, dict] | None:
    """
    Every needed per-sample quantity for both axles of one lap, unfiltered,
    plus the steady-state cornering mask. None when the lap lacks the
    recorder-only Slip Angle channels (native LMU files): we refuse rather
    than guess, since biased absolute values would poison every future lap.
    """
    if not all(f"Slip Angle {w}" in lap.channels for w in WHEEL_LABELS):
        return None
    speed = lap.channels.get("Ground Speed")
    lat_g = lap.channels.get("G Force Lat")
    if speed is None or lat_g is None:
        return None
    long_g = lap.channels.get("G Force Long")
    yaw_acc = lap.channels.get("Yaw Accel")
    lat_abs = np.abs(lat_g)

    steady = speed > MIN_SPEED_KMH_FOR_MODEL
    if long_g is not None:
        steady &= np.abs(long_g) < STEADY_MAX_LONG_G
    if yaw_acc is not None:
        steady &= np.abs(yaw_acc) < STEADY_MAX_YAW_ACCEL

    if load_ref is None:
        load_ref = file_load_reference([lap], calibration)

    try:
        optimal = [float(x) for x in lap.meta.get("TyreOptimalTempC", "").split(",")]
    except ValueError:
        optimal = []
    optimal_by_wheel = dict(zip(WHEEL_LABELS, optimal)) if len(optimal) == 4 else {}

    n = len(speed)
    # A fresh array per use: a shared one returned under several keys/axles
    # would let an in-place edit of one field corrupt the others.
    def nan_arr():
        return np.full(n, np.nan)

    long_g_arr = np.array(long_g, dtype=float) if long_g is not None else nan_arr()

    def _clean_pair_mean(ch, wheels, bad, absolute=False):
        """Axle mean after NaN-ing each wheel's invalid readings (bad(v) -> bool
        array). Mean of a NaN wheel is NaN: one bad wheel makes the axle unknown
        rather than silently halving the value. None when a channel is absent."""
        a, b = lap.channels.get(f"{ch} {wheels[0]}"), lap.channels.get(f"{ch} {wheels[1]}")
        if a is None or b is None:
            return None
        a, b = np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()
        a[bad(a)] = np.nan
        b[bad(b)] = np.nan
        if absolute:
            # Per-wheel |v| BEFORE averaging: the mean of signed values cancels
            # toward 0 when the wheels have opposite signs (e.g. static camber).
            a, b = np.abs(a), np.abs(b)
        return (a + b) / 2

    out = {}
    for axle, wheels in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        slip_deg = np.degrees(_pair_mean(lap, "Slip Angle", wheels))
        temp = _pair_mean(lap, "TyresRubberTemp", wheels)
        load, load_source = _axle_load(lap, wheels, (calibration or {}).get(axle))
        load_ratio = load / load_ref[axle] if load is not None and axle in load_ref else None
        temp_delta = None
        if temp is not None and optimal_by_wheel:
            temp_delta = temp - (optimal_by_wheel[wheels[0]] + optimal_by_wheel[wheels[1]]) / 2
        # PREVIOUSLY the mask ignored the surface. Kerb/grass/gravel samples have
        # tyre forces unrelated to the road-tyre curve and poisoned the peak.
        # Fix: when Surface Type exists, both wheels must be dry (0) or wet (1).
        valid = steady & np.isfinite(slip_deg) & np.isfinite(lat_abs)
        for w in wheels:
            surf = lap.channels.get(f"Surface Type {w}")
            if surf is not None:
                valid &= np.isin(surf, (0.0, 1.0))

        # Camber is radians (typical -0.035..+0.011); a few samples reach ±0.59
        # (physics glitches), so |camber| > 0.1 rad is dropped per wheel.
        camber = _clean_pair_mean("Camber", wheels, lambda v: np.abs(v) > 0.1, absolute=True)
        # Carcass temp carries a -273.15 C sentinel (0 K = no reading).
        carcass = _clean_pair_mean("TyresCarcassTemp", wheels, lambda v: v <= -100.0)
        carcass_delta = None
        if carcass is not None and optimal_by_wheel:
            carcass_delta = carcass - (optimal_by_wheel[wheels[0]] + optimal_by_wheel[wheels[1]]) / 2
        # TyresWear is the REMAINING tread fraction (1.0 = new); store the WORN
        # fraction so larger = more worn.
        remaining = _pair_mean(lap, "TyresWear", wheels)

        # True mu = sum|Fy| / sum Fz over the axle's wheels. Needs both channels;
        # NaN where load < 30 % of the lap's median axle load (unloaded wheel,
        # ratio blows up). GT3 cars record zeros -> median 0 -> all NaN.
        true_mu = nan_arr()
        fy = [lap.channels.get(f"Tire Lateral Force {w}") for w in wheels]
        fz = [lap.channels.get(f"Tire Load {w}") for w in wheels]
        if all(c is not None for c in fy + fz):
            fz_sum = np.asarray(fz[0], dtype=float) + np.asarray(fz[1], dtype=float)
            fy_sum = np.abs(np.asarray(fy[0], dtype=float)) + np.abs(np.asarray(fy[1], dtype=float))
            med = np.nanmedian(fz_sum)
            if np.isfinite(med) and med > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    true_mu = np.where(fz_sum >= 0.3 * med, fy_sum / fz_sum, np.nan)

        out[axle] = {
            "carcass_delta_c": carcass_delta if carcass_delta is not None else nan_arr(),
            "wear": 1.0 - remaining if remaining is not None else nan_arr(),
            "camber_deg": np.degrees(camber) if camber is not None else nan_arr(),
            "long_g": long_g_arr.copy(),
            "speed_kmh": np.array(speed, dtype=float),
            "true_mu": true_mu,
            "slip_angle_deg": slip_deg,
            "lateral_g": lat_abs,
            "tire_temp_c": temp,
            "tire_pressure_kpa": _pair_mean(lap, "TyresPressure", wheels),
            "susp_deflection_m": _pair_mean(lap, "Susp Pos", wheels),
            "load_ratio": load_ratio,
            "temp_delta_c": temp_delta,
            "load_source": load_source,
            "mask": valid,
        }
    return out


def axle_load_source(s: dict) -> str:
    """Where one axle's load ratio came from (a `_lap_axle_samples` axle dict). Shared by
    the v2 observation extraction and the v3 profile ingest, which labels each stored lap."""
    if s["load_ratio"] is not None:
        return s["load_source"]
    if s["load_source"] is not None:
        return "none (no low-speed straight reference in this file)"
    return "none (no load channel)"


def extract_axle_observations_from_lap(
    lap: LapTelemetry, calibration: dict[str, dict] | None = None,
    load_ref: dict[str, float] | None = None,
) -> dict[str, AxleSlipObservations]:
    """Steady-state cornering observations for front and rear axles of one
    lap (empty for laps without the recorder's Slip Angle channel).
    load_ref: file_load_reference over the whole file; defaults to this lap
    alone, which usually has no low-speed straight samples."""
    empty = {"front": AxleSlipObservations(), "rear": AxleSlipObservations()}
    samples = _lap_axle_samples(lap, calibration, load_ref)
    if samples is None:
        return empty
    result = {}
    for axle, s in samples.items():
        obs = AxleSlipObservations()
        obs.load_source = axle_load_source(s)
        m = s["mask"]
        if m.any():
            obs.add_samples(**{
                name: (s[name][m] if s[name] is not None else None) for name in _OBS_FIELDS
            })
        result[axle] = obs
    return result


# Sample-dict key for each MF state variable (STATE_KEY names the fit's variable).
_SAMPLE_KEY = {"load_ratio": "load_ratio", "temp_delta_c": "temp_delta_c",
               "pressure_kpa": "tire_pressure_kpa"}
_STATE_ROUND = {"load_ratio": 0.02, "temp_delta_c": 1.0, "pressure_kpa": 2.0}


def _state_aware_ceiling(fit: MFFit, s: dict, mask: np.ndarray) -> float | None:
    """Mean over the masked samples of the fit's peak mu AT EACH SAMPLE'S OWN state.

    PREVIOUSLY the ceiling was one number (the p90 envelope of the model), so a lap
    through fast corners (high load, higher mu) and one through slow corners were
    divided by the same value. Fix: the denominator follows the state the tyre was in.
    States are rounded and de-duplicated first, since peak_at scans a slip grid."""
    keys = [k for f, k in STATE_KEY.items()
            if fit.factors_status.get(f) in ("fitted", "at_bound") and s.get(_SAMPLE_KEY[k]) is not None]
    if not keys:
        return None
    cols = np.column_stack([np.round(np.asarray(s[_SAMPLE_KEY[k]], dtype=float)[mask] / _STATE_ROUND[k])
                            * _STATE_ROUND[k] for k in keys])
    cols = cols[np.all(np.isfinite(cols), axis=1)]
    if len(cols) < 20:
        return None
    uniq, counts = np.unique(cols, axis=0, return_counts=True)
    # PREVIOUSLY peak_at scanned the whole PEAK_GRID, so an extrapolated peak beyond the
    # data (rising curve) inflated the divisor and CE read low. Fix: same cap as
    # available_grip, the largest slip angle the driver reached.
    grid = PEAK_GRID[PEAK_GRID <= float(fit.max_observed_slip_deg)]
    if len(grid) == 0:
        return None
    peaks = np.array([float(np.max(predict(fit, grid, dict(zip(keys, row))))) for row in uniq])
    return float(np.average(peaks, weights=counts))


def _is_lower_bound_peak(model: dict) -> bool:
    """True when the model's peak is only a lower bound (curve still rising at the data's
    edge), so GU = p95 / peak is an upper bound. A plateau_unconfirmed peak is a fit inside
    the data and is not flagged. Models without peak_status (v2 dicts) fall back to
    peak_observed."""
    status = model.get("peak_status")
    if status is not None:
        return status == "rising_at_edge"
    return not model.get("peak_observed", False)


def compute_grip_utilization_for_lap(
    lap: LapTelemetry, front_model: dict, rear_model: dict,
    calibration: dict[str, dict] | None = None,
    load_ref: dict[str, float] | None = None,
    front_fit: MFFit | None = None,
) -> dict | None:
    """
    Grip utilization + cornering efficiency for THIS lap, scored against the
    car's learned tire model as it was BEFORE this lap's samples were folded
    in (so a lap is never scored against a model containing itself).

    Returns None if either axle's model has no peak yet or the lap lacks
    the channels — no hard-coded fallback peak.

    GRIP UTILIZATION (per axle): this lap's p95 |slip angle| over the same
    steady-state cornering samples the model is built from, ÷ the learned
    peak slip angle. p95, not max: a single-tick kerb strike used to pin
    every lap near the 1.5 cap (v1: 13 of 1680 samples > 10° on one lap).
    When the model's peak was not observed (still rising), the peak is a
    lower bound, so the utilization is an UPPER bound — flagged.

    CORNERING EFFICIENCY (front): this lap's p90 grip ÷ the model's p90
    envelope (available_grip) in its efficient range, in the model's own
    grip measure (mu or lateral G — the lap is converted the same way).
    """
    if front_model.get("peak") is None or rear_model.get("peak") is None:
        return None
    samples = _lap_axle_samples(lap, calibration, load_ref)
    if samples is None:
        return None

    def _axle(axle: str, model: dict) -> tuple[float | None, float | None]:
        s = samples[axle]
        m = s["mask"]
        if m.sum() < 20:
            return None, None
        p95 = float(np.nanpercentile(np.abs(s["slip_angle_deg"][m]), 95))
        return p95, round(min(p95 / model["peak"], 1.5), 3)

    front_p95, front_gu = _axle("front", front_model)
    rear_p95, rear_gu = _axle("rear", rear_model)

    ce = None
    lap_grip = None
    s = samples["front"]
    m = s["mask"]
    if front_model.get("grip_measure") == "mu_load_normalized":
        lr = s["load_ratio"]
        grip = s["lateral_g"] / np.maximum(lr, 0.2) if lr is not None else None
    else:
        grip = s["lateral_g"]
    available = front_model.get("available_grip")
    ce_method = "v2_envelope"
    if front_fit is not None:
        ceiling = _state_aware_ceiling(front_fit, s, m)
        if ceiling:
            available, ce_method = ceiling, "state_aware_mf"
    if grip is not None and m.sum() >= 20 and available:
        lap_grip = float(np.nanpercentile(grip[m], ENVELOPE_PERCENTILE))
        ce = round(min(lap_grip / available, 1.5), 3)

    return {
        "front_grip_utilization": front_gu,
        "rear_grip_utilization": rear_gu,
        "front_utilization_is_upper_bound": _is_lower_bound_peak(front_model),
        "rear_utilization_is_upper_bound": _is_lower_bound_peak(rear_model),
        "front_peak_slip_angle_deg": round(front_p95, 2) if front_p95 is not None else None,
        "rear_peak_slip_angle_deg": round(rear_p95, 2) if rear_p95 is not None else None,
        "lap_grip_p90": round(lap_grip, 3) if lap_grip is not None else None,
        "grip_measure": front_model.get("grip_measure"),
        "cornering_efficiency": ce,
        "ce_method": ce_method,
        "note": (
            "Steady-state cornering samples only. grip_utilization = this lap's "
            "p95 |slip angle| ÷ the learned peak slip angle for that axle (>1.0 "
            "= pushed past the tire's observed optimum; an upper bound when the "
            "model's peak was not actually observed). cornering_efficiency = "
            "this lap's p90 grip ÷ the model's p90 grip envelope near its peak, "
            "in grip_measure units (mu_load_normalized = lateral G per unit of "
            "actual axle load, removing downforce differences between corners). With "
            "ce_method state_aware_mf the divisor is the fitted peak at each sample's "
            "own load, not one envelope value."
        ),
    }
