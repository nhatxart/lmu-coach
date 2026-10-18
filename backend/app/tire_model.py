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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .telemetry import LapTelemetry, WHEEL_LABELS

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


def build_axle_tire_model(obs: AxleSlipObservations) -> dict:
    """
    One axle's model, e.g.
        {"peak": 3.2, "efficient_range": [2.8, 4.0], "peak_observed": true,
         "confidence": 0.71, "sample_count": 4888,
         "grip_measure": "mu_load_normalized", "peak_grip": 1.52,
         "available_grip": 1.78, "tyre_temperature": {...}}
    peak/efficient_range/peak_grip/available_grip are null when there isn't
    enough data. peak_observed False means the curve was still rising at
    the largest well-sampled slip angle: "peak" is then a LOWER BOUND.
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


def build_temp_pressure_load_relationships(obs: AxleSlipObservations) -> dict:
    """
    Observed (not first-principles) relationships between the learned
    optimal slip angle and temperature / pressure / load, by quantile bucket.
    Each sub-relationship checks it has enough data and reports null + a
    reason if not.
    """
    arr = obs.arrays()
    grip, _, has_grip = grip_values(arr)
    arr = {k: v[has_grip] for k, v in arr.items()}
    sa = arr["slip_angle_deg"]
    g = grip[has_grip]
    temp = np.where(np.isfinite(arr["temp_delta_c"]), arr["temp_delta_c"], np.nan)
    temp_label = "temp_delta_c"
    if not np.isfinite(temp).any():
        temp, temp_label = arr["tire_temp_c"], "temp_c"
    load = arr["load_ratio"]
    load_label = "load_ratio"
    if not np.isfinite(load).any():
        load, load_label = arr["susp_deflection_m"], "susp_deflection_m"

    def _bucketed_optimal_slip(bucket_values: np.ndarray, n_buckets: int, label: str) -> dict:
        valid = np.isfinite(bucket_values) & np.isfinite(sa) & np.isfinite(g)
        if valid.sum() < MIN_SAMPLES_FOR_MODEL:
            return {"available": False, "reason": f"insufficient {label} samples ({int(valid.sum())})"}
        bv, sa_v, g_v = bucket_values[valid], sa[valid], g[valid]
        edges = np.unique(np.quantile(bv, np.linspace(0, 1, n_buckets + 1)))
        if len(edges) < 3:
            return {"available": False, "reason": f"{label} range too narrow to bucket"}
        # decimals scale with the bucketed quantity's span (susp deflection
        # in metres vs temps in tens of degrees)
        span = float(edges[-1] - edges[0])
        decimals = 3 if span == 0 else max(0, 3 - int(np.floor(np.log10(abs(span) + 1e-12))))
        idx = np.digitize(bv, edges[1:-1])
        buckets = []
        for b in range(len(edges) - 1):
            mask = idx == b
            if mask.sum() < 30:
                continue
            fit = _fit_curve(sa_v[mask], g_v[mask])
            buckets.append({
                f"{label}_range": [round(float(edges[b]), decimals), round(float(edges[b + 1]), decimals)],
                "optimal_slip_angle_deg": fit["peak"] if fit else None,
                "peak_observed": fit["peak_observed"] if fit else False,
                "sample_count": int(mask.sum()),
            })
        if not buckets:
            return {"available": False, "reason": f"no {label} bucket had enough samples"}
        return {"available": True, "buckets": buckets}

    return {
        "temperature": _bucketed_optimal_slip(temp, 4, temp_label),
        "pressure": _bucketed_optimal_slip(arr["tire_pressure_kpa"], 4, "pressure_kpa"),
        "load_proxy": _bucketed_optimal_slip(load, 4, load_label),
        "note": (
            "OBSERVED correlations, not a physics model. Temperature and "
            "pressure are CONFOUNDED with cornering itself: hard cornering "
            "heats the tyre and raises its pressure, so hot/high-pressure "
            "buckets are simply the hard-cornering samples. Do not read them "
            "as 'the tyre wants X kPa'. temp_delta_c is tyre temp minus the "
            "car's optimal temperature. load_ratio is axle load ÷ low-speed "
            "straight-line load (>1 = downforce/load transfer)."
        ),
    }


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

    out = {}
    for axle, wheels in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        slip_deg = np.degrees(_pair_mean(lap, "Slip Angle", wheels))
        temp = _pair_mean(lap, "TyresRubberTemp", wheels)
        load, load_source = _axle_load(lap, wheels, (calibration or {}).get(axle))
        load_ratio = load / load_ref[axle] if load is not None and axle in load_ref else None
        temp_delta = None
        if temp is not None and optimal_by_wheel:
            temp_delta = temp - (optimal_by_wheel[wheels[0]] + optimal_by_wheel[wheels[1]]) / 2
        out[axle] = {
            "slip_angle_deg": slip_deg,
            "lateral_g": lat_abs,
            "tire_temp_c": temp,
            "tire_pressure_kpa": _pair_mean(lap, "TyresPressure", wheels),
            "susp_deflection_m": _pair_mean(lap, "Susp Pos", wheels),
            "load_ratio": load_ratio,
            "temp_delta_c": temp_delta,
            "load_source": load_source,
            "mask": steady & np.isfinite(slip_deg) & np.isfinite(lat_abs),
        }
    return out


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
        m = s["mask"]
        if m.any():
            obs.add_samples(**{
                name: (s[name][m] if s[name] is not None else None) for name in _OBS_FIELDS
            })
        result[axle] = obs
    return result


def compute_grip_utilization_for_lap(
    lap: LapTelemetry, front_model: dict, rear_model: dict,
    calibration: dict[str, dict] | None = None,
    load_ref: dict[str, float] | None = None,
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
    if grip is not None and m.sum() >= 20 and available:
        lap_grip = float(np.nanpercentile(grip[m], ENVELOPE_PERCENTILE))
        ce = round(min(lap_grip / available, 1.5), 3)

    return {
        "front_grip_utilization": front_gu,
        "rear_grip_utilization": rear_gu,
        "front_utilization_is_upper_bound": not front_model.get("peak_observed", False),
        "rear_utilization_is_upper_bound": not rear_model.get("peak_observed", False),
        "front_peak_slip_angle_deg": round(front_p95, 2) if front_p95 is not None else None,
        "rear_peak_slip_angle_deg": round(rear_p95, 2) if rear_p95 is not None else None,
        "lap_grip_p90": round(lap_grip, 3) if lap_grip is not None else None,
        "grip_measure": front_model.get("grip_measure"),
        "cornering_efficiency": ce,
        "note": (
            "Steady-state cornering samples only. grip_utilization = this lap's "
            "p95 |slip angle| ÷ the learned peak slip angle for that axle (>1.0 "
            "= pushed past the tire's observed optimum; an upper bound when the "
            "model's peak was not actually observed). cornering_efficiency = "
            "this lap's p90 grip ÷ the model's p90 grip envelope near its peak, "
            "in grip_measure units (mu_load_normalized = lateral G per unit of "
            "actual axle load, removing downforce differences between corners)."
        ),
    }
