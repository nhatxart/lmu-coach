"""Factorized Magic Formula fit for one axle (tyre model v3).

    mu = D * sin(C * atan(B*a - E*(B*a - atan(B*a))))          a = |slip angle| in DEGREES

with D and B scaled by state factors (load ratio, tyre temperature vs optimum,
pressure residual given temperature, wear, camber), C and E shared:

    D = D0 * (1 + kDL*(L-L0)) * T(td) * (1 + kDP*dp) * (1 - kDW*(w-w0)) * (1 + kDC*(c-c0))
    B = B0 * (1 + kBL*(L-L0)) * (1 + kBP*dp)
    T(td) = exp(-((td-T0)/wl)^2) for td < T0, exp(-((td-T0)/wh)^2) otherwise

Conventions (all deliberate):
- Slip is in degrees throughout, so B0 is "per degree".
- D0 is the peak mu AT the temperature optimum T0 (T(T0) = 1) and at the reference
  load/wear/camber (L0, w0, c0 = count-weighted medians of the fit's data) and zero
  pressure residual. The peak at the reference *state* is peak_at(fit), which also
  includes T(td_ref).
- dp is the pressure residual given temperature: pressure minus a count-weighted linear
  fit of pressure on temp_delta. Pressure rises with temperature, so raw pressure would
  be collinear with the temperature factor. The regression coefficients are stored in
  params (pt_slope, pt_intercept) so predict() takes RAW pressure + temp_delta.
- A state value that is NaN for a cell contributes that factor at its neutral value
  (L0 / w0 / c0, T = 1, dp = 0). NaN never reaches the residuals.
- Factors the data cannot identify are fixed neutral and reported, never guessed.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .tire_bins import _CELL_FIELDS, Cell, LapBins, cell_arrays

ALL_FACTORS = ("load", "temp", "pressure", "wear", "camber")
ENABLED_FACTORS: tuple = ALL_FACTORS

BOUNDS = {
    "D0": (0.1, 5.0), "B0": (0.05, 3.0), "C": (1.0, 2.0), "E": (-3.0, 1.0),
    "kDL": (-1.0, 0.5), "kBL": (-1.0, 1.0), "T0": (-30.0, 30.0),
    "wl": (5.0, 80.0), "wh": (5.0, 80.0), "kDP": (-0.05, 0.05), "kBP": (-0.05, 0.05),
    "kDW": (0.0, 2.0), "kDC": (-0.2, 0.2),
}
BASE_PARAMS = ("D0", "B0", "C", "E")
FACTOR_PARAMS = {
    "load": ("kDL", "kBL"), "temp": ("T0", "wl", "wh"), "pressure": ("kDP", "kBP"),
    "wear": ("kDW",), "camber": ("kDC",),
}
NEUTRAL = {"kDL": 0.0, "kBL": 0.0, "T0": 0.0, "wl": 30.0, "wh": 30.0,
           "kDP": 0.0, "kBP": 0.0, "kDW": 0.0, "kDC": 0.0}
# state variable name used by predict()/ref_state for each factor
STATE_KEY = {"load": "load_ratio", "temp": "temp_delta_c", "pressure": "pressure_kpa",
             "wear": "wear", "camber": "camber_deg"}
# Identifiability thresholds: (kind, value). std for load, p95-p05 spread for the rest.
GATE = {"load": ("std", 0.05), "temp": ("spread", 10.0), "pressure": ("spread", 5.0),
        "wear": ("spread", 0.02), "camber": ("spread", 0.3)}
GATE_MIN_SLIP_DEG = 1.5
MIN_TOTAL_SAMPLES = 300   # replaces the old "30 cells with n >= 5" gate (Ruling 8)
MIN_SLIP_BINS = 8         # slip bins whose marginal n >= MIN_CELL_N
MIN_CELL_N = 5            # (kept name) marginal n for a slip bin to count as sampled
SLIP_BIN_DEG = 0.4
MIN_SLIP_SPAN_DEG = 2.5
F_SCALE = 0.1
PEAK_GRID = np.arange(0.0, 15.0 + 1e-9, 0.01)
PEAK_OBS_MIN_N = 20
PEAK_OBS_MARGIN_DEG = 0.4
PEAK_OBS_RATIO = 0.97
FACTOR_FLOOR = 0.05  # a state factor that would go <= 0 is clipped, keeping D, B > 0
START_C = (1.3, 1.5, 1.8)
MAX_NFEV = 300


@dataclass
class MFFit:
    params: dict
    ref_state: dict
    factors_status: dict
    measure: str
    n_cells: int
    n_samples: int
    rmse: float
    peak_slip_deg: float
    peak_mu: float
    peak_observed: bool
    max_observed_slip_deg: float
    # What was fitted: "mu" (the proxy `measure`) or "true" (the game's tyre-force mu).
    target: str = "mu"
    # Fitted params within 1 % of their bound width of a bound: the data did not pin them.
    at_bound: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MFFit":
        d = dict(d)
        d.setdefault("target", "mu")
        d["at_bound"] = list(d.get("at_bound") or [])
        d["notes"] = list(d.get("notes") or [])
        return MFFit(**d)


# ---------------------------------------------------------------- model --------------

def _wquantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, x))


def _dp(params, pressure, td):
    """Pressure residual given temperature; NaN where it cannot be computed."""
    slope, icpt = params["pt_slope"], params["pt_intercept"]
    pred = icpt + (slope * td if slope != 0.0 else 0.0)
    return pressure - pred


def _state_arrays(params, ref, state, shape):
    """Broadcast a state dict to arrays; missing keys -> reference, NaN -> neutral."""
    def get(key):
        v = np.asarray(ref[key] if state is None or key not in state else state[key], float)
        return np.broadcast_to(v, shape).astype(float)

    load, wear, camber = get("load_ratio"), get("wear"), get("camber_deg")
    td, pr = get("temp_delta_c"), get("pressure_kpa")
    with np.errstate(invalid="ignore"):
        dp = _dp(params, pr, td)
    return {
        "L": np.where(np.isfinite(load), load, ref["load_ratio"]),
        "w": np.where(np.isfinite(wear), wear, ref["wear"]),
        "c": np.where(np.isfinite(camber), camber, ref["camber_deg"]),
        "td": td,  # NaN kept: T = 1 there
        "dp": np.where(np.isfinite(dp), dp, 0.0),
    }


def _scales(params, temp_on, ref, st):
    """(D multiplier incl. D0 excluded, B multiplier) for state arrays st."""
    p = params
    dl = np.maximum(1 + p["kDL"] * (st["L"] - ref["load_ratio"]), FACTOR_FLOOR)
    bl = np.maximum(1 + p["kBL"] * (st["L"] - ref["load_ratio"]), FACTOR_FLOOR)
    dpf = np.maximum(1 + p["kDP"] * st["dp"], FACTOR_FLOOR)
    bpf = np.maximum(1 + p["kBP"] * st["dp"], FACTOR_FLOOR)
    dw = np.maximum(1 - p["kDW"] * (st["w"] - ref["wear"]), FACTOR_FLOOR)
    dc = np.maximum(1 + p["kDC"] * (st["c"] - ref["camber_deg"]), FACTOR_FLOOR)
    if temp_on:
        td = st["td"]
        width = np.where(td < p["T0"], p["wl"], p["wh"])
        with np.errstate(invalid="ignore"):
            t = np.where(np.isfinite(td), np.exp(-(((td - p["T0"]) / width) ** 2)), 1.0)
    else:
        t = 1.0
    return dl * t * dpf * dw * dc, bl * bpf


def _mf(slip, D, B, C, E):
    ba = B * slip
    return D * np.sin(C * np.arctan(ba - E * (ba - np.arctan(ba))))


def _predict_raw(params, temp_on, ref, slip, state):
    slip = np.asarray(slip, float)
    st = _state_arrays(params, ref, state, slip.shape)
    dm, bm = _scales(params, temp_on, ref, st)
    return _mf(slip, params["D0"] * dm, params["B0"] * bm, params["C"], params["E"])


def predict(fit: MFFit, slip_deg, state=None) -> np.ndarray:
    # "at_bound" temperature params are still used for prediction.
    return _predict_raw(fit.params, fit.factors_status.get("temp") in ("fitted", "at_bound"),
                        fit.ref_state, slip_deg, state)


def peak_at(fit: MFFit, state=None) -> tuple:
    mu = predict(fit, PEAK_GRID, state)
    i = int(np.argmax(mu))
    return float(PEAK_GRID[i]), float(mu[i])


def _shape_peak(C, E):
    """max over slip of sin(C*atan(u - E*(u - atan u))): D scales it, B only stretches u."""
    u = np.arange(0.0, 200.0, 0.02)
    return float(np.max(np.sin(C * np.arctan(u - E * (u - np.arctan(u))))))


# ---------------------------------------------------------------- fitting ------------

def _initial_guess(slip, y, n):
    """D0 from the max smoothed binned mean; B0 from the slip at that max (E=0 start)."""
    bins = np.floor(slip / 0.4).astype(int)
    uniq = np.unique(bins)
    mean = np.array([np.average(y[bins == b], weights=n[bins == b]) for b in uniq])
    sl = np.array([np.average(slip[bins == b], weights=n[bins == b]) for b in uniq])
    if len(mean) >= 3:
        mean = np.convolve(np.pad(mean, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    i = int(np.argmax(mean))
    return float(mean[i]), max(float(sl[i]), 0.5)


def fit_mf(bins: LapBins, target: str = "mu", prior=None, prior_weight: float = 0.0,
           factors=None, x0=None):
    """x0: optional warm start (param name -> value, e.g. a previous fit's params). When
    given it is the ONLY start (no multi-start over C): the bootstrap refits many
    resamples of data the full fit already explained, so one start from its optimum is
    enough and ~3x cheaper. Names not free in this fit are ignored."""
    factors = ENABLED_FACTORS if factors is None else tuple(factors)
    a = cell_arrays(bins)
    if len(a["n"]) == 0:
        return None
    if target == "true":
        y, n = a["true_mean"], a["true_n"]
    else:
        y, n = a["mu_mean"], a["n"]
    slip = a["slip"]
    ok = np.isfinite(y) & np.isfinite(slip) & (n > 0)
    if ok.sum() == 0:
        return None
    y, n, slip = y[ok], n[ok], slip[ok]
    st_all = {k: a[k][ok] for k in ("load_ratio", "temp_delta_c", "pressure_kpa",
                                     "wear", "camber_deg")}

    # PREVIOUSLY these gates counted individual cells (n >= 5). Cells are keyed by five
    # state dimensions, so a dense lap spreads over many 1-2 sample cells and the gate
    # refused data that is perfectly good along slip (HF Hypercar files, 10/19 run).
    # Fix (Ruling 8): "well sampled" is decided on the SLIP MARGINAL - counts summed
    # over all cells sharing a slip bin.
    slip_bin = np.floor(slip / SLIP_BIN_DEG).astype(int)
    bin_ids, inv = np.unique(slip_bin, return_inverse=True)
    inv = inv.ravel()
    bin_n = np.bincount(inv, weights=n)
    if n.sum() < MIN_TOTAL_SAMPLES or (bin_n >= MIN_CELL_N).sum() < MIN_SLIP_BINS:
        return None
    dense = bin_n >= PEAK_OBS_MIN_N
    if not dense.any():
        return None
    centres = (bin_ids + 0.5) * SLIP_BIN_DEG
    if centres[dense].max() - centres[dense].min() < MIN_SLIP_SPAN_DEG:
        return None
    max_obs = float(centres[dense].max())

    # Reference state: count-weighted medians over cells with real slip (the cells that
    # carry the peak information); fall back to all cells if none reach it.
    hi = slip >= GATE_MIN_SLIP_DEG
    sel = hi if hi.any() else np.ones_like(hi)
    fallback = {"load_ratio": 1.0, "temp_delta_c": 0.0, "pressure_kpa": 0.0,
                "wear": 0.0, "camber_deg": 0.0}
    ref = {}
    for k, v in st_all.items():
        f = sel & np.isfinite(v)
        ref[k] = _wquantile(v[f], n[f], 0.5) if f.any() else fallback[k]

    # Pressure ~ temp regression (count-weighted, all cells with both finite).
    # PREVIOUSLY: always fitted, so with temperature NOT fitted pt_slope still fed dp in
    # predict() and predictions depended on temp_delta_c. Fix: the temperature gate runs
    # first; if temp is not fitted the slope stays 0 (dp = pressure - weighted mean).
    pt = {"pt_slope": 0.0, "pt_intercept": 0.0}
    fp = np.isfinite(st_all["pressure_kpa"])
    ft = fp & np.isfinite(st_all["temp_delta_c"])
    if fp.any():
        pt["pt_intercept"] = float(np.average(st_all["pressure_kpa"][fp], weights=n[fp]))
    temp_ok = False
    if "temp" in factors:
        g = hi & np.isfinite(st_all["temp_delta_c"])
        if g.sum() >= 3:
            td_g = st_all["temp_delta_c"][g]
            temp_ok = (_wquantile(td_g, n[g], 0.95) - _wquantile(td_g, n[g], 0.05)
                       >= GATE["temp"][1])
    if temp_ok and ft.sum() >= 2:
        w, x, z = n[ft], st_all["temp_delta_c"][ft], st_all["pressure_kpa"][ft]
        xm, zm = np.average(x, weights=w), np.average(z, weights=w)
        vx = np.sum(w * (x - xm) ** 2)
        if vx > 1e-9:
            pt["pt_slope"] = float(np.sum(w * (x - xm) * (z - zm)) / vx)
            pt["pt_intercept"] = float(zm - pt["pt_slope"] * xm)
    with np.errstate(invalid="ignore"):
        dp_cells = _dp(pt, st_all["pressure_kpa"], st_all["temp_delta_c"])

    # Identifiability gate.
    gate_var = {"load": st_all["load_ratio"], "temp": st_all["temp_delta_c"],
                "pressure": dp_cells, "wear": st_all["wear"], "camber": st_all["camber_deg"]}
    status = {}
    for f in ALL_FACTORS:
        if f not in factors:
            status[f] = "disabled"
            continue
        v = gate_var[f]
        g = hi & np.isfinite(v)
        if g.sum() < 3:
            status[f] = "not_identifiable"
            continue
        kind, thr = GATE[f]
        if kind == "std":
            m = np.average(v[g], weights=n[g])
            val = float(np.sqrt(np.average((v[g] - m) ** 2, weights=n[g])))
        else:
            val = _wquantile(v[g], n[g], 0.95) - _wquantile(v[g], n[g], 0.05)
        status[f] = "fitted" if val >= thr else "not_identifiable"

    free = list(BASE_PARAMS)
    for f in ALL_FACTORS:
        if status[f] == "fitted":
            free += list(FACTOR_PARAMS[f])
    temp_on = status["temp"] == "fitted"

    fixed = {**NEUTRAL, **pt}
    scale = np.array([(BOUNDS[p][1] - BOUNDS[p][0]) / 10 for p in free])
    lo = np.array([BOUNDS[p][0] for p in free])
    hi_b = np.array([BOUNDS[p][1] for p in free])

    st_cell = _state_arrays({**fixed}, ref, st_all, slip.shape)
    sqrt_n = np.sqrt(n)
    n_total = float(n.sum())

    def full(x):
        p = dict(fixed)
        p.update(zip(free, x))
        return p

    # Prior: pulls only parameters that are free here AND meaningful in the prior.
    prior_terms = []
    notes = []
    if prior is not None and prior_weight > 0:
        # A prior fitted on another measure/target is on a different scale of mu; pulling
        # toward it would be wrong, so ignore it and say so.
        if prior.measure != bins.measure:
            notes.append("prior ignored: measure mismatch")
            prior = None
        elif prior.target != target:
            notes.append("prior ignored: target mismatch")
            prior = None
    if prior is not None and prior_weight > 0:
        pp = dict(prior.params)
        # The prior's D0/B0 refer to ITS reference state; re-express at ours so the pull
        # compares like with like (load, wear, camber shifts; temperature/pressure left).
        dl = 1 + pp["kDL"] * (ref["load_ratio"] - prior.ref_state["load_ratio"])
        dw = 1 - pp["kDW"] * (ref["wear"] - prior.ref_state["wear"])
        dc = 1 + pp["kDC"] * (ref["camber_deg"] - prior.ref_state["camber_deg"])
        bl = 1 + pp["kBL"] * (ref["load_ratio"] - prior.ref_state["load_ratio"])
        pp["D0"] = pp["D0"] * max(dl, FACTOR_FLOOR) * max(dw, FACTOR_FLOOR) * max(dc, FACTOR_FLOOR)
        pp["B0"] = pp["B0"] * max(bl, FACTOR_FLOOR)
        owner = {p: f for f, ps in FACTOR_PARAMS.items() for p in ps}
        coef = np.sqrt(prior_weight * n_total / len(free))
        for i, p in enumerate(free):
            f = owner.get(p)
            if f is None or prior.factors_status.get(f) == "fitted":
                prior_terms.append((i, pp[p], coef / scale[i]))

    def resid(x):
        p = full(x)
        pred = _predict_raw(p, temp_on, ref, slip, st_all)
        r = sqrt_n * (y - pred)
        if prior_terms:
            extra = np.array([c * (x[i] - t) for i, t, c in prior_terms])
            r = np.concatenate([r, extra])
        return r

    d0_guess, slip_peak = _initial_guess(slip, y, n)
    wmed_td = ref["temp_delta_c"]
    best = None
    for c0 in (START_C if x0 is None else (START_C[1],)):
        start = {"D0": d0_guess, "B0": np.tan(np.pi / (2 * c0)) / slip_peak, "C": c0,
                 "E": 0.0, "kDL": 0.0, "kBL": 0.0, "T0": wmed_td, "wl": 30.0, "wh": 30.0,
                 "kDP": 0.0, "kBP": 0.0, "kDW": 0.05, "kDC": 0.0}
        if x0 is not None:
            start.update({k: float(v) for k, v in x0.items() if k in start})
        x = np.array([start[p] for p in free])
        x = np.clip(x, lo + 1e-6 * scale, hi_b - 1e-6 * scale)
        try:
            res = least_squares(resid, x, bounds=(lo, hi_b), loss="soft_l1", f_scale=F_SCALE,
                                x_scale=scale, max_nfev=MAX_NFEV)
        except (ValueError, FloatingPointError):
            continue
        if res.status <= 0 or not np.all(np.isfinite(res.x)):
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return None

    params = {k: float(v) for k, v in full(best.x).items()}
    pred = _predict_raw(params, temp_on, ref, slip, st_all)
    rmse = float(np.sqrt(np.sum(n * (y - pred) ** 2) / n_total))
    fit = MFFit(params=params, ref_state={k: float(v) for k, v in ref.items()},
                factors_status=status, measure=bins.measure, n_cells=int(len(y)),
                n_samples=int(n_total), rmse=rmse, peak_slip_deg=0.0, peak_mu=0.0,
                peak_observed=False, max_observed_slip_deg=max_obs, target=target,
                notes=notes)
    for p in free:
        lo_p, hi_p = BOUNDS[p]
        if min(params[p] - lo_p, hi_p - params[p]) <= 0.01 * (hi_p - lo_p):
            fit.at_bound.append(p)
    if temp_on and any(p in fit.at_bound for p in FACTOR_PARAMS["temp"]):
        # Prediction still uses the fitted values; the status tells consumers not to trust them.
        fit.factors_status["temp"] = "at_bound"
    fit.peak_slip_deg, fit.peak_mu = peak_at(fit)

    # Peak observed: a well-sampled bin beyond the peak whose mean sits below the model's
    # peak AT THAT CELL'S STATE (D scales the whole curve; B only stretches slip, so the
    # state-specific peak is D(state) * shape_peak(C, E)).
    # Ruling 8: judged per SLIP BIN (marginal n >= 20) on the count-weighted mean of
    # cell mean / model peak at the cell's state, not per sparse cell.
    dm, _ = _scales(params, temp_on, ref, st_cell)
    cell_peak = params["D0"] * dm * _shape_peak(params["C"], params["E"])
    ratio = y / cell_peak
    for j in np.flatnonzero(dense):
        sel_b = inv == j
        if np.average(slip[sel_b], weights=n[sel_b]) >= fit.peak_slip_deg + PEAK_OBS_MARGIN_DEG:
            if np.average(ratio[sel_b], weights=n[sel_b]) <= PEAK_OBS_RATIO:
                fit.peak_observed = True
                break
    return fit


# ---------------------------------------------------------------- bootstrap ----------

BOOT_SINGLE_LAP_CAP = 0.3   # one lap cannot show lap-to-lap variation
BOOT_CONF_CAP = 0.95        # never claim certainty
BOOT_LAPS_TAU = 8.0         # confidence ~ 1 - exp(-n_laps / tau)
BOOT_UNOBSERVED_PEAK = 0.6  # peak extrapolated beyond the data
BOOT_AT_BOUND = 0.8         # a factor pinned at a bound is an extrapolation
_INT_FIELDS = {f for f in _CELL_FIELDS if f.startswith("n")}


class _CellTable:
    """All cells of all laps as one matrix, so a resample is a weighted group-by-key sum.

    WHY: LapBins.merge walks every cell/field in Python; doing that for each of 50
    resamples costs more than the fits. Cell fields are plain sums, so resampling with
    multiplicity k is exactly "add the cell k times" = weight the row by k.
    """

    def __init__(self, laps):
        keys, rows, owner = [], [], []
        for i, lap in enumerate(laps):
            for k, c in lap.cells.items():
                keys.append(k)
                rows.append([getattr(c, f) for f in _CELL_FIELDS])
                owner.append(i)
        self.measure = laps[0].measure
        self.mat = np.array(rows, dtype=float)
        self.owner = np.array(owner)
        uniq, self.inv = np.unique(np.array(keys), axis=0, return_inverse=True)
        self.inv = self.inv.ravel()
        self.keys = [tuple(int(v) for v in u) for u in uniq]

    def build(self, weights) -> LapBins:
        """LapBins of the rows weighted by `weights` (per row)."""
        m = len(self.keys)
        cols = [np.bincount(self.inv, weights=weights * self.mat[:, j], minlength=m)
                for j in range(self.mat.shape[1])]
        out = LapBins(self.measure)
        n_col = _CELL_FIELDS.index("n")
        for g in np.flatnonzero(cols[n_col] > 0):
            out.cells[self.keys[g]] = Cell(**{
                f: (int(round(cols[j][g])) if f in _INT_FIELDS else float(cols[j][g]))
                for j, f in enumerate(_CELL_FIELDS)})
        return out


def bootstrap_fit(lap_bins: list, n_boot: int = 50, seed: int = 0, **fit_kw) -> dict:
    """Fit all laps, then bootstrap over LAPS to get 90 % CIs on the peak and a confidence.

    Laps (not samples) are the resampling unit: samples within a lap are strongly
    correlated, laps are the independent replicates. With a single lap there is nothing
    to resample, so cells are resampled instead and confidence is capped at 0.3.
    """
    empty = {"fit": None, "peak_slip_ci90": None, "peak_mu_ci90": None,
             "n_boot_ok": 0, "confidence": 0.0}
    laps = [l for l in lap_bins if l.cells]
    if not laps:
        return empty
    n_laps = len(laps)
    fit = fit_mf(LapBins.merge(laps), **fit_kw)
    if fit is None:
        return empty

    table = _CellTable(laps)
    rng = np.random.default_rng(seed)
    slips, mus = [], []
    for _ in range(n_boot):
        if n_laps > 1:
            w = np.bincount(rng.integers(0, n_laps, n_laps), minlength=n_laps)[table.owner]
        else:
            w = np.bincount(rng.integers(0, len(table.owner), len(table.owner)),
                            minlength=len(table.owner))
        # Warm start from the full fit; resamples are perturbations of the same data.
        f = fit_mf(table.build(w.astype(float)), x0=fit.params, **fit_kw)
        if f is not None:
            slips.append(f.peak_slip_deg)
            mus.append(f.peak_mu)

    out = {"fit": fit, "peak_slip_ci90": None, "peak_mu_ci90": None,
           "n_boot_ok": len(slips), "confidence": 0.0}
    if not slips:
        return out
    lo, hi = (float(v) for v in np.percentile(slips, [5, 95]))
    out["peak_slip_ci90"] = [lo, hi]
    out["peak_mu_ci90"] = [float(v) for v in np.percentile(mus, [5, 95])]

    width = float(np.clip(1.0 - (hi - lo) / fit.peak_slip_deg, 0.0, 1.0))
    conf = ((1.0 - np.exp(-n_laps / BOOT_LAPS_TAU)) * width
            * (1.0 if fit.peak_observed else BOOT_UNOBSERVED_PEAK) * (len(slips) / n_boot))
    if "at_bound" in fit.factors_status.values():
        conf *= BOOT_AT_BOUND
    if n_laps == 1:
        conf = min(conf, BOOT_SINGLE_LAP_CAP)
    out["confidence"] = round(min(conf, BOOT_CONF_CAP), 3)
    return out
