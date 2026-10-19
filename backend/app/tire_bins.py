"""Per-lap binned sufficient statistics for the tyre model (v3).

A lap's axle samples are reduced to cells keyed by (slip, load_ratio, temp_delta,
pressure, wear, camber) bin indices. Each cell keeps counts and sums, so laps can be
stored compactly in the profile and merged exactly (sums add) without re-reading
telemetry. Carcass delta and |long G| are carried as per-cell means, not binned.
"""

from dataclasses import dataclass, field, fields

import numpy as np

STATE_VARS = ("load_ratio", "temp_delta_c", "pressure_kpa", "wear", "camber_deg",
              "carcass_delta_c", "long_g")
BIN = {"slip": 0.4, "load_ratio": 0.1, "temp_delta_c": 5.0, "pressure_kpa": 5.0,
       "wear": 0.05, "camber_deg": 0.5}
BINNED_VARS = STATE_VARS[:5]
SLIP_MAX_DEG = 15.0
UNKNOWN_BIN = -999
LOAD_FLOOR = 0.2
MIN_LOAD_FRACTION = 0.5
MEASURE_MU = "mu_load_normalized"
MEASURE_LAT = "lateral_g"

# Keys of the sample dict -> state variable name (Ruling 1: pressure is renamed).
_SAMPLE_KEY = {"pressure_kpa": "tire_pressure_kpa"}


@dataclass
class Cell:
    n: int = 0
    sum_mu: float = 0.0
    sumsq_mu: float = 0.0
    sum_slip: float = 0.0
    n_true: int = 0
    sum_true: float = 0.0
    sumsq_true: float = 0.0
    sum_load_ratio: float = 0.0
    n_load_ratio: int = 0
    sum_temp_delta_c: float = 0.0
    n_temp_delta_c: int = 0
    sum_pressure_kpa: float = 0.0
    n_pressure_kpa: int = 0
    sum_wear: float = 0.0
    n_wear: int = 0
    sum_camber_deg: float = 0.0
    n_camber_deg: int = 0
    sum_carcass_delta_c: float = 0.0
    n_carcass_delta_c: int = 0
    sum_long_g: float = 0.0
    n_long_g: int = 0


_CELL_FIELDS = [f.name for f in fields(Cell)]


def _r(x):
    """Round to ~6 significant digits so the JSON stays compact."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return float(f"{float(x):.6g}")


@dataclass
class LapBins:
    measure: str
    cells: dict = field(default_factory=dict)

    def n_samples(self) -> int:
        return sum(c.n for c in self.cells.values())

    def to_dict(self) -> dict:
        # Rows are flat lists [*key, *cell fields] - far smaller than per-row dicts.
        rows = [list(k) + [_r(getattr(c, f)) for f in _CELL_FIELDS]
                for k, c in self.cells.items()]
        return {"measure": self.measure, "rows": rows}

    @staticmethod
    def from_dict(d: dict) -> "LapBins":
        nk = len(BINNED_VARS) + 1
        cells = {}
        for row in d["rows"]:
            vals = dict(zip(_CELL_FIELDS, row[nk:]))
            cells[tuple(int(v) for v in row[:nk])] = Cell(**vals)
        return LapBins(d["measure"], cells)

    @staticmethod
    def merge(others: list) -> "LapBins":
        if not others:
            raise ValueError("nothing to merge")
        # WHY: an empty LapBins (a lap with no usable samples) defaults to some
        # measure and would trip the mixed-measure check against real laps.
        others = [o for o in others if o.cells]
        if not others:
            return LapBins(MEASURE_MU)
        measure = others[0].measure
        if any(o.measure != measure for o in others):
            raise ValueError("cannot merge bins with different measures")
        out = LapBins(measure)
        for o in others:
            for k, c in o.cells.items():
                t = out.cells.setdefault(k, Cell())
                for f in _CELL_FIELDS:
                    setattr(t, f, getattr(t, f) + getattr(c, f))
        return out


def _arr(s, key, n):
    v = s.get(key)
    if v is None:
        return np.full(n, np.nan)
    return np.asarray(v, dtype=float)


def _bin_index(vals, width):
    idx = np.floor(vals / width)
    out = np.full(len(vals), UNKNOWN_BIN, dtype=np.int64)
    ok = np.isfinite(vals)
    out[ok] = idx[ok].astype(np.int64)
    return out


def bin_axle_samples(s: dict, mask: np.ndarray) -> LapBins:
    slip_raw = _arr(s, "slip_angle_deg", len(s["slip_angle_deg"]))
    n = len(slip_raw)
    slip = np.abs(slip_raw)
    lat = np.abs(_arr(s, "lateral_g", n))
    load = _arr(s, "load_ratio", n)
    state = {}
    for v in STATE_VARS:
        a = _arr(s, _SAMPLE_KEY.get(v, v), n)
        # WHY |long_g|: combined slip depends on magnitude; the sign averages out
        # to ~0 within a cell and would hide braking vs traction load.
        state[v] = np.abs(a) if v == "long_g" else a
    true_mu = _arr(s, "true_mu", n)

    keep = (np.asarray(mask, bool) & np.isfinite(slip) & (slip < SLIP_MAX_DEG)
            & np.isfinite(lat))
    has_load = keep & np.isfinite(load)
    if keep.any() and has_load.sum() >= MIN_LOAD_FRACTION * keep.sum():
        measure, keep = MEASURE_MU, has_load
        grip = lat / np.maximum(load, LOAD_FLOOR)
    else:
        measure = MEASURE_LAT
        grip = lat
    out = LapBins(measure)
    if not keep.any():
        return out

    slip, grip, true_mu = slip[keep], grip[keep], true_mu[keep]
    state = {v: a[keep] for v, a in state.items()}
    idx = np.stack([_bin_index(slip, BIN["slip"])]
                   + [_bin_index(state[v], BIN[v]) for v in BINNED_VARS], axis=1)
    keys, inv = np.unique(idx, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = len(keys)

    def acc(x):
        return np.bincount(inv, weights=x, minlength=m)

    ok_t = np.isfinite(true_mu)
    t0 = np.where(ok_t, true_mu, 0.0)
    cols = {
        "n": acc(np.ones(len(grip))), "sum_mu": acc(grip), "sumsq_mu": acc(grip ** 2),
        "sum_slip": acc(slip), "n_true": acc(ok_t.astype(float)),
        "sum_true": acc(t0), "sumsq_true": acc(t0 ** 2),
    }
    for v in STATE_VARS:
        fin = np.isfinite(state[v])
        cols["sum_" + v] = acc(np.where(fin, state[v], 0.0))
        cols["n_" + v] = acc(fin.astype(float))
    int_fields = {"n", "n_true"} | {"n_" + v for v in STATE_VARS}
    for i, k in enumerate(keys):
        vals = {f: (int(round(cols[f][i])) if f in int_fields else float(cols[f][i]))
                for f in _CELL_FIELDS}
        out.cells[tuple(int(x) for x in k)] = Cell(**vals)
    return out


def cell_arrays(bins: LapBins) -> dict:
    cs = list(bins.cells.values())
    n = np.array([c.n for c in cs], dtype=float)
    sm = np.array([c.sum_mu for c in cs])
    sq = np.array([c.sumsq_mu for c in cs])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sm / n
        res = {
            "n": n, "mu_mean": mean,
            "mu_var": np.maximum(sq / n - mean ** 2, 0.0),
            "slip": np.array([c.sum_slip for c in cs]) / n,
        }
        for v in STATE_VARS:
            sv = np.array([getattr(c, "sum_" + v) for c in cs])
            nv = np.array([getattr(c, "n_" + v) for c in cs], dtype=float)
            res[v] = np.where(nv > 0, sv / nv, np.nan)
        nt = np.array([c.n_true for c in cs], dtype=float)
        res["true_n"] = nt
        res["true_mean"] = np.where(nt > 0, np.array([c.sum_true for c in cs]) / nt, np.nan)
    return res


def coverage(bins: LapBins) -> dict:
    counts: dict = {}
    per_slip: dict = {}
    for k, c in bins.cells.items():
        per_slip[k[0]] = per_slip.get(k[0], 0) + c.n
        if c.n and c.sum_slip / c.n >= 1.5:
            key = (k[1], k[2])  # (load_ratio bin, temp_delta bin)
            counts[key] = counts.get(key, 0) + c.n
    return {"counts": counts,
            "slip_bins_reached": sorted(b for b, n in per_slip.items() if n >= 20)}
