"""Ideal lap, the car model (design 2026-11-10-ideal-lap-design.md §4), fitted
from the car's own recordings. Ported from the spike's il_core.fit_car with the
probe fixes (fable_p2 variant 'both') and the design's changes.

Units: accelerations in g; gx kinematic (+ speeding up); gxt = gx + grade +
drag(v) = the longitudinal force the tyres supply, in g. lam = wheel load / its
static value (only the ratio is used: Susp Force is not wheel load).

Two scopes: the LOAD MODEL is per file (one garage stint = one setup); grip,
braking, traction, drag and engine are per fit group (car model, track, layout).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .ideal_state import G, GRID_HZ, State

# Version of the fit code: part of every group's fingerprint (ideal_store), so
# a profile fitted by older code is refitted. BUMP IT WITH ANY CHANGE THAT CAN
# MOVE A FITTED NUMBER. 2: round 1 (continuous knee, k / c span rule).
FIT_VERSION = 2
FRONT, REAR = (0, 1), (2, 3)
AXLES = ("front", "rear")
LAM_FLOOR = 0.02
K_MAX, C_MAX = 0.15, 0.45          # bounds of the load and speed terms of mu
OBSERVED_MIN = 250                 # saturated samples (5 s) for an axle to be `observed`
KNEE_BIN_DEG, KNEE_MIN_N, KNEE_FRAC, KNEE_SPAN_DEG = 0.5, 60, 0.97, 4.0
GF_SATURATED = 0.3                 # Grip Fraction is LOW when saturated
KNEE_V3_DIFFER_DEG = 1.5
# k, c of an axle are fitted only where its saturated set spans them (§13 R2)
TERMS_MIN_N, TERMS_V_SPAN, TERMS_LAM_SPAN = 250, 15.0, 0.15
K_DEFAULT, C_DEFAULT = 0.03, 0.10          # middle of what has been fitted (k 0-0.08, c 0.05-0.19)
EXTRAP_MARGIN = 5.0                        # m/s above v_sat[1]
DRAG_PRIOR = {"LMGT3": (0.015, 4.8e-5)}    # d0 (g), d2 (g per (m/s)^2)
DRAG_PRIOR_OTHER = (0.015, 4.0e-5)         # Ferrari measured 0.054 g + 3.98e-5 v^2
COAST_MIN, COAST_D2_CAP = 150, 7e-5


# ── 4.1 Load model (per file) ────────────────────────────────────────────────

def load_features(v, gx, gy, gz) -> np.ndarray:
    """[v^2, gx, gy, gz, |gy|] per sample. PREVIOUSLY (spike) there was no |gy|
    term. Fix (F1): in hard cornering the axle-mean load moved against the
    linear model (Porsche COTA esses: front Susp Force +0.15, rear -0.16, likely
    suspension jacking); the |gy| term per wheel carries it."""
    v, gx, gy, gz = np.broadcast_arrays(*(np.asarray(a, float) for a in (v, gx, gy, gz)))
    return np.stack([v * v, gx, gy, gz, np.abs(gy)], axis=-1)


def _lstsq_load(v, gx, gy, gz, L) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([np.ones(np.shape(v) + (1,)), load_features(v, gx, gy, gz)], axis=-1)
    C = np.array([np.linalg.lstsq(X, L[i], rcond=None)[0] for i in range(4)])
    c0 = C[:, 0].copy()
    return c0, C[:, 1:] / c0[:, None]


@dataclass
class LoadModel:
    """L_i = c0_i * (1 + a v^2 + b gx + c gy + d gz + e |gy|); lam = L / c0."""
    static: np.ndarray                    # c0 per wheel (units of the source)
    coef: np.ndarray                      # 4 x 5: a, b, c, d, e
    source: str                           # 'load' (Tire Load) | 'susp' (Susp Force)
    r2: list = field(default_factory=list)   # per wheel, odd laps scored when fitted on even laps
    scope: str = "file"                   # 'file' | 'group' (borrowed, too few laps)

    def lam(self, v, gx, gy, gz) -> np.ndarray:
        X = load_features(v, gx, gy, gz)
        return np.maximum(1.0 + np.einsum("ij,...j->i...", self.coef, X), LAM_FLOOR)

    def to_dict(self) -> dict:
        return {"static": [float(x) for x in self.static], "coef": np.asarray(self.coef).tolist(),
                "source": self.source, "r2": [float(x) for x in self.r2], "scope": self.scope}

    @classmethod
    def from_dict(cls, d: dict) -> "LoadModel":
        return cls(static=np.asarray(d["static"], float), coef=np.asarray(d["coef"], float),
                   source=d["source"], r2=list(d.get("r2") or []), scope=d.get("scope", "file"))


def _cat(states: list[State], key: str) -> np.ndarray:
    return np.concatenate([getattr(s, key) for s in states], axis=-1)


def fit_load_model(states: list[State], source: str, scope: str = "file") -> LoadModel:
    """Least squares per wheel over the laps (design §4.1); R^2 per wheel on the
    odd laps of a fit on the even laps (needs >= 2 laps)."""
    def arrays(sts):
        return [_cat(sts, k) for k in ("v", "gx", "gy", "gz")] + [np.concatenate([s.L(source) for s in sts], axis=-1)]

    v, gx, gy, gz, L = arrays(states)
    c0, coef = _lstsq_load(v, gx, gy, gz, L)
    r2 = []
    even, odd = states[0::2], states[1::2]
    if even and odd:
        e0, ecoef = _lstsq_load(*arrays(even))
        ov, ogx, ogy, ogz, oL = arrays(odd)
        pred = e0[:, None] * (1.0 + ecoef @ load_features(ov, ogx, ogy, ogz).T)
        for i in range(4):
            r2.append(float(1 - np.sum((oL[i] - pred[i]) ** 2) / np.sum((oL[i] - oL[i].mean()) ** 2)))
    return LoadModel(static=c0, coef=coef, source=source, r2=r2, scope=scope)


# ── The car model the solver uses ────────────────────────────────────────────

def axle_capacity(lam: np.ndarray, v, mu1, k, c) -> tuple[np.ndarray, np.ndarray]:
    """(A_front, A_rear): the lateral g of the car when that axle is saturated,
    A_j = 1/2 sum_{i in j} mu_j(lam_i, v) lam_i, mu_j = mu1 max(1 - k(lam-1) - c(v-40)/40, 0.3)."""
    out = []
    for j, idx in enumerate((FRONT, REAR)):
        a = 0.0
        for i in idx:
            a = a + 0.5 * mu1[j] * np.maximum(1 - k[j] * (lam[i] - 1) - c[j] * (v - 40) / 40, 0.3) * lam[i]
        out.append(a)
    return out[0], out[1]


@dataclass
class CarModel:
    """One file's load model + its fit group's grip, braking, traction, drag
    and engine. All JSON-able numbers (to_dict / from_dict)."""
    load: LoadModel
    mu1: list
    k: list
    c: list
    s_front: float
    kb: float
    kt: float
    d0: float
    d2: float
    power_v: list
    power_g: list
    corner_drag: float

    @property
    def sf(self) -> float:
        return self.s_front

    @property
    def sr(self) -> float:
        return 1.0 - self.s_front

    def lam(self, v, gx, gy, gz) -> np.ndarray:
        return self.load.lam(v, gx, gy, gz)

    def axles(self, v, gx, gy, gz):
        return axle_capacity(self.lam(v, gx, gy, gz), v, self.mu1, self.k, self.c)

    def drag(self, v):
        return self.d0 + self.d2 * v * v

    def power(self, v, gy):
        return np.interp(v, self.power_v, self.power_g) - self.corner_drag * gy * gy

    def scaled(self, level: float) -> "CarModel":
        """Grip level x `level` on both axles (design §5.4 levels 1-beta, 1, 1+beta)."""
        d = self.to_dict()
        d["mu1"] = [m * level for m in d["mu1"]]
        return CarModel.from_dict(d)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "load"}
        d["load"] = self.load.to_dict()
        for k in ("mu1", "k", "c", "power_v", "power_g"):
            d[k] = [float(x) for x in d[k]]
        for k in ("s_front", "kb", "kt", "d0", "d2", "corner_drag"):
            d[k] = float(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CarModel":
        kw = {k: v for k, v in d.items() if k != "load"}
        return cls(load=LoadModel.from_dict(d["load"]), **kw)


class InsufficientData(Exception):
    """The group's data cannot give a car model; str(e) is the reason."""


# ── 4.2 Lateral capacity per axle ────────────────────────────────────────────

def knee_mask(slip_ax: np.ndarray, r: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, float]:
    """Saturated without force data: at or past the slip angle where r = |gy| /
    axle-mean lam stops rising. Median r per 0.5 deg bin (>= 60 samples), the
    medians smoothed by a 3-bin running median; plateau = median of the three
    highest; knee = slip angle (linear between bin centres) where the smoothed
    curve first reaches 97 % of the plateau; set = knee <= slip <= knee + 4 deg.
    PREVIOUSLY: knee = lower edge of the first bin whose raw median reached 97 %
    of the highest. On a flat plateau that jumped a whole bin from one sample
    set to the next (Porsche COTA front 4.0 -> 3.0 deg when the drag entered
    the base mask), moving the front capacity 4-5 % (design §13 R3). Fix: the
    continuous rule above."""
    edges = np.arange(0, 14.5, KNEE_BIN_DEG)
    med, mid = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = base & (slip_ax >= a) & (slip_ax < b)
        if m.sum() >= KNEE_MIN_N:
            med.append(np.median(r[m])); mid.append(0.5 * (a + b))
    if len(med) < 3:
        return base & False, float("nan")
    med, mid = np.array(med), np.array(mid)
    pad = np.concatenate([[med[0]], med, [med[-1]]])
    smo = np.median(np.vstack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
    thr = KNEE_FRAC * float(np.median(np.sort(smo)[-3:]))
    i = int(np.argmax(smo >= thr))
    if i == 0:
        knee = float(mid[0])
    else:
        f = (thr - smo[i - 1]) / (smo[i] - smo[i - 1])
        knee = float(mid[i - 1] + f * (mid[i] - mid[i - 1]))
    return base & (slip_ax >= knee) & (slip_ax <= knee + KNEE_SPAN_DEG), knee


def _pqr(lam_ax: np.ndarray, v: np.ndarray):
    """|gy| = mu1 (P - k Q - c R) on a saturated axle."""
    P = 0.5 * lam_ax.sum(0)
    Q = 0.5 * ((lam_ax - 1) * lam_ax).sum(0)
    return P, Q, P * (v - 40) / 40


def _ls_mu(y, P, Q, R, k_fix=None, c_fix=None):
    """Least squares of mu1, k, c with k in [0, 0.15] and c in [0, 0.45]: the
    free solution and the pinned corners, smallest residual sd that respects the
    bounds wins (il_core.fit_car step 3). k_fix / c_fix: given, not fitted."""
    best = None
    k_opts = (None, 0.0, K_MAX) if k_fix is None else (k_fix,)
    c_opts = (None, 0.0, C_MAX) if c_fix is None else (c_fix,)
    for kf in k_opts:
        for cf in c_opts:
            base = P - (kf or 0.0) * Q - (cf or 0.0) * R
            cols = [base] + ([] if kf is not None else [-Q]) + ([] if cf is not None else [-R])
            sol = np.linalg.lstsq(np.c_[tuple(cols)], y, rcond=None)[0]
            m1 = float(sol[0])
            if m1 <= 0:
                continue
            it = iter(sol[1:])
            k = kf if kf is not None else float(next(it) / m1)
            c = cf if cf is not None else float(next(it) / m1)
            if not (0.0 <= k <= K_MAX) or not (0.0 <= c <= C_MAX):
                continue
            err = float(np.std(y - m1 * (P - k * Q - c * R)))
            if best is None or err < best[0] - 1e-9:
                best = (err, m1, k, c)
    if best is None:
        raise InsufficientData("lateral grip fit failed (no positive level)")
    return best[1], best[2], best[3]


@dataclass
class AxleFit:
    mu1: float
    k: float
    c: float
    status: str                 # 'observed' | 'lower_bound'
    saturation: str             # 'grip_fraction' | 'knee'
    knee_deg: float | None
    n_sat: int
    ratio_sd: float             # log sd of observed / predicted on the set
    v3_peak_slip_deg: float | None = None
    knee_vs_v3: str | None = None   # 'agree' | 'differ'
    terms: str = "fitted"           # k, c: 'fitted' | 'borrowed' | 'default' (§13 R2)
    v_sat: list | None = None       # [p02, p98] of the set's speed, m/s

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and not np.isfinite(v):
                d[k] = None
        return d


def is_extrapolated(axle: AxleFit, v) -> bool:
    """Design §13 R2: a speed more than 5 m/s above the fastest saturated sample
    of the axle (its v_sat[1]) is an extrapolation of the speed term."""
    return axle.v_sat is not None and float(v) > axle.v_sat[1] + EXTRAP_MARGIN


# ── The group fit ────────────────────────────────────────────────────────────

@dataclass
class GroupFit:
    """Per fit group (car model, track, layout): everything but the load
    model, plus each file's load model. `car_for(file)` gives the solver's car."""
    car_class: str
    load_models: dict           # file -> LoadModel
    mu1: list
    k: list
    c: list
    s_front: float
    kb: float
    kt: float
    d0: float
    d2: float
    power_v: list
    power_g: list
    corner_drag: float
    info: dict                  # axles, s_front/kappa/drag sources and counts, notes

    def car_for(self, file: str) -> CarModel:
        lm = self.load_models.get(file) or self.load_models.get("__group__")
        if lm is None:
            raise KeyError(file)
        return CarModel(load=lm, mu1=list(self.mu1), k=list(self.k), c=list(self.c), s_front=self.s_front,
                        kb=self.kb, kt=self.kt, d0=self.d0, d2=self.d2, power_v=list(self.power_v),
                        power_g=list(self.power_g), corner_drag=self.corner_drag)

    def axle(self, j: int) -> AxleFit:
        return AxleFit(**self.info["axles"][AXLES[j]])

    def to_dict(self) -> dict:
        """JSON-able. Load models are under `load_models` keyed by file."""
        d = {k: getattr(self, k) for k in ("car_class", "mu1", "k", "c", "s_front", "kb", "kt", "d0", "d2",
                                           "power_v", "power_g", "corner_drag", "info")}
        d["load_models"] = {f: lm.to_dict() for f, lm in self.load_models.items()}
        return _jsonable(d)

    @classmethod
    def from_dict(cls, d: dict) -> "GroupFit":
        kw = {k: v for k, v in d.items() if k != "load_models"}
        return cls(load_models={f: LoadModel.from_dict(x) for f, x in d["load_models"].items()}, **kw)


def _jsonable(x):
    """numpy scalars / arrays to plain Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def _pool(reductions, load_models):
    """Concatenate the group's qualifying laps; lam is each file's MODELLED lam."""
    parts = {k: [] for k in ("v", "gx", "gy", "gz", "grade", "thr", "brk", "abs", "tc", "yawacc")}
    lam, slip, gf, fx, fy, lap_id = [], [], [], [], [], []
    for red in reductions:
        lm = load_models[red.file]
        for s in red.laps:
            for k in parts:
                parts[k].append(getattr(s, k))
            lam.append(lm.lam(s.v, s.gx, s.gy, s.gz))
            slip.append(s.slip); gf.append(s.gf); fx.append(s.fx); fy.append(s.fy)
            lap_id.append(np.full(len(s.v), len(lap_id)))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["lam"] = np.concatenate(lam, axis=1)
    out["lap_id"] = np.concatenate(lap_id)
    for k, arrs in (("slip", slip), ("gf", gf), ("fx", fx), ("fy", fy)):
        out[k] = np.concatenate(arrs, axis=1) if all(a is not None for a in arrs) else None
    return out


# Every fit below takes the pooled dict `d` (v, gx, gy, gz, grade, thr, brk, abs,
# tc, yawacc: 1-D; lam, slip, gf, fx, fy: 4 x n or None) and `q`.
# `q` is TEST ONLY (the build check against the spike at q 0.75); the design
# fixes the median everywhere (§4.2, §4.8) and nothing in the app passes another.

def fit_drag(d: dict, has_force: bool, car_class: str) -> dict:
    """Design §4.6: {d0, d2, source, mass_kg}.
    PREVIOUSLY (spike): without tyre forces, drag was fitted jointly with the
    braking limit and bounded by coasting; it returned 0 on Road Atlanta and
    4.5-6e-5 on COTA, and its bound was the hard-coded default both times
    (< 50 coasting samples), so drag without forces was never measured (F8).
    Fix: coast-down when the group has >= 3 s of it (an upper bound: it holds
    engine braking; capped at 7e-5), else a class prior. Drag only splits gxt
    between tyres and air; the engine curve is kinematic and contains it."""
    v, gx, gy, gr = d["v"], d["gx"], d["gy"], d["grade"]
    if has_force:
        Fx, Fy = d["fx"], d["fy"]
        ok = v > 15
        mass = abs(np.polyfit(gy[ok], Fy.sum(0)[ok], 1)[0]) / G
        sgn = np.sign(np.polyfit(gx[ok], Fx.sum(0)[ok], 1)[0])
        D = (sgn * Fx.sum(0) - mass * G * (gx + gr)) / (mass * G)
        dd = np.linalg.lstsq(np.c_[np.ones(ok.sum()), v[ok] ** 2], D[ok], rcond=None)[0]
        return dict(d0=float(max(dd[0], 0)), d2=float(max(dd[1], 0)), source="forces", mass_kg=float(mass))
    coast = (d["thr"] < 2) & (d["brk"] < 1) & (np.abs(gy) < 0.3) & (v > 40)
    if coast.sum() >= COAST_MIN:
        d2 = float(min(max(np.median(-(gx + gr)[coast] / v[coast] ** 2), 0.0), COAST_D2_CAP))
        return dict(d0=0.0, d2=d2, source="coast_down", mass_kg=None, n=int(coast.sum()))
    d0, d2 = DRAG_PRIOR.get(car_class, DRAG_PRIOR_OTHER)
    return dict(d0=d0, d2=d2, source="prior", mass_kg=None, n=int(coast.sum()))


def fit_share(d: dict, has_force: bool, statics: list) -> dict:
    """Design §4.3: front share of the lateral force. From the forces in
    quasi-steady hard cornering; else the static front share of the load
    source, flagged `estimate` (Susp Force share is not weight share)."""
    if has_force:
        gy, Fy = d["gy"], d["fy"]
        qs = (np.abs(gy) > 1.0) & (np.abs(d["gx"] + d["grade"]) < 0.2) & (np.abs(d["yawacc"]) < 0.15)
        return dict(value=float(np.median((Fy[:2].sum(0) / Fy.sum(0))[qs])), source="lateral_force")
    shares = [float(np.sum(s[:2]) / np.sum(s)) for s in statics]
    return dict(value=float(np.mean(shares)), source="estimate")


def fit_axles(d: dict, gxt: np.ndarray, slip: np.ndarray | None, gf: np.ndarray | None,
              v3_peaks: dict | None = None, q: float = 0.5) -> list:
    """Design §4.2: [AxleFit front, AxleFit rear], fitted on d['lam'], the
    MODELLED load ratio.
    PREVIOUSLY (spike): grip was fitted against the MEASURED load ratio and the
    solver then used the MODELLED one; they differ in hard cornering (Porsche
    front 1.17-1.18 measured vs 1.10-1.11 modelled at 120-180 km/h), so the
    driver "exceeded" the front limit on 41 % of hard samples while the rear
    never passed 0.93 (F1). Fix: every fit uses lam from the load model at the
    sample's own (v, gx, gy, gz), never the measured load.
    Calibration at the median (q 0.5): past the knee a tyre holds its peak
    force, so a saturated sample IS the limit plus noise."""
    v, gy, yawacc, lam = d["v"], d["gy"], d["yawacc"], d["lam"]
    if gf is None and slip is None:
        raise InsufficientData("no slip angles and no Grip Fraction")
    y_all = np.abs(gy)
    gymax = np.percentile(y_all, 99)
    base = (np.abs(gxt) < 0.2) & (y_all > 0.45 * gymax) & (v > 15) & (np.abs(yawacc) < 0.4)
    sets = []
    for idx in (FRONT, REAR):
        idx = list(idx)
        r = y_all / lam[idx].mean(0)
        if slip is not None:
            m_knee, knee = knee_mask(slip[idx].mean(0), r, base)
        else:
            m_knee, knee = base & False, float("nan")
        if gf is not None:
            sets.append((idx, r, base & (gf[idx].mean(0) < GF_SATURATED), "grip_fraction", knee))
        else:
            sets.append((idx, r, m_knee, "knee", knee))

    fits: list = [None, None]

    def finish(j, m1, k, c, m, m_sat, status, terms):
        idx, _, _, how, knee = sets[j]
        P, Q, R = _pqr(lam[idx], v)
        shape = P[m] - k * Q[m] - c * R[m]
        mu = m1 * float(np.quantile(y_all[m] / (m1 * shape), q))
        v3 = (v3_peaks or {}).get(AXLES[j])
        vs = None
        if v3 is not None and np.isfinite(knee):
            vs = "differ" if abs(v3 - knee) > KNEE_V3_DIFFER_DEG else "agree"
        fits[j] = AxleFit(mu1=float(mu), k=float(k), c=float(c), status=status, saturation=how,
                          knee_deg=float(knee) if np.isfinite(knee) else None, n_sat=int(m_sat.sum()),
                          ratio_sd=float(np.std(np.log(y_all[m] / (mu * shape)))),
                          v3_peak_slip_deg=v3, knee_vs_v3=vs, terms=terms, v_sat=vsat(m_sat if m_sat.any() else m))

    def vsat(m):
        return [float(x) for x in np.percentile(v[m], [2, 98])]

    def spans(j):
        """Does the saturated set span the load and speed terms (§13 R2)?"""
        idx, _, m, _, _ = sets[j]
        if m.sum() < TERMS_MIN_N:
            return False
        p = lambda a: float(np.percentile(a[m], 90) - np.percentile(a[m], 10))
        return p(v) >= TERMS_V_SPAN and p(lam[idx].mean(0)) >= TERMS_LAM_SPAN

    # PREVIOUSLY: every observed axle fitted its own k, c. An axle saturated in a
    # narrow speed band cannot tell its speed term, and the least squares pinned
    # it at a bound (synthetic rear seen at slow apexes only: c 0.45, capacity
    # 0.74 x true at 60-90 m/s; real sets are 20-50 m/s). Fix (§13 R2): fit k, c
    # only where the set spans them; else borrow the other axle's fitted terms,
    # else the defaults; mu1 is always fitted on the axle's own set.
    observed = [int(s[2].sum()) >= OBSERVED_MIN for s in sets]
    own = [spans(j) for j in (0, 1)]
    for j in (0, 1):
        if own[j]:
            idx, _, m, _, _ = sets[j]
            P, Q, R = _pqr(lam[idx], v)
            m1, k, c = _ls_mu(y_all[m], P[m], Q[m], R[m])
            finish(j, m1, k, c, m, m, "observed", "fitted")
    for j in (0, 1):
        if own[j]:
            continue
        idx, r, m_sat, _, _ = sets[j]
        if observed[j]:
            m = m_sat
        else:
            # lower_bound: the car was not seen to reach this axle's limit; the
            # capacity is what was shown (top 20 % of r among `base`) (mirrors
            # tyre v3's rising_at_edge).
            if base.sum() < 10:
                raise InsufficientData("too few cornering samples")
            m = base & (r >= np.quantile(r[base], 0.8))
        other = fits[1 - j]
        if other is not None and other.terms == "fitted":
            k, c, terms = other.k, other.c, "borrowed"
        else:
            k, c, terms = K_DEFAULT, C_DEFAULT, "default"
        P, Q, R = _pqr(lam[idx], v)
        m1, k, c = _ls_mu(y_all[m], P[m], Q[m], R[m], k_fix=k, c_fix=c)
        finish(j, m1, k, c, m, m_sat, "observed" if observed[j] else "lower_bound", terms)
    return fits


TRIM_MIN_N, TRIM_HALF_N, TRIM_CAP = 250, 500, 0.08     # design §14 R8


def axle_trim(car: CarModel, axles: list, red) -> tuple[CarModel, list]:
    """Design §14 R8: move the group's grip level per axle to the reference
    file's own saturated samples. t_j = median log(|gy| / A_j) over the file's
    saturated set of axle j (group mu1, k, c; the file's own load model in
    `car`), applied as mu1_j exp(w_j t_j), w_j = n_j / (n_j + 500), only when
    n_j >= 250 (else `none`); |t_j| capped at 0.08 (`capped`: a different car or
    a broken fit). The level only: k, c, kb, kt, drag and engine stay the
    group's (braking and traction then use the trimmed A_j).
    WHY: a setup change (camber, ride height) moved an axle's level by 3-5 %
    between files of one car and track, often the two axles in opposite
    directions, while the same setup agreed within 1 % across sessions.
    Returns (trimmed car, [{trim, t, w, n} front, rear])."""
    ratios = [[], []]
    for s in red.laps:
        gxt = s.gx + s.grade + car.drag(s.v)
        ay = np.abs(s.gy)
        base = (np.abs(gxt) < 0.2) & (ay > 0.45 * np.percentile(ay, 99)) & (s.v > 15) & (np.abs(s.yawacc) < 0.4)
        A = car.axles(s.v, s.gx, s.gy, s.gz)
        for j, idx in enumerate((FRONT, REAR)):
            idx = list(idx)
            ax = axles[j]
            if ax.saturation == "grip_fraction" and s.gf is not None:
                sat = s.gf[idx].mean(0) < GF_SATURATED
            elif ax.knee_deg is not None and s.slip is not None:
                sl = s.slip[idx].mean(0)
                sat = (sl >= ax.knee_deg) & (sl <= ax.knee_deg + KNEE_SPAN_DEG)
            else:
                continue
            m = base & sat
            ratios[j].append(np.log(ay[m] / A[j][m]))
    mu1, out = list(car.mu1), []
    for j in (0, 1):
        x = np.concatenate(ratios[j]) if ratios[j] else np.zeros(0)
        n = int(len(x))
        if n < TRIM_MIN_N:
            out.append({"trim": "none", "t": None, "w": 0.0, "n": n})
            continue
        t = float(np.median(x))
        how = "applied"
        if abs(t) > TRIM_CAP:
            t, how = float(np.sign(t) * TRIM_CAP), "capped"
        w = n / (n + TRIM_HALF_N)
        mu1[j] = mu1[j] * float(np.exp(w * t))
        out.append({"trim": how, "t": t, "w": w, "n": n})
    d = car.to_dict()
    d["mu1"] = mu1
    return CarModel.from_dict(d), out


def fit_braking(d: dict, gxt: np.ndarray, Af, Ar, s_front: float, gf: np.ndarray | None,
                q: float = 0.5) -> dict:
    """Design §4.4: |gxt| <= kb (s_f sqrt(Af^2-gy^2) + s_r sqrt(Ar^2-gy^2)).
    Limit samples, first rule with >= 100; none: kb 1.0 `assumed`."""
    gy, brk = d["gy"], d["brk"]
    S = s_front * np.sqrt(np.maximum(Af ** 2 - gy ** 2, 0)) + (1 - s_front) * np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 0))
    straight = np.abs(gy) < 0.3
    at_lim = d["abs"] > 0.5
    first = "|gy| < 0.3, brake > 40 %, ABS"
    if gf is not None:
        at_lim = at_lim | (gf[:2].mean(0) < GF_SATURATED)
        first = "|gy| < 0.3, brake > 40 %, front Grip Fraction < 0.3 or ABS"
    rules = ((first, straight & (brk > 40) & at_lim),
             ("|gy| < 0.3, brake > 90 %", straight & (brk > 90)),
             ("brake > 85 %", brk > 85))
    for name, lim in rules:
        lim = lim & (S > 0.05)                         # S ~ 0 only past the lateral limit
        if lim.sum() >= 100:
            return dict(value=float(np.quantile(-gxt[lim] / S[lim], q)), rule=name, n=int(lim.sum()),
                        status="fitted")
    return dict(value=1.0, rule="none", n=0, status="assumed")


def fit_traction(d: dict, gxt: np.ndarray, Ar, s_front: float, q: float = 0.5) -> dict:
    """Design §4.5: gxt <= kt s_r sqrt(Ar^2 - gy^2) (rear drive). Fewer than 30
    limit samples: kt 1.0 `assumed`."""
    v, gy, thr = d["v"], d["gy"], d["thr"]
    room = np.sqrt(np.maximum(Ar ** 2 - gy ** 2, 1e-4))
    rest = (gxt > 0.2) & (np.abs(gy) < 0.8 * Ar)
    trl, rule = (d["tc"] > 0.5) & (thr > 30) & (v < 45) & rest, "TC active"
    if trl.sum() < 100:
        trl, rule = (thr > 95) & (v < 40) & rest, "full throttle below 144 km/h"
    if trl.sum() >= 30:
        return dict(value=float(np.quantile(gxt[trl] / ((1 - s_front) * room[trl]), q)), rule=rule,
                    n=int(trl.sum()), status="fitted")
    return dict(value=1.0, rule=rule, n=int(trl.sum()), status="assumed")


def fit_engine(d: dict, q: float = 0.5) -> dict:
    """Design §4.7: full-throttle flat-road (kinematic) acceleration per 2.5 m/s
    bin, smoothed and extended (flat below, last slope above, never rising),
    and the cornering cost cc: power(v, gy) = curve(v) - cc gy^2."""
    v, gx, gy, gr = d["v"], d["gx"], d["gy"], d["grade"]
    ft = (d["thr"] > 98) & (d["tc"] < 0.05)
    if ft.sum() < 25:
        raise InsufficientData("no full-throttle samples for the engine curve")
    es = np.arange(max(8.0, np.floor(v[ft].min())), v[ft].max() + 2.5, 2.5)
    pv, pa = [], []
    for a, b in zip(es[:-1], es[1:]):
        m = ft & (np.abs(gy) < 0.25) & (v >= a) & (v < b)
        if m.sum() >= 25:
            pv.append((a + b) / 2); pa.append(float(np.quantile((gx + gr)[m], q)))
    if len(pv) < 2:
        raise InsufficientData("too few full-throttle speed bins for the engine curve")
    pv, pa = np.array(pv), np.array(pa)
    for _ in range(2):                                 # light smoothing, keeps the ends
        pa[1:-1] = 0.25 * pa[:-2] + 0.5 * pa[1:-1] + 0.25 * pa[2:]
    slope = (pa[-1] - pa[-4]) / (pv[-1] - pv[-4]) if len(pv) > 4 else 0.0
    pv = np.concatenate([[0.0], pv, [pv[-1] + 40]])
    pa = np.concatenate([[pa[0]], pa, [pa[-1] + 40 * min(slope, 0.0)]])
    mc = ft & (np.abs(gy) > 0.8) & (v > pv[1])
    cc = float(np.median((np.interp(v[mc], pv, pa) - (gx + gr)[mc]) / gy[mc] ** 2)) if mc.sum() > 100 else 0.0
    return dict(power_v=pv.tolist(), power_g=pa.tolist(), corner_drag=float(np.clip(cc, 0.0, 0.08)),
                bins=len(pv) - 2, n_corner=int(mc.sum()))


def fit_group(reductions: list, car_class: str, v3_peaks: dict | None = None,
              q: float = 0.5) -> GroupFit:
    """Design §4 on one fit group. `reductions`: ideal_state.FileReduction of the
    group's files (qualifying laps only). `v3_peaks`: tyre v3 peak slip per axle
    ({'front': deg, 'rear': deg}) for the knee cross-check only. `q`: test only."""
    reductions = [r for r in reductions if r.laps]
    if not reductions:
        raise InsufficientData("no qualifying laps")
    notes = []
    # The saturation check comes first: a native LMU file has neither slip
    # angles nor Grip Fraction (nor Susp Force), and "no slip angles" is the
    # reason the user can act on (design §6.1).
    if not all(r.has_slip or r.has_grip_fraction for r in reductions):
        raise InsufficientData("no slip angles and no Grip Fraction")

    # 4.1 load model per file; files with < 2 laps borrow the group's pooled
    # one. The pooled model ("__group__") is always kept: a reference file with
    # no qualifying lap of its own uses it too.
    srcs = {r.load_source for r in reductions}
    if None in srcs:
        raise InsufficientData("no wheel load channel (Tire Load or Susp Force)")
    pooled_src = "load" if srcs == {"load"} else "susp"
    pooled = fit_load_model([s for x in reductions for s in x.laps], pooled_src, scope="group")
    load_models = {"__group__": pooled}
    for r in reductions:
        if len(r.laps) >= 2:
            load_models[r.file] = fit_load_model(r.laps, r.load_source)
        else:
            load_models[r.file] = pooled
            notes.append(f"{r.file}: load model from the group ({len(r.laps)} lap)")

    d = _pool(reductions, load_models)
    has_force = all(r.has_forces for r in reductions)
    gf = d["gf"] if all(r.has_grip_fraction for r in reductions) else None
    drag = fit_drag(d, has_force, car_class)
    gxt = d["gx"] + d["grade"] + drag["d0"] + drag["d2"] * d["v"] ** 2
    share = fit_share(d, has_force, [lm.static for f, lm in load_models.items() if f != "__group__"])
    axles = fit_axles(d, gxt, d["slip"], gf, v3_peaks, q)
    mu1, kk, cc = [a.mu1 for a in axles], [a.k for a in axles], [a.c for a in axles]
    Af, Ar = axle_capacity(d["lam"], d["v"], mu1, kk, cc)
    kb = fit_braking(d, gxt, Af, Ar, share["value"], gf, q)
    kt = fit_traction(d, gxt, Ar, share["value"], q)
    eng = fit_engine(d, q)

    info = dict(has_forces=has_force, load_source=pooled_src,
                n_laps=sum(len(r.laps) for r in reductions), n_files=len(reductions),
                axles={AXLES[j]: axles[j].to_dict() for j in (0, 1)}, s_front=share,
                kappa_brake=kb, kappa_trac=kt, drag=drag,
                engine={k: eng[k] for k in ("bins", "corner_drag", "n_corner")}, q=q, notes=notes)
    return GroupFit(car_class=car_class, load_models=load_models, mu1=mu1, k=kk, c=cc,
                    s_front=share["value"], kb=kb["value"], kt=kt["value"], d0=drag["d0"], d2=drag["d2"],
                    power_v=eng["power_v"], power_g=eng["power_g"], corner_drag=eng["corner_drag"], info=info)

