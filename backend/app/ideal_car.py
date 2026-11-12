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

FRONT, REAR = (0, 1), (2, 3)
AXLES = ("front", "rear")
LAM_FLOOR = 0.02
K_MAX, C_MAX = 0.15, 0.45          # bounds of the load and speed terms of mu
OBSERVED_MIN = 250                 # saturated samples (5 s) for an axle to be `observed`
KNEE_BIN_DEG, KNEE_MIN_N, KNEE_FRAC, KNEE_SPAN_DEG = 0.5, 60, 0.97, 4.0
GF_SATURATED = 0.3                 # Grip Fraction is LOW when saturated
KNEE_V3_DIFFER_DEG = 1.5
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

