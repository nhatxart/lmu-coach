"""
Car attitude (pitch / roll) from the recorder's raw orientation matrix.

The recorder stores `telem.mOri[r].x/y/z` as nine scalar tables
`Ori R0X .. Ori R2Z` (row r, component x/y/z). LMU does not document which
way round that 3x3 matrix is: its rows or its columns may hold the car's
forward / left / up axes, and "forward" may be local +z or -z. Guessing wrong
would silently give garbage pitch and roll, so the layout is detected from the
data: while the car drives fast, its forward axis must point where it moves.

World frame: LMU world, y up. Left of travel for a world velocity (dx, dz) is
(-dz, dx) (derived from the verified non-mirrored map frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORI_CHANNELS: tuple[str, ...] = tuple(
    f"Ori R{r}{c}" for r in range(3) for c in "XYZ"
)

MIN_FAST_SAMPLES = 50
FAST_SPEED_MS = 15.0
MIN_FWD_ALIGNMENT = 0.95
MIN_UP_MEAN_Y = 0.9
MIN_LEFT_ALIGNMENT = 0.9


@dataclass(frozen=True)
class PoseBasis:
    axis_kind: str                # "row" or "col": which slices of M are the car axes
    fwd: tuple[int, int]          # (index, sign)
    up: tuple[int, int]
    left: tuple[int, int]
    fwd_alignment: float          # mean cos(fwd, velocity) over fast samples
    up_mean_y: float


def ori_matrices(rows: dict[str, np.ndarray]) -> np.ndarray | None:
    """(n,3,3) with M[:, r, c] = rows["Ori R<r><XYZ[c]>"]; None if any array
    is missing, empty or of a different length than the others."""
    arrs = []
    for name in ORI_CHANNELS:
        a = rows.get(name)
        if a is None or len(a) == 0:
            return None
        arrs.append(np.asarray(a, dtype=float))
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        return None
    return np.stack(arrs, axis=1).reshape(n, 3, 3)


def _axis(M: np.ndarray, kind: str, i: int) -> np.ndarray:
    return M[:, i, :] if kind == "row" else M[:, :, i]


def _horizontal_unit(v: np.ndarray) -> np.ndarray:
    """(n,2) unit vectors of the (x,z) part; zeros where it vanishes."""
    h = v[:, [0, 2]]
    n = np.linalg.norm(h, axis=1, keepdims=True)
    return np.divide(h, n, out=np.zeros_like(h), where=n > 1e-9)


def detect_pose_basis(M: np.ndarray, vel_x: np.ndarray, vel_z: np.ndarray,
                      speed_ms: np.ndarray) -> PoseBasis | None:
    """Find which axes of M are forward / up / left in the world frame.

    Returns None (never a guess) when the data does not support one layout:
    too few fast samples, forward not tracking the velocity, up not pointing
    up, or left not on the left of travel.
    """
    M = np.asarray(M, dtype=float)
    vx, vz = np.asarray(vel_x, float), np.asarray(vel_z, float)
    sp = np.asarray(speed_ms, float)
    if M.ndim != 3 or M.shape[1:] != (3, 3) or not (len(M) == len(vx) == len(vz) == len(sp)):
        return None
    # BUG FIX / PREVIOUSLY: only speed was checked, so a sample with speed > 15
    # but world velocity (0, 0) (duplicate Pos ticks) divided by zero, every
    # mean became NaN and `nan < 0.95` is False -> a NaN basis was returned.
    # Fix: require a usable velocity direction and reject non-finite results.
    fast = ((sp > FAST_SPEED_MS) & np.isfinite(M).all(axis=(1, 2))
            & np.isfinite(vx) & np.isfinite(vz) & (np.hypot(vx, vz) > 1e-6))
    if int(fast.sum()) < MIN_FAST_SAMPLES:
        return None
    M = M[fast]
    vh = np.stack([vx[fast], vz[fast]], axis=1)
    vh = vh / np.linalg.norm(vh, axis=1, keepdims=True)
    left_dir = np.stack([-vh[:, 1], vh[:, 0]], axis=1)   # world left of travel

    best = None
    for kind in ("row", "col"):
        # forward: the axis (either sign) whose horizontal part follows the velocity
        cos = [float(np.mean(np.sum(_horizontal_unit(_axis(M, kind, i)) * vh, axis=1)))
               for i in range(3)]
        fi = int(np.argmax(np.abs(cos)))
        fsign = 1 if cos[fi] >= 0 else -1
        fwd_align = abs(cos[fi])
        rest = [i for i in range(3) if i != fi]
        # up: of the other two, the one pointing most along world +y
        ymean = [float(np.mean(_axis(M, kind, i)[:, 1])) for i in rest]
        k = int(np.argmax(np.abs(ymean)))
        ui, usign, up_y = rest[k], (1 if ymean[k] >= 0 else -1), abs(ymean[k])
        li = rest[1 - k]
        lal = float(np.mean(np.sum(_horizontal_unit(_axis(M, kind, li)) * left_dir, axis=1)))
        lsign = 1 if lal >= 0 else -1
        cand = PoseBasis(kind, (fi, fsign), (ui, usign), (li, lsign), fwd_align, up_y)
        if best is None or (cand.fwd_alignment, cand.up_mean_y) > (best[0].fwd_alignment, best[0].up_mean_y):
            best = (cand, abs(lal))

    cand, left_align = best
    # Handedness is not enforced: the left sign is chosen to match world-left,
    # which is all pitch/roll need (fwd_y, left_y); a mirrored matrix still
    # yields correct angles because fwd and left are each checked against motion.
    if not np.isfinite([cand.fwd_alignment, cand.up_mean_y, left_align]).all():
        return None
    if (cand.fwd_alignment < MIN_FWD_ALIGNMENT or cand.up_mean_y < MIN_UP_MEAN_Y
            or left_align < MIN_LEFT_ALIGNMENT):
        return None
    return cand


def basis_vectors(M: np.ndarray, basis: PoseBasis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fwd, left, up): unit world vectors per sample, each (n,3)."""
    M = np.asarray(M, dtype=float)

    def get(spec):
        idx, sign = spec
        v = sign * _axis(M, basis.axis_kind, idx)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)

    return get(basis.fwd), get(basis.left), get(basis.up)


def pitch_roll_deg(fwd: np.ndarray, left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pitch (+ = nose up) = asin(fwd_y); roll (+ = right side down, i.e. left
    side up) = asin(left_y)."""
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1.0, 1.0)))
    roll = np.degrees(np.arcsin(np.clip(left[:, 1], -1.0, 1.0)))
    return pitch, roll
