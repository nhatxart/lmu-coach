"""
Track topography (height, banking, kerb/grass/gravel paint) from the game's
own telemetry, for the 3D chase view.

LMU gives no track mesh, but every recorded tick says where the car is
(`Pos Y` = height of the car's reference point), how it is tilted (the `Ori`
matrix, decoded by vehicle_pose) and how far each wheel hangs above the
ground (`Ride Height`). Chassis corner height minus that wheel's ride height
is the ground under that tyre: four ground probes per tick, and the
left/right difference is the banking with the car's own suspension roll
removed. Probes are placed in (lap distance s, lateral offset lat) - the
game's centre-path frame - because that frame is shared by every lap and
every recording of a layout, so laps on different lines add up.

The fit is deliberately simple and local: one plane per 2 m row
(`h + bank*lat`), estimated from the probes of the five rows around it,
rows nobody drove interpolated, then smoothed along the lap. A plane per row
is all a racing line can measure (a car is 1.6 m wide and drives within a
few metres of the centre path); anything more would be invented terrain.

Conventions (docs/superpowers/plans/2026-10-25-chase-view.md):
`lat` positive = right of travel, `bank` = dh/dlat (+ = rises to the right),
`s` = the game's lap distance, periodic in `length_m`.

Surface paint needs no pose at all (`Surface Type` per wheel was recorded
before pose channels existed), so old recordings still give kerbs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .vehicle_pose import PoseBasis, basis_vectors, detect_pose_basis, ori_matrices

ROW_STEP_M = 2.0
PAINT_CELL_M = 0.5
PAINT_KINDS = {2, 3, 4, 5}       # grass, dirt, gravel, kerb (0 dry / 1 wet / 6 special: no paint)
MIN_SPEED_MS = 8.0

FIT_WINDOW_ROWS = 2              # rows r-2..r+2 feed row r's plane
MIN_FIT_WEIGHT = 3.0
MIN_LAT_STD_M = 0.5              # below this the probes cannot see a slope across the row
OUTLIER_M = 0.15                 # kerb strikes, airborne ticks
SMOOTH_H_ROWS = 1.5
SMOOTH_BANK_ROWS = 3.0
CENTRE_PROBE_WEIGHT = 0.5
OTHER_CAR_WEIGHT = 0.3
OTHER_CAR_REF_HEIGHT_M = 0.30    # cg allowance + a typical ride height
WHEELS = ("FL", "FR", "RL", "RR")

# detect_lateral_sign
EDGE_FRACTION = 0.6
MIN_SIGN_CONTACTS = 20
MIN_SIGN_RATIO = 2.0


@dataclass(frozen=True)
class WheelGeometry:
    """Assumed wheel positions relative to the car's reference point.
    Constant per car; the same error at both sides cancels in the banking."""
    half_track_m: float = 0.80
    front_m: float = 1.45
    rear_m: float = -1.45
    cg_height_m: float = 0.25


DEFAULT_GEOMETRY = WheelGeometry()


@dataclass
class ProbeSet:
    """Ground probes and surface-paint hits of one lap (or one car).

    s is the game lap distance in metres (may run past the lap length; the
    fit wraps it), lat right-positive, y absolute LMU ground height, w weight.
    """
    s: np.ndarray
    lat: np.ndarray
    y: np.ndarray
    w: np.ndarray
    paint_s: np.ndarray
    paint_lat: np.ndarray
    paint_kind: np.ndarray


def _empty() -> np.ndarray:
    return np.zeros(0, dtype=float)


def _probe_set(heights: list, paint: list) -> ProbeSet:
    def cat(parts, i, dtype=float):
        return (np.concatenate([p[i] for p in parts]).astype(dtype)
                if parts else np.zeros(0, dtype=dtype))
    return ProbeSet(cat(heights, 0), cat(heights, 1), cat(heights, 2), cat(heights, 3),
                    cat(paint, 0), cat(paint, 1), cat(paint, 2, int))


# ── Lateral sign ─────────────────────────────────────────────────────────────

def detect_lateral_sign(path_lat: np.ndarray, track_edge: np.ndarray,
                        surface: dict[str, np.ndarray]) -> tuple[int, float, str]:
    """Is `Path Lateral` positive to the right? (sign, confidence, evidence).

    Near an edge, the wheels on that side are the ones that touch kerbs and
    grass. So with positive = right, contacts at positive lat are FR/RR and at
    negative lat FL/RL. Counted are paint-kind contacts (PAINT_KINDS) on
    samples with |path_lat| > 0.6 |track_edge|. Too few contacts or no clear
    winner -> the documented default, with confidence 0.
    """
    pl = np.asarray(path_lat, dtype=float)
    te = np.asarray(track_edge, dtype=float)
    near = np.isfinite(pl) & np.isfinite(te) & (np.abs(pl) > EDGE_FRACTION * np.abs(te))

    def hits(w):
        a = surface.get(w)
        if a is None or len(a) != len(pl):
            return np.zeros(len(pl), dtype=bool)
        a = np.asarray(a, dtype=float)
        return np.isin(np.round(np.nan_to_num(a, nan=0.0)).astype(int), list(PAINT_KINDS))

    right = hits("FR").astype(int) + hits("RR")
    left = hits("FL").astype(int) + hits("RL")
    pos, neg = near & (pl > 0), near & (pl < 0)
    agree = int(right[pos].sum() + left[neg].sum())      # fits positive = right
    disagree = int(left[pos].sum() + right[neg].sum())   # fits positive = left
    total = agree + disagree
    evidence = (f"{total} edge contacts: {agree} on the wheels positive=right predicts, "
                f"{disagree} on the other side")
    dom, minor = max(agree, disagree), min(agree, disagree)
    if total < MIN_SIGN_CONTACTS or dom < MIN_SIGN_RATIO * minor:
        return 1, 0.0, f"inconclusive: default positive=right ({evidence})"
    sign = 1 if agree > disagree else -1
    return sign, round((dom - minor) / total, 3), evidence


# ── Probes ───────────────────────────────────────────────────────────────────

def lap_pose_basis(lap) -> PoseBasis | None:
    """Detect the orientation-matrix layout from one lap (None: no usable pose)."""
    ch = lap.channels
    M = ori_matrices(ch)
    px, pz = ch.get("Pos X"), ch.get("Pos Z")
    if M is None or px is None or pz is None or len(lap.time_s) < 3:
        return None
    t = np.asarray(lap.time_s, dtype=float)
    if len(px) != len(t) or len(M) != len(t):
        return None
    vx, vz = np.gradient(np.asarray(px, float), t), np.gradient(np.asarray(pz, float), t)
    gs = ch.get("Ground Speed")
    speed = np.asarray(gs, float) / 3.6 if gs is not None else np.hypot(vx, vz)
    return detect_pose_basis(M, vx, vz, speed)


def lap_probes(lap, lap_dist_offset: float = 0.0, lateral_sign: int = 1,
               geom: WheelGeometry = DEFAULT_GEOMETRY,
               basis: PoseBasis | None = None) -> ProbeSet:
    """Ground probes and paint hits of one lap on its own grid.

    Four wheel probes (weight 1) need Pos Y, a detected basis and Ride Height;
    with Pos Y alone one centre probe (weight 0.5). Paint needs only Surface
    Type and is collected either way.
    """
    ch = lap.channels
    ld = ch.get("Lap Dist")
    if ld is None or len(ld) == 0:
        return _probe_set([], [])
    ld = np.asarray(ld, dtype=float)
    n = len(ld)
    gs = ch.get("Ground Speed")
    speed = np.asarray(gs, float) / 3.6 if gs is not None else np.full(n, np.inf)
    pl = ch.get("Path Lateral")
    path_lat = np.asarray(pl, float) if pl is not None else np.zeros(n)
    # Lap Dist must advance while the car moves. telemetry's unwrap holds the
    # last 10 Hz value flat across the start/finish wrap (the final ~0.1 s of
    # every lap reads up to one 10 Hz step short), and pit/reset glitches can
    # run it backwards; probes there would land metres from where they belong.
    advancing = np.diff(ld, prepend=-np.inf) > 0
    keep = np.isfinite(ld) & np.isfinite(path_lat) & (speed > MIN_SPEED_MS) & advancing
    s_ref = ld + float(lap_dist_offset)
    lat_ref = lateral_sign * path_lat

    wheels = []   # (label, along, lat offset, left-vector factor)
    for w in WHEELS:
        along = geom.front_m if w[0] == "F" else geom.rear_m
        side = -1.0 if w[1] == "L" else 1.0          # lat offset direction (right +)
        wheels.append((w, along, side * geom.half_track_m, -side * geom.half_track_m))

    heights, paint = [], []
    py = ch.get("Pos Y")
    rides = [ch.get(f"Ride Height {w}") for w in WHEELS]
    have_ride = all(r is not None and len(r) == n for r in rides)
    M = ori_matrices(ch) if basis is not None else None
    if py is not None and len(py) == n:
        py = np.asarray(py, dtype=float)
        if M is not None and len(M) == n and have_ride:
            fwd, left, _ = basis_vectors(M, basis)
            for (w, along, dlat, lfac), rh in zip(wheels, rides):
                y = py + fwd[:, 1] * along + left[:, 1] * lfac - np.asarray(rh, float) \
                    - geom.cg_height_m
                m = keep & np.isfinite(y)
                heights.append((s_ref[m] + along, lat_ref[m] + dlat, y[m], np.ones(int(m.sum()))))
        else:
            rh = (np.nanmean(np.stack([np.asarray(r, float) for r in rides]), axis=0)
                  if have_ride else np.zeros(n))
            y = py - np.nan_to_num(rh) - geom.cg_height_m
            m = keep & np.isfinite(y)
            heights.append((s_ref[m], lat_ref[m], y[m],
                            np.full(int(m.sum()), CENTRE_PROBE_WEIGHT)))

    for w, along, dlat, _ in wheels:
        st = ch.get(f"Surface Type {w}")
        if st is None or len(st) != n:
            continue
        kind = np.round(np.nan_to_num(np.asarray(st, float), nan=0.0)).astype(int)
        m = keep & np.isin(kind, list(PAINT_KINDS))
        paint.append((s_ref[m] + along, lat_ref[m] + dlat, kind[m]))
    return _probe_set(heights, paint)


def other_car_probes(rows: np.ndarray, length_m: float, lateral_sign: int = 1) -> ProbeSet:
    """Centre probes from the `Other Cars` table (ts, slot, pos_x, pos_y,
    pos_z, lap_dist, path_lateral, in_pits). Scoring positions at 10 Hz with
    no pose or ride height, unverified in game: low weight, no paint."""
    r = np.asarray(rows, dtype=float)
    if r.ndim != 2 or r.shape[0] == 0 or r.shape[1] < 8:
        return _probe_set([], [])
    m = (r[:, 7] == 0) & np.isfinite(r[:, [3, 5, 6]]).all(axis=1)
    r = r[m]
    s = np.mod(r[:, 5], float(length_m))
    return _probe_set([(s, lateral_sign * r[:, 6], r[:, 3] - OTHER_CAR_REF_HEIGHT_M,
                        np.full(len(r), OTHER_CAR_WEIGHT))], [])


# ── Fit ──────────────────────────────────────────────────────────────────────

def _plane(lat, y, w, ds):
    """Weighted y = h + bank*lat + grade*ds, ds = distance from the row centre.

    Returns (h, bank, m, residuals): h is the height at the row centre and
    lat 0; bank NaN (and h the height at the probes' mean lat m instead)
    when the probes span too little lateral range. None: too little weight.

    The grade term is not in the plan's "y = h + bank*lat": without it the
    five-row (10 m) window sees the hill as scatter - on the synthetic ring's
    4 % grade ~7 % of perfectly clean probes landed beyond OUTLIER_M and were
    dropped, and wherever one lap's samples sat a little further along the
    window than another lap's, the grade leaked into the bank. On a real 10-18 %
    climb (Eau Rouge) the outlier cut would reject most of a row.
    """
    W = float(w.sum())
    if W < MIN_FIT_WEIGHT:
        return None
    m = float((w * lat).sum() / W)
    cols = [np.ones_like(lat)]
    has_bank = math.sqrt(float((w * (lat - m) ** 2).sum() / W)) >= MIN_LAT_STD_M
    if has_bank:
        cols.append(lat - m)
    dm = float((w * ds).sum() / W)
    if math.sqrt(float((w * (ds - dm) ** 2).sum() / W)) >= MIN_LAT_STD_M:
        cols.append(ds)
    A = np.stack(cols, axis=1)
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    res = y - A @ coef
    if has_bank:
        b = float(coef[1])
        return float(coef[0]) - b * m, b, m, res
    return float(coef[0]), math.nan, m, res


def _row_centres(length_m: float, rows: int, step: float) -> np.ndarray:
    c = (np.arange(rows) + 0.5) * step
    c[-1] = (step * (rows - 1) + length_m) / 2.0   # the last row may be short
    return c


def _periodic_fill(v: np.ndarray, rows: int) -> np.ndarray:
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros(rows)
    x = np.arange(rows, dtype=float)
    return np.interp(x, x[ok], v[ok], period=rows)


def build_layout_topography(probes: list[ProbeSet], length_m: float) -> dict | None:
    """Fit one layout's grid from every probe set of that layout (see the
    module docstring and the plan, Task 3, for the algorithm). None when
    there are neither height probes nor paint."""
    L = float(length_m)
    step = ROW_STEP_M
    N = int(math.ceil(L / step))
    def cat(attr):
        return (np.concatenate([np.asarray(getattr(p, attr), float) for p in probes])
                if probes else _empty())

    s, lat, y, w = cat("s"), cat("lat"), cat("y"), cat("w")
    ok = np.isfinite(s) & np.isfinite(lat) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    s, lat, y, w = s[ok], lat[ok], y[ok], w[ok]
    ps, plat, pkind = cat("paint_s"), cat("paint_lat"), cat("paint_kind")
    pok = np.isfinite(ps) & np.isfinite(plat)
    ps, plat, pkind = ps[pok], plat[pok], pkind[pok].astype(int)
    if len(s) == 0 and len(ps) == 0:
        return None

    s = np.mod(s, L)
    row = np.minimum((s // step).astype(int), N - 1)
    centres = _row_centres(L, N, step)
    order = np.argsort(row, kind="stable")
    starts = np.searchsorted(row[order], np.arange(N + 1))
    h_raw = np.full(N, np.nan)
    b_raw = np.full(N, np.nan)
    m_raw = np.zeros(N)
    for r in range(N):
        qs = sorted({(r + k) % N for k in range(-FIT_WINDOW_ROWS, FIT_WINDOW_ROWS + 1)})
        idx = np.concatenate([order[starts[q]:starts[q + 1]] for q in qs])
        if len(idx) == 0:
            continue
        # signed distance from this row's centre, across the start/finish wrap
        ds = np.mod(s[idx] - centres[r] + L / 2, L) - L / 2
        fit = _plane(lat[idx], y[idx], w[idx], ds)
        if fit is None:
            continue
        good = np.abs(fit[3]) <= OUTLIER_M
        if not good.all():
            refit = _plane(lat[idx][good], y[idx][good], w[idx][good], ds[good])
            fit = refit if refit is not None else fit
        h_raw[r], b_raw[r], m_raw[r] = fit[0], fit[1], fit[2]

    has_heights = bool(np.isfinite(h_raw).any())
    bank = _periodic_fill(b_raw, N)
    # rows fitted without a bank of their own: h was their mean at lat m;
    # move it to lat 0 with the neighbouring bank
    flat_rows = np.isfinite(h_raw) & ~np.isfinite(b_raw)
    h_raw[flat_rows] -= bank[flat_rows] * m_raw[flat_rows]
    h = _periodic_fill(h_raw, N)
    if has_heights:
        h = gaussian_filter1d(h, SMOOTH_H_ROWS, mode="wrap")
        bank = gaussian_filter1d(bank, SMOOTH_BANK_ROWS, mode="wrap")
    else:
        h, bank = np.zeros(N), np.zeros(N)

    count = np.bincount(row, minlength=N)
    lat_min = np.full(N, np.inf)
    lat_max = np.full(N, -np.inf)
    np.minimum.at(lat_min, row, lat)
    np.maximum.at(lat_max, row, lat)
    lat_min[count == 0] = 0.0
    lat_max[count == 0] = 0.0

    prow = np.minimum((np.mod(ps, L) // step).astype(int), N - 1)
    pcell = np.round(plat / PAINT_CELL_M).astype(int)
    cells: dict[tuple[int, int], Counter] = {}
    for key, k in zip(zip(prow.tolist(), pcell.tolist()), pkind.tolist()):
        cells.setdefault(key, Counter())[k] += 1
    paint = []
    for (r, c), cnt in sorted(cells.items()):
        if sum(cnt.values()) >= 2:
            kind = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            paint.append([int(r), int(c), int(kind)])

    rnd = lambda a: [round(float(v), 3) for v in a]
    return {
        "length_m": round(L, 3),
        "row_step_m": step,
        "rows": N,
        "h": rnd(h),
        "bank": rnd(bank),
        "n": [int(v) for v in count],
        "lat_min": rnd(lat_min),
        "lat_max": rnd(lat_max),
        "paint_cell_m": PAINT_CELL_M,
        "paint": paint,
        "has_heights": has_heights,
        "coverage": round(float(np.mean(count > 0)), 3),
    }


# ── Lookups ──────────────────────────────────────────────────────────────────

def _interp_rows(layout: dict, key: str, s: np.ndarray) -> np.ndarray:
    L = float(layout["length_m"])
    c = _row_centres(L, int(layout["rows"]), float(layout["row_step_m"]))
    return np.interp(np.mod(np.asarray(s, dtype=float), L), c,
                     np.asarray(layout[key], dtype=float), period=L)


def lookup_height(layout: dict, s: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Ground height h(s) + bank(s)*lat, periodic in s."""
    return _interp_rows(layout, "h", s) + _interp_rows(layout, "bank", s) * np.asarray(lat, float)


def grade_bank(layout: dict, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dh/ds, bank) at s. The grade is a central difference over one row
    each side of s on the interpolated height."""
    s = np.asarray(s, dtype=float)
    d = float(layout["row_step_m"])
    grade = (_interp_rows(layout, "h", s + d) - _interp_rows(layout, "h", s - d)) / (2 * d)
    return grade, _interp_rows(layout, "bank", s)


def paint_at(layout: dict, s: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Paint kind of the cell under (s, lat); 0 where none."""
    L = float(layout["length_m"])
    step = float(layout["row_step_m"])
    cell_m = float(layout["paint_cell_m"])
    table = {(int(r), int(c)): int(k) for r, c, k in layout.get("paint", [])}
    s = np.asarray(s, dtype=float)
    rows = np.minimum((np.mod(s, L) // step).astype(int), int(layout["rows"]) - 1)
    cells = np.round(np.asarray(lat, dtype=float) / cell_m).astype(int)
    return np.array([table.get((r, c), 0) for r, c in zip(rows.tolist(), cells.tolist())],
                    dtype=int)
