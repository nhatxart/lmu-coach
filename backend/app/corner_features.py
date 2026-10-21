"""
Per-corner driving features for the coach (race-engineer-coach skill).

The lap-level grip report says WHERE the car hit a limit (ABS, TC, understeer
candidates). It cannot say WHY, because the usual causes sit in the driver's
inputs around the corner: when the brake was released relative to turn-in, how
much of the rotation was done before the throttle came on, whether the brake was
re-applied, and so on. This module segments the lap into corners and computes
those features (the list in coach_skill/references/diagnostics.md), plus a short
list of evidence-backed flags per corner and the habits that repeat across
corners of the same type.

Everything here runs on the lap's 20 Hz analysis grid, so timings are good to
about +-0.05 s; the output says so. Thresholds marked (EN) are engineering
heuristics from the skill: tune them per car against the tyre model and the
driver's best laps. Nothing is guessed: a feature that cannot be computed from
the channels in the file is left out (None is stripped), and the coach prompt
treats a missing feature as "unknown".
"""

from __future__ import annotations

import numpy as np

from .telemetry import LapTelemetry

# --- corner segmentation ---------------------------------------------------
CORNER_MIN_LAT_G = 0.5        # (EN) |lateral G| that counts as cornering
CORNER_MIN_SPEED_KMH = 30.0
CORNER_MERGE_GAP_S = 0.4      # same-direction runs closer than this are one corner
CORNER_MIN_DURATION_S = 0.6
CORNER_MIN_HEADING_DEG = 15.0  # drop kinks that barely turn the car
APEX_MIN_PROMINENCE_KMH = 12.0  # (EN) speed rise between two apexes of one run
APEX_MIN_SEPARATION_S = 1.5
MAX_CORNERS_IN_REPORT = 20

# --- pedals ---------------------------------------------------------------
BRAKE_ON_PCT = 5.0
BRAKE_ZERO_PCT = 3.0
BRAKE_MERGE_GAP_S = 0.3
BRAKE_LOOKBACK_S = 12.0
REAPPLY_DROP_PCT = 8.0        # (EN) fall then rise by this much = re-application
REAPPLY_RISE_PCT = 8.0
THROTTLE_PICKUP_PCT = 10.0
THROTTLE_FULL_PCT = 90.0
THROTTLE_LIFT_PCT = 10.0

# --- flag thresholds (EN) ---------------------------------------------------
UNDER_ROTATED_SHARE = 0.45
SMEARED_MRP_S = 0.3
YAW_PLATEAU_S = 0.3
UNDERSTEER_ONSET_S = 0.15
EARLY_UNWIND_PCT = 20.0
TC_EXIT_S = 0.15
RELEASE_GAP_S = 0.15
RELEASE_GAP_DIP_PCT = 15.0
HEAVY_TRAIL_BRAKE_PCT = 12.0
COAST_S = 0.4


def _pct(a: np.ndarray) -> np.ndarray:
    """Pedals are 0-100 in recorder files; treat a 0-1 channel as a fraction."""
    a = np.nan_to_num(np.asarray(a, dtype=float))
    return a * 100.0 if a.size and np.max(np.abs(a)) <= 1.5 else a


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of contiguous True runs."""
    if not mask.size:
        return []
    m = np.concatenate(([False], mask.astype(bool), [False]))
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _smooth(x: np.ndarray, n: int = 3) -> np.ndarray:
    if n <= 1 or x.size < n:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def _count_reapplications(vals: np.ndarray, drop: float, rise: float) -> int:
    """Falls of >= drop followed by a rise of >= rise, starting from the first
    peak of vals. Used for the brake (re-application) and the throttle (lift)."""
    if not vals.size:
        return 0
    hi = lo = float(vals[0])
    count = 0
    for b in vals[1:]:
        if b > hi and lo == hi:
            hi = lo = float(b)           # still climbing to the first peak
        elif b < lo:
            lo = float(b)
        elif b - lo >= rise and hi - lo >= drop:
            count += 1
            hi = lo = float(b)
    return count


class _Sig:
    """The lap's channels as plain arrays on the 20 Hz grid."""

    def __init__(self, lap: LapTelemetry):
        ch = lap.channels
        n = len(lap.time_s)
        zeros = np.zeros(n)
        self.t = np.asarray(lap.time_s, dtype=float)
        self.dt = float(np.median(np.diff(self.t))) if n > 1 else 0.05
        self.n = n
        self.dist = np.nan_to_num(ch.get("Lap Dist", zeros))
        self.speed = np.nan_to_num(ch.get("Ground Speed", zeros))
        self.lat_g = np.nan_to_num(ch.get("G Force Lat", zeros))
        self.long_g = np.nan_to_num(ch.get("G Force Long", zeros))
        self.steer = np.nan_to_num(ch.get("Steering Pos", zeros))
        self.brake = _pct(ch.get("Brake Pos", zeros))
        self.thr = _pct(ch.get("Throttle Pos", zeros))
        self.gear = np.nan_to_num(ch["Gear"]) if "Gear" in ch else None
        self.abs_on = np.nan_to_num(ch.get("ABS", zeros)) > 0.5
        self.tc_on = np.nan_to_num(ch.get("TC", zeros)) > 0.5
        self.comb_g = np.hypot(self.lat_g, self.long_g)

        speed_ms = np.maximum(self.speed / 3.6, 8.0)
        if "Yaw Rate" in ch:
            self.yaw = np.abs(np.nan_to_num(ch["Yaw Rate"]))
            self.yaw_source = "recorded"
        else:
            self.yaw = np.abs(self.lat_g * 9.81 / speed_ms)
            self.yaw_source = "lateral_g_estimate"
        self.yaw_s = _smooth(self.yaw, 3)

        def axle(a: str, b: str):
            if f"Slip Angle {a}" in ch and f"Slip Angle {b}" in ch:
                m = (np.nan_to_num(ch[f"Slip Angle {a}"]) + np.nan_to_num(ch[f"Slip Angle {b}"])) / 2
                return np.abs(np.degrees(_smooth(m, 3)))
            return None
        self.front_slip = axle("FL", "FR")
        self.rear_slip = axle("RL", "RR")

        # which way the steering channel turns for positive lateral G
        if np.std(self.steer) > 1e-6 and np.std(self.lat_g) > 1e-6:
            c = np.corrcoef(self.steer, self.lat_g)[0, 1]
            self.steer_orient = 1.0 if (np.isnan(c) or c >= 0) else -1.0
        else:
            self.steer_orient = 1.0

    def idx(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.dt)))


def _split_at_speed_minima(s: _Sig, regions: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """A same-direction lateral-G run can hold several apexes (sector-3 climb,
    Fuji T13-T16). Split it at the fastest point between two distinct speed
    minima so each apex is analysed on its own."""
    from scipy.signal import find_peaks
    out: list[tuple[int, int, float]] = []
    spd = _smooth(s.speed, 5)
    for a, b, sign in regions:
        if b - a < 2 * s.idx(APEX_MIN_SEPARATION_S):
            out.append((a, b, sign))
            continue
        mins, _ = find_peaks(-spd[a:b + 1], prominence=APEX_MIN_PROMINENCE_KMH,
                             distance=s.idx(APEX_MIN_SEPARATION_S))
        if len(mins) < 2:
            out.append((a, b, sign))
            continue
        cuts = [a + int(np.argmax(spd[a + int(m0):a + int(m1) + 1])) + int(m0)
                for m0, m1 in zip(mins[:-1], mins[1:])]
        edges = [a] + cuts + [b + 1]
        for k in range(len(edges) - 1):
            out.append((edges[k], edges[k + 1] - 1 if k < len(edges) - 2 else b, sign))
    return out


def _detect_corners(s: _Sig) -> list[dict]:
    moving = (np.abs(s.lat_g) >= CORNER_MIN_LAT_G) & (s.speed >= CORNER_MIN_SPEED_KMH)
    merge_gap = s.idx(CORNER_MERGE_GAP_S)
    regions: list[tuple[int, int, float]] = []
    for sign in (1.0, -1.0):
        merged: list[list[int]] = []
        for a, b in _runs(moving & (np.sign(s.lat_g) == sign)):
            if merged and a - merged[-1][1] <= merge_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        regions += [(a, b, sign) for a, b in merged]
    regions.sort()
    out = []
    for a, b, sign in _split_at_speed_minima(s, regions):
        if (s.t[b] - s.t[a]) < CORNER_MIN_DURATION_S:
            continue
        heading = float(np.degrees(np.trapezoid(s.yaw[a:b + 1], s.t[a:b + 1])))
        if heading < CORNER_MIN_HEADING_DEG:
            continue
        out.append({"i0": a, "i1": b, "sign": sign, "heading_deg": heading})
    # a later region may overlap an earlier one after merging opposite signs
    clean: list[dict] = []
    for c in out:
        if clean and c["i0"] <= clean[-1]["i1"]:
            c["i0"] = clean[-1]["i1"] + 1
            if c["i0"] >= c["i1"]:
                continue
        clean.append(c)
    return clean


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def _corner_features(s: _Sig, c: dict, prev_i1: int, next_i0: int | None) -> dict:
    i0, i1 = c["i0"], c["i1"]
    t, dt = s.t, s.dt
    seg = slice(i0, i1 + 1)
    yaw_max = i0 + int(np.argmax(s.yaw_s[seg]))
    mrp = i0 + int(np.argmin(s.speed[seg]))
    # a flat-out sweep that only accelerates has its slowest sample on the edge
    # of the corner; the peak of yaw is the better anchor there
    edge = max(1, int(0.15 * (i1 - i0)))
    if mrp - i0 < edge or i1 - mrp < edge:
        mrp = yaw_max
    steer_abs = np.abs(s.steer)
    steer_peak = float(np.max(steer_abs[seg])) if i1 >= i0 else 0.0
    steer_max = i0 + int(np.argmax(steer_abs[seg]))

    # ---- braking zone leading into this corner --------------------------
    lookback = max(prev_i1, i0 - s.idx(BRAKE_LOOKBACK_S))
    gap = s.idx(BRAKE_MERGE_GAP_S)
    zone_runs: list[list[int]] = []
    for a, b in _runs(s.brake > BRAKE_ON_PCT):
        # the corner starts when lateral G passes a threshold, a little after the
        # wheel moves, so a brake zone that ended up to 3 s earlier still belongs to it
        if b < i0 - s.idx(3.0) or a > mrp or a < lookback:
            continue
        if zone_runs and a - zone_runs[-1][1] <= gap:
            zone_runs[-1][1] = b
        else:
            zone_runs.append([a, b])
    zone = max(zone_runs, key=lambda r: float(np.max(s.brake[r[0]:r[1] + 1]))) if zone_runs else None

    brake_f: dict = {}
    z0 = zb_end = rel0 = peak_i = None
    if zone:
        z0, zb_end = zone
        zb = s.brake[z0:zb_end + 1]
        peak = float(np.max(zb))
        peak_i = z0 + int(np.argmax(zb))
        first90 = z0 + int(np.argmax(zb >= 0.9 * peak))
        after = np.flatnonzero(s.brake[peak_i:zb_end + 1] >= 0.85 * peak)
        rel0 = peak_i + int(after[-1])
        zero_after = np.flatnonzero(s.brake[rel0:] < BRAKE_ZERO_PCT)
        rel_zero = rel0 + int(zero_after[0]) if zero_after.size else zb_end
        abs_mask = s.abs_on[z0:zb_end + 1]
        brake_f = {
            "peak_pct": _r(peak, 0),
            "point_dist_m": _r(s.dist[z0], 0),
            "time_to_peak_s": _r(t[first90] - t[z0]),
            "release_s": _r(t[rel_zero] - t[rel0]),
            "reapplications": _count_reapplications(
                s.brake[peak_i:zb_end + 1], REAPPLY_DROP_PCT, REAPPLY_RISE_PCT),
            "abs_s": _r(abs_mask.sum() * dt) if abs_mask.any() else 0.0,
            "abs_max_steer_pct": _r(np.max(steer_abs[z0:zb_end + 1][abs_mask]), 0) if abs_mask.any() else None,
            "at_min_speed_pct": _r(s.brake[mrp], 0),
        }

    # ---- turn-in ---------------------------------------------------------
    lb = max(prev_i1, z0 if z0 is not None else i0 - s.idx(2.0))
    turn_candidates = np.flatnonzero(steer_abs[lb:steer_max + 1] >= 0.15 * steer_peak) if steer_peak > 0 else []
    steer_start = lb + int(turn_candidates[0]) if len(turn_candidates) else i0
    half = np.flatnonzero(steer_abs[steer_start:steer_max + 1] >= 0.5 * steer_peak)
    steer_half = steer_start + int(half[0]) if len(half) else steer_max

    # ---- throttle pickup -------------------------------------------------
    # The COMMITTED pickup is where the final rise to full throttle starts. Part
    # throttle held earlier ("maintenance throttle") is reported separately as
    # part_throttle_before_pickup_s: counting its first touch as the pickup made
    # every compound corner look like it was rotating on power.
    pick_from = max(zb_end if zb_end is not None else i0, steer_start)
    limit = min(s.n - 1, i1 + s.idx(3.0))
    thr_s = _smooth(s.thr, 3)
    first_full = None
    full_hits = np.flatnonzero(thr_s[max(mrp, pick_from):limit + 1] >= THROTTLE_FULL_PCT)
    if len(full_hits):
        first_full = max(mrp, pick_from) + int(full_hits[0])
    pickup = None
    if first_full is not None and thr_s[pick_from] < THROTTLE_FULL_PCT:
        k = first_full
        while k > pick_from and thr_s[k - 1] < thr_s[k] - 0.2:   # walk back to where the rise began
            k -= 1
        pickup = k
    elif first_full is None:
        pk = np.flatnonzero(s.thr[pick_from:limit + 1] >= THROTTLE_PICKUP_PCT)
        pickup = pick_from + int(pk[0]) if len(pk) else None
    # throttle already (nearly) full at turn-in: a flat-out sweep, no pickup to judge
    flat_out = bool(s.thr[steer_start] >= THROTTLE_FULL_PCT)
    if flat_out:
        pickup = None

    # ---- rotation --------------------------------------------------------
    total_rot = float(np.trapezoid(s.yaw[seg], t[seg])) or 1e-9

    def share_to(idx: int) -> float:
        a, b = i0, min(max(idx, i0), i1)
        return float(np.trapezoid(s.yaw[a:b + 1], t[a:b + 1])) / total_rot if b > a else 0.0

    # alignment of the three physical markers of the apex (slowest, most yaw, most
    # steering); the throttle pickup is judged separately below
    marks = [t[i0 + int(np.argmin(s.speed[seg]))], t[yaw_max], t[steer_max]]
    plateau = (s.yaw_s[seg] >= 0.95 * s.yaw_s[yaw_max]) & (s.thr[seg] >= 5) & (s.thr[seg] <= 60)

    a, b = steer_start, (pickup if pickup is not None else i1)
    if b - a >= 3:
        d_steer = np.gradient(_smooth(steer_abs, 5), t)[a:b + 1]
        d_yaw = np.gradient(_smooth(s.yaw, 5), t)[a:b + 1]
        under = (d_steer > 2.0) & (d_yaw <= 0.0) & (s.speed[a:b + 1] > 40)
        us_s = float(under.sum() * dt)
    else:
        us_s = 0.0

    expected = c["sign"] * s.steer_orient
    counter = (s.steer[seg] * expected) < -3.0
    counter_events = [r for r in _runs(counter) if (r[1] - r[0] + 1) * dt >= 0.1]

    # ---- entry coupling --------------------------------------------------
    entry_f: dict = {}
    if rel0 is not None and steer_peak > 0:
        lag = t[steer_start] - t[rel0]
        zero_idx = np.flatnonzero(s.brake[rel0:] < BRAKE_ZERO_PCT)
        t_zero = t[rel0 + int(zero_idx[0])] if zero_idx.size else None
        dip = None
        # only meaningful when the brake really is (nearly) off at turn-in; with the
        # brake still on, G falls with speed and aero, which is not a grip gap
        if rel0 < steer_half and s.brake[steer_start] < 10.0:
            ref = min(float(np.max(s.comb_g[z0:rel0 + 1])), float(np.max(s.comb_g[steer_start:mrp + 1]))) if mrp > steer_start else None
            if ref and ref > 0.3:
                dip = max(0.0, 1.0 - float(np.min(s.comb_g[rel0:steer_half + 1])) / ref) * 100
        entry_f = {
            "brake_release_to_turn_in_s": _r(lag),
            "brake_zero_before_turn_in_s": _r(max(0.0, t[steer_start] - t_zero)) if t_zero is not None else None,
            "grip_dip_at_turn_in_pct": _r(dip, 0),
            "steer_ramp_s": _r(t[steer_max] - t[steer_start]),
            "brake_at_turn_in_pct": _r(s.brake[steer_start], 0),
        }
    downshifts_after = None
    last_ds_vs_turn_in = None
    if s.gear is not None:
        w0 = z0 if z0 is not None else max(prev_i1, i0 - s.idx(3.0))
        shifts = [k for k in range(max(w0, 1), mrp + 1) if s.gear[k] < s.gear[k - 1] and s.gear[k] >= 1]
        if shifts:
            last_ds_vs_turn_in = t[shifts[-1]] - t[steer_start]
            downshifts_after = sum(1 for k in shifts if k > steer_start)
        else:
            downshifts_after = 0
    fs = s.front_slip
    entry_f["downshifts_after_turn_in"] = downshifts_after
    entry_f["last_downshift_vs_turn_in_s"] = _r(last_ds_vs_turn_in)
    entry_f["front_slip_peak_deg"] = _r(np.max(fs[steer_start:mrp + 1]), 1) if fs is not None else None

    # ---- exit ------------------------------------------------------------
    exit_f: dict = {"throttle_pct_at_min_speed": _r(s.thr[mrp], 0)}
    coast_s = 0.0
    c0 = zb_end if zb_end is not None else steer_start
    c1 = pickup if pickup is not None else mrp
    if c1 > c0:
        coast_s = float(((s.brake[c0:c1 + 1] < BRAKE_ZERO_PCT) & (s.thr[c0:c1 + 1] < 5)).sum() * dt)
    exit_f["coast_before_throttle_s"] = _r(coast_s)
    tc_s = 0.0
    if pickup is not None:
        end_x = min(s.n - 1, i1 + s.idx(1.5))
        to90 = t[first_full] - t[pickup] if first_full is not None else None
        tc_s = float(s.tc_on[pickup:end_x + 1].sum() * dt)
        part = (s.thr[pick_from:pickup + 1] >= THROTTLE_PICKUP_PCT) & (s.thr[pick_from:pickup + 1] < THROTTLE_FULL_PCT)
        # lifts: throttle falls then rises again anywhere between turn-in and full power
        lift_to = first_full if first_full is not None else end_x

        def thr_at(sec: float) -> float:
            return float(s.thr[min(s.n - 1, pickup + s.idx(sec))])
        peak_st = steer_abs[steer_max]
        exit_f.update({
            "throttle_pickup_vs_min_speed_s": _r(t[pickup] - t[mrp]),
            "steer_unwound_at_throttle_pct": _r(max(0.0, (peak_st - steer_abs[pickup]) / peak_st * 100), 0) if pickup > steer_max and peak_st > 0 else 0.0,
            "throttle_pct_after_0_5s": _r(thr_at(0.5), 0),
            "throttle_pct_after_1s": _r(thr_at(1.0), 0),
            "time_to_90pct_throttle_s": _r(to90),
            "throttle_pct_at_pickup": _r(s.thr[pickup], 0),
            "part_throttle_before_pickup_s": _r(part.sum() * dt),
            "throttle_lifts": _count_reapplications(s.thr[pick_from:lift_to + 1], THROTTLE_LIFT_PCT, THROTTLE_LIFT_PCT),
        })
    if pickup is None and not flat_out and limit >= s.n - 1:
        exit_f["lap_ended_before_throttle_pickup"] = True   # corner sits at the lap boundary
        exit_f.pop("coast_before_throttle_s", None)
        coast_s = 0.0
    exit_f["tc_s"] = _r(tc_s)
    exit_f["rear_slip_peak_deg"] = _r(np.max(s.rear_slip[mrp:min(s.n, i1 + s.idx(1.0)) + 1]), 1) if s.rear_slip is not None else None

    rot_f = {
        "heading_change_deg": _r(c["heading_deg"], 0),
        "rotation_share_before_throttle": _r(share_to(pickup), 2) if pickup is not None else None,
        "rotation_share_at_min_speed": _r(share_to(mrp), 2),
        "min_speed_marks_spread_s": _r(max(marks) - min(marks)),
        "yaw_plateau_on_part_throttle_s": _r(plateau.sum() * dt),
        "understeer_onset_s": _r(us_s),
        "countersteer_events": len(counter_events),
    }

    # ---- flags (evidence is in the numbers above) ----------------------------
    flags: list[str] = []
    lag = entry_f.get("brake_release_to_turn_in_s")
    zero_gap = entry_f.get("brake_zero_before_turn_in_s")
    dip = entry_f.get("grip_dip_at_turn_in_pct")
    if lag is not None and lag > RELEASE_GAP_S and ((zero_gap or 0) >= RELEASE_GAP_S or (dip or 0) >= RELEASE_GAP_DIP_PCT):
        # brake off (or nearly) before the wheel moves: nothing carries the tyres
        # between braking and cornering. A brake still pressed at turn-in is trail
        # braking, not this.
        flags.append("brake_released_before_turning")
    if brake_f.get("reapplications", 0) >= 1:
        flags.append("brake_reapplied_after_release")
    if brake_f.get("abs_max_steer_pct") is not None and brake_f["abs_max_steer_pct"] >= 25:
        flags.append("abs_while_steering")
    share = rot_f["rotation_share_before_throttle"]
    if share is not None and share < UNDER_ROTATED_SHARE:
        # includes throttle that starts before the slowest point: that is power
        # asked for while the car still has most of its turning to do
        flags.append("under_rotated_before_throttle")
    if rot_f["min_speed_marks_spread_s"] > SMEARED_MRP_S:
        flags.append("min_speed_point_smeared")
    if rot_f["yaw_plateau_on_part_throttle_s"] >= YAW_PLATEAU_S:
        flags.append("waiting_on_part_throttle")
    if us_s >= UNDERSTEER_ONSET_S:
        flags.append("adding_steering_while_yaw_flat")
    if rot_f["countersteer_events"] >= 1:
        flags.append("countersteer_needed")
    if (brake_f.get("at_min_speed_pct") or 0) >= HEAVY_TRAIL_BRAKE_PCT:
        flags.append("heavy_brake_at_min_speed")
    if (downshifts_after or 0) >= 1:
        flags.append("downshift_after_turn_in")
    if coast_s >= COAST_S:
        flags.append("coasting_before_throttle")
    if (exit_f.get("steer_unwound_at_throttle_pct") or 0) >= EARLY_UNWIND_PCT:
        flags.append("steering_unwound_before_throttle")
    if tc_s >= TC_EXIT_S:
        flags.append("tc_on_exit")
    if exit_f.get("throttle_lifts", 0) >= 1:
        flags.append("throttle_lift_on_exit")

    v_entry = float(s.speed[z0]) if z0 is not None else float(s.speed[steer_start])
    return {
        "_mrp_i": mrp,
        "lap_dist_start_m": _r(s.dist[steer_start], 0),
        "lap_dist_end_m": _r(s.dist[i1], 0),
        "mrp_dist_m": _r(s.dist[mrp], 0),
        "speeds_kmh": {"entry": _r(v_entry, 0), "min": _r(s.speed[mrp], 0), "exit": _r(s.speed[min(s.n - 1, i1)], 0)},
        "braking": brake_f or None,
        "entry": entry_f,
        "rotation": rot_f,
        "exit": exit_f,
        "flags": flags,
        "_brake_peak": brake_f.get("peak_pct", 0) or 0,
    }


def _classify(c: dict, followed_by_opposite: bool) -> str:
    vmin = c["speeds_kmh"]["min"]
    peak = c["_brake_peak"]
    if followed_by_opposite:
        return "direction_change"
    if peak >= 50 and vmin < 110:
        return "hard_braking_slow"
    if peak < 20 and vmin >= 150:
        return "fast_lift_or_flat"
    return "medium"


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if v is not None and not k.startswith("_")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def compute_corner_analysis(lap: LapTelemetry) -> dict:
    """Corner-by-corner features for one lap, or {'available': False, ...}."""
    need = ("Ground Speed", "G Force Lat", "Steering Pos", "Brake Pos", "Throttle Pos", "Lap Dist")
    missing = [k for k in need if k not in lap.channels]
    if missing or len(lap.time_s) < 40:
        return {"available": False, "reason": f"channels missing: {', '.join(missing) or 'lap too short'}"}
    s = _Sig(lap)
    regions = _detect_corners(s)
    if not regions:
        return {"available": False, "reason": "no corners detected on this lap"}

    corners = []
    for k, c in enumerate(regions):
        prev_i1 = regions[k - 1]["i1"] if k else 0
        nxt = regions[k + 1]["i0"] if k + 1 < len(regions) else None
        f = _corner_features(s, c, prev_i1, nxt)
        opp = bool(nxt is not None and regions[k + 1]["sign"] != c["sign"] and (s.t[nxt] - s.t[c["i1"]]) < 1.2)
        f["type"] = _classify(f, opp)
        f["n"] = k + 1
        corners.append(f)

    habits = []
    flag_names = sorted({fl for f in corners for fl in f["flags"]})
    for fl in flag_names:
        hit = [f for f in corners if fl in f["flags"]]
        if len(hit) >= 2:
            by_type: dict[str, list[int]] = {}
            for f in hit:
                by_type.setdefault(f["type"], []).append(f["n"])
            habits.append({"flag": fl, "corners": [f["n"] for f in hit], "of": len(corners),
                           "by_type": by_type})
    habits.sort(key=lambda h: -len(h["corners"]))

    omitted = 0
    if len(corners) > MAX_CORNERS_IN_REPORT:
        ranked = sorted(corners, key=lambda f: (-len(f["flags"]), -f["rotation"]["heading_change_deg"]))
        keep = {f["n"] for f in ranked[:MAX_CORNERS_IN_REPORT]}
        omitted = len(corners) - len(keep)
        corners = [f for f in corners if f["n"] in keep]

    report = {
        "available": True,
        "grid_hz": round(1.0 / s.dt),
        "timing_resolution_s": round(s.dt, 3),
        "yaw_source": s.yaw_source,
        "corner_count": len(regions),
        "corners_omitted_for_length": omitted,
        "corners": [_strip(f) for f in corners],
        "repeated_habits": habits,
        "note": ("Turn-in = steering reaches 15% of its corner peak; min speed point = slowest "
                 "sample in the corner; throttle pickup = first throttle >= 10% after braking. "
                 "Features are measured from the inputs, not from tyre forces."),
    }
    return report
