"""
Grip-limit analysis from a single lap's telemetry — no reference lap needed.

The core idea: LMU doesn't expose a raw tire slip-angle/slip-ratio trace, but
we can build solid proxies for "was the driver at/over the grip limit, and on
which axle" from what it does give us:

  1. ABS / TC activation events — ground truth from the game's own systems.
     ABS firing means the front axle locked under braking (exceeded braking
     grip). TC firing means the rear axle exceeded traction grip under power.
     These are the most reliable signals in the file because they come
     straight from LMU's physics, not a derived estimate.

  2. Wheel Speed vs Ground Speed delta — a genuine slip proxy. Convert wheel
     speed (m/s) to km/h and compare against Ground Speed. A wheel spinning
     faster than the car (positive delta under power) or locking (negative
     delta under braking) is measurably slipping relative to the road.

  3. Combined lateral+longitudinal G (friction circle magnitude) — tells us
     how close to the car's overall grip ceiling the driver is running at
     each point, and whether they're leaving grip on the table (low combined
     G in a corner with no obvious reason) or overdriving (G spikes right
     before an ABS/TC event, or a G Force Lat "plateau" while steering input
     keeps increasing — classic understeer signature).

None of this requires knowing the tire model's actual curve — we're reading
the car's behavior, which is what a driver needs anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .telemetry import LapTelemetry, WHEEL_LABELS
from .vehicle_dynamics import compute_slip_trace, summarize_slip_trace


@dataclass
class GripEvent:
    kind: str  # "abs_lockup" | "tc_intervention" | "understeer" | "oversteer_snap"
    start_time_s: float
    end_time_s: float
    lap_dist_start_m: float
    lap_dist_end_m: float
    peak_severity: float  # meaning depends on kind; see notes in build_grip_events
    notes: str


def _contiguous_true_regions(mask: np.ndarray, time_s: np.ndarray) -> list[tuple[int, int]]:
    """Returns [(start_idx, end_idx), ...] for contiguous True runs in mask."""
    regions = []
    in_region = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_region:
            start = i
            in_region = True
        elif not v and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, len(mask) - 1))
    return regions


def find_abs_lockups(lap: LapTelemetry, min_duration_s: float = 0.05) -> list[GripEvent]:
    if "ABS" not in lap.channels:
        return []
    mask = lap.channels["ABS"] > 0.5
    events = []
    for start_i, end_i in _contiguous_true_regions(mask, lap.time_s):
        duration = lap.time_s[end_i] - lap.time_s[start_i]
        if duration < min_duration_s:
            continue
        events.append(
            GripEvent(
                kind="abs_lockup",
                start_time_s=float(lap.time_s[start_i]),
                end_time_s=float(lap.time_s[end_i]),
                lap_dist_start_m=float(lap.channels["Lap Dist"][start_i]),
                lap_dist_end_m=float(lap.channels["Lap Dist"][end_i]),
                peak_severity=float(duration),
                notes=f"ABS active for {duration:.2f}s — front axle exceeded braking grip",
            )
        )
    return events


def find_tc_interventions(
    lap: LapTelemetry, min_duration_s: float = 0.05, merge_gap_s: float = 0.15
) -> list[GripEvent]:
    """
    TC in LMU's raw log flickers rapidly (on/off/on within tens of ms) as the
    system modulates power cuts — this is real behavior, not noise, but at
    our analysis grid (20Hz by default) individual flickers aren't
    resolvable and would either get merged or dropped depending on where
    grid points happen to land. Rather than report a possibly-wrong flicker
    count, we merge any TC-active regions separated by less than
    `merge_gap_s` into one intervention "zone" and report the zone's total
    active duration — that's the number that's actually meaningful for
    coaching (how much time was spent with the rear axle at/over the limit
    in this corner), not a jittery event count.
    """
    if "TC" not in lap.channels:
        return []
    mask = lap.channels["TC"] > 0.5
    raw_regions = _contiguous_true_regions(mask, lap.time_s)
    if not raw_regions:
        return []

    # merge regions that are close together in time
    merged: list[list[int]] = []
    for start_i, end_i in raw_regions:
        if merged and lap.time_s[start_i] - lap.time_s[merged[-1][1]] < merge_gap_s:
            merged[-1][1] = end_i
            merged[-1].append(end_i)  # track for active-time accounting below
        else:
            merged.append([start_i, end_i])

    events = []
    for region in merged:
        start_i, end_i = region[0], region[1]
        duration = lap.time_s[end_i] - lap.time_s[start_i]
        if duration < min_duration_s:
            continue
        # active_fraction: how much of this zone TC was actually cutting power,
        # vs just the zone's wall-clock span (which includes the merge gaps)
        zone_mask = mask[start_i : end_i + 1]
        active_fraction = float(np.mean(zone_mask)) if len(zone_mask) else 0.0
        events.append(
            GripEvent(
                kind="tc_intervention",
                start_time_s=float(lap.time_s[start_i]),
                end_time_s=float(lap.time_s[end_i]),
                lap_dist_start_m=float(lap.channels["Lap Dist"][start_i]),
                lap_dist_end_m=float(lap.channels["Lap Dist"][end_i]),
                peak_severity=active_fraction,
                notes=(
                    f"TC active {active_fraction*100:.0f}% of a {duration:.2f}s zone — "
                    "rear axle repeatedly at/over traction limit here"
                ),
            )
        )
    return events


def compute_wheel_slip(lap: LapTelemetry) -> dict[str, np.ndarray]:
    """
    Returns per-wheel slip ratio (unitless fraction, 0 = no slip, positive =
    spinning faster than car, negative = locking up).

    Two code paths depending on data source:

    RECORDER files (shared memory): use mLongitudinalGroundVel per wheel,
    stored as "Slip Angle {FL/FR/RL/RR}" — wait, that's slip angle. The
    longitudinal velocity is what we need. TinyPedal's slip_ratio() formula
    is: abs(w_rotation) * w_radius / v_speed - 1. Since w_radius =
    longitudinal_vel / w_rotation (rotation_radius() in TinyPedal), this
    simplifies to: abs(longitudinal_vel) / v_speed - 1. Radius cancels out
    entirely — no hardcoded tyre radius needed.

    But: recorder doesn't currently store longitudinal_vel separately from
    slip angle (we used it to compute slip angle but didn't store it raw).
    Fall back to: derive from Wheel Speed (mRotation in rad/s) using a
    per-car estimated tyre radius. This is less accurate but physically
    correct in magnitude.

    NATIVE LMU export files: Wheel Speed is stored in m/s (linear speed),
    so wheel_speed * 3.6 gives km/h directly. Simple delta against Ground
    Speed.

    We distinguish the two cases by checking the channelsList unit field
    (native export uses 'm/s' via LMU, recorder stores rad/s from mRotation).
    In practice we check whether Slip Angle channels exist — if they do,
    we're reading recorder output.
    """
    ground_ms = lap.channels["Ground Speed"] / 3.6
    ground_ms_safe = np.maximum(ground_ms, 0.5)
    slip = {}

    recorder_file = "Slip Angle FL" in lap.channels

    for label in WHEEL_LABELS:
        ws_key = f"Wheel Speed {label}"
        if ws_key not in lap.channels:
            continue

        if recorder_file:
            # mRotation in rad/s; derive linear speed via estimated radius.
            # Standard GT/hypercar tyre radius: ~0.33m (varies ~0.31-0.35m
            # depending on tyre size). Not exact but magnitude is correct —
            # good enough to distinguish lockup from spin events.
            # TinyPedal's rotation_radius() would give a dynamic per-sample
            # radius if we had mLongitudinalGroundVel stored separately, but
            # we don't yet — that's a recorder enhancement to add next.
            ESTIMATED_TYRE_RADIUS_M = 0.33
            wheel_rotation_rad_s = lap.channels[ws_key]
            wheel_linear_ms = np.abs(wheel_rotation_rad_s) * ESTIMATED_TYRE_RADIUS_M
            slip[label] = wheel_linear_ms / ground_ms_safe - 1.0
        else:
            # native LMU export: Wheel Speed already in m/s linear
            wheel_ms = lap.channels[ws_key]
            slip[label] = wheel_ms / ground_ms_safe - 1.0

    return slip


def compute_combined_g(lap: LapTelemetry) -> np.ndarray:
    """Friction-circle magnitude: sqrt(lat^2 + long^2), ignoring vertical G."""
    lat = lap.channels.get("G Force Lat", np.zeros_like(lap.time_s))
    lon = lap.channels.get("G Force Long", np.zeros_like(lap.time_s))
    return np.sqrt(lat**2 + lon**2)


def find_understeer_signatures(
    lap: LapTelemetry,
    steering_rate_threshold: float = 0.15,  # normalized steering units/sec
    lat_g_plateau_threshold: float = 0.05,  # G/sec — near-zero lateral G rate = plateaued
    min_speed_kmh: float = 60.0,  # ignore low-speed sections (hairpins, chicanes) where this proxy is noisy
) -> list[GripEvent]:
    """
    Understeer signature: driver keeps adding steering input (steering rate
    stays meaningfully positive) while lateral G stops increasing (its rate
    flattens toward zero) — the front tires have stopped generating more
    cornering force despite more steering angle being asked of them.

    This is a heuristic, not a certainty — flag it as a candidate for the
    LLM/driver to weigh against what actually happened at that point on
    track, not as ground truth the way ABS/TC events are.
    """
    if "Steering Pos" not in lap.channels or "G Force Lat" not in lap.channels:
        return []

    dt = np.gradient(lap.time_s)
    steer_rate = np.abs(np.gradient(lap.channels["Steering Pos"], lap.time_s))
    lat_g_rate = np.abs(np.gradient(lap.channels["G Force Lat"], lap.time_s))
    speed = lap.channels.get("Ground Speed", np.full_like(lap.time_s, min_speed_kmh))

    mask = (
        (steer_rate > steering_rate_threshold)
        & (lat_g_rate < lat_g_plateau_threshold)
        & (speed > min_speed_kmh)
        & (np.abs(lap.channels["Steering Pos"]) > 0.1)  # actually turning, not straight-line noise
    )

    events = []
    for start_i, end_i in _contiguous_true_regions(mask, lap.time_s):
        duration = lap.time_s[end_i] - lap.time_s[start_i]
        if duration < 0.2:  # filter noise-length blips
            continue
        events.append(
            GripEvent(
                kind="understeer",
                start_time_s=float(lap.time_s[start_i]),
                end_time_s=float(lap.time_s[end_i]),
                lap_dist_start_m=float(lap.channels["Lap Dist"][start_i]),
                lap_dist_end_m=float(lap.channels["Lap Dist"][end_i]),
                peak_severity=float(duration),
                notes=(
                    f"Steering input increasing for {duration:.2f}s with little lateral G "
                    "gain — possible understeer (front grip saturated)"
                ),
            )
        )
    return events


def build_grip_report(lap: LapTelemetry, car_setup_json: dict | None = None) -> dict:
    """
    Top-level entry point: runs all grip/slip detectors on a lap and returns
    a compact summary dict suitable for handing to the LLM coach or a UI.

    car_setup_json: pass the lap's parsed CarSetup metadata (from
    telemetry.py's metadata read) when available — it improves slip angle
    accuracy via the real steering lock/ratio for this specific setup
    rather than falling back to per-car defaults.
    """
    abs_events = find_abs_lockups(lap)
    tc_events = find_tc_interventions(lap)
    understeer_events = find_understeer_signatures(lap)
    combined_g = compute_combined_g(lap)
    wheel_slip = compute_wheel_slip(lap)

    slip_trace = compute_slip_trace(lap, car_setup_json=car_setup_json)
    slip_summary = summarize_slip_trace(slip_trace, lap)

    return {
        "lap_number": lap.lap_number,
        "lap_time_s": lap.lap_time_s,
        "driver_name": lap.driver_name,
        "car_name": lap.car_name,
        "track_name": lap.track_name,
        "peak_combined_g": float(np.nanmax(combined_g)) if len(combined_g) else None,
        "abs_events": [e.__dict__ for e in abs_events],
        "tc_events": [e.__dict__ for e in tc_events],
        "understeer_candidates": [e.__dict__ for e in understeer_events],
        "max_wheel_slip_ratio": {
            label: round(float(np.nanmax(np.abs(v))), 4) for label, v in wheel_slip.items()
        },
        "slip_angle_analysis": slip_summary,
    }
