"""
Computes slip angle, slip ratio, and yaw rate from telemetry using standard
bicycle-model vehicle dynamics equations — the same category of math any
racing telemetry tool (TinyPedal included) is built on. We can't verify
TinyPedal's exact source against this (GitHub blocked automated fetching of
their repo during development), so treat this as "standard textbook vehicle
dynamics," not "TinyPedal's exact formula." If you want to verify/tune
against TinyPedal specifically, clone their repo and drop calculation.py
here for comparison.

LMU's .duckdb files don't expose wheelbase or track width as numeric values
(VM_FRONT_WHEEL_TRACK / VM_REAR_WHEEL_TRACK come through as "Non-adjustable"
placeholder strings, not measurements) — so those are hardcoded per car
model below, sourced from public real-world specs. This is the one part of
this module that needs maintaining as you add cars.

Core equations (standard bicycle model):
  - Slip ratio (per wheel): (wheel_speed - ground_speed) / ground_speed
  - Yaw rate (r): approximated from lateral G and speed for a point mass:
        r ≈ G_lat_ms2 / speed_ms
    This is an approximation (ignores the vehicle's own slip angle rate,
    which a full bicycle model would include) but is standard practice when
    a direct yaw rate sensor/channel isn't available — same assumption
    underlies most GPS-lap-timer tools when no gyro is present.
  - Vehicle (chassis) slip angle (β): standard steady-state simplification,
    β ≈ 0. A more complete model would derive β by integrating lateral
    velocity from lateral acceleration, but that approach was tried during
    development and produced unusable results — see compute_slip_trace()
    for the full explanation of why, and what this means for result
    interpretation (short version: trust the front/rear slip angle
    difference and its sign, not the absolute axle values in degrees).
  - Front axle slip angle (α_f): δ - β - (a·r)/V
  - Rear axle slip angle (α_r): -β + (b·r)/V
    where δ = steering angle at the wheel (steering input / steering ratio),
    a = distance from CG to front axle, b = distance from CG to rear axle,
    r = yaw rate, V = speed. This is the standard linear bicycle model used
    throughout vehicle dynamics literature (see e.g. Milliken's "Race Car
    Vehicle Dynamics" or the OptimumG seminar materials).

Everything here is an approximation built from available channels — not a
readout of LMU's actual internal tire model. Good enough to say "the front
axle was sliding more than the rear here" (understeer) with confidence; not
precise enough to claim an exact slip angle in degrees matches what LMU's
physics engine computed internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .telemetry import LapTelemetry

# Real-world specs per car, since LMU's setup export doesn't include them.
# wheelbase_m: front axle to rear axle distance.
# cg_to_front_m / cg_to_rear_m: approximate 50/50-ish split if unknown;
# refine with real weight distribution specs when available (GT3 cars are
# usually close to 45/55 or so depending on model — treat these as
# reasonable estimates, not precise figures).
# steering_ratio: wheel degrees of driver input per degree of front wheel
# angle. Pulled from VM_STEER_LOCK when available (see note below);
# fallback used if a car isn't in this table yet.
CAR_SPECS: dict[str, dict] = {
    "McLaren 720S LMGT3 Evo": {
        "wheelbase_m": 2.670,
        "cg_to_front_m": 1.28,  # estimate; refine if real weight distribution is known
        "cg_to_rear_m": 1.39,
        "steering_ratio_fallback": 13.0,  # used only if VM_STEER_LOCK parsing fails
        "max_front_wheel_deg": 18.5,  # from this car's VM_STEER_LOCK setup value; varies by steering lock setting, re-derive per-setup if changed
    },
}

DEFAULT_SPECS = {
    "wheelbase_m": 2.7,
    "cg_to_front_m": 1.35,
    "cg_to_rear_m": 1.35,
    "steering_ratio_fallback": 13.0,
    "max_front_wheel_deg": 18.5,
}


@dataclass
class SlipTrace:
    time_s: np.ndarray
    yaw_rate_deg_s: np.ndarray
    vehicle_slip_angle_deg: np.ndarray
    front_slip_angle_deg: np.ndarray
    rear_slip_angle_deg: np.ndarray
    slip_angle_difference_deg: np.ndarray  # front - rear; positive = understeer tendency


def get_car_specs(car_name: str) -> dict:
    return CAR_SPECS.get(car_name, DEFAULT_SPECS)


def _extract_steering_ratio(car_setup_json: dict | None, fallback: float) -> float:
    """
    VM_STEER_LOCK in LMU's setup JSON encodes both the wheel rotation range
    and the resulting front wheel angle at full lock, e.g. a stringValue of
    "524deg (18.5deg)" means 524 degrees of driver wheel rotation maps to
    18.5 degrees of front wheel angle at full lock -> ratio = 524/18.5.
    Falls back to a fixed estimate if this can't be parsed (setup JSON
    missing, unexpected format, etc.) rather than crashing the whole
    analysis over a cosmetic detail.
    """
    if not car_setup_json:
        return fallback
    try:
        steer_lock = car_setup_json.get("VM_STEER_LOCK", {})
        string_value = steer_lock.get("stringValue", "")
        # format: "524deg (18.5deg)"
        parts = string_value.replace("deg", "").replace(")", "").split("(")
        wheel_deg = float(parts[0].strip())
        front_wheel_deg = float(parts[1].strip())
        if front_wheel_deg == 0:
            return fallback
        return wheel_deg / front_wheel_deg
    except (KeyError, IndexError, ValueError, AttributeError):
        return fallback


def _extract_max_front_wheel_deg(car_setup_json: dict | None, fallback: float) -> float:
    """Same VM_STEER_LOCK field as steering ratio; this pulls the front wheel
    angle at full lock directly, e.g. "524deg (18.5deg)" -> 18.5."""
    if not car_setup_json:
        return fallback
    try:
        steer_lock = car_setup_json.get("VM_STEER_LOCK", {})
        string_value = steer_lock.get("stringValue", "")
        parts = string_value.replace("deg", "").replace(")", "").split("(")
        return float(parts[1].strip())
    except (KeyError, IndexError, ValueError, AttributeError):
        return fallback


def compute_slip_trace(
    lap: LapTelemetry, car_setup_json: dict | None = None
) -> SlipTrace:
    specs = get_car_specs(lap.car_name)
    steering_ratio = _extract_steering_ratio(
        car_setup_json, specs["steering_ratio_fallback"]
    )
    max_front_wheel_deg = _extract_max_front_wheel_deg(
        car_setup_json, specs["max_front_wheel_deg"]
    )

    speed_ms = lap.channels["Ground Speed"] / 3.6  # km/h -> m/s
    # avoid division by zero/instability at low speed. TinyPedal's verified
    # source (calculation.py: yaw_rate()) uses an 8 m/s minimum speed
    # threshold before trusting this approximation, returning 0 below it —
    # adopted directly here rather than the looser 0.5 m/s floor used
    # earlier, since this constant is now confirmed rather than guessed.
    min_yaw_speed_ms = 8.0
    speed_ms_safe = np.maximum(speed_ms, 0.5)  # still need a nonzero denominator

    lat_g = lap.channels.get("G Force Lat", np.zeros_like(speed_ms))
    lat_accel_ms2 = lat_g * 9.81

    # --- Yaw rate: prefer the real channel from the recorder (mLocalRot.y,
    # direct from LMU's physics engine, no approximation), fall back to the
    # a_lat/V approximation for native LMU .duckdb export files which don't
    # include it. Both branches are guarded below min_yaw_speed_ms.
    if "Yaw Rate" in lap.channels:
        yaw_rate_rad_s = np.where(
            speed_ms > min_yaw_speed_ms, lap.channels["Yaw Rate"], 0.0
        )
    else:
        # approximation: r ≈ a_lat / V (matches TinyPedal's verified formula)
        yaw_rate_rad_s = np.where(
            speed_ms > min_yaw_speed_ms, lat_accel_ms2 / speed_ms_safe, 0.0
        )

    # --- Per-wheel slip angle: prefer the real values from the recorder
    # (computed from mLateralGroundVel/mLongitudinalGroundVel per wheel,
    # exactly TinyPedal's formula: atan(lateral/longitudinal)), stored as
    # "Slip Angle FL/FR/RL/RR" after telemetry.py splits the value1-4 columns.
    # When present, front/rear axle slip angle is just the mean of the two
    # wheels on each axle — no bicycle model needed.
    # Fall back to the β≈0 bicycle model for native export files.
    fl_key = "Slip Angle FL"
    if fl_key in lap.channels:
        front_slip_rad = (lap.channels["Slip Angle FL"] + lap.channels["Slip Angle FR"]) / 2
        rear_slip_rad = (lap.channels["Slip Angle RL"] + lap.channels["Slip Angle RR"]) / 2
        vehicle_slip_rad = np.zeros_like(speed_ms)  # not needed when real slip angles are present

        return SlipTrace(
            time_s=lap.time_s,
            yaw_rate_deg_s=np.degrees(yaw_rate_rad_s),
            vehicle_slip_angle_deg=np.zeros_like(speed_ms),
            front_slip_angle_deg=np.degrees(front_slip_rad),
            rear_slip_angle_deg=np.degrees(rear_slip_rad),
            slip_angle_difference_deg=np.degrees(front_slip_rad - rear_slip_rad),
        )

    # --- Fallback: β≈0 bicycle model for native LMU .duckdb export files.
    # Vehicle (chassis) slip angle: NOT computed by integrating lateral
    # acceleration. That was tried and produced unusable results — over a
    # ~140s lap, integrating noisy/biased accelerometer data with no
    # independent correction accumulates tens of m/s of drift (verified:
    # raw integration swung +/-38 m/s, faster than the car itself), which
    # is a known limitation of dead-reckoning without a reference to
    # correct against. Rather than patch that with an ad-hoc filter and
    # risk quietly-wrong numbers, this uses the standard steady-state
    # simplification instead: assume vehicle slip angle β ≈ 0.
    #
    # This is a real, textbook simplification (not a shortcut taken to
    # dodge the bug) — for quasi-steady-state cornering, which is most of
    # a lap outside of sudden transients/catches, β stays small relative to
    # the axle slip angles, and yaw rate + steering input alone already
    # capture the understeer/oversteer signature we actually need for
    # coaching. It means: don't trust absolute slip angle values from this
    # module as precise degrees, but DO trust the front-vs-rear slip angle
    # DIFFERENCE and its sign — that comparison is much less sensitive to
    # the β≈0 assumption than either axle's absolute value is.
    vehicle_slip_rad = np.zeros_like(speed_ms)

    # steering input -> front wheel angle in radians.
    # LMU's channelsList labels "Steering Pos" as unit "%", and the raw
    # values in this file range up to ~±55.6 — inconsistent with either a
    # true percentage (expected max 100) or a normalized -1..1 signal. The
    # working interpretation used here: value is percent of the car's full
    # steering lock (VM_STEER_LOCK), i.e. a value of 55.6 means the driver
    # was at 55.6% of max lock at that instant. This is validated below by
    # checking the resulting front wheel angle never exceeds the car's
    # physical lock limit — if it does, this interpretation is wrong for
    # your data and needs re-deriving (see the assert/warning below).
    steering_pct = lap.channels.get("Steering Pos", np.zeros_like(speed_ms))
    front_wheel_deg = (steering_pct / 100.0) * max_front_wheel_deg
    front_wheel_angle_rad = np.radians(front_wheel_deg)

    if np.nanmax(np.abs(front_wheel_deg)) > max_front_wheel_deg * 1.05:
        # the interpretation above doesn't hold for this data — flag loudly
        # rather than silently produce wrong slip angles
        import warnings
        warnings.warn(
            f"Steering Pos interpretation produced front wheel angle "
            f"{np.nanmax(np.abs(front_wheel_deg)):.1f} deg, exceeding the "
            f"car's physical lock ({max_front_wheel_deg} deg) by >5%. The "
            f"percent-of-lock assumption for this channel is likely wrong "
            f"for this file — slip angle values below should not be trusted "
            f"until this is re-derived against a known reference (e.g. a "
            f"logged full-lock moment in the pits)."
        )

    a = specs["cg_to_front_m"]
    b = specs["cg_to_rear_m"]

    front_slip_rad = (
        front_wheel_angle_rad - vehicle_slip_rad - (a * yaw_rate_rad_s) / speed_ms_safe
    )
    rear_slip_rad = -vehicle_slip_rad + (b * yaw_rate_rad_s) / speed_ms_safe

    return SlipTrace(
        time_s=lap.time_s,
        yaw_rate_deg_s=np.degrees(yaw_rate_rad_s),
        vehicle_slip_angle_deg=np.degrees(vehicle_slip_rad),
        front_slip_angle_deg=np.degrees(front_slip_rad),
        rear_slip_angle_deg=np.degrees(rear_slip_rad),
        slip_angle_difference_deg=np.degrees(front_slip_rad - rear_slip_rad),
    )


def summarize_slip_trace(slip: SlipTrace, lap: LapTelemetry) -> dict:
    """
    Compact summary for the LLM prompt: peak values and where (by lap
    distance) they occurred, rather than the full per-tick trace.
    """
    def _peak_with_location(arr: np.ndarray, use_abs: bool = True) -> dict:
        idx = int(np.nanargmax(np.abs(arr))) if use_abs else int(np.nanargmax(arr))
        return {
            "value": round(float(arr[idx]), 2),
            "time_s": round(float(slip.time_s[idx]), 2),
            "lap_dist_m": round(float(lap.channels["Lap Dist"][idx]), 1),
        }

    return {
        "peak_yaw_rate_deg_s": _peak_with_location(slip.yaw_rate_deg_s),
        "peak_vehicle_slip_angle_deg": _peak_with_location(slip.vehicle_slip_angle_deg),
        "peak_front_slip_angle_deg": _peak_with_location(slip.front_slip_angle_deg),
        "peak_rear_slip_angle_deg": _peak_with_location(slip.rear_slip_angle_deg),
        "peak_slip_angle_difference_deg": _peak_with_location(
            slip.slip_angle_difference_deg
        ),
        "note": (
            "Positive slip angle difference (front - rear) indicates understeer "
            "tendency; negative indicates oversteer tendency, at the flagged moment."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Handling balance: understeer / neutral / oversteer from tyre slip angles
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the UI's old US/Neu/OS bars, which counted the sign of the
# suspension front-minus-rear deflection over the whole lap: that is rake and
# pitch (aero load, braking), not balance, and it counted straights.
# Here: US/OS = which axle runs the bigger slip angle INTO the turn, only
# while actually cornering, and only when the slip agrees with the way the
# car is rotating (yaw rate) - so kerb strikes and direction changes don't
# count. MEASURED 10/14/2026 on the Fuji HF file (laps 2, 6, 11): signed
# slip (screen convention) correlates -0.88..-0.95 with Yaw Rate, so the
# slip "into the turn" is -sign(yaw) * slip. The turn-projected delta had a
# median of about +0.45 deg; a 0.12 deg band left only 8 % neutral (noise
# level), 0.25 deg about 20 %.
HANDLING_NEUTRAL_BAND_DEG = 0.25
HANDLING_MIN_SPEED_KMH = 40.0
HANDLING_MIN_YAW_RAD_S = 0.05
HANDLING_MIN_LAT_G = 0.3


def handling_balance(front_slip_deg, rear_slip_deg, yaw_rate, lat_g, speed_kmh) -> dict:
    """Per-sample balance and lap shares, all inputs on the same samples.

    front/rear_slip_deg: axle-mean SIGNED slip angle, /trackmap screen
    convention. Returns delta_deg (front minus rear slip into the turn;
    + = understeer, - = oversteer, 0 outside counted cornering) and the
    us/neutral/os shares of counted cornering samples (None if none).
    """
    f = np.asarray(front_slip_deg, dtype=np.float64)
    r = np.asarray(rear_slip_deg, dtype=np.float64)
    yaw = np.asarray(yaw_rate, dtype=np.float64)
    g = np.asarray(lat_g, dtype=np.float64)
    v = np.asarray(speed_kmh, dtype=np.float64)
    into = -np.sign(yaw)
    f_turn, r_turn = f * into, r * into
    cornering = ((v > HANDLING_MIN_SPEED_KMH) & (np.abs(yaw) > HANDLING_MIN_YAW_RAD_S)
                 & (np.abs(g) > HANDLING_MIN_LAT_G)
                 # yaw check: at least one axle must slip the way the car turns
                 & ((f_turn > 0) | (r_turn > 0)))
    delta = np.where(cornering, f_turn - r_turn, 0.0)
    n = int(cornering.sum())
    out = {"delta_deg": [round(float(d), 3) for d in delta],
           "cornering_samples": n, "samples": len(delta),
           "band_deg": HANDLING_NEUTRAL_BAND_DEG,
           "us": None, "neutral": None, "os": None}
    if n:
        d = delta[cornering]
        out["us"] = float(np.mean(d > HANDLING_NEUTRAL_BAND_DEG))
        out["os"] = float(np.mean(d < -HANDLING_NEUTRAL_BAND_DEG))
        out["neutral"] = 1.0 - out["us"] - out["os"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Body slip angle β (chassis heading minus direction of travel)
# ─────────────────────────────────────────────────────────────────────────────
def _wrap(a):
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def body_slip_from_local_vel_deg(local_vel_x, local_vel_z, native_x_lap, speed_kmh,
                                 min_speed_kmh: float = 25.0) -> list:
    """Body slip β in DEGREES from the car's own velocity (recorder.py
    files record Local Vel X/Z every tick). Same convention as
    compute_body_slip_deg: positive = body rotated clockwise on screen
    relative to the velocity vector.

    Preferred over compute_body_slip_deg when present: it is measured, not
    reconstructed, so it needs no straights to calibrate on and works on
    partial laps. Car forward is -z in LMU's local frame. The sign of x was
    MEASURED on the 10/14/2026 Fuji HF file: atan2(vx, -vz) agrees with the
    yaw-integral estimate at r = 0.90..0.94, 0.3 deg RMS (laps 2, 6, 11).

    native_x_lap: fractional row of each lap sample in the Local Vel tables.
    Returns [] when the channels are missing.
    """
    vx = np.asarray(local_vel_x, dtype=np.float64)
    vz = np.asarray(local_vel_z, dtype=np.float64)
    if len(vx) < 2 or len(vx) != len(vz):
        return []
    rows = np.arange(len(vx), dtype=np.float64)
    nx = np.asarray(native_x_lap, dtype=np.float64)
    beta = np.degrees(np.arctan2(np.interp(nx, rows, vx), -np.interp(nx, rows, vz)))
    beta[np.asarray(speed_kmh, dtype=np.float64) < min_speed_kmh] = 0.0
    return [round(float(b), 3) for b in beta]


def compute_body_slip_deg(yaw_rate_100hz, time_100hz, s_idx, e_idx, full_len,
                          course_rad, speed_kmh, yaw_lap,
                          min_speed_kmh: float = 25.0,
                          native_x_lap=None) -> list:
    """Estimate body slip β in DEGREES for one lap (positive = body rotated
    clockwise on screen relative to the velocity vector).

    β = ψ − course, where ψ is body heading (integral of Yaw Rate) and
    course is the direction of travel derived from position. The integral
    has an unknown offset and an unknown sign convention, so both are
    calibrated per lap on straights (|yaw|<0.03 rad/s, v>50 km/h) where
    β≈0. Returns [] (caller falls back to heading-only) if the fit is not
    credible, e.g. native LMU files with no Yaw Rate channel.

    native_x_lap: fractional 100 Hz row of each lap sample, by game time.
    Without it the 100 Hz rows are assumed evenly spaced across the file,
    which is wrong once the game skips ticks (10/14 Fuji HF file: ψ and the
    course were then seconds apart and every lap was refused).
    """
    try:
        course = np.asarray(course_rad, dtype=np.float64)
        speed = np.asarray(speed_kmh, dtype=np.float64)
        yl = np.asarray(yaw_lap, dtype=np.float64)
        yaw = np.asarray(yaw_rate_100hz, dtype=np.float64) if len(yaw_rate_100hz) else np.zeros(0)
        n, N = len(course), len(yaw)
        if n < 50 or N < 50 or len(speed) != n or len(yl) != n or full_len < 2:
            return []
        if native_x_lap is not None and len(native_x_lap) == n:
            nx = np.asarray(native_x_lap, dtype=np.float64)
            i0 = int(np.floor(nx[0]))
            i1 = min(int(np.ceil(nx[-1])), N - 1)
        else:
            nx = None
            i0 = int(round(s_idx * (N - 1) / (full_len - 1)))
            i1 = int(round(e_idx * (N - 1) / (full_len - 1)))
        r = yaw[i0:i1 + 1]
        if len(r) < 50 or not np.all(np.isfinite(r)) or float(np.std(r)) < 1e-3:
            return []
        if time_100hz is not None and len(time_100hz) == N:
            # Real per-tick steps: a skipped tick is a 20 ms step, not 10 ms.
            dts = np.diff(np.asarray(time_100hz[i0:i1 + 1], dtype=np.float64))
        else:
            dts = np.full(len(r) - 1, 0.01)
        psi100 = np.concatenate([[0.0], np.cumsum((r[1:] + r[:-1]) * 0.5 * dts)])
        if nx is not None:
            psi = np.interp(nx - i0, np.arange(len(psi100)), psi100)
        else:
            psi = np.interp(np.linspace(0, len(psi100) - 1, n), np.arange(len(psi100)), psi100)
        crs = np.unwrap(course)

        straight = (np.abs(yl) < 0.03) & (speed > 50.0)
        fast = speed > min_speed_kmh
        if straight.sum() < 40 or fast.sum() < 100:
            return []

        best = None
        for sgn in (1.0, -1.0):
            d = crs - sgn * psi
            c = np.arctan2(np.mean(np.sin(d[straight])), np.mean(np.cos(d[straight])))
            res = np.abs(_wrap(d - c))[fast]
            med = float(np.median(res))
            if best is None or med < best[0]:
                best = (med, sgn, c, float(np.mean(res > np.radians(20))))
        med, sgn, c, frac_big = best
        if np.degrees(med) > 4.0 or frac_big > 0.25:
            return []

        beta = _wrap(sgn * psi + c - crs)
        k = np.ones(3) / 3.0
        beta = np.convolve(np.pad(beta, 1, mode="edge"), k, mode="valid")
        beta[~fast] = 0.0
        return [round(float(np.degrees(b)), 3) for b in beta]
    except Exception:
        return []
