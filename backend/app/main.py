"""
FastAPI backend for the local LMU telemetry coach.

Endpoints:
  GET  /sessions              — list .duckdb files found in mounted telemetry folders
  GET  /sessions/{file}/laps  — list laps found in a given file, with basic stats
  POST /analyze                — run grip analysis + LLM coaching on one lap
  POST /analyze/followup       — ask a follow-up question about a prior analysis

Session state (grip report + conversation history) is kept in memory only,
keyed by a session_id returned from /analyze — this is a single-user local
tool, not a multi-tenant service, so there's no need for a real DB here yet.
"""

from __future__ import annotations

import copy
import logging
import math
import uuid
from pathlib import Path

import duckdb
import numpy as np

try:
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY = True
except ImportError:
    # Falls back to a simple EMA smoother if scipy isn't installed in this
    # container yet (e.g. requirements.txt not rebuilt). Not as clean as a
    # zero-phase Butterworth but keeps the endpoint working either way.
    _HAS_SCIPY = False

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .coach import LLMError, ask_followup, get_coaching_analysis
from .track_db import find_track_in_registry, find_layout_in_track, annotate_report_with_corners
from .grip_analysis import build_grip_report
from .grip_profile import load_profile_for_lap, save_profile
from .tire_model import TIRE_GRID_HZ, compute_grip_utilization_for_lap, file_load_reference
from .vehicle_dynamics import (body_slip_from_local_vel_deg, compute_body_slip_deg,
                               handling_balance)
from .telemetry import load_telemetry_file, get_car_setup_json, session_time_axes, reference_lap_time

log = logging.getLogger(__name__)

app = FastAPI(title="LMU Telemetry Coach")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Low-pass filter for noisy dynamics channels ──────────────────────────────
# FFT analysis on real recorded laps (Yaw Rate, Slip Angle) showed 90%+ of
# signal energy sits below 1Hz — the genuine driving input (corner entry/
# exit, braking rotation) — with only a thin, low-amplitude tail above
# ~3-5Hz that looks like physics-engine/tire-model jitter rather than real
# vehicle behaviour. A real event (e.g. a lockup or a sharp trail-braking
# moment) showed up as a large, COHERENT spike across yaw rate and all four
# slip-angle channels at once — that's signal, not noise, and a cutoff this
# low leaves it untouched: an 8Hz cutoff only removes content happening
# faster than 1/8s, far quicker than any real chassis rotation event.
YAW_SLIP_FILTER_CUTOFF_HZ = 8.0
YAW_SLIP_FILTER_ORDER     = 2   # 2nd-order Butterworth — gentle rolloff


def _lowpass_filter(arr: list[float], sample_hz: float,
                     cutoff_hz: float = YAW_SLIP_FILTER_CUTOFF_HZ) -> list[float]:
    """
    Zero-phase low-pass filter for noisy per-sample dynamics channels
    (Yaw Rate, Slip Angle). Zero-phase (filtfilt, forward+backward) matters
    here specifically because this data drives a car-position chevron and
    a live trace scrubber — a phase-shifting filter would make the chevron
    visibly lag the actual car position, which is worse than the noise it
    was meant to remove.

    sample_hz is the array's OWN native rate (100Hz for Yaw Rate/Slip Angle
    as recorded), not full_len or the position array's rate — those can
    differ (see full_len's GPS-vs-Pos-X/Z note earlier in this file), so
    callers must pass the correct source rate for the array being filtered.

    Falls back to a simple exponential moving average (still smooths, but
    NOT zero-phase, so it does add a small lag) if scipy isn't available —
    keeps the endpoint working rather than hard-failing on a missing
    dependency.
    """
    n = len(arr)
    if n < 15:  # too short for a stable filtfilt pad; not worth filtering anyway
        return arr
    if not _HAS_SCIPY:
        alpha = 0.3
        out = [arr[0]]
        for v in arr[1:]:
            out.append(alpha * v + (1 - alpha) * out[-1])
        return out

    nyquist = sample_hz / 2.0
    norm_cutoff = min(cutoff_hz / nyquist, 0.99)  # clamp — cutoff must stay < Nyquist
    b, a = butter(YAW_SLIP_FILTER_ORDER, norm_cutoff, btype="low")
    # filtfilt needs padlen < signal length; guard short lap slices too.
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return arr
    return filtfilt(b, a, np.asarray(arr, dtype=np.float64)).tolist()


# ── Per-wheel slip angle for the tyre gauges ─────────────────────────────────
# +1 or -1: turns the recorder's raw atan(lateral/longitudinal ground
# velocity) into the SCREEN convention the gauge draws - positive = the tyre
# is travelling to the RIGHT of where it points (clockwise on the dial).
# In a right-hand corner a tyre travels to the left of where it points, so a
# right-hander must come out negative. MEASURED 10/12/2026 through /trackmap
# on three recorder files (Aston #007, Cadillac #12, Akkodis #78): with +1,
# front slip correlates -0.89..-0.92 with the screen's right-turn rate
# (about -2.6 deg in hard rights, +2.3 deg in hard lefts). So the recorder's
# raw sign already is the screen convention.
# backend/tests/test_trackmap_real_file.py re-checks this on a local file.
SLIP_ANGLE_SCREEN_SIGN = 1.0
# Below this speed atan(lat/long) is a ratio of two near-zero numbers and
# swings to +-90 deg (seen in every recording while parked), which would peg
# the gauge in the pits. Same idea as body slip being zeroed at low speed.
SLIP_ANGLE_MIN_SPEED_KMH = 20.0
SLIP_ANGLE_CLIP_DEG = 45.0
_WHEEL_KEYS = ("fl", "fr", "rl", "rr")


def _wheel_slip_angles_deg(sa_raw: list, speed_kmh_raw: list, s_idx: int,
                           e_idx: int, full_len: int,
                           native_x: np.ndarray | None = None) -> dict[str, list[float]]:
    """
    Signed slip angle per wheel, degrees, for the lap slice [s_idx, e_idx].

    sa_raw is the recorder's "Slip Angle" table (rows of FL, FR, RL, RR in
    radians, native 100Hz); speed_kmh_raw is "Ground Speed" at its own
    native length. Follows the file-wide rule: filter at native rate,
    interpolate to full_len, slice LAST. Returns four empty lists when the
    file has no Slip Angle table (native LMU exports). native_x, when given,
    is the fractional Slip Angle row of each of the full_len output rows
    (by game time); without it the rows are assumed evenly spaced.
    """
    if not sa_raw or full_len <= 0:
        return {k: [] for k in _WHEEL_KEYS}
    sa = np.asarray(sa_raw, dtype=np.float64)
    n = len(sa)
    full_x = (np.asarray(native_x, dtype=np.float64)
              if native_x is not None and len(native_x) == full_len
              else np.linspace(0.0, n - 1, full_len))
    native_x = np.arange(n, dtype=np.float64)

    slow_native = slow_full = None
    if len(speed_kmh_raw) > 1:
        spd = np.asarray(speed_kmh_raw, dtype=np.float64)
        spd_x = np.linspace(0.0, n - 1, len(spd))
        slow_native = np.interp(native_x, spd_x, spd) < SLIP_ANGLE_MIN_SPEED_KMH
        slow_full = np.interp(full_x, spd_x, spd) < SLIP_ANGLE_MIN_SPEED_KMH

    out: dict[str, list[float]] = {}
    for c, key in enumerate(_WHEEL_KEYS):
        deg = np.clip(np.degrees(sa[:, c]) * SLIP_ANGLE_SCREEN_SIGN,
                      -SLIP_ANGLE_CLIP_DEG, SLIP_ANGLE_CLIP_DEG)
        if slow_native is not None:
            # Zeroed BEFORE filtering: otherwise the low-pass smears the
            # parked-car +-90 deg spikes into the first moving samples.
            deg[slow_native] = 0.0
        filtered = np.asarray(_lowpass_filter(deg.tolist(), sample_hz=100.0))
        full = np.clip(np.interp(full_x, native_x, filtered),
                       -SLIP_ANGLE_CLIP_DEG, SLIP_ANGLE_CLIP_DEG)
        if slow_full is not None:
            full[slow_full] = 0.0  # and again after: filter ringing at the edge
        out[key] = np.round(full[s_idx : e_idx + 1], 2).tolist()
    return out


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "UI not found — static/index.html missing")
    return FileResponse(str(index))

# Read-only bind mounts set up in docker-compose.yml. OWN_SESSIONS is your
# own LMU UserData/Telemetry folder; REFERENCE_LAPS is the local folder
# where you manually drop .duckdb files (e.g. ones friends send you, or your
# own exports) that aren't part of your live LMU session history.
OWN_SESSIONS_DIR = Path("/data/sessions")
REFERENCE_LAPS_DIR = Path("/data/reference_laps")

# In-memory session store: session_id -> {"grip_report": dict, "history": [...]}
_analysis_sessions: dict[str, dict] = {}


class LapRef(BaseModel):
    file: str  # filename only, resolved against one of the two data dirs
    source: str = "sessions"  # "sessions" | "reference_laps"
    lap_number: int | None = None  # if None, uses the longest/fastest lap found
    # Optional per-request LLM settings from the web UI's Settings panel.
    # Omitted = fall back to the container environment defaults (see coach.py).
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float | None = None


class FollowupRequest(BaseModel):
    session_id: str
    question: str
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float | None = None


def _resolve_file(file: str, source: str) -> Path:
    base = OWN_SESSIONS_DIR if source == "sessions" else REFERENCE_LAPS_DIR
    path = (base / file).resolve()
    # guard against path traversal outside the intended data dir
    if base.resolve() not in path.parents and path != base.resolve():
        raise HTTPException(400, "Invalid file path")
    if not path.exists():
        raise HTTPException(404, f"File not found: {file}")
    return path


def _list_duckdb_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.glob("*.duckdb"))


@app.get("/sessions")
def list_sessions():
    return {
        "sessions": _list_duckdb_files(OWN_SESSIONS_DIR),
        "reference_laps": _list_duckdb_files(REFERENCE_LAPS_DIR),
    }


@app.get("/sessions/{source}/{file}/laps")
def list_laps(source: str, file: str):
    path = _resolve_file(file, source)
    laps = load_telemetry_file(path)
    return {
        "laps": [
            {
                "lap_number": lap.lap_number,
                "lap_time_s": round(lap.lap_time_s, 3),
                "car_name": lap.car_name,
                "track_name": lap.track_name,
                "session_type": lap.session_type,
            }
            for lap in laps
            if lap.lap_time_s < 600  # filter obviously-partial trailing laps
        ]
    }


@app.post("/analyze")
async def analyze(ref: LapRef):
    path = _resolve_file(ref.file, ref.source)
    laps = load_telemetry_file(path)
    valid_laps = [l for l in laps if l.lap_time_s < 600]
    if not valid_laps:
        raise HTTPException(422, "No complete laps found in this file")

    if ref.lap_number is not None:
        matches = [l for l in valid_laps if l.lap_number == ref.lap_number]
        if not matches:
            raise HTTPException(404, f"Lap {ref.lap_number} not found in file")
        lap = matches[0]
    else:
        # fastest REAL lap: min() alone picked the unfinished last lap
        ref_time = reference_lap_time(valid_laps)
        lap = (next(l for l in valid_laps if l.lap_time_s == ref_time and l.is_complete)
               if ref_time is not None else min(valid_laps, key=lambda l: l.lap_time_s))

    grip_report = build_grip_report(lap, car_setup_json=get_car_setup_json(path))
    annotate_report_with_corners(
        grip_report, lap.track_name,
        lap.meta.get("TrackLayout", lap.meta.get("Layout", "")))

    # Grip utilization / cornering efficiency (tire_model.py) needs to be
    # scored against the model as it existed BEFORE this lap's own samples
    # are folded in — otherwise a lap would always partially include
    # itself in its own reference curve. So: load the current profile's
    # baselines first, compute this lap's utilization against THAT, and
    # only then fold this lap's samples into the saved profile.
    # PREVIOUSLY the profile was keyed by lap.car_name (one file per livery) and the
    # tyre model read the same 20 Hz load as the coaching report, which leaves too few
    # steady samples per lap to populate the v3 state cells. Fix: the profile is the car
    # MODEL's, and the tyre steps load the file a second time at TIRE_GRID_HZ (same lap
    # number); the coaching report above keeps the 20 Hz load.
    pre_update_profile = load_profile_for_lap(lap)
    profile = pre_update_profile
    tire_laps: list = []
    tire_error: str | None = None
    grip_report["grip_utilization"] = None
    try:
        tire_laps = load_telemetry_file(path, grid_hz=TIRE_GRID_HZ)
        tire_valid = [l for l in tire_laps if l.lap_time_s < 600]
        tire_lap = next((l for l in tire_valid if l.lap_number == lap.lap_number), None)
        pre_update_baselines = pre_update_profile.compute_baselines()
        if tire_lap is not None:
            grip_report["grip_utilization"] = compute_grip_utilization_for_lap(
                tire_lap,
                pre_update_baselines["front_tire_model"],
                pre_update_baselines["rear_tire_model"],
                calibration=pre_update_profile.deflection_calibration,
                load_ref=file_load_reference(tire_valid, pre_update_profile.deflection_calibration),
                front_fit=pre_update_profile.cached_fit("front"),
            )
    except Exception as e:  # /analyze must never fail because of the tyre model
        log.exception("tyre model failed on %s lap %s", path.name, lap.lap_number)
        tire_error = f"{type(e).__name__}: {e}"
        tire_laps = []
        grip_report["grip_utilization"] = None

    # fold this lap into the car's running grip profile (including its raw
    # slip-angle/lateral-G/temp/pressure/load samples for the probabilistic
    # tire model), then use the updated baselines (including this lap) to
    # give the coach something to compare against — this is the "memory"
    # for grip limits: a local file that accumulates across sessions,
    # rather than anything the LLM itself has to recall
    # The profile is only SAVED after the LLM answers (below). Previously it
    # was saved first, so an LLM failure followed by a retry folded the same
    # lap into the tire model twice.
    # The tire model takes every usable lap of the file, once per lap (see
    # GripProfile.ingest_tire_laps); the ABS/TC/slip baselines still take
    # only the analyzed lap. Caveat: analyzing a second lap of an already
    # ingested file scores it against a model that already contains it.
    profile.update_from_grip_report(grip_report)
    # PREVIOUSLY a refit failure after a successful ingest was unguarded (HTTP 500) and a
    # failed ingest could leave half-ingested laps in the profile that was then saved.
    # Fix: the tyre ingest + refit is transactional. The tyre state is copied first and
    # put back if either step fails; the ABS/TC baselines above are kept and saved.
    grip_baselines = None
    if tire_error is None:
        snapshot = (copy.deepcopy(profile.tire_files), copy.deepcopy(profile.tire_fit_cache),
                    copy.deepcopy(profile.deflection_calibration), profile.car_class)
        try:
            profile.ingest_tire_laps(tire_laps, file_key=path.name)
            grip_baselines = profile.compute_baselines()
        except Exception as e:
            log.exception("tyre ingest/refit failed on %s", path.name)
            tire_error = f"{type(e).__name__}: {e}"
            (profile.tire_files, profile.tire_fit_cache,
             profile.deflection_calibration, profile.car_class) = snapshot
    if grip_baselines is None:
        grip_baselines = profile.compute_baselines(tire_unavailable=tire_error)

    try:
        analysis_text = await get_coaching_analysis(
            grip_report, grip_baselines,
            model=ref.llm_model, timeout_s=ref.llm_timeout_s, base_url=ref.llm_base_url,
        )
    except LLMError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    save_profile(profile)

    session_id = str(uuid.uuid4())
    _analysis_sessions[session_id] = {
        "grip_report": grip_report,
        "grip_baselines": grip_baselines,
        "history": [{"role": "assistant", "content": analysis_text}],
    }

    return {
        "session_id": session_id,
        "grip_report": grip_report,
        "grip_baselines": grip_baselines,
        "analysis": analysis_text,
    }


@app.post("/analyze/followup")
async def analyze_followup(req: FollowupRequest):
    session = _analysis_sessions.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id — run /analyze first")

    try:
        answer = await ask_followup(
            session["grip_report"],
            session["history"],
            req.question,
            grip_baselines=session.get("grip_baselines"),
            model=req.llm_model, timeout_s=req.llm_timeout_s, base_url=req.llm_base_url,
        )
    except LLMError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    session["history"].append({"role": "user", "content": req.question})
    session["history"].append({"role": "assistant", "content": answer})

    return {"session_id": req.session_id, "answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Track map endpoint ────────────────────────────────────────────────────────
# Handles both native LMU .duckdb files and recorder files.
#
# Native LMU:  GPS Latitude/Longitude at 10Hz + Path Lateral + Track Edge
#              GPS Time as master clock, ABS/TC/In Pits as event tables
# Recorder v2: same channels (GPS synthesized from mPos), same schema
# Recorder v1: Pos X/Z only, no Path Lateral/Track Edge — edges skipped

@app.get("/sessions/{source}/{file}/trackmap")
def get_trackmap(source: str, file: str, lap_number: int | None = None):
    path = _resolve_file(file, source)
    con  = duckdb.connect(str(path), read_only=True)
    tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}

    def _col(table: str, col: str = "value") -> list:
        try:
            return [r[0] for r in con.execute(
                f'SELECT {col} FROM "{table}" ORDER BY rowid'
            ).fetchall()]
        except Exception:
            return []

    def _cols4(table: str) -> list:
        try:
            return con.execute(
                f'SELECT value1,value2,value3,value4 FROM "{table}" ORDER BY rowid'
            ).fetchall()
        except Exception:
            return []

    def _event_active_col(table: str, ref_ts: list[float]) -> list:
        """
        Converts an (ts, value) event-log table — e.g. ABS/TC, which log a
        row only when the on/off state CHANGES (see actual data: sparse
        rows like (0.4, 0), (51.8, 1), (52.0, 0)...) — into a per-sample
        0/1 step-function array aligned to ref_ts (the same GPS Time array
        used to size every other full-length channel in this endpoint).
        Native LMU stores `value` as a bool; the recorder stores 0.0/1.0 —
        both come back truthy/falsy the same way via float(), so one path
        handles both sources.

        This does NOT use _interp_to_length()/np.interp — interpolating a
        step function between two ts/value points would blend a 0 and a 1
        into a meaningless 0.5, smearing exactly the transition moment
        that matters. Instead each ref_ts sample gets the value of the
        most recent event at or before it (forward-fill / step-hold),
        which is what "ABS was active from this instant until the next
        change" actually means.
        """
        try:
            rows = con.execute(f'SELECT ts, value FROM "{table}" ORDER BY ts').fetchall()
        except Exception:
            return []
        if not rows or not ref_ts:
            return []
        events = [(float(ts), 1.0 if value else 0.0) for ts, value in rows]
        out = []
        ei = 0
        cur = events[0][1] if events and events[0][0] <= ref_ts[0] else 0.0
        for t in ref_ts:
            while ei < len(events) and events[ei][0] <= t:
                cur = events[ei][1]
                ei += 1
            out.append(cur)
        return out

    # ── 1. Position data — which coordinate system? ──────────────────────────
    # Prefer GPS Lat/Lon (present in native LMU AND recorder v2).
    # Fall back to Pos X/Z world metres (recorder v1 only).
    use_gps = "GPS Latitude" in tables and "GPS Longitude" in tables

    if use_gps:
        lat_raw = _col("GPS Latitude")
        lon_raw = _col("GPS Longitude")
        if not lat_raw:
            raise HTTPException(422, "GPS Latitude table is empty")
        centre_lat = (min(lat_raw) + max(lat_raw)) / 2
        centre_lon = (min(lon_raw) + max(lon_raw)) / 2
        cos_lat    = math.cos(math.radians(centre_lat))
        R          = 111_320.0
        # Screen-space projection: +lon (east) → +x (right), +lat (north) → -z
        # (screen y-down convention, so "up" on screen = -z).
        # PREVIOUSLY both axes were negated ("mirror fix"), which is actually
        # a 180° rotation — it preserves turn handedness (a right-hander
        # still traces as a right-hander) but rotates the whole track
        # layout, which is why the overall shape looked right but start/
        # finish orientation and turn-by-turn direction read backwards
        # against a reference map. A true mirror flips exactly ONE axis
        # and DOES reverse handedness — negating both is never a mirror.
        # Verified against this session's own telemetry: negating both (or
        # neither) axis gives an IDENTICAL signed-curvature sign for the
        # early part of the lap, proving the old code was a rotation, not
        # a mirror, and could never have fixed a handedness problem.
        # Road Atlanta's Turn 1 is a documented right-hander; only
        # negating longitude (not latitude) reproduces that correctly.
        pos_x = [(lo - centre_lon) * R * cos_lat for lo in lon_raw]
        pos_z = [-(la - centre_lat) * R           for la in lat_raw]
        pos_hz = 10.0
    else:
        # Pos X/Z fallback (recorder v1, no GPS reprojection available).
        # Apply the SAME single-axis convention as the GPS path above for
        # consistency between the two position sources.
        raw_x  = _col("Pos X")
        pos_x  = list(raw_x)
        pos_z  = _col("Pos Z")
        pos_hz = 100.0

    if not pos_x:
        con.close()
        raise HTTPException(422, "No position data found in file")

    # ── 2. Master clock ───────────────────────────────────────────────────────
    # GPS Time is the session-relative clock at 100Hz (synthesized in recorder
    # v2, native in LMU files). Used to slice laps from multi-lap files.
    gps_time = _col("GPS Time") if "GPS Time" in tables else []

    # Game time of every position row, and where each one falls in the
    # 100 Hz rows (fractional index). BUG FIX: everything below used to map
    # the two by index fraction (row i of M  <->  row i*N/M of N), which is
    # only true if no game tick was ever skipped. The 10/14 Fuji HF file
    # skipped ~18 s of ticks in lap 0, so lap 11's position ran 7.8 s ahead
    # of its speed/brake/throttle (S/F label and brake points far down the
    # straight, body slip refused). See telemetry.session_time_axes.
    t_fast, t_slow = session_time_axes(con)
    t_pos = None
    if t_fast is not None:
        t_pos = t_slow if use_gps else t_fast
        if t_pos is not None and len(t_pos) != len(pos_x):
            t_pos = None
    native_x = None   # fractional 100 Hz row index of each position row
    if t_pos is not None:
        native_x = np.interp(t_pos, t_fast, np.arange(len(t_fast), dtype=np.float64))

    # ── 3. Lap boundaries ─────────────────────────────────────────────────────
    # Lap table: (ts, value) where value = lap number. In recorder v2 / LMU
    # files there may be multiple rows; recorder v1 has exactly one row.
    try:
        lap_events = con.execute(
            'SELECT ts, value FROM "Lap" ORDER BY ts'
        ).fetchall()
    except Exception:
        lap_events = []

    # Auto-select fastest complete lap when none specified
    if lap_number is None and len(lap_events) >= 2:
        candidates = []
        for i in range(len(lap_events) - 1):
            dur = lap_events[i+1][0] - lap_events[i][0]
            if 30 < dur < 600:
                candidates.append((dur, int(lap_events[i][1])))
        if candidates:
            lap_number = min(candidates)[1]

    def _lap_slice() -> tuple[int, int]:
        """Return (start_idx, end_idx) into pos_x/pos_z for the chosen lap."""
        if len(lap_events) < 2:
            # Single-lap recorder file — use the whole array
            return 0, len(pos_x) - 1

        target = [(ts, v) for ts, v in lap_events
                  if lap_number is None or int(v) == lap_number]
        if not target:
            return 0, len(pos_x) - 1
        ts_start = target[0][0]
        later    = [ts for ts, _ in lap_events if ts > ts_start]
        ts_end   = later[0] if later else (gps_time[-1] if gps_time else ts_start + 600)

        if t_pos is not None:
            s = int(np.searchsorted(t_pos, ts_start, side="left"))
            e = int(np.searchsorted(t_pos, ts_end, side="left"))
        elif gps_time:
            s = next((i for i, t in enumerate(gps_time) if t >= ts_start), 0)
            e = next((i for i, t in enumerate(gps_time) if t >= ts_end),   len(gps_time) - 1)
            # GPS Time and pos_x may have different lengths if pos is at higher Hz
            # Scale indices to pos array length
            if len(gps_time) != len(pos_x):
                ratio = len(pos_x) / max(len(gps_time), 1)
                s = int(s * ratio)
                e = int(e * ratio)
        else:
            s = int(ts_start * pos_hz)
            e = int(ts_end   * pos_hz)
        return max(0, s), min(e, len(pos_x) - 1)

    s_idx, e_idx = _lap_slice()
    px_lap = pos_x[s_idx : e_idx + 1]
    pz_lap = pos_z[s_idx : e_idx + 1]
    lap_len = len(px_lap)

    if lap_len < 10:
        con.close()
        raise HTTPException(422, f"Lap slice too short ({lap_len} pts) — file may be partial")

    # ── 4. Track surface geometry ─────────────────────────────────────────────
    # Path Lateral = car's lateral offset from centreline, metres (positive = right)
    # Track Edge   = half-width of track on the car's side, metres (signed same as PathLat)
    # Both at 10Hz; interpolated to match pos array length for edge reconstruction.
    path_lat_raw   = _col("Path Lateral")
    track_edge_raw = _col("Track Edge")

    has_edges = bool(path_lat_raw) and bool(track_edge_raw)

    def _interp_to_length(arr: list, target: int) -> list:
        if not arr:
            return [0.0] * target
        if len(arr) == target:
            return arr
        src = np.array(arr, dtype=np.float64)
        if native_x is not None and target == len(native_x) and len(src) == len(t_fast):
            # A 100 Hz channel onto the position rows, by game time.
            return np.interp(native_x, np.arange(len(src)), src).tolist()
        return np.interp(
            np.linspace(0, len(src) - 1, target),
            np.arange(len(src)), src
        ).tolist()

    full_len   = len(pos_x)
    path_lat   = _interp_to_length(path_lat_raw,   full_len)[s_idx : e_idx + 1]
    track_edge = _interp_to_length(track_edge_raw, full_len)[s_idx : e_idx + 1]

    # Build left/right edge points in the same coordinate space as pos_x/pos_z
    left_x, left_z, right_x, right_z = [], [], [], []
    for i in range(lap_len):
        i1  = max(0, i - 4)
        i2  = min(lap_len - 1, i + 4)
        dx  = px_lap[i2] - px_lap[i1]
        dz  = pz_lap[i2] - pz_lap[i1]
        mag = math.sqrt(dx * dx + dz * dz) or 1e-9
        # Normal 90° to the RIGHT of travel in screen-y-down convention.
        # Screen space: y increases downward, so 90° right of (dx,dz) = (dz, -dx).
        # This puts the "left" edge (inner) correctly to the left of travel direction.
        nx, nz = dz / mag, -dx / mag

        pl = path_lat[i]
        te = abs(track_edge[i]) if has_edges else 7.5   # fallback 7.5m half-width

        left_x.append( px_lap[i] + nx * (te + pl))
        left_z.append( pz_lap[i] + nz * (te + pl))
        right_x.append(px_lap[i] - nx * (te - pl))
        right_z.append(pz_lap[i] - nz * (te - pl))

    # ── 5. Driven line channels ───────────────────────────────────────────────
    def _interp_lap(raw: list) -> list:
        """Interpolate a full-session array, then slice to this lap."""
        if not raw:
            return [0.0] * lap_len
        full = _interp_to_length(raw, full_len)
        return full[s_idx : e_idx + 1]

    throttle = _interp_lap(_col("Throttle Pos"))
    brake    = _interp_lap(_col("Brake Pos"))
    # Yaw Rate: low-pass filtered BEFORE slicing to the lap window, not
    # after. Filtering only reflects genuine dynamics below ~8Hz (see
    # YAW_SLIP_FILTER_CUTOFF_HZ docstring), but filtfilt needs context
    # outside the slice to avoid edge artefacts right at the lap
    # boundaries — which is exactly corner entry/exit at start/finish,
    # the part most worth getting right. Source rate here is 100Hz
    # (Yaw Rate's own native recording rate), not full_len/pos_hz, which
    # can be 10Hz when GPS Lat/Lon drives position — filtering at the
    # wrong assumed rate would apply the wrong cutoff.
    yaw_raw_full = _col("Yaw Rate")
    yaw_filtered_full = _lowpass_filter(yaw_raw_full, sample_hz=100.0) if yaw_raw_full else []
    yaw      = _interp_lap(yaw_filtered_full)
    gflat    = _interp_lap(_col("G Force Lat"))
    glong    = _interp_lap(_col("G Force Long"))
    steering = _interp_lap(_col("Steering Pos"))
    # Ground Speed (100Hz, km/h) is preferred; fall back to GPS Speed
    # (10Hz, native LMU units = m/s) if Ground Speed is missing or empty —
    # e.g. a file where that channel failed to record. GPS Speed must be
    # converted to km/h (×3.6) so the two sources are never silently
    # mixed-unit on the same chart.
    ground_speed_raw = _col("Ground Speed")
    if ground_speed_raw:
        speed = _interp_lap(ground_speed_raw)   # km/h
    else:
        gps_speed_raw = _col("GPS Speed")       # m/s
        speed = _interp_lap([v * 3.6 for v in gps_speed_raw]) if gps_speed_raw else []

    # ABS/TC: sparse (ts, value) event logs, expanded to a per-sample 0/1
    # step function aligned to gps_time (see _event_active_col), then
    # sliced/resampled to the lap window the same way every other channel
    # in this endpoint is — via _interp_lap, which itself calls
    # _interp_to_length(..., full_len)[s_idx:e_idx+1]. Interpolating a
    # 0/1 step array with linear interpolation does blur the exact
    # transition sample slightly (a 0→1 edge briefly reads ~0.5 for one
    # output point), but that only matters at the display's own pixel
    # resolution — negligible here, and far simpler than a second
    # step-aware resampler. Empty list (table missing) flows through
    # cleanly since _interp_lap/_interp_to_length both handle [] input.
    abs_active_full = _event_active_col("ABS", gps_time) if gps_time else []
    tc_active_full  = _event_active_col("TC",  gps_time) if gps_time else []
    abs_active = _interp_lap(abs_active_full)
    tc_active  = _interp_lap(tc_active_full)

    # Throttle/Brake from recorder v2 are 0-100%; normalise to 0-1 for colour math
    thr_max = max(throttle) if throttle else 1.0
    brk_max = max(brake)    if brake    else 1.0
    if thr_max > 1.5:   # clearly in percent
        throttle = [v / 100.0 for v in throttle]
    if brk_max > 1.5:
        brake    = [v / 100.0 for v in brake]

    # ── 6. Balance metric ─────────────────────────────────────────────────────
    # Primary: front axle suspension deflection − rear axle deflection,
    # baseline-corrected per wheel before combining (see below).
    #   Positive → front compressing MORE than its own baseline → understeer loading.
    #   Negative → rear compressing MORE than its own baseline → oversteer tendency.
    # Fallback: yaw_rate × speed (dimensionless lateral proxy).
    #
    # BUG FIX: this used to combine raw absolute Susp Pos values directly
    # (front_sum - rear_sum) with no baseline correction. That's wrong
    # whenever the car has any static rake (rear ride height ≠ front ride
    # height — the normal case for most of these cars), because rake alone
    # makes rear suspension travel numerically larger than front travel at
    # ALL times, including in a straight line at rest. The result: the
    # balance metric read essentially 100% oversteer for an entire lap,
    # which is physically impossible — it was measuring static rake, not
    # dynamic cornering behaviour. Confirmed on a real file: FL/FR mean
    # ~0.060/0.057, RL/RR mean ~0.099/0.094 — rear is structurally ~0.04m
    # larger than front even averaged over a whole lap, swamping any real
    # cornering-induced signal by roughly two orders of magnitude.
    #
    # Fix: subtract each wheel's OWN session-mean baseline first, so what
    # remains is the DYNAMIC deviation from that wheel's normal position —
    # i.e. "how much MORE compressed than usual is this corner right now",
    # which is what load transfer during cornering actually looks like.
    # Verified on the same file: this produces a 38% US / 60% OS / 1%
    # neutral split across the lap instead of ~100% OS, and the balance
    # trace's mean is now ~0 by construction (baseline-subtraction always
    # zeroes the session mean), which is the physically correct behaviour
    # for a metric meant to show relative loading, not absolute position.
    susp_raw = _cols4("Susp Pos")
    if susp_raw:
        susp_arr    = np.array(susp_raw, dtype=np.float64)   # (N_full, 4) FL FR RL RR
        susp_baseline = susp_arr.mean(axis=0)                 # per-wheel static offset
        susp_delta  = susp_arr - susp_baseline
        balance_raw = (susp_delta[:, 0] + susp_delta[:, 1]
                       - susp_delta[:, 2] - susp_delta[:, 3]).tolist()
        balance = balance_raw[s_idx : e_idx + 1]
        if len(balance) != lap_len:
            balance = _interp_to_length(balance_raw, full_len)[s_idx : e_idx + 1]
    else:
        speed_raw = _col("Ground Speed")
        speed_lap = _interp_lap(speed_raw)
        # km/h → m/s
        balance = [y * (s / 3.6) for y, s in zip(yaw, speed_lap)]

    # ── 6b. Per-corner suspension and slip ratio ──────────────────────────────
    susp_fl = susp_fr = susp_rl = susp_rr = []
    slip_fl = slip_fr = slip_rl = slip_rr = []

    if susp_raw:
        def _susp_corner(col_idx):
            raw = [row[col_idx] for row in susp_raw]
            return _interp_to_length(raw, full_len)[s_idx : e_idx + 1]
        susp_fl = _susp_corner(0)
        susp_fr = _susp_corner(1)
        susp_rl = _susp_corner(2)
        susp_rr = _susp_corner(3)

    # Slip ratio per wheel for lockup detection.
    # Wheel Speed in native LMU = m/s; in recorder = rad/s * R_tyre.
    # slip < -0.15 = locked wheel.
    #
    # IMPORTANT — full_len here is NOT a fixed sample rate. It's len(pos_x),
    # which is 10Hz-length when GPS Lat/Lon drive position (this file) or
    # 100Hz-length when Pos X/Z do (recorder v1 fallback). Ground Speed and
    # Wheel Speed are always logged at 100Hz regardless of which position
    # source is active, so BOTH must be interpolated down/up to full_len
    # before any elementwise math — never assumed to already match it.
    # This is what broke: v_ms_full was built from `speed`, which is
    # lap-sliced (via _interp_lap) rather than full-session-length, while
    # ws_arr came from the raw 100Hz table untouched — two different
    # lengths, neither of which was full_len. Fix: interpolate both raw
    # columns to full_len explicitly, then slice to the lap window last.
    ws_raw = _cols4("Wheel Speed")
    speed_raw_full = _col("Ground Speed")
    if ws_raw and speed_raw_full:
        ws_arr = np.array(
            [_interp_to_length([row[c] for row in ws_raw], full_len)
             for c in range(4)],
            dtype=np.float64,
        ).T  # (full_len, 4)
        v_ms_full = np.array(_interp_to_length(speed_raw_full, full_len),
                              dtype=np.float64) / 3.6

        # Detect rad/s vs m/s: recorder wheel speed is large (>50) at driving speed;
        # native LMU is ~vehicle_speed/3.6 (small). Heuristic: max > 50 = rad/s.
        ws_max = float(np.abs(ws_arr).max())
        R_TYRE = 0.33  # metres
        is_radps = ws_max > 50.0

        def _slip_ratio_col(col_idx):
            ws_col = np.abs(ws_arr[:, col_idx])
            if is_radps:
                ws_ms = ws_col * R_TYRE
            else:
                ws_ms = ws_col  # already m/s
            # ws_ms and v_ms_full are both full_len now — the boolean mask
            # and the arrays it indexes are always the same shape.
            v = v_ms_full
            sr = np.zeros(len(ws_ms))
            safe = v > 2.0
            sr[safe] = (ws_ms[safe] - v[safe]) / v[safe]
            return sr.tolist()[s_idx : e_idx + 1]

        slip_fl = _slip_ratio_col(0)
        slip_fr = _slip_ratio_col(1)
        slip_rl = _slip_ratio_col(2)
        slip_rr = _slip_ratio_col(3)

    # ── 6c. Heading angle per sample (canvas radians) ───────────────────────
    # Computed from consecutive position derivative (velocity direction).
    # canvas angle 0 = pointing right; positive = clockwise.
    # Use a small lookahead window to smooth noise.
    # Also compute front/rear slip angle difference for OS/US arrow colour.
    W_HEAD = max(2, int(round(0.2 * pos_hz)))  # ~0.2 s half-window
    heading = []
    for i in range(lap_len):
        i1 = max(0, i - W_HEAD)
        i2 = min(lap_len - 1, i + W_HEAD)
        dx = px_lap[i2] - px_lap[i1]
        dz = pz_lap[i2] - pz_lap[i1]  # already screen-Z (negated in projection)
        heading.append(math.atan2(dz, dx))  # canvas angle: 0=right, positive=CW

    # Per-wheel signed slip angle (deg) for the tyre gauges' direction dial.
    sa_raw = _cols4("Slip Angle")
    wheel_slip_angle = _wheel_slip_angles_deg(
        sa_raw, _col("Ground Speed"), s_idx, e_idx, full_len, native_x=native_x)

    # Front vs rear slip angle delta → US/OS (chevron colour + balance bars).
    # Positive = front slipping more into the turn = understeer; negative =
    # oversteer. PREVIOUSLY |front| - |rear| on every sample: straights,
    # kerbs and direction changes all coloured the chevron, and the UI's
    # US/Neu/OS bars came from the suspension delta instead (rake, not
    # balance). Now one definition for both, gated to real cornering and
    # checked against yaw rate - see vehicle_dynamics.handling_balance.
    # Built from the per-wheel angles above, which are already low-passed
    # at 100 Hz and zeroed at low speed.
    slip_delta = []  # DEGREES
    handling = None
    if wheel_slip_angle["fl"] and len(wheel_slip_angle["fl"]) == lap_len and speed:
        wsa = {k: np.asarray(v) for k, v in wheel_slip_angle.items()}
        hb = handling_balance((wsa["fl"] + wsa["fr"]) / 2, (wsa["rl"] + wsa["rr"]) / 2,
                              yaw, gflat, speed)
        slip_delta = hb.pop("delta_deg")
        handling = hb

    # Body slip β (deg): chassis orientation vs direction of travel.
    # Measured from Local Vel X/Z when the file has them (recorder_hf.py),
    # otherwise reconstructed from the yaw-rate integral.
    body_slip = []
    lv_x = _col("Local Vel X") if "Local Vel X" in tables else []
    if lv_x and native_x is not None and len(lv_x) == len(t_fast):
        body_slip = body_slip_from_local_vel_deg(
            lv_x, _col("Local Vel Z"), native_x[s_idx : e_idx + 1],
            speed_kmh=speed if speed else [0.0] * lap_len)
    if not body_slip:
        body_slip = compute_body_slip_deg(
            yaw_raw_full, gps_time, s_idx, e_idx, full_len,
            course_rad=heading, speed_kmh=speed if speed else [0.0] * lap_len,
            yaw_lap=yaw,
            native_x_lap=native_x[s_idx : e_idx + 1] if native_x is not None else None)

    # ── 6d. Corner labels from track_db ──────────────────────────────────────
    # Match ref_points (dist in metres) to driven-line positions using Lap Dist.
    corner_labels = []  # [{x, z, label, dist}]
    try:
        # Get track metadata for lookup
        meta_rows = {r[0]: r[1] for r in con.execute('SELECT key, value FROM metadata').fetchall()}             if "metadata" in tables else {}
        track_name_meta = meta_rows.get("TrackName", "")

        if not track_name_meta and "GPS Time" in tables:
            # Try from channelsList or other sources
            pass

        if track_name_meta:
            track_key, track_data = find_track_in_registry(track_name_meta)
            if track_data:
                layout_name_meta = meta_rows.get("TrackLayout", meta_rows.get("Layout", "Default"))
                _, layout_data = find_layout_in_track(track_data, layout_name_meta, track_name_meta)
                if not layout_data:
                    # fallback to first layout
                    layouts = track_data.get("layouts", {})
                    layout_data = next(iter(layouts.values()), None) if layouts else None

                if layout_data:
                    ref_pts = layout_data.get("ref_points", [])
                    # Get lap dist array at line resolution
                    lapdist_raw = _col("Lap Dist") or _col("Lap Dist Integrated")
                    if lapdist_raw and ref_pts:
                        lapdist_interp = _interp_to_length(lapdist_raw, full_len)[s_idx:e_idx+1]
                        max_dist = max(lapdist_interp) if lapdist_interp else 0
                        for rp in ref_pts:
                            target_dist = rp.get("dist", 0)
                            label = rp.get("corner", "")
                            if not label or target_dist > max_dist * 1.05:
                                continue
                            # Find closest index in lapdist_interp
                            best_i = min(range(len(lapdist_interp)),
                                         key=lambda i: abs(lapdist_interp[i] - target_dist))
                            if best_i < len(px_lap):
                                corner_labels.append({
                                    "x": px_lap[best_i],
                                    "z": pz_lap[best_i],
                                    "label": label,
                                    "dist": target_dist,
                                })
    except Exception as _e:
        pass  # corner labels are non-critical

    # ── 7. Downsample for wire transfer ───────────────────────────────────────
    def _ds(arr: list, n: int) -> list:
        if len(arr) <= n:
            return arr
        step = len(arr) / n
        return [arr[int(i * step)] for i in range(n)]

    EDGE_PTS = 600
    LINE_PTS = 2000

    line_idx = [int(i * (lap_len - 1) / (LINE_PTS - 1)) for i in range(LINE_PTS)] \
               if lap_len >= LINE_PTS else list(range(lap_len))

    con.close()

    return {
        "coordinate_system": "gps_projected" if use_gps else "world_metres",
        "lap_number":  lap_number,
        "has_edges":   has_edges,
        "lap_len_pts": lap_len,
        "track": {
            "left_x":  _ds(left_x,  EDGE_PTS),
            "left_z":  _ds(left_z,  EDGE_PTS),
            "right_x": _ds(right_x, EDGE_PTS),
            "right_z": _ds(right_z, EDGE_PTS),
        },
        "driven_line": {
            "x":        [px_lap[i]   for i in line_idx],
            "z":        [pz_lap[i]   for i in line_idx],
            "throttle": [throttle[i] for i in line_idx],
            "brake":    [brake[i]    for i in line_idx],
            "balance":  [balance[i]  for i in line_idx],
            "yaw":      [yaw[i]      for i in line_idx],
            "gflat":    [gflat[i]    for i in line_idx],
            "glong":    [glong[i]    for i in line_idx],
            "steering": [steering[i] for i in line_idx],
            "speed":    [speed[i]    for i in line_idx] if speed else [],
            "heading":  [heading[i]  for i in line_idx],
            "slip_delta_deg": [slip_delta[i] if slip_delta else 0.0 for i in line_idx],
            "body_slip_deg": [body_slip[i] for i in line_idx] if body_slip else [],
            "abs_active": [abs_active[i] for i in line_idx] if abs_active else [],
            "tc_active":  [tc_active[i]  for i in line_idx] if tc_active  else [],
        },
        "susp": {
            "fl": _ds(susp_fl, LINE_PTS) if susp_fl else [],
            "fr": _ds(susp_fr, LINE_PTS) if susp_fr else [],
            "rl": _ds(susp_rl, LINE_PTS) if susp_rl else [],
            "rr": _ds(susp_rr, LINE_PTS) if susp_rr else [],
        },
        "slip": {
            "fl": _ds(slip_fl, LINE_PTS) if slip_fl else [],
            "fr": _ds(slip_fr, LINE_PTS) if slip_fr else [],
            "rl": _ds(slip_rl, LINE_PTS) if slip_rl else [],
            "rr": _ds(slip_rr, LINE_PTS) if slip_rr else [],
            "lockup_threshold": -0.15,
        },
        # Signed degrees, positive = tyre travelling right of where it
        # points. Same downsampling as "slip" so one index reads both.
        "slip_angle": {k: _ds(v, LINE_PTS) for k, v in wheel_slip_angle.items()},
        # US/neutral/OS shares of cornering samples (None without Slip Angle).
        "handling": handling,
        # Higher-res positions for car dot animation
        "car_pos_x": _ds(px_lap, 4000),
        "car_pos_z": _ds(pz_lap, 4000),
        "corner_labels": corner_labels,
    }
