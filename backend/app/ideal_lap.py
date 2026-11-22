"""Ideal lap, orchestration (design 2026-11-10-ideal-lap-design.md §5.4-5.6, §7).

From a recording and a lap to the /ideal-lap response: the fitted car of the
lap's group (ideal_store, trimmed per file), the reference lap's line with the
saturation clamp, three grip levels (envelope with the driven lap), the
unclamped plain model, the ceiling, grip used per axle, the corner table, the
trace on /trackmap's driven-line points, and the public lap-time line.

`analyse_lap` is the core and needs only a fitted car and one lap's raw arrays,
so the validation tool (backend/tools/ideal_lap_report.py) runs it on held-out
fits; `build_ideal_lap` adds loading, the store, lap choice and the cache.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import ideal_solver as SV
from .ideal_car import AxleFit, AXLES, is_extrapolated
from .ideal_state import G, GRID_HZ, lap_raw, lap_state, _ensure_wheel_channels

CORNER_AT_LIMIT_S = 0.02            # §5.6 verdict
ASSUMED_BINDS_SHARE = 0.10          # §6.3: an assumed part that binds on more of the lap -> low
PUBLIC_MARGIN = 0.995               # §14 R9: ideal faster than public best x 0.995 -> flag
FALLBACK_CURV = 1 / 300.0           # corner windows without a track reference: radius < 300 m
_CACHE: dict = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def _mono(ld: np.ndarray, L: float) -> np.ndarray:
    """Game Lap Dist made monotonic over one lap (ideal_solver.unwrap_lap_dist):
    the first rows of a lap can still read the previous lap's end (negative
    distances), and the last rows the next lap's start (L + d). The grid's
    lap_dist is already unwrapped by line_of; unwrapping again changes nothing."""
    return SV.unwrap_lap_dist(ld, L)


def _nearest_index(p_ld: np.ndarray, ld: np.ndarray) -> np.ndarray:
    """Index of the grid point NEAREST to each lap distance.
    PREVIOUSLY searchsorted took the next grid point, up to 2 m off, for the
    trace's binding / saturated_ref (review minor 3)."""
    k = np.rint(np.interp(p_ld, ld, np.arange(len(ld), dtype=float)))
    return np.clip(k.astype(int), 0, len(ld) - 1)


def _trace_t(p_ld, p_t, ld, dt_ref, dt_i, ds) -> np.ndarray:
    """Ideal game-clock time at the driven line's points: the driven time minus
    the gain accumulated up to there.
    PREVIOUSLY the cumulative times stopped at the last grid point, so a point on
    the line lost the last step's gain and t ended one 2 m step short of
    ideal.time_s (review minor 8). Fix: the closing step is part of the cumulative
    times (at the last grid point + ds)."""
    ld_ext = np.append(ld, ld[-1] + ds)
    cum_ref = np.concatenate([[0.0], np.cumsum(dt_ref)])
    cum_i = np.concatenate([[0.0], np.cumsum(dt_i)])
    return np.asarray(p_t, float) - np.interp(p_ld, ld_ext, cum_ref - cum_i)


def _axle_fits(info: dict) -> list:
    names = {f.name for f in fields(AxleFit)}
    return [AxleFit(**{k: v for k, v in info["axles"][a].items() if k in names}) for a in AXLES]


def _kinematic_gxt(car, ln, v):
    gx = (np.roll(v, -1) ** 2 - v ** 2) / (2 * G * ln["ds"])
    return gx + ln["grade"] + car.drag(v)


def _corner_windows(track_name: str, lap_len: float | None, ln: dict, L: float) -> tuple[list, str]:
    """[(n, name, s0, s1)] in game Lap Dist, from the track reference; without
    one, runs of path curvature tighter than 300 m radius (padded 40 m)."""
    try:
        from .track_reference import load_reference
        ref = load_reference(track_name, lap_len) if track_name else None
    except Exception:
        ref = None
    if ref:
        cs = sorted((c for c in ref["layout"].get("corners", []) if c.get("lmu_apex_m") is not None),
                    key=lambda c: c["lmu_apex_m"])
        out = [(c.get("number") or i + 1, c.get("display") or c.get("name") or f"C{i + 1}",
                float(c.get("lmu_start_m", c["lmu_apex_m"] - 20)), float(c.get("lmu_end_m", c["lmu_apex_m"] + 20)))
               for i, c in enumerate(cs)]
        if out:
            return out, "track_reference"
    ld = ln["lap_dist_mono"]
    on = np.abs(ln["ks"]) > FALLBACK_CURV
    out, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            if j - i >= 5:
                s0, s1 = ld[i] - 40.0, ld[j] + 40.0
                if out and s0 <= out[-1][3]:
                    out[-1] = (out[-1][0], out[-1][1], out[-1][2], s1)
                else:
                    out.append((len(out) + 1, f"C{len(out) + 1}", s0, s1))
            i = j + 1
        else:
            i += 1
    return out, "curvature"


def _window_mask(ld: np.ndarray, s0: float, s1: float, L: float) -> np.ndarray:
    if s1 >= s0:
        return (ld >= s0) & (ld <= s1)
    return (ld >= s0) | (ld <= s1)             # across the line


# ── the core ─────────────────────────────────────────────────────────────────

def analyse_lap(car, info: dict, raw: dict, lap_time_s: float, beta: float | None = None,
                track_name: str = "", lap_len: float | None = None, points: dict | None = None,
                clamp: bool = True) -> dict:
    """Everything of §5.4-5.6 for one lap with a fitted car. `info`: the store's
    fit info (axles with knee / saturation / v_sat, kappa statuses).
    `points`: {lap_dist[], t[]} of /trackmap's driven line (None: the 2 m grid)."""
    beta = info["beta"] if beta is None else beta
    st = lap_state(raw)
    axles = _axle_fits(info)
    sat, ax = (SV.saturated_ref(st, car, axles, per_axle=True) if clamp else (None, None))
    ln = SV.line_of(raw, sat_ref=sat, sat_axles=ax)
    ds = ln["ds"]
    lv = SV.levels(car, ln, beta)
    ceil = SV.solve(car, ln, "ceiling")
    conflict = SV.clamp_conflict(car, ln) if clamp else {a: {"n": 0, "conflicts": 0, "share": None} for a in AXLES}
    v_ref = ln["v"]
    t_ref = lv["t_ref_grid"]
    exp, mn = lv["exp"], lv["min"]
    v_i = exp["env"]["v"]
    from_ref = exp["env"]["from_ref"]
    code = np.where(from_ref, "driven", exp["solve"]["binding"]).astype(object)

    # lap distance of the grid (game Lap Dist when recorded)
    L = float(lap_len) if lap_len else float(ln["s"][-1] + ds)
    ln["lap_dist_mono"] = _mono(ln["lap_dist"], L) if ln.get("lap_dist") is not None else ln["s"]

    # grip used per axle: the ideal from its own profile, the driven lap from its state
    uf_i, ur_i = SV.usage_on_line(car, ln, v_i, _kinematic_gxt(car, ln, v_i))
    gxt_d = st.gx + st.grade + car.drag(st.v)
    uf_d, ur_d = SV.usage(car, st.v, gxt_d, st.gy, st.gz)
    s_raw = raw["s"]
    uf_r, ur_r = np.interp(ln["s"], s_raw, uf_d), np.interp(ln["s"], s_raw, ur_d)

    # times: official lap time - (T_ref_grid - T_x): the grid's timing error cancels
    shown = lambda t_grid: float(lap_time_s - (t_ref - t_grid))       # noqa: E731
    shares = {c: float(np.mean(code == c)) for c in SV.BINDING + ("driven",)}
    notes = list(info.get("notes", []))
    confidence = info.get("confidence", "medium")
    if (info.get("kappa_brake", {}).get("status") == "assumed" and shares["brake"] > ASSUMED_BINDS_SHARE) or \
       (info.get("kappa_trac", {}).get("status") == "assumed" and shares["traction"] > ASSUMED_BINDS_SHARE):
        confidence = "low"
        notes.append("an assumed braking / traction limit binds on more than 10 % of the lap")

    # corners
    windows, wsrc = _corner_windows(track_name, lap_len, ln, L)
    ld = ln["lap_dist_mono"]
    dt_ref, dt_i, dt_min = ds / v_ref, ds / v_i, ds / mn["env"]["v"]
    corners, covered = [], np.zeros(len(ld), bool)
    for n_c, name, s0, s1 in windows:
        m = _window_mask(ld, s0, s1, L)
        if m.sum() < 2:
            continue
        covered |= m
        idx = np.flatnonzero(m)
        k = idx[int(np.argmin(v_i[idx]))]
        gain = float(dt_ref[m].sum() - dt_i[m].sum())
        gmin = float(dt_ref[m].sum() - dt_min[m].sum())
        thirds = np.array_split(idx, 3)
        where = ("entry", "mid", "exit")[int(np.argmax([dt_ref[t].sum() - dt_i[t].sum() for t in thirds]))]
        b = str(code[k])
        extrap = False
        if b in AXLES:
            extrap = bool(is_extrapolated(axles[AXLES.index(b)], v_i[k]))
        corners.append({
            "kind": "corner", "n": n_c, "name": name, "s0": round(s0, 1), "s1": round(s1, 1),
            "t_ref": round(float(dt_ref[m].sum()), 3), "t_ideal": round(float(dt_i[m].sum()), 3),
            "gain_s": round(gain, 3), "gain_min_s": round(gmin, 3),
            "vmin_ref_kmh": round(float(v_ref[m].min() * 3.6), 1), "vmin_ideal_kmh": round(float(v_i[k] * 3.6), 1),
            "binding": b, "u_front_ref": round(float(uf_r[m].max()), 3), "u_rear_ref": round(float(ur_r[m].max()), 3),
            "verdict": "at_limit" if gmin <= CORNER_AT_LIMIT_S else "time_available",
            "where": None if gmin <= CORNER_AT_LIMIT_S else where,
            "measured_share": round(float(np.mean(ln["sat_ref"][m])) if ln.get("sat_ref") is not None else 0.0, 3),
            "extrapolated": extrap})
    # time outside the windows: straight rows between corners (in lap order)
    rest = ~covered
    if rest.any():
        runs = np.flatnonzero(np.diff(np.concatenate([[0], rest.astype(int), [0]])))
        for a, b_ in zip(runs[0::2], runs[1::2]):
            m = np.zeros(len(ld), bool)
            m[a:b_] = True
            corners.append({"kind": "straight", "n": None, "name": "straight", "s0": round(float(ld[a]), 1),
                            "s1": round(float(ld[b_ - 1]), 1), "t_ref": round(float(dt_ref[m].sum()), 3),
                            "t_ideal": round(float(dt_i[m].sum()), 3),
                            "gain_s": round(float(dt_ref[m].sum() - dt_i[m].sum()), 3),
                            "gain_min_s": round(float(dt_ref[m].sum() - dt_min[m].sum()), 3)})
    corners.sort(key=lambda c: c["s0"])
    extrap_gain = sum(c["gain_s"] for c in corners if c.get("extrapolated"))

    # trace at the driven line's points
    cum_ref = np.concatenate([[0.0], np.cumsum(dt_ref)])[:-1]
    if points and points.get("lap_dist"):
        p_ld = _mono(np.asarray(points["lap_dist"], float), L)
        p_t = np.asarray(points.get("t") or [], float)
        if len(p_t) != len(p_ld):
            p_t = np.interp(p_ld, ld, cum_ref)
    else:
        p_ld, p_t = ld, cum_ref
    at = lambda a: np.interp(p_ld, ld, a)                               # noqa: E731
    t_trace = _trace_t(p_ld, p_t, ld, dt_ref, dt_i, ds)
    near = _nearest_index(p_ld, ld)
    ideal_time = shown(exp["env"]["time"])
    trace = {"lap_dist": [round(float(x), 2) for x in p_ld], "v_kmh": [round(float(x) * 3.6, 2) for x in at(v_i)],
             "v_ref_kmh": [round(float(x) * 3.6, 2) for x in at(v_ref)], "t": [round(float(x), 3) for x in t_trace],
             "binding": [str(code[i]) for i in near],
             "saturated_ref": [bool(ln["sat_ref"][i]) if ln.get("sat_ref") is not None else False for i in near],
             "u_front": [round(float(x), 3) for x in at(uf_i)], "u_rear": [round(float(x), 3) for x in at(ur_i)],
             "u_front_ref": [round(float(x), 3) for x in at(uf_r)], "u_rear_ref": [round(float(x), 3) for x in at(ur_r)]}

    fit = dict(info)
    fit["axles"] = {a: dict(info["axles"][a], clamp_conflict=conflict[a]["share"], clamp_conflict_n=conflict[a]["n"])
                    for a in AXLES}
    fit["confidence"], fit["notes"], fit["beta"] = confidence, notes, beta
    return {
        "fit": fit,
        "ideal": {"time_s": round(ideal_time, 3), "gain_s": round(lv["gain"], 3), "gain_min_s": round(lv["gain_min"], 3),
                  "gain_max_s": round(lv["gain_max"], 3), "clamped_share": round(lv["clamped_share"], 4)},
        "ceiling": {"time_s": round(shown(ceil["time"]), 3), "gain_s": round(t_ref - ceil["time"], 3)},
        "model_only": {"time_s": round(shown(lv["model_only_time"]), 3), "under_driven_share": round(lv["under_driven_share"], 4)},
        "trace": trace, "corners": corners,
        "diagnostics": {"t_ref_grid": t_ref, "gain_unclamped_s": lv["gain_unclamped"], "binding_share": shares,
                        "corner_source": wsrc, "extrapolated_gain_s": round(extrap_gain, 3),
                        "grade_source": raw.get("grade_source")},
        "_levels": lv, "_line": ln, "_ceiling": ceil,
    }


def note_for(res: dict, lap_time_s: float) -> str:
    i = res["ideal"]
    return ("Ideal lap on your own line: at least %.2f s and expected %.2f s faster (%.3f s), with each axle "
            "at its grip limit where you were not; where your tyres were already at the limit it keeps your speed. "
            "Range %.2f-%.2f s from the grip level's uncertainty." % (
                i["gain_min_s"], i["gain_s"], i["time_s"], i["gain_min_s"], i["gain_max_s"]))


def public_block(car_model: str, car_class: str, slug: str | None, layout_len_m: float | None,
                 game_version: str | None, lap_time_s: float, ideal_s: float, ceiling_s: float) -> tuple:
    """(public, public_check) of §14 R9; (None, None) without a match or file."""
    try:
        from . import public_laptimes as PL
    except Exception:
        return None, None
    m = PL.match(slug, layout_len_m, car_model, car_class, game_version)
    if not m:
        return None, None
    best = m.get("car_best_s") or m.get("class_ref_s")
    if not best:
        return None, None
    pub = dict(m)
    pub["driven_pct"] = round(100 * lap_time_s / best, 2)
    pub["ideal_pct"] = round(100 * ideal_s / best, 2)
    pub["band"] = PL.band(pub["driven_pct"])
    check = "faster_than_public_best" if min(ideal_s, ceiling_s) < best * PUBLIC_MARGIN else "ok"
    return pub, check


def _public_strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ── the request ──────────────────────────────────────────────────────────────

def _pick_lap(laps: list, lap_number: int | None):
    from .telemetry import real_lap_flags
    if lap_number is not None:
        return next((l for l in laps if int(l.lap_number) == int(lap_number)), None)
    flags = real_lap_flags([l.lap_time_s for l in laps], [l.is_complete for l in laps])
    real = [l for l, f in zip(laps, flags) if f]
    return min(real, key=lambda l: l.lap_time_s) if real else None


def _refusal(reason: str, **kw) -> dict:
    out = {"status": "insufficient_data", "reason": reason, "car_model": None, "car_class": None, "track": None,
           "lap_number": None, "lap_time_s": None, "fit": None, "ideal": None, "ceiling": None, "model_only": None,
           "trace": None, "corners": None, "note": None, "public": None, "public_check": None}
    out.update(kw)
    return out


def build_ideal_lap(path: str | Path, lap_number: int | None = None, store=None,
                    points: dict | None = None) -> dict:
    """The /ideal-lap response (§7). Never raises for data reasons: a missing
    fit or lap is `insufficient_data` with the reason."""
    from .ideal_store import Refusal, default_store
    from . import track_topography as tt
    path = Path(path)
    store = store or default_store([path.parent])
    try:
        st = path.stat()
    except OSError as exc:
        return _refusal(f"recording unreadable: {exc}")
    got = store.car_for_lap(path, lap_number)
    if isinstance(got, Refusal):
        return _refusal(got.reason, lap_number=lap_number)
    car, info = got
    key = (str(path), st.st_mtime, st.st_size, lap_number, info.get("fingerprint"),
           len((points or {}).get("lap_dist") or []))
    if key in _CACHE:
        return _CACHE[key]
    _ensure_wheel_channels()
    from .telemetry import load_telemetry_file
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            laps = load_telemetry_file(path, grid_hz=GRID_HZ)
    except Exception as exc:
        return _refusal(f"recording unreadable: {type(exc).__name__}: {exc}")
    lap = _pick_lap(laps, lap_number)
    if lap is None or not lap.is_complete:
        return _refusal("no complete lap %s in this recording" % ("" if lap_number is None else lap_number),
                        lap_number=lap_number)
    layout = None
    lap_len = tt._lap_length_m(lap)
    if "Pos Y" not in lap.channels:
        hit = tt.load_topography(lap.track_name, lap_len)
        layout = hit[1] if hit else None
    raw = lap_raw(lap, layout)
    if raw.get("slip") is None and raw.get("gf") is None:
        return _refusal("no slip angles and no Grip Fraction on this lap", lap_number=int(lap.lap_number))
    try:
        res = analyse_lap(car, info, raw, float(lap.lap_time_s), track_name=lap.track_name, lap_len=lap_len,
                          points=points)
    except Exception as exc:                    # numerical failure on an odd lap: say so, never a 500
        return _refusal(f"ideal lap failed on this lap: {type(exc).__name__}: {exc}", lap_number=int(lap.lap_number))
    red = store._reduction(path)
    from .session_index import file_entry
    try:
        entry = file_entry(path, "sessions")
    except Exception:
        entry = {}
    pub, check = public_block(red.car_model, red.car_class, entry.get("track_slug"), entry.get("layout_len_m"),
                              red.game_version, float(lap.lap_time_s), res["ideal"]["time_s"], res["ceiling"]["time_s"])
    if check == "faster_than_public_best":
        res["fit"]["confidence"] = "low"
        res["fit"]["notes"].append("the ideal lap beats the public best for this car: almost surely a model error")
    out = {"status": "ok", "reason": None, "car_model": red.car_model, "car_class": red.car_class,
           "track": lap.track_name, "lap_number": int(lap.lap_number), "lap_time_s": round(float(lap.lap_time_s), 3)}
    out.update(_public_strip(res))
    out["note"] = note_for(res, float(lap.lap_time_s))
    out["public"], out["public_check"] = pub, check
    _CACHE[key] = out
    return out
