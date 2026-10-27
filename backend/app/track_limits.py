"""
Car class / dimensions and track limits for the chase view's car model.

The car's dimensions come from static/car_models.json, the one source the UI
(car_models.js) and the STL exporter also read, so the wheels the backend
judges are the wheels the UI draws.

Track limits (LMU's rule: the white line is track; a wheel is off when its
whole tyre is beyond it): per driven-line row, each wheel's lateral position
in the centre-path frame (right of travel positive) is

    lat_wheel = lat + a * sin(psi) + w * cos(psi)

with (a, w) the wheel's offset from the car's reference point (along,
right-positive), lat the car's lateral offset, and psi the body's angle to
the path: the direction of travel against the path (d lat / d s, from Lap
Dist) plus body slip beta (nose turned right of the velocity = positive,
the /trackmap body_slip_deg convention). A wheel is out when the inner edge
of its tyre (tyre_width / 2 inside its centre) is beyond the edge line.

The edge is LMU's Track Edge: the half-width on the car's side only, so the
other side is assumed symmetric. Approximate, and labelled as such.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .car_identity import identify_car

CAR_MODELS_JSON = Path(__file__).parent / "static" / "car_models.json"
WHEELS = ("FL", "FR", "RL", "RR")            # bit 0..3 of wheels_out

# Body angle to the path from d lat / d s over +-PATH_ANGLE_ROWS rows; below
# MIN_DS_M of lap distance (standing, pit stop) the direction is not
# measurable and only beta counts. Clipped: a spin is not a track-limits case.
PATH_ANGLE_ROWS = 2
MIN_DS_M = 1.0
MAX_PATH_ANGLE_DEG = 45.0

NOTE_OK = "Track edge from LMU on the car's side, assumed symmetric: approximate."
NOTE_NO_EDGES = "No Track Edge / Path Lateral in this file: track limits unavailable."
NOTE_FAILED = "Track limits could not be worked out for this lap."


@lru_cache(maxsize=1)
def load_car_models() -> dict:
    return json.loads(CAR_MODELS_JSON.read_text(encoding="utf-8"))


def class_dims(models: dict, car_class: str | None) -> tuple[str, dict, str]:
    """(class, dims, source): the same rule as car_models.js classDims, so
    the UI and the backend pick the same car. Unknown -> LMGT3, 'fallback'."""
    classes = (models or {}).get("classes", {})
    if car_class and car_class in classes:
        return car_class, classes[car_class], "exact"
    alias = (models or {}).get("aliases", {}).get(car_class) if car_class else None
    if alias and alias in classes:
        return alias, classes[alias], "alias"
    return "LMGT3", classes["LMGT3"], "fallback"


def car_payload(meta: dict, car_name: str) -> dict:
    """/trackmap `car`: {class, source, requested, model, dims}. requested is
    the class car_identity found ("unknown" when none)."""
    ident = identify_car(meta or {}, car_name or "")
    cls, dims, source = class_dims(load_car_models(), ident.car_class)
    return {"class": cls, "source": source, "requested": ident.car_class,
            "model": ident.model, "dims": dims}


def wheel_offsets(dims: dict) -> list[tuple[float, float, float]]:
    """(along, right, tyre width) per wheel FL, FR, RL, RR: the reference
    point is mid-wheelbase on the centre line, as in car_models.js."""
    a = float(dims["wheelbase"]) / 2
    tf, tr = float(dims["track_front"]) / 2, float(dims["track_rear"]) / 2
    wf, wr = float(dims["tyre_width_front"]), float(dims["tyre_width_rear"])
    return [(a, -tf, wf), (a, tf, wf), (-a, -tr, wr), (-a, tr, wr)]


def wheels_out(lat, edge_half, angle_deg, dims: dict) -> np.ndarray:
    """4-bit mask per row (bit 0 FL .. bit 3 RR): the wheel's whole tyre is
    beyond the edge line on either side."""
    lat = np.asarray(lat, dtype=float)
    e = np.abs(np.asarray(edge_half, dtype=float))
    psi = np.radians(np.asarray(angle_deg, dtype=float))
    s, c = np.sin(psi), np.cos(psi)
    mask = np.zeros(len(lat), dtype=int)
    for bit, (a, w, tw) in enumerate(wheel_offsets(dims)):
        wl = lat + a * s + w * c
        out = (wl - tw / 2 > e) | (wl + tw / 2 < -e)
        mask |= np.where(np.isfinite(wl) & out, 1 << bit, 0)
    return mask


def path_angle_deg(lat, s) -> np.ndarray:
    """Direction of travel against the centre path (deg, + = heading to the
    right of it) from d lat / d s over a few rows. 0 where Lap Dist does not
    advance enough or jumps (start/finish wrap, reset)."""
    lat = np.asarray(lat, dtype=float)
    s = np.asarray(s, dtype=float)
    n, k = len(lat), PATH_ANGLE_ROWS
    out = np.zeros(n)
    if n < 2 * k + 1:
        return out
    i0 = np.clip(np.arange(n) - k, 0, n - 1)
    i1 = np.clip(np.arange(n) + k, 0, n - 1)
    ds, dl = s[i1] - s[i0], lat[i1] - lat[i0]
    ok = np.isfinite(ds) & np.isfinite(dl) & (ds > MIN_DS_M) & (ds < 200.0)
    out[ok] = np.degrees(np.arctan2(dl[ok], ds[ok]))
    return np.clip(out, -MAX_PATH_ANGLE_DEG, MAX_PATH_ANGLE_DEG)


def track_limits_payload(*, has_edges: bool, path_lat, track_edge, lateral_sign: int,
                         s_lap, body_slip_deg, dims: dict) -> dict:
    """Lap-length arrays {lat, edge_half, wheels_out} plus {available, note}.
    Without Track Edge / Path Lateral the arrays are empty (not zeros: a
    zero would read as "on track")."""
    pl = np.asarray(path_lat, dtype=float)
    te = np.asarray(track_edge, dtype=float)
    if not has_edges or len(pl) == 0 or len(te) != len(pl) or not np.any(np.abs(te) > 0):
        return {"available": False, "note": NOTE_NO_EDGES, "lat": [], "edge_half": [], "wheels_out": []}
    lat = int(lateral_sign) * pl
    ang = path_angle_deg(lat, s_lap) if s_lap is not None and len(s_lap) == len(lat) else np.zeros(len(lat))
    if body_slip_deg is not None and len(body_slip_deg) == len(lat):
        ang = ang + np.nan_to_num(np.asarray(body_slip_deg, dtype=float))
    edge = np.abs(te)
    return {"available": True, "note": NOTE_OK, "lat": lat, "edge_half": edge,
            "wheels_out": wheels_out(lat, edge, ang, dims)}
