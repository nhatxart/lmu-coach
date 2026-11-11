"""Ideal-lap spike, stage 1: per-lap arrays from a recording (cached), read-only.
Conventions here: gx = longitudinal g, + = speeding up; gy = lateral g as recorded;
gz = vertical g as recorded; wheels FL, FR, RL, RR."""
import os, pickle, sys, hashlib
from pathlib import Path
import numpy as np

BACKEND = os.environ.get("IL_BACKEND", r"C:\Users\hoang\Projects\lmu-coach\backend")
SESS = os.environ.get("IL_SESSIONS", r"C:\Users\hoang\Projects\lmu-coach\data\sessions")
REF = os.environ.get("IL_REF", r"C:\Users\hoang\Projects\lmu-coach\data\track_reference")
CACHE = Path(os.environ.get("TEMP", ".")) / "il_cache"
CACHE.mkdir(exist_ok=True)
HZ = 50.0
G = 9.81
W4 = ("FL", "FR", "RL", "RR")
VERSION = 3


def _four(lap, name):
    ks = [f"{name} {w}" for w in W4]
    if not all(k in lap.channels for k in ks):
        return None
    a = np.vstack([np.nan_to_num(np.asarray(lap.channels[k], float)) for k in ks])
    return a if np.nanmax(np.abs(a)) > 1e-9 else None


def _one(lap, name, default=0.0):
    if name not in lap.channels:
        return None if default is None else np.full(len(lap.time_s), default)
    return np.nan_to_num(np.asarray(lap.channels[name], float))


def load_file(name: str) -> dict:
    """{meta, track, model, cls, laps:[{...arrays...}]}; laps complete and under 10 min."""
    path = Path(name) if os.path.isabs(name) else Path(SESS) / name
    key = hashlib.sha1(f"{path}|{path.stat().st_mtime}|{path.stat().st_size}|{VERSION}".encode()).hexdigest()[:16]
    cp = CACHE / f"{key}.pkl"
    if cp.exists():
        return pickle.loads(cp.read_bytes())
    os.environ.setdefault("TRACK_REFERENCE_DIR", REF)
    if BACKEND not in sys.path:
        sys.path.insert(0, BACKEND)
    import io, contextlib
    from app import telemetry
    from app.car_identity import identify_car
    for extra in ("Tire Longitudinal Force", "Grip Fraction"):
        if extra not in telemetry.WHEEL_CHANNELS:
            telemetry.WHEEL_CHANNELS.append(extra)
    with contextlib.redirect_stdout(io.StringIO()):
        laps = telemetry.load_telemetry_file(path, grid_hz=HZ)
    out = []
    for l in laps:
        if not l.is_complete or not (25 < l.lap_time_s < 600):
            continue
        t = np.asarray(l.time_s, float)
        v = _one(l, "Ground Speed") / 3.6
        thr, brk = _one(l, "Throttle Pos"), _one(l, "Brake Pos")
        if np.nanmax(thr) <= 1.5: thr = thr * 100
        if np.nanmax(brk) <= 1.5: brk = brk * 100
        s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
        d = dict(n=int(l.lap_number), time=float(l.lap_time_s), t=t, v=v, s=s,
                 gx=-_one(l, "G Force Long"), gy=_one(l, "G Force Lat"), gz=_one(l, "G Force Vert"),
                 thr=thr, brk=brk, abs=_one(l, "ABS"), tc=_one(l, "TC"), steer=_one(l, "Steering Pos"),
                 yaw=_one(l, "Yaw Rate"), yawacc=_one(l, "Yaw Accel"), lapdist=_one(l, "Lap Dist", None),
                 posy=_one(l, "Pos Y", None), posx=_one(l, "Pos X", None), posz=_one(l, "Pos Z", None),
                 pathlat=_one(l, "Path Lateral", None),
                 load=_four(l, "Tire Load"), susp=_four(l, "Susp Force"), fy=_four(l, "Tire Lateral Force"),
                 fx=_four(l, "Tire Longitudinal Force"), gf=_four(l, "Grip Fraction"), slip=_four(l, "Slip Angle"),
                 rh=_four(l, "Ride Height"), surf=_four(l, "Surface Type"), wspd=_four(l, "Wheel Speed"))
        out.append(d)
    meta = dict(laps[0].meta) if laps else {}
    ident = identify_car(meta, laps[0].car_name if laps else "")
    res = dict(file=path.name, meta=meta, track=meta.get("TrackName", ""), model=ident.model, cls=ident.car_class, laps=out)
    cp.write_bytes(pickle.dumps(res))
    return res


def real_laps(f: dict, within=1.05) -> list:
    if not f["laps"]:
        return []
    best = min(l["time"] for l in f["laps"])
    return [l for l in f["laps"] if l["time"] <= best * within]


def sm(a, n):
    if n <= 1:
        return a
    h = n // 2                               # edge-padded: a plain "same" convolution ramps to 0 at lap ends
    return np.convolve(np.pad(a, (h, n - 1 - h), mode="edge"), np.ones(n) / n, mode="valid")


def cat(laps, key, n=1, idx=None):
    """Concatenate a channel over laps, each lap smoothed by n samples first."""
    if idx is None:
        return np.concatenate([sm(l[key], n) for l in laps])
    return np.concatenate([sm(l[key][idx], n) for l in laps])
