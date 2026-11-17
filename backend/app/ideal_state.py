"""Ideal lap (design 2026-11-10-ideal-lap-design.md): from a recording to the
smoothed per-lap state arrays the car fit (ideal_car) and the solver
(ideal_solver) work on. Ported from the spike's il_data.py / il_core.lap_state.

Conventions (measured, see 2026-11-10-ideal-lap-strategy.md):
- gx = kinematic longitudinal g, + = speeding up (= -G Force Long, which is
  kinematic: the hill does not appear in it).
- gy = lateral g as recorded; gz = path vertical g (crest -, compression +).
- grade = dh/ds from Pos Y; else the track topography at the lap's Lap Dist;
  else zero (`grade_source` says which).
- v in m/s; throttle and brake in percent; wheels FL, FR, RL, RR.
- Grip Fraction is LOW when the tyre is saturated.
"""
from __future__ import annotations

import hashlib
import io
import contextlib
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

G = 9.81
GRID_HZ = 50.0          # = tire_model.TIRE_GRID_HZ; files are loaded at this rate
SMOOTH_N = 10           # 0.2 s at 50 Hz: every signal used in a fit
GRADE_N = 25            # 0.5 s: height and speed before the grade is taken
QUALIFY_WITHIN = 1.05   # qualifying lap: lap time <= 1.05 x the file's best
MAX_GROUP_LAPS = 40     # per fit group, the most recent (cost)
REDUCTION_VERSION = 1
WHEELS = ("FL", "FR", "RL", "RR")

# Per-wheel arrays a State may carry (4 x n each, None when the file lacks them).
WHEEL_KEYS = ("load", "susp", "slip", "fx", "fy", "gf")
SCALAR_KEYS = ("v", "gx", "gy", "gz", "grade", "thr", "brk", "abs", "tc", "yawacc")


def smooth(a: np.ndarray, n: int) -> np.ndarray:
    """Moving average over n samples, edge-padded (a plain 'same' convolution
    ramps to 0 at the lap ends)."""
    a = np.asarray(a, dtype=float)
    if n <= 1:
        return a
    h = n // 2
    return np.convolve(np.pad(a, (h, n - 1 - h), mode="edge"), np.ones(n) / n, mode="valid")


@dataclass
class State:
    """One lap's smoothed state on the 50 Hz grid (see module docstring)."""
    lap_number: int
    lap_time_s: float
    v: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray
    grade: np.ndarray
    thr: np.ndarray
    brk: np.ndarray
    abs: np.ndarray
    tc: np.ndarray
    yawacc: np.ndarray
    load: np.ndarray | None = None      # Tire Load (N), Hypercars
    susp: np.ndarray | None = None      # Susp Force (N), every recorder file
    slip: np.ndarray | None = None      # |slip angle| in degrees
    fx: np.ndarray | None = None        # Tire Longitudinal Force (N)
    fy: np.ndarray | None = None        # Tire Lateral Force (N)
    gf: np.ndarray | None = None        # Grip Fraction (LOW = saturated)

    def L(self, source: str) -> np.ndarray:
        """The load source of the file's load model ('load' or 'susp')."""
        return getattr(self, source)

    def __len__(self) -> int:
        return len(self.v)


def _four(lap, name: str) -> np.ndarray | None:
    ks = [f"{name} {w}" for w in WHEELS]
    if not all(k in lap.channels for k in ks):
        return None
    a = np.vstack([np.nan_to_num(np.asarray(lap.channels[k], float)) for k in ks])
    return a if np.nanmax(np.abs(a)) > 1e-9 else None


def _one(lap, name: str, default: float | None = 0.0) -> np.ndarray | None:
    if name not in lap.channels:
        return None if default is None else np.full(len(lap.time_s), default)
    return np.nan_to_num(np.asarray(lap.channels[name], float))


def lap_raw(lap, topo_layout: dict | None = None) -> dict:
    """Unsmoothed per-lap arrays of a telemetry.LapTelemetry (il_data.load_file).
    `s` = distance integrated from Ground Speed (the solver's line coordinate).
    `topo_layout`: the track topography layout, used for the grade only when the
    lap has no Pos Y."""
    t = np.asarray(lap.time_s, float)
    v = _one(lap, "Ground Speed") / 3.6
    thr, brk = _one(lap, "Throttle Pos"), _one(lap, "Brake Pos")
    if len(thr) and np.nanmax(thr) <= 1.5:
        thr = thr * 100
    if len(brk) and np.nanmax(brk) <= 1.5:
        brk = brk * 100
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    d = dict(n=int(lap.lap_number), time=float(lap.lap_time_s), t=t, v=v, s=s,
             gx=-_one(lap, "G Force Long"), gy=_one(lap, "G Force Lat"), gz=_one(lap, "G Force Vert"),
             thr=thr, brk=brk, abs=_one(lap, "ABS"), tc=_one(lap, "TC"),
             yaw=_one(lap, "Yaw Rate", None), yawacc=_one(lap, "Yaw Accel"),
             posy=_one(lap, "Pos Y", None), lapdist=getattr(lap, "lap_dist_raw", None),
             load=_four(lap, "Tire Load"), susp=_four(lap, "Susp Force"), fy=_four(lap, "Tire Lateral Force"),
             fx=_four(lap, "Tire Longitudinal Force"), gf=_four(lap, "Grip Fraction"),
             slip=_four(lap, "Slip Angle"))
    d["grade"], d["grade_source"] = raw_grade(d, topo_layout)
    return d


def raw_grade(raw: dict, topo_layout: dict | None = None) -> tuple[np.ndarray, str]:
    """(dh/ds, source). Pos Y of the lap, else the topography at Lap Dist, else 0."""
    posy = raw.get("posy")
    if posy is not None and np.nanmax(np.abs(posy - posy[0])) > 1e-6:
        g = np.gradient(smooth(posy, GRADE_N), 1 / GRID_HZ) / np.maximum(smooth(raw["v"], GRADE_N), 5)
        return np.clip(g, -0.3, 0.3), "pos_y"
    if topo_layout is not None and raw.get("lapdist") is not None:
        from .track_topography import grade_bank
        g, _ = grade_bank(topo_layout, np.asarray(raw["lapdist"], float))
        return np.clip(np.nan_to_num(g), -0.3, 0.3), "topography"
    return np.zeros(len(raw["v"])), "none"


def lap_state(raw: dict) -> State:
    """Smoothed state of one lap (il_core.lap_state). ABS/TC 0.1 s, the rest 0.2 s;
    throttle and brake are not smoothed (thresholds on them are about the pedal)."""
    n = SMOOTH_N
    wheel = {}
    for k in WHEEL_KEYS:
        a = raw.get(k)
        wheel[k] = None if a is None else np.vstack([smooth(a[i], n) for i in range(4)])
    if wheel["slip"] is not None:
        wheel["slip"] = np.degrees(np.abs(wheel["slip"]))       # recorder: radians
    return State(lap_number=int(raw["n"]), lap_time_s=float(raw["time"]),
                 v=smooth(raw["v"], n), gx=smooth(raw["gx"], n), gy=smooth(raw["gy"], n),
                 gz=smooth(raw["gz"], n), grade=smooth(raw["grade"], n),
                 thr=np.asarray(raw["thr"], float), brk=np.asarray(raw["brk"], float),
                 abs=smooth(raw["abs"], 5), tc=smooth(raw["tc"], 5), yawacc=smooth(raw["yawacc"], n),
                 **wheel)


def qualifying_laps(laps: list, pit_laps: set[int] | None = None,
                    within: float = QUALIFY_WITHIN) -> list:
    """Design §6.2: complete, real (telemetry.real_lap_flags, as the session
    index), not a pit lap, lap time <= 1.05 x the file's best real lap."""
    from .telemetry import real_lap_flags
    flags = real_lap_flags([l.lap_time_s for l in laps], [l.is_complete for l in laps])
    real = [l for l, f in zip(laps, flags) if f and int(l.lap_number) not in (pit_laps or set())]
    if not real:
        return []
    best = min(l.lap_time_s for l in real)
    return [l for l in real if l.lap_time_s <= best * within]


@dataclass
class FileReduction:
    """What the fit needs from one recording: its qualifying laps as States."""
    file: str
    mtime: float
    size: int
    meta: dict
    track: str
    car_model: str
    car_class: str
    game_version: str
    load_source: str | None          # 'load' | 'susp' | None (no load channel)
    grade_source: str
    laps: list[State] = field(default_factory=list)

    @property
    def has_forces(self) -> bool:
        return bool(self.laps) and all(s.fx is not None and s.fy is not None and s.load is not None
                                       for s in self.laps)

    @property
    def has_grip_fraction(self) -> bool:
        return bool(self.laps) and all(s.gf is not None for s in self.laps)

    @property
    def has_slip(self) -> bool:
        return bool(self.laps) and all(s.slip is not None for s in self.laps)


def file_load_source(states: list[State]) -> str | None:
    """Tire Load when every lap has it, else Susp Force (design §4.1)."""
    if states and all(s.load is not None for s in states):
        return "load"
    if states and all(s.susp is not None for s in states):
        return "susp"
    return None


def reduce_laps(laps: list, path: Path, pit_laps: set[int] | None = None,
                topo_layout: dict | None = None) -> FileReduction:
    """FileReduction from already loaded LapTelemetry (no I/O but stat)."""
    from .car_identity import identify_car
    meta = dict(laps[0].meta) if laps else {}
    ident = identify_car(meta, laps[0].car_name if laps else "")
    raws = [lap_raw(l, topo_layout) for l in qualifying_laps(laps, pit_laps)]
    states = [lap_state(r) for r in raws]
    st = Path(path).stat()
    srcs = sorted({r["grade_source"] for r in raws})
    return FileReduction(file=Path(path).name, mtime=st.st_mtime, size=st.st_size, meta=meta,
                         track=meta.get("TrackName", ""), car_model=ident.model, car_class=ident.car_class,
                         game_version=ident.game_version, load_source=file_load_source(states),
                         grade_source=",".join(srcs) if srcs else "none", laps=states)


def _ensure_wheel_channels() -> None:
    """Design §6.1: the fit needs two wheel channels the rest of the app does not."""
    from . import telemetry
    for extra in ("Tire Longitudinal Force", "Grip Fraction"):
        if extra not in telemetry.WHEEL_CHANNELS:
            telemetry.WHEEL_CHANNELS.append(extra)


def _cache_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.name}|{st.st_mtime}|{st.st_size}|{REDUCTION_VERSION}".encode()).hexdigest()[:20]


def _save_npz(red: FileReduction, p: Path) -> None:
    import json
    arrays = {}
    for i, s in enumerate(red.laps):
        for f in fields(State):
            val = getattr(s, f.name)
            if isinstance(val, np.ndarray):
                arrays[f"{i}|{f.name}"] = val
    head = {k: getattr(red, k) for k in ("file", "mtime", "size", "meta", "track", "car_model", "car_class",
                                         "game_version", "load_source", "grade_source")}
    head["laps"] = [[s.lap_number, s.lap_time_s] for s in red.laps]
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, __head__=np.array(json.dumps(head)), **arrays)
    tmp.replace(p)


def _load_npz(p: Path) -> FileReduction:
    import json
    z = np.load(p, allow_pickle=False)
    head = json.loads(str(z["__head__"]))
    laps = []
    for i, (num, t) in enumerate(head.pop("laps")):
        kw = {f.name: (z[f"{i}|{f.name}"] if f"{i}|{f.name}" in z.files else None)
              for f in fields(State) if f.name not in ("lap_number", "lap_time_s")}
        laps.append(State(lap_number=num, lap_time_s=t, **kw))
    return FileReduction(laps=laps, **head)


def load_reduction(path: str | Path, cache_dir: str | Path | None = None) -> FileReduction:
    """Load a recording at 50 Hz and reduce it to its qualifying laps.
    With `cache_dir`, the reduction is kept as <sha>.npz keyed by file name,
    mtime, size and REDUCTION_VERSION (a first load takes minutes)."""
    path = Path(path)
    cp = None
    if cache_dir is not None:
        cp = Path(cache_dir) / f"{_cache_key(path)}.npz"
        if cp.exists():
            return _load_npz(cp)
    _ensure_wheel_channels()
    from .telemetry import load_telemetry_file
    from . import track_topography as tt
    with contextlib.redirect_stdout(io.StringIO()):
        laps = load_telemetry_file(path, grid_hz=GRID_HZ)
    pit = tt._read_pit_laps(path)
    layout = None
    if laps and "Pos Y" not in laps[0].channels:
        # PREVIOUSLY laps[0] picked the layout, and the first lap is often a
        # partial out-lap whose Lap Dist span matches no layout. Fix: a complete
        # lap's length (design §13 R4).
        whole = [l for l in laps if l.is_complete] or laps
        got = tt.load_topography(whole[0].track_name, tt._lap_length_m(whole[0]))
        layout = got[1] if got else None
    red = reduce_laps(laps, path, pit, layout)
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        _save_npz(red, cp)
    return red
