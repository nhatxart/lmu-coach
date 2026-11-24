"""Ideal lap, fit groups and storage (design 2026-11-10-ideal-lap-design.md §6.2-6.4).

Which recordings fit the car model of a lap, the JSON profile they are kept in,
when a group is refitted, the fallbacks and the refusal.

- Fit group, track scope: (car model, track slug, layout within 3 % of length),
  >= 3 qualifying laps. Car scope (fallback): every track of the car model,
  >= 6 laps, flagged `transferred`. Else a Refusal saying what is missing.
- Game version as tyre v3: the newest version with >= 3 laps, else all of them
  with the note `mixed_versions`. At most the 40 most recent laps.
- Profile: <root>/<car model>.json (root = <profiles>/ideal_lap), groups keyed
  "<slug>|<layout length>", refitted when the fingerprint (sorted file#lap keys
  + file mtime/size + VERSION) changes. Per-file reductions are cached as
  <root>/cache/<sha>.npz (ideal_state.load_reduction).
- Nothing here writes outside `root`; recordings are only read.

The session index (session_index.file_entry) gives car, track and layout per
file without loading channels; only the group's files are reduced.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ideal_car as C
from .ideal_state import LRU, FileReduction, MAX_GROUP_LAPS, load_reduction, setup_fingerprint

REDUCTIONS_IN_MEMORY = 12          # ~8 MB each; a fit group rarely has more files (the npz cache keeps all)

VERSION = 3            # 2: validation round 1 (§15, §15.x, §15.y); 3: kt at q 0.75 (§15.z). Profiles from before refit
LAYOUT_TOLERANCE = 0.03
MIN_TRACK_LAPS, MIN_CAR_LAPS, HIGH_LAPS = 3, 6, 6
BETA_TIGHT, BETA_TRACK, BETA_CAR = 0.02, 0.03, 0.05
SELFCHECK_SHIFT, SELFCHECK_SD = 0.02, 0.075     # §9 V2 (odd/even) and V3
SELFCHECK_SD_LOW = 0.075                         # §15 V4 C3: above it the fit is `low`, beta >= 0.05
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T[\d_]+Z)")


@dataclass
class Refusal:
    """No ideal lap: `reason` says what is missing (design §6.3, T7)."""
    reason: str
    status: str = "insufficient_data"


@dataclass
class GroupEntry:
    """One fit group in the profile: the fit and what it was built from."""
    key: str
    scope: str                       # 'track' | 'car'
    fingerprint: str
    game_version: str
    files: list
    n_laps: int
    fit: C.GroupFit
    beta: float
    confidence: str                  # 'high' | 'medium' | 'low'
    selfcheck: dict | None
    notes: list = field(default_factory=list)
    # Full paths of `files` (same order). PREVIOUSLY only the names were kept and
    # the store looked them up next to the reference file, so a group file in the
    # other session folder (reference_laps) silently dropped out of the setup
    # median (review minor 7). Profiles without it fall back to that lookup.
    paths: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "scope": self.scope, "fingerprint": self.fingerprint,
                "game_version": self.game_version, "files": list(self.files), "n_laps": self.n_laps,
                "fit": self.fit.to_dict(), "beta": self.beta, "confidence": self.confidence,
                "selfcheck": self.selfcheck, "notes": list(self.notes), "paths": list(self.paths)}

    @classmethod
    def from_dict(cls, d: dict) -> "GroupEntry":
        kw = dict(d)
        kw["fit"] = C.GroupFit.from_dict(d["fit"])
        return cls(**kw)

    def info(self) -> dict:
        """The `fit` block of the API response (§7)."""
        i = self.fit.info
        return {"scope": self.scope, "key": self.key, "n_laps": self.n_laps, "n_files": len(self.files),
                "load_source": i.get("load_source"), "confidence": self.confidence, "beta": self.beta,
                "game_version": self.game_version, "notes": list(self.notes) + list(i.get("notes", [])),
                "axles": i["axles"], "s_front": i["s_front"], "kappa_brake": i["kappa_brake"],
                "kappa_trac": i["kappa_trac"], "drag": i["drag"], "selfcheck": self.selfcheck,
                "transferred": self.scope == "car", "fingerprint": self.fingerprint}


def safe_name(car_model: str) -> str:
    """As grip_profile._profile_path."""
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in car_model)


def fingerprint(reductions: list, v3_peaks: dict | None = None) -> str:
    """sha1 of the sorted file#lap keys, each file's mtime and size, VERSION,
    the fit code's version (ideal_car.FIT_VERSION), the reduction version and
    the tyre-v3 peaks of the knee cross-check.
    PREVIOUSLY only VERSION: a profile written before a change of the fit code
    (this branch had several) was reused as if nothing had changed (review
    minor 5)."""
    from . import ideal_state
    keys = sorted(f"{r.file}#{s.lap_number}" for r in reductions for s in r.laps)
    stamps = sorted(f"{r.file}|{r.mtime}|{r.size}" for r in reductions)
    code = [f"v{VERSION}", f"fit{C.FIT_VERSION}", f"red{ideal_state.REDUCTION_VERSION}",
            "v3 " + json.dumps(v3_peaks or {}, sort_keys=True)]
    return hashlib.sha1("\n".join(keys + stamps + code).encode()).hexdigest()


def _version_key(v):
    from .grip_profile import _version_key as vk
    return vk(v)


def pick_version(reductions: list) -> tuple[list, str, list]:
    """(reductions, game_version, notes): the newest version with >= 3 laps,
    else every file with the note `mixed_versions` (as tyre v3)."""
    by_v: dict = {}
    for r in reductions:
        by_v.setdefault(r.game_version, []).append(r)
    for v in sorted(by_v, key=_version_key, reverse=True):
        if sum(len(r.laps) for r in by_v[v]) >= MIN_TRACK_LAPS:
            return by_v[v], v, []
    return reductions, "mixed", ["mixed_versions"] if len(by_v) > 1 else []


def newest_laps(reductions: list, n: int = MAX_GROUP_LAPS) -> list:
    """At most the n most recent laps (file time stamp, then lap number)."""
    order = sorted(((STAMP.search(r.file).group(1) if STAMP.search(r.file) else r.file, s.lap_number, i, j)
                    for i, r in enumerate(reductions) for j, s in enumerate(r.laps)), reverse=True)[:n]
    keep = {(i, j) for *_, i, j in order}
    out = []
    for i, r in enumerate(reductions):
        laps = [s for j, s in enumerate(r.laps) if (i, j) in keep]
        if laps:
            out.append(FileReduction(**{**r.__dict__, "laps": laps}))
    return out


def selfcheck(reductions: list, car_class: str) -> dict | None:
    """Design §6.3: fit on the even laps, median and sd of log(observed /
    predicted) on the saturated samples of the odd laps (pooled axles). The
    saturated set of the odd laps uses the even fit's own rules (Grip Fraction,
    else its knee). None when either half is too thin to fit."""
    even = [FileReduction(**{**r.__dict__, "laps": r.laps[0::2]}) for r in reductions]
    odd = [FileReduction(**{**r.__dict__, "laps": r.laps[1::2]}) for r in reductions]
    even, odd = [r for r in even if r.laps], [r for r in odd if r.laps]
    if not even or not odd:
        return None
    try:
        g = C.fit_group(even, car_class)
    except (C.InsufficientData, ValueError, IndexError, np.linalg.LinAlgError):
        return None
    ratios = []
    for r in odd:
        car = g.car_for(r.file)
        for s in r.laps:
            gxt = s.gx + s.grade + car.drag(s.v)
            Af, Ar = car.axles(s.v, s.gx, s.gy, s.gz)
            gymax = np.percentile(np.abs(s.gy), 99)
            base = (np.abs(gxt) < 0.2) & (np.abs(s.gy) > 0.45 * gymax) & (s.v > 15) & (np.abs(s.yawacc) < 0.4)
            for j, (idx, A) in enumerate(((C.FRONT, Af), (C.REAR, Ar))):
                ax = g.axle(j)
                if ax.status != "observed":
                    continue
                if ax.saturation == "grip_fraction" and s.gf is not None:
                    sat = s.gf[list(idx)].mean(0) < C.GF_SATURATED
                elif ax.knee_deg is not None and s.slip is not None:
                    sl = s.slip[list(idx)].mean(0)
                    sat = (sl >= ax.knee_deg) & (sl <= ax.knee_deg + C.KNEE_SPAN_DEG)
                else:
                    continue
                m = base & sat
                ratios.append(np.log(np.abs(s.gy[m]) / A[m]))
    x = np.concatenate(ratios) if ratios else np.zeros(0)
    if len(x) < 50:
        return None
    # shift: the level's own statistic (q 0.75, §15 V2 / V5; the median of a q 0.75
    # calibration sits ~0.67 sd below 0 by construction); the median is kept too
    return {"shift": float(np.quantile(x, C.LEVEL_Q)), "median": float(np.median(x)),
            "sd": float(np.std(x)), "n": int(len(x))}


def beta_and_confidence(scope: str, n_laps: int, fit: C.GroupFit, sc: dict | None) -> tuple[float, str, list]:
    """Design §5.4 (β, with the round-1 knee rule) and §6.3 (confidence).
    `low` also with `default` grip terms; an assumed kb / kt or a drag prior is
    listed and lowers it only where it binds on > 10 % of the lap (ideal_lap)."""
    axles = [fit.axle(0), fit.axle(1)]
    knee_based = any(a.saturation == "knee" for a in axles)
    both_observed = all(a.status == "observed" for a in axles)
    weak = []
    i = fit.info
    if i["kappa_brake"]["status"] == "assumed":
        weak.append("kappa_brake assumed")
    if i["kappa_trac"]["status"] == "assumed":
        weak.append("kappa_trac assumed")
    if i["drag"]["source"] == "prior":
        weak.append("drag prior")
    weak += [f"{C.AXLES[j]} terms default" for j, a in enumerate(axles) if a.terms == "default"]
    if scope == "car":
        return BETA_CAR, "low", weak
    beta = BETA_TIGHT if (n_laps >= HIGH_LAPS and both_observed and not knee_based) else BETA_TRACK
    passes = sc is not None and abs(sc["shift"]) <= SELFCHECK_SHIFT and sc["sd"] <= SELFCHECK_SD
    # PREVIOUSLY an assumed kb / kt made the fit `low` here unconditionally. §6.3:
    # low only when the assumed part binds on > 10 % of the lap, which
    # ideal_lap.analyse_lap checks from the solve's binding shares. Fix: only
    # `default` grip terms (§13 R2) lower it here; kappa and the drag prior are
    # notes, judged per lap.
    hard_weak = [w for w in weak if w.endswith("terms default")]
    # §15 V4 C3 (data quality): a self-check spread above the V3 limit means the
    # grip varies more than this model holds (v4 recordings: sd 0.076-0.167, no
    # Pos Y, no Yaw Accel; HF1 0.030-0.071) and, with the level at q 0.75, lifts
    # it by 0.67 sd: low confidence and a range of at least +-5 %.
    if sc is not None and sc["sd"] > SELFCHECK_SD_LOW:
        weak.append("grip varies more than usual in these recordings (older recorder files "
                    "have no height or yaw-acceleration channels)")
        return max(beta, BETA_CAR), "low", weak
    if hard_weak:
        conf = "low"
    elif n_laps >= HIGH_LAPS and both_observed and passes:
        conf = "high"
    else:
        conf = "medium"
    return beta, conf, weak


# Design §14 R8: a file's setup differs from its group when an item is off the
# group median by MORE than this (either axle). Same-setup files in the probe
# (fable_p7) stay inside all of these; every setup change found exceeds one.
SETUP_THRESHOLDS = (("camber", 0.3), ("toe", 0.1), ("rh_mid", 4.0), ("rh_fast", 4.0), ("press", 3.0),
                    ("aero_f", 0.015), ("lltd_f", 0.015))
SETUP_REPORT_ONLY = ("compound",)          # listed, but alone it neither widens beta nor lowers confidence
SETUP_BETA_ADD = 0.03
CONF_ORDER = ("high", "medium", "low")


def setup_differs(fp: dict, med: dict) -> list:
    """Items of the file's setup fingerprint off the group median (§14 R8),
    in a fixed order; a missing measurement on either side is not a difference.
    Compound is listed when different; temperature is never listed."""
    out = []
    for key, tol in SETUP_THRESHOLDS:
        a, b = fp.get(key), med.get(key)
        if a is None or b is None:
            continue
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        if a.shape == b.shape and np.any(np.abs(a - b) > tol):
            out.append(key)
    if fp.get("compound") and med.get("compound") and fp["compound"] != med["compound"]:
        out.append("compound")
    return out


def group_median(fps: list) -> dict:
    """Element-wise median of the group's setup fingerprints (None ignored);
    compound = the most common one."""
    out = {}
    keys = {k for fp in fps for k in fp}
    for k in keys:
        vals = [fp.get(k) for fp in fps if fp.get(k) is not None]
        if not vals:
            out[k] = None
        elif k == "compound":
            out[k] = max(dict.fromkeys(vals), key=vals.count)
        elif isinstance(vals[0], (list, tuple)):
            same = [v for v in vals if len(v) == len(vals[0])]
            out[k] = [float(x) for x in np.median(np.asarray(same, float), axis=0)]
        else:
            out[k] = float(np.median(np.asarray(vals, float)))
    return out


def _default_v3_peaks(car_model: str) -> dict | None:
    """Tyre v3's fitted peak slip per axle, for the knee cross-check only."""
    try:
        from .grip_profile import load_profile
        cache = load_profile(car_model).tire_fit_cache or {}
        out = {}
        for ax in ("front", "rear"):
            fit = (cache.get(ax) or {}).get("fit")
            if fit and fit.get("peak_slip_deg"):
                out[ax] = float(fit["peak_slip_deg"])
        return out or None
    except Exception:
        return None


class IdealStore:
    """Fit groups over a set of session folders, kept under `root`.

    reducer(path) -> FileReduction and indexer(path) -> session-index entry
    are injectable (the tests use synthetic ones)."""

    def __init__(self, root: str | Path, sessions_dirs: list, reducer=None, indexer=None,
                 v3_peaks=_default_v3_peaks):
        self.root = Path(root)
        self.sessions_dirs = [Path(p) for p in sessions_dirs]
        self.cache_dir = self.root / "cache"
        self.reducer = reducer or (lambda p: load_reduction(p, self.cache_dir))
        if indexer is None:
            from .session_index import file_entry
            indexer = lambda p: file_entry(p, "sessions")   # noqa: E731
        self.indexer = indexer
        self.v3_peaks = v3_peaks
        self.fits_done = 0                       # how many group fits ran (tests, ingest log)
        self._red = LRU(REDUCTIONS_IN_MEMORY)

    # ── files ────────────────────────────────────────────────────────────
    def _paths(self) -> list[Path]:
        out = []
        for d in self.sessions_dirs:
            if d.exists():
                out += sorted(d.glob("*.duckdb"))
        return out

    def _reduction(self, path: Path) -> FileReduction:
        key = str(path)
        if key not in self._red:
            self._red[key] = self.reducer(path)
        return self._red[key]

    def profile_path(self, car_model: str) -> Path:
        return self.root / f"{safe_name(car_model)}.json"

    def load_profile(self, car_model: str) -> dict:
        p = self.profile_path(car_model)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("version") == VERSION:
                return d
        return {"version": VERSION, "car_model": car_model, "car_class": None, "groups": {},
                "car_scope": None, "files": {}}

    def save_profile(self, prof: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.profile_path(prof["car_model"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(C._jsonable(prof)), encoding="utf-8")
        tmp.replace(p)

    # ── groups ───────────────────────────────────────────────────────────
    def _entries(self, car_model: str) -> list[tuple[Path, dict]]:
        out = []
        for p in self._paths():
            e = self.indexer(p)
            if e.get("error") or e.get("car_model") != car_model:
                continue
            out.append((p, e))
        return out

    @staticmethod
    def _track_id(e: dict) -> str | None:
        return e.get("track_slug") or e.get("track")

    @staticmethod
    def _layout_groups(lengths: list[float]) -> list[list[int]]:
        """Indices grouped so every member is within 3 % of the group's
        shortest (as track_topography._group_by_length)."""
        groups: list[list[int]] = []
        for i in sorted(range(len(lengths)), key=lambda k: lengths[k]):
            if groups and lengths[i] <= lengths[groups[-1][0]] * (1 + LAYOUT_TOLERANCE):
                groups[-1].append(i)
            else:
                groups.append([i])
        return groups

    def track_group(self, path: Path, entry: dict) -> tuple[str, list[Path]]:
        """(key, files) of the track-scope group of `path`'s file."""
        tid = self._track_id(entry)
        same = [(p, e) for p, e in self._entries(entry["car_model"]) if self._track_id(e) == tid]
        if not any(p.name == path.name for p, _ in same):
            same.append((path, entry))
        lengths = [e.get("layout_len_m") or 0.0 for _, e in same]
        for g in self._layout_groups(lengths):
            names = [same[i][0].name for i in g]
            if path.name in names:
                base = min(lengths[i] for i in g)
                return f"{tid}|{base:.0f}", [same[i][0] for i in g]
        return f"{tid}|{entry.get('layout_len_m') or 0:.0f}", [path]

    def _fit_entry(self, key: str, scope: str, paths: list[Path], car_class: str, car_model: str,
                   prof: dict) -> GroupEntry | Refusal:
        reds, where = [], {}
        for p in paths:
            try:
                reds.append(self._reduction(p))
                where.setdefault(p.name, str(p))
            except Exception as exc:                         # one unreadable file must not refuse the group
                prof.setdefault("unreadable", {})[p.name] = f"{type(exc).__name__}: {exc}"
        reds = [r for r in reds if r.laps]
        if not reds:
            return Refusal("no qualifying laps in this car's recordings")
        reds, version, vnotes = pick_version(reds)
        reds = newest_laps(reds)
        n = sum(len(r.laps) for r in reds)
        need = MIN_TRACK_LAPS if scope == "track" else MIN_CAR_LAPS
        if n < need:
            where = "on this track" if scope == "track" else "of this car"
            return Refusal(f"{need} clean laps {where} needed, have {n}")
        peaks = self.v3_peaks(car_model) if self.v3_peaks else None
        fp = fingerprint(reds, peaks)
        old = (prof["groups"].get(key) if scope == "track" else prof.get("car_scope")) or None
        if old and old.get("fingerprint") == fp:
            ge = GroupEntry.from_dict(old)
            ge.paths = [where.get(f, f) for f in ge.files]   # where the files are now
            return ge
        try:
            fit = C.fit_group(reds, car_class, peaks)
        except C.InsufficientData as exc:
            return Refusal(str(exc))
        self.fits_done += 1
        sc = selfcheck(reds, car_class)
        beta, conf, weak = beta_and_confidence(scope, n, fit, sc)
        notes = vnotes + weak + (["transferred: estimate from other tracks"] if scope == "car" else [])
        ge = GroupEntry(key=key, scope=scope, fingerprint=fp, game_version=version,
                        files=[r.file for r in reds], n_laps=n, fit=fit, beta=beta, confidence=conf,
                        selfcheck=sc, notes=notes, paths=[where.get(r.file, r.file) for r in reds])
        if scope == "track":
            prof["groups"][key] = ge.to_dict()
        else:
            prof["car_scope"] = ge.to_dict()
        for r in reds:
            lm = fit.load_models.get(r.file)
            prof["files"][r.file] = {"mtime": r.mtime, "size": r.size,
                                     "load_model": lm.to_dict() if lm else None,
                                     "laps": [s.lap_number for s in r.laps], "path": where.get(r.file),
                                     "setup": setup_fingerprint(None, None, lm, measured=r.setup or {})}
        prof["car_class"] = car_class
        self.save_profile(prof)
        return ge

    def group_for(self, path: str | Path) -> GroupEntry | Refusal:
        """The fit group of a recording: track scope, else car scope, else a
        Refusal. Never raises for data reasons."""
        path = Path(path)
        try:
            entry = self.indexer(path)
        except Exception as exc:
            return Refusal(f"recording unreadable: {type(exc).__name__}: {exc}")
        if entry.get("error"):
            return Refusal(f"recording unreadable: {entry['error']}")
        car_model, car_class = entry.get("car_model"), entry.get("car_class") or "unknown"
        if not car_model:
            return Refusal("car model unknown")
        try:
            own = self._reduction(path)
        except Exception as exc:
            return Refusal(f"recording unreadable: {type(exc).__name__}: {exc}")
        # (a file without qualifying laps says nothing about its channels; the group decides)
        if own.laps and not (own.has_slip or own.has_grip_fraction):
            return Refusal("no slip angles and no Grip Fraction (a native LMU file, or a recording "
                           "without them)")
        prof = self.load_profile(car_model)
        key, paths = self.track_group(path, entry)
        track = self._fit_entry(key, "track", paths, car_class, car_model, prof)
        if isinstance(track, GroupEntry):
            return track
        all_paths = [p for p, _ in self._entries(car_model)]
        if path.name not in {p.name for p in all_paths}:
            all_paths.append(path)
        car = self._fit_entry("__car__", "car", all_paths, car_class, car_model, prof)
        if isinstance(car, GroupEntry):
            return car
        return Refusal(track.reason)

    def car_for_lap(self, path: str | Path, lap_number: int | None = None):
        """(CarModel, fit info) for a lap of `path`, or a Refusal (design §8).
        The car uses the file's own load model; a file with fewer than 2
        qualifying laps (or a car-scope fit from other tracks without this
        file) uses the group's pooled one. Round 2 (§14 R8): the grip level per
        axle is trimmed to the file's own saturated samples, and the file's
        setup is compared with the group's (fit.setup, fit.setup_differs)."""
        import copy
        g = self.group_for(path)
        if isinstance(g, Refusal):
            return g
        path = Path(path)
        name = path.name
        car = g.fit.car_for(name)
        info = copy.deepcopy(g.info())
        info["load_model"] = "file" if name in g.fit.load_models and g.fit.load_models[name].scope == "file" else "group"
        axles = [g.fit.axle(0), g.fit.axle(1)]
        red = self._reduction(path)
        car, trims = C.axle_trim(car, axles, red)
        for j, ax in enumerate(C.AXLES):
            info["axles"][ax].update(trim=trims[j]["trim"], trim_w=trims[j]["w"], trim_t=trims[j]["t"],
                                     trim_n=trims[j]["n"], mu1_file=car.mu1[j])
        lm = lambda f: g.fit.load_models.get(f) or g.fit.load_models.get("__group__")   # noqa: E731
        fp = setup_fingerprint(None, None, lm(name), measured=red.setup or {})
        fps = []
        paths = g.paths if len(g.paths) == len(g.files) else [str(path.with_name(f)) for f in g.files]
        for f, p in zip(g.files, paths):
            try:
                fps.append(setup_fingerprint(None, None, lm(f), measured=self._reduction(Path(p)).setup or {}))
            except Exception:
                continue
        differs = setup_differs(fp, group_median(fps)) if fps else []
        info["setup"], info["setup_differs"] = fp, differs
        flagged = [d for d in differs if d not in SETUP_REPORT_ONLY]
        conf = info["confidence"]
        if flagged and any(t["trim"] == "none" for t in trims):
            info["beta"] = info["beta"] + SETUP_BETA_ADD
            conf = CONF_ORDER[min(CONF_ORDER.index(conf) + 1, len(CONF_ORDER) - 1)]
            info["notes"].append("setup differs from the laps the grip was learned on (%s)" % ", ".join(flagged))
        if any(t["trim"] == "capped" for t in trims):
            conf = "low"
            info["notes"].append("axle trim capped at 8 %: a different car or a broken fit")
        info["confidence"] = conf
        return car, info

    # ── ingest ───────────────────────────────────────────────────────────
    def fit_all(self, paths: list | None = None, log=print) -> dict:
        """The ingest pass (design §6.4): reduce every file (builds the npz
        caches) and fit every track group and car scope with enough laps.
        {key: 'fitted' | 'cached' | reason}."""
        out = {}
        seen = set()
        for p in paths or self._paths():
            p = Path(p)
            try:
                e = self.indexer(p)
                if e.get("error") or not e.get("car_model"):
                    out[p.name] = e.get("error") or "car model unknown"
                    continue
                key, members = self.track_group(p, e)
            except Exception as exc:
                out[p.name] = f"{type(exc).__name__}: {exc}"
                continue
            gid = (e["car_model"], key)
            if gid in seen:
                continue
            seen.add(gid)
            n0 = self.fits_done
            res = self.group_for(p)
            what = res.reason if isinstance(res, Refusal) else (
                "fitted" if self.fits_done > n0 else "cached") + f" ({res.scope}, {res.n_laps} laps, {res.confidence})"
            out[f"{e['car_model']} {key}"] = what
            log(f"  {e['car_model']} {key}: {what}")
        return out


def default_store(sessions_dirs: list, profiles_dir: str | Path | None = None) -> IdealStore:
    """The app's store: <profiles>/ideal_lap (grip_profile.PROFILES_DIR)."""
    from . import grip_profile
    root = Path(profiles_dir or grip_profile.PROFILES_DIR) / "ideal_lap"
    return IdealStore(root, sessions_dirs)
