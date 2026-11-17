"""
A local, file-based "grip profile" per car — the durable memory the LLM
coach itself doesn't have between calls. Every /analyze run feeds this lap's
grip_report into the relevant car's profile; the profile accumulates
percentile-based baselines over time (ABS lockup severity, TC intervention
frequency, slip angle ranges, etc.) so future coaching calls can say "this
lockup was worse than your usual" instead of treating every lap in
isolation.

Also accumulates the PROBABILISTIC TIRE MODEL (tyre model v3, see tire_mf.py):
every analysed lap is reduced to per-axle binned sufficient statistics
(tire_bins.LapBins) and stored per lap under `tire_files`; the Magic Formula
fit (with a bootstrap-over-laps confidence) is recomputed from those bins
whenever the set of laps changes and cached in `tire_fit_cache`. With zero
history the model has no peak at all (null, not a guess).

v3 profile layout (new keys; the ABS/TC/slip baseline keys are unchanged):

    car_name        the car MODEL (car_identity.identify_car(...).model), so every
                    livery of a model shares one file
    car_class       "Hypercar" / "LMGT3" / ... / "unknown" (class prior pooling)
    tire_files      {"<file>": {"game_version", "track",
                                "laps": {"<lap number>": {"load_source",
                                                          "front": LapBins dict,
                                                          "rear": LapBins dict}}}}
                    game version and track live ONCE per file, never per lap
    tire_fit_cache  {"fingerprint", "front": {...}, "rear": {...}} (see refit_tyre_model)

Stored as plain JSON files, one per car, under a local directory — chosen
over SQLite specifically because you mentioned wanting this distributable
later: a JSON file is trivially inspectable, diffable, shareable (drop it
in a repo, send it to a friend, back it up), and needs no schema migration
tooling as fields evolve. If this ever needs to scale past "one person's
local car profiles" — many users, concurrent writes, querying across
thousands of laps — that's the point to move to SQLite/a real DB, not
before.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import tire_class
from .car_identity import identify_car
from .telemetry import reference_lap_time
from .tire_bins import MEASURE_MU, LapBins, bin_axle_samples
from .tire_class import CLASS_PRIOR_WEIGHT, fit_car_with_prior, fit_class_prior
from .tire_mf import MFFit, fit_mf
from .tire_model import (
    _lap_axle_samples,
    axle_load_source,
    build_axle_tire_model,
    file_load_reference,
    fit_deflection_calibration,
)

PROFILES_DIR = Path("/data/grip_profiles")

# Bump when the meaning of the stored tire data changes. v1 stored every sample
# above 40 km/h, v2 stored raw (slip, lat G, ...) observations; both are discarded
# on load rather than mixed with v3's per-lap bins (see GripProfile.from_dict).
TIRE_MODEL_VERSION = 3

# Laps slower than this multiple of the file's fastest lap are not folded
# into the tire model (out laps, pit stops, spins, traffic). Their samples
# are not wrong as such, but they are mostly far below the limit and on
# out laps the tyres are cold.
TIRE_LAP_MAX_RATIO = 1.15

# A game version needs this many laps to be fitted on its own. Below that in every
# version, all laps are fitted together and the result says "mixed_versions".
MIN_VERSION_LAPS = 3

AXLES = ("front", "rear")


def _version_key(version: str | None) -> tuple:
    """Sort key for game versions: numeric-aware ("1.10" > "1.9"), "unknown" oldest.

    The string is the tiebreak so two versions with the same digits still order stably.
    """
    if not version or version == "unknown":
        return ((-1,), "")
    nums = tuple(int(x) for x in re.findall(r"\d+", version))
    return (nums or (0,), version)


def _majority_measure(bin_dicts: list) -> str:
    """Most common grip measure among stored LapBins dicts (mu wins a tie).

    WHY: LapBins.merge refuses mixed measures, and a curve half in mu and half in raw
    lateral G is meaningless, so an axle is fitted on its majority measure only.
    """
    counts = Counter(b["measure"] for b in bin_dicts)
    return min(counts, key=lambda m: (-counts[m], m != MEASURE_MU))


def _keys_fingerprint(tire_files: dict, axle: str) -> str:
    keys = sorted(f"{fk}#{ln}" for fk, f in tire_files.items()
                  for ln, lap in f.get("laps", {}).items() if lap.get(axle, {}).get("rows"))
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()


def _first_calibration(laps: list) -> dict:
    """The first lap's deflection->force fit that worked, else {} (a file with
    Susp Force calibrates the cars whose files have only Susp Pos)."""
    for lap in laps:
        cal = fit_deflection_calibration(lap)
        if cal:
            return cal
    return {}


def file_tyre_context(laps: list, profile_calibration: dict | None) -> dict:
    """
    The file-level context ingest_tire_laps takes from a WHOLE recording, for a
    kept lap of it (spec 2026-11-03 4.3): a one-lap file cannot recompute the
    load reference (low-speed straights are on the out lap / pit lane), the
    slow-lap cutoff (the file's fastest real lap) or the calibration.

    calibration: the original's own first deflection fit, else the car profile's
    (as ingest_tire_laps would use it), else None. load_ref: file_load_reference
    over every < 600 s lap with that calibration. fastest_s: reference_lap_time.
    `laps` are the original's at tire_model.TIRE_GRID_HZ. JSON-safe (stored in
    the kept file's metadata KeptTyreContext).
    """
    valid = [l for l in laps if l.lap_time_s < 600]
    cal = _first_calibration(valid) or profile_calibration or None
    load_ref = file_load_reference(valid, cal) if valid else {}
    # over the < 600 s laps, the same set ingest_tire_laps takes it from
    fastest = reference_lap_time(valid)
    return {
        "calibration": json.loads(json.dumps(cal)) if cal else None,
        "load_ref": {k: float(v) for k, v in load_ref.items()},
        "fastest_s": None if fastest is None else float(fastest),
    }


def tire_file_key(path_name: str, meta: dict) -> tuple[str, dict]:
    """
    (file key, context) for ingest_tire_laps. A kept file (metadata KeptFromFile)
    is keyed by its ORIGINAL's name, so its lap (same lap number) is the
    original's lap: present once whichever file is ingested first, and still
    counted after the original is deleted. Its context is KeptTyreContext
    ({} when absent or unreadable: the lap is then still keyed by its original,
    so it is still counted once, but binned with a context recomputed from one
    lap (often no load reference), and it may refresh the profile's calibration).
    A recording: (its own name, {}).
    """
    origin = (meta or {}).get("KeptFromFile")
    if not origin:
        return path_name, {}
    try:
        context = json.loads(meta.get("KeptTyreContext") or "{}")
    except (TypeError, ValueError):
        context = {}
    return origin, context if isinstance(context, dict) else {}


@dataclass
class GripProfile:
    car_name: str
    lap_count: int = 0
    # raw historical samples kept per metric, capped, so percentiles can be
    # recomputed as new laps arrive without needing a running-stats library
    abs_lockup_durations_s: list[float] = field(default_factory=list)
    tc_active_fractions: list[float] = field(default_factory=list)
    peak_combined_g_values: list[float] = field(default_factory=list)
    peak_front_slip_angle_values: list[float] = field(default_factory=list)
    peak_rear_slip_angle_values: list[float] = field(default_factory=list)
    max_wheel_slip_ratio_values: list[float] = field(default_factory=list)

    tire_model_version: int = TIRE_MODEL_VERSION
    car_class: str = "unknown"
    # Per-lap binned tyre statistics, see the module docstring. The load source of a
    # lap is "Tire Load", "Susp Force", "Susp Pos (calibrated to Susp Force)" or
    # "none (...)" (front axle, or "<front> (front), <rear> (rear)" when they differ):
    # LMU writes zero Tire Load for every LMGT3 recorded so far, so those cars' mu
    # rests on the Susp Force stand-in (DevNotes 10/18/2026 pilot).
    tire_files: dict = field(default_factory=dict)
    tire_fit_cache: dict = field(default_factory=dict)
    # per axle {"slope","intercept","r","n"}: Susp Pos -> Susp Force, learned
    # from recordings that had both (tire_model.fit_deflection_calibration)
    deflection_calibration: dict = field(default_factory=dict)

    # cap each list so the file doesn't grow unbounded over months of use —
    # recent-ish history is what matters for "what's normal for me on this
    # car right now", not literally every lap ever recorded
    MAX_SAMPLES: int = 200

    # -- derived views of tire_files (v2 had these as stored fields) -------------

    @property
    def tire_model_laps(self) -> list[str]:
        """"<file>#<lap>" for every lap in the tyre model."""
        return [f"{fk}#{ln}" for fk, f in self.tire_files.items() for ln in f["laps"]]

    @property
    def tire_model_load_sources(self) -> dict:
        return {f"{fk}#{ln}": lap["load_source"]
                for fk, f in self.tire_files.items() for ln, lap in f["laps"].items()}

    def to_dict(self) -> dict:
        return {
            "car_name": self.car_name,
            "car_class": self.car_class,
            "lap_count": self.lap_count,
            "abs_lockup_durations_s": self.abs_lockup_durations_s,
            "tc_active_fractions": self.tc_active_fractions,
            "peak_combined_g_values": self.peak_combined_g_values,
            "peak_front_slip_angle_values": self.peak_front_slip_angle_values,
            "peak_rear_slip_angle_values": self.peak_rear_slip_angle_values,
            "max_wheel_slip_ratio_values": self.max_wheel_slip_ratio_values,
            "tire_model_version": self.tire_model_version,
            "deflection_calibration": self.deflection_calibration,
            "tire_files": self.tire_files,
            "tire_fit_cache": self.tire_fit_cache,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GripProfile":
        v3 = d.get("tire_model_version", 1) == TIRE_MODEL_VERSION
        # PREVIOUSLY v2 profiles kept raw per-sample observations (front_axle_obs /
        # rear_axle_obs, up to 20k samples/axle). v3 stores per-lap bins instead and the
        # two are not convertible (the v2 samples have no state binning), so a profile of
        # any other version loses its tyre data and keeps its ABS/TC/slip baselines.
        # The deflection calibration is not tyre data and stays.
        return cls(
            car_name=d["car_name"],
            car_class=d.get("car_class", "unknown"),
            lap_count=d.get("lap_count", 0),
            abs_lockup_durations_s=d.get("abs_lockup_durations_s", []),
            tc_active_fractions=d.get("tc_active_fractions", []),
            peak_combined_g_values=d.get("peak_combined_g_values", []),
            peak_front_slip_angle_values=d.get("peak_front_slip_angle_values", []),
            peak_rear_slip_angle_values=d.get("peak_rear_slip_angle_values", []),
            max_wheel_slip_ratio_values=d.get("max_wheel_slip_ratio_values", []),
            deflection_calibration=d.get("deflection_calibration", {}),
            tire_files=d.get("tire_files", {}) if v3 else {},
            tire_fit_cache=d.get("tire_fit_cache", {}) if v3 else {},
        )

    def _append_capped(self, lst: list[float], value: float) -> list[float]:
        lst.append(value)
        return lst[-self.MAX_SAMPLES :]

    def update_from_grip_report(self, grip_report: dict) -> None:
        """Folds one analyzed lap's grip-report summary (ABS/TC/slip
        baselines) into the profile. The tire model is fed separately, from
        every lap of the file, by ingest_tire_laps."""
        self.lap_count += 1

        for event in grip_report.get("abs_events", []):
            self.abs_lockup_durations_s = self._append_capped(
                self.abs_lockup_durations_s, event["peak_severity"]
            )
        for event in grip_report.get("tc_events", []):
            self.tc_active_fractions = self._append_capped(
                self.tc_active_fractions, event["peak_severity"]
            )
        if grip_report.get("peak_combined_g") is not None:
            self.peak_combined_g_values = self._append_capped(
                self.peak_combined_g_values, grip_report["peak_combined_g"]
            )
        slip = grip_report.get("slip_angle_analysis", {})
        if "peak_front_slip_angle_deg" in slip:
            self.peak_front_slip_angle_values = self._append_capped(
                self.peak_front_slip_angle_values,
                abs(slip["peak_front_slip_angle_deg"]["value"]),
            )
        if "peak_rear_slip_angle_deg" in slip:
            self.peak_rear_slip_angle_values = self._append_capped(
                self.peak_rear_slip_angle_values,
                abs(slip["peak_rear_slip_angle_deg"]["value"]),
            )
        for wheel_max in grip_report.get("max_wheel_slip_ratio", {}).values():
            self.max_wheel_slip_ratio_values = self._append_capped(
                self.max_wheel_slip_ratio_values, wheel_max
            )

    # ── tyre model: ingest ────────────────────────────────────────────────────

    def ingest_tire_laps(self, laps: list, file_key: str,
                         context: dict | None = None) -> list[int]:
        """
        Folds every usable lap of one recording into the tire model, once.
        PREVIOUSLY only the analyzed lap was added (one lap per /analyze, so
        an 8-lap file contributed 1 lap). Now all laps within
        TIRE_LAP_MAX_RATIO of the file's fastest are added, keyed by file and
        lap number so repeat analyses add nothing. Returns the lap numbers added.

        Each lap is stored as per-axle LapBins (not raw samples). Laps without the
        recorder's Slip Angle channels (native LMU files) add nothing. The game
        version and track are stored once on the file entry, never per lap.
        `laps` must be loaded at tire_model.TIRE_GRID_HZ (the caller's job).

        context: a kept lap's file_tyre_context (tire_file_key gives it, with the
        ORIGINAL's file name as file_key). PREVIOUSLY a kept lap was ingested as a
        file of its own: a second copy of the lap under a new key, binned against
        a load reference, slow-lap cutoff and calibration recomputed from one lap
        plus margins (a flying lap has no low-speed straight, so often no load
        reference at all). Fix: with a non-empty context its calibration,
        load_ref and fastest_s are used as they are, and the profile's
        calibration is not refreshed from the kept file.
        """
        valid = [l for l in laps if l.lap_time_s < 600]
        if not valid:
            return []
        if context:
            calibration = context.get("calibration") or {}
            load_ref = context.get("load_ref") or {}
            fastest = context.get("fastest_s") or min(l.lap_time_s for l in valid)
        else:
            # BUG FIX (10/18/2026): the cutoff reference was min(lap_time_s), which
            # the unfinished last lap or a mid-lap join usually won, so only that
            # partial lap got in. Reference = fastest real lap; partial laps are
            # still ingested (their steady-state samples are as good as any).
            fastest = reference_lap_time(valid) or min(l.lap_time_s for l in valid)
            # Learn/refresh the deflection->force calibration from any lap that
            # has Susp Force, before extracting, so this file can use it too.
            cal = _first_calibration(valid)
            if cal:
                self.deflection_calibration = cal
            calibration = self.deflection_calibration
            # reference load from the whole file incl. slow laps (the out lap /
            # pit lane is usually the only low-speed straight-line running)
            load_ref = file_load_reference(valid, calibration)
        identity = identify_car(valid[0].meta, valid[0].car_name)
        if self.car_class == "unknown":
            self.car_class = identity.car_class
        stored = self.tire_files.get(file_key, {}).get("laps", {})
        added = []
        for lap in valid:
            if str(lap.lap_number) in stored or lap.lap_time_s > fastest * TIRE_LAP_MAX_RATIO:
                continue
            samples = _lap_axle_samples(lap, calibration, load_ref)
            if samples is None:
                continue
            bins = {a: bin_axle_samples(samples[a], samples[a]["mask"]) for a in AXLES}
            if not any(b.cells for b in bins.values()):
                continue
            front_src, rear_src = (axle_load_source(samples[a]) for a in AXLES)
            entry = self.tire_files.setdefault(file_key, {
                "game_version": identity.game_version, "track": valid[0].track_name,
                "laps": {}})
            entry["laps"][str(lap.lap_number)] = {
                "load_source": front_src if front_src == rear_src
                else f"{front_src} (front), {rear_src} (rear)",
                **{a: bins[a].to_dict() for a in AXLES},
            }
            added.append(lap.lap_number)
        return added

    def tire_sample_count(self, axle: str = "front") -> int:
        """Total stored samples of one axle over all laps (for tools and logs)."""
        return sum(LapBins.from_dict(lap[axle]).n_samples()
                   for f in self.tire_files.values() for lap in f["laps"].values()
                   if lap.get(axle, {}).get("rows"))

    def load_normalization_summary(self) -> dict | None:
        """Laps in the tire model by load source, for the baselines / LLM.
        None until there is a tire model lap."""
        sources = self.tire_model_load_sources
        if not sources:
            return None
        counts: dict[str, int] = {}
        for label in sources.values():
            counts[label] = counts.get(label, 0) + 1
        out = {"laps_by_load_source": counts}
        if any(l.startswith("Susp") for l in counts):
            out["note"] = (
                "Where the source is Susp Force (pushrod load) the game gave no Tire Load for "
                "this class. In a Hypercar pilot, mu from Susp Force matched the game's own tyre "
                "grip a little worse than mu from the real tyre load (cornering correlation "
                "0.74-0.88 vs 0.85-0.94) and read ~5-15 % high in absolute terms: use it to "
                "compare laps and corners, not as a true friction coefficient. Laps marked "
                "none had no load ratio, so they add no samples to the load-normalised curve."
            )
        return out

    # ── tyre model: versioned fit with cache ─────────────────────────────────

    def _plan_axle(self, axle: str) -> dict:
        """Which laps one axle is fitted on (no bins parsed, no fitting).

        {"fit": [(key, bins dict)], "older": [...], "version": str|None, "note": str|None,
         "measure": str|None}. Version rule: the newest game version with >=
        MIN_VERSION_LAPS laps is the fit set and older versions' laps are a prior
        candidate; if no version has that many, all laps are fitted together.
        """
        items = [(_version_key(f["game_version"]), f"{fk}#{ln}", lap[axle])
                 for fk, f in self.tire_files.items() for ln, lap in f["laps"].items()
                 if lap.get(axle, {}).get("rows")]
        if not items:
            return {"fit": [], "older": [], "version": None, "note": None, "measure": None,
                    "unused_newer": []}
        measure = _majority_measure([b for _, _, b in items])
        items = [it for it in items if it[2]["measure"] == measure]
        by_version: dict = {}
        for vk, key, b in items:
            by_version.setdefault(vk, []).append((key, b))
        eligible = [vk for vk, v in by_version.items() if len(v) >= MIN_VERSION_LAPS]
        if eligible:
            newest = max(eligible)
            # PREVIOUSLY laps of a newer game version with < MIN_VERSION_LAPS laps were
            # dropped without a word, so the coach did not know the physics had changed.
            # Fix: report them so the version note can say they are not used yet.
            unused = [(vk[1] or "unknown", len(v)) for vk, v in sorted(by_version.items())
                      if vk > newest]
            return {"fit": by_version[newest],
                    "older": [kb for vk, v in by_version.items() if vk < newest for kb in v],
                    "version": newest[1] or "unknown", "note": None, "measure": measure,
                    "unused_newer": unused}
        return {"fit": [(k, b) for _, k, b in items], "older": [], "version": None,
                "note": "mixed_versions" if len(by_version) > 1 else None, "measure": measure,
                "unused_newer": []}

    def _resolve_prior(self, axle: str, plan: dict, class_prior, load_class_prior: bool):
        """(prior MFFit | None, kind, weight) for one axle."""
        if len(plan["fit"]) >= tire_class.PRIOR_MAX_LAPS or not plan["fit"]:
            return None, "class", 0.0
        # DORMANT BY DESIGN (Ruling 11b): PRIOR_MAX_LAPS == MIN_VERSION_LAPS, so the guard
        # above returns first whenever the fit set has >= MIN_VERSION_LAPS laps, and a
        # fit set smaller than that cannot coexist with a newest eligible version. The
        # branch stays so the older-version prior comes back if the constants diverge.
        # older versions of this car first: same tyres, previous physics
        if len(plan["older"]) >= MIN_VERSION_LAPS:
            older = LapBins.merge([LapBins.from_dict(b) for _, b in plan["older"]])
            return fit_mf(older), "older_version", CLASS_PRIOR_WEIGHT / 2
        prior = (class_prior or {}).get(axle)
        if prior is None and load_class_prior:
            prior = class_prior_for(self.car_class, self.car_name, axle, plan["measure"])
        return prior, "class", CLASS_PRIOR_WEIGHT

    def refit_tyre_model(self, class_prior: dict | None = None,
                         load_class_prior: bool = False) -> dict:
        """The cached per-axle bootstrap fits, recomputed only if the inputs changed.

        class_prior: {"front": MFFit | None, "rear": ...} used for axles that need one
        (fewer than tire_class.PRIOR_MAX_LAPS laps); with load_class_prior the other cars
        of this class are pooled when none is given. The fingerprint (sha1 of the fit
        set's lap keys, measure, prior kind/weight and the prior's parameters rounded to
        4 dp) decides whether tire_fit_cache is still valid: compute_baselines runs on
        every /analyze and the 50-resample bootstrap must not.
        """
        plans, priors, fp = {}, {}, {}
        for axle in AXLES:
            plan = self._plan_axle(axle)
            prior, kind, weight = self._resolve_prior(axle, plan, class_prior, load_class_prior)
            plans[axle], priors[axle] = plan, (prior, kind, weight)
            fp[axle] = {
                "laps": sorted(k for k, _ in plan["fit"]), "measure": plan["measure"],
                "prior": None if prior is None or weight <= 0 else {
                    "kind": kind, "weight": weight,
                    "params": {k: round(v, 4) for k, v in sorted(prior.params.items())}},
            }
        fingerprint = hashlib.sha1(json.dumps(
            [TIRE_MODEL_VERSION, fp], sort_keys=True).encode()).hexdigest()
        cache = self.tire_fit_cache
        if cache.get("fingerprint") == fingerprint and all(a in cache for a in AXLES):
            return cache

        out = {"fingerprint": fingerprint}
        for axle in AXLES:
            plan = plans[axle]
            prior, kind, weight = priors[axle]
            lap_bins = [LapBins.from_dict(b) for _, b in plan["fit"]]
            res = fit_car_with_prior(lap_bins, prior, weight=weight, prior_kind=kind)
            fit = res["fit"]
            out[axle] = {
                "fit": fit.to_dict() if fit is not None else None,
                "peak_slip_ci90": res["peak_slip_ci90"], "peak_mu_ci90": res["peak_mu_ci90"],
                "n_boot_ok": res["n_boot_ok"], "confidence": res["confidence"],
                "prior_used": res["prior_used"], "version_note": plan["note"],
                "fit_game_version": plan["version"],
                "n_fit_laps": len(lap_bins),
                "n_samples": sum(b.n_samples() for b in lap_bins),
                "measure": plan["measure"],
            }
        self.tire_fit_cache = out
        return out

    def cached_fit(self, axle: str) -> MFFit | None:
        """The cached Magic Formula fit of one axle (None before the first refit)."""
        entry = self.tire_fit_cache.get(axle) or {}
        return MFFit.from_dict(entry["fit"]) if entry.get("fit") else None

    def _axle_tire_model(self, axle: str, entry: dict) -> dict:
        """v3 model of one axle from its cache entry and the stored bins of its fit set."""
        plan = self._plan_axle(axle)
        lap_bins = [LapBins.from_dict(b) for _, b in plan["fit"]]
        out = build_axle_tire_model(entry, lap_bins)
        # version_note is rebuilt from the plan on each call, NOT read from the cache: the
        # newer-version laps it describes are not part of the cache fingerprint.
        notes = [plan["note"]] if plan["note"] else []
        for version, n in plan["unused_newer"]:
            notes.append(f"newest game version {version} has {n} lap{'s' if n != 1 else ''} "
                         f"(< {MIN_VERSION_LAPS}), not used yet; the model is from "
                         f"{plan['version']} laps")
        out["version_note"] = "; ".join(notes) or None
        return out

    @staticmethod
    def _factor_block(model: dict, reason: str | None) -> dict:
        fx = model.get("factor_effects") or {}
        if not fx:
            return {"available": False,
                    "note": reason or "no fitted factor (load, temperature, pressure) yet"}
        return {
            "available": True, "factor_effects": fx,
            "note": ("Peak mu at the p10 / median / p90 of this driver's own data for each "
                     "fitted factor. These are model estimates inside the driven range: never "
                     "present them as a setup target, and never outside that range."),
        }

    def compute_baselines(self, tire_unavailable: str | None = None) -> dict:
        """
        Percentile-based baselines for prompt injection. Returns None for
        any metric with too few samples to be meaningful (avoid treating a
        single lap's noise as an established norm).

        Also includes the tyre model (front_tire_model / rear_tire_model) built
        from the cached Magic Formula fit. The v2 keys (peak, efficient_range,
        peak_observed, confidence, sample_count, grip_measure, peak_grip,
        available_grip, max_observed_slip_deg, tyre_temperature) keep their names;
        v3 adds the CI, prior and version fields. The v2 temperature/pressure/load
        bucket tables needed raw samples and are gone (None).
        """
        MIN_SAMPLES = 5

        def _percentiles(values: list[float]) -> dict | None:
            if len(values) < MIN_SAMPLES:
                return None
            arr = np.array(values)
            return {
                "median": round(float(np.median(arr)), 3),
                "p90": round(float(np.percentile(arr, 90)), 3),
                "sample_count": len(values),
            }

        # `/analyze` must never fail because of the tyre model: with tire_unavailable set
        # (the caller caught an error) no fit is attempted and the axles degrade to the
        # empty v2-fallback shape carrying the reason.
        if tire_unavailable:
            models = {}
            for axle in AXLES:
                m = build_axle_tire_model(None, [])
                m["note"] = f"tyre model unavailable: {tire_unavailable}"
                models[axle] = m
        else:
            fits = self.refit_tyre_model(load_class_prior=True)
            models = {axle: self._axle_tire_model(axle, fits[axle]) for axle in AXLES}

        return {
            "car_name": self.car_name,
            "total_laps_analyzed": self.lap_count,
            "tire_model_laps": len(self.tire_model_laps),
            "load_normalization": self.load_normalization_summary(),
            "abs_lockup_duration_s": _percentiles(self.abs_lockup_durations_s),
            "tc_active_fraction": _percentiles(self.tc_active_fractions),
            "peak_combined_g": _percentiles(self.peak_combined_g_values),
            "peak_front_slip_angle_deg": _percentiles(self.peak_front_slip_angle_values),
            "peak_rear_slip_angle_deg": _percentiles(self.peak_rear_slip_angle_values),
            "max_wheel_slip_ratio": _percentiles(self.max_wheel_slip_ratio_values),
            "front_tire_model": models["front"],
            "rear_tire_model": models["rear"],
            "front_temp_pressure_load": self._factor_block(models["front"], tire_unavailable),
            "rear_temp_pressure_load": self._factor_block(models["rear"], tire_unavailable),
        }


# ── class prior across the other cars of a class ─────────────────────────────

_class_prior_cache: dict = {}


def clear_class_prior_cache() -> None:
    _class_prior_cache.clear()


def class_prior_for(car_class: str, exclude_model: str, axle: str = "front",
                    measure: str = MEASURE_MU) -> MFFit | None:
    """Pooled fit over every other profile of `car_class` (tire_class.fit_class_prior).

    Only profiles whose stored bins for this axle are mostly in `measure` are pooled:
    a mu curve and a raw-lateral-G curve are on different scales. Cached per process by
    (class, axle, measure, {(model, fingerprint of its lap keys)}), so a profile that
    gains a lap invalidates it and everything else does not.
    """
    if not car_class or car_class == "unknown" or not PROFILES_DIR.exists():
        return None
    pool: dict = {}
    for path in sorted(PROFILES_DIR.glob("*.json")):
        if path.name.startswith("."):  # .ingest_ledger.json and friends
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (not isinstance(d, dict) or d.get("tire_model_version") != TIRE_MODEL_VERSION
                or d.get("car_class") != car_class or d.get("car_name") in (None, exclude_model)):
            continue
        files = d.get("tire_files", {})
        laps = [lap[axle] for f in files.values() for lap in f.get("laps", {}).values()
                if lap.get(axle, {}).get("rows")]
        if laps and _majority_measure(laps) == measure:
            pool[d["car_name"]] = (laps, _keys_fingerprint(files, axle))
    if not pool:
        return None
    key = (car_class, axle, measure, tuple(sorted((m, fp) for m, (_, fp) in pool.items())))
    if key not in _class_prior_cache:
        _class_prior_cache[key] = fit_class_prior(
            {m: [LapBins.from_dict(b) for b in laps] for m, (laps, _) in pool.items()}, measure)
    return _class_prior_cache[key]


# ── files ────────────────────────────────────────────────────────────────────

def _profile_path(car_name: str) -> Path:
    # sanitize car name into a safe filename
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in car_name)
    return PROFILES_DIR / f"{safe_name}.json"


def load_profile(car_name: str) -> GripProfile:
    """Profile stored under `car_name`. The v3 key is the car MODEL (use
    load_profile_for_lap); any name still loads, so older callers and the tools keep working."""
    path = _profile_path(car_name)
    if not path.exists():
        return GripProfile(car_name=car_name)
    with open(path, "r") as f:
        return GripProfile.from_dict(json.load(f))


def load_profile_for_lap(lap) -> GripProfile:
    """The profile of a lap's car MODEL, so every livery shares one file."""
    identity = identify_car(lap.meta, lap.car_name)
    profile = load_profile(identity.model)
    if profile.car_class == "unknown":
        profile.car_class = identity.car_class
    return profile


def save_profile(profile: GripProfile) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.car_name)
    # No indent: indent=2 puts every number on its own line.
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f)
