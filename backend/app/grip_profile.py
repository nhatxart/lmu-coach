"""
A local, file-based "grip profile" per car — the durable memory the LLM
coach itself doesn't have between calls. Every /analyze run feeds this lap's
grip_report into the relevant car's profile; the profile accumulates
percentile-based baselines over time (ABS lockup severity, TC intervention
frequency, slip angle ranges, etc.) so future coaching calls can say "this
lockup was worse than your usual" instead of treating every lap in
isolation.

Also accumulates the PROBABILISTIC TIRE MODEL (see tire_model.py) — raw
(slip_angle, lateral_g, temp, pressure, load) observations per axle, from
which peak/efficient_range/confidence/sample_count are recomputed on every
read. This is genuinely probabilistic, not a fixed curve: with zero history
the model has no peak at all (returns null, not a guess), and confidence
rises smoothly as more laps are folded in — see tire_model.py's
_confidence_from_sample_count for the exact curve.

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

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .telemetry import reference_lap_time
from .tire_model import (
    AxleSlipObservations,
    build_axle_tire_model,
    build_temp_pressure_load_relationships,
    extract_axle_observations_from_lap,
    file_load_reference,
    fit_deflection_calibration,
)

PROFILES_DIR = Path("/data/grip_profiles")

# Bump when the meaning of the stored tire observations changes. v1 stored
# every sample above 40 km/h (no steady-state filter, no load ratio, no
# temperature reference) — those are discarded on load rather than mixed
# with v2 samples, see tire_model.py's "Model v2" docstring.
TIRE_MODEL_VERSION = 2

# Laps slower than this multiple of the file's fastest lap are not folded
# into the tire model (out laps, pit stops, spins, traffic). Their samples
# are not wrong as such, but they are mostly far below the limit and on
# out laps the tyres are cold.
TIRE_LAP_MAX_RATIO = 1.15


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

    # Probabilistic tire model: raw per-sample observations, front/rear
    # axle, accumulated across every analyzed lap. See tire_model.py for
    # how these become peak/efficient_range/confidence on read.
    front_axle_obs: AxleSlipObservations = field(default_factory=AxleSlipObservations)
    rear_axle_obs: AxleSlipObservations = field(default_factory=AxleSlipObservations)
    tire_model_version: int = TIRE_MODEL_VERSION
    # "<file name>#<lap number>" for every lap already folded into the tire
    # model, so re-analyzing a file (or another lap of it) never adds the
    # same samples twice.
    tire_model_laps: list[str] = field(default_factory=list)
    # per axle {"slope","intercept","r","n"}: Susp Pos -> Susp Force, learned
    # from recordings that had both (tire_model.fit_deflection_calibration)
    deflection_calibration: dict = field(default_factory=dict)

    # cap each list so the file doesn't grow unbounded over months of use —
    # recent-ish history is what matters for "what's normal for me on this
    # car right now", not literally every lap ever recorded
    MAX_SAMPLES: int = 200

    def to_dict(self) -> dict:
        return {
            "car_name": self.car_name,
            "lap_count": self.lap_count,
            "abs_lockup_durations_s": self.abs_lockup_durations_s,
            "tc_active_fractions": self.tc_active_fractions,
            "peak_combined_g_values": self.peak_combined_g_values,
            "peak_front_slip_angle_values": self.peak_front_slip_angle_values,
            "peak_rear_slip_angle_values": self.peak_rear_slip_angle_values,
            "max_wheel_slip_ratio_values": self.max_wheel_slip_ratio_values,
            "tire_model_version": self.tire_model_version,
            "tire_model_laps": self.tire_model_laps,
            "deflection_calibration": self.deflection_calibration,
            "front_axle_obs": self.front_axle_obs.to_dict(),
            "rear_axle_obs": self.rear_axle_obs.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GripProfile":
        if d.get("tire_model_version", 1) != TIRE_MODEL_VERSION:
            # see TIRE_MODEL_VERSION: start the tire model over, keep the rest
            d = {k: v for k, v in d.items()
                 if k not in ("front_axle_obs", "rear_axle_obs", "tire_model_laps")}
        return cls(
            car_name=d["car_name"],
            lap_count=d.get("lap_count", 0),
            abs_lockup_durations_s=d.get("abs_lockup_durations_s", []),
            tc_active_fractions=d.get("tc_active_fractions", []),
            peak_combined_g_values=d.get("peak_combined_g_values", []),
            peak_front_slip_angle_values=d.get("peak_front_slip_angle_values", []),
            peak_rear_slip_angle_values=d.get("peak_rear_slip_angle_values", []),
            max_wheel_slip_ratio_values=d.get("max_wheel_slip_ratio_values", []),
            front_axle_obs=AxleSlipObservations.from_dict(d.get("front_axle_obs", {})),
            rear_axle_obs=AxleSlipObservations.from_dict(d.get("rear_axle_obs", {})),
            tire_model_laps=d.get("tire_model_laps", []),
            deflection_calibration=d.get("deflection_calibration", {}),
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

    def ingest_tire_laps(self, laps: list, file_key: str) -> list[int]:
        """
        Folds every usable lap of one recording into the tire model, once.
        PREVIOUSLY only the analyzed lap was added (one lap per /analyze, so
        an 8-lap file contributed 1 lap). Now all laps within
        TIRE_LAP_MAX_RATIO of the file's fastest are added, keyed by
        "<file>#<lap>" so repeat analyses add nothing. Returns the lap
        numbers added.
        """
        valid = [l for l in laps if l.lap_time_s < 600]
        if not valid:
            return []
        # BUG FIX (10/18/2026): the cutoff reference was min(lap_time_s), which
        # the unfinished last lap or a mid-lap join usually won, so only that
        # partial lap got in. Reference = fastest real lap; partial laps are
        # still ingested (their steady-state samples are as good as any).
        fastest = reference_lap_time(valid) or min(l.lap_time_s for l in valid)
        # Learn/refresh the deflection->force calibration from any lap that
        # has Susp Force, before extracting, so this file can use it too.
        for lap in valid:
            cal = fit_deflection_calibration(lap)
            if cal:
                self.deflection_calibration = cal
                break
        # reference load from the whole file incl. slow laps (the out lap /
        # pit lane is usually the only low-speed straight-line running)
        load_ref = file_load_reference(valid, self.deflection_calibration)
        added = []
        for lap in valid:
            key = f"{file_key}#{lap.lap_number}"
            if key in self.tire_model_laps or lap.lap_time_s > fastest * TIRE_LAP_MAX_RATIO:
                continue
            obs = extract_axle_observations_from_lap(lap, self.deflection_calibration, load_ref)
            self.front_axle_obs.extend(obs["front"])
            self.rear_axle_obs.extend(obs["rear"])
            self.tire_model_laps.append(key)
            added.append(lap.lap_number)
        self.tire_model_laps = self.tire_model_laps[-2000:]
        return added

    def compute_baselines(self) -> dict:
        """
        Percentile-based baselines for prompt injection. Returns None for
        any metric with too few samples to be meaningful (avoid treating a
        single lap's noise as an established norm).

        Also includes the accumulated probabilistic tire model
        (front_tire_model / rear_tire_model, in the requested peak /
        efficient_range / confidence / sample_count shape) and, when the
        model has enough history, the observed temperature/pressure/load
        relationships. Both are recomputed fresh from raw observations on
        every call — nothing here is a stale cached fit.
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

        front_model = build_axle_tire_model(self.front_axle_obs)
        rear_model = build_axle_tire_model(self.rear_axle_obs)

        return {
            "car_name": self.car_name,
            "total_laps_analyzed": self.lap_count,
            "tire_model_laps": len(self.tire_model_laps),
            "abs_lockup_duration_s": _percentiles(self.abs_lockup_durations_s),
            "tc_active_fraction": _percentiles(self.tc_active_fractions),
            "peak_combined_g": _percentiles(self.peak_combined_g_values),
            "peak_front_slip_angle_deg": _percentiles(self.peak_front_slip_angle_values),
            "peak_rear_slip_angle_deg": _percentiles(self.peak_rear_slip_angle_values),
            "max_wheel_slip_ratio": _percentiles(self.max_wheel_slip_ratio_values),
            "front_tire_model": front_model,
            "rear_tire_model": rear_model,
            "front_temp_pressure_load": (
                build_temp_pressure_load_relationships(self.front_axle_obs)
                if front_model["sample_count"] >= 300 else None
            ),
            "rear_temp_pressure_load": (
                build_temp_pressure_load_relationships(self.rear_axle_obs)
                if rear_model["sample_count"] >= 300 else None
            ),
        }


def _profile_path(car_name: str) -> Path:
    # sanitize car name into a safe filename
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in car_name)
    return PROFILES_DIR / f"{safe_name}.json"


def load_profile(car_name: str) -> GripProfile:
    path = _profile_path(car_name)
    if not path.exists():
        return GripProfile(car_name=car_name)
    with open(path, "r") as f:
        return GripProfile.from_dict(json.load(f))


def save_profile(profile: GripProfile) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.car_name)
    # No indent: indent=2 puts each of the ~140k observation floats on its
    # own line (the v1 WRT profile was 538 KB for one lap).
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f)
