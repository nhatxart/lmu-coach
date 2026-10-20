"""backend/tools/ingest_sessions.py: livery -> model profile migration."""

import sys
from pathlib import Path

from app import grip_profile as gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ingest_sessions as ing  # noqa: E402


def _livery(name, abs_vals, laps, car_class="LMGT3"):
    return gp.GripProfile(car_name=name, car_class=car_class, lap_count=laps,
                          abs_lockup_durations_s=list(abs_vals),
                          peak_combined_g_values=[float(laps)] * laps)


def test_migrate_merges_baselines():
    model = gp.GripProfile(car_name="BMW M4 LMGT3", abs_lockup_durations_s=[9.0], lap_count=1)
    a = _livery("Team WRT 2026 _69_LP", [1.0, 2.0], 4)
    b = _livery("Team WRT 2025 _31_LP", [3.0], 2)
    ing.merge_baselines(model, [a, b])
    assert model.abs_lockup_durations_s == [9.0, 1.0, 2.0, 3.0]
    assert model.lap_count == 1 + 4 + 2
    assert len(model.peak_combined_g_values) == 6
    assert model.car_class == "LMGT3"


def test_migrate_caps_to_max_samples():
    model = gp.GripProfile(car_name="M", abs_lockup_durations_s=[0.0] * 150)
    ing.merge_baselines(model, [_livery("L", [1.0] * 150, 1)])
    assert len(model.abs_lockup_durations_s) == model.MAX_SAMPLES
    assert model.abs_lockup_durations_s[-1] == 1.0     # newest kept
    assert model.abs_lockup_durations_s[0] == 0.0


def test_livery_stem_maps_to_model_via_table(tmp_path):
    (tmp_path / "Team WRT 2026 _69_LP.json").write_text("{}")
    (tmp_path / "BMW M4 LMGT3.json").write_text("{}")
    empty = tmp_path / "sessions"
    empty.mkdir()
    assert ing.livery_models(empty, tmp_path) == {"Team WRT 2026 _69_LP": "BMW M4 LMGT3"}
