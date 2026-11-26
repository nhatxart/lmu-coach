"""Ideal lap corner rows (UX round 11/26/2026).

Without an aligned track reference the rows came from runs of high path
curvature, merged when their padding touched: on COTA one row covered T1-T9
and another T12-T19. Now a run is split at the map's own corner labels (the
track_db points the map draws T1..T20 at, in game lap distance), one row per
label or per small group of labels a few tens of metres apart. A reference
whose corners sit far from the map's labels (not aligned with lap distance)
is not used. A row's Limit is the binding code where its time is gained, not
the code at its slowest point, so a gaining row never says "at the limit
already" (measured) or "your lap is faster here" (driven).
"""
import numpy as np
import pytest

from app import ideal_lap as IL
from app import track_reference

L = 1200.0


def _line():
    """2 m grid on a 1200 m lap; tight (r 60 m) in 400-600 and 1000-1190."""
    s = np.arange(0.0, L, 2.0)
    ks = np.where(((s >= 400) & (s <= 600)) | ((s >= 1000) & (s <= 1190)), 1 / 60.0, 0.0)
    return {"s": s, "ks": ks, "lap_dist_mono": s}


LABELS = [(0.0, "Start/Finish Line"), (470.0, "T1 Apex (North)"), (540.0, "T2 Apex"),
          (1090.0, "T3 Apex"), (1110.0, "T4 Apex (Hairpin)")]


@pytest.fixture
def no_reference(monkeypatch):
    monkeypatch.setattr(track_reference, "load_reference", lambda *a, **k: None)


def test_curvature_runs_are_split_at_the_maps_labels(no_reference):
    w, src = IL._corner_windows("Synthetic Hill Ring", L, _line(), L, map_labels=LABELS)
    assert src == "map_labels"
    names = [n for _, n, _, _ in w]
    # T1 and T2 are 70 m apart: two rows; T3 and T4 are 20 m apart: one group
    assert names == ["T1", "T2", "T3/T4"]
    t1, t2, t34 = w
    assert t1[3] == pytest.approx(t2[2]) == pytest.approx(505.0)       # split at the midpoint
    assert t1[2] < 470 < t1[3] and t2[2] < 540 < t2[3]
    assert t1[2] == pytest.approx(360.0, abs=3) and t2[3] == pytest.approx(640.0, abs=3)  # the run's padded ends
    assert t34[2] < 1090 and t34[3] > 1110


def test_a_label_outside_every_run_still_gets_its_row(no_reference):
    w, _ = IL._corner_windows("x", L, _line(), L, map_labels=LABELS + [(250.0, "T0 Kink")])
    row = next(r for r in w if r[1] == "T0")
    assert row[2] < 250 < row[3] and row[3] - row[2] <= 100


def test_without_labels_the_curvature_runs_stay(no_reference):
    w, src = IL._corner_windows("x", L, _line(), L, map_labels=[])
    assert src == "curvature" and [n for _, n, _, _ in w] == ["C1", "C2"]


def _ref(offset):
    cs = [{"number": i + 1, "display": f"Turn {i + 1}", "lmu_apex_m": d + offset,
           "lmu_start_m": d + offset - 30, "lmu_end_m": d + offset + 30}
          for i, d in enumerate((470.0, 540.0, 1090.0, 1110.0))]
    return {"slug": "x", "display_name": "x", "layout": {"corners": cs}}


def test_a_reference_far_from_the_maps_labels_is_not_used(monkeypatch):
    """Offset by 300 m: not aligned with the game's lap distance."""
    monkeypatch.setattr(track_reference, "load_reference", lambda *a, **k: _ref(300.0))
    w, src = IL._corner_windows("x", L, _line(), L, map_labels=LABELS)
    assert src == "map_labels" and [n for _, n, _, _ in w] == ["T1", "T2", "T3/T4"]


def test_an_aligned_reference_is_used(monkeypatch):
    monkeypatch.setattr(track_reference, "load_reference", lambda *a, **k: _ref(10.0))
    w, src = IL._corner_windows("x", L, _line(), L, map_labels=LABELS)
    assert src == "track_reference" and len(w) == 4


def test_the_limit_is_where_the_time_is():
    codes = np.array(["measured"] * 6 + ["front"] * 3 + ["brake"] * 3, dtype=object)
    gain = np.array([0.0] * 6 + [0.01] * 3 + [0.002] * 3)
    # the slowest point (k = 0) is clamped; the time is under "front"
    assert IL._row_binding(codes, gain, 0) == "front"
    # no time anywhere: the slowest point's code, as before
    assert IL._row_binding(codes, np.zeros(12), 0) == "measured"
    # time only where the ideal runs your speed cannot be; still never a clamp label
    # when another code has any gain
    g2 = np.array([0.0] * 6 + [0.0] * 3 + [0.001] * 3)
    assert IL._row_binding(codes, g2, 0) == "brake"


def test_a_gaining_row_never_shows_the_clamp_or_driven_label(tmp_path, monkeypatch):
    """The endpoint on the synthetic ring (driven at 0.6 speed, a realistic car)."""
    from fastapi.testclient import TestClient
    from urllib.parse import quote
    from app import main, grip_profile
    from synthetic_track import write_hilly_session
    from test_ideal_endpoint import FakeStore
    from test_ideal_solver import _car
    f = write_hilly_session(tmp_path / "Synthetic Hill Ring-Synthetic Car #1-2026-11-01T10_00_00Z.duckdb",
                            lap_offsets=(0.0, 1.0, -1.0), slip_angles=True, speed_wave=0.15, speed_scale=0.6)
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", f.parent)
    monkeypatch.setattr(main, "REFERENCE_LAPS_DIR", tmp_path / "none")
    monkeypatch.setattr(grip_profile, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(main, "_ideal_store", lambda: FakeStore(_car(mu=(1.5, 1.6), power=0.6)))
    IL._CACHE.clear()
    d = TestClient(main.app).get(f"/sessions/sessions/{quote(f.name)}/ideal-lap?lap_number=1").json()
    rows = [c for c in d["corners"] if c["kind"] == "corner"]
    assert rows and any(c["gain_s"] > 0.01 for c in rows)
    for c in rows:
        if c["gain_s"] > 0.01:
            assert c["binding"] not in ("measured", "driven"), c
        assert "binding_slowest" in c
