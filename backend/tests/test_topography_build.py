"""Per-track topography build (build_topography), its file IO and the CLI.

Uses the synthetic hilly ring (synthetic_track.py), whose ground is known."""

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

from app import telemetry
from app import track_topography as tt
from synthetic_track import RING_LENGTH_M, truth, write_hilly_session

L = RING_LENGTH_M
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def ref_dir(tmp_path, monkeypatch):
    d = tmp_path / "ref"
    d.mkdir()
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(d))
    monkeypatch.setattr(tt, "slug_for_track", lambda n: "synthetic-ring")
    tt._CACHE.clear()
    return d


def test_build_and_load_roundtrip(tmp_path, ref_dir):
    a = write_hilly_session(tmp_path / "a.duckdb")
    b = write_hilly_session(tmp_path / "b.duckdb", with_pose=False)
    topo = tt.build_topography([a, b], "synthetic-ring", "Synthetic Hill Ring")
    assert topo["schema"] == 1 and topo["slug"] == "synthetic-ring"
    lay0 = topo["layouts"][0]
    assert lay0["pose"] == {"files_with_pose": 1, "files_refused": 1}
    assert lay0["lateral_sign"]["value"] == 1
    p = tt.save_topography(topo)
    assert p == ref_dir / "synthetic-ring.topography.json"
    topo2, lay = tt.load_topography("Synthetic Hill Ring", L)
    assert topo2["slug"] == "synthetic-ring"
    assert lay["has_heights"] and lay["coverage"] > 0.9
    assert lay["sources"]["files"] == 2
    assert topo["datum_m"] == pytest.approx(min(lay["h"]), abs=0.01)   # hill dips to ~-8 m
    s = np.arange(5.0, L, 7.0)
    for la in (0.0, 1.0):
        err = tt.lookup_height(lay, s, np.full_like(s, la)) - truth(s, np.full_like(s, la))
        assert np.max(np.abs(err)) < 0.08, (la, np.max(np.abs(err)))


def test_load_topography_length_mismatch(tmp_path, ref_dir):
    a = write_hilly_session(tmp_path / "a.duckdb")
    tt.save_topography(tt.build_topography([a], "synthetic-ring", "Synthetic Hill Ring"))
    assert tt.load_topography("Synthetic Hill Ring", L * 1.1) is None
    assert tt.load_topography("Synthetic Hill Ring", L * 1.02) is not None
    assert tt.load_topography("Synthetic Hill Ring", None) is not None


def test_load_topography_no_file(ref_dir):
    assert tt.load_topography("Synthetic Hill Ring", L) is None


def test_load_topography_caches_by_mtime(tmp_path, ref_dir):
    a = write_hilly_session(tmp_path / "a.duckdb")
    p = tt.save_topography(tt.build_topography([a], "synthetic-ring", "X"))
    first = tt.load_topography("Synthetic Hill Ring", L)[0]
    assert tt.load_topography("Synthetic Hill Ring", L)[0] is first
    data = json.loads(p.read_text())
    data["display_name"] = "Changed"
    import os
    p.write_text(json.dumps(data))
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
    assert tt.load_topography("Synthetic Hill Ring", L)[0]["display_name"] == "Changed"


def test_build_without_pose_files_is_paint_only(tmp_path, ref_dir):
    a = write_hilly_session(tmp_path / "a.duckdb", with_pose=False)
    topo = tt.build_topography([a], "synthetic-ring", "Synthetic Hill Ring")
    lay = topo["layouts"][0]
    assert lay["has_heights"] is False
    assert topo["datum_m"] == 0.0
    assert lay["paint"]
    assert lay["pose"]["files_with_pose"] == 0


def test_build_nothing_usable_returns_none(tmp_path, ref_dir):
    a = write_hilly_session(tmp_path / "a.duckdb", with_pose=False, with_surface=False)
    assert tt.build_topography([a], "synthetic-ring", "Synthetic Hill Ring") is None


def test_layouts_split_by_length(tmp_path, ref_dir):
    """Laps 3 % apart share a layout; the same file set with a different
    ring length would not. Two files of one ring give one layout."""
    a = write_hilly_session(tmp_path / "a.duckdb")
    b = write_hilly_session(tmp_path / "b.duckdb", lap_offsets=(0.0, 1.0))
    topo = tt.build_topography([a, b], "synthetic-ring", "X")
    assert len(topo["layouts"]) == 1
    assert topo["layouts"][0]["length_m"] == pytest.approx(L, rel=0.01)
    assert topo["layouts"][0]["sources"]["laps"] == 5


def test_other_cars_add_coverage(tmp_path, ref_dir):
    plain = write_hilly_session(tmp_path / "p.duckdb", lap_offsets=(0.0,))
    crowd = write_hilly_session(tmp_path / "c.duckdb", lap_offsets=(0.0,), other_cars=3)
    t0 = tt.build_topography([plain], "synthetic-ring", "X")["layouts"][0]
    t1 = tt.build_topography([crowd], "synthetic-ring", "X")["layouts"][0]
    w0 = np.array(t0["lat_max"]) - np.array(t0["lat_min"])
    w1 = np.array(t1["lat_max"]) - np.array(t1["lat_min"])
    assert w1.mean() > w0.mean() + 0.5
    assert t1["sources"]["other_car_samples"] > 0
    assert t0["sources"]["other_car_samples"] == 0


def test_lap_dist_raw_matches_game_with_crossing_between_rows(tmp_path):
    """BUG FIX / PREVIOUSLY: probes took telemetry's re-zeroed Lap Dist, which
    loses the 10 Hz step the wrap falls in (here the first row after each
    crossing reads ~3 m), so topography sat metres off the game's own lap
    distance that /trackmap looks up with. Fix: LapTelemetry.lap_dist_raw."""
    path = write_hilly_session(tmp_path / "x.duckdb", cross_offset_ticks=-8)
    laps = [l for l in telemetry.load_telemetry_file(path, grid_hz=50.0) if l.is_complete]
    assert len(laps) == 3
    con = duckdb.connect(str(path), read_only=True)
    raw = np.array([r[0] for r in con.execute('SELECT value FROM "Lap Dist" ORDER BY rowid').fetchall()])
    _, t_slow = telemetry.session_time_axes(con)
    starts = {int(n): float(ts) for ts, n in con.execute('SELECT ts, value FROM "Lap"').fetchall()}
    con.close()
    # the crossing really falls between rows: first row after it reads 2-5 m
    after = [raw[np.searchsorted(t_slow, starts[i])] for i in (1, 2)]
    assert all(2.0 < a < 5.0 for a in after), after
    for lap in laps:
        assert lap.channels["Lap Dist"][0] == 0.0      # existing behaviour untouched
        assert lap.lap_dist_raw is not None and len(lap.lap_dist_raw) == len(lap.time_s)
        t_abs = lap.time_s + starts[lap.lap_number]
        i = np.clip(np.searchsorted(t_slow, t_abs, side="right") - 1, 0, len(raw) - 2)
        seg_ok = (raw[i + 1] - raw[i]) > 0            # skip the interval holding the wrap
        want = raw[i] + (raw[i + 1] - raw[i]) * (t_abs - t_slow[i]) / (t_slow[i + 1] - t_slow[i])
        assert seg_ok.sum() > 0.9 * len(seg_ok)
        assert np.max(np.abs(lap.lap_dist_raw[seg_ok] - want[seg_ok])) < 0.5


def test_probes_use_game_lap_dist(tmp_path):
    path = write_hilly_session(tmp_path / "x.duckdb", cross_offset_ticks=-8)
    laps = [l for l in telemetry.load_telemetry_file(path, grid_hz=50.0) if l.is_complete]
    lap = laps[1]
    ps = tt.lap_probes(lap, lateral_sign=1, basis=tt.lap_pose_basis(lap))
    want = np.array([truth(s % L, la) for s, la in zip(ps.s, ps.lat)])
    # a 3 m s error on the 4 % grade would be ~0.12 m; the probes' own noise is a few cm
    assert np.median(np.abs(ps.y - want)) < 0.04
    assert np.percentile(np.abs(ps.y - want), 95) < 0.12


def test_cli_topography_only(tmp_path, ref_dir, monkeypatch, capsys):
    sess = tmp_path / "sessions"
    sess.mkdir()
    write_hilly_session(sess / "Fuji_Speedway-Synthetic_Car-2026-10-25T10_00_00Z.duckdb",
                        track_name="Fuji Speedway")
    monkeypatch.undo()                           # keep TRACK_REFERENCE_DIR below
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(ref_dir))
    monkeypatch.syspath_prepend(str(TOOLS))
    import build_track_reference as btr
    monkeypatch.setattr(sys, "argv", ["x", "--topography-only", "--sessions", str(sess),
                                      "--track", "fuji"])
    assert btr.main() == 0
    out = capsys.readouterr().out
    assert "topography: fuji layouts=1" in out and "(1 with pose)" in out
    assert (ref_dir / "fuji.topography.json").exists()
    assert not (ref_dir / "fuji.json").exists()      # no OSM/corner work
