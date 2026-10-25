"""Track topography: ground probes from pose + ride height, per-track grid
fit, kerb paint and lookups, checked against a synthetic hilly, banked ring
whose ground is known exactly (synthetic_track.truth)."""

import numpy as np
import pytest

from app import telemetry
from app import track_topography as tt
from synthetic_track import (RING_LENGTH_M, true_bank, truth,
                             write_hilly_session)

L = RING_LENGTH_M


def _probes_from_file(path, lateral_sign=1):
    laps = [l for l in telemetry.load_telemetry_file(path) if l.is_complete]
    out = []
    for lap in laps:
        basis = tt.lap_pose_basis(lap)
        out.append(tt.lap_probes(lap, 0.0, lateral_sign, basis=basis))
    return laps, out


@pytest.fixture(scope="module")
def hilly(tmp_path_factory):
    path = write_hilly_session(tmp_path_factory.mktemp("topo") / "hill.duckdb")
    laps, probes = _probes_from_file(path)
    return laps, probes, tt.build_layout_topography(probes, L)


def test_flat_ring_fits_zero():
    rng = np.random.default_rng(1)
    n = 20000
    s = rng.uniform(0, L, n)
    lat = rng.uniform(-3, 3, n)
    ps = tt.ProbeSet(s=s, lat=lat, y=np.zeros(n), w=np.ones(n),
                     paint_s=np.array([]), paint_lat=np.array([]),
                     paint_kind=np.array([], dtype=int))
    lay = tt.build_layout_topography([ps], L)
    assert lay["has_heights"] is True
    assert lay["rows"] == int(np.ceil(L / tt.ROW_STEP_M))
    assert np.max(np.abs(lay["h"])) < 0.01
    assert np.max(np.abs(lay["bank"])) < 0.005
    assert lay["coverage"] == pytest.approx(1.0)


def test_steep_grade_is_not_scatter():
    """15 % grade, three lines sampled every 2 m at different phases (like
    laps at 20 Hz): the row plane must model the grade, otherwise the outlier
    cut eats the probes and the grade leaks into the bank."""
    Lr = 600.0
    amp = 0.15 * Lr / (2 * np.pi)
    s_parts, lat_parts = [], []
    for k, off in enumerate((0.0, 1.5, -1.5)):
        s0 = np.arange(0.0, Lr, 2.0) + 0.7 * k
        for d_s, d_lat in ((1.45, -0.8), (1.45, 0.8), (-1.45, -0.8), (-1.45, 0.8)):
            s_parts.append(s0 + d_s)
            lat_parts.append(np.full_like(s0, off + d_lat))
    s, lat = np.concatenate(s_parts), np.concatenate(lat_parts)
    y = amp * np.sin(2 * np.pi * s / Lr) + 0.05 * lat
    n = len(s)
    ps = tt.ProbeSet(s=s, lat=lat, y=y, w=np.ones(n), paint_s=np.array([]),
                     paint_lat=np.array([]), paint_kind=np.array([], dtype=int))
    lay = tt.build_layout_topography([ps], Lr)
    q = np.arange(0.0, Lr, 3.0)
    for la in (-2.0, 0.0, 2.0):
        err = tt.lookup_height(lay, q, np.full_like(q, la)) - (
            amp * np.sin(2 * np.pi * q / Lr) + 0.05 * la)
        assert np.max(np.abs(err)) < 0.05, (la, np.max(np.abs(err)))
    assert np.max(np.abs(np.asarray(lay["bank"]) - 0.05)) < 0.005


def test_synthetic_laps_load_cleanly(hilly):
    laps, probes, _ = hilly
    assert [l.lap_number for l in laps] == [0, 1, 2]
    for lap in laps:
        assert lap.lap_time_s == pytest.approx(30.0, abs=0.05)
        assert tt.lap_pose_basis(lap) is not None
        assert lap.channels["Lap Dist"][-1] == pytest.approx(L, abs=8.0)
    assert all(len(p.s) > 1000 and np.all(p.w == 1.0) for p in probes)
    # body roll/pitch: each wheel has its own ride height, so wheel order matters
    rh = np.stack([laps[0].channels[f"Ride Height {w}"] for w in telemetry.WHEEL_LABELS])
    assert np.max(np.abs(rh[0] - rh[1])) > 0.03     # FL vs FR (roll)
    assert np.max(np.abs(rh[0] - rh[2])) > 0.02     # FL vs RL (pitch)
    assert rh.min() > 0.0


def _planar_probes(rng, n=6000, length=600.0):
    s = rng.uniform(0, length, n)
    lat = rng.uniform(-3, 3, n)
    y = 2.0 + 0.06 * lat     # level at 2.0 m, bank 0.06
    return s, lat, y


def test_outlier_spikes_are_rejected():
    """~5 % of probes 0.5 m too high (kerb strikes, airborne ticks) must not
    move the plane: reject-and-refit."""
    rng = np.random.default_rng(7)
    s, lat, y = _planar_probes(rng)
    spike = rng.random(len(s)) < 0.05
    y = y + np.where(spike, 0.5, 0.0)
    ps = tt.ProbeSet(s=s, lat=lat, y=y, w=np.ones(len(s)), paint_s=np.array([]),
                     paint_lat=np.array([]), paint_kind=np.array([], dtype=int))
    lay = tt.build_layout_topography([ps], 600.0)
    assert np.max(np.abs(np.asarray(lay["h"]) - 2.0)) < 0.01
    assert np.max(np.abs(np.asarray(lay["bank"]) - 0.06)) < 0.005


def test_outlier_spikes_in_wheel_probes(hilly):
    _, probes, _ = hilly
    rng = np.random.default_rng(11)
    spiked = []
    for p in probes:
        bump = np.where(rng.random(len(p.y)) < 0.05, 0.5, 0.0)
        spiked.append(tt.ProbeSet(p.s, p.lat, p.y + bump, p.w,
                                  p.paint_s, p.paint_lat, p.paint_kind))
    lay = tt.build_layout_topography(spiked, L)
    s = np.arange(0.0, L, 5.0)
    for lat in (-3.0, 0.0, 3.0):
        err = tt.lookup_height(lay, s, np.full_like(s, lat)) - truth(s, lat)
        assert np.max(np.abs(err)) < (0.03 if lat == 0.0 else 0.08)
    _, bank = tt.grade_bank(lay, np.arange(1040.0, 1160.0, 2.0))
    assert float(np.mean(bank)) == pytest.approx(0.08, abs=0.01)


def test_hill_and_bank_recovered(hilly):
    _, _, lay = hilly
    assert lay is not None
    s = np.arange(0.0, L, 5.0)
    worst = 0.0
    for lat in (-3.0, 0.0, 3.0):
        got = tt.lookup_height(lay, s, np.full_like(s, lat))
        want = np.array([truth(si, lat) for si in s])
        worst = max(worst, float(np.max(np.abs(got - want))))
        if lat == 0.0:
            # tighter on the centre path: a dropped Ride Height subtraction
            # (~0.05 m) must not pass
            assert np.max(np.abs(got - want)) < 0.03
    assert worst < 0.08, worst
    mid = np.arange(1040.0, 1160.0, 2.0)
    _, bank = tt.grade_bank(lay, mid)
    assert float(np.mean(bank)) == pytest.approx(0.08, abs=0.01)
    assert np.all(true_bank(mid) == pytest.approx(0.08))

    # The pooled fit averages a swapped wheel's ride height away (laps on
    # different lines dominate the bank), so check the probes themselves and
    # a single lap whose bank comes from its own wheel pairs only. Ride
    # heights differ per wheel by 0.025 m (pitch) to 0.042 m (roll).
    _, probes, _ = hilly
    err = np.concatenate([p.y - truth(np.mod(p.s, L), p.lat) for p in probes])
    assert np.percentile(np.abs(err), 99) < 0.015
    _, bank0 = tt.grade_bank(tt.build_layout_topography(probes[:1], L), mid)
    assert float(np.mean(bank0)) == pytest.approx(0.08, abs=0.01)


def test_grade_follows_hill(hilly):
    _, _, lay = hilly
    s = np.arange(0.0, L, 10.0)
    grade, _ = tt.grade_bank(lay, s)
    want = 8.0 * 2 * np.pi / L * np.cos(2 * np.pi * s / L)
    assert np.max(np.abs(grade - want)) < 0.005


def test_paint_kerb_cells(hilly):
    _, _, lay = hilly
    assert lay["paint"], "the kerb excursion should leave paint"
    kerb = [c for c in lay["paint"] if c[2] == 5]
    assert kerb
    assert all(c[1] * lay["paint_cell_m"] > 5.0 for c in lay["paint"])
    rows = {c[0] for c in kerb}
    assert min(rows) * lay["row_step_m"] >= 580 and max(rows) * lay["row_step_m"] <= 702
    s = np.array([620.0, 620.0, 300.0])
    lat = np.array([5.8, 0.0, 5.8])
    assert list(tt.paint_at(lay, s, lat)) == [5, 0, 0]


def test_lookup_wraps_start_finish(hilly):
    _, _, lay = hilly
    a = float(tt.lookup_height(lay, np.array([L - 0.5]), np.array([0.0]))[0])
    b = float(tt.lookup_height(lay, np.array([0.5]), np.array([0.0]))[0])
    assert abs(a - b) < 0.05
    assert abs(a - truth(L - 0.5, 0.0)) < 0.08
    assert abs(b - truth(0.5, 0.0)) < 0.08
    # periodic: s beyond L and negative s wrap
    c = tt.lookup_height(lay, np.array([L + 0.5, -0.5]), np.array([0.0, 0.0]))
    assert c[0] == pytest.approx(b) and c[1] == pytest.approx(a)


def _surface(n, touching):
    surf = {w: np.zeros(n) for w in telemetry.WHEEL_LABELS}
    for w in touching:
        surf[w][:] = 5
    return surf


def test_detect_lateral_sign():
    n = 50
    path_lat = np.full(n, 5.5)
    edge = np.full(n, 6.0)
    sign, conf, ev = tt.detect_lateral_sign(path_lat, edge, _surface(n, ("FR", "RR")))
    assert sign == 1 and conf > 0.5, ev
    sign, conf, ev = tt.detect_lateral_sign(path_lat, edge, _surface(n, ("FL", "RL")))
    assert sign == -1 and conf > 0.5, ev
    # the mirror side: negative lat, left wheels touching -> positive = right
    sign, conf, _ = tt.detect_lateral_sign(-path_lat, -edge, _surface(n, ("FL", "RL")))
    assert sign == 1 and conf > 0.5
    sign, conf, ev = tt.detect_lateral_sign(path_lat, edge, _surface(n, ()))
    assert (sign, conf) == (1, 0.0) and ev.startswith("inconclusive")


def test_detect_lateral_sign_on_synthetic_file(hilly):
    laps, _, _ = hilly
    lap = laps[1]   # the 1.5 m lap with the kerb excursion
    surf = {w: lap.channels[f"Surface Type {w}"] for w in telemetry.WHEEL_LABELS}
    pl = lap.channels["Path Lateral"]
    # telemetry does not load Track Edge; the file's is +-6.0 signed like pl
    edge = np.where(pl < 0, -6.0, 6.0)
    sign, conf, _ = tt.detect_lateral_sign(pl, edge, surf)
    assert sign == 1 and conf > 0.5


def test_legacy_lap_gives_paint_only(tmp_path):
    path = write_hilly_session(tmp_path / "legacy.duckdb", with_pose=False)
    laps, probes = _probes_from_file(path)
    assert all(tt.lap_pose_basis(l) is None for l in laps)
    assert all(len(p.s) == 0 for p in probes)
    lay = tt.build_layout_topography(probes, L)
    assert lay is not None
    assert lay["has_heights"] is False
    assert lay["paint"]
    assert lay["coverage"] == 0.0


def test_no_data_gives_none(tmp_path):
    path = write_hilly_session(tmp_path / "bare.duckdb", with_pose=False,
                               with_surface=False)
    _, probes = _probes_from_file(path)
    assert tt.build_layout_topography(probes, L) is None


def test_centre_probe_without_basis(hilly):
    laps, _, _ = hilly
    ps = tt.lap_probes(laps[0], 0.0, 1, basis=None)
    assert len(ps.s) > 0 and np.all(ps.w == 0.5)
    err = ps.y - np.array([truth(s % L, lat) for s, lat in zip(ps.s, ps.lat)])
    assert np.median(np.abs(err)) < 0.05


def test_other_car_probes_weighted(tmp_path):
    rows = np.array([
        # ts, slot, x, y, z, lap_dist, path_lateral, in_pits
        [1.0, 3, 0.0, 10.30, 0.0, 100.0, 2.0, 0],
        [1.1, 3, 0.0, 10.30, 0.0, 1250.0, -1.0, 0],
        [1.0, 4, 0.0, 5.0, 0.0, 50.0, 0.0, 1],
    ])
    ps = tt.other_car_probes(rows, L, lateral_sign=-1)
    assert len(ps.s) == 2
    assert np.all(ps.w == 0.3)
    assert ps.y == pytest.approx([10.0, 10.0])
    assert ps.s == pytest.approx([100.0, 50.0])
    assert ps.lat == pytest.approx([-2.0, 1.0])
    assert len(ps.paint_s) == 0

    path = write_hilly_session(tmp_path / "others.duckdb", other_cars=2)
    import duckdb
    con = duckdb.connect(str(path), read_only=True)
    raw = np.array(con.execute('SELECT * FROM "Other Cars" ORDER BY ts, slot').fetchall(),
                   dtype=float)
    con.close()
    assert len(np.unique(raw[:, 1])) == 2
    ps = tt.other_car_probes(raw, L)
    err = ps.y - np.array([truth(s, lat) for s, lat in zip(ps.s, ps.lat)])
    assert np.max(np.abs(err)) < 1e-6
