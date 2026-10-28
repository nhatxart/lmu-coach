"""The synthetic ring's per-wheel slip angles (write_hilly_session
slip_angles=True), as /trackmap sends them to the chase view's tyre slip
arrows: known values on the constant-radius corner, and the kinematic
steer estimate the UI makes from them recovers the written steer angle."""

from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, track_topography, trackmap_elevation

from synthetic_track import (RADIUS_M, SLIP_A_M, SLIP_FRONT_DEG, SLIP_HALF_TRACK_F,
                             STEER_LOCK_DEG, SPEED_MS, write_hilly_session)


@pytest.fixture(autouse=True)
def reference_dir(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    ref.mkdir()
    monkeypatch.setenv("TRACK_REFERENCE_DIR", str(ref))
    monkeypatch.setattr(track_topography, "slug_for_track", lambda name: None)
    track_topography._CACHE.clear()
    trackmap_elevation._LAP_CACHE.clear()


def _trackmap(path, monkeypatch, lap=0):
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", path.parent)
    r = TestClient(main.app).get(
        f"/sessions/sessions/{quote(path.name)}/trackmap", params={"lap_number": lap})
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _mid_corner(d):
    """Driven-line indices in the middle 60 % of the first semicircle
    (s 440..560 m: past the 40 m slip build-up, constant radius)."""
    n = len(d["driven_line"]["x"])
    f = np.arange(n) / (n - 1)
    return np.where((f > 440 / 1200) & (f < 560 / 1200))[0]


def test_slip_angles_absent_by_default(tmp_path, monkeypatch):
    d = _trackmap(write_hilly_session(tmp_path / "a.duckdb"), monkeypatch)
    assert all(v == [] for v in d["slip_angle"].values())


def test_slip_angles_known_in_the_corner(tmp_path, monkeypatch):
    d = _trackmap(write_hilly_session(tmp_path / "s.duckdb", slip_angles=True), monkeypatch)
    sa = {k: np.asarray(v) for k, v in d["slip_angle"].items()}
    i = _mid_corner(d)
    assert len(i) > 20          # 10 Hz position: ~300 points per 30 s lap
    r = SPEED_MS / RADIUS_M                                   # left turn, rad/s
    th_f = np.degrees(np.arctan2(-r * SLIP_A_M, SPEED_MS))  # front travel, right-positive (about -1.2)
    th_r = -th_f                                             # rear swings out: right of the body axis
    # fronts: outward (positive) by SLIP_FRONT_DEG plus the per-wheel
    # track-width term (hundredths of a degree); rears: their travel angle
    for k in ("fl", "fr"):
        assert np.allclose(sa[k][i], SLIP_FRONT_DEG, atol=0.06), (k, sa[k][i].min(), sa[k][i].max())
    for k in ("rl", "rr"):
        assert np.allclose(sa[k][i], th_r, atol=0.06), (k, sa[k][i].min(), sa[k][i].max())
    # straights: no slip
    n = len(d["driven_line"]["x"])
    straight = np.arange(int(0.05 * n), int(0.28 * n))
    assert np.abs(sa["fl"][straight]).max() < 0.05


def test_kinematic_steer_estimate_recovers_written_steer(tmp_path, monkeypatch):
    """The UI's estimate (index.html steerSeries): delta = mean over the two
    fronts of atan2(V sin(-beta) + s r a, V cos(beta) + s r w_wheel) minus
    the wheel's slip angle; s (the yaw-rate sign) by the best correlation
    with Steering Pos. On this file Yaw Rate is left-positive, so s = -1."""
    d = _trackmap(write_hilly_session(tmp_path / "s.duckdb", slip_angles=True), monkeypatch)
    dl = d["driven_line"]
    v = np.asarray(dl["speed"]) / 3.6
    r = np.asarray(dl["yaw"])
    beta = np.radians(np.asarray(dl["body_slip_deg"])) if dl["body_slip_deg"] else np.zeros_like(v)
    sa = {k: np.radians(np.asarray(d["slip_angle"][k])) for k in ("fl", "fr")}
    steer = np.asarray(dl["steering"])

    def est(s):
        fl = np.arctan2(v * np.sin(-beta) + s * r * SLIP_A_M, v * np.cos(beta) + s * r * SLIP_HALF_TRACK_F) - sa["fl"]
        fr = np.arctan2(v * np.sin(-beta) + s * r * SLIP_A_M, v * np.cos(beta) - s * r * SLIP_HALF_TRACK_F) - sa["fr"]
        return np.degrees((fl + fr) / 2)

    c = {s: np.corrcoef(est(s), steer)[0, 1] for s in (1, -1)}
    assert c[-1] > 0.99 and c[-1] > c[1] + 0.02, c
    i = _mid_corner(d)
    want = np.degrees(np.arctan2(-(SPEED_MS / RADIUS_M) * SLIP_A_M, SPEED_MS)) - SLIP_FRONT_DEG
    assert np.allclose(est(-1)[i], want, atol=0.1)
    assert np.allclose(steer[i] * STEER_LOCK_DEG, want, atol=0.1)
