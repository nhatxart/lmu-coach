"""Orientation-matrix convention detection (vehicle_pose.py) and the telemetry
loader picking up the new pose channels."""

import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import telemetry, vehicle_pose as vp  # noqa: E402
from synthetic_session import tick_times, write_session  # noqa: E402

PITCH, ROLL = 2.0, -3.0


def make_car(n_s=20.0, hz=100, speed=30.0, radius=150.0, pitch=PITCH, roll=ROLL):
    """A car circling at constant speed with a fixed true attitude.
    Returns world fwd/left/up (n,3) and the world velocity (vx, vz, speed)."""
    t = np.arange(int(n_s * hz)) / hz
    psi = speed / radius * t                      # course angle, 0 = +x, toward +z
    z0 = np.zeros_like(psi)
    fwd0 = np.stack([np.cos(psi), z0, np.sin(psi)], axis=1)
    left0 = np.stack([-np.sin(psi), z0, np.cos(psi)], axis=1)   # world left of (dx, dz) is (-dz, dx)
    up0 = np.stack([z0, z0 + 1.0, z0], axis=1)
    p, r = np.radians(pitch), np.radians(roll)
    fwd = np.cos(p) * fwd0 + np.sin(p) * up0
    up1 = np.cos(p) * up0 - np.sin(p) * fwd0
    # choose the mix so that left_y == sin(roll) exactly
    phi = np.arcsin(np.sin(r) / np.cos(p))
    left = np.cos(phi) * left0 + np.sin(phi) * up1
    up = np.cross(fwd, left)
    return fwd, left, up, speed * fwd0[:, 0], speed * fwd0[:, 2], np.full(len(t), speed)


def matrix(cols):
    """M whose columns are the given (n,3) vectors."""
    return np.stack(cols, axis=2)


def check(M, vx, vz, sp):
    basis = vp.detect_pose_basis(M, vx, vz, sp)
    assert basis is not None
    fwd, left, up = vp.basis_vectors(M, basis)
    pitch, roll = vp.pitch_roll_deg(fwd, left)
    assert np.allclose(pitch, PITCH, atol=0.05)
    assert np.allclose(roll, ROLL, atol=0.05)
    assert np.all(up[:, 1] > 0.9)
    return basis


def test_detect_rows_convention():
    # rF2 style: local +x left, +y up, +z rear; world = M @ local -> columns of M
    fwd, left, up, vx, vz, sp = make_car()
    basis = check(matrix([left, up, -fwd]), vx, vz, sp)
    assert basis.axis_kind == "col"


def test_detect_transposed():
    fwd, left, up, vx, vz, sp = make_car()
    M = matrix([left, up, -fwd]).transpose(0, 2, 1)
    basis = check(M, vx, vz, sp)
    assert basis.axis_kind == "row"


def test_detect_forward_positive_z():
    fwd, left, up, vx, vz, sp = make_car()
    check(matrix([left, up, fwd]), vx, vz, sp)
    check(matrix([-left, up, fwd]), vx, vz, sp)


def test_detect_refuses_garbage():
    _, _, _, vx, vz, sp = make_car()
    n = len(vx)
    assert vp.detect_pose_basis(np.tile(np.eye(3), (n, 1, 1)), vx, vz, sp) is None
    assert vp.detect_pose_basis(np.zeros((n, 3, 3)), vx, vz, sp) is None
    # too few fast samples
    fwd, left, up, vx, vz, sp = make_car()
    slow = sp.copy()
    slow[40:] = 5.0
    assert vp.detect_pose_basis(matrix([left, up, -fwd]), vx, vz, slow) is None


def test_ori_matrices_missing_channel():
    n = 10
    rows = {name: np.arange(n, dtype=float) for name in vp.ORI_CHANNELS}
    M = vp.ori_matrices(rows)
    assert M.shape == (n, 3, 3)
    assert M[3, 1, 2] == 3.0 and np.all(M[:, 1, 2] == rows["Ori R1Z"])
    del rows["Ori R2Y"]
    assert vp.ori_matrices(rows) is None
    rows["Ori R2Y"] = np.arange(n - 1, dtype=float)
    assert vp.ori_matrices(rows) is None
    rows["Ori R2Y"] = np.array([])
    assert vp.ori_matrices(rows) is None


def test_telemetry_loads_pose_channels(tmp_path):
    times = tick_times()
    path = write_session(tmp_path / "pose.duckdb", times=times)
    span = times[-1] - times[0]
    hz = (len(times) - 1) / span
    con = duckdb.connect(str(path))
    for name in ("Pos Y",) + vp.ORI_CHANNELS:
        con.execute(f'CREATE TABLE "{name}" (value DOUBLE)')
        con.executemany(f'INSERT INTO "{name}" VALUES (?)', [(t,) for t in times])
        con.execute("INSERT INTO channelsList VALUES (?, ?, '')", (name, hz))
    con.execute('CREATE TABLE "Ride Height" (value1 DOUBLE, value2 DOUBLE, value3 DOUBLE, value4 DOUBLE)')
    con.executemany('INSERT INTO "Ride Height" VALUES (?, ?, ?, ?)', [(t, t, t, t) for t in times])
    con.execute("INSERT INTO channelsList VALUES ('Ride Height', ?, '')", (hz,))
    con.close()

    laps = telemetry.load_telemetry_file(path, grid_hz=50)
    assert laps
    for lap in laps:
        assert "Pos Y" in lap.channels
        assert "Ori R1Y" in lap.channels
        assert "Ride Height FL" in lap.channels and "Ride Height RR" in lap.channels
