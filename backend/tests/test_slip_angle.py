"""Per-wheel slip angle for the tyre gauges (/trackmap `slip_angle`).

Run from the repo root (the backend's own deps live in Docker):
    uv run --no-project --with pytest --with fastapi --with duckdb --with numpy `
        --with scipy --with httpx --with pydantic python -m pytest backend/tests -v
"""

import math

import pytest

from app import main

KEYS = ("fl", "fr", "rl", "rr")
N = 400  # 4 s at the recorder's 100 Hz


def _call(rad, speed_kmh, n=N, s_idx=0, e_idx=None, full_len=None):
    full_len = n if full_len is None else full_len
    e_idx = full_len - 1 if e_idx is None else e_idx
    sa_raw = [(rad, rad, rad, rad)] * n
    return main._wheel_slip_angles_deg(sa_raw, [speed_kmh] * n, s_idx, e_idx, full_len)


def test_converts_radians_to_signed_degrees():
    out = _call(0.05, 100.0)
    expected = main.SLIP_ANGLE_SCREEN_SIGN * math.degrees(0.05)  # 2.865
    for k in KEYS:
        assert out[k] == pytest.approx([expected] * N, abs=0.01)


def test_opposite_slip_gives_opposite_sign():
    left, right = _call(0.05, 100.0), _call(-0.05, 100.0)
    assert left["fl"][N // 2] == pytest.approx(-right["fl"][N // 2], abs=0.01)
    assert left["fl"][N // 2] != 0


def test_zero_below_minimum_speed():
    # atan(lat/long) swings to +-90 deg when nearly stopped; the gauge must not peg.
    out = _call(1.2, 10.0)
    for k in KEYS:
        assert out[k] == [0.0] * N


def test_clipped_to_45_degrees():
    out = _call(1.5, 100.0)  # 86 deg
    assert max(abs(v) for v in out["rr"]) == pytest.approx(45.0, abs=0.01)


def test_output_is_the_lap_slice():
    out = _call(0.05, 100.0, s_idx=50, e_idx=249)
    assert [len(out[k]) for k in KEYS] == [200] * 4


def test_interpolates_to_position_length():
    # 100 Hz slip angle against a 10 Hz position grid (GPS files).
    out = _call(0.05, 100.0, n=400, full_len=40)
    assert len(out["fl"]) == 40


def test_slow_stretch_does_not_leak_into_fast_samples():
    sa_raw = [(1.2,) * 4] * 200 + [(0.05,) * 4] * 200
    speed = [5.0] * 200 + [120.0] * 200
    out = main._wheel_slip_angles_deg(sa_raw, speed, 0, 399, 400)
    assert out["fl"][:200] == [0.0] * 200
    # One second into the fast stretch the parked-car spike must be gone.
    assert out["fl"][300] == pytest.approx(
        main.SLIP_ANGLE_SCREEN_SIGN * math.degrees(0.05), abs=0.05)


def test_no_slip_angle_table_gives_empty_lists():
    assert main._wheel_slip_angles_deg([], [100.0] * 10, 0, 9, 10) == {k: [] for k in KEYS}


def test_missing_speed_still_returns_angles():
    out = main._wheel_slip_angles_deg([(0.05,) * 4] * N, [], 0, N - 1, N)
    assert len(out["fl"]) == N
    assert abs(out["fl"][N // 2]) == pytest.approx(math.degrees(0.05), abs=0.01)
