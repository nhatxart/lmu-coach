"""Body slip from the HF recorder's Local Vel X/Z (measured, no calibration).

Sign MEASURED 10/14/2026 on the Fuji HF file: atan2(vx, -vz) matches the
yaw-integral estimate (compute_body_slip_deg) with r = 0.90..0.94 and
0.3 deg RMS on laps 2, 6 and 11; the opposite sign gives r = -0.9.
"""

import math

import numpy as np
import pytest

from app.vehicle_dynamics import body_slip_from_local_vel_deg


def _vel(beta_deg, v=50.0, n=200):
    b = math.radians(beta_deg)
    return [v * math.sin(b)] * n, [-v * math.cos(b)] * n


def test_returns_beta_in_degrees_at_the_given_rows():
    vx, vz = _vel(4.0)
    out = body_slip_from_local_vel_deg(vx, vz, native_x_lap=np.linspace(10, 150, 50),
                                       speed_kmh=[180.0] * 50)
    assert out == pytest.approx([4.0] * 50, abs=0.01)


def test_sign_follows_lateral_velocity():
    vx, vz = _vel(-6.0)
    out = body_slip_from_local_vel_deg(vx, vz, native_x_lap=np.arange(100.0),
                                       speed_kmh=[180.0] * 100)
    assert out[50] == pytest.approx(-6.0, abs=0.01)


def test_zero_below_minimum_speed():
    vx, vz = _vel(30.0, v=3.0)   # parked/crawling: ratio of tiny numbers
    out = body_slip_from_local_vel_deg(vx, vz, native_x_lap=np.arange(100.0),
                                       speed_kmh=[10.0] * 100)
    assert not any(out)


def test_empty_without_channels():
    assert body_slip_from_local_vel_deg([], [], np.arange(10.0), [100.0] * 10) == []
