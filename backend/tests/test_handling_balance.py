"""US / neutral / OS from tyre slip angles, checked against yaw rate.

Replaces the old bars, which counted the sign of the suspension F-R
deflection delta over the whole lap (straights included) - a rake/pitch
signal, not handling balance.

Inputs use the /trackmap screen convention: signed slip in degrees, and in a
corner the slip has the OPPOSITE sign to Yaw Rate (MEASURED 10/14/2026 on the
Fuji HF file: r = -0.88..-0.95 front and rear, laps 2, 6, 11).
"""

import pytest

from app.vehicle_dynamics import HANDLING_NEUTRAL_BAND_DEG, handling_balance

N = 100


def _run(front, rear, yaw=0.3, lat_g=1.2, speed=150.0):
    return handling_balance([front] * N, [rear] * N, [yaw] * N, [lat_g] * N, [speed] * N)


def test_front_slipping_more_into_the_turn_is_understeer():
    out = _run(front=-2.0, rear=-1.0)            # yaw +: slip negative into the turn
    assert out["us"] == 1.0 and out["os"] == 0.0
    assert out["delta_deg"][0] == pytest.approx(1.0)


def test_rear_slipping_more_is_oversteer_in_either_direction():
    assert _run(front=-1.0, rear=-2.0)["os"] == 1.0
    assert _run(front=1.0, rear=2.0, yaw=-0.3, lat_g=-1.2)["os"] == 1.0


def test_countersteer_is_oversteer():
    # Fronts pointed out of the turn, rear sliding into it.
    out = _run(front=0.5, rear=-3.0)
    assert out["os"] == 1.0
    assert out["delta_deg"][0] == pytest.approx(-3.5)


def test_small_difference_is_neutral():
    out = _run(front=-1.0, rear=-1.0 + HANDLING_NEUTRAL_BAND_DEG / 2)
    assert out["neutral"] == 1.0


def test_straights_and_slow_corners_are_not_counted():
    for kw in (dict(yaw=0.01, lat_g=0.05), dict(speed=30.0)):
        out = _run(front=-2.0, rear=-1.0, **kw)
        assert out["cornering_samples"] == 0
        assert out["us"] is None
        assert not any(out["delta_deg"])


def test_slip_against_the_yaw_direction_is_not_counted():
    # Both axles slipping as if turning the other way: a transient or kerb
    # strike, not a balance reading.
    out = _run(front=2.0, rear=1.0)
    assert out["cornering_samples"] == 0


def test_shares_add_up_over_cornering_samples_only():
    front = [-2.0] * 40 + [-1.0] * 30 + [-1.0] * 30
    rear = [-1.0] * 40 + [-2.0] * 30 + [-1.0] * 30
    yaw = [0.3] * 70 + [0.0] * 30                # last 30: straight
    out = handling_balance(front, rear, yaw, [1.0] * 100, [150.0] * 100)
    assert out["cornering_samples"] == 70
    assert out["us"] == pytest.approx(40 / 70)
    assert out["os"] == pytest.approx(30 / 70)
    assert out["us"] + out["neutral"] + out["os"] == pytest.approx(1.0)
