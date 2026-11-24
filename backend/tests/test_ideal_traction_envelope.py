"""Ideal lap, design §15.z2: the traction efficiency kt (q 0.75 of the TC-active
ratio gxt / (s_r room)) must not claim more traction than the car showed.
envelope_ratio = median over bins (v 5 m/s x |gy| 0.3 g, >= 40 drive samples,
TC share >= 20 %) of (model limit at the fitted kt) / (achieved p99 of gxt);
> 1 caps kt at kt / envelope_ratio (status `capped_by_envelope`)."""
import numpy as np
import pytest

from app import ideal_car as C

AR, SF = 1.5, 0.5


def _drive(parts):
    """parts: (n, v, gy, ratio, tc, thr): samples at that state whose tyre demand is
    ratio x s_r x room."""
    v, gy, ratio, tc, thr = (np.concatenate([np.full(p[0], p[i]) for p in parts]) for i in range(1, 6))
    rng = np.random.default_rng(4)
    ratio = ratio * (1 + 0.01 * rng.standard_normal(len(ratio)))
    room = np.sqrt(AR ** 2 - gy ** 2)
    gxt = ratio * (1 - SF) * room
    return dict(v=v, gy=gy, thr=thr, tc=tc), gxt, np.full(len(v), AR)


def test_kt_above_the_achieved_envelope_is_capped():
    # bin X (v 10-15, |gy| 0.6-0.9): TC on 100 % of 200 samples, the car reached ratio ~0.6 there;
    # bin Y (v 30-35, |gy| 0-0.3): 300 TC samples at ratio 1.2 among 3000 drive samples (TC 9 %:
    # not an envelope bin), which pull the q 0.75 kt to ~1.2
    d, gxt, Ar = _drive([(200, 12.0, 0.7, 0.6, 1.0, 90.0), (300, 32.0, 0.1, 1.2, 1.0, 90.0),
                         (2700, 32.0, 0.1, 0.5, 0.0, 90.0)])
    kt = C.fit_traction(d, gxt, Ar, SF, C.LEVEL_Q)
    assert kt["envelope_ratio"] > 1.5
    assert kt["status"] == "capped_by_envelope"
    assert kt["value"] == pytest.approx(kt["uncapped"] / kt["envelope_ratio"])
    # after the cap the model limit in bin X equals the achieved p99
    m = d["v"] < 20
    assert kt["value"] * (1 - SF) * np.sqrt(AR ** 2 - 0.7 ** 2) == pytest.approx(np.percentile(gxt[m], 99), rel=1e-6)


def test_kt_inside_the_envelope_is_not_capped():
    d, gxt, Ar = _drive([(400, 12.0, 0.7, 1.0, 1.0, 90.0), (400, 22.0, 0.4, 1.0, 1.0, 90.0)])
    kt = C.fit_traction(d, gxt, Ar, SF, C.LEVEL_Q)
    assert kt["status"] == "fitted" and kt["envelope_ratio"] < 1.0
    assert kt["value"] == pytest.approx(kt["uncapped"])


def test_no_envelope_bins_no_cap():
    # TC never active: the full-throttle rule, no TC bins -> envelope_ratio None
    d, gxt, Ar = _drive([(400, 12.0, 0.7, 1.0, 0.0, 99.0)])
    kt = C.fit_traction(d, gxt, Ar, SF, C.LEVEL_Q)
    assert kt["envelope_ratio"] is None and kt["status"] == "fitted"
