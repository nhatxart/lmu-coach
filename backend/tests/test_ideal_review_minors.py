"""Branch review (11/21/2026), Minor findings that are code defects: one test
each, failing on the code before the fix."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from app import ideal_car as C
from app import ideal_store as ST


def _fit(kb_status="fitted", kt_status="fitted", terms=("fitted", "fitted")):
    ax = {a: {"mu1": 1.4, "k": 0.0, "c": 0.0, "status": "observed", "saturation": "grip_fraction",
              "knee_deg": None, "n_sat": 1000, "ratio_sd": 0.05, "terms": t, "v_sat": [15.0, 45.0]}
          for a, t in zip(C.AXLES, terms)}
    info = {"axles": ax, "kappa_brake": {"status": kb_status}, "kappa_trac": {"status": kt_status},
            "drag": {"source": "forces"}, "notes": []}
    return SimpleNamespace(info=info, axle=lambda j: C.AxleFit(**info["axles"][C.AXLES[j]]))


# ── Minor 1: assumed kb / kt alone do not make the store confidence low ──────

def test_assumed_kappa_is_left_to_the_binding_share_rule():
    sc = {"shift": 0.0, "sd": 0.05}
    assert ST.beta_and_confidence("track", 8, _fit(kb_status="assumed"), sc)[1] == "high"
    assert ST.beta_and_confidence("track", 8, _fit(kt_status="assumed"), sc)[1] == "high"
    assert "kappa_brake assumed" in ST.beta_and_confidence("track", 8, _fit(kb_status="assumed"), sc)[2]
    # default terms stay low (§13 R2)
    assert ST.beta_and_confidence("track", 8, _fit(terms=("fitted", "default")), sc)[1] == "low"


# ── Minor 2: Lap Dist is unwrapped before it goes on the grid ────────────────

def _raw(n=1000, v=40.0, L=790.0, wrap_tail=True, wrap_head=False):
    from app.ideal_state import GRID_HZ
    s = np.arange(n) * v / GRID_HZ                     # 0 .. 799.2 m
    ld = s.copy()
    if wrap_tail:
        ld[s >= L] -= L                                # the closing rows already read the next lap (0 .. 9 m)
    if wrap_head:
        ld[:3] = L - 2.0 + s[:3]                       # the first rows still read the previous lap's end
    z = np.zeros(n)
    return dict(n=1, time=n / GRID_HZ, v=np.full(n, v), s=s, gy=z + 0.5, gz=z, grade=z, thr=z, brk=z, lapdist=ld)


@pytest.mark.parametrize("head", [False, True])
def test_line_lap_dist_is_monotonic_across_the_wraps(head):
    from app import ideal_solver as SV
    ln = SV.line_of(_raw(wrap_head=head))
    ld = ln["lap_dist"]
    assert (np.diff(ld) > 0).all()
    assert ld[-1] > 790.0                              # past the line, not a dip back towards 0


# ── Minor 3: the trace's codes come from the NEAREST grid point ──────────────

def test_trace_codes_take_the_nearest_grid_point():
    from app.ideal_lap import _nearest_index
    ld = np.arange(0.0, 20.0, 2.0)
    np.testing.assert_array_equal(_nearest_index(np.array([0.4, 2.9, 3.1, 7.99, 25.0, -1.0]), ld),
                                  [0, 1, 2, 4, 9, 0])


# ── Minor 8: the trace's t ends at the ideal lap time ────────────────────────

def test_trace_t_includes_the_closing_step():
    from app.ideal_lap import _trace_t
    ds, n = 2.0, 10                                     # a 20 m "lap": the driven 20 m/s, the ideal 25 m/s
    ld = np.arange(n) * ds
    dt_ref, dt_i = np.full(n, ds / 20.0), np.full(n, ds / 25.0)
    p_ld = np.array([0.0, 5.0, 10.0, 20.0])             # the last point on the line
    p_t = p_ld / 20.0
    t = _trace_t(p_ld, p_t, ld, dt_ref, dt_i, ds)
    np.testing.assert_allclose(t, p_ld / 25.0, atol=1e-9)
    assert t[-1] == pytest.approx(dt_i.sum())           # = the ideal lap time on the grid
