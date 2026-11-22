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
