"""Ideal lap, design §15 V4 C3 (data quality, applied by the app): a fit whose
self-check spread (odd-lap log sd on the saturated set, §6.3) exceeds 0.075
gets confidence `low`, beta >= 0.05 and a note."""
import numpy as np
import pytest

from app import ideal_store as ST
from app.ideal_state import lap_state
from synthetic_ideal import true_car, line, perfect_laps
from test_ideal_store import World, _red, _entry, NAME


def _group(tmp_path, sd, seed):
    raws, _, _ = perfect_laps(true_car("B"), line(), 6, np.random.default_rng(seed), place_sd=sd, lap_sd=sd)
    w = World(tmp_path)
    w.add(_red(NAME, [lap_state(r) for r in raws]), _entry(NAME))
    return w.store().group_for(w.path(NAME))


def test_noisy_recordings_get_low_confidence_and_a_wider_range(tmp_path):
    g = _group(tmp_path, 0.06, 31)
    assert g.selfcheck is not None and g.selfcheck["sd"] > ST.SELFCHECK_SD_LOW
    assert g.confidence == "low" and g.beta >= 0.05
    assert any("grip varies more than usual" in n for n in g.notes)


def test_clean_recordings_are_not_flagged(tmp_path):
    g = _group(tmp_path, 0.0, 32)
    assert g.selfcheck is not None and g.selfcheck["sd"] <= ST.SELFCHECK_SD_LOW
    assert g.confidence != "low" and g.beta < 0.05
    assert not any("grip varies more than usual" in n for n in g.notes)


def test_rule_on_the_statistic():
    from test_ideal_review_minors import _fit
    beta, conf, notes = ST.beta_and_confidence("track", 8, _fit(), {"shift": 0.0, "sd": 0.08})
    assert (conf, beta) == ("low", 0.05) and any("grip varies" in n for n in notes)
    beta, conf, _ = ST.beta_and_confidence("track", 8, _fit(), {"shift": 0.0, "sd": 0.07})
    assert conf != "low" and beta == pytest.approx(0.02)
