import importlib.util
from pathlib import Path

import numpy as np

from app.tire_bins import LapBins
from synthetic_tyre import make_bins

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "mf_experiment.py"
_spec = importlib.util.spec_from_file_location("mf_experiment", _TOOL)
mfx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mfx)


def _laps(seed, n_laps=2, factors=("load", "temp")):
    # true_scale=1.0: the "true mu" channel equals the generator mu (+ noise)
    return make_bins(n_laps=n_laps, factors=factors, seed=seed, n_per_lap=20000,
                     true_scale=1.0)


def test_heldout_r2_separates_a_good_model_from_a_constant():
    train = LapBins.merge(_laps(1))
    test = LapBins.merge(_laps(2))
    fit = mfx.tire_mf.fit_mf(train, target="true", factors=("load", "temp"))
    good = mfx.heldout_r2(fit, test)
    flat = mfx.heldout_r2(None, test, predictor=lambda s: np.full(len(s), 0.5))
    assert good["r2"] > 0.9
    assert good["rmse"] < 0.1
    assert flat["r2"] < good["r2"] - 0.5
    assert good["n_cells"] > 10


def test_heldout_r2_none_when_no_cornering_true_cells():
    empty = make_bins(n_laps=1, n_per_lap=500, slip_range=(0.0, 1.0), true_scale=1.0)[0]
    assert mfx.heldout_r2(None, empty, predictor=lambda s: s) is None


def test_binned_predictor_interpolates_and_clamps():
    pred = mfx.binned_predictor(LapBins.merge(_laps(3, factors=())), "true")
    out = pred(np.array([-1.0, 3.0, 50.0]))
    assert np.all(np.isfinite(out)) and out[1] > out[0]
    assert out[2] == pred(np.array([100.0]))[0]  # nearest bin outside the range


def test_score_axle_runs_and_load_temp_beat_slip_only():
    units = [(f"file{i}", b) for i in range(3) for b in _laps(10 + i)]
    res = mfx.score_axle(units)
    assert res["n_folds"] == 3
    assert set(mfx.VARIANT_NAMES) <= set(res)
    assert res["+load+temp"]["folds"] == 3
    assert res["+load+temp"]["r2"] > res["mf_slip_only"]["r2"] + 0.02
    assert "temp:fit" in res["+load+temp"]["status"]
    table = mfx.format_table("synthetic", res)
    assert table.count("\n| ") >= len(mfx.VARIANT_NAMES)


def test_fold_with_unfittable_training_is_counted_as_skipped():
    good = _laps(20)
    tiny = make_bins(n_laps=1, n_per_lap=100, slip_range=(0.0, 3.0), true_scale=1.0)
    # holding out "a" trains on the tiny fold only -> MF fits return None
    res = mfx.score_axle([("a", good[0]), ("b", tiny[0])])
    assert res["+load"]["skipped"] >= 1

def test_held_out_fold_is_excluded_from_training(monkeypatch):
    units = [(f"f{i}", b) for i in range(3) for b in _laps(30 + i, n_laps=1)]
    totals = {f: b.n_samples() for f, b in units}
    seen = []
    real = mfx.tire_mf.fit_mf

    def spy(train, **kw):
        seen.append(train.n_samples())
        return real(train, **kw)

    monkeypatch.setattr(mfx.tire_mf, "fit_mf", spy)
    mfx.score_axle(units)
    total = sum(totals.values())
    assert set(seen) == {total - n for n in totals.values()}


def test_outlier_fold_changes_heldout_score_not_training_fit():
    normal = [(f"f{i}", b) for i in range(2) for b in _laps(40 + i, n_laps=2)]
    outlier = ("odd", make_bins(n_laps=1, factors=("load", "temp"), seed=49,
                                n_per_lap=20000, true_scale=3.0)[0])
    base = mfx.score_axle(normal)
    mixed = mfx.score_axle(normal + [outlier])
    # holding out an outlier must be judged by a model trained only on normal folds:
    # the outlier fold scores badly; the normal folds' training now contains the outlier
    assert base["+load+temp"]["r2"] > 0.9
    assert mixed["+load+temp"]["r2_all"] < base["+load+temp"]["r2"] - 0.1


def test_true_n_below_three_is_not_scored():
    lap = _laps(50, n_laps=1)[0]
    full = mfx.heldout_r2(None, lap, predictor=lambda s: np.full(len(s), 0.5))
    for c in lap.cells.values():
        c.n_true = min(c.n_true, 2)
    assert mfx.heldout_r2(None, lap, predictor=lambda s: s) is None
    assert full is not None


def test_folds_without_cells_are_counted():
    good = _laps(60, n_laps=2)
    empty = make_bins(n_laps=1, n_per_lap=500, slip_range=(0.0, 1.0), true_scale=1.0)[0]
    res = mfx.score_axle([("a", good[0]), ("b", good[1]), ("c", empty)])
    assert res["no_cells"] == 1 and res["n_folds"] == 3
    assert "1 without scorable" in mfx.format_table("t", res)
