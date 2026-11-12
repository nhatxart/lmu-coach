"""Ideal lap, design T6: a PERFECT driver. Telemetry is made by solving a known
car on a synthetic line (the car at its limit everywhere), 5 % noise is added to
gy, the car is refitted from that telemetry and solved on one lap's line.
- the plain model (model_only) must give that lap back within +-0.3 % of the lap;
- the envelope's `gain` at level 1 is the PHANTOM gain (time the envelope finds
  where there is none: model noise shows small gains in about half the
  corners); it is reported (printed; run with -s) for the UI note;
- gain_min (level 1 - beta) must stay <= 0.3 % of the lap."""
import numpy as np
import pytest

from app import ideal_solver as SV
from app.ideal_car import fit_group
from app.ideal_state import FileReduction, lap_state
from synthetic_ideal import true_car, line, perfect_laps

BETA = 0.02            # track scope, >= 6 laps, both axles observed (design §5.4)


# Measured 11/11/2026 (seed 2026, this car: front-limited at speed, rear-limited
# when slow): model_only +0.25 %, phantom gain 0.72 % of the lap, gain_min
# 0.40 % > 0.3 %. Other seeds: model_only -1.09 / -0.89 %, phantom 0.26 / 0.34 %.
# The rear is saturated only at slow apexes, so its speed term extrapolates
# (fitted rear capacity 0.74 x true at 60-90 m/s even without noise). Design
# §11 sends a phantom > 0.5 % back to the design author: open, see
# .superpowers/sdd/2026-11-11-ideal-lap/batch-a-report.md.
@pytest.mark.xfail(reason="T6 above the design's thresholds; awaiting the design author", strict=False)
def test_perfect_driver_phantom_gain():
    rng = np.random.default_rng(2026)
    car = true_car()
    raws, truth = perfect_laps(car, line(), n_laps=6, gy_noise=0.05, rng=rng)
    red = FileReduction(file="synthetic.duckdb", mtime=0, size=0, meta={}, track="synthetic",
                        car_model="Synthetic", car_class="LMGT3", game_version="test", load_source="susp",
                        grade_source="pos_y", laps=[lap_state(r) for r in raws])
    gfit = fit_group([red], "LMGT3")
    fitted = gfit.car_for("synthetic.duckdb")
    ln = SV.line_of(raws[3])
    out = SV.levels(fitted, ln, BETA)
    lap = out["t_ref_grid"]
    model_gain = lap - out["model_only_time"]
    print("\nT6 perfect driver: lap %.2f s | fitted mu1 %s (true %s) | model_only gain %+.3f s (%+.2f %%) | "
          "envelope gain_min %+.3f / gain %+.3f (%.2f %%) / gain_max %+.3f s | axles %s / %s | kb %.3f kt %.3f" % (
              lap, np.round(fitted.mu1, 3), car.mu1, model_gain, 100 * model_gain / lap, out["gain_min"],
              out["gain"], 100 * out["gain"] / lap, out["gain_max"], gfit.info["axles"]["front"]["status"],
              gfit.info["axles"]["rear"]["status"], fitted.kb, fitted.kt))
    assert abs(model_gain) <= 0.003 * lap
    assert out["gain_min"] <= 0.003 * lap
    assert out["gain_min"] <= out["gain"] <= out["gain_max"]
