"""Review I1: the solver's drive / braking demand precomputed per grid point on
a speed grid (vectorised) gives the same laps as the per-step search it
replaces, within 0.01 s per lap, on synthetic lines: the synthetic perfect-driver
line (loads, aero, crests, grade), its clamped variant (clamp up and down), the
ceiling, and a capacity-noise line."""
import numpy as np
import pytest

from app import ideal_solver as SV
from synthetic_ideal import true_car, line, noise_field


def _clamped(ln, car, rng):
    v = SV.solve(car, ln, fast=False)["v"]
    ln = dict(ln, v=v * (1 + 0.04 * np.sin(np.arange(len(v)) / 37.0)))   # driven: up and down of the model
    sat = (np.abs(ln["ks"]) > 1 / 200) & (rng.random(len(v)) < 0.7)
    return SV.apply_clamp(ln, sat, np.vstack([sat, np.zeros_like(sat)]))


@pytest.mark.parametrize("kind", ["A", "B"])
def test_fast_solver_matches_the_per_step_search(kind):
    car = true_car(kind)
    ln = line()
    rng = np.random.default_rng(3)
    cases = [("balanced", ln), ("ceiling", ln), ("balanced", _clamped(ln, car, rng)),
             ("balanced", dict(ln, cap_scale=np.exp(noise_field(len(ln["ks"]), ln["ds"], 0.05, 50.0, rng))))]
    for mode, l in cases:
        old = SV.solve(car, l, mode, fast=False)
        new = SV.solve(car, l, mode)
        assert abs(new["time"] - old["time"]) <= 0.01, (mode, new["time"], old["time"])
        np.testing.assert_allclose(new["v"], old["v"], atol=0.05)          # m/s, everywhere
        assert np.mean(new["binding"] == old["binding"]) > 0.99


def test_fast_solver_on_scaled_levels_matches():
    car = true_car("B")
    ln = line()
    for lev in (0.97, 1.03):
        old = SV.solve(car.scaled(lev), ln, fast=False)["time"]
        assert abs(SV.solve(car.scaled(lev), ln)["time"] - old) <= 0.01
