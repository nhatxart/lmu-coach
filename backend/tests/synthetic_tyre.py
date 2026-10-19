"""Synthetic axle laps drawn from a known factorized Magic Formula (for tire_mf tests).

The truth is expressed relative to a generator reference state (GEN_REF) that sits at
the medians of the sampling ranges, so the fit's data-median reference state lands
(almost) on it and D0 / T0 can be compared directly.
"""

import numpy as np

from app import tire_bins as tb

ALL = ("load", "temp", "pressure", "wear", "camber")

# slip in deg; B0 per deg; k's per unit of their variable (kPa for pressure, deg for camber).
TRUE_PARAMS = {
    "D0": 1.8, "B0": 0.5, "C": 1.5, "E": -0.5,
    "kDL": -0.4, "kBL": 0.2, "T0": -5.0, "wl": 15.0, "wh": 25.0,
    "kDP": -0.004, "kBP": 0.003, "kDW": 1.0, "kDC": 0.05,
}
# Medians of the ranges drawn in make_bins; dp is the pressure residual (ref 0).
GEN_REF = {"load_ratio": 1.25, "temp_delta_c": -5.0, "wear": 0.2, "camber_deg": 2.25}
PT_SLOPE, PT_INTERCEPT = 0.8, 160.0  # pressure = intercept + slope * temp_delta + noise


def true_mu(p, slip, load, td, dp, wear, camber, factors):
    """Noise-free mu from the generator truth; factors not listed stay neutral."""
    on = lambda f: f in factors  # noqa: E731
    dmul = np.ones_like(slip)
    bmul = np.ones_like(slip)
    if on("load"):
        dmul = dmul * (1 + p["kDL"] * (load - GEN_REF["load_ratio"]))
        bmul = bmul * (1 + p["kBL"] * (load - GEN_REF["load_ratio"]))
    if on("temp"):
        w = np.where(td < p["T0"], p["wl"], p["wh"])
        dmul = dmul * np.exp(-(((td - p["T0"]) / w) ** 2))
    if on("pressure"):
        dmul = dmul * (1 + p["kDP"] * dp)
        bmul = bmul * (1 + p["kBP"] * dp)
    if on("wear"):
        dmul = dmul * (1 - p["kDW"] * (wear - GEN_REF["wear"]))
    if on("camber"):
        dmul = dmul * (1 + p["kDC"] * (camber - GEN_REF["camber_deg"]))
    ba = p["B0"] * bmul * slip
    return p["D0"] * dmul * np.sin(p["C"] * np.arctan(ba - p["E"] * (ba - np.arctan(ba))))


def make_bins(n_laps=6, true_params=None, factors=("load", "temp", "pressure"),
              noise=0.03, seed=0, n_per_lap=4000, true_scale=None, td_range=(-20.0, 10.0),
              slip_range=(0.0, 7.0), measure="mu"):
    """List of LapBins (one per lap). measure="lat" drops load_ratio -> lateral_g bins."""
    p = {**TRUE_PARAMS, **(true_params or {})}
    rng = np.random.default_rng(seed)
    out = []
    for lap in range(n_laps):
        n = n_per_lap
        slip = rng.uniform(*slip_range, n)
        load = rng.uniform(0.9, 1.6, n)
        td = rng.uniform(*td_range, n)
        dp = rng.uniform(-6.0, 6.0, n)
        pressure = PT_INTERCEPT + PT_SLOPE * td + dp
        # Wear grows lap to lap (ref 0.2 = median of 0.05 + 0.03 * lap over 6 laps).
        wear = np.full(n, 0.125 + 0.03 * lap) if "wear" in factors else np.full(n, 0.2)
        camber = rng.uniform(1.5, 3.0, n) if "camber" in factors else np.full(n, 2.25)
        mu = true_mu(p, slip, load, td, dp, wear, camber, factors)
        mu_noisy = mu + rng.normal(0, noise, n)
        s = {
            "slip_angle_deg": slip * rng.choice([-1.0, 1.0], n),
            "lateral_g": mu_noisy * load,  # grip = lateral_g / max(load, 0.2) = mu
            "load_ratio": load, "temp_delta_c": td, "tire_pressure_kpa": pressure,
            "wear": wear, "camber_deg": camber,
            "carcass_delta_c": np.zeros(n), "long_g": np.zeros(n),
            "true_mu": (true_scale * (mu + rng.normal(0, noise, n))
                        if true_scale else np.full(n, np.nan)),
        }
        if measure == "lat":
            s["lateral_g"] = mu_noisy
            s["load_ratio"] = np.full(n, np.nan)
        out.append(tb.bin_axle_samples(s, np.ones(n, bool)))
    return out
