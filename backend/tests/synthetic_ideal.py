"""Synthetic "perfect driver" laps for the ideal-lap tests (design §8 T6).

A known car (ideal_car.CarModel) is solved on a closed synthetic line with
slow, medium and fast corners and crests; each solved speed profile becomes
50 Hz telemetry in the shape of ideal_state.lap_raw: speed, kinematic gx, gy,
gz, grade, pedals, ABS / TC where brakes / traction bind, Susp Force per wheel
from the true load model, Grip Fraction LOW and the slip angle past the knee
where an axle is at its limit.

Round 1 (design §13 R1): the noise is on the CAPACITY, not on gy. The true
capacity of both axles is multiplied by a field over lap distance: a place part
(the same on every lap) times a lap part (new on every lap), each log sd 0.035
with a 50 m correlation length. The perfect driver drives every lap at that
lap's capacity; gy, speed and the line are exact. (The first version put 5 %
noise on gy, which also bends the line: no recording does that.)
"""
import numpy as np

from app import ideal_car as C
from app import ideal_solver as SV
from app.ideal_state import G, GRID_HZ

DS = 2.0
KNEE_DEG = 4.0


def true_car(kind: str = "B") -> C.CarModel:
    """B: front-limited at speed (rear aero), rear-limited when slow; on line()
    each axle saturates on hundreds of samples per lap.
    A: rear-limited almost everywhere (the front saturates only at speed)."""
    a_f, a_r = (0.6e-4, 2.4e-4) if kind == "B" else (1.0e-4, 1.6e-4)
    lm = C.LoadModel(static=np.array([3000.0, 3000.0, 3667.0, 3667.0]),     # static front share 0.45
                     coef=np.array([[a_f, -0.12, +0.30, 0.9, +0.05],
                                    [a_f, -0.12, -0.30, 0.9, +0.05],
                                    [a_r, +0.12, +0.25, 0.9, -0.05],
                                    [a_r, +0.12, -0.25, 0.9, -0.05]]),
                     source="susp")
    d0, d2 = C.DRAG_PRIOR["LMGT3"]                     # the fit takes the class prior
    pv = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    pg = [0.85, 0.85, 0.70, 0.50, 0.33, 0.20, 0.10]
    return C.CarModel(load=lm, mu1=[1.42, 1.40], k=[0.06, 0.04], c=[0.10, 0.15], s_front=0.45,
                      kb=0.95, kt=0.90, d0=d0, d2=d2, power_v=pv, power_g=pg, corner_drag=0.0)


def line(ds: float = DS) -> dict:
    """Closed line, about 3.9 km: hairpin, medium and fast corners, an S, crests."""
    segs = [(400, 0), (120, 1 / 45), (350, 0), (260, -1 / 130), (300, 0), (420, 1 / 260),
            (250, 0), (110, -1 / 70), (90, 1 / 70), (500, 0), (300, 1 / 180), (400, 0),
            (180, -1 / 90), (450, 0)]
    ks = np.concatenate([np.full(int(L / ds), k) for L, k in segs])
    ks = np.convolve(np.pad(ks, 15, mode="wrap"), np.ones(31) / 31, mode="valid")    # 60 m transitions
    s = np.arange(len(ks)) * ds
    L = s[-1] + ds
    h = 6.0 * np.sin(2 * np.pi * 3 * s / L) + 2.0 * np.sin(2 * np.pi * 7 * s / L)
    grade = np.gradient(h, ds)
    kv = np.gradient(grade, ds)                        # crest (h'' < 0) gives gz < 0
    z = np.zeros(len(ks))
    return dict(s=s, ks=ks, kv=kv, grade=grade, v=z + 30.0, thr=z, brk=z, ds=ds, time=0.0)


def noise_field(n: int, ds: float, sd: float, corr_m: float, rng) -> np.ndarray:
    """Periodic Gaussian field over the lap, log sd `sd`, correlated over
    `corr_m` (moving average of white noise over that length)."""
    if sd <= 0:
        return np.zeros(n)
    w = max(1, int(round(corr_m / ds)))
    e = rng.standard_normal(n)
    f = np.convolve(np.pad(e, (w, w), mode="wrap"), np.ones(w) / w, mode="same")[w:-w]
    return sd * (f - f.mean()) / f.std()


def perfect_laps(car, ln: dict, n_laps: int, rng, place_sd: float = 0.035, lap_sd: float = 0.035,
                 corr_m: float = 50.0, gf_usage: str = "combined") -> tuple:
    """(raws, sols, scales): n_laps lap_raw-shaped dicts of the car driven at
    its limit, each lap at its own capacity exp(place + lap) on the grid of
    `ln`; the true solves and the capacity scales per lap.
    gf_usage: "combined" (as the game: the usage of §5.5, drive and braking
    included) or "lateral" (|gy| / A only), to test a mechanism without the
    combined-slip bias of the lateral fit."""
    n, ds = len(ln["ks"]), ln["ds"]
    place = noise_field(n, ds, place_sd, corr_m, rng)
    raws, sols, scales = [], [], []
    for k in range(n_laps):
        scale = np.exp(place + noise_field(n, ds, lap_sd, corr_m, rng))
        sol = SV.solve(car, dict(ln, cap_scale=scale), "balanced")
        raws.append(_telemetry(car, ln, sol, scale, k, gf_usage))
        sols.append(sol)
        scales.append(scale)
    return raws, sols, scales


def _telemetry(car, ln, sol, scale, lap_no, gf_usage="combined"):
    v, ds = sol["v"], ln["ds"]
    t_grid = np.concatenate([[0.0], np.cumsum(ds / v)])     # time at each grid point (closing point too)
    T = t_grid[-1]
    s_grid = np.arange(len(v) + 1) * ds
    t = np.arange(0.0, T, 1.0 / GRID_HZ)
    s = np.interp(t, t_grid, s_grid)
    per = lambda a: np.interp(s, s_grid, np.append(a, a[0]))      # noqa: E731
    vt = per(v)
    gxt = per(sol["gxt"])
    grade = per(ln["grade"])
    gx = gxt - grade - car.drag(vt)
    gy = per(ln["ks"]) * vt * vt / G
    gz = np.clip(per(ln["kv"]) * vt * vt / G, SV.GZ_MIN, SV.GZ_MAX)
    code = sol["binding"][np.minimum((s / ds).astype(int), len(v) - 1)]
    thr = np.where(np.isin(code, ("power", "traction")), 100.0, np.where(gxt > 0, 40.0, 0.0))
    brk = np.where(code == "brake", 100.0, np.where(gxt < -0.3, 50.0, 0.0))
    abs_ = (code == "brake").astype(float)
    tc = (code == "traction").astype(float)
    sc = per(scale)
    uf, ur = SV.usage(car, vt, gxt, gy, gz)
    if gf_usage == "lateral":
        Af, Ar = car.axles(vt, gx, gy, gz)
        uf, ur = np.abs(gy) / Af, np.abs(gy) / Ar
    uf, ur = uf / sc, ur / sc                      # usage against THIS lap's capacity
    susp = car.load.static[:, None] * car.lam(vt, gx, gy, gz)
    gf, slip = [], []
    for u in (uf, uf, ur, ur):
        sat = u >= 0.995
        gf.append(np.where(sat, 0.15, 0.35 + 0.65 * np.clip(1 - u, 0, 1)))
        slip.append(np.radians(np.where(sat, KNEE_DEG + 1.5, KNEE_DEG * np.clip(u, 0, 1))))
    return dict(n=lap_no, time=float(T), t=t, v=vt,
                s=np.concatenate([[0.0], np.cumsum(0.5 * (vt[1:] + vt[:-1]) / GRID_HZ)]),
                gx=gx, gy=gy, gz=gz, thr=thr, brk=brk, abs=abs_, tc=tc, yaw=None,
                yawacc=np.zeros(len(t)), posy=None, lapdist=s, load=None, susp=susp, fy=None, fx=None,
                gf=np.vstack(gf), slip=np.vstack(slip), grade=grade, grade_source="pos_y")
