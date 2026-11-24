"""
Ideal lap validation (design docs/superpowers/specs/2026-11-10-ideal-lap-design.md §9).

One PAIR = (fit set, test laps); the test laps are never in the fit.
  --holdout odd  : fit on the even qualifying laps of --sim (plus any other --fit
                   files), test on its odd laps.
  --holdout file : fit on the --fit files only, test on --sim (cross-session, or
                   cross-track when the tracks differ: V11, report only).
Default: odd when --sim is among --fit, else file. Test laps = the 3 fastest
qualifying laps of the test set (--laps).

Prints one line per test lap with the §9 metrics, then the pair's summary:
  V1  load model R2 per wheel (test file) and axle-mean lam bias in hard cornering
  GATE the pair's gate row (§15 V5, §15.x 3): main | low_confidence (still in the gate) |
      excluded_few_laps (any class with < 3 fit laps); the fit set's self-check (C3 gives `low` above sd 0.075)
  V2  level: 0.75 quantile of log(observed / predicted) on the held-out saturated samples (per
      axle, pooled; the calibration's own statistic, §15 V5), the median printed after it
  V3  spread: log sd of the same
  V4  V2 inside the fit's beta (per pair; the share over pairs is the summary's job)
  V5  plain model (unclamped) at level x 1.10: share of the lap > 2 km/h under driven, worst km/h
  V6  unclamped envelope time over the test laps (spread %), clamped reported; report only (§15 V5)
  V6b fit on the even / odd / all laps of the fit set, plain model (no clamp, no measured power) on
      the first test lap's line: spread % (report only; its limit comes from the first run, §15.x 4)
  V7  (file holdout) ideal time fitted without vs with the test file (%)
  V8  (no tyre forces) ideal time for drag d2 x 0.7 / x 1.3 (s)
  V9  gain for laps within 1 % of the file's best: 0..3 % of the lap, gain_min > 0
  V10 axle status, terms, knee (vs tyre v3), binding-code shares
  V12 run time of one lap's analysis (s)
  V13 clamped share, gain with / without the clamp, gain share in extrapolated corners
  V14 (file holdout) per-axle V2 on the test file's odd laps, without and with the trim
      taken from its even laps; setup_differs of the test file against the fit files
  V15 public lap times: driven %, ideal %, public_check (needs public_laptimes.json)

Read-only on the recordings. Reductions are cached in --cache (default
%TEMP%/ideal_lap_cache), never under data/.

    python -I backend/tools/ideal_lap_report.py --fit "COTA*Ferrari*10-27" --sim "COTA*Ferrari*10-30"
    python -I backend/tools/ideal_lap_report.py --fit Porsche_COTA --sim Porsche_COTA --holdout odd --json out.json
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app import ideal_car as C  # noqa: E402
from app import ideal_solver as SV  # noqa: E402
from app import ideal_store as ST  # noqa: E402
from app.ideal_lap import analyse_lap, public_block  # noqa: E402
from app.ideal_state import (GRID_HZ, FileReduction, lap_raw, lap_state, load_reduction,  # noqa: E402
                             qualifying_laps, setup_fingerprint, _ensure_wheel_channels)


def resolve(sessions: Path, spec: str) -> Path:
    """A file name, a glob, or a unique substring of one."""
    p = sessions / spec
    if p.exists():
        return p
    names = sorted(x.name for x in sessions.glob("*.duckdb"))
    hits = [n for n in names if fnmatch.fnmatch(n, spec)] or [n for n in names if spec in n]
    if len(hits) != 1:
        raise SystemExit(f"'{spec}' matches {len(hits)} files: {hits[:8]}")
    return sessions / hits[0]


def level_stats(car, axles, states) -> dict:
    """V2 / V3 per axle and pooled: log(|gy| / A_j) on base & saturated samples
    (the fit's own rules), as ideal_store.selfcheck."""
    out, pool = {}, []
    for j, (idx, name) in enumerate(((C.FRONT, "front"), (C.REAR, "rear"))):
        x = []
        for s in states:
            gxt = s.gx + s.grade + car.drag(s.v)
            ay = np.abs(s.gy)
            base = (np.abs(gxt) < 0.2) & (ay > 0.45 * np.percentile(ay, 99)) & (s.v > 15) & (np.abs(s.yawacc) < 0.4)
            ax = axles[j]
            if ax.saturation == "grip_fraction" and s.gf is not None:
                sat = s.gf[list(idx)].mean(0) < C.GF_SATURATED
            elif ax.knee_deg is not None and s.slip is not None:
                sl = s.slip[list(idx)].mean(0)
                sat = (sl >= ax.knee_deg) & (sl <= ax.knee_deg + C.KNEE_SPAN_DEG)
            else:
                continue
            m = base & sat
            x.append(np.log(ay[m] / car.axles(s.v, s.gx, s.gy, s.gz)[j][m]))
        x = np.concatenate(x) if x else np.zeros(0)
        out[name] = _stat(x)
        pool.append(x)
    out["all"] = _stat(np.concatenate(pool))
    return out


def _stat(x) -> dict:
    """V2 shift = the 0.75 quantile, the level's own statistic (design §15 V5;
    PREVIOUSLY the median, which a q 0.75 calibration puts ~0.67 sd below 0);
    the median is reported next to it."""
    if not len(x):
        return {"n": 0, "shift": None, "median": None, "sd": None}
    return {"n": int(len(x)), "shift": float(np.quantile(x, C.LEVEL_Q)), "median": float(np.median(x)),
            "sd": float(np.std(x))}


def lam_bias(lm, states) -> dict:
    """V1: axle-mean lam in hard cornering, modelled vs measured (relative)."""
    v, gx, gy, gz = (np.concatenate([getattr(s, k) for s in states]) for k in ("v", "gx", "gy", "gz"))
    L = np.concatenate([s.L(lm.source) for s in states], axis=1)
    meas, mod = L / lm.static[:, None], lm.lam(v, gx, gy, gz)
    hard = np.abs(gy) > 0.6 * np.percentile(np.abs(gy), 99)
    f = lambda a, ix: float(a[ix].mean(0)[hard].mean())               # noqa: E731
    return {"front": f(mod, [0, 1]) / f(meas, [0, 1]) - 1, "rear": f(mod, [2, 3]) / f(meas, [2, 3]) - 1}


def fit_info(g, n_laps: int, scope: str = "track", sc: dict | None = None) -> dict:
    """As the store: with the self-check `sc` (ST.selfcheck of the fit set) the
    C3 rule applies (sd > 0.075 -> low, beta >= 0.05)."""
    beta, conf, weak = ST.beta_and_confidence(scope, n_laps, g, sc)
    i = g.info
    return {"scope": scope, "n_laps": n_laps, "n_files": i["n_files"], "load_source": i["load_source"],
            "confidence": conf, "beta": beta, "notes": list(weak), "selfcheck": sc,
            "axles": i["axles"], "s_front": i["s_front"],
            "kappa_brake": i["kappa_brake"], "kappa_trac": i["kappa_trac"], "drag": i["drag"]}


def car_with(g, lm) -> C.CarModel:
    d = g.car_for(next(iter(g.load_models))).to_dict()
    d["load"] = lm.to_dict()
    return C.CarModel.from_dict(d)


def subset(red: FileReduction, laps) -> FileReduction:
    return FileReduction(**{**red.__dict__, "laps": list(laps)})


HF1 = "lmu-coach-recorder hf1"


def legacy_recorder(meta: dict) -> bool:
    """True for a recording made before the HF1 recorder (10/19/2026): its
    metadata RecordedBy is not 'lmu-coach-recorder hf1' (v4, v3, the first
    shared-memory recorder, or none at all). Kept files copy the original's
    metadata, so they count as what they were cut from."""
    return not str((meta or {}).get("RecordedBy") or "").startswith(HF1)


def gate_row(car_class: str, n_fit: int, confidence: str, legacy: bool = False) -> dict:
    """Design §15 V5 with §15.x 3: the gate takes every pair (Hypercar and
    LMGT3) whose fit set has >= 3 qualifying laps. Low-confidence pairs
    (C3 or default terms) stay IN the gate (excluding them would make V3
    tautological) and are reported in their own row.
    BUG FIX (validation r2): the >= 3 rule was applied to LMGT3 only, so the
    1-2-lap Hypercar fits (Genesis Sarthe, Ferrari Road Atlanta, Valkyrie,
    Aston 2to1) were labelled main / low_confidence and counted in the gate.
    Fix: they are `excluded_few_laps`, whatever the class.
    Gate decision 11/23/2026 (design §16): a pair with any file from the old
    recorder (`legacy`, see legacy_recorder) is `excluded_legacy_recorder`; such
    files stay low confidence in the app (the C3 rule), they are not gated."""
    if legacy:
        return {"gated": False, "row": "excluded_legacy_recorder",
                "reason": "a file of the pair is from the old recorder (RecordedBy not hf1)"}
    gated = n_fit >= 3
    row = ("low_confidence" if confidence == "low" else "main") if gated else "excluded_few_laps"
    return {"gated": gated, "row": row,
            "reason": None if gated else f"{n_fit} fit laps (< 3, {car_class})"}


def v6b(fit_reds: list, car_class: str, lm, raw_line: dict) -> dict:
    """Design §15 V4 C2 / §15.x 4 (report only, no limit yet): the fit on the
    even / odd / all laps of the fit set, each car's plain model (no clamp, no
    measured power) on the first test lap's line; spread = (max - min) / mean."""
    out = {}
    for name, sl in (("even", slice(0, None, 2)), ("odd", slice(1, None, 2)), ("all", slice(None))):
        reds = [subset(r, sorted(r.laps, key=lambda s: s.lap_number)[sl]) for r in fit_reds]
        reds = [r for r in reds if r.laps]
        try:
            g = C.fit_group(reds, car_class)
        except (C.InsufficientData, ValueError, IndexError, np.linalg.LinAlgError) as e:
            out[name] = None
            out.setdefault("errors", {})[name] = str(e)
            continue
        out[name] = SV.solve(car_with(g, lm), raw_line, clamp=False)["time"]
    ts = [out[k] for k in ("even", "odd", "all") if out.get(k) is not None]
    out["spread_pct"] = 100 * (max(ts) - min(ts)) / float(np.mean(ts)) if len(ts) >= 2 else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-dir", type=Path, default=REPO / "data" / "sessions")
    ap.add_argument("--fit", required=True, help="comma-separated files (name, glob or unique substring)")
    ap.add_argument("--sim", required=True, help="the test file")
    ap.add_argument("--holdout", choices=("odd", "file"))
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--levels", action="store_true", help="print gain_min / gain / gain_max (default: gain only)")
    ap.add_argument("--no-v7", action="store_true", help="skip the second fit of V7")
    ap.add_argument("--no-v8", action="store_true", help="skip the drag sensitivity of V8")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("TEMP", tempfile.gettempdir())) / "ideal_lap_cache")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if not os.environ.get("TRACK_REFERENCE_DIR") and (REPO / "data" / "track_reference").exists():
        os.environ["TRACK_REFERENCE_DIR"] = str(REPO / "data" / "track_reference")

    sim = resolve(a.sessions_dir, a.sim)
    fits = [resolve(a.sessions_dir, s.strip()) for s in a.fit.split(",") if s.strip()]
    holdout = a.holdout or ("odd" if sim in fits else "file")
    t0 = time.time()
    reds = {p.name: load_reduction(p, a.cache) for p in set(fits) | {sim}}
    sim_red = reds[sim.name]
    q = sorted(sim_red.laps, key=lambda s: s.lap_number)
    if holdout == "odd":
        fit_reds = [subset(sim_red, q[0::2])] + [reds[p.name] for p in fits if p != sim]
        test_states = q[1::2]
    else:
        fit_reds = [reds[p.name] for p in fits if p != sim]
        test_states = q
    if not fit_reds or not test_states:
        print("nothing to fit or nothing to test")
        return 1
    try:
        g = C.fit_group(fit_reds, sim_red.car_class)
    except C.InsufficientData as e:
        print(f"PAIR insufficient_data: {e}")
        return 0
    n_fit = sum(len(r.laps) for r in fit_reds)
    lm = g.load_models.get(sim.name) if holdout == "odd" else C.fit_load_model(q, sim_red.load_source)
    car = car_with(g, lm)
    sc = ST.selfcheck(fit_reds, sim_red.car_class)
    info = fit_info(g, n_fit, sc=sc)
    legacy = any(legacy_recorder(r.meta) for r in [*fit_reds, sim_red])
    gate = gate_row(sim_red.car_class, n_fit, info["confidence"], legacy=legacy)
    axles = [g.axle(0), g.axle(1)]
    cross_track = any(r.track != sim_red.track for r in fit_reds)
    print(f"PAIR {holdout}{' cross-track' if cross_track else ''} | fit {[r.file[-31:-7] for r in fit_reds]} "
          f"({n_fit} laps) | test {sim.name[-31:-7]} | {sim_red.car_model} ({sim_red.car_class}) | "
          f"beta {info['beta']:.2f} conf {info['confidence']} | load {time.time() - t0:.0f} s")
    print(f"  GATE {gate['row']}" + (f" ({gate['reason']})" if gate["reason"] else "")
          + " | selfcheck " + ("n/a" if sc is None else f"q75 {sc['shift']:+.3f} median {sc['median']:+.3f} "
                                                        f"sd {sc['sd']:.3f} (n {sc['n']})")
          + (f" | notes {info['notes']}" if info["notes"] else ""))
    v6b_out = None

    lv2 = level_stats(car, axles, test_states)                      # V2 / V3
    v1 = {"r2": [round(x, 3) for x in lm.r2], "bias": lam_bias(lm, test_states)}
    v14 = None
    if holdout == "file" and len(q) >= 2:
        trimmed, trims = C.axle_trim(car, axles, subset(sim_red, q[0::2]))
        fps = [setup_fingerprint(None, None, g.load_models.get(r.file), measured=r.setup or {}) for r in fit_reds]
        fp = setup_fingerprint(None, None, lm, measured=sim_red.setup or {})
        v14 = {"without": level_stats(car, axles, q[1::2]), "with": level_stats(trimmed, axles, q[1::2]),
               "trims": trims, "setup_differs": ST.setup_differs(fp, ST.group_median(fps))}
    g2 = None
    if holdout == "file" and not a.no_v7:
        g2 = C.fit_group(fit_reds + [sim_red], sim_red.car_class)

    # the test laps' raw arrays (the line needs them)
    _ensure_wheel_channels()
    from app.telemetry import load_telemetry_file
    from app import track_topography as tt
    from app.session_index import file_entry
    with contextlib.redirect_stdout(io.StringIO()):
        laps = load_telemetry_file(sim, grid_hz=GRID_HZ)
    keep = {s.lap_number for s in test_states}
    test_laps = sorted((l for l in qualifying_laps(laps) if l.lap_number in keep), key=lambda l: l.lap_time_s)[: a.laps]
    best = min(l.lap_time_s for l in qualifying_laps(laps))
    entry = file_entry(sim, "sessions")
    rows = []
    for lap in test_laps:
        lap_len = tt._lap_length_m(lap)
        layout = None
        if "Pos Y" not in lap.channels:
            hit = tt.load_topography(lap.track_name, lap_len)
            layout = hit[1] if hit else None
        raw = lap_raw(lap, layout)
        t1 = time.time()
        res = analyse_lap(car, info, raw, lap.lap_time_s, track_name=lap.track_name, lap_len=lap_len)
        v12 = time.time() - t1
        ln, lv = res["_line"], res["_levels"]
        T = lap.lap_time_s
        if v6b_out is None:                                              # V6b: first test lap's line
            v6b_out = v6b(fit_reds, sim_red.car_class, lm, SV.line_of(raw))
        p110 = SV.solve(car.scaled(1.10), ln, clamp=False)                # V5
        under = ln["v"] - p110["v"]
        r = {"lap": int(lap.lap_number), "lap_time": T, "ideal": res["ideal"], "model_only": res["model_only"],
             "ceiling": res["ceiling"], "v1": v1, "v2": lv2, "v4_in_beta": (lv2["all"]["shift"] is not None
                                                                           and abs(lv2["all"]["shift"]) <= info["beta"]),
             "v5": {"share": float((under > 2 / 3.6).mean()), "worst_kmh": float(under.max() * 3.6)},
             "v6_unclamped": T - lv["gain_unclamped"], "v6_clamped": res["ideal"]["time_s"], "v12": v12,
             "v13": {"clamped_share": lv["clamped_share"], "gain": lv["gain"], "gain_unclamped": lv["gain_unclamped"],
                     "extrapolated_gain_s": res["diagnostics"]["extrapolated_gain_s"],
                     "terms": [x.terms for x in axles], "knee": [x.knee_deg for x in axles]},
             "v10": {"status": [x.status for x in axles], "knee_vs_v3": [x.knee_vs_v3 for x in axles],
                     "binding": res["diagnostics"]["binding_share"]},
             "within_1pct": T <= best * 1.01}
        r["v9_ok"] = (not r["within_1pct"]) or (0 <= lv["gain"] / T <= 0.03 and lv["gain_min"] > 0)
        if g2 is not None:                                               # V7
            car2 = car_with(g2, lm)
            r2 = analyse_lap(car2, fit_info(g2, n_fit + len(q)), raw, T, track_name=lap.track_name, lap_len=lap_len)
            r["v7_pct"] = 100 * (r2["ideal"]["time_s"] - res["ideal"]["time_s"]) / T
        if not sim_red.has_forces and not a.no_v8:                       # V8
            out = []
            for f in (0.7, 1.3):
                d = car.to_dict()
                d["d2"] = d["d2"] * f
                out.append(analyse_lap(C.CarModel.from_dict(d), info, raw, T, track_name=lap.track_name,
                                       lap_len=lap_len)["ideal"]["time_s"] - res["ideal"]["time_s"])
            r["v8_s"] = out
        pub, chk = public_block(sim_red.car_model, sim_red.car_class, entry.get("track_slug"),
                                entry.get("layout_len_m"), sim_red.game_version, T, res["ideal"]["time_s"],
                                res["ceiling"]["time_s"])
        r["v15"] = {"public": pub, "public_check": chk}
        rows.append(r)
        i = res["ideal"]
        gains = (f"{i['gain_min_s']:+.2f}/{i['gain_s']:+.2f}/{i['gain_max_s']:+.2f}" if a.levels else f"{i['gain_s']:+.2f}")
        print(f"  L{r['lap']:<3d} {T:8.3f} ideal {i['time_s']:8.3f} gain {gains} s | model_only {res['model_only']['time_s']:.2f} "
              f"| V5 {100 * r['v5']['share']:.1f}% {r['v5']['worst_kmh']:.0f}km/h | V9 {'ok' if r['v9_ok'] else 'FAIL'} "
              f"| V13 clamp {100 * lv['clamped_share']:.0f}% unclamped {lv['gain_unclamped']:+.2f} s extrap "
              f"{r['v13']['extrapolated_gain_s']:+.2f} s ceiling {res['ceiling']['time_s']:.2f}"
              + (f" | V7 {r['v7_pct']:+.2f}%" if "v7_pct" in r else "")
              + (f" | V8 {r['v8_s'][0]:+.2f}/{r['v8_s'][1]:+.2f} s" if "v8_s" in r else "")
              + (f" | V15 {pub['driven_pct']:.1f}%/{pub['ideal_pct']:.1f}% {chk}" if pub else " | V15 -")
              + f" | V12 {v12:.1f} s")
    if rows:
        un = [r["v6_unclamped"] for r in rows]
        cl = [r["v6_clamped"] for r in rows]
        mean_t = float(np.mean([r["lap_time"] for r in rows]))
        v6 = {"unclamped_pct": 100 * (max(un) - min(un)) / mean_t, "clamped_pct": 100 * (max(cl) - min(cl)) / mean_t}
    else:
        v6 = None
    # V2 = the 0.75 quantile (the shape tally.py parses: "<q75> sd <sd> (n N)"), median after it
    fmt = lambda d: ("n/a" if d["shift"] is None else                                      # noqa: E731
                     f"{d['shift']:+.3f} sd {d['sd']:.3f} (n {d['n']}) med {d['median']:+.3f}")
    b = v6b_out or {}
    v6b_txt = ("n/a" if b.get("spread_pct") is None else
               f"{b['spread_pct']:.2f}% (even/odd/all " + "/".join("-" if b.get(k) is None else f"{b[k]:.2f}"
                                                               for k in ("even", "odd", "all")) + " s)")
    print(f"  V1 R2 {v1['r2']} lam bias F {100 * v1['bias']['front']:+.1f}% R {100 * v1['bias']['rear']:+.1f}%"
          f" | V2/V3 F {fmt(lv2['front'])} R {fmt(lv2['rear'])} all {fmt(lv2['all'])}"
          f" | V4 in beta {rows[0]['v4_in_beta'] if rows else None}"
          + (f" | V6 unclamped {v6['unclamped_pct']:.2f}% clamped {v6['clamped_pct']:.2f}% (report only)" if v6 else "")
          + f" | V6b {v6b_txt} (report only, no limit yet)")
    ktf = info["kappa_trac"]                                           # §15.z2 guard
    print(f"  V10 status {[x.status for x in axles]} terms {[x.terms for x in axles]} knee {[x.knee_deg for x in axles]} "
          f"knee_vs_v3 {[x.knee_vs_v3 for x in axles]} binding {rows[0]['v10']['binding'] if rows else None}"
          f" | kt {ktf['value']:.3f} {ktf['status']} envelope_ratio "
          + ("n/a" if ktf.get("envelope_ratio") is None else f"{ktf['envelope_ratio']:.3f}"))
    if v14:
        print(f"  V14 setup_differs {v14['setup_differs']} | trims {[(t['trim'], t['t'] and round(t['t'], 3)) for t in v14['trims']]}"
              f" | V2 F without {fmt(v14['without']['front'])} with {fmt(v14['with']['front'])}"
              f" | R without {fmt(v14['without']['rear'])} with {fmt(v14['with']['rear'])}")
    if a.json:
        a.json.write_text(json.dumps(C._jsonable({"pair": {"holdout": holdout, "cross_track": cross_track,
                                                            "fit": [r.file for r in fit_reds], "sim": sim.name,
                                                            "car_model": sim_red.car_model, "car_class": sim_red.car_class,
                                                            "beta": info["beta"], "confidence": info["confidence"], "kappa_trac": info["kappa_trac"],
                                                            "n_fit_laps": n_fit, "selfcheck": sc, "gate": gate,
                                                            "notes": info["notes"]},
                                                   "laps": rows, "v6": v6, "v6b": v6b_out, "v14": v14}), indent=1))
        print(f"  JSON: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
