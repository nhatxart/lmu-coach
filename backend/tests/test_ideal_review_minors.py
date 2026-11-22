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


# ── Minor 4: the in-memory caches are bounded ────────────────────────────────

def test_lru_keeps_the_most_recent():
    from app.ideal_state import LRU
    c = LRU(3)
    for k in "abc":
        c[k] = k
    assert c["a"] == "a"                                # a is now the most recent
    c["d"] = "d"
    assert list(c) == ["c", "a", "d"] and "b" not in c
    assert c.get("zz") is None


def test_caches_are_bounded(tmp_path):
    from app import ideal_lap, main
    from app.ideal_state import LRU
    assert isinstance(ideal_lap._CACHE, LRU) and isinstance(main._IDEAL_STORES, LRU)
    loads = []
    st = ST.IdealStore(tmp_path, [tmp_path], reducer=lambda p: loads.append(p.name) or p.name,
                       indexer=lambda p: {})
    assert isinstance(st._red, LRU)
    for i in range(st._red.maxsize + 3):
        st._reduction(tmp_path / f"f{i}.duckdb")
    assert len(st._red) == st._red.maxsize
    st._reduction(tmp_path / "f0.duckdb")              # evicted: loaded again
    assert loads.count("f0.duckdb") == 2


# ── Minor 5: a fit-code change, a new reduction version or new tyre-v3 peaks refit ──

@pytest.fixture(scope="module")
def states():
    from app.ideal_state import lap_state
    from synthetic_ideal import true_car, line, perfect_laps
    raws, _, _ = perfect_laps(true_car(), line(), n_laps=6, rng=np.random.default_rng(4), place_sd=0.0, lap_sd=0.0)
    return [lap_state(r) for r in raws]


def test_fingerprint_holds_the_fit_version_and_the_peaks(tmp_path, states, monkeypatch):
    from test_ideal_store import World, _red, _entry, NAME
    w = World(tmp_path)
    w.add(_red(NAME, states), _entry(NAME))
    peaks = {"front": 4.0, "rear": 3.5}

    def store():
        s = w.store()
        s.v3_peaks = lambda model: dict(peaks)
        return s

    s1 = store(); s1.group_for(w.path(NAME)); assert s1.fits_done == 1
    s2 = store(); s2.group_for(w.path(NAME)); assert s2.fits_done == 0          # unchanged: from the profile
    monkeypatch.setattr(C, "FIT_VERSION", C.FIT_VERSION + 1)
    s3 = store(); s3.group_for(w.path(NAME)); assert s3.fits_done == 1          # the fit code changed
    from app import ideal_state
    monkeypatch.setattr(ideal_state, "REDUCTION_VERSION", ideal_state.REDUCTION_VERSION + 1)
    s4 = store(); s4.group_for(w.path(NAME)); assert s4.fits_done == 1
    peaks["rear"] = 3.9
    s5 = store(); s5.group_for(w.path(NAME)); assert s5.fits_done == 1          # new tyre-v3 peaks (knee_vs_v3)


# ── Minor 6: the npz key follows the topography of files without Pos Y ──────

def test_reduction_key_follows_the_topography_file(tmp_path, monkeypatch):
    import os
    from app import ideal_state, track_topography as tt
    from synthetic_track import write_hilly_session
    no_y = write_hilly_session(tmp_path / "noy.duckdb", lap_offsets=(0.0,), with_pose=False)
    with_y = write_hilly_session(tmp_path / "y.duckdb", lap_offsets=(0.0,), with_pose=True)
    topo = tmp_path / "synth.topography.json"
    topo.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tt, "slug_for_track", lambda name: "synth")
    monkeypatch.setattr(tt, "topography_path", lambda slug: topo)
    k_no_y, k_y = ideal_state._cache_key(no_y), ideal_state._cache_key(with_y)
    st = topo.stat()
    os.utime(topo, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))       # the topography was rebuilt
    assert ideal_state._cache_key(no_y) != k_no_y                              # grade from it: rebuild
    assert ideal_state._cache_key(with_y) == k_y                               # grade from Pos Y: keep


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
