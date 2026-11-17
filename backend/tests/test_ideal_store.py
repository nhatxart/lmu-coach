"""Ideal lap store, design T7 (refusals) and T8 (refit on fingerprint change,
borrowed load model, JSON round trip). Synthetic reductions only: the session
index and the reducer are injected, files are empty placeholders in tmp dirs."""
import json
from dataclasses import replace

import numpy as np
import pytest

from app import ideal_store as ST
from app.ideal_car import CarModel
from app.ideal_state import FileReduction, lap_state
from synthetic_ideal import true_car, line, perfect_laps

MODEL = "Synthetic GT"


@pytest.fixture(scope="module")
def states():
    raws, _ = perfect_laps(true_car(), line(), n_laps=8, gy_noise=0.0, rng=np.random.default_rng(4))
    return [lap_state(r) for r in raws]


def _red(name, laps, **kw):
    d = dict(file=name, mtime=1.0, size=100, meta={}, track="Synth", car_model=MODEL, car_class="LMGT3",
             game_version="1.0", load_source="susp", grade_source="pos_y", laps=laps)
    d.update(kw)
    return FileReduction(**d)


def _entry(name, slug="synth", layout=3900.0, **kw):
    e = dict(source="sessions", file=name, track=slug.title(), track_slug=slug, car_model=MODEL,
             car_class="LMGT3", layout_len_m=layout, coordinate_system="world_metres", laps=[])
    e.update(kw)
    return e


class World:
    """tmp sessions dir with placeholder files; index and reductions by name."""

    def __init__(self, tmp_path):
        self.sessions = tmp_path / "sessions"
        self.sessions.mkdir()
        self.root = tmp_path / "profiles" / "ideal_lap"
        self.reds, self.entries = {}, {}

    def add(self, red, entry):
        (self.sessions / red.file).write_bytes(b"")
        self.reds[red.file], self.entries[red.file] = red, entry

    def store(self):
        def reducer(p):
            r = self.reds[p.name]
            if isinstance(r, Exception):
                raise r
            return r
        return ST.IdealStore(self.root, [self.sessions], reducer=reducer,
                             indexer=lambda p: self.entries[p.name], v3_peaks=None)

    def path(self, name):
        return self.sessions / name


NAME = "Synth-Synthetic GT-2026-11-01T10_00_00Z.duckdb"
NAME2 = "Synth-Synthetic GT-2026-11-02T10_00_00Z.duckdb"


# ── T7: refusals ─────────────────────────────────────────────────────────────

def test_refuses_without_slip_angles_and_grip_fraction(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, [replace(s, gf=None, slip=None) for s in states[:6]]), _entry(NAME))
    out = w.store().car_for_lap(w.path(NAME))
    assert isinstance(out, ST.Refusal) and out.status == "insufficient_data"
    assert "no slip angles" in out.reason


def test_refuses_with_fewer_than_three_laps(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:2]), _entry(NAME))
    out = w.store().car_for_lap(w.path(NAME))
    assert isinstance(out, ST.Refusal)
    assert out.reason == "3 clean laps on this track needed, have 2"


def test_refuses_a_native_file(tmp_path, states):
    """LMU's own export: no slip angles, no Grip Fraction, no Susp Force."""
    w = World(tmp_path)
    native = [replace(s, gf=None, slip=None, susp=None, load=None) for s in states[:6]]
    w.add(_red(NAME, native, load_source=None), _entry(NAME))
    out = w.store().car_for_lap(w.path(NAME))
    assert isinstance(out, ST.Refusal) and "no slip angles" in out.reason


def test_unreadable_recording_is_a_refusal_not_an_exception(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:6]), _entry(NAME))
    w.reds[NAME] = OSError("damaged file")
    out = w.store().car_for_lap(w.path(NAME))
    assert isinstance(out, ST.Refusal) and "damaged file" in out.reason
    w.entries[NAME] = _entry(NAME, error="IOException: not a DuckDB file")
    assert isinstance(w.store().car_for_lap(w.path(NAME)), ST.Refusal)


# ── T8: store ────────────────────────────────────────────────────────────────

def test_refits_only_when_the_fingerprint_changes(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:6]), _entry(NAME))
    s1 = w.store()
    car, info = s1.car_for_lap(w.path(NAME))
    assert s1.fits_done == 1 and info["scope"] == "track" and info["n_laps"] == 6
    assert (w.root / f"{MODEL}.json").exists()
    s2 = w.store()                                    # a fresh process: reads the JSON profile
    car2, _ = s2.car_for_lap(w.path(NAME))
    assert s2.fits_done == 0
    np.testing.assert_allclose(car2.mu1, car.mu1)
    w.reds[NAME] = replace(w.reds[NAME], mtime=2.0)    # the file changed on disk
    s3 = w.store()
    s3.car_for_lap(w.path(NAME))
    assert s3.fits_done == 1
    w.reds[NAME] = replace(w.reds[NAME], laps=states[:7])   # a lap more
    s4 = w.store()
    s4.car_for_lap(w.path(NAME))
    assert s4.fits_done == 1


def test_a_file_with_one_lap_borrows_the_group_load_model(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:5]), _entry(NAME))
    w.add(_red(NAME2, states[5:6]), _entry(NAME2, layout=3910.0))
    st = w.store()
    _, info2 = st.car_for_lap(w.path(NAME2))
    _, info1 = st.car_for_lap(w.path(NAME))
    assert info2["load_model"] == "group" and info1["load_model"] == "file"
    assert info2["n_files"] == 2 and info2["n_laps"] == 6
    prof = json.loads((w.root / f"{MODEL}.json").read_text(encoding="utf-8"))
    assert prof["files"][NAME2]["load_model"]["scope"] == "group"
    assert prof["files"][NAME]["load_model"]["scope"] == "file"
    assert any("load model from the group" in n for n in info2["notes"])


def test_json_round_trip(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:6]), _entry(NAME))
    g = w.store().group_for(w.path(NAME))
    back = ST.GroupEntry.from_dict(json.loads(json.dumps(g.to_dict())))
    assert back.fingerprint == g.fingerprint and back.beta == g.beta and back.confidence == g.confidence
    a, b = g.fit.car_for(NAME), back.fit.car_for(NAME)
    assert CarModel.from_dict(json.loads(json.dumps(a.to_dict()))).to_dict() == a.to_dict()
    v, gx, gy, gz = np.array([30.0, 60.0]), np.zeros(2), np.array([1.2, 1.5]), np.zeros(2)
    np.testing.assert_allclose(a.axles(v, gx, gy, gz), b.axles(v, gx, gy, gz))
    assert back.fit.info == g.fit.info


def test_layouts_more_than_three_percent_apart_are_separate_groups(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:4]), _entry(NAME, layout=3900.0))
    w.add(_red(NAME2, states[4:8]), _entry(NAME2, layout=4100.0))
    g1 = w.store().group_for(w.path(NAME))
    assert g1.files == [NAME] and g1.key == "synth|3900"


def test_car_scope_fallback_from_other_tracks(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:2]), _entry(NAME, slug="here"))
    w.add(_red(NAME2, states[2:8]), _entry(NAME2, slug="elsewhere"))
    car, info = w.store().car_for_lap(w.path(NAME))
    assert info["scope"] == "car" and info["transferred"]
    assert info["beta"] == ST.BETA_CAR and info["confidence"] == "low"
    assert info["n_laps"] == 8


def test_newest_version_with_three_laps_is_the_fit_set(tmp_path, states):
    w = World(tmp_path)
    w.add(_red(NAME, states[:4], game_version="1.0"), _entry(NAME))
    w.add(_red(NAME2, states[4:8], game_version="1.1"), _entry(NAME2))
    g = w.store().group_for(w.path(NAME))
    assert g.game_version == "1.1" and g.files == [NAME2]
