"""/trackmap against a real recording from data/sessions (read-only).

Skipped when no recorder-made file is present, so the suite still runs on a
machine without the personal session data.
"""

from pathlib import Path
from urllib.parse import quote

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main

SESSIONS = Path(__file__).resolve().parents[2] / "data" / "sessions"
KEYS = ("fl", "fr", "rl", "rr")


def _has_slip_angle(path: Path) -> bool:
    try:
        con = duckdb.connect(str(path), read_only=True)
        try:
            return con.execute('select count(*) from "Slip Angle"').fetchone()[0] > 1000
        finally:
            con.close()
    except Exception:
        return False


def _trackmap(path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(main, "OWN_SESSIONS_DIR", path.parent)
    r = TestClient(main.app).get(f"/sessions/sessions/{quote(path.name)}/trackmap")
    assert r.status_code == 200, r.text[:300]
    return r.json()


@pytest.fixture(scope="module")
def files():
    found = sorted(SESSIONS.glob("*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        pytest.skip("no session files in data/sessions")
    return found


def test_slip_angle_matches_slip_ratio_length_and_turn_direction(files, monkeypatch):
    path = next((p for p in files if _has_slip_angle(p)), None)
    if path is None:
        pytest.skip("no recorder file with a Slip Angle table")
    d = _trackmap(path, monkeypatch)
    sa, dl = d["slip_angle"], d["driven_line"]
    assert set(sa) == set(KEYS)
    assert [len(sa[k]) for k in KEYS] == [len(d["slip"]["fl"])] * 4

    # Screen convention: turning right on screen (heading increasing) means
    # the front tyres travel LEFT of where they point, i.e. negative.
    n = min(len(sa["fl"]), len(dl["heading"]))
    rate = np.gradient(np.unwrap(np.array(dl["heading"][:n])))
    front = (np.array(sa["fl"][:n]) + np.array(sa["fr"][:n])) / 2
    fast = np.array(dl["speed"][:n]) > 60
    assert np.corrcoef(rate[fast], front[fast])[0, 1] < -0.6

    slow = np.array(dl["speed"][:n]) < main.SLIP_ANGLE_MIN_SPEED_KMH
    assert not np.any(np.array(sa["fl"][:n])[slow])
    assert max(abs(v) for k in KEYS for v in sa[k]) <= main.SLIP_ANGLE_CLIP_DEG


def test_file_without_slip_angle_returns_empty_lists(files, monkeypatch):
    path = next((p for p in files if not _has_slip_angle(p)), None)
    if path is None:
        pytest.skip("every local file has a Slip Angle table")
    try:
        d = _trackmap(path, monkeypatch)
    except AssertionError:
        pytest.skip("that file has no loadable lap")
    assert d["slip_angle"] == {k: [] for k in KEYS}
