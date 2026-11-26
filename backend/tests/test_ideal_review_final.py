"""Final whole-branch review (final-review.md), backend findings M7 and M8."""
import threading
from pathlib import Path

from app import ideal_store as ST
from app import session_index as SI


def test_the_store_asks_the_index_with_each_folders_source(tmp_path, monkeypatch):
    """M7: the store asked file_entry(p, "sessions") for kept files too."""
    sess, kept = tmp_path / "sessions", tmp_path / "kept"
    sess.mkdir()
    kept.mkdir()
    seen = []
    monkeypatch.setattr(SI, "file_entry", lambda p, source: seen.append((Path(p).parent.name, source)) or {})
    store = ST.IdealStore(tmp_path / "root", [sess, kept])
    store.indexer(sess / "a.duckdb")
    store.indexer(kept / "b.duckdb")
    assert seen == [("sessions", "sessions"), ("kept", "reference_laps")]


def test_the_index_cache_is_not_thrashed_by_the_source(tmp_path, monkeypatch):
    """M7: the cache hit required the same source, so asking one file with
    two sources re-read it every time."""
    f = tmp_path / "x.duckdb"
    f.write_bytes(b"not a real file")
    reads = []

    def fake_read(path, source):
        reads.append(source)
        return {"source": source, "file": Path(path).name}
    monkeypatch.setattr(SI, "_read_entry", fake_read)
    SI._CACHE.clear()
    a = SI.file_entry(f, "sessions")
    b = SI.file_entry(f, "reference_laps")
    c = SI.file_entry(f, "sessions")
    assert reads == ["sessions"]
    assert (a["source"], b["source"], c["source"]) == ("sessions", "reference_laps", "sessions")
    SI._CACHE.clear()


def test_concurrent_saves_use_their_own_tmp_files(tmp_path, monkeypatch):
    """M8: two requests for one car wrote the same <car>.json.tmp."""
    store = ST.IdealStore(tmp_path / "root", [tmp_path])
    tmps, gate = [], threading.Barrier(2)
    real = Path.replace

    def rec(self, target):
        tmps.append(self.name)
        return real(self, target)
    monkeypatch.setattr(Path, "replace", rec)

    def save():
        gate.wait()
        store.save_profile({"car_model": "Car X", "version": ST.VERSION, "groups": {}})
    ts = [threading.Thread(target=save) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(tmps) == 2 and tmps[0] != tmps[1]
    assert (tmp_path / "root" / f"{ST.safe_name('Car X')}.json").exists()
