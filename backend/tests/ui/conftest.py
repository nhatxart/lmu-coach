"""Browser smoke tests of index.html (chase view): a real backend in a
subprocess, on synthetic sessions, driven by headless Chromium.

Skips cleanly when Playwright, uvicorn or the Chromium build are missing, so
the plain backend command (no --with playwright) still passes.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # synthetic_track / synthetic_session
sys.path.insert(0, str(HERE))               # serve_for_ui (file names)

from serve_for_ui import (CMP_A, CMP_B, CMP_HYPER, CMP_KEPT, CMP_LAYOUT, CMP_NOLD,  # noqa: E402
                          CMP_V1, CMP_WAVE, HILLY, LEGACY, LIMITS, PARTIAL, SLIP)

# SwiftShader: software WebGL, so the chase view renders without a GPU.
# Verified with Chromium 156 headless (Playwright 1.5x) on Windows 11.
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_compare_files(sessions: Path, kept: Path | None) -> None:
    """The lap-comparison fixtures (serve_for_ui.CMP_*), all on the
    Synthetic Hill Ring. CMP_KEPT goes to the reference-laps folder."""
    from synthetic_track import write_hilly_session

    write_hilly_session(sessions / CMP_A, vehicle_class="LMGT3", lap_offsets=(0.0, 1.0, -1.0))
    write_hilly_session(sessions / CMP_B, vehicle_class="LMGT3", lap_offsets=(2.0, 2.0), speed_scale=0.97)
    if kept is None:
        return
    write_hilly_session(kept / CMP_KEPT, vehicle_class="LMGT3", lap_offsets=(2.0, 2.0), speed_scale=0.97)
    write_hilly_session(sessions / CMP_HYPER, vehicle_class="Hypercar")
    write_hilly_session(sessions / CMP_LAYOUT, vehicle_class="LMGT3", lap_dist_scale=0.9)
    write_hilly_session(sessions / CMP_V1, vehicle_class="LMGT3", with_gps=False)
    # offered (its layout is unknown without Lap Dist), so slower than CMP_B
    # to keep "Best other file" on CMP_B / CMP_KEPT
    write_hilly_session(sessions / CMP_NOLD, vehicle_class="LMGT3", with_lap_dist=False, speed_scale=0.96)
    # slower than CMP_B, so it is never "Best other file"; its speed varies
    # along the lap, so the delta to it both rises and falls
    write_hilly_session(sessions / CMP_WAVE, vehicle_class="LMGT3", lap_offsets=(0.5, 0.5),
                        speed_scale=0.95, speed_wave=0.1)


def _start(sessions: Path, ref: Path, kept: Path | None = None):
    port = _free_port()
    args = [sys.executable, str(HERE / "serve_for_ui.py"), str(sessions), str(port), str(ref)]
    if kept is not None:
        args.append(str(kept))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        if proc.poll() is not None or time.time() > deadline:
            out = proc.stdout.read().decode(errors="replace") if proc.poll() is not None else ""
            proc.kill()
            pytest.fail(f"backend did not start: {out[-2000:]}")
        time.sleep(0.3)
    return proc, base


def _stop(proc):
    proc.kill()
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    pytest.importorskip("uvicorn")
    from synthetic_track import GPS_REF_LAT, GPS_REF_LON, write_hilly_session

    root = tmp_path_factory.mktemp("ui")
    sessions, ref, kept = root / "sessions", root / "ref", root / "kept"
    sessions.mkdir()
    ref.mkdir()        # no built topography, so the legacy file is flat
    kept.mkdir()
    # The generator's GPS reference as the track's fixed origin, so every
    # fixture file shares one frame like a catalog track (track_origin). A
    # centre alone builds no topography.
    (ref / "sources").mkdir()
    (ref / "sources" / "synthetic_hill_ring.json").write_text(
        json.dumps({"center": [GPS_REF_LAT, GPS_REF_LON]}), encoding="utf-8")
    write_hilly_session(sessions / HILLY, with_pose=True)
    write_hilly_session(sessions / LEGACY, with_pose=False)
    write_hilly_session(sessions / LIMITS, with_pose=True, track_edge_m=5.5)
    write_hilly_session(sessions / SLIP, with_pose=True, slip_angles=True)
    write_hilly_session(sessions / PARTIAL, with_pose=True, trailing_partial_m=300)
    _write_compare_files(sessions, kept)

    proc, base = _start(sessions, ref, kept)
    yield base
    _stop(proc)


@pytest.fixture(scope="session")
def server_no_origin(tmp_path_factory):
    """CMP_A and CMP_B with an EMPTY track reference dir: no track origin,
    so each file is projected around its own centre (coordinate_origin
    kind "file") and a comparison has no shared frame."""
    pytest.importorskip("uvicorn")
    root = tmp_path_factory.mktemp("ui_no_origin")
    sessions, ref = root / "sessions", root / "ref"
    sessions.mkdir()
    ref.mkdir()
    _write_compare_files(sessions, None)
    proc, base = _start(sessions, ref)
    yield base
    _stop(proc)


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    pw = sync_api.sync_playwright().start()
    try:
        b = pw.chromium.launch(args=CHROMIUM_ARGS)
    except Exception as e:          # Chromium not installed (playwright install chromium)
        pw.stop()
        pytest.skip(f"Chromium unavailable: {e}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture
def page(browser):
    """A fresh context per test (clean localStorage); console errors and
    uncaught exceptions are collected in page.errors."""
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    p = ctx.new_page()
    p.errors = []

    def on_console(m):
        # The page probes the host agent (recorder_control, :8420), which
        # is not running here; that refused connection is expected.
        if m.type == "error" and ":8420" not in (m.location or {}).get("url", ""):
            p.errors.append(m.text)
    p.on("console", on_console)
    p.on("pageerror", lambda e: p.errors.append(str(e)))
    yield p
    ctx.close()


@pytest.fixture
def shot_dir():
    """UI_SHOT_DIR=<dir> keeps screenshots for a human to look at."""
    d = os.environ.get("UI_SHOT_DIR")
    return Path(d) if d else None
