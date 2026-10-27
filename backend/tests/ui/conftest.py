"""Browser smoke tests of index.html (chase view): a real backend in a
subprocess, on synthetic sessions, driven by headless Chromium.

Skips cleanly when Playwright, uvicorn or the Chromium build are missing, so
the plain backend command (no --with playwright) still passes.
"""

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

from serve_for_ui import HILLY, LEGACY, LIMITS  # noqa: E402

# SwiftShader: software WebGL, so the chase view renders without a GPU.
# Verified with Chromium 156 headless (Playwright 1.5x) on Windows 11.
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    pytest.importorskip("uvicorn")
    from synthetic_track import write_hilly_session

    root = tmp_path_factory.mktemp("ui")
    sessions, ref = root / "sessions", root / "ref"
    sessions.mkdir()
    ref.mkdir()        # empty: no built topography, so the legacy file is flat
    write_hilly_session(sessions / HILLY, with_pose=True)
    write_hilly_session(sessions / LEGACY, with_pose=False)
    write_hilly_session(sessions / LIMITS, with_pose=True, track_edge_m=5.5)

    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "serve_for_ui.py"), str(sessions), str(port), str(ref)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    yield base
    proc.kill()
    proc.wait(timeout=10)


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
