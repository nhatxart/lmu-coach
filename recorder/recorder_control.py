"""
recorder_control.py — a small local HTTP service that wraps recorder.py as
a managed subprocess, so the web UI (backend/app/static/index.html's
Recording tab) has something to start/stop/watch.

WHY THIS IS A SEPARATE SERVICE, NOT PART OF THE DOCKERIZED BACKEND
────────────────────────────────────────────────────────────────────────────
The main backend (backend/app/main.py) runs inside a Docker container (see
docker-compose.yml). recorder.py needs direct access to LMU's shared-memory
mapped file on the WINDOWS HOST — that's an OS-level mechanism a container
cannot reach at all, containerized or not, regardless of volume mounts
(volume mounts share files, not live memory-mapped IPC segments). So the
Dockerized backend can never itself launch or control the recorder.

This service is the actual answer to "integrate the recorder into the web
UI": it runs directly on the Windows host (same machine as LMU and the
recorder), started the same way the recorder itself already is (see
START_RECORDER.ps1's pattern), and exposes plain HTTP endpoints. The
browser's Recording tab talks to THIS service directly at
http://localhost:8420 (not through the Dockerized backend, which has no
path to reach it) to start/stop recording and stream logs.

This also directly serves the "detect session exit automatically" ask: it
does NOT replace recorder.py's own freeze-detection fix (that's a real fix,
already applied, and is the primary mechanism) — it's the fallback control
surface for when a person wants to explicitly start/stop instead of, or in
addition to, that automatic detection, and a way to actually SEE that the
recorder is alive and what it's doing, which a bare terminal window habit
doesn't give you once you've alt-tabbed into the game.

RUNNING THIS
────────────────────────────────────────────────────────────────────────────
From the recorder/ directory (same place recorder.py lives):
    uv run recorder_control.py
Then open the web UI's Recording tab — it talks to http://localhost:8420.
Leave this running in the background (or add it to a startup script)
instead of manually launching recorder.py yourself; the Recording tab's
start button launches recorder.py FOR you as a subprocess of this service.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import urllib.error
import urllib.request
import os
from contextlib import asynccontextmanager
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

RECORDER_DIR = Path(__file__).parent
RECORDER_SCRIPT = RECORDER_DIR / "recorder.py"
# Must match STOP_FILE in recorder.py. Creating this file asks the recorder
# to save the current session and exit (see recorder.py for why a file is
# used instead of a signal: on Windows Popen.terminate() force-kills and no
# signal handler ever runs).
STOP_FILE = RECORDER_DIR / ".stop_requested"

# How long a requested stop may take before the process is force-killed.
# Saving a long session now takes seconds, but a force-kill DURING the save
# destroys the whole session (this is what produced the empty 12KB .tmp
# files), so the limit is deliberately generous.
GRACEFUL_STOP_TIMEOUT_S = 300.0
IS_WINDOWS = sys.platform == "win32"

# How many recent log lines to keep in memory for a client that connects
# AFTER the recorder already started (so opening the Recording tab mid-
# session still shows useful recent context, not just an empty log from
# that point forward).
LOG_BUFFER_LINES = 500

# ════════════════════════════════════════════════════════════════════════════
# HOST AGENT additions: shared config file, request guard, LLM auto-start.
#
# This service already runs on the Windows host (it has to, to reach LMU's
# shared memory). The same property makes it the only place that can start
# your LLM server for the Dockerized backend, which cannot launch programs on
# the host. So it now also owns: the stack config file, LLM health checks, and
# running the LLM launcher script when the server is not up.
# ════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(os.environ.get("LMU_COACH_CONFIG")
                   or (RECORDER_DIR.parent / "lmu-coach.config.json"))
LLM_LOG_PATH = CONFIG_PATH.with_name("lmu-coach-llm.log")

DEFAULT_CONFIG: dict = {
    "config_version": 1,
    "host_agent": {
        # Where the browser finds this service, and which web-UI origins may
        # talk to it (see GuardMiddleware). Edit the JSON file by hand if you
        # serve the UI from a different port.
        "url": "http://localhost:8420",
        "allowed_origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
    },
    "llm": {
        "auto_start": True,
        "launcher_path": "",
        # Checked FROM THE HOST. llama-server: /health (503 while the model
        # loads, 200 when ready). LM Studio/Ollama: use .../v1/models.
        "health_url": "http://127.0.0.1:8001/health",
        # Used BY THE DOCKER BACKEND to call the model.
        "backend_base_url": "http://host.docker.internal:8001/v1",
        "model": "local",
        "startup_timeout_s": 240,
        "request_timeout_s": 300,
        "show_window": True,
    },
    "recorder": {"output_dir": ""},
    "stack_paths": {
        "lmu_telemetry_dir": "",
        "reference_laps_dir": "",
        "grip_profiles_dir": "",
    },
}

_BOOL_FIELDS = {("llm", "auto_start"), ("llm", "show_window")}
_INT_LIMITS = {("llm", "startup_timeout_s"): (10, 1800),
               ("llm", "request_timeout_s"): (10, 3600)}
_URL_FIELDS = {("host_agent", "url"), ("llm", "health_url"), ("llm", "backend_base_url")}


def _clean_str(v) -> str:
    s = str(v).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()  # tolerate a path pasted with surrounding quotes
    return s


def validate_config(raw) -> tuple[dict, list[str]]:
    """Normalise a config object. Unknown keys are dropped, missing keys take
    defaults, wrong types/ranges are reported in the returned error list."""
    errors: list[str] = []
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not isinstance(raw, dict):
        return cfg, ["config must be a JSON object"]
    for section, defaults in DEFAULT_CONFIG.items():
        if section == "config_version":
            continue
        incoming = raw.get(section)
        if incoming is None:
            continue
        if not isinstance(incoming, dict):
            errors.append(f"'{section}' must be an object")
            continue
        for key in defaults:
            if key not in incoming:
                continue
            val, where = incoming[key], f"{section}.{key}"
            if key == "allowed_origins":
                if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                    errors.append(f"{where} must be a list of origins")
                    continue
                origins = [_clean_str(x).rstrip("/") for x in val if _clean_str(x)]
                if any(not re.match(r"^https?://[^/\s]+$", o) for o in origins):
                    errors.append(f"{where} entries must look like http://localhost:8080")
                    continue
                cfg[section][key] = origins
            elif (section, key) in _BOOL_FIELDS:
                if not isinstance(val, bool):
                    errors.append(f"{where} must be true or false")
                    continue
                cfg[section][key] = val
            elif (section, key) in _INT_LIMITS:
                lo, hi = _INT_LIMITS[(section, key)]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not lo <= val <= hi:
                    errors.append(f"{where} must be a number between {lo} and {hi}")
                    continue
                cfg[section][key] = int(val)
            else:
                if not isinstance(val, str):
                    errors.append(f"{where} must be text")
                    continue
                s = _clean_str(val)
                if len(s) > 1000:
                    errors.append(f"{where} is too long")
                    continue
                if (section, key) in _URL_FIELDS:
                    if s and not re.match(r"^https?://", s):
                        errors.append(f"{where} must start with http:// or https://")
                        continue
                    s = s.rstrip("/")
                cfg[section][key] = s
    return cfg, errors


def load_config() -> tuple[dict, Optional[str]]:
    if not CONFIG_PATH.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG)), None
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))  # -sig: tolerate a BOM
    except Exception as e:  # noqa: BLE001
        return json.loads(json.dumps(DEFAULT_CONFIG)), f"could not read {CONFIG_PATH.name}: {e}"
    cfg, errs = validate_config(raw)
    return cfg, ("; ".join(errs) if errs else None)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)  # atomic: a crash never leaves a half-written config


# ── Request guard ─────────────────────────────────────────────────────────────
# This service can start programs and write a config file, and it listens on
# localhost. Any web page open in your browser can send requests to localhost,
# so without a guard a random website could trigger it. Two checks:
#   * Host header must be localhost/127.0.0.1 (blocks DNS-rebinding tricks).
#   * Browser requests carry an Origin header; state-changing requests from an
#     origin that is not the lmu-coach web UI are refused, and CORS headers are
#     only ever granted to the allowed origins.
_ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1", "[::1]"}


def _hostname(host_header: str) -> str:
    h = host_header.strip().lower()
    if h.startswith("["):
        return h[: h.find("]") + 1] if "]" in h else h
    return h.rsplit(":", 1)[0] if ":" in h else h


def _allowed_origins() -> set[str]:
    return set(load_config()[0]["host_agent"]["allowed_origins"])


class GuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def _reply(self, send, status: int, body: bytes = b"", extra=()):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()), *extra]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if _hostname(headers.get("host", "")) not in _ALLOWED_HOSTNAMES:
            await self._reply(send, 403, b'{"error":"host not allowed"}')
            return
        origin = headers.get("origin")
        origin_ok = origin is None or origin.rstrip("/") in _allowed_origins()
        method = scope["method"]
        if method == "OPTIONS" and origin is not None:  # CORS preflight
            if not origin_ok:
                await self._reply(send, 403, b'{"error":"origin not allowed"}')
                return
            await self._reply(send, 204, b"", [
                (b"access-control-allow-origin", origin.encode()),
                (b"access-control-allow-methods", b"GET, POST, PUT, OPTIONS"),
                (b"access-control-allow-headers", b"content-type"),
                (b"access-control-allow-private-network", b"true"),
                (b"access-control-max-age", b"600"), (b"vary", b"Origin")])
            return
        if origin is not None and not origin_ok and method not in ("GET", "HEAD"):
            await self._reply(send, 403, b'{"error":"origin not allowed"}')
            return
        if origin is not None and origin_ok:
            async def send_with_cors(message):
                if message["type"] == "http.response.start":
                    message = dict(message)
                    message["headers"] = list(message.get("headers", [])) + [
                        (b"access-control-allow-origin", origin.encode()), (b"vary", b"Origin")]
                await send(message)
            await self.app(scope, receive, send_with_cors)
        else:
            await self.app(scope, receive, send)


# ── LLM server: health check + launcher ───────────────────────────────────────
def _probe_health(url: str, timeout: float = 2.0) -> tuple[str, Optional[int]]:
    """-> (state, http_status). llama-server answers 503 while the model is
    loading and 200 once ready; connection refused means nothing is listening."""
    if not url:
        return "down", None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return ("ready" if 200 <= r.status < 300 else "loading"), r.status
    except urllib.error.HTTPError as e:
        return ("loading" if e.code == 503 else "down"), e.code
    except Exception:  # noqa: BLE001 — refused/timeout/DNS: not up
        return "down", None


def _launcher_command(path: str) -> list[str]:
    ext = Path(path).suffix.lower()
    if ext == ".ps1":
        exe = ((shutil.which("powershell") if IS_WINDOWS else None)
               or shutil.which("pwsh") or shutil.which("powershell") or "powershell")
        return [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path]
    if ext in (".bat", ".cmd"):
        return ["cmd", "/c", path]
    if ext == ".py":
        return [sys.executable, path]
    return [path]


def _spawn_llm(path: str, show_window: bool) -> subprocess.Popen:
    """Start the launcher DETACHED so the LLM server outlives this service.
    The launcher runs llama-server in the foreground, so it needs a console
    of its own: a visible new window (you see the server log) or a hidden one
    with output sent to lmu-coach-llm.log."""
    kwargs: dict = {"cwd": str(Path(path).parent)}
    if IS_WINDOWS:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if show_window:
            flags |= subprocess.CREATE_NEW_CONSOLE
        else:
            flags |= subprocess.CREATE_NO_WINDOW
            kwargs.update(stdin=subprocess.DEVNULL, stdout=open(LLM_LOG_PATH, "ab"),
                          stderr=subprocess.STDOUT)
        kwargs["creationflags"] = flags
    else:
        kwargs.update(start_new_session=True, stdin=subprocess.DEVNULL,
                      stdout=open(LLM_LOG_PATH, "ab"), stderr=subprocess.STDOUT)
    return subprocess.Popen(_launcher_command(path), **kwargs)


class LlmLauncher:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.started_at: Optional[float] = None
        self.lock = asyncio.Lock()  # two browser tabs must not launch two servers

    async def status(self) -> dict:
        cfg = load_config()[0]["llm"]
        loop = asyncio.get_running_loop()
        state, http_status = await loop.run_in_executor(None, _probe_health, cfg["health_url"])
        alive = self.proc is not None and self.proc.poll() is None
        failed = None
        if (self.proc is not None and not alive and state != "ready" and self.started_at
                and time.time() - self.started_at < cfg["startup_timeout_s"]):
            # exited soon after launch without the server coming up = a failed
            # launch. (A later exit, e.g. you closed the server, is not a failure.)
            failed = (f"The launcher exited with code {self.proc.returncode} before the "
                      "server came up. Run the launcher by hand to see the error.")
        path = cfg["launcher_path"]
        return {
            "state": state, "http_status": http_status,
            "launching": alive and state != "ready", "failed": failed,
            "pid": self.proc.pid if alive else None,
            "elapsed_s": round(time.time() - self.started_at, 1) if (alive and self.started_at) else None,
            "health_url": cfg["health_url"],
            "launcher_configured": bool(path),
            "launcher_exists": bool(path) and Path(path).is_file(),
        }

    async def ensure(self) -> dict:
        async with self.lock:
            st = await self.status()
            if st["state"] == "ready":
                return {"ok": True, "state": "ready", "started": False}
            if st["state"] == "loading" or st["launching"]:
                return {"ok": True, "state": "loading", "started": False}  # already starting
            cfg = load_config()[0]["llm"]
            path = cfg["launcher_path"]
            if not path:
                return {"ok": False, "error": "No launcher script is configured. Set one in "
                                              "Settings > LLM server, or start your LLM server manually."}
            if not Path(path).is_file():
                return {"ok": False, "error": f"Launcher script not found: {path}"}
            try:
                self.proc = _spawn_llm(path, cfg["show_window"])
            except OSError as e:
                return {"ok": False, "error": f"Could not start the launcher: {e}"}
            self.started_at = time.time()
            return {"ok": True, "state": "loading", "started": True}


llm = LlmLauncher()


@asynccontextmanager
async def lifespan(_app):
    yield
    # Service is shutting down (Ctrl+C in its window): if a recording is
    # running, save it first instead of orphaning or killing it. Windows
    # gives a console only a few seconds after its X button is clicked, so
    # use Ctrl+C or the Stop button rather than closing the window.
    if state.is_running:
        await state.stop()
        if state._stop_task is not None:
            await state._stop_task


app = FastAPI(title="lmu-coach recorder control", lifespan=lifespan)

# CORS + request guard (see GuardMiddleware): only the lmu-coach web UI origin
# may call this service from a browser. This replaces a wildcard CORS policy.
app.add_middleware(GuardMiddleware)


class RecorderState:
    """
    Owns the recorder.py subprocess and its log buffer. One instance for
    the lifetime of this control service (there is only ever meant to be
    one recorder process at a time — starting a second while one is
    already running is rejected rather than silently spawning two
    processes writing to the same output directory).
    """

    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.log_lines: deque[str] = deque(maxlen=LOG_BUFFER_LINES)
        self.started_at: Optional[float] = None
        self.stopping: bool = False
        self._reader_task: Optional[asyncio.Task] = None
        self._stop_task: Optional[asyncio.Task] = None
        # New-line "subscribers" for live streaming: each active SSE
        # connection gets its own asyncio.Queue that the reader task pushes
        # every new line into, so multiple browser tabs can watch the same
        # live log independently without one client's slow consumption
        # blocking another's.
        self._subscribers: set[asyncio.Queue] = set()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _emit(self, line: str) -> None:
        self.log_lines.append(line)
        for q in list(self._subscribers):
            q.put_nowait(line)

    async def start(self) -> dict:
        if self.is_running:
            return {"ok": False, "error": "Recorder is already running."}
        if not RECORDER_SCRIPT.exists():
            return {"ok": False, "error": f"recorder.py not found at {RECORDER_SCRIPT}"}

        self.log_lines.clear()
        self.stopping = False
        STOP_FILE.unlink(missing_ok=True)  # never start with a stale stop request
        self.started_at = time.time()
        # Launch recorder.py DIRECTLY with this service's own interpreter
        # (this service runs inside the recorder project's environment, so
        # duckdb is already importable). The previous version launched
        # `uv run recorder.py`; uv starts the real recorder as a CHILD of
        # uv.exe, so stopping "the process" only touched the wrapper on
        # Windows and left the actual recorder alive, still holding its
        # output file open — the reason the file could not be opened until
        # the whole PowerShell window was closed. A direct launch means
        # self.process IS the recorder.
        # PYTHONUNBUFFERED: without it, output piped to us is block-buffered
        # and log lines would arrive in delayed bursts (or only at exit).
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-u", str(RECORDER_SCRIPT)]
        out_dir = load_config()[0]["recorder"]["output_dir"]
        if out_dir:
            cmd += ["--output-dir", out_dir]
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(RECORDER_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge streams — recorder.py's prints and any traceback both matter
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered
                env=env,
            )
        except OSError as e:
            return {"ok": False, "error": f"Could not launch recorder.py: {e}"}
        self._reader_task = asyncio.create_task(self._read_output())
        return {"ok": True}

    async def stop(self) -> dict:
        """Request a graceful stop and return IMMEDIATELY. Saving a session
        can take a while; the browser must not hold a request open (or give
        up and assume failure) for that long. Progress is visible through
        the log stream and the `stopping` flag on /status."""
        if not self.is_running:
            return {"ok": False, "error": "Recorder is not running."}
        if self.stopping:
            return {"ok": True, "stopping": True}
        self.stopping = True
        self._emit("[control] stop requested — recorder will save the session, then exit")
        STOP_FILE.write_text("stop")
        self._stop_task = asyncio.create_task(self._await_exit_or_kill())
        return {"ok": True, "stopping": True}

    async def _await_exit_or_kill(self) -> None:
        deadline = time.monotonic() + GRACEFUL_STOP_TIMEOUT_S
        while self.is_running and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        if self.is_running:
            self._emit(f"[control] recorder did not exit within {GRACEFUL_STOP_TIMEOUT_S:.0f}s "
                       "— force-killing (the session being saved may be lost)")
            self._kill_tree()
        STOP_FILE.unlink(missing_ok=True)

    def _kill_tree(self) -> None:
        """Last resort. /T also kills any child processes on Windows."""
        if self.process is None:
            return
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               capture_output=True, timeout=15)
            else:
                self.process.kill()
        except Exception as e:  # noqa: BLE001 — nothing more we can do; report it
            self._emit(f"[control] force-kill failed: {e}")

    async def _read_output(self) -> None:
        """Runs until the recorder's stdout closes (i.e. it has exited),
        pushing each line into the shared buffer and out to any live
        subscribers. Keeping this reader alive for the recorder's whole
        life also prevents the pipe from filling up and blocking the
        recorder's own print() calls."""
        proc = self.process
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            self._emit(line.rstrip("\r\n"))
        await loop.run_in_executor(None, proc.wait)
        self._emit(f"[recorder process exited, code {proc.returncode}]")
        self.stopping = False
        STOP_FILE.unlink(missing_ok=True)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)


state = RecorderState()


@app.get("/status")
async def status():
    return {
        "running": state.is_running,
        "stopping": state.stopping,
        "started_at": state.started_at,
        "recent_log": list(state.log_lines),
    }


@app.post("/start")
async def start_recording():
    return await state.start()


@app.post("/stop")
async def stop_recording():
    return await state.stop()


@app.get("/logs/stream")
async def stream_logs():
    """
    Server-Sent Events stream of new log lines as they arrive. The
    Recording tab opens this via EventSource so logs appear live rather
    than needing to poll /status repeatedly.
    """
    q = state.subscribe()

    async def event_gen():
        try:
            # Replay recent buffered lines first so a client connecting
            # mid-session isn't starting from a blank log.
            for line in list(state.log_lines):
                yield f"data: {line}\n\n"
            while True:
                line = await q.get()
                yield f"data: {line}\n\n"
        finally:
            state.unsubscribe(q)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── Config, path checks, LLM endpoints ────────────────────────────────────────
@app.get("/ping")
async def ping():
    return {"ok": True, "service": "lmu-coach host agent", "config_path": str(CONFIG_PATH)}


@app.get("/config")
async def get_config():
    cfg, err = load_config()
    return {"config": cfg, "path": str(CONFIG_PATH), "exists": CONFIG_PATH.exists(), "load_error": err}


@app.put("/config")
async def put_config(request: Request):
    """Replace the config with the submitted one (missing keys take defaults)."""
    body = await request.body()
    if len(body) > 65536:
        return JSONResponse({"ok": False, "errors": ["config too large"]}, status_code=413)
    try:
        raw = json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "errors": ["body is not valid JSON"]}, status_code=400)
    cfg, errors = validate_config(raw)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    save_config(cfg)
    return {"ok": True, "config": cfg, "path": str(CONFIG_PATH)}


def _check_paths(cfg: dict) -> dict:
    items = [
        ("LLM launcher script", cfg["llm"]["launcher_path"], "file"),
        ("Recorder output folder", cfg["recorder"]["output_dir"], "dir_or_create"),
        ("LMU telemetry folder", cfg["stack_paths"]["lmu_telemetry_dir"], "dir"),
        ("Reference laps folder", cfg["stack_paths"]["reference_laps_dir"], "dir"),
        ("Grip profiles folder", cfg["stack_paths"]["grip_profiles_dir"], "dir"),
    ]
    out = []
    for label, path, kind in items:
        if not path:
            out.append({"label": label, "path": "", "status": "not set"})
            continue
        p = Path(path)
        if kind == "file":
            status = "ok" if p.is_file() else ("wrong type (is a folder)" if p.exists() else "missing")
        elif p.is_dir():
            status = "ok"
        elif p.exists():
            status = "wrong type (is a file)"
        else:
            status = "will be created" if kind == "dir_or_create" else "missing"
        out.append({"label": label, "path": path, "status": status})
    return {"paths": out}


@app.get("/paths/check")
async def check_paths():
    """Check the SAVED config's paths."""
    return _check_paths(load_config()[0])


@app.post("/paths/check")
async def check_paths_for(request: Request):
    """Check the paths in a submitted (not yet saved) config, so the Settings
    form can be validated before saving. Nothing is written."""
    try:
        raw = json.loads((await request.body()).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "errors": ["body is not valid JSON"]}, status_code=400)
    cfg, errors = validate_config(raw)
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    return _check_paths(cfg)


@app.get("/llm/status")
async def llm_status():
    return await llm.status()


@app.post("/llm/ensure")
async def llm_ensure():
    """If the LLM server is not up, run the configured launcher. Returns at
    once; the caller polls /llm/status until state is 'ready'."""
    return await llm.ensure()


if __name__ == "__main__":
    import uvicorn
    print("lmu-coach recorder control service")
    print(f"  recorder.py: {RECORDER_SCRIPT}")
    _port = int(os.environ.get("LMU_COACH_AGENT_PORT", "8420"))
    print(f"  config file: {CONFIG_PATH}")
    print(f"  listening on http://localhost:{_port}")
    print("  (the web UI's Recording tab talks to this service directly — "
          "leave this window open)")
    uvicorn.run(app, host="127.0.0.1", port=_port, log_level="warning")
