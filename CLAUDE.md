# LMU Coach

Telemetry coaching for Le Mans Ultimate (LMU). A recorder captures sim telemetry to DuckDB; a FastAPI backend analyzes laps (grip, tires, vehicle dynamics) and asks a local LLM for coaching; a single-page web UI shows track map, playback scrubber and traces.

## Layout
- `recorder/` - Windows-only Python (uv). Reads LMU shared memory via `pyLMUSharedMemory/` and writes `.duckdb` sessions. `recorder.py`, `recorder_control.py`.
- `backend/app/` - FastAPI app (`main.py`): `coach.py` (LLM prompts), `telemetry.py`, `grip_analysis.py`, `grip_profile.py`, `tire_model.py`, `vehicle_dynamics.py`, `track_db.py`. UI is one file: `static/index.html`.
- `backend/tools/` - CLI tools on the backend code, each with a `.ps1` wrapper: bulk grip-profile ingest, tyre-model validation (scorecard in DevNotes), lap list, tyre-force check. See its `README.md`.
- `data/` - `sessions/*.duckdb` (recordings), `grip_profiles/*.json` (per-car, written by the backend).
- Config: `.env` (Docker paths), `lmu-coach.config.json` (host agent, local LLM launcher). Examples are checked in as `*.example*`.

## Run
- Backend: `docker compose up --build` -> http://localhost:8080
- Recorder: `recorder\START_RECORDER.ps1` (does `uv sync; uv run recorder.py`)
- Local LLM is expected at `http://127.0.0.1:8001/v1` (OpenAI-compatible); the UI Settings send URL/model per request.

## Conventions
- Shell is PowerShell on Windows: chain with `;`, not `&&`.
- Log findings, bugs and TODOs in `DevNotes.md` under a dated heading (`### M/D/YYYY`), newest last.
- Backend runs in Docker with Python 3.12; the recorder uses Python 3.13 locally.
- Keep UI changes inside `static/index.html` unless asked otherwise.

## Don't touch
- `.env` and `lmu-coach.config.json` (personal paths).
- `data/sessions/*.duckdb` (large, personal recordings). Read-only queries are fine; never modify or delete.

## Prior context
The app was originally built in a claude.ai chat. Handoff notes from that session (decisions, known issues, next steps) live in @docs/HANDOFF.md when present.
