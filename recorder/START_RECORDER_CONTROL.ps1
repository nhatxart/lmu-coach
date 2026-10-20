# Run this from the recorder\ directory
# PowerShell doesn't support && - use semicolons or separate lines
#
# Starts the lmu-coach host agent the web UI talks to. It runs on this PC because
# the recorder (LMU shared memory) and your LLM launcher can only be started from
# the host, not from Docker. Settings are stored in ..\lmu-coach.config.json.
# Leave this window open - it manages recorder.py as a subprocess of
# itself (start/stop it from the browser, or from here with Ctrl+C).
# This does NOT run recorder.py directly; use START_RECORDER.ps1 for that
# if you'd rather run the recorder standalone without the web UI.
# Either way it is the same recorder: do not run both at once.
cd $PSScriptRoot
uv sync
uv run recorder_control.py
