# Run this from the recorder\ directory
# PowerShell doesn't support && — use semicolons or separate lines
cd $PSScriptRoot
uv sync
uv run recorder.py
