# Run this from the recorder\ directory
# PowerShell doesn't support && - use semicolons or separate lines
#
# EXPERIMENTAL high-frequency recorder (recorder_hf.py). The normal recorder
# (START_RECORDER.ps1 / the web Record tab) is unchanged; do not run both at once.
#
#   .\START_RECORDER_HF.ps1                 record, polling at 1000 Hz
#   .\START_RECORDER_HF.ps1 --probe 10      measure LMU's real update rate for 10 s
#                                           while you drive; records nothing
#   .\START_RECORDER_HF.ps1 --every-poll    store every poll, not one per game tick
#                                           (for inspecting raw data; not for the web UI)
#   .\START_RECORDER_HF.ps1 --poll-hz 200   poll slower. 500 Hz and above keep one CPU
#                                           core fully busy; 200 Hz uses far less
#   .\START_RECORDER_HF.ps1 --recover       convert a leftover .spool_hf safety copy
cd $PSScriptRoot
uv sync
uv run recorder_hf.py @args
