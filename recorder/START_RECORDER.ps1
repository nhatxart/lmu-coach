# Run this from the recorder\ directory
# PowerShell doesn't support && - use semicolons or separate lines
#
# THE recorder (recorder.py). Start it once and leave the window open: it waits
# for LMU, records every session it finds, saves the file when you go back to
# the garage or leave the session, and starts the next one by itself.
# The web UI's Record tab (START_RECORDER_CONTROL.ps1) runs this same recorder;
# do not run both at once.
#
#   .\START_RECORDER.ps1                    record, polling at 1000 Hz
#   .\START_RECORDER.ps1 --probe 10         measure LMU's real update rate for 10 s
#                                           while you drive; records nothing
#   .\START_RECORDER.ps1 --every-poll       store every poll, not one per game tick
#                                           (for inspecting raw data; not for the web UI)
#   .\START_RECORDER.ps1 --poll-hz 200      poll slower. 500 Hz and above keep one CPU
#                                           core fully busy; 200 Hz uses far less
#   .\START_RECORDER.ps1 --recover          convert a leftover safety copy (.spool)
#   .\START_RECORDER.ps1 --selftest         time a save on this PC, no game needed
cd $PSScriptRoot
uv sync
uv run recorder.py @args
