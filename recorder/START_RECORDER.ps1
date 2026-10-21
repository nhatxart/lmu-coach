# Run this from the recorder\ directory
# PowerShell doesn't support && - use semicolons or separate lines
#
# THE recorder (recorder.py). Start it once and leave the window open: it waits
# for LMU, records every session it finds, saves the file when you go back to
# the garage or leave the session, and starts the next one by itself.
# The web UI's Record tab (START_RECORDER_CONTROL.ps1) runs this same recorder;
# do not run both at once.
#
# By default it follows the game's own updates (about 100 a second): it sleeps
# between them, so it uses very little CPU, and every row is stamped with the
# game's clock. It runs at below-normal priority so the game wins any contention.
#
#   .\START_RECORDER.ps1                    record (follows the game's updates)
#   .\START_RECORDER.ps1 --probe 10         measure LMU's update rate, skipped updates
#                                           and the recorder's CPU use for 10 s while
#                                           you drive; records nothing. Run it again with
#                                           --poll-hz 1000 in the same session to compare.
#   .\START_RECORDER.ps1 --poll-hz 1000     poll at a FIXED rate instead (the old way).
#                                           500 Hz and above keep one CPU core fully busy
#   .\START_RECORDER.ps1 --every-poll       store every poll, not one per game tick
#                                           (fixed polling; for inspecting raw data)
#   .\START_RECORDER.ps1 --normal-priority  do not lower the recorder's priority
#   .\START_RECORDER.ps1 --recover          convert a leftover safety copy (.spool)
#   .\START_RECORDER.ps1 --selftest         time a save on this PC, no game needed
cd $PSScriptRoot
uv sync
uv run recorder.py @args
