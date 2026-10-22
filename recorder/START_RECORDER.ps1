# Run this from the recorder\ directory
# PowerShell doesn't support && - use semicolons or separate lines
#
# THE recorder (recorder.py). Start it once and leave the window open: it waits
# for LMU, records every session it finds, saves the file when you go back to
# the garage or leave the session, and starts the next one by itself.
# The web UI's Record tab (START_RECORDER_CONTROL.ps1) runs this same recorder;
# do not run both at once.
#
# Default: follows the game's own updates (about 100 a second, one row each,
# stamped with the game's clock). It sleeps between them, so it uses about 1% of
# a core. It stays at NORMAL priority on purpose: LMU runs at High priority and a
# sleeping thread below it wakes late and loses updates.
# If a saved recording lost more than 2% of the game's updates it says WARNING.
#
#   .\START_RECORDER.ps1                    record
#   .\START_RECORDER.ps1 --probe 30         while driving: measure update rate, skipped
#                                           updates, CPU use, loop time and wake-up lateness;
#                                           records nothing
#   .\START_RECORDER.ps1 --probe 30 --poll-hz 1000
#                                           the same for fixed 1000 Hz polling, to compare
#   .\START_RECORDER.ps1 --poll-hz 1000     poll at a FIXED rate instead (96% of a core;
#                                           500 Hz and above keep one core fully busy)
#   .\START_RECORDER.ps1 --every-poll       store every poll, not one per game tick
#                                           (fixed polling; for inspecting raw data)
#   .\START_RECORDER.ps1 --low-priority     NOT recommended: below-normal priority lost
#                                           9% to 44% of the updates in testing
#   .\START_RECORDER.ps1 --recover          convert a leftover safety copy (.spool)
#   .\START_RECORDER.ps1 --selftest         time a save on this PC, no game needed
cd $PSScriptRoot
uv sync
uv run recorder.py @args
