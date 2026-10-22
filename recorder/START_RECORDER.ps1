# Run this from the recorder\ directory
# PowerShell doesn't support && - use semicolons or separate lines
#
# THE recorder (recorder.py). Start it once and leave the window open: it waits
# for LMU, records every session it finds, saves the file when you go back to
# the garage or leave the session, and starts the next one by itself.
# The web UI's Record tab (START_RECORDER_CONTROL.ps1) runs this same recorder;
# do not run both at once.
#
# Default: polls at a fixed 1000 Hz. That catches every game update (100 a
# second, one row each, stamped with the game's clock) but keeps one CPU core
# ~96% busy, which can stutter the game. It runs at below-normal priority.
#
#   .\START_RECORDER.ps1                    record (fixed 1000 Hz)
#   .\START_RECORDER.ps1 --probe 30         while driving: measure update rate, skipped
#                                           updates, CPU use and loop timing; records nothing
#   .\START_RECORDER.ps1 --probe 30 --tick-sync
#                                           the same for the experimental mode below, plus
#                                           how late its wake-ups are (that is the number
#                                           that explains lost updates)
#   .\START_RECORDER.ps1 --tick-sync        EXPERIMENTAL: follow the game's updates, ~1% CPU.
#                                           On its first real drive it caught only 56 of 100
#                                           updates a second. Do not record with it until
#                                           --probe shows ~100 Hz and ~0 skipped.
#   .\START_RECORDER.ps1 --poll-hz 200      poll at another fixed rate
#   .\START_RECORDER.ps1 --every-poll       store every poll, not one per game tick
#                                           (fixed polling; for inspecting raw data)
#   .\START_RECORDER.ps1 --normal-priority  do not lower the recorder's priority
#   .\START_RECORDER.ps1 --recover          convert a leftover safety copy (.spool)
#   .\START_RECORDER.ps1 --selftest         time a save on this PC, no game needed
cd $PSScriptRoot
uv sync
uv run recorder.py @args
