# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Build the track reference (corners, straights, names) used by the coach. No LLM involved.
#
#   .\BUILD_TRACK_REFERENCE.ps1                         every catalog track, from the local cache
#   .\BUILD_TRACK_REFERENCE.ps1 --track fuji --details  one track, list every corner
#   .\BUILD_TRACK_REFERENCE.ps1 --track newtrack --fetch   new track: download OSM/Wikipedia once
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\build_track_reference.py @args
} finally {
    Pop-Location
}
