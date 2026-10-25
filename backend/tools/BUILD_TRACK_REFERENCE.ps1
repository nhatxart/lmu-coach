# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Build the track reference (corners, straights, names) used by the coach. No LLM involved.
#
#   .\BUILD_TRACK_REFERENCE.ps1                         every catalog track, from the local cache
#   .\BUILD_TRACK_REFERENCE.ps1 --track fuji --details  one track, list every corner
#   .\BUILD_TRACK_REFERENCE.ps1 --track newtrack --fetch   new track: download OSM/Wikipedia once
#
# Also builds <slug>.topography.json (height, banking, kerb paint for the 3D chase view) for
# every track that has recordings:
#   .\BUILD_TRACK_REFERENCE.ps1 --topography-only   only that file; corners/names untouched, no downloads
#   .\BUILD_TRACK_REFERENCE.ps1 --no-topography     skip it
#   add --sessions <dir> to read recordings from another folder (opened read-only)
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\build_track_reference.py @args
} finally {
    Pop-Location
}
