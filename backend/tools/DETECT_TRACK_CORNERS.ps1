# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Detect each track's corners from your recordings and compare with the official count.
#
#   .\DETECT_TRACK_CORNERS.ps1                    table for every track (read-only)
#   .\DETECT_TRACK_CORNERS.ps1 --track Fuji --details
#   .\DETECT_TRACK_CORNERS.ps1 --save             also writes data\track_corners\<track>.json
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\detect_track_corners.py @args
} finally {
    Pop-Location
}
