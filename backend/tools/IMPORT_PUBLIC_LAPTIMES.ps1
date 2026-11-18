# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Imports Ohne Speed's public LMU lap-time sheet into data\track_reference\public_laptimes.json
# (design section 14 R9). This is the only tool that goes online, and only when you run it.
#
#   .\IMPORT_PUBLIC_LAPTIMES.ps1 --dry-run      fetch and print, write nothing
#   .\IMPORT_PUBLIC_LAPTIMES.ps1                write the JSON (restart the backend to pick it up)
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\import_public_laptimes.py @args
} finally {
    Pop-Location
}
