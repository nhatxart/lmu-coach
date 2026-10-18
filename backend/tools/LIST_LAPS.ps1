# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# List laps per session file as the backend sees them (read-only).
#
#   .\LIST_LAPS.ps1                   every file
#   .\LIST_LAPS.ps1 "*2026-10-*"      glob on the file name
$repo = Resolve-Path "$PSScriptRoot\..\.."
$fileArgs = @($args | ForEach-Object { if ($_ -is [string] -and (Test-Path -LiteralPath $_ -PathType Leaf)) { (Resolve-Path -LiteralPath $_).Path } else { $_ } })
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\list_laps.py @fileArgs
} finally {
    Pop-Location
}
