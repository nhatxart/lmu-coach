# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Pilot: can Susp Force stand in for Tire Load in the grip proxy? (read-only)
#
#   .\COMPARE_LOAD_SOURCES.ps1 "*Genesis*" "*Ferrari AF Corse*"
$repo = Resolve-Path "$PSScriptRoot\..\.."
$fileArgs = @($args | ForEach-Object { if ($_ -is [string] -and (Test-Path -LiteralPath $_ -PathType Leaf)) { (Resolve-Path -LiteralPath $_).Path } else { $_ } })
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\compare_load_sources.py @fileArgs
} finally {
    Pop-Location
}
