# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Does a recording have usable tyre-force channels? (read-only)
#
#   .\CHECK_TIRE_FORCES.ps1 "..\..\data\sessions\<file>.duckdb"
$repo = Resolve-Path "$PSScriptRoot\..\.."
$fileArgs = @($args | ForEach-Object { if ($_ -is [string] -and (Test-Path -LiteralPath $_ -PathType Leaf)) { (Resolve-Path -LiteralPath $_).Path } else { $_ } })
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\check_tire_forces.py @fileArgs
} finally {
    Pop-Location
}
