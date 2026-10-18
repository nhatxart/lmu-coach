# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Read-only accuracy check of one car's tyre model; prints a scorecard row for DevNotes.md.
#
#   .\VALIDATE_TIRE_MODEL.ps1 WRT
#   .\VALIDATE_TIRE_MODEL.ps1 Cadillac --boot 100     fewer bootstrap resamples = faster
$repo = Resolve-Path "$PSScriptRoot\..\.."
$fileArgs = @($args | ForEach-Object { if ($_ -is [string] -and (Test-Path -LiteralPath $_ -PathType Leaf)) { (Resolve-Path -LiteralPath $_).Path } else { $_ } })
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\validate_tire_model.py @fileArgs
} finally {
    Pop-Location
}
