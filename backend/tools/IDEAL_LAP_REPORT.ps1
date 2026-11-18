# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Read-only validation of the ideal lap (design section 9): one line per test lap with the
# V1-V15 metrics, then the pair's summary. Reductions are cached in %TEMP%\ideal_lap_cache.
#
#   .\IDEAL_LAP_REPORT.ps1 --fit "*Ferrari*10-27*" --sim "*Ferrari*10-30*"          cross-session
#   .\IDEAL_LAP_REPORT.ps1 --fit 11-09T04_29_26 --sim 11-09T04_29_26 --levels       odd/even
#   .\IDEAL_LAP_REPORT.ps1 --fit A,B --sim C --json C:\temp\pair.json
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\ideal_lap_report.py @args
} finally {
    Pop-Location
}
