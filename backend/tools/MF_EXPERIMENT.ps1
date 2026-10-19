# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Hypercar factor experiment for tyre model v3 (read-only):
#
#   .\MF_EXPERIMENT.ps1 --sessions-dir C:\path\to\data\sessions --class Hypercar [--files "*Valkyrie*"]
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\mf_experiment.py @args
} finally {
    Pop-Location
}
