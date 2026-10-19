# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Class-collapse experiment for tyre model v3 (read-only): how much should a car lean on
# the pooled curve of the other cars of its class?
#
#   .\CLASS_COLLAPSE.ps1 --sessions-dir C:\path\to\data\sessions --class LMGT3
$repo = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\class_collapse.py @args
} finally {
    Pop-Location
}
