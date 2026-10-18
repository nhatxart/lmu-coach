# Run from anywhere; uses the backend's pinned requirements (no venv to manage).
# PowerShell doesn't support && - use semicolons or separate lines
#
# Bulk-fold recorded sessions into the per-car grip profiles (no LLM).
# Dry run unless --save; --save backs the profiles up first.
#
#   .\INGEST_SESSIONS.ps1 --month 2026-10            preview October
#   .\INGEST_SESSIONS.ps1 --month 2026-10 --save     write it
#   .\INGEST_SESSIONS.ps1 --since 2026-10-14 --until 2026-10-18 --save
#   .\INGEST_SESSIONS.ps1 --files "Fuji*" --tyre-only --save
#
# Stop the backend first (docker compose down): it reads the same profile files.
$repo = Resolve-Path "$PSScriptRoot\..\.."
$fileArgs = @($args | ForEach-Object { if ($_ -is [string] -and (Test-Path -LiteralPath $_ -PathType Leaf)) { (Resolve-Path -LiteralPath $_).Path } else { $_ } })
Push-Location $repo
try {
    uv run --no-project --with-requirements backend\requirements.txt python -I backend\tools\ingest_sessions.py @fileArgs
} finally {
    Pop-Location
}
