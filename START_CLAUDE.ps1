# Opens Claude Code in the project root. Use -Continue to resume the last session.
# PowerShell doesn't support && - use semicolons or separate lines
param([switch]$Continue)
cd $PSScriptRoot
if ($Continue) { claude --continue } else { claude }
