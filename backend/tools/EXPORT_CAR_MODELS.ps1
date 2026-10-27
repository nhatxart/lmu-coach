# Run from anywhere; needs Node.js (any recent version), nothing else.
# PowerShell doesn't support && - use semicolons or separate lines
#
# Write the chase view's car models (backend/app/static/car_models.js, sized
# by car_models.json) as STL previews for a 3D-printer slicer:
# docs/car_models/<class>_1to1_mm.stl and <class>_1to43_mm.stl.
#
#   .\EXPORT_CAR_MODELS.ps1                 into docs\car_models
#   .\EXPORT_CAR_MODELS.ps1 C:\some\folder  somewhere else
$repo = Resolve-Path "$PSScriptRoot\..\.."
$out = if ($args.Count -ge 1) { $args[0] } else { Join-Path $repo "docs\car_models" }
node (Join-Path $repo "backend\tools\export_car_models.mjs") $out
