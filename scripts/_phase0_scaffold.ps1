# Phase 0.1 — Archive old code and scaffold new structure

Set-Location $PSScriptRoot\..

# ── Archive old model scripts ─────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path 'archive\old_model' | Out-Null

Move-Item -Path 'old_model\arima_modelling.py' -Destination 'archive\old_model\arima_modelling.py' -Force
Move-Item -Path 'old_model\cleaning.py'        -Destination 'archive\old_model\cleaning.py' -Force
Move-Item -Path 'old_model\eda.py'             -Destination 'archive\old_model\eda.py' -Force
Move-Item -Path 'old_model\reg_modelling.py'   -Destination 'archive\old_model\reg_modelling.py' -Force

Move-Item -Path 'better_model\plan.txt' -Destination 'archive\better_model_plan.txt' -Force

Remove-Item -Path 'old_model'    -Recurse -Force
Remove-Item -Path 'better_model' -Recurse -Force

# ── Delete deprecated cleaned CSVs ───────────────────────────────────────────
Remove-Item -Path 'data\cleaned\weather_cleaned_sorted.csv' -Force -ErrorAction SilentlyContinue
Remove-Item -Path 'data\cleaned\weather_data_2025.csv'      -Force -ErrorAction SilentlyContinue
Remove-Item -Path 'data\cleaned\weather_data_cleaned.csv'   -Force -ErrorAction SilentlyContinue

# ── Create new production directory structure ─────────────────────────────────
$dirs = @(
    'data\processed',
    'src\data',
    'src\models',
    'api',
    'scripts',
    'models',
    'results\plots',
    'notebooks'
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# ── Create all __init__.py files ─────────────────────────────────────────────
$inits = @(
    'src\__init__.py',
    'src\data\__init__.py',
    'src\models\__init__.py',
    'api\__init__.py'
)
foreach ($f in $inits) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType File -Force -Path $f | Out-Null
    }
}

# ── .gitkeep so git tracks the empty models/ dir ─────────────────────────────
New-Item -ItemType File -Force -Path 'models\.gitkeep' | Out-Null

Write-Host 'Phase 0.1 complete — directory scaffold created.'
