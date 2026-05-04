# scripts/setup_scheduler.ps1
# ─────────────────────────────────────────────────────────────
# Creates two Windows Task Scheduler tasks for the Cork Weather
# Predictor pipeline:
#
#   CorkWeather-FetchHourly  — runs fetch_latest.py every hour
#   CorkWeather-RetrainNightly — runs retrain.py at 03:00 daily
#
# Run this script ONCE as Administrator to register the tasks.
# After that they run automatically in the background.
#
# Usage:
#   Right-click PowerShell > "Run as Administrator"
#   cd C:\Users\clanc\Desktop\College\Weather-Predictor
#   .\scripts\setup_scheduler.ps1
# ─────────────────────────────────────────────────────────────

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$PythonExe   = (Get-Command python).Source
$LogDir      = Join-Path $ProjectRoot "logs"

# Create logs directory if it doesn't exist
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-Host "Created logs/ directory at $LogDir"
}

# ── Task 1: Hourly fetch ──────────────────────────────────────────────────────

$FetchAction = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "scripts\fetch_latest.py --api-url http://localhost:8000" `
    -WorkingDirectory $ProjectRoot

$FetchTrigger = New-ScheduledTaskTrigger `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -Once `
    -At (Get-Date).Date

$FetchSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName   "CorkWeather-FetchHourly" `
    -Action     $FetchAction `
    -Trigger    $FetchTrigger `
    -Settings   $FetchSettings `
    -RunLevel   Highest `
    -Force

Write-Host "Registered: CorkWeather-FetchHourly (runs every hour)"

# ── Task 2: Nightly retrain ───────────────────────────────────────────────────

$RetrainAction = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "scripts\retrain.py" `
    -WorkingDirectory $ProjectRoot

$RetrainTrigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "03:00"

$RetrainSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName   "CorkWeather-RetrainNightly" `
    -Action     $RetrainAction `
    -Trigger    $RetrainTrigger `
    -Settings   $RetrainSettings `
    -RunLevel   Highest `
    -Force

Write-Host "Registered: CorkWeather-RetrainNightly (runs at 03:00 daily)"

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======================================================"
Write-Host "  Scheduler setup complete."
Write-Host "  Fetch  : every hour   -> fetch_latest.py + API reload"
Write-Host "  Retrain: 03:00 daily  -> retrain.py (full pipeline)"
Write-Host "  Logs   : $LogDir"
Write-Host "======================================================"
Write-Host ""
Write-Host "To verify tasks were created:"
Write-Host "  Get-ScheduledTask -TaskName 'CorkWeather-*'"
Write-Host ""
Write-Host "To remove tasks:"
Write-Host "  Unregister-ScheduledTask -TaskName 'CorkWeather-FetchHourly' -Confirm:`$false"
Write-Host "  Unregister-ScheduledTask -TaskName 'CorkWeather-RetrainNightly' -Confirm:`$false"
