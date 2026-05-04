#!/usr/bin/env bash
# scripts/setup_cron.sh
# ─────────────────────────────────────────────────────────────────────────────
# Registers cron jobs for the Cork Weather Predictor pipeline.
# Works on: Linux servers, WSL, macOS, Docker containers.
#
# Installs two jobs:
#   - Hourly fetch   : runs fetch_latest.py every hour at :05
#   - Nightly retrain: runs retrain.py every day at 03:00
#
# Usage:
#   bash scripts/setup_cron.sh
#   bash scripts/setup_cron.sh --api-url http://your-server:8000
#
# To remove the cron jobs:
#   crontab -e   (manually delete the CorkWeather lines)
#   — or —
#   crontab -l | grep -v CorkWeather | crontab -
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
API_URL="${1:-http://localhost:8000}"
LOG_DIR="$PROJECT_ROOT/logs"

# ── Validate ──────────────────────────────────────────────────────────────────

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: '$PYTHON' not found. Set PYTHON env var or install python3."
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Project root : $PROJECT_ROOT"
echo "Python       : $($PYTHON --version)"
echo "API URL      : $API_URL"
echo "Log dir      : $LOG_DIR"
echo ""

# ── Build cron lines ──────────────────────────────────────────────────────────

# Fetch every hour at :05 (slight offset avoids clock-on-the-hour traffic)
FETCH_CRON="5 * * * * cd $PROJECT_ROOT && $PYTHON scripts/fetch_latest.py --api-url $API_URL >> $LOG_DIR/fetch.log 2>&1  # CorkWeather-FetchHourly"

# Full retrain nightly at 03:00
RETRAIN_CRON="0 3 * * * cd $PROJECT_ROOT && $PYTHON scripts/retrain.py >> $LOG_DIR/retrain.log 2>&1  # CorkWeather-RetrainNightly"

# ── Install (idempotent: remove existing CorkWeather entries first) ───────────

echo "Installing cron jobs..."

# Get current crontab (empty string if none exists)
CURRENT_CRON="$(crontab -l 2>/dev/null || true)"

# Remove any previous CorkWeather entries to avoid duplicates
CLEAN_CRON="$(echo "$CURRENT_CRON" | grep -v "CorkWeather" || true)"

# Add new entries
NEW_CRON="$(printf '%s\n%s\n%s\n' "$CLEAN_CRON" "$FETCH_CRON" "$RETRAIN_CRON")"

echo "$NEW_CRON" | crontab -

# ── Verify ────────────────────────────────────────────────────────────────────

echo ""
echo "======================================================"
echo "  Cron setup complete."
echo "======================================================"
echo ""
echo "Registered jobs:"
crontab -l | grep CorkWeather
echo ""
echo "Logs will be written to:"
echo "  $LOG_DIR/fetch.log"
echo "  $LOG_DIR/retrain.log"
echo ""
echo "To view live logs:"
echo "  tail -f $LOG_DIR/fetch.log"
echo "  tail -f $LOG_DIR/retrain.log"
echo ""
echo "To remove all CorkWeather cron jobs:"
echo "  crontab -l | grep -v CorkWeather | crontab -"
