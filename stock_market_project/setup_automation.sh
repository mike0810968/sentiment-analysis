#!/bin/bash

# Automated Daily Data Update Script
# This script sets up cron jobs to update stock prices and news daily

PROJECT_DIR="/Users/namanvyas/Desktop/sentiment analysis/stock_market_project"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"
UPDATE_SCRIPT="$PROJECT_DIR/src/data_updater.py"

# Function to setup cron job
setup_cron_job() {
    local schedule="$1"
    local job_name="$2"
    local command="$3"

    # Remove existing job if it exists
    crontab -l | grep -v "$job_name" | crontab -

    # Add new job
    (crontab -l ; echo "$schedule $command # $job_name") | crontab -

    echo "✅ Added cron job: $job_name"
}

echo "🚀 Setting up automated daily data updates..."

# Create log directory
mkdir -p "$PROJECT_DIR/logs"

# Setup daily update at 6:00 AM (market open time)
# This runs Monday through Friday
DAILY_SCHEDULE="0 6 * * 1-5"

UPDATE_COMMAND="cd '$PROJECT_DIR' && source '$VENV_PATH' && python '$UPDATE_SCRIPT' >> '$PROJECT_DIR/logs/daily_update.log' 2>&1"

setup_cron_job "$DAILY_SCHEDULE" "stock_market_daily_update" "$UPDATE_COMMAND"

# Alternative: Setup update every 4 hours during market hours (9 AM - 3 PM)
# MARKET_HOURS_SCHEDULE="0 9,11,13,15 * * 1-5"
# setup_cron_job "$MARKET_HOURS_SCHEDULE" "stock_market_market_hours" "$UPDATE_COMMAND"

echo ""
echo "📋 Current cron jobs:"
crontab -l | grep stock_market

echo ""
echo "� To monitor updates:"
echo "  tail -f $PROJECT_DIR/logs/daily_update.log"
echo ""
echo "🔧 To modify schedule:"
echo "  crontab -e"
echo ""
echo "🛑 To stop updates:"
echo "  crontab -l | grep -v stock_market | crontab -"
echo ""
echo "✅ Setup complete! Your dashboard will now update daily at 6:00 AM."