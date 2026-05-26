#!/bin/bash
# Wrapper for NSE Daily Updater Service (Relocated)

# Set working directory to new home location
cd "/Users/sumeetdas/Antigravity_NSE_Data" || exit 1

# Log start
echo "Starting NSE Daily Updater Service (Relocated) at $(date)" >> daily_updater_wrapper.log

# Execute the python script with the virtualenv python
exec "/Users/sumeetdas/Antigravity_NSE_Data/.venv/bin/python" "nse_daily_update_service.py" \
    --service \
    --timezone "Asia/Kolkata" \
    --slots "16:30,17,18,19,20" \
    "$@" \
    >> daily_updater_stdout.log 2>> daily_updater_stderr.log
