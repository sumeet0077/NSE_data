#!/bin/bash -x
# Natively triggers the React/Next.js JSON export and auto-pushes to GitHub for Cloudflare deployment.
set -e

# Ensure common paths are available even in background service (launchd)
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"

LOG_FILE="/Users/sumeetdas/Antigravity_NSE_Data/frontend_build.log"
echo "========================================" >> "$LOG_FILE"
echo "Starting frontend data build at $(date) [PATH=$PATH]" >> "$LOG_FILE"

# 1. Generate breadth CSVs
echo "[1/3] Generating Breadth CSVs..." >> "$LOG_FILE"
cd "/Users/sumeetdas/Projects/nifty-breadth"
/Users/sumeetdas/Antigravity_NSE_Data/.venv/bin/python fetch_breadth_data.py >> "$LOG_FILE" 2>&1

# 2. Export JSONs to the Next.js app
echo "[2/3] Exporting JSON payload to nse-industry-insights..." >> "$LOG_FILE"
cd "/Users/sumeetdas/Projects/nse-industry-insights"
/Users/sumeetdas/Antigravity_NSE_Data/.venv/bin/python scripts/export_json.py --output data --source "../nifty-breadth" >> "$LOG_FILE" 2>&1

# 3. Commit and push
echo "[3/3] Committing and Pushing to GitHub..." >> "$LOG_FILE"
# Stage both data/ (JSON outputs) and any modified source files (lib/, scripts/)
# so Vercel always builds with the latest frontend code, not just data.
git add data/ lib/ scripts/
# Commit will fail if no changes, so we use || true
git commit -m "Auto-update market breadth data for $(date +'%Y-%m-%d')" >> "$LOG_FILE" 2>&1 || true
git push origin main >> "$LOG_FILE" 2>&1 || true

echo "Frontend data build completed at $(date)" >> "$LOG_FILE"
