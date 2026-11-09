#!/bin/bash -x
# Natively triggers the React/Next.js JSON export and auto-pushes to GitHub for Cloudflare deployment.
set -e

# Ensure common paths are available even in background service (launchd)
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
export GIT_TERMINAL_PROMPT=0

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
git fetch origin >> "$LOG_FILE" 2>&1
git reset --hard origin/main >> "$LOG_FILE" 2>&1
/Users/sumeetdas/Antigravity_NSE_Data/.venv/bin/python scripts/export_json.py --output data --source "../nifty-breadth" >> "$LOG_FILE" 2>&1

# 3. Commit and push
echo "[3/3] Committing and Pushing to GitHub..." >> "$LOG_FILE"
# Configure git buffer to prevent RPC timeouts on large JSON datasets
git config http.postBuffer 524288000
git config http.version HTTP/1.1
# Stage both data/ (JSON outputs) and any modified source files (lib/, scripts/)
git add data/ lib/ scripts/
# Commit will fail if no changes, so we use || true
git commit -m "Auto-update market breadth data for $(date +'%Y-%m-%d')" >> "$LOG_FILE" 2>&1 || true

PUSH_SUCCESS=0
for attempt in 1 2 3 4 5; do
    echo "Push attempt $attempt of 5..." >> "$LOG_FILE"
    if git push origin main >> "$LOG_FILE" 2>&1; then
        echo "Push succeeded on attempt $attempt" >> "$LOG_FILE"
        PUSH_SUCCESS=1
        break
    fi
    echo "Push attempt $attempt failed, retrying in 15s..." >> "$LOG_FILE"
    sleep 15
done

if [ $PUSH_SUCCESS -ne 1 ]; then
    echo "ERROR: git push origin main failed after 5 attempts" >> "$LOG_FILE"
    exit 1
fi

echo "Frontend data build completed at $(date)" >> "$LOG_FILE"
