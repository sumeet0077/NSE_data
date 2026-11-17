#!/bin/bash
# Autonomous frontend data build and auto-push script (Mac & OCI Cloud VPS)
set -e

# Auto-detect environment
if [ -d "/home/ubuntu/nifty-breadth" ]; then
    BREADTH_DIR="/home/ubuntu/nifty-breadth"
    INSIGHTS_DIR="/home/ubuntu/nse-industry-insights"
    LOG_FILE="/home/ubuntu/NSE_data/frontend_build.log"
    PYTHON_BIN="/home/ubuntu/NSE_data/.venv/bin/python"
    export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
else
    BREADTH_DIR="/Users/sumeetdas/Projects/nifty-breadth"
    INSIGHTS_DIR="/Users/sumeetdas/Projects/nse-industry-insights"
    LOG_FILE="/Users/sumeetdas/Antigravity_NSE_Data/frontend_build.log"
    PYTHON_BIN="/Users/sumeetdas/Antigravity_NSE_Data/.venv/bin/python"
    export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
fi

export GIT_TERMINAL_PROMPT=0

echo "========================================" >> "$LOG_FILE"
echo "Starting frontend data build at $(date) [PATH=$PATH]" >> "$LOG_FILE"

# 1. Reset repositories to stay clean and in-sync with GitHub origin
echo "[1/4] Syncing repositories with origin/main..." >> "$LOG_FILE"
cd "$BREADTH_DIR"
git fetch origin >> "$LOG_FILE" 2>&1
git reset --hard origin/main >> "$LOG_FILE" 2>&1
git clean -fd >> "$LOG_FILE" 2>&1

cd "$INSIGHTS_DIR"
git fetch origin >> "$LOG_FILE" 2>&1
git reset --hard origin/main >> "$LOG_FILE" 2>&1
git clean -fd >> "$LOG_FILE" 2>&1

# 2. Generate breadth CSVs
echo "[2/4] Generating Breadth CSVs..." >> "$LOG_FILE"
cd "$BREADTH_DIR"
nice -n 5 ionice -c 2 -n 4 "$PYTHON_BIN" fetch_breadth_data.py >> "$LOG_FILE" 2>&1 || "$PYTHON_BIN" fetch_breadth_data.py >> "$LOG_FILE" 2>&1

# Commit breadth CSVs locally (no push to avoid deploy key permission issues)
git config --local user.email "sumeetdas@users.noreply.github.com" 2>/dev/null || true
git config --local user.name "sumeet0077-vps" 2>/dev/null || true
git add *.csv >> "$LOG_FILE" 2>&1 || true
git commit -m "Auto-update breadth CSVs for $(date +'%Y-%m-%d')" >> "$LOG_FILE" 2>&1 || true

# 3. Export JSONs to the Next.js app
echo "[3/4] Exporting JSON payload to nse-industry-insights..." >> "$LOG_FILE"
cd "$INSIGHTS_DIR"
nice -n 5 ionice -c 2 -n 4 "$PYTHON_BIN" scripts/export_json.py --output data --source "$BREADTH_DIR" >> "$LOG_FILE" 2>&1 || "$PYTHON_BIN" scripts/export_json.py --output data --source "$BREADTH_DIR" >> "$LOG_FILE" 2>&1

# 4. Commit and push to nse-industry-insights
echo "[4/4] Committing and Pushing to GitHub..." >> "$LOG_FILE"
git config --local user.email "sumeetdas@users.noreply.github.com" 2>/dev/null || true
git config --local user.name "sumeet0077-vps" 2>/dev/null || true
git config http.postBuffer 524288000
git config http.version HTTP/1.1
git config pack.window 1
git config pack.depth 1
git config pack.threads 1

git add data/
git commit -m "Auto-update market breadth data for $(date +'%Y-%m-%d') [Autonomous Pipeline]" >> "$LOG_FILE" 2>&1 || true

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
