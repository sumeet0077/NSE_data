#!/bin/bash
# ──────────────────────────────────────────────────────
#  Camarilla Scanner — One-Click Daily Runner
#  Usage: ./run_camarilla.sh
#  Or double-click from Finder after running:
#    chmod +x run_camarilla.sh
# ──────────────────────────────────────────────────────

SCRIPT_DIR="/Users/sumeetdas/Antigravity_NSE_Data"
PYTHON=$(which python3)

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║    Camarilla Monthly Compression Scanner      ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║  $(date '+%A, %d %B %Y  %H:%M IST')          "
echo "╟───────────────────────────────────────────────╢"
echo ""

cd "$SCRIPT_DIR" || { echo "ERROR: Could not cd to $SCRIPT_DIR"; exit 1; }

$PYTHON export_strict_camarilla.py
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "✅  Scanner completed successfully!"
else
    echo "❌  Scanner failed with exit code $STATUS. Check logs above."
fi
echo ""
