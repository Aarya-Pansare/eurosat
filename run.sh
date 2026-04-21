#!/usr/bin/env bash
# ──────────────────────────────────────────────
#  TERRAVIEW – Land Cover Classification Server
# ──────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-$SCRIPT_DIR/model.pth}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  echo "Usage: ./run.sh [path/to/model.pth]"
  exit 1
fi

export MODEL_PATH="$MODEL_PATH"

echo ""
echo "  ████████╗███████╗██████╗ ██████╗  █████╗ ██╗   ██╗██╗███████╗██╗    ██╗"
echo "     ██╔══╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██║   ██║██║██╔════╝██║    ██║"
echo "     ██║   █████╗  ██████╔╝██████╔╝███████║██║   ██║██║█████╗  ██║ █╗ ██║"
echo "     ██║   ██╔══╝  ██╔══██╗██╔══██╗██╔══██║╚██╗ ██╔╝██║██╔══╝  ██║███╗██║"
echo "     ██║   ███████╗██║  ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████╗╚███╔███╔╝"
echo "     ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝"
echo ""
echo "  Sentinel-2 EuroSAT Land Cover Classification"
echo "  Model: $MODEL_PATH"
echo "  URL:   http://localhost:5000"
echo ""

cd "$SCRIPT_DIR"
python3 app.py
