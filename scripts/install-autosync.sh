#!/usr/bin/env bash
# Install the automation on macOS:
#   1. turn on the pre-commit hook (auto-gzip on every commit)
#   2. load the launchd watcher (auto push when data folders change)
# Run once from your macOS Terminal:  bash scripts/install-autosync.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "• enabling pre-commit hook"
git config core.hooksPath .githooks

PLIST="$HOME/Library/LaunchAgents/com.starlinkd2c.datasync.plist"
echo "• installing launchd watcher -> $PLIST"
mkdir -p "$HOME/Library/LaunchAgents"
cp scripts/com.starlinkd2c.datasync.plist "$PLIST"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✓ done. Drop a CSV in 'QP CSV/' or 'PRISM CSV/' and it will push itself."
echo "  log: /tmp/starlinkd2c-sync.log"
echo "  to stop: launchctl unload \"$PLIST\" && rm \"$PLIST\""
