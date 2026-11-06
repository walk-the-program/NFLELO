#!/usr/bin/env bash
# Exact-time trigger for the weekly update, run by the com.walker.nfelo-update LaunchAgent.
# Dispatches the GitHub Action and logs the outcome. The Action's own cron entries are backups.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
log="$HOME/Library/Logs/nfelo-update.log"
mkdir -p "$(dirname "$log")"
if out=$(gh workflow run weekly.yml --ref main -R walk-the-program/NFLELO 2>&1); then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') dispatch OK" >> "$log"
else
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') dispatch FAILED: $out" >> "$log"
  exit 1
fi
