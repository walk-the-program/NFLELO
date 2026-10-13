#!/usr/bin/env bash
# Force a full update now: runs the same GitHub Action as the scheduled runs (Elo rebuild, model
# predictions logged before kickoff, playoff simulation, site data), waits for it, then pulls the
# result. The hosted site reads its data from GitHub, so it shows the new numbers on next page load.
# Safe to run any time: the ledger refuses rows at or after a game's kickoff.
# Needs the GitHub CLI signed in (gh auth status).
set -euo pipefail
cd "$(dirname "$0")/.."

gh workflow run weekly.yml --ref main
echo "Started. Waiting for GitHub to pick it up..."
for _ in $(seq 1 20); do
  sleep 3
  id=$(gh run list --workflow weekly.yml --event workflow_dispatch --limit 1 --json databaseId,createdAt \
       --jq '.[0] | select((now - (.createdAt | fromdateiso8601)) < 120) | .databaseId')
  [ -n "${id:-}" ] && break
done
[ -n "${id:-}" ] || { echo "Run not found yet; check: gh run list --workflow weekly.yml"; exit 1; }

gh run watch "$id" --exit-status --interval 15
git pull --rebase --autostash --quiet
echo "Done. Site data updated on main ($(git log -1 --format=%h))."
