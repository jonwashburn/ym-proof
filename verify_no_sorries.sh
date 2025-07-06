#!/usr/bin/env bash
# Fail the build if any Lean source outside .lake/ contains the token `sorry`.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
COUNT=$(grep -R --exclude-dir=.lake --include='*.lean' -n "sorry" "$ROOT" | wc -l)

if [ "$COUNT" -ne 0 ]; then
  echo "ERROR: project contains $COUNT occurrences of 'sorry'."
  grep -R --exclude-dir=.lake --include='*.lean' -n "sorry" "$ROOT" || true
  exit 1
else
  echo "verify_no_sorries: no sorry found – good!"
fi 