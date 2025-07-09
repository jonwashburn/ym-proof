#!/usr/bin/env bash
# Fail the build if any Lean source outside .lake/ contains actual `sorry` statements (not in comments).

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")

# Count actual sorry statements (not in comments)
# This regex looks for 'sorry' that's not preceded by -- (line comment) 
CODE_SORRY_COUNT=$(grep -R --exclude-dir=.lake --include='*.lean' -E "^[[:space:]]*sorry|[^-]sorry" "$ROOT" 2>/dev/null | wc -l | tr -d ' ' || echo 0)

# Count all occurrences including in comments
TOTAL_COUNT=$(grep -R --exclude-dir=.lake --include='*.lean' -n "sorry" "$ROOT" 2>/dev/null | wc -l | tr -d ' ' || echo 0)

if [ "$CODE_SORRY_COUNT" -ne 0 ]; then
  echo "ERROR: project contains $CODE_SORRY_COUNT actual 'sorry' statements in code."
  grep -R --exclude-dir=.lake --include='*.lean' -E "^[[:space:]]*sorry|[^-]sorry" "$ROOT" 2>/dev/null | head -10 || true
  exit 1
else
  if [ "$TOTAL_COUNT" -ne 0 ]; then
    echo "Note: Found $TOTAL_COUNT occurrences of the word 'sorry' (all in comments/documentation)."
    echo "Examples:"
    grep -R --exclude-dir=.lake --include='*.lean' -n "sorry" "$ROOT" 2>/dev/null | head -3 || true
  fi
  echo "verify_no_sorries: no actual sorry statements found – good!"
fi 