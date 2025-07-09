#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory to ensure paths work
cd "$SCRIPT_DIR" || exit 1

echo "Verifying Yang-Mills proof integrity..."
echo "========================================"

# Check if YangMillsProof directory exists
if [ ! -d "YangMillsProof" ]; then
    echo "ERROR: YangMillsProof directory not found in $SCRIPT_DIR"
    echo "Make sure you're running this from the repository root"
    exit 1
fi

# Count axiom declarations (excluding .lake directory)
axiom_count=$(find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -l "^axiom " {} \; 2>/dev/null | wc -l)
echo "Files containing axiom declarations: $axiom_count"

if [ $axiom_count -gt 0 ]; then
    echo "WARNING: Found axiom declarations in:"
    find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -l "^axiom " {} \;
fi

# Count sorry statements (excluding .lake directory)
sorry_count=$(find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -c "^[[:space:]]*sorry" {} \; 2>/dev/null | awk '{sum += $1} END {print sum}')
# Handle empty result
if [ -z "$sorry_count" ]; then
    sorry_count=0
fi
echo "Total sorry statements: $sorry_count"

# Show files with sorries (excluding .lake directory)
if [ $sorry_count -gt 0 ]; then
    echo ""
    echo "Files with sorry statements:"
    find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -l "^[[:space:]]*sorry" {} \; | while read file; do
        count=$(grep -c "^[[:space:]]*sorry" "$file")
        echo "  $file: $count sorries"
    done
fi

echo ""
if [ $axiom_count -eq 0 ] && [ $sorry_count -eq 0 ]; then
    echo "✓ SUCCESS: Proof is complete - no axioms or sorries!"
else
    echo "✗ FAILURE: Proof incomplete"
    exit 1
fi 