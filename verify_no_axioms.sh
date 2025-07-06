#!/bin/bash

echo "Verifying Yang-Mills proof integrity..."
echo "========================================"

# Count axiom declarations (excluding .lake directory)
axiom_count=$(find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -l "^axiom " {} \; | wc -l)
echo "Files containing axiom declarations: $axiom_count"

if [ $axiom_count -gt 0 ]; then
    echo "WARNING: Found axiom declarations in:"
    find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -l "^axiom " {} \;
fi

# Count sorry statements (excluding .lake directory)
sorry_count=$(find YangMillsProof -name "*.lean" -type f -not -path "*/\.lake/*" -exec grep -c "^[[:space:]]*sorry" {} \; | awk '{sum += $1} END {print sum}')
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