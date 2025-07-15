#!/usr/bin/env bash
# Fail the build if any lakefile root modules contain sorry or axiom statements.
# This ensures that modules marked as "working" in lakefile.lean are actually complete.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
cd "$ROOT"

echo "Verifying lakefile roots are complete (no sorries or axioms)..."
echo "============================================================="

# Define the lakefile roots based on current lakefile.lean
# These are the modules we claim are "working" and should be complete
LAKEFILE_ROOTS=(
    # Analysis library
    "Analysis/Trig/MonotoneCos.lean"
    
    # RSImport library  
    "RSImport/BasicDefinitions.lean"
    
    # YangMillsProof library roots
    "YangMillsProof/YangMillsProof.lean"
    "YangMillsProof/foundation_clean/Core/MetaPrinciple.lean"
    "YangMillsProof/foundation_clean/Core/EightFoundations.lean"
    "YangMillsProof/foundation_clean/Core.lean"
    "YangMillsProof/foundation_clean/MinimalFoundation.lean"
    "YangMillsProof/Stage0_RS_Foundation/ActivityCost.lean"
    "YangMillsProof/Stage1_GaugeEmbedding/VoxelLattice.lean"
    "YangMillsProof/Stage2_LatticeTheory/TransferMatrixGap.lean"
)

TOTAL_ERRORS=0
CHECKED_FILES=0

echo "Checking ${#LAKEFILE_ROOTS[@]} lakefile root modules..."
echo

for file in "${LAKEFILE_ROOTS[@]}"; do
    if [ -f "$file" ]; then
        echo "Checking: $file"
        CHECKED_FILES=$((CHECKED_FILES + 1))
        
        # Check for sorry statements (simplified pattern)
        SORRY_COUNT=0
        if grep -q "sorry" "$file" 2>/dev/null; then
            SORRY_COUNT=$(grep -c "sorry" "$file" 2>/dev/null | head -1 || echo 0)
        fi
        
        # Check for axiom declarations (simplified pattern)
        AXIOM_COUNT=0
        if grep -q "^[[:space:]]*axiom[[:space:]]" "$file" 2>/dev/null; then
            AXIOM_COUNT=$(grep -c "^[[:space:]]*axiom[[:space:]]" "$file" 2>/dev/null | head -1 || echo 0)
        fi
        
        if [ "$SORRY_COUNT" -ne 0 ]; then
            echo "  ❌ ERROR: Found $SORRY_COUNT sorry statements:"
            grep -n "sorry" "$file" 2>/dev/null | head -3 || true
            TOTAL_ERRORS=$((TOTAL_ERRORS + SORRY_COUNT))
        fi
        
        if [ "$AXIOM_COUNT" -ne 0 ]; then
            echo "  ❌ ERROR: Found $AXIOM_COUNT axiom declarations:"
            grep -n "^[[:space:]]*axiom[[:space:]]" "$file" 2>/dev/null | head -3 || true
            TOTAL_ERRORS=$((TOTAL_ERRORS + AXIOM_COUNT))
        fi
        
        if [ "$SORRY_COUNT" -eq 0 ] && [ "$AXIOM_COUNT" -eq 0 ]; then
            echo "  ✅ CLEAN: No sorries or axioms found"
        fi
        
    else
        echo "  ⚠️  WARNING: File not found: $file"
    fi
    echo
done

echo "============================================================="
echo "Summary:"
echo "- Files checked: $CHECKED_FILES"
echo "- Total errors found: $TOTAL_ERRORS"

if [ "$TOTAL_ERRORS" -ne 0 ]; then
    echo
    echo "❌ FAILURE: Lakefile roots contain $TOTAL_ERRORS sorries or axioms!"
    echo "All modules in lakefile roots should be complete (no sorry/axiom statements)."
    echo "Either:"
    echo "1. Complete the proofs in the listed files, or"
    echo "2. Remove incomplete modules from lakefile.lean roots"
    exit 1
else
    echo
    echo "✅ SUCCESS: All lakefile roots are complete!"
    echo "No sorries or axioms found in working modules."
fi 