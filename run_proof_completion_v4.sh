#!/bin/bash
# Run the v4 AI proof completion system

echo "Yang-Mills Lean Proof Completion System V4"
echo "=========================================="
echo "Using Claude 4 Sonnet with optimistic proof application"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

# Create backup
echo "Creating backup of Lean files..."
backup_dir="backups/YangMillsProof_v4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp -r YangMillsProof/* "$backup_dir/"

# Run the v4 system
echo ""
echo "Starting AI proof completion..."
echo "Features:"
echo "  ✓ Claude 4 Sonnet exclusively"
echo "  ✓ Basic syntax validation"
echo "  ✓ Optimistic proof application"
echo "  ✓ Up to 3 retry attempts per proof"
echo ""

python3 lean_ai_agents_v4.py

echo ""
echo "Proof completion finished!"
echo "Backup saved in: $backup_dir"
echo ""
echo "Check the summary above for:"
echo "  - Number of successful proofs"
echo "  - Final build status"
echo "  - Any proofs that need manual attention" 