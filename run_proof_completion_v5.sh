#!/bin/bash
# Run the v5 AI proof completion system

echo "Yang-Mills Lean Proof Completion System V5"
echo "=========================================="
echo "Improved application logic + real compilation testing"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

# Create backup
echo "Creating backup of current state..."
backup_dir="backups/YangMillsProof_v5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp -r YangMillsProof/* "$backup_dir/"

echo ""
echo "Starting v5 proof completion..."
echo "Features:"
echo "  ✓ Claude 4 Sonnet"
echo "  ✓ Improved proof application logic"
echo "  ✓ Real compilation testing"
echo "  ✓ Safer string replacement"
echo ""

python3 lean_ai_agents_v5.py

echo ""
echo "V5 completion finished!"
echo "Backup saved in: $backup_dir" 