#!/bin/bash

# Yang-Mills Lean Proof Completion Runner V3
# Using Claude 4 Sonnet with Proof Verification

echo "Yang-Mills Lean Proof Completion System V3"
echo "=========================================="
echo "Using Claude 4 Sonnet ONLY (no other models)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade anthropic

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY environment variable not set!"
    echo ""
    echo "Please run:"
    echo "  ANTHROPIC_API_KEY='your-api-key-here' ./run_proof_completion_v3.sh"
    echo ""
    exit 1
fi

# Create backup of current Lean files
echo "Creating backup of Lean files..."
mkdir -p backups
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r YangMillsProof "backups/YangMillsProof_v3_$timestamp"

echo ""
echo "Starting AI proof completion with verification..."
echo "Features:"
echo "  ✓ Claude 4 Sonnet exclusively"
echo "  ✓ Proof verification before applying"
echo "  ✓ Up to 3 retry attempts per proof"
echo "  ✓ Only applies valid proofs"
echo ""
echo "This will take longer but ensures only valid proofs are applied."
echo ""

python3 lean_ai_agents_v3.py

echo ""
echo "Proof completion finished!"
echo "Backup saved in: backups/YangMillsProof_v3_$timestamp"
echo ""
echo "Check the summary above for:"
echo "  - Number of successful proofs"
echo "  - Final build status"
echo "  - Any proofs that need manual attention" 