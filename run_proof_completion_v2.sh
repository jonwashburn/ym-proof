#!/bin/bash

# Yang-Mills Lean Proof Completion Runner V2
# Uses improved proof extraction

echo "Yang-Mills Lean Proof Completion System V2"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY environment variable not set!"
    echo ""
    echo "Please run:"
    echo "  ANTHROPIC_API_KEY='your-api-key-here' ./run_proof_completion_v2.sh"
    echo ""
    exit 1
fi

# Create backup of current Lean files
echo "Creating backup of Lean files..."
mkdir -p backups
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r YangMillsProof "backups/YangMillsProof_v2_$timestamp"

echo ""
echo "Starting improved proof completion agents..."
echo "Features:"
echo "  - Better proof extraction from AI responses"
echo "  - Improved indentation handling" 
echo "  - Smarter categorization"
echo ""

python3 lean_ai_agents_v2.py

echo ""
echo "Proof completion finished!"
echo "Backup saved in: backups/YangMillsProof_v2_$timestamp" 