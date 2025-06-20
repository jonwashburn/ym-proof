#!/bin/bash

# Yang-Mills Lean Proof Completion Runner
# This script safely runs the parallel AI agents to complete the Lean proofs

echo "Yang-Mills Lean Proof Completion System"
echo "======================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY environment variable not set!"
    echo ""
    echo "Please run one of the following:"
    echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
    echo "  ANTHROPIC_API_KEY='your-api-key-here' ./run_proof_completion.sh"
    echo ""
    exit 1
fi

# Create backup of current Lean files
echo "Creating backup of Lean files..."
mkdir -p backups
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r YangMillsProof "backups/YangMillsProof_$timestamp"

# Run the proof completion
echo "Starting parallel proof completion agents..."
echo "This may take 1-4 hours depending on API response times."
echo ""

python lean_ai_agents.py

echo ""
echo "Proof completion finished!"
echo "Backup saved in: backups/YangMillsProof_$timestamp"
echo ""
echo "To verify the build manually, run:"
echo "  lake build" 