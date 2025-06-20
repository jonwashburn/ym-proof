#!/usr/bin/env python3
"""Test script to verify AI agent setup before running full proof completion"""

import os
import sys
from pathlib import Path

def test_setup():
    print("Testing Yang-Mills Lean AI Proof Completion Setup...")
    print("=" * 50)
    
    # Check Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ required")
        return False
    
    # Check API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("✗ ANTHROPIC_API_KEY not set")
        return False
    print(f"✓ API key found ({len(api_key)} characters)")
    
    # Check Lean files exist
    lean_files = list(Path("YangMillsProof").glob("*.lean"))
    if not lean_files:
        print("✗ No Lean files found in YangMillsProof/")
        return False
    print(f"✓ Found {len(lean_files)} Lean files")
    
    # Count sorries
    sorry_count = 0
    for file in Path("YangMillsProof").rglob("*.lean"):
        with open(file, 'r') as f:
            content = f.read()
            sorry_count += content.count('sorry')
    print(f"✓ Found {sorry_count} sorries to resolve")
    
    # Check imports
    try:
        import anthropic
        print("✓ Anthropic library available")
    except ImportError:
        print("✗ Anthropic library not installed")
        print("  Run: pip install anthropic")
        return False
    
    # Test API connection (optional)
    print("\nSetup test passed! Ready to run proof completion.")
    print("\nNext step: ./run_proof_completion.sh")
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1) 