#!/usr/bin/env python3
"""
Check for 'sorry' in core Yang-Mills proof files.
Returns 0 if no sorries found in core chain, 1 otherwise.
"""

import os
import sys
import re

# Core proof files that must be sorry-free
CORE_FILES = [
    "Parameters/Constants.lean",
    "Parameters/DerivedConstants.lean",
    "Parameters/Assumptions.lean", 
    "Wilson/LedgerBridge.lean",
    "Measure/ReflectionPositivity.lean",
    "RG/ContinuumLimit.lean",
    "Topology/ChernWhitney.lean",
    "RG/StepScaling.lean",
    "TransferMatrix.lean",
    "Complete.lean"
]

# Directories to ignore
IGNORE_DIRS = ["working", "gravity", "recognition-ledger", "backups"]

def check_file_for_sorry(filepath):
    """Check if a file contains 'sorry' (excluding comments)."""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove single-line comments
    content = re.sub(r'--.*$', '', content, flags=re.MULTILINE)
    
    # Remove multi-line comments
    content = re.sub(r'/-.*?-/', '', content, flags=re.DOTALL)
    
    # Check for 'sorry'
    if re.search(r'\bsorry\b', content):
        # Count occurrences
        count = len(re.findall(r'\bsorry\b', content))
        return True, f"Found {count} sorry(s) in {filepath}"
    
    return False, None

def main():
    """Check all core files for sorry."""
    found_sorry = False
    results = []
    
    # Find YangMillsProof directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up from scripts/
    
    print("Checking core Yang-Mills proof files for 'sorry'...\n")
    
    for file in CORE_FILES:
        filepath = os.path.join(base_dir, file)
        has_sorry, message = check_file_for_sorry(filepath)
        
        if has_sorry:
            found_sorry = True
            results.append(f"❌ {message}")
        else:
            if message:  # File not found
                results.append(f"⚠️  {message}")
            else:
                results.append(f"✅ {file} - clean")
    
    # Print results
    for result in results:
        print(result)
    
    print("\n" + "="*50)
    
    if found_sorry:
        print("❌ FAILED: Found 'sorry' in core proof files!")
        print("   Please complete all proofs before merging.")
        return 1
    else:
        print("✅ SUCCESS: No 'sorry' found in core proof chain!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 