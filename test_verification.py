#!/usr/bin/env python3
"""Test the verification process to see why it's failing"""

import subprocess
import tempfile
import shutil

def test_verification():
    # Create a minimal test file
    test_content = """
import Mathlib.Data.Real.Basic

def test : 1 + 1 = 2 := by
  rfl
"""
    
    # Create temp file
    temp_dir = tempfile.mkdtemp()
    temp_file = f"{temp_dir}/test.lean"
    
    try:
        with open(temp_file, 'w') as f:
            f.write(test_content)
        
        print(f"Testing verification with file: {temp_file}")
        
        # Try to compile with lake env lean
        result = subprocess.run(
            ['lake', 'env', 'lean', temp_file],
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
        
        # Also try a simpler approach
        print("\n" + "="*50 + "\n")
        print("Testing with direct lean command:")
        
        result2 = subprocess.run(
            ['lean', temp_file],
            capture_output=True,
            text=True
        )
        
        print(f"Return code: {result2.returncode}")
        print(f"Stdout:\n{result2.stdout}")
        print(f"Stderr:\n{result2.stderr}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_verification() 