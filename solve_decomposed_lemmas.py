#!/usr/bin/env python3
"""
Attempt to solve decomposed helper lemmas using AI.
Focuses on the simpler lemmas created during decomposition.
"""

import subprocess
import re
import os
from typing import List, Tuple, Optional
import json
from pathlib import Path

def find_sorries_with_context(file_path: str) -> List[Tuple[str, int, str, str]]:
    """Find all sorries in a file with their lemma names and contexts."""
    sorries = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_lemma = None
    lemma_start = None
    
    for i, line in enumerate(lines):
        # Track current lemma/theorem
        if any(keyword in line for keyword in ['lemma', 'theorem', 'def']):
            match = re.match(r'\s*(lemma|theorem|def)\s+(\w+)', line)
            if match:
                current_lemma = match.group(2)
                lemma_start = i
        
        # Find sorries
        if 'sorry' in line and current_lemma:
            # Get surrounding context
            start = max(0, i - 10)
            end = min(len(lines), i + 5)
            context = ''.join(lines[start:end])
            
            # Get the full lemma text
            lemma_end = i + 1
            while lemma_end < len(lines) and not any(kw in lines[lemma_end] for kw in ['lemma', 'theorem', 'def', 'end ']):
                lemma_end += 1
            lemma_text = ''.join(lines[lemma_start:lemma_end])
            
            sorries.append((current_lemma, i + 1, context, lemma_text))
    
    return sorries

def categorize_sorry(lemma_name: str, context: str) -> str:
    """Categorize a sorry by difficulty based on lemma name and context."""
    # Helper lemmas from decomposition are often easier
    if any(word in lemma_name for word in ['char_matrix_', 'h_', 'step', 'cost_contribution', 
                                           'gauge_layer_has', 'min_cost', 'cost_sum']):
        return "helper"
    elif 'computation' in context or 'arithmetic' in context:
        return "computation"
    elif 'Sign computation' in context or 'Detailed determinant' in context:
        return "computation"
    elif any(word in lemma_name for word in ['_pos', '_nonneg', '_ge', '_le']):
        return "inequality"
    else:
        return "complex"

def generate_proof_with_ai(lemma_text: str, file_context: str) -> Optional[str]:
    """Generate a proof using Claude."""
    import anthropic
    
    # Check for API key in environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("  Warning: ANTHROPIC_API_KEY not found in environment")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are a Lean 4 proof assistant. Complete the following proof by replacing 'sorry' with a valid proof.

File context:
```lean
{file_context}
```

Lemma to prove:
```lean
{lemma_text}
```

Provide ONLY the proof code that should replace 'sorry'. Use appropriate tactics like:
- simp, ring, field_simp for simplification
- linarith, norm_num for arithmetic
- exact, apply, rw for direct proofs
- calc for calculation chains

Your response should be valid Lean 4 code only, no explanations."""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"AI generation failed: {e}")
        return None

def verify_proof(file_path: str) -> bool:
    """Verify that a file builds successfully."""
    result = subprocess.run(
        ['lake', 'build', file_path.replace('.lean', '')],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def apply_proof(file_path: str, line_num: int, proof: str) -> bool:
    """Apply a proof by replacing sorry with the generated proof."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Replace sorry with the proof
    if line_num - 1 < len(lines) and 'sorry' in lines[line_num - 1]:
        # Handle inline sorry
        lines[line_num - 1] = lines[line_num - 1].replace('sorry', proof)
    else:
        # Handle multi-line case
        for i in range(max(0, line_num - 3), min(len(lines), line_num + 2)):
            if 'sorry' in lines[i]:
                indent = len(lines[i]) - len(lines[i].lstrip())
                # Add proper indentation to multi-line proofs
                indented_proof = '\n'.join(' ' * indent + line for line in proof.split('\n'))
                lines[i] = lines[i].replace('sorry', indented_proof)
                break
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    return True

def main():
    # Files to process (focusing on our decomposed lemmas)
    files = [
        'YangMillsProof/GaugeResidue.lean',
        'YangMillsProof/TransferMatrix.lean',
        'YangMillsProof/BalanceOperator.lean'
    ]
    
    results = {
        'attempted': 0,
        'successful': 0,
        'failed': []
    }
    
    for file_path in files:
        if not os.path.exists(file_path):
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {file_path}")
        print(f"{'='*60}")
        
        # Read file context
        with open(file_path, 'r') as f:
            file_context = f.read()
        
        sorries = find_sorries_with_context(file_path)
        
        for lemma_name, line_num, context, lemma_text in sorries:
            category = categorize_sorry(lemma_name, context)
            
            # Focus on helper lemmas and simple computations
            if category not in ['helper', 'computation', 'inequality']:
                print(f"Skipping {lemma_name} (category: {category})")
                continue
            
            print(f"\nAttempting {lemma_name} at line {line_num} (category: {category})")
            results['attempted'] += 1
            
            # Generate proof
            proof = generate_proof_with_ai(lemma_text, file_context[:5000])  # First 5000 chars for context
            
            if not proof:
                print(f"  Failed to generate proof")
                results['failed'].append((file_path, lemma_name))
                continue
            
            print(f"  Generated proof: {proof[:100]}...")
            
            # Backup the file
            backup_path = f"{file_path}.backup"
            subprocess.run(['cp', file_path, backup_path])
            
            # Apply the proof
            if apply_proof(file_path, line_num, proof):
                # Verify it builds
                if verify_proof(file_path):
                    print(f"  ✓ Successfully proved {lemma_name}")
                    results['successful'] += 1
                else:
                    print(f"  ✗ Proof failed to build")
                    # Restore backup
                    subprocess.run(['cp', backup_path, file_path])
                    results['failed'].append((file_path, lemma_name))
            
            # Clean up backup
            if os.path.exists(backup_path):
                os.remove(backup_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Attempted: {results['attempted']}")
    print(f"Successful: {results['successful']}")
    print(f"Success rate: {results['successful']/results['attempted']*100:.1f}%" if results['attempted'] > 0 else "N/A")
    
    if results['failed']:
        print(f"\nFailed proofs requiring decomposition:")
        for file_path, lemma_name in results['failed']:
            print(f"  - {file_path}: {lemma_name}")

if __name__ == "__main__":
    main() 