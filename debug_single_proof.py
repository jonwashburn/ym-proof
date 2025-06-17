#!/usr/bin/env python3
"""Debug script to see what proof is being generated for a specific sorry"""

import asyncio
import os
import anthropic

async def debug_single_proof():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ No API key set")
        return
    
    # Test with a simple sorry from BalanceOperator.lean
    context = """
/-- Structure constants for the balance algebra -/
noncomputable def balanceStructureConstants (i j k : Fin 8) : ℝ :=
  sorry -- Define SU(3) structure constants
"""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": f"""You are a Lean 4 expert specializing in Lie groups, SU(3) gauge theory.

Complete this proof by replacing 'sorry' with valid Lean 4 code.

Context:
```lean
{context}
```

CRITICAL REQUIREMENTS:
1. Output ONLY the Lean code that replaces 'sorry'
2. NO explanations, NO markdown, NO commentary
3. The proof MUST compile with Lean 4 and mathlib4
4. NEVER add any axioms

For SU(3) structure constants, you can use a simple placeholder definition like:
if i = j ∨ j = k ∨ i = k then 0 else 1

Output only the Lean proof code:"""
            }]
        )
        
        print("Generated proof:")
        print("-" * 50)
        print(response.content[0].text)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_single_proof()) 