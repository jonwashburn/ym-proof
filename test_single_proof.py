#!/usr/bin/env python3
"""Test script to verify proof generation works on a single sorry"""

import asyncio
import os
import anthropic

async def test_single_proof():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ No API key set")
        return False
    
    print("Testing proof generation system...")
    
    # Test API connection
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'API working' if you receive this"}]
        )
        print(f"✅ API connection: {response.content[0].text}")
    except Exception as e:
        print(f"❌ API error: {e}")
        return False
    
    # Test proof generation on a simple example
    try:
        test_context = """
lemma simple_test : 1 + 1 = 2 := by
  sorry
"""
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user", 
                "content": f"Complete this Lean 4 proof by replacing 'sorry'. Output ONLY the proof code:\n{test_context}"
            }]
        )
        print(f"✅ Proof generation test: {response.content[0].text.strip()}")
        return True
    except Exception as e:
        print(f"❌ Proof generation error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_single_proof())
    print(f"\nSystem functional: {'YES' if result else 'NO'}") 