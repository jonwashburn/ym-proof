#!/usr/bin/env python3
"""
Parallel AI Agents for Lean Proof Completion
Completes sorries in Yang-Mills proof using multiple specialized agents
"""

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional
import anthropic
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Sorry:
    file: str
    line: int
    context: str
    category: str

@dataclass
class Agent:
    name: str
    specialty: str
    files: List[str]
    temperature: float = 0.1

class LeanProofAgent:
    def __init__(self, agent: Agent, api_key: str):
        self.agent = agent
        self.client = anthropic.Anthropic(api_key=api_key)
        
    async def find_sorries(self) -> List[Sorry]:
        """Find all sorries in agent's assigned files"""
        sorries = []
        for file_pattern in self.agent.files:
            files = Path(".").glob(file_pattern)
            for file in files:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if 'sorry' in line and not line.strip().startswith('--'):
                            # Get context (5 lines before and after)
                            start = max(0, i - 5)
                            end = min(len(lines), i + 6)
                            context = ''.join(lines[start:end])
                            sorries.append(Sorry(
                                file=str(file),
                                line=i + 1,
                                context=context,
                                category=self._categorize_sorry(context)
                            ))
        return sorries
    
    def _categorize_sorry(self, context: str) -> str:
        """Categorize the type of sorry for better prompting"""
        if 'instance' in context:
            return 'typeclass_instance'
        elif 'charpoly' in context or 'eigenvalue' in context:
            return 'spectral_theory'
        elif 'Matrix' in context:
            return 'matrix_computation'
        elif 'SU(' in context or 'structure constant' in context:
            return 'gauge_theory'
        elif 'path integral' in context or 'correlation' in context:
            return 'quantum_field_theory'
        else:
            return 'general'
    
    async def generate_proof(self, sorry: Sorry) -> str:
        """Generate a proof for a single sorry"""
        prompt = self._build_prompt(sorry)
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=self.agent.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating proof: {e}")
            return f"-- Failed to generate proof: {e}"
    
    def _build_prompt(self, sorry: Sorry) -> str:
        """Build specialized prompt based on sorry category"""
        base_prompt = f"""You are a Lean 4 expert specializing in {self.agent.specialty}.
        
Complete this proof by replacing the 'sorry' with a valid Lean 4 proof.
Use mathlib4 tactics and lemmas where appropriate.

File: {sorry.file}
Line: {sorry.line}

Context:
```lean
{sorry.context}
```

Requirements:
1. The proof must compile with Lean 4 and mathlib4
2. Use appropriate tactics for {sorry.category}
3. Be concise but complete
4. Include necessary imports if missing

Provide ONLY the proof code that replaces 'sorry', nothing else."""

        # Add category-specific hints
        if sorry.category == 'typeclass_instance':
            base_prompt += "\n\nHint: Use pattern { field1 := ..., field2 := ... } and prove each field goal."
        elif sorry.category == 'spectral_theory':
            base_prompt += "\n\nHint: Use Matrix.charpoly, eigenvalue lemmas from mathlib."
        elif sorry.category == 'matrix_computation':
            base_prompt += "\n\nHint: Use simp, ring, norm_num for computation."
            
        return base_prompt
    
    async def complete_sorries(self) -> List[Tuple[Sorry, str]]:
        """Complete all sorries for this agent"""
        sorries = await self.find_sorries()
        logger.info(f"{self.agent.name} found {len(sorries)} sorries")
        
        results = []
        for sorry in sorries:
            logger.info(f"{self.agent.name} working on {sorry.file}:{sorry.line}")
            proof = await self.generate_proof(sorry)
            results.append((sorry, proof))
            
        return results

async def apply_proof(sorry: Sorry, proof: str) -> bool:
    """Apply a generated proof to the file"""
    try:
        with open(sorry.file, 'r') as f:
            lines = f.readlines()
        
        # Find the exact sorry to replace
        if sorry.line - 1 < len(lines):
            line = lines[sorry.line - 1]
            if 'sorry' in line:
                # Replace sorry with the proof
                indent = len(line) - len(line.lstrip())
                indented_proof = '\n'.join(' ' * indent + l for l in proof.split('\n'))
                lines[sorry.line - 1] = line.replace('sorry', indented_proof)
                
                with open(sorry.file, 'w') as f:
                    f.writelines(lines)
                return True
    except Exception as e:
        logger.error(f"Failed to apply proof: {e}")
    return False

async def verify_lean_builds() -> bool:
    """Check if the Lean project builds successfully"""
    try:
        result = await asyncio.create_subprocess_exec(
            'lake', 'build',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Build verification failed: {e}")
        return False

async def main():
    # Get API key from environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Define specialized agents
    agents = [
        Agent(
            name="AlgebraicAgent",
            specialty="algebraic structures, typeclass instances, module theory",
            files=["YangMillsProof/TransferMatrix.lean", "YangMillsProof/RSImport/*.lean"],
            temperature=0.1
        ),
        Agent(
            name="SpectralAgent", 
            specialty="matrix theory, eigenvalues, spectral decomposition",
            files=["YangMillsProof/TransferMatrix.lean"],
            temperature=0.2
        ),
        Agent(
            name="GaugeAgent",
            specialty="Lie groups, SU(3) gauge theory, representation theory",
            files=["YangMillsProof/BalanceOperator.lean", "YangMillsProof/GaugeResidue.lean"],
            temperature=0.1
        ),
        Agent(
            name="QFTAgent",
            specialty="quantum field theory, path integrals, correlation functions",
            files=["YangMillsProof/OSReconstruction.lean"],
            temperature=0.3
        )
    ]
    
    # Create agent instances
    proof_agents = [LeanProofAgent(agent, api_key) for agent in agents]
    
    # Phase 1: Collect all sorries
    logger.info("Phase 1: Collecting sorries...")
    all_results = await asyncio.gather(
        *[agent.complete_sorries() for agent in proof_agents]
    )
    
    # Phase 2: Apply proofs
    logger.info("Phase 2: Applying proofs...")
    success_count = 0
    total_count = 0
    
    for agent_results in all_results:
        for sorry, proof in agent_results:
            total_count += 1
            if await apply_proof(sorry, proof):
                success_count += 1
                logger.info(f"✓ Applied proof to {sorry.file}:{sorry.line}")
            else:
                logger.error(f"✗ Failed to apply proof to {sorry.file}:{sorry.line}")
    
    # Phase 3: Verify build
    logger.info("Phase 3: Verifying build...")
    if await verify_lean_builds():
        logger.info("✓ Lean project builds successfully!")
    else:
        logger.error("✗ Build failed - manual intervention needed")
    
    logger.info(f"Completed {success_count}/{total_count} proofs")

if __name__ == "__main__":
    asyncio.run(main()) 