#!/usr/bin/env python3
"""
Improved Parallel AI Agents for Lean Proof Completion
Version 2: Better proof extraction from AI responses
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
    full_line: str

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
                            # Get context (10 lines before and after)
                            start = max(0, i - 10)
                            end = min(len(lines), i + 11)
                            context = ''.join(lines[start:end])
                            sorries.append(Sorry(
                                file=str(file),
                                line=i + 1,
                                context=context,
                                category=self._categorize_sorry(context),
                                full_line=line
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
    
    def _extract_lean_code(self, response: str) -> str:
        """Extract only Lean code from AI response"""
        # Try to find code blocks
        code_blocks = re.findall(r'```lean\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Try to find code between 'by' and end of proof
        by_match = re.search(r'\bby\b\s*(.*?)(?=\n\s*(?:def|lemma|theorem|example|/--|$))', response, re.DOTALL)
        if by_match:
            return 'by\n  ' + by_match.group(1).strip()
        
        # If no code blocks, try to extract lines that look like Lean code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip obvious non-code lines
            if any(phrase in line.lower() for phrase in ['explanation:', 'here is', 'this proof', 'note:']):
                in_code = False
                continue
            
            # Start collecting if we see Lean keywords
            if any(kw in line for kw in ['by', 'apply', 'exact', 'intro', 'have', 'rw', 'simp', 'constructor']):
                in_code = True
            
            if in_code and line.strip():
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Last resort: return the whole response, but cleaned
        cleaned = re.sub(r'(Explanation:|Note:|Here is|This proof).*$', '', response, flags=re.MULTILINE)
        return cleaned.strip()
    
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
            raw_response = response.content[0].text
            return self._extract_lean_code(raw_response)
        except Exception as e:
            logger.error(f"Error generating proof: {e}")
            return f"-- Failed to generate proof: {e}"
    
    def _build_prompt(self, sorry: Sorry) -> str:
        """Build specialized prompt based on sorry category"""
        base_prompt = f"""You are a Lean 4 expert specializing in {self.agent.specialty}.
        
Complete this proof by replacing 'sorry' with valid Lean 4 code.

File: {sorry.file}
Context:
```lean
{sorry.context}
```

The specific line with sorry is:
{sorry.full_line}

IMPORTANT: 
- Provide ONLY the Lean code that replaces 'sorry'
- Do NOT include explanations, markdown formatting, or any other text
- The code should compile with Lean 4 and mathlib4
- Start with 'by' if it's a tactic proof

Just give me the pure Lean code, nothing else."""

        # Add category-specific hints
        if sorry.category == 'typeclass_instance':
            base_prompt += "\n\nFor this instance, use the pattern: { field1 := value1, field2 := value2, ... }"
        elif sorry.category == 'matrix_computation':
            base_prompt += "\n\nUse tactics like: simp, ring, norm_num, field_simp"
            
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
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
            
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
                # Determine proper indentation
                indent = len(line) - len(line.lstrip())
                
                # If proof doesn't start with 'by', add it
                if not proof.strip().startswith('by'):
                    proof = 'by\n  ' + proof
                
                # Properly indent the proof
                proof_lines = proof.split('\n')
                indented_proof = proof_lines[0]  # First line (usually 'by')
                for pline in proof_lines[1:]:
                    if pline.strip():
                        indented_proof += '\n' + ' ' * (indent + 2) + pline.strip()
                
                # Replace sorry with the proof
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
    
    # Define specialized agents (focusing on specific files to avoid duplication)
    agents = [
        Agent(
            name="AlgebraicAgent",
            specialty="algebraic structures, typeclass instances, basic definitions",
            files=["YangMillsProof/RSImport/*.lean"],
            temperature=0.1
        ),
        Agent(
            name="SpectralAgent", 
            specialty="matrix theory, eigenvalues, spectral decomposition, transfer matrices",
            files=["YangMillsProof/TransferMatrix.lean"],
            temperature=0.2
        ),
        Agent(
            name="GaugeAgent",
            specialty="Lie groups, SU(3) gauge theory, balance operators",
            files=["YangMillsProof/BalanceOperator.lean", "YangMillsProof/GaugeResidue.lean"],
            temperature=0.1
        ),
        Agent(
            name="QFTAgent",
            specialty="quantum field theory, path integrals, Osterwalder-Schrader reconstruction",
            files=["YangMillsProof/OSReconstruction.lean"],
            temperature=0.2
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
        logger.error("✗ Build failed - check 'lake build' for details")
    
    logger.info(f"Completed {success_count}/{total_count} proofs")

if __name__ == "__main__":
    asyncio.run(main()) 