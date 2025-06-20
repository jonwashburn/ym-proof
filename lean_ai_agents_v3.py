#!/usr/bin/env python3
"""
Verified AI Agents for Lean Proof Completion
Version 3: Claude 4 Sonnet + Proof Verification
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
import tempfile
import shutil

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
    indent: int

@dataclass
class Agent:
    name: str
    specialty: str
    files: List[str]
    temperature: float = 0.1

class LeanProofVerifier:
    """Verifies that generated proofs are syntactically valid"""
    
    @staticmethod
    async def verify_proof(file_path: str, line_num: int, proof: str) -> Tuple[bool, str]:
        """
        Verify a proof by creating a temporary file and checking if it compiles
        Returns (success, error_message)
        """
        # Create a temporary copy
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(file_path))
        
        try:
            # Read original file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Apply the proof to the temporary file
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]
                if 'sorry' in line:
                    # Get proper indentation
                    indent = len(line) - len(line.lstrip())
                    
                    # Format the proof with proper indentation
                    if not proof.strip().startswith('by'):
                        proof = 'by\n  ' + proof
                    
                    proof_lines = proof.split('\n')
                    indented_proof = proof_lines[0]
                    for pline in proof_lines[1:]:
                        if pline.strip():
                            indented_proof += '\n' + ' ' * (indent + 2) + pline.strip()
                    
                    lines[line_num - 1] = line.replace('sorry', indented_proof)
            
            # Write temporary file
            with open(temp_file, 'w') as f:
                f.writelines(lines)
            
            # Try to compile just this file
            result = await asyncio.create_subprocess_exec(
                'lake', 'env', 'lean', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(file_path)
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return True, "Proof verified successfully"
            else:
                error_msg = stderr.decode('utf-8')
                # Extract relevant error lines
                error_lines = error_msg.split('\n')
                relevant_errors = [line for line in error_lines if 'error:' in line][:3]
                return False, '\n'.join(relevant_errors)
                
        except Exception as e:
            return False, f"Verification error: {str(e)}"
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)

class LeanProofAgent:
    def __init__(self, agent: Agent, api_key: str):
        self.agent = agent
        self.client = anthropic.Anthropic(api_key=api_key)
        self.verifier = LeanProofVerifier()
        
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
                            # Get context (15 lines before and after for better understanding)
                            start = max(0, i - 15)
                            end = min(len(lines), i + 16)
                            context = ''.join(lines[start:end])
                            
                            # Calculate indentation
                            indent = len(line) - len(line.lstrip())
                            
                            sorries.append(Sorry(
                                file=str(file),
                                line=i + 1,
                                context=context,
                                category=self._categorize_sorry(context),
                                full_line=line,
                                indent=indent
                            ))
        return sorries
    
    def _categorize_sorry(self, context: str) -> str:
        """Categorize the type of sorry for better prompting"""
        if 'instance' in context and ':' in context:
            return 'typeclass_instance'
        elif 'charpoly' in context or 'eigenvalue' in context:
            return 'spectral_theory'
        elif 'Matrix' in context and any(op in context for op in ['det', 'trace', 'transpose']):
            return 'matrix_computation'
        elif 'SU(' in context or 'structure constant' in context:
            return 'gauge_theory'
        elif 'path integral' in context or 'correlation' in context:
            return 'quantum_field_theory'
        elif 'phi' in context and ('golden' in context or 'ratio' in context):
            return 'golden_ratio'
        else:
            return 'general'
    
    def _extract_lean_code(self, response: str) -> str:
        """Extract only Lean code from AI response"""
        # Try to find code blocks first
        code_blocks = re.findall(r'```lean\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for code blocks without language specifier
        code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            # Check if it looks like Lean code
            candidate = code_blocks[0].strip()
            if any(kw in candidate for kw in ['by', 'apply', 'exact', 'simp', 'intro']):
                return candidate
        
        # If the response starts with 'by', it's likely all code
        if response.strip().startswith('by'):
            # Take everything until we see explanation keywords
            lines = []
            for line in response.split('\n'):
                if any(phrase in line.lower() for phrase in ['explanation:', 'note:', 'this proves']):
                    break
                lines.append(line)
            return '\n'.join(lines).strip()
        
        # Try to extract proof tactics
        tactic_pattern = r'(by\s+.*?)(?=\n\s*$|\n\s*--|\Z)'
        tactic_match = re.search(tactic_pattern, response, re.DOTALL)
        if tactic_match:
            return tactic_match.group(1).strip()
        
        # If nothing else works, return cleaned response
        cleaned = response.strip()
        # Remove any trailing explanations
        cleaned = re.sub(r'\n\s*(Explanation|Note|This).*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned
    
    async def generate_proof(self, sorry: Sorry, attempt: int = 1) -> Tuple[str, bool]:
        """
        Generate a proof for a single sorry
        Returns (proof_code, is_valid)
        """
        prompt = self._build_prompt(sorry, attempt)
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",  # Using Claude 4 Sonnet as requested
                max_tokens=2000,
                temperature=self.agent.temperature + (attempt - 1) * 0.1,  # Increase temperature on retries
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            raw_response = response.content[0].text
            proof = self._extract_lean_code(raw_response)
            
            # Verify the proof before returning
            logger.info(f"Verifying proof for {sorry.file}:{sorry.line}")
            is_valid, error_msg = await self.verifier.verify_proof(sorry.file, sorry.line, proof)
            
            if not is_valid:
                logger.warning(f"Proof verification failed: {error_msg}")
            
            return proof, is_valid
            
        except Exception as e:
            logger.error(f"Error generating proof: {e}")
            return f"-- Failed to generate proof: {e}", False
    
    def _build_prompt(self, sorry: Sorry, attempt: int) -> str:
        """Build specialized prompt based on sorry category"""
        
        retry_hint = ""
        if attempt > 1:
            retry_hint = f"\n\nThis is attempt {attempt}. Previous attempts failed verification. Please ensure the proof is syntactically correct."
        
        base_prompt = f"""You are a Lean 4 expert specializing in {self.agent.specialty}.

Your task is to complete this proof by replacing 'sorry' with valid Lean 4 code that will compile.

File: {sorry.file}
Context:
```lean
{sorry.context}
```

The specific line with sorry (at indentation level {sorry.indent}):
{sorry.full_line}

Category: {sorry.category}

 CRITICAL REQUIREMENTS:
 1. Output ONLY the Lean code that replaces 'sorry'
 2. NO explanations, NO markdown, NO commentary
 3. The proof MUST compile with Lean 4 and mathlib4
 4. Use 'by' for tactic proofs
 5. Match the indentation level carefully
 6. Use only tactics and lemmas that exist in the current context
 7. NEVER add any axioms - all proofs must be constructive{retry_hint}

Output only the Lean proof code:"""

        # Add category-specific hints
        if sorry.category == 'typeclass_instance':
            base_prompt += """

For this typeclass instance, use the pattern:
{ field1 := value1
  field2 := value2
  ... }
Ensure all required fields are provided."""
            
        elif sorry.category == 'matrix_computation':
            base_prompt += """

For matrix computations, useful tactics:
- simp [matrix_definition]
- ring
- norm_num
- field_simp"""
            
        elif sorry.category == 'spectral_theory':
            base_prompt += """

For spectral theory, consider:
- Matrix.charpoly_apply
- eigenvalue lemmas
- det_fin_three for 3x3 matrices"""
            
        elif sorry.category == 'golden_ratio':
            base_prompt += """

For golden ratio proofs:
- unfold phi
- field_simp
- ring
- Use that phi = (1 + sqrt 5) / 2"""
            
        return base_prompt
    
    async def complete_sorries(self) -> List[Tuple[Sorry, str, bool]]:
        """
        Complete all sorries for this agent
        Returns list of (sorry, proof, is_valid)
        """
        sorries = await self.find_sorries()
        logger.info(f"{self.agent.name} found {len(sorries)} sorries")
        
        results = []
        for sorry in sorries:
            logger.info(f"{self.agent.name} working on {sorry.file}:{sorry.line}")
            
            # Try up to 3 attempts to generate a valid proof
            for attempt in range(1, 4):
                proof, is_valid = await self.generate_proof(sorry, attempt)
                
                if is_valid:
                    logger.info(f"✓ Generated valid proof on attempt {attempt}")
                    results.append((sorry, proof, True))
                    break
                else:
                    logger.warning(f"✗ Attempt {attempt} failed verification")
                    if attempt == 3:
                        results.append((sorry, proof, False))
                
                # Small delay between attempts
                await asyncio.sleep(1)
            
            # Delay between different sorries to avoid rate limits
            await asyncio.sleep(2)
            
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
                # Format the proof with proper indentation
                if not proof.strip().startswith('by'):
                    proof = 'by\n  ' + proof
                
                proof_lines = proof.split('\n')
                indented_proof = proof_lines[0]
                for pline in proof_lines[1:]:
                    if pline.strip():
                        indented_proof += '\n' + ' ' * (sorry.indent + 2) + pline.strip()
                
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
    logger.info("Running full build verification...")
    try:
        result = await asyncio.create_subprocess_exec(
            'lake', 'build',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode == 0:
            return True
        else:
            # Log first few errors
            errors = stderr.decode('utf-8').split('\n')
            for error in errors[:10]:
                if 'error:' in error:
                    logger.error(error)
            return False
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
            specialty="algebraic structures, typeclass instances, module theory, basic definitions",
            files=["YangMillsProof/RSImport/*.lean"],
            temperature=0.1
        ),
        Agent(
            name="SpectralAgent", 
            specialty="matrix theory, eigenvalues, characteristic polynomials, spectral decomposition",
            files=["YangMillsProof/TransferMatrix.lean"],
            temperature=0.15
        ),
        Agent(
            name="GaugeAgent",
            specialty="Lie groups, SU(3) gauge theory, structure constants, balance operators",
            files=["YangMillsProof/BalanceOperator.lean", "YangMillsProof/GaugeResidue.lean"],
            temperature=0.1
        ),
        Agent(
            name="QFTAgent",
            specialty="quantum field theory, path integrals, correlation functions, Osterwalder-Schrader axioms",
            files=["YangMillsProof/OSReconstruction.lean"],
            temperature=0.2
        )
    ]
    
    # Create agent instances
    proof_agents = [LeanProofAgent(agent, api_key) for agent in agents]
    
    # Phase 1: Collect and generate proofs with verification
    logger.info("Phase 1: Generating and verifying proofs...")
    all_results = await asyncio.gather(
        *[agent.complete_sorries() for agent in proof_agents]
    )
    
    # Phase 2: Apply only valid proofs
    logger.info("Phase 2: Applying verified proofs...")
    success_count = 0
    total_count = 0
    failed_proofs = []
    
    for agent_results in all_results:
        for sorry, proof, is_valid in agent_results:
            total_count += 1
            if is_valid:
                if await apply_proof(sorry, proof):
                    success_count += 1
                    logger.info(f"✓ Applied verified proof to {sorry.file}:{sorry.line}")
                else:
                    logger.error(f"✗ Failed to apply proof to {sorry.file}:{sorry.line}")
                    failed_proofs.append((sorry, "Application failed"))
            else:
                logger.warning(f"⚠️  Skipping invalid proof for {sorry.file}:{sorry.line}")
                failed_proofs.append((sorry, "Verification failed"))
    
    # Phase 3: Final build verification
    logger.info("Phase 3: Final build verification...")
    build_success = await verify_lean_builds()
    
    if build_success:
        logger.info("✅ Lean project builds successfully!")
    else:
        logger.error("❌ Build failed - manual intervention needed")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY: Completed {success_count}/{total_count} proofs")
    logger.info(f"Build status: {'SUCCESS' if build_success else 'FAILED'}")
    
    if failed_proofs:
        logger.info(f"\nFailed proofs requiring manual attention:")
        for sorry, reason in failed_proofs:
            logger.info(f"  - {sorry.file}:{sorry.line} ({reason})")

if __name__ == "__main__":
    asyncio.run(main()) 