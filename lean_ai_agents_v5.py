#!/usr/bin/env python3
"""
Verified AI Agents for Lean Proof Completion
Version 5: Fixed application logic with search_replace + better proof handling
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
        Verify a proof by checking if it would compile
        Returns (success, error_message)
        """
        # Check for basic syntax errors
        if not proof.strip():
            return False, "Empty proof"
        
        # Check for common issues
        if '```' in proof:
            return False, "Contains markdown formatting"
        
        if any(phrase in proof.lower() for phrase in ['explanation:', 'note:', 'here is', 'this proof']):
            return False, "Contains explanatory text"
        
        # Check for balanced parentheses/brackets
        open_parens = proof.count('(') + proof.count('[') + proof.count('{')
        close_parens = proof.count(')') + proof.count(']') + proof.count('}')
        if open_parens != close_parens:
            return False, "Unbalanced parentheses/brackets"
        
        # Basic Lean syntax validation
        if 'sorry' in proof:
            return False, "Proof contains 'sorry'"
        
        if 'axiom' in proof:
            return False, "Proof contains 'axiom' (forbidden)"
        
        # Check for incomplete definitions
        if proof.strip().endswith(',') or proof.strip().endswith(';'):
            return False, "Definition ends with invalid character"
        
        # Real verification: try to compile a test file
        return await LeanProofVerifier._compile_test(file_path, line_num, proof)
    
    @staticmethod
    async def _compile_test(original_file: str, line_num: int, proof: str) -> Tuple[bool, str]:
        """Test compilation by creating a modified copy"""
        try:
            # Read original file
            with open(original_file, 'r') as f:
                lines = f.readlines()
            
            if line_num - 1 >= len(lines):
                return False, "Line number out of range"
            
            # Create modified lines
            modified_lines = lines.copy()
            original_line = lines[line_num - 1]
            
            if 'sorry' not in original_line:
                return False, "No sorry found in line"
            
            # Replace sorry with proof
            if ':=' in original_line and 'by' not in proof:
                # Simple definition replacement
                modified_lines[line_num - 1] = original_line.replace('sorry', proof)
            else:
                # Proof replacement
                if not proof.strip().startswith('by'):
                    proof = 'by ' + proof.strip()
                modified_lines[line_num - 1] = original_line.replace('sorry', proof)
            
            # Write test file
            test_file = f"/tmp/lean_test_{os.getpid()}.lean"
            with open(test_file, 'w') as f:
                f.writelines(modified_lines)
            
            # Try to compile with lake env lean
            result = await asyncio.create_subprocess_exec(
                'lake', 'env', 'lean', test_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            stdout, stderr = await result.communicate()
            
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)
            
            if result.returncode == 0:
                return True, "Compilation successful"
            else:
                error_msg = stderr.decode() if stderr else stdout.decode()
                return False, f"Compilation failed: {error_msg[:200]}"
            
        except Exception as e:
            return False, f"Test compilation error: {str(e)}"

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
                            # Get context (10 lines before and after)
                            start = max(0, i - 8)
                            end = min(len(lines), i + 6)
                            context = ''.join(lines[start:end])
                            
                            # Calculate indentation
                            indent = len(line) - len(line.lstrip())
                            
                            sorries.append(Sorry(
                                file=str(file),
                                line=i + 1,
                                context=context,
                                category=self._categorize_sorry(context, line),
                                full_line=line,
                                indent=indent
                            ))
        return sorries
    
    def _categorize_sorry(self, context: str, line: str) -> str:
        """Categorize the type of sorry for better prompting"""
        if 'def' in line and ':=' in line:
            return 'definition'
        elif 'lemma' in context or 'theorem' in context:
            return 'proof'
        elif 'instance' in context:
            return 'instance'
        else:
            return 'general'
    
    def _extract_lean_code(self, response: str) -> str:
        """Extract only Lean code from AI response"""
        # Remove any markdown code blocks
        response = re.sub(r'```lean\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Remove explanatory text
        lines = response.split('\n')
        code_lines = []
        in_code = True
        
        for line in lines:
            # Skip lines that are clearly explanations
            if any(phrase in line.lower() for phrase in [
                'explanation:', 'note:', 'this proof', 'here we', 'we need to', 'first,', 'then,', 'finally,'
            ]):
                in_code = False
                continue
            
            # Skip empty lines after explanations
            if not in_code and line.strip() == '':
                continue
            
            # Include lines that look like code
            if line.strip():
                in_code = True
                code_lines.append(line)
        
        result = '\n'.join(code_lines).strip()
        
        # Clean up common issues
        result = re.sub(r'\s*--.*$', '', result, flags=re.MULTILINE)  # Remove comments
        result = re.sub(r'\n\s*\n', '\n', result)  # Remove extra blank lines
        
        return result
    
    async def generate_proof(self, sorry: Sorry, attempt: int = 1) -> Tuple[str, bool]:
        """
        Generate a proof for a single sorry
        Returns (proof_code, is_valid)
        """
        prompt = self._build_prompt(sorry, attempt)
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=self.agent.temperature + (attempt - 1) * 0.05,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            raw_response = response.content[0].text
            proof = self._extract_lean_code(raw_response)
            
            # Verify the proof
            logger.info(f"Verifying proof for {sorry.file}:{sorry.line}")
            is_valid, error_msg = await self.verifier.verify_proof(sorry.file, sorry.line, proof)
            
            if not is_valid:
                logger.warning(f"Proof validation failed: {error_msg}")
            else:
                logger.info(f"✓ Proof validation passed")
            
            return proof, is_valid
            
        except Exception as e:
            logger.error(f"Error generating proof: {e}")
            return f"-- Failed to generate proof: {e}", False
    
    def _build_prompt(self, sorry: Sorry, attempt: int) -> str:
        """Build specialized prompt based on sorry category"""
        
        retry_hint = ""
        if attempt > 1:
            retry_hint = f"\n\nThis is attempt {attempt}. Previous attempts failed validation. Be more careful with syntax and ensure the proof is complete."
        
        base_requirements = """
CRITICAL REQUIREMENTS:
- Output ONLY the code that replaces 'sorry'
- NO explanations, comments, or markdown
- NEVER use axioms
- Ensure proper Lean 4 syntax"""
        
        if sorry.category == 'definition':
            prompt = f"""Complete this Lean 4 definition by replacing 'sorry' with a simple expression.

Context:
```lean
{sorry.context}
```

Line with sorry:
{sorry.full_line}

{base_requirements}
- For definitions, do NOT use 'by' tactics
- Use simple expressions like: 0, 1, if-then-else, etc.
- For structure constants: if i = j ∨ j = k ∨ i = k then 0 else 1{retry_hint}

Output only the expression:"""

        elif sorry.category == 'proof':
            prompt = f"""Complete this Lean 4 proof by replacing 'sorry'.

Context:
```lean  
{sorry.context}
```

Line with sorry:
{sorry.full_line}

{base_requirements}
- Start with 'by' for tactic proofs
- Use simple tactics: simp, ring, norm_num, rfl, exact, apply, intro, linarith
- Keep proofs concise{retry_hint}

Output only the proof:"""

        elif sorry.category == 'instance':
            prompt = f"""Complete this Lean 4 typeclass instance by replacing 'sorry'.

Context:
```lean
{sorry.context}
```

Line with sorry:
{sorry.full_line}

{base_requirements}
- Use structural definitions with := 
- Reference existing instances where possible{retry_hint}

Output only the instance definition:"""

        else:
            # Generic prompt
            prompt = f"""Complete this Lean 4 code by replacing 'sorry'.

Context:
```lean
{sorry.context}
```

Line with sorry:
{sorry.full_line}

{base_requirements}{retry_hint}

Output only the Lean code:"""
            
        return prompt
    
    async def complete_sorries(self) -> List[Tuple[Sorry, str, bool]]:
        """
        Complete all sorries for this agent
        Returns list of (sorry, proof, is_valid)
        """
        sorries = await self.find_sorries()
        logger.info(f"{self.agent.name} found {len(sorries)} sorries")
        
        results = []
        for sorry in sorries:
            logger.info(f"{self.agent.name} working on {sorry.file}:{sorry.line} ({sorry.category})")
            
            # Try up to 3 attempts
            proof = None
            is_valid = False
            
            for attempt in range(1, 4):
                proof, is_valid = await self.generate_proof(sorry, attempt)
                
                if is_valid:
                    logger.info(f"✓ Generated valid proof on attempt {attempt}")
                    results.append((sorry, proof, True))
                    break
                else:
                    logger.warning(f"✗ Attempt {attempt} failed validation")
                
                await asyncio.sleep(0.5)
            
            if not is_valid:
                results.append((sorry, proof, False))
            
            # Delay between sorries
            await asyncio.sleep(1)
            
        return results

async def apply_proof_safely(sorry: Sorry, proof: str) -> bool:
    """Apply a generated proof using search_replace for better handling"""
    try:
        with open(sorry.file, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        if sorry.line - 1 >= len(lines):
            return False
        
        target_line = lines[sorry.line - 1]
        if 'sorry' not in target_line:
            return False
        
        # Build context for unique identification
        start_line = max(0, sorry.line - 3)
        end_line = min(len(lines), sorry.line + 2)
        context_lines = lines[start_line:end_line]
        old_string = '\n'.join(context_lines)
        
        # Create new string with replacement
        new_context_lines = context_lines.copy()
        relative_line = sorry.line - 1 - start_line
        
        if ':=' in target_line and 'by' not in proof:
            # Simple definition replacement
            new_context_lines[relative_line] = target_line.replace('sorry', proof)
        else:
            # Proof replacement - ensure proper formatting
            if not proof.strip().startswith('by'):
                proof = 'by ' + proof.strip()
            new_context_lines[relative_line] = target_line.replace('sorry', proof)
        
        new_string = '\n'.join(new_context_lines)
        
        # Apply the replacement
        new_content = content.replace(old_string, new_string)
        
        # Write back only if changed
        if new_content != content:
            with open(sorry.file, 'w') as f:
                f.write(new_content)
            return True
        
        return False
            
    except Exception as e:
        logger.error(f"Failed to apply proof: {e}")
        return False

async def verify_lean_builds() -> bool:
    """Check if the Lean project builds successfully"""
    logger.info("Running build verification...")
    try:
        result = await asyncio.create_subprocess_exec(
            'lake', 'build',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            error_output = stderr.decode() if stderr else stdout.decode()
            logger.error(f"Build errors: {error_output[:500]}")
        
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Build verification failed: {e}")
        return False

async def main():
    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Define specialized agents with better file targeting
    agents = [
        Agent(
            name="DefinitionAgent",
            specialty="simple definitions and expressions",
            files=["YangMillsProof/BalanceOperator.lean", "YangMillsProof/RSImport/BasicDefinitions.lean"],
            temperature=0.05
        ),
        Agent(
            name="ProofAgent", 
            specialty="mathematical proofs and lemmas",
            files=["YangMillsProof/RSImport/GoldenRatio.lean"],
            temperature=0.1
        ),
        Agent(
            name="MatrixAgent",
            specialty="matrix and spectral theory",
            files=["YangMillsProof/TransferMatrix.lean"],
            temperature=0.1
        ),
        Agent(
            name="GaugeAgent",
            specialty="gauge theory definitions",
            files=["YangMillsProof/GaugeResidue.lean"],
            temperature=0.1
        ),
        Agent(
            name="QFTAgent",
            specialty="quantum field theory",
            files=["YangMillsProof/OSReconstruction.lean"],
            temperature=0.15
        )
    ]
    
    # Create agent instances
    proof_agents = [LeanProofAgent(agent, api_key) for agent in agents]
    
    # Phase 1: Generate and verify proofs
    logger.info("Phase 1: Generating and verifying proofs...")
    all_results = await asyncio.gather(
        *[agent.complete_sorries() for agent in proof_agents]
    )
    
    # Phase 2: Apply verified proofs
    logger.info("Phase 2: Applying verified proofs...")
    success_count = 0
    total_count = 0
    applied_proofs = []
    failed_proofs = []
    
    for agent_results in all_results:
        for sorry, proof, is_valid in agent_results:
            total_count += 1
            if is_valid:
                if await apply_proof_safely(sorry, proof):
                    success_count += 1
                    logger.info(f"✓ Applied proof to {sorry.file}:{sorry.line}")
                    applied_proofs.append((sorry, proof))
                else:
                    logger.error(f"✗ Failed to apply proof to {sorry.file}:{sorry.line}")
                    failed_proofs.append((sorry, "Application failed"))
            else:
                logger.warning(f"⚠️  Skipping invalid proof for {sorry.file}:{sorry.line}")
                failed_proofs.append((sorry, "Invalid proof"))
    
    # Phase 3: Final build check
    logger.info("Phase 3: Final build verification...")
    build_success = await verify_lean_builds()
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY: Applied {success_count}/{total_count} verified proofs")
    logger.info(f"Build status: {'SUCCESS' if build_success else 'FAILED'}")
    
    if applied_proofs:
        logger.info(f"\nSuccessfully applied proofs:")
        for sorry, _ in applied_proofs:
            logger.info(f"  ✓ {sorry.file}:{sorry.line}")
    
    if failed_proofs:
        logger.info(f"\nFailed proofs requiring manual attention:")
        for sorry, reason in failed_proofs[:10]:  # Show first 10
            logger.info(f"  ✗ {sorry.file}:{sorry.line} ({reason})")
        if len(failed_proofs) > 10:
            logger.info(f"  ... and {len(failed_proofs) - 10} more")

if __name__ == "__main__":
    asyncio.run(main()) 