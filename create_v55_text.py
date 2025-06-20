#!/usr/bin/env python3
import re

def latex_to_text(latex_content):
    """Convert LaTeX content to plain text."""
    
    # Remove comments
    text = re.sub(r'%.*$', '', latex_content, flags=re.MULTILINE)
    
    # Remove preamble
    text = re.sub(r'\\documentclass.*?\\begin\{document\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\end\{document\}.*$', '', text, flags=re.DOTALL)
    
    # Convert sections
    text = re.sub(r'\\section\*?\{([^}]+)\}', r'\n\n# \1\n', text)
    text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'\n\n## \1\n', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]+)\}', r'\n\n### \1\n', text)
    
    # Convert environments
    text = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', r'\nABSTRACT:\n\1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{theorem\}(?:\[[^\]]*\])?(.*?)\\end\{theorem\}', r'\nTHEOREM: \1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{lemma\}(?:\[[^\]]*\])?(.*?)\\end\{lemma\}', r'\nLEMMA: \1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{definition\}(?:\[[^\]]*\])?(.*?)\\end\{definition\}', r'\nDEFINITION: \1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{proof\}(.*?)\\end\{proof\}', r'\nPROOF: \1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{remark\}(?:\[[^\]]*\])?(.*?)\\end\{remark\}', r'\nREMARK: \1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{proposition\}(?:\[[^\]]*\])?(.*?)\\end\{proposition\}', r'\nPROPOSITION: \1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{corollary\}(?:\[[^\]]*\])?(.*?)\\end\{corollary\}', r'\nCOROLLARY: \1\n', text, flags=re.DOTALL)
    
    # Convert equations
    text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'\n\1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', r'\n\1\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'\n\1\n', text, flags=re.DOTALL)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    
    # Convert lists
    text = re.sub(r'\\begin\{itemize\}', '', text)
    text = re.sub(r'\\end\{itemize\}', '', text)
    text = re.sub(r'\\begin\{enumerate\}(?:\[[^\]]*\])?', '', text)
    text = re.sub(r'\\end\{enumerate\}', '', text)
    text = re.sub(r'\\item', '• ', text)
    
    # Convert formatting
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
    text = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
    text = re.sub(r'\\texttt\{([^}]+)\}', r'\1', text)
    
    # Convert math symbols
    text = re.sub(r'\\mathfrak\{su\}\(3\)', 'su(3)', text)
    text = re.sub(r'\\mathfrak\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathcal\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\operatorname\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\varphi', 'φ', text)
    text = re.sub(r'\\phi', 'φ', text)
    text = re.sub(r'\\Delta', 'Δ', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', text)
    text = re.sub(r'\\sqrt', '√', text)
    text = re.sub(r'\\infty', '∞', text)
    text = re.sub(r'\\sum', 'Σ', text)
    text = re.sub(r'\\prod', 'Π', text)
    text = re.sub(r'\\int', '∫', text)
    text = re.sub(r'\\partial', '∂', text)
    text = re.sub(r'\\nabla', '∇', text)
    text = re.sub(r'\\times', '×', text)
    text = re.sub(r'\\cdot', '·', text)
    text = re.sub(r'\\geq', '≥', text)
    text = re.sub(r'\\leq', '≤', text)
    text = re.sub(r'\\neq', '≠', text)
    text = re.sub(r'\\approx', '≈', text)
    text = re.sub(r'\\equiv', '≡', text)
    text = re.sub(r'\\subset', '⊂', text)
    text = re.sub(r'\\subseteq', '⊆', text)
    text = re.sub(r'\\in', '∈', text)
    text = re.sub(r'\\notin', '∉', text)
    text = re.sub(r'\\forall', '∀', text)
    text = re.sub(r'\\exists', '∃', text)
    text = re.sub(r'\\rightarrow', '→', text)
    text = re.sub(r'\\leftarrow', '←', text)
    text = re.sub(r'\\Rightarrow', '⇒', text)
    text = re.sub(r'\\Leftarrow', '⇐', text)
    text = re.sub(r'\\iff', '⇔', text)
    text = re.sub(r'\\mapsto', '↦', text)
    text = re.sub(r'\\to', '→', text)
    text = re.sub(r'\\circ', '∘', text)
    text = re.sub(r'\\oplus', '⊕', text)
    text = re.sub(r'\\otimes', '⊗', text)
    text = re.sub(r'\\wedge', '∧', text)
    text = re.sub(r'\\vee', '∨', text)
    text = re.sub(r'\\cap', '∩', text)
    text = re.sub(r'\\cup', '∪', text)
    text = re.sub(r'\\emptyset', '∅', text)
    text = re.sub(r'\\varnothing', '∅', text)
    
    # Remove remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    text = re.sub(r'\{', '', text)
    text = re.sub(r'\}', '', text)
    text = re.sub(r'\\\\', '\n', text)
    text = re.sub(r'&', ' ', text)
    text = re.sub(r'~', ' ', text)
    
    # Clean up
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

# Read the LaTeX file
with open('Yang-Mills-v55.tex', 'r') as f:
    latex_content = f.read()

# Convert to text
text_content = latex_to_text(latex_content)

# Write the text file
with open('Yang-Mills-v55.txt', 'w') as f:
    f.write(text_content)

print("Conversion complete. Output saved to Yang-Mills-v55.txt") 