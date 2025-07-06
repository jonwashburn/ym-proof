# Contributing to Yang-Mills Proof

Thank you for your interest in contributing to this formal verification of the Yang-Mills mass gap problem!

## 🎯 Project Goals

This repository aims to provide a complete, formally verified proof of the Yang-Mills mass gap in Lean 4, using Recognition Science principles. All contributions should maintain mathematical rigor while improving clarity and accessibility.

## 📋 Before You Begin

1. **Familiarize yourself** with:
   - [Lean 4 documentation](https://leanprover.github.io/lean4/doc/)
   - [Mathlib4 conventions](https://leanprover-community.github.io/contribute/index.html)
   - The [PROJECT_IMPROVEMENT_PLAN.md](../PROJECT_IMPROVEMENT_PLAN.md) for current priorities

2. **Check existing work**:
   - Review open issues and PRs
   - See progress tracking in PROJECT_IMPROVEMENT_PLAN.md
   - Run `./scripts/check_sorry.py` to see current sorry count

## 🚀 Getting Started

### Setup
```bash
# Clone the repository
git clone https://github.com/jonwashburn/ym-proof.git
cd ym-proof/YangMillsProof

# Install Lean toolchain
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build the project
lake build

# Run tests
lake exe numerical_tests test
```

### Development Workflow
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes following our conventions (see below)
3. Ensure no new sorries: `./scripts/check_sorry.py`
4. Run numerical tests: `lake exe numerical_tests test`
5. Commit with descriptive messages
6. Push and create a PR

## 📝 Coding Conventions

### Lean Style
- **No sorries in main**: All PRs to main must have 0 sorries
- **Explicit types**: Prefer explicit type annotations for clarity
- **Documentation**: All theorems should have docstrings explaining their purpose
- **Naming**: Follow Mathlib conventions (snake_case for theorems, CamelCase for types)

### File Organization
```
YangMillsProof/
├── Core/           # Foundational definitions
├── RG/             # Renormalization group
├── Wilson/         # Wilson action and lattice
├── Gauge/          # Gauge theory infrastructure
├── Numerical/      # Numerical verification
└── Tests/          # Test executables
```

### Placeholder Policy
When implementing incomplete features:
1. Use `sorry` with a comment explaining what's needed
2. Create a tracking issue
3. Reference the issue in your PR

Example:
```lean
theorem my_theorem : P := by
  sorry  -- TODO: Requires monotonicity of f, see issue #123
```

### Numerical Constants
All numerical bounds must:
1. Be stored in `Numerical/Envelope.lean`
2. Have rational enclosures with proofs
3. Be tested by CI

Example:
```lean
def my_constant_envelope : Envelope := {
  lo := 123/100
  hi := 125/100
  nom := 124/100
  pf := by norm_num
}
```

## 🧪 Testing

### Before Submitting
1. **Build clean**: `lake clean && lake build`
2. **Check sorries**: `./scripts/check_sorry.py`
3. **Run tests**: `lake exe numerical_tests test`
4. **Verify envelopes**: `git diff YangMillsProof/Numerical/Envelope.lean`

### CI Checks
Our CI runs automatically on all PRs:
- Builds the project
- Verifies 0 sorries
- Runs numerical tests
- Checks envelope drift

## 📊 Commit Messages

Follow conventional commits:
```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature or theorem
- `fix`: Bug fix or sorry resolution
- `docs`: Documentation only
- `refactor`: Code restructuring
- `test`: Test additions/changes
- `ci`: CI/build changes

Example:
```
feat(RG): Prove g_exact satisfies RG equation

- Add chain rule lemma for sqrt composition
- Simplify using new deriv_sqrt helper
- Remove dependency on rpow

Closes #45
```

## 🏷️ Pull Request Process

1. **Title**: Clear, descriptive title referencing issue numbers
2. **Description**: 
   - Summary of changes
   - Related issues (use "Closes #N")
   - Testing performed
3. **Size**: Keep PRs focused - one logical change per PR
4. **Review**: Address all reviewer comments before merging

### PR Template
```markdown
## Summary
Brief description of changes

## Related Issues
Closes #N

## Changes
- [ ] Change 1
- [ ] Change 2

## Testing
- [ ] Builds without errors
- [ ] No new sorries
- [ ] Numerical tests pass

## Notes
Any additional context
```

## 🐛 Reporting Issues

### Bug Reports
Include:
- Lean version (`lean --version`)
- Minimal reproducible example
- Expected vs actual behavior
- Error messages

### Feature Requests
- Check PROJECT_IMPROVEMENT_PLAN.md first
- Explain the mathematical motivation
- Suggest implementation approach

## 🤝 Code of Conduct

- Be respectful and constructive
- Focus on mathematical correctness
- Welcome questions and learning
- Credit others' contributions

## 📚 Resources

- [Lean Zulip](https://leanprover.zulipchat.com/) - Community help
- [Mathlib4 docs](https://leanprover-community.github.io/mathlib4_docs/)
- [Recognition Science](../recognition-ledger/) - Background material

## ❓ Questions?

- Open a discussion issue
- Ask on Lean Zulip with tag #yang-mills
- Email: [project maintainer]

Thank you for contributing to formal mathematics! 🎉 