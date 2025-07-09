# Continuous Integration Setup

This repository includes a comprehensive CI system that automatically verifies the Yang-Mills proof integrity.

## GitHub Actions Workflow

### Main CI Workflow (`.github/workflows/ci.yml`)

The CI workflow runs on every push and pull request to the main branch:

1. **Environment Setup**
   - Installs Lean 4 via `elan` (Lean version manager)
   - Gets dependencies with `lake update`

2. **Build Verification**
   - Runs `lake build` to compile the entire project
   - Ensures all Lean files elaborate successfully

3. **Axiom-Free Verification**
   - Runs `verify_no_axioms.sh` to confirm no axioms are declared
   - Checks all `.lean` files for `axiom` declarations

4. **Sorry-Free Verification**
   - Runs `verify_no_sorries.sh` to confirm no sorry statements exist
   - Distinguishes between actual sorries and the word in comments

5. **Status Reporting**
   - Generates verification summary
   - Updates GitHub status badges

## Status Badges

The repository displays real-time status via GitHub badges:

- **Build Status**: Shows if `lake build` succeeds
- **Axiom-Free**: Shows if no axioms are declared
- **Sorry-Free**: Shows if no sorry statements exist
- **Lean Version**: Shows Lean 4.12 compliance

## Local CI Testing

### `ci_status.sh` Script

Run the full CI check locally:

```bash
./ci_status.sh
```

This script:
- Builds the project
- Verifies axiom-free status
- Verifies sorry-free status
- Provides detailed status summary

### Individual Checks

Run specific verification scripts:

```bash
# Check for axioms
./verify_no_axioms.sh

# Check for sorries
./verify_no_sorries.sh

# Build only
lake build
```

## CI Outputs

### Success Output
```
🎯 OVERALL STATUS: ✅ ALL CHECKS PASSED
🏆 Yang-Mills proof is complete and formally verified!
```

### Failure Output
```
🔴 OVERALL STATUS: ❌ SOME CHECKS FAILED
🔧 Please fix the issues above
```

## Badge URLs

The badges link to:
- Build badge: GitHub Actions workflow page
- Axiom-Free badge: GitHub Actions workflow page
- Sorry-Free badge: GitHub Actions workflow page
- Lean version badge: Lean prover website

## Maintenance

The CI system automatically:
- Runs on every commit to main
- Validates all pull requests
- Updates status badges
- Provides detailed logs

No manual intervention required - the CI system maintains itself and provides continuous verification of the proof's integrity. 