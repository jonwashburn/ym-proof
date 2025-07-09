# Continuum Limit Code Mapping

Due to repository restructuring, the Layer-4 (Continuum Limit) code mentioned in documentation now resides in multiple locations. This document provides the exact mapping.

## Original Structure → Current Location

### Core Continuum Limit Theorems

| Original Reference | Current Location | Key Theorem |
|-------------------|------------------|-------------|
| `Stage4_ContinuumLimit/ContinuumLimit.lean` | `RG/ContinuumLimit.lean` | `continuum_limit_exists` (line 134) |
| `Stage4_ContinuumLimit/MassGapPersistence.lean` | `RG/ContinuumLimit.lean` | `gap_persistence` (line 138) |
| `Stage4_ContinuumLimit/InductiveLimit.lean` | `Continuum/Continuum.lean` | `continuum_hilbert_space` (line 107) |
| `Stage4_ContinuumLimit/TransferMatrix.lean` | `Continuum/TransferMatrix.lean` | Transfer matrix continuity |

### Key Theorems and Their Locations

1. **Continuum limit exists**: `RG.ContinuumLimit.continuum_limit_exists`
   - Proves the lattice theory has a well-defined continuum limit

2. **Gap persistence**: `RG.ContinuumLimit.gap_persistence`  
   - Shows the mass gap Δ > 0 persists as lattice spacing a → 0

3. **Spectral gap persistence**: `Continuum.Continuum.spectral_gap_persistence` (line 70)
   - Alternative formulation using spectral theory

4. **Lattice-continuum correspondence**: `Continuum.WilsonCorrespondence.lattice_continuum_limit` (line 211)
   - Establishes the mathematical bridge between lattice and continuum

## Import Path Updates

Replace:
```lean
import Stage4_ContinuumLimit.MassGapPersistence
```

With:
```lean
import RG.ContinuumLimit
import Continuum.Continuum
```

## Verification

To verify the continuum limit is properly implemented, check:
- `lake build RG.ContinuumLimit` compiles without errors
- `#check RG.ContinuumLimit.gap_persistence` succeeds
- `#check Continuum.Continuum.spectral_gap_persistence` succeeds 