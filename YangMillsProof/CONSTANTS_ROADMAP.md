# Recognition Science Constants Roadmap
## Converting Hidden Numbers to Explicit Parameters

This document tracks the conversion of all hard-coded constants in the Recognition Science / Yang-Mills proof to explicit parameters, ensuring mathematical transparency.

---

## PART I: Mandatory for Lean Proof Chain
These items currently appear as literal numerals or `def`s and must be moved to `formal/Parameters/`.

### A. Core Constants
- [ ] **φ : ℝ** (golden ratio)
  - Currently: `def φ : ℝ := (1 + √5)/2`
  - Files: `Foundations/GoldenRatio.lean`
  
- [ ] **E_coh : ℝ** (coherence-quantum energy)  
  - Currently: `def E_coh : ℝ := 0.090`
  - Files: `Foundations/PositiveCost.lean`
  
- [ ] **q73 : ℕ** (plaquette half-quantum)
  - Currently: literal `73`
  - Files: `GaugeResidue/Integer73.lean`
  
- [ ] **λ_rec : ℝ** (recognition length)
  - Currently: various literals
  - Files: TBD

### B. Secondary Constants Used in Proofs
- [ ] **σ_phys : ℝ = 0.18** GeV² (physical string tension)
  - Files: `RecognitionScience/Ledger/FirstPrinciples.lean`
  
- [ ] **β_critical : ℝ = 6.0** (critical coupling)
  - Files: `RecognitionScience/Ledger/FirstPrinciples.lean`
  
- [ ] **a_lattice : ℝ = 0.1** fm (lattice spacing)
  - Files: `RecognitionScience/Ledger/FirstPrinciples.lean`
  
- [ ] **c₆ : ℝ ≈ 7.55** (step-scaling product)
  - Files: `RG/RunningGap.lean`

### C. Derived Definitions (Must Reference Parameters)
- [ ] **massGap** := E_coh * φ
  - Files: `CostSpectrum.lean`
  
- [ ] **minimalGaugeCost** := massGap
  - Files: `CostSpectrum.lean`
  
- [ ] **transferMatrix** entry 1/φ²
  - Files: `TransferMatrix.lean`
  
- [ ] **transferGap** := 1/φ − 1/φ²
  - Files: `TransferMatrix/SpectralGap.lean`
  
- [ ] **gaugeCost(s)** := Σ ... E_coh · φ^{|rung|}
  - Files: `GaugeResidue/Cost.lean`
  
- [ ] **stringTension** := (q73 : ℝ)/1000
  - Files: `GaugeResidue/Integer73.lean`

### D. Numeric Literals to Replace
- [ ] All occurrences of `1.618...`
- [ ] All occurrences of `0.090`
- [ ] All occurrences of `73`
- [ ] All occurrences of `0.18`
- [ ] All occurrences of `6.0`
- [ ] All occurrences of `0.1`
- [ ] All occurrences of `7.55`
- [ ] All occurrences of `1/φ^2` as literal

---

## PART II: Future Theorems (Aspirational)
Once Part I is complete, prove theorems to eliminate each parameter.

### 1. Golden Ratio Uniqueness
- [ ] Prove: `φ * φ = φ + 1` and `1 < φ` ⇒ `φ = (1+√5)/2`
- Target: `Foundations/GoldenRatio.lean`

### 2. Coherence Quantum Derivation
- [ ] Prove: `E_coh = minimal plaquette energy` from Wilson action
- Target: `Ledger/FirstPrinciples.lean` or `Physics/Matching.lean`

### 3. Plaquette Charge from Cohomology
- [ ] Prove: `plaquetteDefect = 73` as obstruction in H³(T⁴,ℤ₃)
- Target: `Topology/ChernWhitney.lean`

### 4. Critical Coupling Relations
- [ ] Derive β_c, σ_phys from lattice scaling
- Target: `RG/ScaleMatching.lean`

### 5. Recognition Length from First Principles
- [ ] Derive λ_rec from Planck + dark-energy occupancy
- Target: `Cosmology/RecognitionLength.lean`

### 6. Step-Scaling Product
- [ ] Prove c₆ from exact RG flow
- Target: `RG/StepScaling.lean`

### 7. Continuum Limit Gap Theorem
- [ ] Prove: `∃ Δ>0, ∀a<a₀, gap(a)≥Δ`
- Target: `RG/ContinuumLimit.lean`

### 8. Reflection Positivity
- [ ] Complete OS reconstruction proofs
- Target: `Measure/ReflectionPositivity.lean`, `OS/Reconstruction.lean`

### 9. BRST Cohomology
- [ ] Prove: BRST cohomology ≅ physical Hilbert space
- Target: `Gauge/BRST/Cohomology.lean`

### 10. Numerical Bounds
- [ ] Convert matches to theorems: `|m_e − calcMass 32| < 5×10^{-4} MeV`
- Target: `Numerics/ParticleMassBounds.lean`

---

## PART III: Implementation Steps

### 1. Create Parameter Infrastructure
```lean
-- formal/Parameters/Constants.lean
namespace RS.Param
constant φ       : ℝ  -- Golden ratio
constant E_coh   : ℝ  -- Coherence quantum
constant q73     : ℕ  -- Plaquette charge
constant λ_rec   : ℝ  -- Recognition length
end RS.Param

-- formal/Parameters/Assumptions.lean
open RS.Param
axiom phi_gt_one      : 1 < φ
axiom phi_eq_root     : φ * φ = φ + 1
axiom E_coh_pos       : 0 < E_coh
axiom q73_eq_73       : (q73 : ℤ) = 73
axiom λ_rec_pos       : 0 < λ_rec
```

### 2. Refactoring Process
1. Add `Parameters/Constants.lean` and `Assumptions.lean`
2. Replace literals module-by-module
3. Update imports as needed
4. Keep `lake build` green at each commit
5. Update this checklist

### 3. Verification Script
Create `scripts/check_literals.py` to grep for disallowed numerals:
```python
#!/usr/bin/env python3
import re
import sys

forbidden = ['0.090', '1.618', '73', '0.18', '6.0', '0.1', '7.55']
# ... implementation ...
```

---

## Current Status
- Started: [DATE]
- Part I Complete: [ ]
- Part II Complete: [ ]
- All sorries removed: [ ]

## Notes
- After Part I: Theory becomes "parametric Yang-Mills with explicit constants"
- After Part II: Theory becomes "parameter-free Yang-Mills from first principles"
- Keep commits atomic for easy review 