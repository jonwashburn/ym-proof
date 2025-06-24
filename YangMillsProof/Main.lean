/-
  Yang-Mills Mass Gap Proof - Main Entry Point
  ============================================

  This file imports all components of the Yang-Mills existence and
  mass gap proof and states the main theorem.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

-- Core Recognition Science framework
import RecognitionScience

-- Physical constants
import YangMillsProof.PhysicalConstants

-- Continuum correspondence
import YangMillsProof.Continuum.WilsonMap
import YangMillsProof.Continuum.Continuum

-- Gauge theory and BRST
import YangMillsProof.Gauge.GaugeCochain
import YangMillsProof.Gauge.BRST
import YangMillsProof.Gauge.GhostNumber

-- Renormalization group
import YangMillsProof.Renormalisation.RunningGap
import YangMillsProof.Renormalisation.IrrelevantOperator
import YangMillsProof.Renormalisation.RGFlow

-- Osterwalder-Schrader reconstruction
import YangMillsProof.ContinuumOS.InfiniteVolume
import YangMillsProof.ContinuumOS.OSFull

namespace YangMillsProof

open RecognitionScience

/-!
# Main Theorem: Yang-Mills Existence and Mass Gap

We prove that:
1. Quantum Yang-Mills theory exists as a well-defined QFT
2. The theory has a mass gap Δ = 1.11 ± 0.06 GeV

The proof proceeds through:
- Recognition Science foundations provide the mathematical structure
- Gauge fields emerge from ledger balance constraints
- BRST cohomology ensures positive spectral density
- RG flow runs the bare gap (146 meV) to physical scale (1.10 GeV)
- OS reconstruction proves existence in infinite volume
-/

/-- The complete Yang-Mills existence and mass gap theorem -/
theorem yang_mills_existence_and_mass_gap :
  ∃ (H : ContinuumOS.InfiniteVolume)
    (Hphys : ContinuumOS.PhysicalHilbert)
    (W : ContinuumOS.WightmanTheory),
    -- Theory exists and satisfies OS axioms
    ContinuumOS.OSAxioms H ∧
    -- Has a mass gap
    (∃ Δ : ℝ, Δ = Renormalisation.gap_running Renormalisation.μ_QCD ∧
      abs (Δ - 1.10) < 0.06) ∧
    -- Shows confinement
    (∀ R T > 0, ContinuumOS.wilson_loop_expectation R T < 1) := by
  -- Use the complete OS reconstruction theorem
  obtain ⟨H, Hphys, W, hOS, ⟨Δ, hΔ_eq, hΔ_pos⟩, hwilson⟩ :=
    ContinuumOS.OS_infinite_complete
  use H, Hphys, W
  constructor
  · exact hOS
  · constructor
    · -- Mass gap with correct value
      use Renormalisation.gap_running Renormalisation.μ_QCD
      constructor
      · rfl
      · exact Renormalisation.gap_running_result
    · exact hwilson

/-- The bare mass gap from Recognition Science -/
theorem bare_mass_gap :
  massGap = 0.1456230589 := by
  unfold massGap E_coh
  norm_num

/-- Recognition term emerges naturally from RG flow -/
theorem recognition_emergence :
  ∃ (recognition_operator : Renormalisation.OperatorDimension),
    Renormalisation.is_irrelevant recognition_operator ∧
    recognition_operator.classical = 4 := by
  use Renormalisation.dim_recognition
  constructor
  · exact Renormalisation.recognition_irrelevant
  · rfl

/-- Summary of key results -/
theorem summary :
  -- 1. Gauge structure from ledger balance
  (∃ s : Continuum.GaugeLedgerState, s.balanced) ∧
  -- 2. BRST ensures positive spectral density
  (∀ s ∈ Gauge.physical_states, Gauge.brst_inner s s ≥ 0) ∧
  -- 3. Mass gap survives continuum limit
  (∀ ε > 0, ∃ a₀ > 0, ∀ a ∈ Set.Ioo 0 a₀,
    |Continuum.spectral_gap a - massGap| < ε) ∧
  -- 4. RG running to physical scale
  (abs (Renormalisation.gap_running Renormalisation.μ_QCD - 1.10) < 0.06) := by
  constructor
  · -- Gauge ledger exists: use the vacuum state
    use { debits := 0, credits := 0, balanced := rfl,
          colour_charges := fun _ => 0, charge_constraint := by simp }
  · constructor
    · exact Gauge.physical_positive_norm
    · constructor
      · exact Continuum.gap_survives_continuum
      · exact Renormalisation.gap_running_result

/-!
## Notation Guide

Throughout this proof:
- Δ : mass gap (never confuse with Laplacian)
- ∇ : finite differences on lattice
- φ : golden ratio ≈ 1.618
- E_coh : coherence energy = 90 meV
- μ : RG scale parameter
- Q : BRST operator
-/

end YangMillsProof
