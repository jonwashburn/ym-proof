import Mathlib
import ym.OSPositivity
import ym.Reflection
import ym.Transfer
import ym.Continuum
import ym.Interfaces

/-!
Finite-state lattice/transfer toy model used to assemble an explicit
`ExplicitPersistenceCertificate` at a given positive rate `γ0`.

This keeps the model minimal (Prop-level for blocks), but provides a
single place to construct a uniform block-positivity assumption across
scales so that `mass_gap_final_explicit` can be applied.
-/

namespace YM

/- Trivial reflection used in certificates. -/
def trivialReflection : Reflection where
  act := id
  involutive := by intro x; rfl

/- Constant-in-scale scaling family placeholders. -/
def trivialSF : ScalingFamily where
  μ_at := fun _ => (default : LatticeMeasure)
  K_at := fun _ => (default : TransferKernel)

/-- Uniform block positivity for the trivial model across all scales. -/
theorem trivial_uniform_blocks :
    ∀ s : Scale, ∀ b : Block, BlockPositivity (trivialSF.μ_at s) (trivialSF.K_at s) b := by
  intro _ _; trivial

/-- Build an explicit-persistence certificate at any positive rate `γ0`. -/
def explicitCert (γ0 : ℝ) (hγ0 : 0 < γ0) : ExplicitPersistenceCertificate :=
  { R := trivialReflection
  , sf := trivialSF
  , γ0 := γ0
  , hRef0 := trivial
  , hBlkU := by intro s b; simpa using (trivial_uniform_blocks s b)
  , hγ0 := hγ0 }

/-- A tiny quantitative example: exports a continuum mass gap at rate `γ0`. -/
theorem quantitative_final_example {γ0 : ℝ} (hγ0 : 0 < γ0) : MassGapCont γ0 :=
  mass_gap_final_explicit (explicitCert γ0 hγ0)

end YM


