import ym.Interfaces
import ym.Reflection
import ym.Transfer
import ym.Continuum
import ym.Adapter.MatrixToTransfer

/-!
Quantitative-style example wiring a base-scale quantitative PF gap into the
pipeline export. Uses the interface-level adapter `pf_gap_via_reflection_blocks`
to obtain `∃ γ > 0`, then applies `pipeline_mass_gap_export_quant`.
-/

namespace YM
namespace ExamplesQuant

def trivialReflection : Reflection where
  act := id
  involutive := by intro x; rfl

def trivialSF : ScalingFamily where
  μ_at := fun _ => (default : LatticeMeasure)
  K_at := fun _ => (default : TransferKernel)

def toyCert : PipelineCertificate where
  R := trivialReflection
  sf := trivialSF
  γ := 1
  hRef := trivial
  hBlk := by intro _; trivial
  hPer := by
    refine And.intro (by norm_num) ?pf
    intro _; trivial

/-- Quantitative end-to-end example: exports `∃ γ, MassGapCont γ`. -/
theorem quant_end_to_end : ∃ γ : ℝ, MassGapCont γ := by
  -- Base-scale quantitative PF gap via reflection + blocks (interface-level adapter).
  have hQuant : ∃ γ : ℝ, 0 < γ ∧
      TransferPFGap (toyCert.sf.μ_at ⟨0⟩) (toyCert.sf.K_at ⟨0⟩) γ := by
    exact pf_gap_via_reflection_blocks
      (μ := toyCert.sf.μ_at ⟨0⟩) (K := toyCert.sf.K_at ⟨0⟩) (R := toyCert.R)
      toyCert.hRef toyCert.hBlk
  -- Compose into the pipeline export.
  exact pipeline_mass_gap_export_quant toyCert hQuant

/-- Alternative quantitative example using the matrix adapter at the base scale. -/
theorem quant_end_to_end_matrix : ∃ γ : ℝ, MassGapCont γ := by
  -- Base-scale quantitative PF gap via a 1×1 toy matrix adapter.
  have hGap : TransferPFGap (toyCert.sf.μ_at ⟨0⟩) (toyCert.sf.K_at ⟨0⟩) 1 := by
    simpa using YM.transfer_gap_of_matrix_gap (A := YM.Examples.toy1x1) (γ := 1)
      (YM.Examples.toy1x1_matrix_gap)
  have hQuant : ∃ γ : ℝ, 0 < γ ∧
      TransferPFGap (toyCert.sf.μ_at ⟨0⟩) (toyCert.sf.K_at ⟨0⟩) γ := by
    exact ⟨1, by norm_num, hGap⟩
  exact pipeline_mass_gap_export_quant toyCert hQuant

end ExamplesQuant
end YM
