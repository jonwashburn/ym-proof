import ym.Interfaces
import ym.Reflection
import ym.Transfer
import ym.Continuum

/-!
End-to-end pipeline glue: OS → PF gap → persistence → continuum gap.
This module composes the existing adapters and exposes a compact
export plus a tiny example.
-/

namespace YM

/-- End-to-end export from reflection positivity, block positivity family,
    and persistence to a continuum mass gap at rate `γ`. -/
structure EndToEndCert where
  R     : Reflection
  sf    : ScalingFamily
  γ     : ℝ
  hRef  : ReflectionPositivity (sf.μ_at ⟨0⟩) R
  hBlk  : ∀ b : Block, BlockPositivity (sf.μ_at ⟨0⟩) (sf.K_at ⟨0⟩) b
  hPer  : PersistenceCert sf γ

/-- Compact export assembling the pipeline in one step. -/
theorem end_to_end_mass_gap (c : EndToEndCert) : MassGapCont c.γ := by
  -- OS from reflection
  have hOS : OSPositivity (c.sf.μ_at ⟨0⟩) :=
    os_of_reflection (μ := c.sf.μ_at ⟨0⟩) (R := c.R) c.hRef
  -- PF gap from blocks
  have hPF : TransferPFGap (c.sf.μ_at ⟨0⟩) (c.sf.K_at ⟨0⟩) c.γ :=
    pf_gap_of_block_pos (μ := c.sf.μ_at ⟨0⟩) (K := c.sf.K_at ⟨0⟩) c.γ c.hBlk (hirr := trivial)
  -- Persistence
  have hPer : GapPersists c.γ := gap_persists_of_cert (sf := c.sf) (γ := c.γ) c.hPer
  -- Package and finish
  have hGap : MassGap (c.sf.μ_at ⟨0⟩) c.γ := mass_gap_of_OS_PF hOS hPF
  exact mass_gap_continuum (μ := c.sf.μ_at ⟨0⟩) (γ := c.γ) hGap hPer

/-- Tiny toy certificate and export (Prop-level). -/
namespace Examples

private def trivialReflection : Reflection where
  act := id
  involutive := by intro x; rfl

private def trivialSF : ScalingFamily where
  μ_at := fun _ => (default : LatticeMeasure)
  K_at := fun _ => (default : TransferKernel)

private def toyCert : EndToEndCert where
  R := trivialReflection
  sf := trivialSF
  γ := 1
  hRef := trivial
  hBlk := by intro _; trivial
  hPer := by
    -- Provide a basic persistence certificate: γ>0 and uniform PF gap across scales
    refine And.intro (by norm_num) ?h
    intro _; trivial

/-- Toy end-to-end example that compiles with the Prop-level interfaces. -/
theorem toy_end_to_end : MassGapCont 1 :=
  end_to_end_mass_gap toyCert

end Examples

end YM
