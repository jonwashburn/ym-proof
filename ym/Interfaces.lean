import ym.OSPositivity
import ym.Reflection
import ym.Transfer
import ym.Continuum

/-!
YM high-level interfaces: collect assumptions into a certificate and export a grand
mass-gap consequence for the continuum.
-/

namespace YM

/-- Bundles the interface assumptions needed to derive a continuum mass gap. -/
structure GapCertificate where
  μ    : LatticeMeasure
  K    : TransferKernel
  γ    : ℝ
  hOS  : OSPositivity μ
  hPF  : TransferPFGap μ K γ
  hPer : GapPersists γ

/-- Grand export: from the certificate, produce a continuum mass gap. -/
theorem grand_mass_gap_export (c : GapCertificate) : MassGapCont c.γ := by
  have hGap : MassGap c.μ c.γ :=
    mass_gap_of_OS_PF (μ:=c.μ) (K:=c.K) (γ:=c.γ) c.hOS c.hPF
  exact mass_gap_continuum (μ:=c.μ) (γ:=c.γ) hGap c.hPer

/-- Full pipeline export: reflection positivity + block positivity family + persistence
delivers a continuum mass gap at rate `γ`. -/
structure PipelineCertificate where
  R     : Reflection
  sf    : ScalingFamily
  γ     : ℝ
  hRef  : ReflectionPositivity (sf.μ_at ⟨0⟩) R
  hBlk  : ∀ b : Block, BlockPositivity (sf.μ_at ⟨0⟩) (sf.K_at ⟨0⟩) b
  hPer  : PersistenceCert sf γ

/-- Pipeline export theorem, composing the adapters. -/
theorem pipeline_mass_gap_export (p : PipelineCertificate) : MassGapCont p.γ := by
  -- OS from reflection
  have hOS : OSPositivity (sf := p.sf) :=
    os_of_reflection (μ := p.sf.μ_at ⟨0⟩) (R := p.R) p.hRef
  -- PF gap from blocks
  have hPF : TransferPFGap (p.sf.μ_at ⟨0⟩) (p.sf.K_at ⟨0⟩) p.γ :=
    pf_gap_of_block_pos (μ := p.sf.μ_at ⟨0⟩) (K := p.sf.K_at ⟨0⟩) p.γ p.hBlk
  -- Persistence
  have hPer : GapPersists p.γ := gap_persists_of_cert (sf := p.sf) (γ := p.γ) p.hPer
  -- Assemble via GapCertificate
  let c : GapCertificate :=
    { μ := p.sf.μ_at ⟨0⟩, K := p.sf.K_at ⟨0⟩, γ := p.γ, hOS := hOS, hPF := hPF, hPer := hPer }
  exact grand_mass_gap_export c

end YM
