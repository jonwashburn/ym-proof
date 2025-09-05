import ym.OSPositivity
import ym.Reflection
import ym.Transfer
import ym.Continuum
import ym.Kato.SimpleEigenStability

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

/-- Builder that assembles a `GapCertificate` from base-scale reflection positivity,
block positivity, and a persistence certificate on a scaling family. This is a
non-quantitative wrapper; quantitative routes can refine `hPF` via
`pf_gap_via_reflection_blocks` when available. -/
def buildGapCertificate (p : PipelineCertificate) : GapCertificate :=
  let μ0 := p.sf.μ_at ⟨0⟩
  let K0 := p.sf.K_at ⟨0⟩
  have hOS : OSPositivity μ0 := os_of_reflection (μ := μ0) (R := p.R) p.hRef
  have hPF : TransferPFGap μ0 K0 p.γ :=
    pf_gap_of_block_pos (μ := μ0) (K := K0) p.γ p.hBlk
  have hPer : GapPersists p.γ := gap_persists_of_cert (sf := p.sf) (γ := p.γ) p.hPer
  { μ := μ0, K := K0, γ := p.γ, hOS := hOS, hPF := hPF, hPer := hPer }

/-- Quantitative export: if `pf_gap_via_reflection_blocks` provides an explicit `γ`,
assemble a `PipelineCertificate` with that `γ` and export the mass gap. -/
theorem pipeline_mass_gap_export_quant
    (p : PipelineCertificate)
    (hQuant : ∃ γ : ℝ, 0 < γ ∧ TransferPFGap (p.sf.μ_at ⟨0⟩) (p.sf.K_at ⟨0⟩) γ)
    : ∃ γ : ℝ, MassGapCont γ := by
  rcases hQuant with ⟨γ, hγpos, hpf⟩
  have hPer : GapPersists γ := gap_persists_of_cert (sf := p.sf) (γ := γ)
    (p := p).hPer
  have hOS : OSPositivity (p.sf.μ_at ⟨0⟩) := os_of_reflection (μ := _) (R := p.R) p.hRef
  have hGap : MassGap (p.sf.μ_at ⟨0⟩) γ := mass_gap_of_OS_PF hOS hpf
  exact ⟨γ, mass_gap_continuum (μ := _) (γ := γ) hGap hPer⟩

/--
Certificate for an explicit, scale-uniform persistence gap `γ0`, together with
base-scale reflection positivity. This allows exporting a continuum mass gap at
the explicit rate `γ0` using only block positivity uniformly across scales.
-/
structure ExplicitPersistenceCertificate where
  R     : Reflection
  sf    : ScalingFamily
  γ0    : ℝ
  hRef0 : ReflectionPositivity (sf.μ_at ⟨0⟩) R
  hBlkU : ∀ s : Scale, ∀ b : Block, BlockPositivity (sf.μ_at s) (sf.K_at s) b
  hγ0   : 0 < γ0

/-- Quantitative final export at an explicit rate `γ0`.
From base-scale reflection positivity and a uniform block-positivity family,
derive a persistence certificate at `γ0`, a base-scale PF gap at `γ0`, and
conclude a continuum mass gap at rate `γ0`.
-/
theorem mass_gap_final_explicit (c : ExplicitPersistenceCertificate) : MassGapCont c.γ0 := by
  -- Persistence from uniform block positivity at explicit γ0
  have hPer : PersistenceCert c.sf c.γ0 :=
    persistence_of_uniform_block_pos (sf := c.sf) (γ := c.γ0) c.hγ0 (by
      intro s b; exact c.hBlkU s b)
  -- Base-scale OS positivity
  have hOS0 : OSPositivity (c.sf.μ_at ⟨0⟩) := os_of_reflection (μ := _) (R := c.R) c.hRef0
  -- Base-scale PF gap at explicit γ0 from block positivity and `0<γ0`
  have hPF0 : TransferPFGap (c.sf.μ_at ⟨0⟩) (c.sf.K_at ⟨0⟩) c.γ0 := by
    -- package `0 < γ0` as `UniformGamma` for the adapter
    have hUG : UniformGamma (c.sf.μ_at ⟨0⟩) (c.sf.K_at ⟨0⟩) c.γ0 := by simpa
    exact pf_gap_of_block_pos_uniform
      (μ := c.sf.μ_at ⟨0⟩) (K := c.sf.K_at ⟨0⟩) (γ := c.γ0)
      (hpos := by intro b; exact c.hBlkU ⟨0⟩ b) (hγ := hUG)
  -- Base-scale lattice mass gap
  have hGap0 : MassGap (c.sf.μ_at ⟨0⟩) c.γ0 := mass_gap_of_OS_PF hOS0 hPF0
  -- Export to the continuum
  exact mass_gap_continuum (μ := c.sf.μ_at ⟨0⟩) (γ := c.γ0) hGap0 (gap_persists_of_cert hPer)

end YM
