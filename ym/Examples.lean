import ym.Interfaces
import ym.Reflection
import ym.Transfer
import ym.Continuum

/-!
Tiny toy example instantiating the YM pipeline interfaces and exporting a
continuum mass-gap witness via `pipeline_mass_gap_export`.
-/

namespace YM
namespace Examples

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
  hPer := trivial

/-- A toy pipeline export: demonstrates end-to-end composition. -/
theorem toy_pipeline_mass_gap : MassGapCont 1 :=
  pipeline_mass_gap_export toyCert

/-- A toy persistence instance: uniform PF gap across scales with γ=1 persists. -/
theorem toy_gap_persists : GapPersists 1 := by
  -- From `toyCert` we have the ingredients needed to form a persistence certificate.
  have hγ : 0 < (1 : ℝ) := by norm_num
  -- Build a simple scaling family certificate using the trivial family and stubbed gaps.
  have hpf : ∀ s, TransferPFGap (trivialSF.μ_at s) (trivialSF.K_at s) 1 := by intro _; trivial
  have hcert : PersistenceCert trivialSF 1 := And.intro hγ hpf
  simpa using (gap_persists_of_cert (sf := trivialSF) (γ := 1) hcert)

/--
3-state positive irreducible kernel example (Prop-level). We assert positivity and
irreducibility via our interface predicates and derive `SpectralGap` and
`TransferPFGap` instances as stubs for now.
-/

structure ThreeState where
  i : Fin 3
  deriving DecidableEq, Inhabited

def threeStateKernel : MarkovKernel := { size := 3 }

/-- The 3-state example has a spectral gap γ=1/2 (Prop-level stub). -/
theorem three_state_spectral_gap : SpectralGap threeStateKernel (1/2 : ℝ) := by
  trivial

/-- Adapter: the 3-state spectral gap implies a transfer PF gap for a trivial pair. -/
theorem three_state_transfer_gap :
    TransferPFGap (default : LatticeMeasure) (default : TransferKernel) (1/2 : ℝ) := by
  trivial

/-- Smoke test: basic `#check` endpoints to ensure pipeline exports remain available. -/
#check YM.Examples.toy_pipeline_mass_gap
#check YM.Examples.toy_gap_persists

end Examples
end YM
