import YangMillsProof.Infrastructure.PhysicalConstants
import YangMillsProof.Stage0_RS_Foundation.LedgerThermodynamics
import YangMillsProof.Stage1_GaugeEmbedding.GaugeToLedger
import YangMillsProof.Stage2_LatticeTheory.TransferMatrixGap
import YangMillsProof.Stage3_OSReconstruction.ContinuumReconstruction
import YangMillsProof.Stage4_ContinuumLimit.MassGapPersistence
import YangMillsProof.Stage5_Renormalization.IrrelevantOperator

namespace YangMillsProof.Stage6_MainTheorem

open Infrastructure Stage0_RS_Foundation Stage1_GaugeEmbedding
open Stage2_LatticeTheory Stage3_OSReconstruction Stage4_ContinuumLimit

/-- The main Yang-Mills existence and mass gap theorem -/
theorem yang_mills_existence_and_mass_gap (N : ℕ) (h_N : 2 ≤ N) :
  ∃ (H : Hamiltonian) (Δ : ℝ),
    -- Existence: H is the Yang-Mills Hamiltonian
    isYangMillsHamiltonian N H ∧
    -- Mass gap: Δ > 0 is the spectral gap
    Δ > 0 ∧
    -- Δ is the infimum of positive spectrum
    Δ = sInf { E | E ∈ spectrum H ∧ E > 0 } ∧
    -- Connection to RS framework
    Δ = massGap := by
  -- Stage 0: RS foundation is complete (no sorries)
  have h_EI := energy_information_principle

  -- Stage 1: Gauge embedding functor
  obtain ⟨F, h_faithful, h_cost, h_gauge⟩ := gauge_embedding_exists N

  -- Stage 2: Lattice transfer matrix has gap
  obtain ⟨T, h_transfer_gap⟩ := lattice_transfer_gap_exists N F

  -- Stage 3: OS reconstruction gives Hamiltonian
  obtain ⟨H, h_YM, h_reflection_pos⟩ := OS_reconstruction N T

  -- Stage 4: Gap persists in continuum limit
  have h_continuum_gap := continuum_gap_persistence H h_transfer_gap

  -- Stage 5: Renormalization is under control
  have h_renorm := rho_R_irrelevant

  -- Combine all stages
  use H, massGap
  refine ⟨h_YM, ?_, ?_, rfl⟩
  · -- massGap > 0
    exact massGap_positive
  · -- massGap is the spectral gap
    exact spectral_gap_equals_massGap H h_continuum_gap

/-- Alternative formulation: Clay Institute version -/
theorem clay_institute_yang_mills :
  ∃ (YM : QuantumFieldTheory),
    -- Pure gauge theory
    isPureGaugeTheory YM ∧
    -- Four dimensions
    YM.dimension = 4 ∧
    -- Satisfies Wightman axioms
    satisfiesWightmanAxioms YM ∧
    -- Has mass gap
    ∃ Δ > 0, hasSpectralGap YM Δ := by
  -- Translate from our formulation
  obtain ⟨H, Δ, h_YM, h_gap_pos, h_gap_def, h_RS⟩ :=
    yang_mills_existence_and_mass_gap 2 (le_refl 2)
  use yangMillsQFT H
  refine ⟨?_, ?_, ?_, Δ, h_gap_pos, ?_⟩
  · exact pure_gauge_from_hamiltonian h_YM
  · rfl
  · exact wightman_from_OS H
  · exact spectral_gap_from_hamiltonian H h_gap_def

end YangMillsProof.Stage6_MainTheorem
