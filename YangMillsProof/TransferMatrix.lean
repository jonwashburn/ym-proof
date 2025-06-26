/-
  Transfer Matrix Re-export
  ========================

  This file re-exports the transfer matrix definitions from Continuum/TransferMatrix.lean
-/

import YangMillsProof.Continuum.TransferMatrix

-- Re-export key definitions
export YangMillsProof.Continuum (massGap GaugeLedgerState)

namespace YangMillsProof

open YangMillsProof.Continuum

/-- Predicate stating that Δ is the Yang-Mills mass gap -/
def IsYangMillsMassGap (Δ : ℝ) : Prop :=
  -- The mass gap equals the spectral gap of the transfer matrix
  Δ = massGap ∧
  -- It persists in the continuum limit
  ∃ (Δ_cont : ℝ), Δ_cont = Δ ∧ RG.continuum_gap_exists

end YangMillsProof
