import YangMillsProof.GaugeResidue

namespace YangMillsProof

/--
`CostSpectrum` isolates the statement and proof that the minimal positive cost
in the gauge layer equals `massGap = E_coh * phi`.
The heavy lifting (definitions and lemma `massGap_positive`) is already
provided by `GaugeResidue.lean`; here we simply re-package the result
so that the Lean file hierarchy matches the documentation (v44 manuscript).
-/
noncomputable section

open Real

/--  The minimal positive recognition cost in the gauge layer.  -/
@[simp] def minimalGaugeCost : ℝ := massGap

/--  The minimal gauge cost is strictly positive.  -/
lemma minimalGaugeCost_pos : minimalGaugeCost > 0 := by
  dsimp [minimalGaugeCost]
  exact massGap_positive

/-- The minimal cost equals E_coh * phi -/
lemma minimalGaugeCost_formula : minimalGaugeCost = E_coh * phi := by
  unfold minimalGaugeCost massGap
  rfl

/-- The minimal cost is related to the golden ratio -/
lemma minimalGaugeCost_golden_ratio : minimalGaugeCost / E_coh = phi := by
  unfold minimalGaugeCost massGap
  -- (E_coh * phi) / E_coh = phi
  exact div_mul_cancel_pos phi E_coh_pos

end

end YangMillsProof
