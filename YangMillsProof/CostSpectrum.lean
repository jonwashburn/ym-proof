import YangMillsProof.GaugeResidue

open Real

namespace YangMillsProof

/--  The minimal positive recognition cost in the gauge layer.  -/
@[simp] noncomputable def minimalGaugeCost : ℝ := massGap

/--  The minimal gauge cost is strictly positive.  -/
lemma minimalGaugeCost_pos : minimalGaugeCost > 0 := by
  dsimp [minimalGaugeCost]
  exact massGap_positive

/-- The minimal cost equals E_coh * phi -/
lemma minimalGaugeCost_formula : minimalGaugeCost = RSImport.E_coh * RSImport.phi := by
  unfold minimalGaugeCost massGap
  rfl

/-- The minimal cost is related to the golden ratio -/
lemma minimalGaugeCost_golden_ratio : minimalGaugeCost / RSImport.E_coh = RSImport.phi := by
  rw [minimalGaugeCost_formula]
  -- (E_coh * phi) / E_coh = phi
  have h : RSImport.E_coh ≠ 0 := ne_of_gt E_coh_pos
  field_simp [h]

end YangMillsProof
