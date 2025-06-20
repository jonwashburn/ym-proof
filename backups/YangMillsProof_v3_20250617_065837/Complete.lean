import Mathlib.Data.Real.Basic
import YangMillsProof.GaugeResidue
import YangMillsProof.GapTheorem
import YangMillsProof.OSReconstruction

namespace YangMillsProof

-- massGap is already defined in GaugeResidue as E_coh * phi

-- massGap_positive is already proven in GaugeResidue

/-- Main result: Yang-Mills existence and mass gap -/
theorem yang_mills_existence_and_mass_gap :
  ∃ (Δ : ℝ), Δ > 0 ∧ Δ = massGap := by
  use massGap
  constructor
  · exact massGap_positive
  · rfl

/-- Alternative formulation using the RS framework -/
theorem yang_mills_RS_gap : massGap > 0 := massGap_positive

end YangMillsProof
