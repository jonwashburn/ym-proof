import Stage0_RS_Foundation.ActivityCost

namespace YangMillsProof.Stage0_RS_Foundation

open YangMillsProof.Stage0_RS_Foundation

/-- Non-vacuum states have positive activity cost -/
theorem non_vacuum_positive_cost (S : LedgerState) (h : S ≠ vacuumState) :
  ∃ ε > 0, activityCost S ≥ ε := by
  sorry

/-- Vacuum state has zero activity cost -/
theorem vacuum_zero_cost (S : LedgerState) :
  activityCost S = 0 → S = vacuumState := by
  sorry

end YangMillsProof.Stage0_RS_Foundation
