import Stage0_RS_Foundation.ActivityCost

namespace YangMillsProof.Stage0_RS_Foundation

open YangMillsProof.Stage0_RS_Foundation

/-- Non-vacuum states have positive activity cost -/
theorem non_vacuum_positive_cost (S : LedgerState) (h : S ≠ vacuumState) :
  ∃ ε > 0, activityCost S ≥ ε := by
  -- We'll show that activityCost S ≥ phi, and phi > 0
  use phi
  constructor
  · -- phi > 0
    exact phi_pos
  · -- activityCost S ≥ phi
    -- activityCost S = (S.fst + S.snd) * phi
    unfold activityCost
    -- Since S ≠ vacuumState, we have S.fst + S.snd ≠ 0
    have h_sum_nonzero : S.fst + S.snd ≠ 0 := by
      intro h_sum_zero
      have h_vacuum : S = vacuumState := by
        rw [vacuumState]
        have h_both_zero := Nat.add_eq_zero_iff.mp h_sum_zero
        exact Prod.ext h_both_zero.1 h_both_zero.2
      exact h h_vacuum
    -- Since S.fst + S.snd ≠ 0 and they're natural numbers, S.fst + S.snd ≥ 1
    have h_sum_pos : (S.fst + S.snd : ℝ) ≥ 1 := by
      have h_pos : (S.fst + S.snd : ℕ) ≥ 1 := Nat.one_le_iff_ne_zero.mpr h_sum_nonzero
      exact_mod_cast h_pos
    -- Therefore (S.fst + S.snd) * phi ≥ 1 * phi = phi
    calc (S.fst + S.snd : ℝ) * phi
      ≥ 1 * phi := by exact mul_le_mul_of_nonneg_right h_sum_pos (le_of_lt phi_pos)
      _ = phi := by ring

/-- Vacuum state has zero activity cost -/
theorem vacuum_zero_cost (S : LedgerState) :
  activityCost S = 0 → S = vacuumState := by
  -- This is exactly the forward direction of activity_zero_iff_vacuum
  exact (activity_zero_iff_vacuum S).mp

end YangMillsProof.Stage0_RS_Foundation
