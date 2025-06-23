import YangMillsProof.RSImport.BasicDefinitions

namespace YangMillsProof.Stage0_RS_Foundation

open RSImport

/-- Activity cost functional measuring total ledger activity -/
noncomputable def activityCost (S : LedgerState) : ℝ :=
  ∑' n, ((S.entries n).debit + (S.entries n).credit) * phi^(n+1)

/-- Activity cost is non-negative -/
lemma activity_nonneg (S : LedgerState) : 0 ≤ activityCost S := by
  unfold activityCost
  apply tsum_nonneg
  intro n
  apply mul_nonneg
  · apply add_nonneg
    · exact (S.entries n).debit_nonneg
    · exact (S.entries n).credit_nonneg
  · exact pow_nonneg (le_of_lt phi_pos) (n+1)

/-- Activity cost is zero iff all entries are zero -/
theorem activity_zero_iff_vacuum (S : LedgerState) :
  activityCost S = 0 ↔ S = vacuumState := by
  constructor
  · -- Forward direction: if activity is zero, then S is vacuum
    intro h_zero
    unfold activityCost at h_zero
    -- Since all terms are non-negative and sum is zero, each term must be zero
    have h_all_zero : ∀ n, ((S.entries n).debit + (S.entries n).credit) * phi^(n+1) = 0 := by
      intro n
      apply tsum_eq_zero_iff_all_eq_zero at h_zero
      · exact h_zero n
      · intro m
        apply mul_nonneg
        · apply add_nonneg
          · exact (S.entries m).debit_nonneg
          · exact (S.entries m).credit_nonneg
        · exact pow_nonneg (le_of_lt phi_pos) (m+1)
    -- Since phi^(n+1) > 0, we must have debit + credit = 0 for all n
    have h_entries_zero : ∀ n, (S.entries n).debit = 0 ∧ (S.entries n).credit = 0 := by
      intro n
      have h_phi_pos : 0 < phi^(n+1) := pow_pos phi_pos (n+1)
      have h_prod_zero := h_all_zero n
      rw [mul_eq_zero] at h_prod_zero
      cases h_prod_zero with
      | inl h_sum =>
        have h_debit_nonneg := (S.entries n).debit_nonneg
        have h_credit_nonneg := (S.entries n).credit_nonneg
        have h_sum_nonneg : 0 ≤ (S.entries n).debit + (S.entries n).credit :=
          add_nonneg h_debit_nonneg h_credit_nonneg
        have h_both_zero := le_antisymm (le_of_eq h_sum.symm) h_sum_nonneg
        rw [add_eq_zero_iff_of_nonneg h_debit_nonneg h_credit_nonneg] at h_both_zero
        exact h_both_zero
      | inr h_phi =>
        -- This case is impossible since phi^(n+1) > 0
        exact absurd h_phi (ne_of_gt h_phi_pos)
    -- Now we can show S = vacuumState
    ext n
    · exact (h_entries_zero n).1
    · exact (h_entries_zero n).2
  · -- Backward direction: if S is vacuum, then activity is zero
    intro h_vacuum
    rw [h_vacuum]
    unfold activityCost
    simp only [vacuumState_entries, zero_add, mul_zero]
    exact tsum_zero

end YangMillsProof.Stage0_RS_Foundation
