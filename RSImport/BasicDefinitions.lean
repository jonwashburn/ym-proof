/-
  RSImport Basic Definitions
  ==========================

  This file provides the basic Recognition Science definitions and theorems
  that are used throughout the Yang-Mills proof project.

  It imports from the actual Recognition Science foundation and re-exports
  the necessary definitions with the expected names.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Topology.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace RSImport

-- Direct definition of phi without external imports
noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

-- Local proof of phi_pos
theorem phi_pos : 0 < phi := by
  unfold phi
  have h_sqrt5_pos : 0 < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num : 0 < 5)
  linarith

-- Local proof of phi_gt_one
theorem phi_gt_one : 1 < phi := by
  unfold phi
  have h_sqrt5_gt1 : 1 < Real.sqrt 5 := Real.one_lt_sqrt_iff.mpr (by norm_num : 1 < 5)
  linarith

-- Ledger state structure
structure LedgerEntry where
  debit : ℝ
  credit : ℝ
  debit_nonneg : 0 ≤ debit
  credit_nonneg : 0 ≤ credit

structure LedgerState where
  entries : ℕ → LedgerEntry

-- Vacuum state
def vacuumState : LedgerState where
  entries := fun _ => {
    debit := 0
    credit := 0
    debit_nonneg := le_refl 0
    credit_nonneg := le_refl 0
  }

-- Vacuum state property
theorem vacuumState_entries : ∀ n, (vacuumState.entries n).debit = 0 ∧ (vacuumState.entries n).credit = 0 := by
  intro n
  simp [vacuumState]

-- Activity cost functional
noncomputable def activityCost (S : LedgerState) : ℝ :=
  ∑' n, ((S.entries n).debit + (S.entries n).credit) * phi^(n+1)

-- Activity cost is non-negative
theorem activity_nonneg (S : LedgerState) : 0 ≤ activityCost S := by
  unfold activityCost
  apply tsum_nonneg
  intro n
  apply mul_nonneg
  · apply add_nonneg
    · exact (S.entries n).debit_nonneg
    · exact (S.entries n).credit_nonneg
  · exact pow_nonneg (le_of_lt phi_pos) (n+1)

-- Activity cost is zero iff vacuum state
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
    constructor
    · exact (h_entries_zero n).1
    · exact (h_entries_zero n).2
  · -- Backward direction: if S is vacuum, then activity is zero
    intro h_vacuum
    rw [h_vacuum]
    unfold activityCost
    simp only [vacuumState_entries, zero_add, mul_zero]
    exact tsum_zero

-- Balance predicate
def isBalanced (S : LedgerState) : Prop :=
  ∀ n, (S.entries n).debit = (S.entries n).credit

-- Zero cost functional (for compatibility)
def zeroCostFunctional (S : LedgerState) : ℝ := activityCost S

-- Cost zero iff vacuum (for compatibility)
theorem cost_zero_iff_vacuum (S : LedgerState) :
  zeroCostFunctional S = 0 ↔ S = vacuumState := by
  unfold zeroCostFunctional
  exact activity_zero_iff_vacuum S

end RSImport
