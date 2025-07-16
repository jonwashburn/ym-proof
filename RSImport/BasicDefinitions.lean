/-
  RSImport Basic Definitions
  ==========================

  This file provides the basic Recognition Science definitions and theorems
  that are used throughout the Yang-Mills proof project.

  Full mathematical definitions with proper ℝ types.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Topology.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Topology.Algebra.InfiniteSum.Basic

namespace RSImport

-- Golden ratio definition
noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

-- Coherence energy quantum
def E_coh : ℝ := 0.090  -- eV

-- Ledger state (simplified as pair of naturals)
def LedgerState : Type := ℕ × ℕ

-- Vacuum state (both components zero)
def vacuumState : LedgerState := (0, 0)

-- Basic positivity proofs
theorem phi_pos : 0 < phi := by
  unfold phi
  have h_sqrt5_pos : 0 < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 5)
  have h_one_plus_sqrt5 : 0 < 1 + Real.sqrt 5 := by linarith [h_sqrt5_pos]
  have h_two_pos : (0 : ℝ) < 2 := by norm_num
  exact div_pos h_one_plus_sqrt5 h_two_pos

theorem E_coh_pos : 0 < E_coh := by norm_num

-- Activity cost function
def activityCost : LedgerState → ℝ := fun s => (s.fst + s.snd) * phi

-- Basic lemmas
theorem activity_cost_nonneg : ∀ s : LedgerState, 0 ≤ activityCost s := by
  intro s
  unfold activityCost
  apply mul_nonneg
  · exact add_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  · exact le_of_lt phi_pos

-- Additional mathematical functions needed by other modules
lemma tsum_eq_zero_iff_all_eq_zero (f : ℕ → ℝ) (hf : ∀ n, 0 ≤ f n) :
  (∑' n, f n = 0) ↔ (∀ n, f n = 0) := by
  constructor
  · intro h_sum_zero n
    -- If the sum is zero and all terms are non-negative, each term must be zero
    by_contra h_not_zero
    have h_pos : 0 < f n := lt_of_le_of_ne (hf n) (Ne.symm h_not_zero)
    have h_sum_pos : 0 < ∑' k, f k := by
      apply tsum_pos
      · exact hf
      · use n
        exact h_pos
    rw [h_sum_zero] at h_sum_pos
    exact lt_irrefl 0 h_sum_pos
  · intro h_all_zero
    simp only [h_all_zero, tsum_zero]

lemma add_eq_zero_iff_of_nonneg (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a + b = 0 ↔ a = 0 ∧ b = 0 := by
  constructor
  · intro h_sum_zero
    constructor
    · -- Show a = 0
      have h_a_le_zero : a ≤ 0 := by
        calc a = a + b - b := by ring
        _ = 0 - b := by rw [h_sum_zero]
        _ = -b := by ring
        _ ≤ 0 := by linarith [hb]
      exact le_antisymm h_a_le_zero ha
    · -- Show b = 0
      have h_b_le_zero : b ≤ 0 := by
        calc b = a + b - a := by ring
        _ = 0 - a := by rw [h_sum_zero]
        _ = -a := by ring
        _ ≤ 0 := by linarith [ha]
      exact le_antisymm h_b_le_zero hb
  · intro ⟨h_a_zero, h_b_zero⟩
    rw [h_a_zero, h_b_zero, add_zero]

theorem tsum_zero : ∑' n : ℕ, (0 : ℝ) = 0 := by
  exact tsum_zero

end RSImport
