/-
Copyright (c) 2024 YangMillsProof Authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: YangMillsProof Contributors
-/
import Mathlib.Data.Real.Pi
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Numerical Constants for Yang-Mills Proof

This file centralizes all numerical constants and bounds used throughout the proof.
It serves as the single source of truth for:
- π bounds
- Logarithm values
- Beta function constants
- Other numerical approximations

All other files should import this module rather than defining their own constants.
-/

namespace YangMillsProof.Numerical.Constants

open Real

/-! ## π estimates -/

/-- Lower bound for π -/
lemma pi_lower : (π : ℝ) > 3.1415 := by
  have := Real.pi_gt_31415
  linarith

/-- Upper bound for π -/
lemma pi_upper : (π : ℝ) < 3.1416 := by
  have := Real.pi_lt_31416
  linarith

/-- Convenient bound: π > 22/7 -/
lemma pi_gt_22_7 : (π : ℝ) > 22 / 7 := Real.pi_gt_d0

/-- Convenient bound: 3 < π -/
lemma three_lt_pi : (3 : ℝ) < π := Real.three_lt_pi

/-! ## Logarithm table -/

/-- Bounds for log 2 -/
@[simp]
lemma log_two_bounds : 0.6931 < log 2 ∧ log 2 < 0.6932 := by
  constructor
  · calc 0.6931 < log (exp 0.6931) := by norm_num
      _ = 0.6931 := log_exp 0.6931
      _ < log 2 := by
        apply log_lt_log
        · norm_num
        · norm_num
  · calc log 2 < log (exp 0.6932) := by
        apply log_lt_log
        · norm_num
        · norm_num
      _ = 0.6932 := log_exp 0.6932

/-- Lower bound for log 2 -/
lemma log_two_lower : 0.6931 < log 2 := log_two_bounds.1

/-- Upper bound for log 2 -/
lemma log_two_upper : log 2 < 0.6932 := log_two_bounds.2

/-- Bounds for log 4 -/
@[simp]
lemma log_four_bounds : 1.3862 < log 4 ∧ log 4 < 1.3864 := by
  have h2 := log_two_bounds
  rw [← log_pow (by norm_num : (0:ℝ) < 2)]
  constructor
  · calc 1.3862 = 2 * 0.6931 := by norm_num
      _ < 2 * log 2 := by linarith [h2.1]
  · calc 2 * log 2 < 2 * 0.6932 := by linarith [h2.2]
      _ = 1.3864 := by norm_num

/-- Bounds for log 8 -/
@[simp]
lemma log_eight_bounds : 2.0793 < log 8 ∧ log 8 < 2.0796 := by
  have h2 := log_two_bounds
  rw [← log_pow (by norm_num : (0:ℝ) < 2)]
  constructor
  · calc 2.0793 = 3 * 0.6931 := by norm_num
      _ < 3 * log 2 := by linarith [h2.1]
  · calc 3 * log 2 < 3 * 0.6932 := by linarith [h2.2]
      _ = 2.0796 := by norm_num

/-- Bounds for log 64 -/
@[simp]
lemma log_64_bounds : 4.1586 < log 64 ∧ log 64 < 4.1592 := by
  have h2 := log_two_bounds
  rw [← log_pow (by norm_num : (0:ℝ) < 2)]
  constructor
  · calc 4.1586 = 6 * 0.6931 := by norm_num
      _ < 6 * log 2 := by linarith [h2.1]
  · calc 6 * log 2 < 6 * 0.6932 := by linarith [h2.2]
      _ = 4.1592 := by norm_num

/-- Bounds for log 512 -/
@[simp]
lemma log_512_bounds : 6.2379 < log 512 ∧ log 512 < 6.2388 := by
  have h2 := log_two_bounds
  rw [← log_pow (by norm_num : (0:ℝ) < 2)]
  constructor
  · calc 6.2379 = 9 * 0.6931 := by norm_num
      _ < 9 * log 2 := by linarith [h2.1]
  · calc 9 * log 2 < 9 * 0.6932 := by linarith [h2.2]
      _ = 6.2388 := by norm_num

/-- Bounds for log 4096 -/
@[simp]
lemma log_4096_bounds : 8.3172 < log 4096 ∧ log 4096 < 8.3184 := by
  have h2 := log_two_bounds
  rw [← log_pow (by norm_num : (0:ℝ) < 2)]
  constructor
  · calc 8.3172 = 12 * 0.6931 := by norm_num
      _ < 12 * log 2 := by linarith [h2.1]
  · calc 12 * log 2 < 12 * 0.6932 := by linarith [h2.2]
      _ = 8.3184 := by norm_num

/-- Bounds for log 32768 -/
@[simp]
lemma log_32768_bounds : 10.3965 < log 32768 ∧ log 32768 < 10.398 := by
  have h2 := log_two_bounds
  rw [← log_pow (by norm_num : (0:ℝ) < 2)]
  constructor
  · calc 10.3965 = 15 * 0.6931 := by norm_num
      _ < 15 * log 2 := by linarith [h2.1]
  · calc 15 * log 2 < 15 * 0.6932 := by linarith [h2.2]
      _ = 10.398 := by norm_num

/-- Helper: log(2^n) = n * log 2 -/
lemma log_pow_two (n : ℕ) : log (2^n) = n * log 2 := by
  cases' n with n
  · simp
  · rw [log_pow (by norm_num : (0:ℝ) < 2)]

/-! ## Beta-function constants -/

/-- One-loop beta function coefficient for SU(3) Yang-Mills -/
def b₀ : ℝ := 11 / (3 * 16 * π^2)

/-- Lower bound for b₀ -/
lemma b₀_lower : 0.0232 < b₀ := by
  unfold b₀
  have h_pi := pi_upper
  calc 0.0232 < 11 / (3 * 16 * (3.1416)^2) := by norm_num
    _ < 11 / (3 * 16 * π^2) := by
      apply div_lt_div_of_lt_left
      · norm_num
      · apply mul_pos; apply mul_pos; norm_num; exact sq_pos_of_pos pi_pos
      · apply mul_lt_mul_of_pos_left
        apply mul_lt_mul_of_pos_left
        apply sq_lt_sq'
        · linarith [pi_pos]
        · exact h_pi
        · norm_num
        · norm_num

/-- Upper bound for b₀ -/
lemma b₀_upper : b₀ < 0.0234 := by
  unfold b₀
  have h_pi := pi_lower
  calc 11 / (3 * 16 * π^2) < 11 / (3 * 16 * (3.1415)^2) := by
      apply div_lt_div_of_lt_left
      · norm_num
      · apply mul_pos; apply mul_pos; norm_num; norm_num
      · apply mul_lt_mul_of_pos_left
        apply mul_lt_mul_of_pos_left
        apply sq_lt_sq'
        · norm_num
        · linarith
        · norm_num
        · norm_num
    _ < 0.0234 := by norm_num

/-- b₀ is positive -/
@[simp]
lemma b₀_pos : 0 < b₀ := by
  linarith [b₀_lower]

/-- Convenient existential form of b₀ bounds -/
lemma b₀_value : ∃ b : ℝ, b = b₀ ∧ 0.0232 < b ∧ b < 0.0234 := by
  use b₀
  exact ⟨rfl, b₀_lower, b₀_upper⟩

/-! ## Helper monotonicity facts -/

/-- Square root is monotone -/
lemma sqrt_mono {x y : ℝ} (hx : 0 ≤ x) (hxy : x ≤ y) : sqrt x ≤ sqrt y :=
  Real.sqrt_le_sqrt hxy

/-- Division by larger positive number gives smaller result -/
lemma div_lt_div_of_lt_left' {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hbc : b < c) : a / c < a / b :=
  div_lt_div_of_lt_left ha hb hbc

end YangMillsProof.Numerical.Constants
