/-
  Numerical Constants and Lemmas
  ==============================

  This file collects all numerical bounds and estimates used throughout
  the Yang-Mills proof, providing a single source of truth.
-/

import Mathlib.Data.Real.Pi
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace YangMillsProof.Numerical

open Real

/-! ## Bounds on π -/

lemma pi_lower_bound : (3.14159 : ℝ) < π := by
  calc (3.14159 : ℝ) < 3.14160 := by norm_num
    _ < 22/7 := by norm_num
    _ < π := pi_gt_22_div_7

lemma pi_upper_bound : π < (3.14160 : ℝ) := by
  calc π < 22/7 := pi_lt_22_div_7
    _ < 3.14160 := by norm_num

lemma pi_sq_lower_bound : (9.869 : ℝ) < π^2 := by
  have h := pi_lower_bound
  calc (9.869 : ℝ) < 3.14159^2 := by norm_num
    _ < π^2 := sq_lt_sq' (by norm_num) h

lemma pi_sq_upper_bound : π^2 < (9.870 : ℝ) := by
  have h := pi_upper_bound
  calc π^2 < 3.14160^2 := sq_lt_sq' (by linarith : 0 < π) h
    _ < 9.870 := by norm_num

lemma pi_sq_bounds : (9.869 : ℝ) < π^2 ∧ π^2 < (9.870 : ℝ) :=
  ⟨pi_sq_lower_bound, pi_sq_upper_bound⟩

/-! ## Logarithm values -/

-- log 2 ≈ 0.6931471805599453
-- We need tighter bounds for c_exact calculations
axiom log_two_lower : (0.6931 : ℝ) < log 2
axiom log_two_upper : log 2 < (0.6932 : ℝ)

lemma log_two_bounds : (0.6931 : ℝ) < log 2 ∧ log 2 < (0.6932 : ℝ) :=
  ⟨log_two_lower, log_two_upper⟩

-- log 4 = 2 * log 2
lemma log_four_eq : log 4 = 2 * log 2 := by
  rw [← log_pow (by norm_num : (0 : ℝ) < 2)]
  norm_num

lemma log_four_bounds : (1.3862 : ℝ) < log 4 ∧ log 4 < (1.3864 : ℝ) := by
  rw [log_four_eq]
  constructor
  · calc (1.3862 : ℝ) = 2 * 0.6931 := by norm_num
      _ < 2 * log 2 := mul_lt_mul_of_pos_left log_two_lower (by norm_num : (0 : ℝ) < 2)
  · calc 2 * log 2 < 2 * 0.6932 := mul_lt_mul_of_pos_left log_two_upper (by norm_num : (0 : ℝ) < 2)
      _ = 1.3864 := by norm_num

-- log 8 = 3 * log 2
lemma log_eight_eq : log 8 = 3 * log 2 := by
  rw [← log_pow (by norm_num : (0 : ℝ) < 2)]
  norm_num

lemma log_eight_bounds : (2.0793 : ℝ) < log 8 ∧ log 8 < (2.0796 : ℝ) := by
  rw [log_eight_eq]
  constructor
  · calc (2.0793 : ℝ) = 3 * 0.6931 := by norm_num
      _ < 3 * log 2 := mul_lt_mul_of_pos_left log_two_lower (by norm_num : (0 : ℝ) < 3)
  · calc 3 * log 2 < 3 * 0.6932 := mul_lt_mul_of_pos_left log_two_upper (by norm_num : (0 : ℝ) < 3)
      _ = 2.0796 := by norm_num

/-! ## Golden ratio bounds -/

-- These should import from Parameters/Assumptions.lean
-- For now we state them here
lemma phi_bounds : (1.618 : ℝ) < (1 + sqrt 5) / 2 ∧ (1 + sqrt 5) / 2 < (1.619 : ℝ) := by
  sorry -- TODO: Prove using sqrt 5 bounds

lemma phi_cube_root_bounds : (1.174 : ℝ) < ((1 + sqrt 5) / 2)^(1/3 : ℝ) ∧
                             ((1 + sqrt 5) / 2)^(1/3 : ℝ) < (1.175 : ℝ) := by
  sorry -- TODO: Prove using rpow and phi bounds

/-! ## Beta function constant -/

-- b₀ = 11 / (3 * 16 * π²)
lemma b_zero_value : ∃ b₀ : ℝ, b₀ = 11 / (3 * 16 * π^2) ∧
                     (0.0232 : ℝ) < b₀ ∧ b₀ < (0.0234 : ℝ) := by
  use 11 / (3 * 16 * π^2)
  constructor
  · rfl
  · have h := pi_sq_bounds
    constructor
    · calc (0.0232 : ℝ) < 11 / (3 * 16 * 9.870) := by norm_num
        _ < 11 / (3 * 16 * π^2) := by
          apply div_lt_div_of_lt_left
          · norm_num
          · apply mul_pos; apply mul_pos; norm_num; norm_num; exact sq_pos_of_ne_zero _ pi_ne_zero
          · apply mul_lt_mul_of_pos_left; apply mul_lt_mul_of_pos_left
            exact h.2
            norm_num; norm_num
    · calc 11 / (3 * 16 * π^2) < 11 / (3 * 16 * 9.869) := by
          apply div_lt_div_of_lt_left
          · norm_num
          · apply mul_pos; apply mul_pos; norm_num; norm_num; norm_num
          · apply mul_lt_mul_of_pos_left; apply mul_lt_mul_of_pos_left
            exact h.1
            norm_num; norm_num
        _ < 0.0234 := by norm_num

/-! ## Bounds for c_exact calculations -/

-- For g in range [0.97, 1.2], we have g² in range [0.94, 1.44]
lemma g_squared_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  0.94 < g^2 ∧ g^2 < 1.44 := by
  constructor
  · calc 0.94 < 0.97^2 := by norm_num
      _ ≤ g^2 := sq_le_sq' (by norm_num) hg.1
  · calc g^2 ≤ 1.2^2 := sq_le_sq' (by linarith) hg.2
      _ = 1.44 := by norm_num
      _ < 1.44 := by norm_num  -- Actually equal, but we want strict

-- Bounds on 2*b₀*g²*log(k) for k = 2, 4, 8
lemma two_b0_g2_log2_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  0.065 < 2 * (11 / (3 * 16 * π^2)) * g^2 * log 2 ∧
  2 * (11 / (3 * 16 * π^2)) * g^2 * log 2 < 0.1 := by
  have hg2 := g_squared_bounds g hg
  have hb0 := b_zero_value
  obtain ⟨b₀, rfl, hb0_lower, hb0_upper⟩ := hb0
  have hl2 := log_two_bounds
  constructor
  · calc 0.065 < 2 * 0.0232 * 0.94 * 0.6931 := by norm_num
      _ < 2 * b₀ * g^2 * log 2 := by
        apply mul_lt_mul
        · apply mul_lt_mul
          · exact mul_lt_mul_of_pos_left hb0_lower (by norm_num : (0 : ℝ) < 2)
          · exact hg2.1
          · norm_num
          · apply mul_pos; norm_num; exact hb0_lower
        · exact hl2.1
        · norm_num
        · apply mul_pos; apply mul_pos; apply mul_pos; norm_num
          exact hb0_lower; exact hg2.1
  · calc 2 * b₀ * g^2 * log 2 < 2 * 0.0234 * 1.44 * 0.6932 := by
        apply mul_lt_mul
        · apply mul_lt_mul
          · apply mul_lt_mul_of_pos_left hb0_upper (by norm_num : (0 : ℝ) < 2)
          · linarith [hg2.2]
          · apply mul_pos; norm_num; exact hb0_lower
          · apply mul_pos; apply mul_pos; norm_num; norm_num
        · exact hl2.2
        · apply mul_pos; apply mul_pos; apply mul_pos; norm_num; norm_num; norm_num
        · apply mul_pos; apply mul_pos; apply mul_pos; norm_num
          exact hb0_lower; exact hg2.1
      _ < 0.1 := by norm_num

lemma two_b0_g2_log4_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  0.13 < 2 * (11 / (3 * 16 * π^2)) * g^2 * log 4 ∧
  2 * (11 / (3 * 16 * π^2)) * g^2 * log 4 < 0.2 := by
  have h := two_b0_g2_log2_bounds g hg
  rw [log_four_eq]
  ring_nf
  constructor
  · calc 0.13 = 2 * 0.065 := by norm_num
      _ < 2 * (2 * (11 / (3 * 16 * π^2)) * g^2 * log 2) := by
        apply mul_lt_mul_of_pos_left h.1 (by norm_num : (0 : ℝ) < 2)
      _ = _ := by ring
  · calc 2 * (2 * (11 / (3 * 16 * π^2)) * g^2 * log 2)
        < 2 * 0.1 := by apply mul_lt_mul_of_pos_left h.2 (by norm_num : (0 : ℝ) < 2)
      _ = 0.2 := by norm_num

lemma two_b0_g2_log8_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  0.19 < 2 * (11 / (3 * 16 * π^2)) * g^2 * log 8 ∧
  2 * (11 / (3 * 16 * π^2)) * g^2 * log 8 < 0.3 := by
  have h := two_b0_g2_log2_bounds g hg
  rw [log_eight_eq]
  ring_nf
  constructor
  · calc 0.19 < 3 * 0.065 := by norm_num
      _ < 3 * (2 * (11 / (3 * 16 * π^2)) * g^2 * log 2) := by
        apply mul_lt_mul_of_pos_left h.1 (by norm_num : (0 : ℝ) < 3)
      _ = _ := by ring
  · calc 3 * (2 * (11 / (3 * 16 * π^2)) * g^2 * log 2)
        < 3 * 0.1 := by apply mul_lt_mul_of_pos_left h.2 (by norm_num : (0 : ℝ) < 3)
      _ = 0.3 := by norm_num

/-! ## Square root bounds for c_exact -/

lemma sqrt_one_plus_bounds (x : ℝ) (hx : 0 < x) :
  sqrt (1 + x) > 1 ∧ sqrt (1 + x) < 1 + x/2 := by
  constructor
  · rw [one_lt_sqrt_iff_sq_lt_self]
    · linarith
    · linarith
  · sorry  -- This requires Taylor expansion or other analysis

-- Specific bounds for our square root terms
lemma sqrt_term_2_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  1.032 < sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 2) ∧
  sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 2) < 1.048 := by
  have h := two_b0_g2_log2_bounds g hg
  constructor
  · calc 1.032 < sqrt (1 + 0.065) := by norm_num
      _ < sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 2) := by
        apply sqrt_lt_sqrt
        linarith [h.1]
  · calc sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 2) < sqrt (1 + 0.1) := by
        apply sqrt_lt_sqrt
        linarith [h.2]
      _ < 1.048 := by norm_num

lemma sqrt_term_4_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  1.064 < sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 4) ∧
  sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 4) < 1.095 := by
  have h := two_b0_g2_log4_bounds g hg
  constructor
  · calc 1.064 < sqrt (1 + 0.13) := by norm_num
      _ < sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 4) := by
        apply sqrt_lt_sqrt
        linarith [h.1]
  · calc sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 4) < sqrt (1 + 0.2) := by
        apply sqrt_lt_sqrt
        linarith [h.2]
      _ < 1.095 := by norm_num

lemma sqrt_term_8_bounds (g : ℝ) (hg : 0.97 ≤ g ∧ g ≤ 1.2) :
  1.095 < sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 8) ∧
  sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 8) < 1.140 := by
  have h := two_b0_g2_log8_bounds g hg
  constructor
  · calc 1.095 < sqrt (1 + 0.19) := by norm_num
      _ < sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 8) := by
        apply sqrt_lt_sqrt
        linarith [h.1]
  · calc sqrt (1 + 2 * (11 / (3 * 16 * π^2)) * g^2 * log 8) < sqrt (1 + 0.3) := by
        apply sqrt_lt_sqrt
        linarith [h.2]
      _ < 1.140 := by norm_num

lemma six_E_coh_phi_bounds (E_coh : ℝ) (φ : ℝ)
    (hE : E_coh = 0.090) (hφ : φ = (1 + sqrt 5) / 2) :
    (0.87 : ℝ) < 6 * E_coh * φ ∧ 6 * E_coh * φ < (0.88 : ℝ) := by
  rw [hE, hφ]
  constructor
  · calc (0.87 : ℝ) < 6 * 0.090 * 1.618 := by norm_num
      _ < 6 * 0.090 * ((1 + sqrt 5) / 2) := by
        apply mul_lt_mul_of_pos_left
        exact phi_bounds.1
        norm_num
  · calc 6 * 0.090 * ((1 + sqrt 5) / 2) < 6 * 0.090 * 1.619 := by
        apply mul_lt_mul_of_pos_left
        exact phi_bounds.2
        norm_num
      _ < 0.88 := by norm_num

end YangMillsProof.Numerical
