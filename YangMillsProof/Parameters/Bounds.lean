/-
  Recognition Science Parameter Bounds
  ===================================

  Positivity and numerical inequalities for Recognition Science parameters.
  These establish the mathematical properties needed for the Yang-Mills proof.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

import YangMillsProof.Parameters.Definitions
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace RS.Param

open Real

-- Basic positivity properties
theorem φ_positive : 0 < φ := by
  unfold φ
  apply div_pos
  · apply add_pos
    · norm_num
    · exact sqrt_pos.mpr (by norm_num)
  · norm_num

theorem E_coh_positive : 0 < E_coh := by
  unfold E_coh χ
  apply div_pos
  · apply div_pos
    · exact φ_positive
    · exact pi_pos
  · unfold λ_rec
    apply sqrt_pos.mpr
    apply div_pos
    · exact log_pos (by norm_num)
    · exact pi_pos

theorem τ₀_positive : 0 < τ₀ := by
  unfold τ₀
  apply div_pos
  · unfold λ_rec
    apply sqrt_pos.mpr
    apply div_pos
    · exact log_pos (by norm_num)
    · exact pi_pos
  · apply mul_pos
    · norm_num
    · have h1 : 1 < φ := by
        unfold φ
        simp [add_div]
        have h2 : 1 < sqrt 5 := by
          rw [lt_sqrt (by norm_num) (by norm_num)]
          norm_num
        linarith
      exact log_pos h1

-- E_coh is greater than or equal to 1
theorem E_coh_ge_one : (1 : ℝ) ≤ E_coh := by
  unfold E_coh χ
  rw [div_le_div_iff]
  · simp only [one_mul]
    unfold λ_rec
    rw [div_le_sqrt_iff]
    · unfold φ
      simp only [pow_two]
      field_simp
      rw [add_pow_two]
      simp only [one_pow, mul_one]
      have h_sqrt5 : (2 : ℝ) < sqrt 5 := by
        rw [lt_sqrt (by norm_num) (by norm_num)]
        norm_num
      have h_left : (6 : ℝ) + 2 * sqrt 5 > 6 + 2 * 2 := by
        linarith [h_sqrt5]
      have h_right : (4 : ℝ) * π * log 2 < 4 * 3.2 * 0.7 := by
        apply mul_lt_mul_of_pos_left
        · apply mul_lt_mul_of_pos_left
          · norm_num
          · exact log_pos (by norm_num)
        · norm_num
      simp at h_left h_right
      linarith [h_left, h_right]
    · exact pi_pos
    · exact log_pos (by norm_num)
  · exact pi_pos
  · unfold λ_rec
    apply sqrt_pos.mpr
    apply div_pos
    · exact log_pos (by norm_num)
    · exact pi_pos

end RS.Param
