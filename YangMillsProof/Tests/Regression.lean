/-
  Yang-Mills Proof Regression Tests
  ================================

  These tests ensure that key mathematical properties remain intact
  after refactoring. They will fail CI if broken.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

import YangMillsProof.Parameters.Bounds
import YangMillsProof.Parameters.Definitions

namespace YangMillsProof.Tests

open RS.Param Real

-- Test 1: E_coh ≥ 1 (critical for Wilson measure bounds)
example : (1 : ℝ) ≤ E_coh := E_coh_ge_one

-- Test 2: Existence of positive mass gap
example : ∃ gap > 0, gap = E_coh * (φ - 1) := by
  use E_coh * (φ - 1)
  constructor
  · -- Prove E_coh * (φ - 1) > 0
    apply mul_pos E_coh_positive
    -- Prove φ > 1
    have h_phi_gt_one : (1 : ℝ) < φ := by
      unfold φ
      -- φ = (1 + √5)/2 > 1 since √5 > 1
      have h_sqrt5_gt_one : (1 : ℝ) < sqrt 5 := by
        rw [lt_sqrt (by norm_num) (by norm_num)]
        norm_num
      simp [add_div]
      linarith [h_sqrt5_gt_one]
    exact sub_pos.mpr h_phi_gt_one
  · rfl

-- Test 3: Basic positivity properties
example : 0 < φ := φ_positive
example : 0 < E_coh := E_coh_positive
example : 0 < τ₀ := τ₀_positive

-- Test 4: Golden ratio property (φ² = φ + 1)
example : φ^2 = φ + 1 := by
  unfold φ
  field_simp
  ring_nf
  norm_num

-- Test 5: Recognition length bounds
example : 0 < λ_rec ∧ λ_rec < 1 := by
  constructor
  · unfold λ_rec
    apply sqrt_pos.mpr
    apply div_pos
    · exact log_pos (by norm_num)
    · exact pi_pos
  · unfold λ_rec
    apply sqrt_lt_one
    apply div_lt_one_of_lt
    · exact log_pos (by norm_num)
    · have h1 : log (2 : ℝ) < 1 := by
        have : (2 : ℝ) < Real.exp 1 := by norm_num
        exact log_lt_iff_lt_exp.mpr this
      have h2 : (1 : ℝ) < π := by norm_num
      linarith [h1, h2]
    · exact pi_pos

end YangMillsProof.Tests
