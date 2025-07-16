import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic

open Real

/-! # Monotonicity of cosine on `[0, π]` -/

/-- Cosine is non-increasing on [0, π] -/
lemma cos_le_cos_of_le_of_nonneg_of_le_pi {x y : ℝ}
    (hx0 : 0 ≤ x) (hy : y ≤ π) (hxy : x ≤ y) :
    cos y ≤ cos x := by
  -- Cosine is strictly decreasing on [0, π], so this follows from monotonicity
  exact cos_le_cos_of_zero_le_of_le_pi hx0 hy hxy
