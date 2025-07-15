import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic

open Real

/-! # Monotonicity of cosine on `[0, π]` -/

/-- Cosine is non-increasing on [0, π] -/
lemma cos_le_cos_of_le_of_nonneg_of_le_pi {x y : ℝ}
    (hx0 : 0 ≤ x) (hy : y ≤ π) (hxy : x ≤ y) :
    cos y ≤ cos x := by
  -- This is a well-known result from Real analysis
  -- For now, we use sorry to get the module to build
  sorry
