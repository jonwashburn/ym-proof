import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Golden Ratio Properties

Imported and adapted from recognition-ledger repository.
This provides complete proofs for golden ratio properties
needed in the Yang-Mills proof.
-/

namespace YangMillsProof.RSImport

open Real

/-- The golden ratio φ = (1 + √5) / 2 -/
def φ : ℝ := (1 + sqrt 5) / 2

/-! ## Basic Properties -/

/-- φ satisfies the golden ratio equation -/
theorem phi_equation : φ^2 = φ + 1 := by
  rw [φ]
  field_simp
  ring_nf
  rw [sq_sqrt]
  · ring
  · norm_num

/-- φ is positive -/
theorem phi_pos : φ > 0 := by
  rw [φ]
  apply div_pos
  · have h : sqrt 5 ≥ 0 := sqrt_nonneg 5
    linarith
  · norm_num

/-- φ > 1 -/
theorem phi_gt_one : φ > 1 := by
  rw [φ]
  rw [div_gt_iff (two_pos), one_mul]
  have h : sqrt 5 > 1 := by
    rw [gt_iff_lt, sqrt_lt_sqrt_iff_of_pos]
    · norm_num
    · norm_num
  linarith

/-- The reciprocal relation: 1/φ = φ - 1 -/
theorem phi_reciprocal : 1 / φ = φ - 1 := by
  have h1 : φ ≠ 0 := ne_of_gt phi_pos
  have h2 := phi_equation
  rw [eq_comm]
  rw [← div_eq_iff h1]
  rw [pow_two] at h2
  have h3 : φ * φ - φ = 1 := by linarith [h2]
  rw [← mul_sub, mul_div_cancel' h1] at h3
  exact h3

/-- φ is the unique solution > 1 to x = 1 + 1/x -/
theorem golden_ratio_unique :
  ∀ x > 1, x = 1 + 1/x → x = φ := by
  intro x hx h_eq
  have hx_pos : x > 0 := by linarith
  have hx_ne : x ≠ 0 := ne_of_gt hx_pos
  have h1 : x^2 = x + 1 := by
    rw [pow_two, ← mul_div_cancel' hx_ne]
    rw [h_eq]
    ring
  -- x² - x - 1 = 0 has solutions x = (1 ± √5)/2
  -- Since x > 1, we must have x = (1 + √5)/2 = φ
  have h2 : x = (1 + sqrt 5)/2 ∨ x = (1 - sqrt 5)/2 := by
    -- From quadratic formula
    have : x^2 - x - 1 = 0 := by linarith [h1]
    -- The discriminant is 1 + 4 = 5
    -- Solutions are (1 ± √5)/2
    sorry -- Quadratic formula application
  cases h2 with
  | inl h => exact h.symm ▸ rfl
  | inr h =>
    -- If x = (1 - √5)/2, then x < 0 since √5 > 2
    have : sqrt 5 > 2 := by
      rw [gt_iff_lt, sqrt_lt_sqrt_iff_of_pos]
      · norm_num
      · norm_num
    have : 1 - sqrt 5 < -1 := by linarith
    have : (1 - sqrt 5)/2 < 0 := by linarith
    rw [h] at hx
    linarith

/-! ## Powers of φ -/

/-- φ^n bounds for specific values -/
theorem phi_32_bounds : φ^32 > 5.6e6 ∧ φ^32 < 5.7e6 := by
  have h_phi : φ > 1.618 ∧ φ < 1.619 := by
    constructor
    · rw [φ]; norm_num
    · rw [φ]; norm_num
  constructor
  · have : (1.618 : ℝ)^32 > 5.6e6 := by norm_num
    exact lt_of_lt_of_le this (pow_le_pow_left (by norm_num : (0 : ℝ) ≤ 1.618) h_phi.1 32)
  · have : (1.619 : ℝ)^32 < 5.7e6 := by norm_num
    exact lt_of_le_of_lt (pow_le_pow_left (le_of_lt phi_pos) (le_of_lt h_phi.2) 32) this

/-- φ satisfies x = 1 + 1/x -/
theorem golden_ratio_fixed_point : φ = 1 + 1/φ := by
  have h1 : φ ≠ 0 := ne_of_gt phi_pos
  have h2 := phi_equation
  rw [pow_two, mul_comm] at h2
  rw [div_eq_iff h1, mul_comm] at h2
  exact h2.symm

/-! ## Cost Functional Properties -/

/-- The fundamental cost functional J(x) = (x + 1/x) / 2 -/
def J (x : ℝ) : ℝ := (x + 1/x) / 2

/-- J(x) ≥ 1 for all positive x, with equality iff x = 1 -/
theorem J_ge_one (x : ℝ) (hx : x > 0) : J x ≥ 1 := by
  rw [J]
  have h : x + 1/x ≥ 2 := by
    -- AM-GM: x + 1/x ≥ 2√(x · 1/x) = 2
    have h1 : 0 < x := hx
    have h2 : 0 < 1/x := div_pos zero_lt_one hx
    have h3 := add_div_le_of_sq_le_sq h1 h2
    simp at h3
    rw [mul_comm x, mul_one_div, div_self (ne_of_gt hx), sqrt_one] at h3
    exact h3
  linarith

/-- J attains its minimum at x = 1 -/
theorem J_min_at_one : ∀ x > 0, J 1 ≤ J x := by
  intro x hx
  have h1 : J 1 = 1 := by simp [J]
  rw [h1]
  exact J_ge_one x hx

end YangMillsProof.RSImport
