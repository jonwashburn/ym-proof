/-
Golden Ratio Complete Proofs
Vendor-copied from github.com/jonwashburn/recognition-ledger
Extended with all necessary lemmas for Yang-Mills proof
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.Field.Basic

namespace YangMillsProof.RSImport

open Real

/-! ## Golden Ratio Definition and Basic Properties -/

/-- The golden ratio φ = (1 + √5)/2 -/
noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

/-- Alternative form: φ = 1 + 1/φ -/
lemma phi_recursive : phi = 1 + 1/phi := by
  unfold phi
  field_simp
  ring_nf
  rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
  ring

/-- Phi is positive -/
lemma phi_pos : 0 < phi := by
  unfold phi
  have h : 0 < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 5)
  linarith

/-- Phi is greater than 1 -/
lemma phi_gt_one : 1 < phi := by
  unfold phi
  have h : 1 < Real.sqrt 5 := by
    rw [← sqrt_one]
    apply sqrt_lt_sqrt
    norm_num
    norm_num
  linarith

/-- The fundamental equation: φ² = φ + 1 -/
lemma phi_sq : phi^2 = phi + 1 := by
  unfold phi
  field_simp
  ring_nf
  rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
  ring

/-- Phi satisfies the quadratic x² - x - 1 = 0 -/
lemma phi_quadratic : phi^2 - phi - 1 = 0 := by
  rw [phi_sq]
  ring

/-- The conjugate of phi: (1 - √5)/2 -/
noncomputable def phi_conj : ℝ := (1 - Real.sqrt 5) / 2

/-- Phi and its conjugate are roots of x² - x - 1 = 0 -/
lemma phi_conj_quadratic : phi_conj^2 - phi_conj - 1 = 0 := by
  unfold phi_conj
  field_simp
  ring_nf
  rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
  ring

/-- Product of phi and its conjugate equals -1 -/
lemma phi_conj_product : phi * phi_conj = -1 := by
  unfold phi phi_conj
  field_simp
  ring_nf
  rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
  ring

/-- Sum of phi and its conjugate equals 1 -/
lemma phi_conj_sum : phi + phi_conj = 1 := by
  unfold phi phi_conj
  field_simp

/-- Phi scaling property: φ^(n+1) = φ^n + φ^(n-1) for n ≥ 1 -/
lemma phi_fibonacci (n : ℕ) (hn : n ≥ 1) : phi^(n+1) = phi^n + phi^(n-1) := by
  -- This follows from the golden ratio recurrence φ² = φ + 1
  -- We use the fact that φ^(n+1) = φ^n * φ and φ² = φ + 1
  have h_base : phi^2 = phi + 1 := phi_sq
  -- Case analysis on n
  cases n with
  | zero =>
    -- n = 0 contradicts hn : n ≥ 1
    omega
  | succ m =>
    -- n = m + 1, so n + 1 = m + 2, n - 1 = m
    -- We need to show φ^(m+2) = φ^(m+1) + φ^m
    rw [pow_succ, pow_succ]
    -- φ^(m+2) = φ * φ^(m+1) = φ * (φ * φ^m) = φ² * φ^m
    rw [← mul_assoc, ← pow_succ]
    rw [h_base]
    rw [add_mul]
    rw [one_mul]

/-- Recognition Science scaling: φ^n represents the n-th rung cost multiplier -/
lemma phi_rung_scaling (n : ℕ) : ∃ (cost : ℝ), cost = phi^n ∧ cost > 0 := by
  use phi^n
  constructor
  · rfl
  · exact pow_pos phi_pos n

/-- The golden ratio is the unique positive solution to x² = x + 1 -/
theorem golden_ratio_unique : ∀ x : ℝ, x > 0 → x^2 = x + 1 → x = phi := by
  intro x hx h_eq
  -- x² - x - 1 = 0 has solutions x = (1 ± √5)/2
  -- Since x > 0, we must have x = (1 + √5)/2 = phi
  have h1 : x = (1 + Real.sqrt 5)/2 ∨ x = (1 - Real.sqrt 5)/2 := by
    -- From quadratic formula for x² - x - 1 = 0
    have h_quad : x^2 - x - 1 = 0 := by linarith [h_eq]
    -- For ax² + bx + c = 0 with a=1, b=-1, c=-1
    -- Discriminant = b² - 4ac = 1 - 4(1)(-1) = 5
    -- Solutions: x = (-b ± √discriminant) / 2a = (1 ± √5) / 2
    -- We use that if p(x) = x² - x - 1, then p((1+√5)/2) = 0 and p((1-√5)/2) = 0
    have h_root1 : ((1 + Real.sqrt 5)/2)^2 - (1 + Real.sqrt 5)/2 - 1 = 0 := by
      field_simp
      ring_nf
      rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
      ring
    have h_root2 : ((1 - Real.sqrt 5)/2)^2 - (1 - Real.sqrt 5)/2 - 1 = 0 := by
      field_simp
      ring_nf
      rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
      ring
    -- Since quadratic has at most 2 roots and we found 2, x must be one of them
    have h_poly : ∀ y : ℝ, y^2 - y - 1 = (y - (1 + Real.sqrt 5)/2) * (y - (1 - Real.sqrt 5)/2) := by
      intro y
      field_simp
      ring_nf
      rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
      ring
    rw [← h_poly] at h_quad
    have h_factor := mul_eq_zero.mp h_quad
    cases h_factor with
    | inl h =>
      left
      linarith
    | inr h =>
      right
      linarith
  cases h1 with
  | inl h =>
    -- x = (1 + √5)/2 = phi
    rw [h]
    rfl
  | inr h =>
    -- x = (1 - √5)/2 < 0, contradicting x > 0
    have h_neg : (1 - Real.sqrt 5)/2 < 0 := by
      have h_sqrt : Real.sqrt 5 > 2 := by
        rw [Real.sqrt_lt_sqrt_iff_of_pos]
        · norm_num
        · norm_num
        · norm_num
      linarith
    rw [h] at hx
    linarith

/-- Powers of phi are positive -/
lemma phi_power_pos (n : ℕ) : 0 < phi^n := by
  exact pow_pos phi_pos n

/-- The inverse of phi is positive -/
lemma phi_inv_pos : (1 / phi) > 0 := by
  exact div_pos one_pos phi_pos

/-- The inverse of phi is less than 1 -/
lemma phi_inv_lt_one : (1 / phi) < 1 := by
  rw [div_lt_one phi_pos]
  exact phi_gt_one

/-- Helper: phi squared is phi plus 1 -/
lemma phi_sq_eq_phi_add_one : phi^2 = phi + 1 := phi_sq

/-- The inverse of phi squared -/
lemma phi_inv_sq : (1/phi)^2 = 1/phi^2 := by
  rw [div_pow]

end YangMillsProof.RSImport
