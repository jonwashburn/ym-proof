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

/-- Phi is the unique positive solution to x² = x + 1 -/
theorem phi_unique_positive : ∀ x : ℝ, x > 0 → x^2 = x + 1 → x = phi := by
  intro x hx_pos hx_eq
  -- The quadratic x² - x - 1 = 0 has two roots: phi and phi_conj
  -- Since x > 0 and phi_conj < 0, we must have x = phi
  have h_quad : x^2 - x - 1 = 0 := by linarith [hx_eq]
  -- The quadratic formula gives x = (1 ± √5)/2
  -- The positive root is phi = (1 + √5)/2
  -- The negative root is phi_conj = (1 - √5)/2
  have h_phi_root : phi^2 - phi - 1 = 0 := by
    rw [phi_sq]
    ring
  have h_conj_neg : phi_conj < 0 := by
    unfold phi_conj
    -- (1 - √5)/2 < 0 since √5 > 1
    have h_sqrt5_gt2 : Real.sqrt 5 > 2 := by
      -- √5 > 2 iff 5 > 4
      have h : (2 : ℝ) ^ 2 < 5 := by norm_num
      exact Real.lt_sqrt_of_sq_lt_sq (by norm_num : 0 ≤ 2) h
    linarith
  -- Since the quadratic has exactly two roots and x is positive,
  -- x must equal the positive root phi
  have h_unique : ∀ y : ℝ, y^2 - y - 1 = 0 → y = phi ∨ y = phi_conj := by
    intro y hy
    -- The quadratic y² - y - 1 = 0 has discriminant Δ = 1 + 4 = 5
    -- By the quadratic formula: y = (1 ± √5)/2
    -- These are exactly phi and phi_conj
    -- We verify by showing (y - phi)(y - phi_conj) = y² - y - 1
    have h_factor : y^2 - y - 1 = (y - phi) * (y - phi_conj) := by
      unfold phi phi_conj
      field_simp
      ring_nf
      rw [sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
      ring
    rw [h_factor] at hy
    -- Since (y - phi)(y - phi_conj) = 0, either y - phi = 0 or y - phi_conj = 0
    have h_prod_zero : (y - phi) * (y - phi_conj) = 0 := hy
    obtain h_zero := mul_eq_zero.mp h_prod_zero
    cases h_zero with
    | inl h => left; linarith
    | inr h => right; linarith
  obtain h_cases := h_unique x h_quad
  cases h_cases with
  | inl h_phi => exact h_phi
  | inr h_conj =>
    -- Contradiction: x > 0 but phi_conj < 0
    rw [h_conj] at hx_pos
    exact absurd hx_pos (not_lt.mpr (le_of_lt h_conj_neg))

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
