/-
  Wilson Correspondence Proofs
  ============================

  This file proves the phase periodicity axiom from WilsonCorrespondence.lean.

  Author: Jonathan Washburn
-/

import YangMillsProof.Continuum.WilsonCorrespondence
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace YangMillsProof.Continuum

open Real

/-- Proof of phase periodicity under gauge transformations -/
theorem phase_periodicity_proof (θ : ℝ) (n : ℕ) (hn : n < 3) :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 2 * Real.pi ∧
  Real.cos φ = Real.cos (θ + 2 * Real.pi * n / 3) := by
  -- Take φ to be the reduced angle modulo 2π
  let θ' := θ + 2 * Real.pi * n / 3
  let k := ⌊θ' / (2 * Real.pi)⌋
  let φ := θ' - 2 * Real.pi * k

  use φ
  constructor
  · -- φ ≥ 0
    unfold φ
    apply sub_nonneg.mpr
    exact Int.le_floor_iff_le.mp (le_refl _)
  · constructor
    · -- φ < 2π
      unfold φ
      have h := Int.sub_floor_div_mul_lt θ' (2 * Real.pi) (by norm_num : 0 < 2 * Real.pi)
      simp at h
      exact h
    · -- cos φ = cos θ'
      unfold φ
      -- cos is periodic with period 2π
      have h_period : ∀ x : ℝ, ∀ m : ℤ, cos (x + 2 * pi * m) = cos x := by
        intro x m
        induction m using Int.induction_on with
        | hz => simp
        | hp m ih =>
          rw [Int.cast_add, Int.cast_one, mul_add, add_assoc]
          rw [cos_add_two_pi, ih]
        | hn m ih =>
          rw [Int.cast_sub, Int.cast_one, mul_sub, sub_eq_add_neg]
          rw [add_assoc, ← sub_eq_add_neg]
          rw [← cos_sub_two_pi]
          simp [ih]

      -- Apply periodicity
      rw [← h_period θ' (-k)]
      simp [Int.cast_neg]
      ring_nf
      rfl

end YangMillsProof.Continuum
