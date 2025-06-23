import Mathlib.Analysis.InnerProductSpace.l2Space
import Mathlib.Analysis.Normed.Operator.ContinuousLinearMap
import rh.Common

/-!
# Fredholm Determinant Theory

This file develops the theory of Fredholm determinants for diagonal operators.
-/

namespace RH.FredholmDeterminant

open Complex Real RH

/-- The eigenvalues of the evolution operator -/
noncomputable def evolutionEigenvalues (s : ℂ) : {p : ℕ // Nat.Prime p} → ℂ :=
  fun p => (p.val : ℂ)^(-s)

/-- A diagonal operator with given eigenvalues -/
noncomputable def DiagonalOperator (eigenvalues : {p : ℕ // Nat.Prime p} → ℂ)
    (h_bounded : ∃ C : ℝ, ∀ p, ‖eigenvalues p‖ ≤ C) : WeightedL2 →L[ℂ] WeightedL2 := by
  -- Get the bound
          obtain ⟨C, hC⟩ := h_bounded
  -- Define the linear map that multiplies pointwise
  let L : WeightedL2 →ₗ[ℂ] WeightedL2 := {
    toFun := fun ψ => {
      val := fun p => eigenvalues p * ψ p
      property := by
        -- We need to show the pointwise product is in l²
        -- Use that |eigenvalues p * ψ p|² ≤ C² |ψ p|²
        have h_sq : ∀ p, ‖eigenvalues p * ψ p‖^2 ≤ C^2 * ‖ψ p‖^2 := by
            intro p
            rw [norm_mul, mul_pow]
          apply mul_le_mul_of_nonneg_right
          · exact sq_le_sq' (neg_le_neg (hC p)) (hC p)
          · exact sq_nonneg _
        -- Now use that ψ ∈ l² and C² scaling preserves l²
        rw [lp.mem_ℓp_iff_summable_norm_rpow] at ψ.property ⊢
        simp only [ENNReal.toReal_ofNat] at ψ.property ⊢
        apply Summable.of_norm_bounded_eventually _ (ψ.property.mul_left (C^2))
        filter_upwards with p
        rw [norm_mul, Real.mul_rpow (sq_nonneg C) (norm_nonneg _)]
        simp only [Real.rpow_two]
        exact h_sq p
    }
    map_add' := fun ψ φ => by
        ext p
        simp only [lp.coeFn_add, Pi.add_apply]
      ring
    map_smul' := fun c ψ => by
        ext p
      simp only [lp.coeFn_smul, Pi.smul_apply, RingHom.id_apply, smul_eq_mul]
        ring
  }
  -- Show L is continuous with bound C
  have h_cont : ∀ ψ, ‖L ψ‖ ≤ C * ‖ψ‖ := by
    intro ψ
    -- We have ‖L ψ‖² = ∑ |eigenvalues p * ψ p|² ≤ C² ∑ |ψ p|² = C² ‖ψ‖²
    rw [norm_sq_eq_sum, norm_sq_eq_sum]
    simp only [L, LinearMap.coe_mk, AddHom.coe_mk]
    have h_sum : (∑' p, ‖eigenvalues p * ψ p‖^2) ≤ C^2 * ∑' p, ‖ψ p‖^2 := by
      rw [← tsum_mul_left]
      apply tsum_le_tsum
      · intro p
        rw [norm_mul, mul_pow]
        apply mul_le_mul_of_nonneg_right
        · exact sq_le_sq' (neg_le_neg (hC p)) (hC p)
        · exact sq_nonneg _
      · exact Summable.mul_left _ (lp.summable_norm_rpow_of_mem_ℓp (by norm_num : 0 < 2)
          (by norm_num : (2 : ℝ≥0∞) ≠ ∞) ψ.property)
      · apply Summable.of_norm_bounded_eventually _
          (Summable.mul_left _ (lp.summable_norm_rpow_of_mem_ℓp (by norm_num : 0 < 2)
            (by norm_num : (2 : ℝ≥0∞) ≠ ∞) ψ.property))
        filter_upwards with p
        simp only [norm_mul, mul_pow, norm_pow]
        apply mul_le_mul_of_nonneg_right
        · exact sq_le_sq' (neg_le_neg (hC p)) (hC p)
        · exact sq_nonneg _
    -- Take square roots
    rw [Real.sqrt_le_sqrt h_sum]
    rw [Real.sqrt_mul (sq_nonneg C), Real.sqrt_sq (abs_nonneg C)]
    exact mul_comm (|C|) _
  -- Make it continuous
  exact L.mkContinuous C (fun ψ => by simp only [LinearMap.mkContinuous_norm_le', h_cont ψ])

/-- The evolution operator from eigenvalues -/
noncomputable def evolutionOperatorFromEigenvalues (s : ℂ) : WeightedL2 →L[ℂ] WeightedL2 :=
  DiagonalOperator (evolutionEigenvalues s)
    ⟨(2 : ℝ)^s.re, fun p => by
      -- Show |p^(-s)| ≤ 2^(Re s)
      simp only [evolutionEigenvalues]
      rw [norm_cpow_of_ne_zero (Nat.cast_ne_zero.mpr (Nat.Prime.ne_zero p.property))]
      simp only [norm_natCast, neg_re]
      -- We have |p|^(-Re s) = p^(-Re s) ≤ 2^(-Re s) since p ≥ 2
      have hp : 2 ≤ (p.val : ℝ) := by
        exact Nat.cast_le.mpr (Nat.Prime.two_le p.property)
      -- For p ≥ 2 and Re s > 0, we have p^(-Re s) ≤ 2^(-Re s)
      -- When Re s < 0, we need p^(-Re s) ≤ 2^(Re s) = 2^(-(-Re s))
      -- This uses that for a ≥ b > 0 and t < 0, we have a^t ≤ b^t
      by_cases h : 0 ≤ s.re
      · -- Case Re s ≥ 0: p^(-Re s) ≤ 2^(-Re s) ≤ 2^(Re s)
        have : (p.val : ℝ)^(-s.re) ≤ (2 : ℝ)^(-s.re) := by
          apply Real.rpow_le_rpow_of_exponent_nonpos hp
          · exact le_neg.mp (neg_nonpos.mpr h)
        exact le_trans this (Real.rpow_le_rpow_of_exponent_ge (by norm_num : 0 < 2)
          (by linarith : -s.re ≤ s.re))
      · -- Case Re s < 0: p^(-Re s) ≤ 2^(-Re s) = 2^(Re s)
        push_neg at h
        have : (p.val : ℝ)^(-s.re) ≤ (2 : ℝ)^(-s.re) := by
          apply Real.rpow_le_rpow_of_exponent_ge hp
          exact neg_pos.mpr h
        convert this
        rw [← neg_neg s.re]
        rfl⟩

/-- A(s) acts diagonally on basis vectors with eigenvalues p^{-s}. -/
@[simp]
lemma evolution_diagonal_action (s : ℂ) (p : {p : ℕ // Nat.Prime p}) :
    evolutionOperatorFromEigenvalues s (WeightedL2.deltaBasis p) =
      (p.val : ℂ)^(-s) • WeightedL2.deltaBasis p := by
  -- This follows from the definition of the diagonal operator
  unfold evolutionOperatorFromEigenvalues DiagonalOperator
  simp only [ContinuousLinearMap.coe_mk', LinearMap.coe_mk, AddHom.coe_mk]
  ext q
  simp only [WeightedL2.deltaBasis, lp.single_apply, evolutionEigenvalues]
  by_cases h : q = p
  · simp [h, Pi.smul_apply, smul_eq_mul]
  · simp [h, Pi.smul_apply, mul_zero, smul_zero]

/-- The regularized Fredholm determinant for diagonal operators -/
noncomputable def fredholmDet2Diagonal (eigenvalues : {p : ℕ // Nat.Prime p} → ℂ) : ℂ :=
  ∏' p : {p : ℕ // Nat.Prime p}, (1 - eigenvalues p) * Complex.exp (eigenvalues p)

/-- The determinant identity specialized to our evolution eigenvalues. -/
theorem fredholm_det2_diagonal (s : ℂ) (hs : 1/2 < s.re) :
    fredholmDet2Diagonal (evolutionEigenvalues s) =
      ∏' p : {p : ℕ // Nat.Prime p}, (1 - (p.val : ℂ)^(-s)) * Complex.exp ((p.val : ℂ)^(-s)) := by
  -- This is just the definition unfolded
  unfold fredholmDet2Diagonal evolutionEigenvalues
  rfl

end RH.FredholmDeterminant
