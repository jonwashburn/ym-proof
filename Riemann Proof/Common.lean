import Mathlib.Analysis.InnerProductSpace.l2Space
import Mathlib.Data.Real.GoldenRatio
import Mathlib.Topology.Instances.ENNReal
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Topology.Algebra.InfiniteSum.Basic

/-!
# Common Infrastructure Definitions

This file contains common definitions used across all infrastructure modules.
-/

namespace RH

open Complex Real BigOperators

/-- The weighted ℓ² space over primes -/
noncomputable abbrev WeightedL2 := lp (fun _ : {p : ℕ // Nat.Prime p} => ℂ) 2

namespace WeightedL2

instance : Fact (1 ≤ 2) := ⟨by norm_num⟩

/-- Basis vectors δ_p for each prime p -/
noncomputable def deltaBasis (p : {p : ℕ // Nat.Prime p}) : WeightedL2 :=
  lp.single 2 p 1

/-- Domain of the arithmetic Hamiltonian -/
def domainH : Set WeightedL2 :=
  {ψ | Summable fun p => ‖ψ p‖^2 * (Real.log p.val)^2}

/-- The inner product on WeightedL2 -/
noncomputable instance : InnerProductSpace ℂ WeightedL2 := by
  -- This follows from the general lp inner product structure
  infer_instance

/-- Norm squared equals sum of component norms squared -/
lemma norm_sq_eq_sum (ψ : WeightedL2) :
    ‖ψ‖^2 = ∑' p : {p : ℕ // Nat.Prime p}, ‖ψ p‖^2 := by
  -- For l2 spaces (p = 2), the norm is (∑' i, ‖f i‖^2)^(1/2)
  -- So the norm squared is ∑' i, ‖f i‖^2
  rw [lp.norm_eq_tsum_rpow]
  simp only [ENNReal.toReal_ofNat]
  -- Now we have ((∑' i, ‖ψ i‖^2)^(1/2))^2 = ∑' i, ‖ψ i‖^2
  -- This follows from (x^(1/2))^2 = x for x ≥ 0
  have h_nonneg : 0 ≤ ∑' (p : { p // Nat.Prime p }), ‖ψ p‖ ^ 2 := tsum_nonneg (fun _ => sq_nonneg _)
  calc
    ((∑' (p : { p // Nat.Prime p }), ‖ψ p‖ ^ 2) ^ (1 / 2)) ^ 2
      = (∑' (p : { p // Nat.Prime p }), ‖ψ p‖ ^ 2) ^ ((1 / 2) * 2) := by rw [← Real.rpow_natCast, Real.rpow_mul h_nonneg]
    _ = (∑' (p : { p // Nat.Prime p }), ‖ψ p‖ ^ 2) ^ 1 := by norm_num
    _ = ∑' (p : { p // Nat.Prime p }), ‖ψ p‖ ^ 2 := by rw [Real.rpow_one]

end WeightedL2

-- Type alias for compatibility
noncomputable abbrev WeightedHilbertSpace := WeightedL2

-- Re-export for convenience
export WeightedL2 (deltaBasis domainH)

/-- The regularized Fredholm determinant -/
noncomputable def fredholm_det2 (s : ℂ) : ℂ :=
  ∏' p : {p : ℕ // Nat.Prime p}, (1 - (p.val : ℂ)^(-s)) * Complex.exp ((p.val : ℂ)^(-s))

/-- The renormalization factor -/
noncomputable def renormE (s : ℂ) : ℂ :=
  Complex.exp (∑' k : ℕ, ∑' p : {p : ℕ // Nat.Prime p}, (p.val : ℂ)^(-(k + 1) * s) / (k + 1 : ℕ))

end RH
