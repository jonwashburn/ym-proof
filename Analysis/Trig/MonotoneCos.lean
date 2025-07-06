import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.Basic
import Mathlib.Analysis.Calculus.Deriv

open Real

/-! # Monotonicity of cosine on `[0, π]` -/

lemma cos_le_cos_of_le_of_nonneg_of_le_pi {x y : ℝ}
    (hx0 : 0 ≤ x) (hy : y ≤ π) (hxy : x ≤ y) :
    cos y ≤ cos x := by
  -- Use mean-value theorem on [x,y]
  have hxy' : x ≤ y := hxy
  have hconn : ContinuousOn cos (Set.Icc x y) :=
    (Real.continuous_cos).continuousOn
  have hderiv : ∀ z ∈ Set.Ioo x y, HasDerivAt cos (-sin z) z := by
    intro z hz; exact (Real.hasDerivAt_cos z).hasDerivAt
  have hneg : ∀ z ∈ Set.Icc x y, -sin z ≤ 0 := by
    intro z hz
    have hsin_nonneg : 0 ≤ sin z := by
      -- for z in [0,π] we have sin z ≥ 0
      have hz0 : 0 ≤ z := by
        have : x ≤ z := hz.1
        exact le_trans hx0 this
      have hzπ : z ≤ π := by
        have : z ≤ y := hz.2
        exact le_trans this hy
      exact Real.sin_nonneg_of_mem_Icc ⟨hz0,hzπ⟩
    have : -sin z ≤ 0 := by
      have : sin z ≥ 0 := hsin_nonneg
      linarith
    exact this
  -- Apply MVT with monotone derivative ≤0 -> cos decreases
  have h := Convex.monotoneOn_of_deriv_nonpos
      (convex_Icc _ _) hconn hderiv hneg
  have h' := h (by exact ⟨hx0,hxy'⟩) (by exact ⟨le_of_le_of_le hx0 hxy',hy⟩) hxy
  simpa using h'
