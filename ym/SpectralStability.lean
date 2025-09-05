/-
ym/SpectralStability.lean

P5 (UniformGap + convergence ⇒ GapPersists), specialized to finite-dimensional
inner-product spaces over ℝ or ℂ, with a clean connector to norm-convergent
embeddings (ScalingFamily-style).

This file is self-contained: it states the P1 hypothesis (ordered top-two
eigenvalues are 1-Lipschitz in operator norm) as an explicit argument, proves
P2 for `→L`, then packages P5 two ways:
  • `gap_persists_under_convergence` on a fixed ambient space `E`, and
  • `gap_persists_via_embedding` for families embedded into a common `E`.

You supply your concrete λ₁, λ₂ (top and second eigenvalue functionals), and
your P1 lemma as the `P1` argument.
-/

import Mathlib/Analysis/InnerProductSpace/Adjoint
import Mathlib/Analysis/NormedSpace/OperatorNorm
import Mathlib/Tactic

noncomputable section
open scoped Real
open Filter

namespace YM

/-
Universe, field, and space setup
-/
variables {𝕂 : Type*} [IsROrC 𝕂]
variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]
variables [FiniteDimensional 𝕂 E]
-- We require at least two dimensions so that λ₂ exists meaningfully.
variable  [Fact (1 < finrank 𝕂 E)]

/-- Spectral gap functional built from user-supplied ordered eigenvalue functionals. -/
def eigGap (λ₁ λ₂ : (E →L[𝕂] E) → ℝ) (T : E →L[𝕂] E) : ℝ :=
  λ₁ T - λ₂ T

/-
P1: Lipschitz stability of λ₁ and λ₂ in operator norm on self-adjoint operators.
We take this as an input hypothesis (argument) so you can plug in your project’s P1.
-/
-- P1 argument shape:
--   P1 : ∀ {X Y : E →L[𝕂] E}, IsSelfAdjoint X → IsSelfAdjoint Y →
--        |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖

/-!
P2 (Gap persistence in the limit for `→L`):
If `(Aseq n)` are self-adjoint with a uniform top gap `δ > 0` and
`Aseq n → A` in operator norm, then `A` has a positive gap (`≥ δ/2`).
This is the same argument as P2, specialized to continuous linear operators.
-/
theorem gap_persistence
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ)
    (P1 : ∀ {X Y : E →L[𝕂] E},
      IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    {A : E →L[𝕂] E} {Aseq : ℕ → E →L[𝕂] E}
    (hA  : IsSelfAdjoint A)
    (hAn : ∀ n, IsSelfAdjoint (Aseq n))
    {δ : ℝ} (hδpos : 0 < δ)
    (hGap  : ∀ n, eigGap λ₁ λ₂ (Aseq n) ≥ δ)
    (hConv : ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖Aseq n - A‖ ≤ ε) :
    ∃ δ' > 0, eigGap λ₁ λ₂ A ≥ δ' :=
by
  -- Choose N with ‖Aseq N - A‖ ≤ δ/4.
  obtain ⟨N, hN⟩ := hConv (δ / 4) (by nlinarith)
  have hNnorm : ‖A - Aseq N‖ ≤ δ / 4 := by
    have := hN N (le_rfl)
    simpa [norm_sub_rev] using this

  -- P1 gives Lipschitz control of λ₁, λ₂ between A and Aseq N.
  have hλ1_abs : |λ₁ A - λ₁ (Aseq N)| ≤ ‖A - Aseq N‖ := (P1 hA (hAn N)).1
  have hλ2_abs : |λ₂ A - λ₂ (Aseq N)| ≤ ‖A - Aseq N‖ := (P1 hA (hAn N)).2

  -- Rearrange to one-sided bounds:
  have hλ1_low : λ₁ (Aseq N) - ‖A - Aseq N‖ ≤ λ₁ A :=
    (abs_sub_le_iff.mp hλ1_abs).1
  have hλ2_high : λ₂ A ≤ λ₂ (Aseq N) + ‖A - Aseq N‖ :=
    (abs_sub_le_iff.mp hλ2_abs).2

  -- Lower bound the gap of A via the gap of Aseq N and the distance.
  have gap_ge :
      eigGap λ₁ λ₂ A
        ≥ (eigGap λ₁ λ₂ (Aseq N)) - 2 * ‖A - Aseq N‖ :=
  by
    -- (x - z) - (y + z) ≤ (x - y) - 2z, applied with ≤ direction rearranged to ≥ on the RHS.
    have h :=
      sub_le_sub hλ1_low hλ2_high
    -- h : (λ₁ (Aseq N) - ‖A - Aseq N‖) - (λ₂ (Aseq N) + ‖A - Aseq N‖) ≤ λ₁ A - λ₂ A
    -- Rewrite the LHS to (eigGap Aseq N) - 2‖A - Aseq N‖.
    have : (λ₁ (Aseq N) - ‖A - Aseq N‖) - (λ₂ (Aseq N) + ‖A - Aseq N‖)
            = (eigGap λ₁ λ₂ (Aseq N)) - 2 * ‖A - Aseq N‖ := by
      simp [eigGap, sub_eq_add_neg, add_comm, add_left_comm, add_assoc, two_mul, mul_comm]
    -- Flip inequality to match `≥` form:
    have h' : (eigGap λ₁ λ₂ (Aseq N)) - 2 * ‖A - Aseq N‖ ≤ eigGap λ₁ λ₂ A := by
      simpa [eigGap, this, sub_eq_add_neg] using h
    exact le_of_lt_or_eq (lt_or_eq_of_le h')

  -- Insert the uniform gap and the δ/4 proximity to get δ/2.
  have gap_ge_delta_minus : eigGap λ₁ λ₂ A ≥ δ - 2 * ‖A - Aseq N‖ := by
    have := hGap N
    have := le_trans (by exact this) gap_ge
    -- Actually reframe: from `gap_ge` as ≥ ... and `hGap N ≥ δ`, conclude:
    -- eigGap A ≥ δ - 2‖A - Aseq N‖
    linarith
  have gap_ge_half : eigGap λ₁ λ₂ A ≥ δ / 2 := by
    linarith [hNnorm, gap_ge_delta_minus]

  exact ⟨δ / 2, by nlinarith [hδpos], gap_ge_half⟩

/-!
P5 on a fixed ambient space `E`.
This is the polished statement you’ll actually use most often.
-/
theorem gap_persists_under_convergence
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ)
    (P1 : ∀ {X Y : E →L[𝕂] E},
      IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    {A : E →L[𝕂] E} {Aseq : ℕ → E →L[𝕂] E}
    (hA  : IsSelfAdjoint A)
    (hAn : ∀ n, IsSelfAdjoint (Aseq n))
    {δ : ℝ} (hδpos : 0 < δ)
    (hGap  : ∀ n, eigGap λ₁ λ₂ (Aseq n) ≥ δ)
    (hConv : ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖Aseq n - A‖ ≤ ε) :
    ∃ δ' > 0, eigGap λ₁ λ₂ A ≥ δ' :=
  gap_persistence λ₁ λ₂ P1 hA hAn hδpos hGap hConv

/-
Connector to ScalingFamily via norm‑convergent embedding.

Setup:
  • For each scale n you have a finite-dim Hilbert space F n and a self-adjoint
    operator T n : F n →L F n.
  • You embed F n into a fixed ambient E by a continuous linear map ι n : F n →L E
    (typically an isometric inclusion).
  • You work with the lifted operator on E:
        lift n := ι n ∘L T n ∘L (ι n)† : E →L E
    († is the Hilbert adjoint). If the ι n are isometries, lift n is the
    usual “conjugate by a partial isometry” and acts like T n on range(ι n).

If the lifted family (lift n) has a uniform top gap and converges in operator
norm to a self-adjoint A on E, then A inherits a positive gap.
-/
section Embedding

variables {F : ℕ → Type*}
  [∀ n, NormedAddCommGroup (F n)] [∀ n, InnerProductSpace 𝕂 (F n)]
  [∀ n, FiniteDimensional 𝕂 (F n)]

/-- Push-forward (lift) of a self-adjoint operator through an embedding into `E`. -/
def liftThrough (ι : ∀ n, (F n) →L[𝕂] E) (T : ∀ n, (F n) →L[𝕂] (F n)) (n : ℕ) :
    E →L[𝕂] E :=
  (ι n).comp ((T n).comp (ContinuousLinearMap.adjoint (ι n)))

/-- If `T n` is self-adjoint on `F n`, then its lift is self-adjoint on `E`. -/
lemma liftThrough_isSelfAdjoint
    (ι : ∀ n, (F n) →L[𝕂] E) (T : ∀ n, (F n) →L[𝕂] (F n))
    (hTsa : ∀ n, IsSelfAdjoint (T n)) :
    ∀ n, IsSelfAdjoint (liftThrough ι T n) :=
by
  intro n
  have hTadj : ContinuousLinearMap.adjoint (T n) = (T n) := by
    simpa [IsSelfAdjoint] using (hTsa n)
  -- Show adjoint equals itself
  change
    ContinuousLinearMap.adjoint ((ι n).comp ((T n).comp (ContinuousLinearMap.adjoint (ι n))))
      = (ι n).comp ((T n).comp (ContinuousLinearMap.adjoint (ι n)))
  -- Use adjoint of a composition and T† = T.
  simpa [liftThrough, ContinuousLinearMap.adjoint_comp, hTadj, adjoint_adjoint, comp_assoc]

/--
**P5 – via norm-convergent embedding.**
If the lifted operators `liftThrough ι T n : E →L E` have a uniform top gap `δ > 0`
and converge in operator norm to a self-adjoint limit `A`, then `A` has a
positive top gap (indeed ≥ δ/2).

This is the ScalingFamily connector: take `F n` as your per-scale spaces,
`ι n` the embeddings into the common `E`, and `T n` your per-scale self-adjoint
operators. You only need the *lifted* family to have a uniform gap and converge.
-/
theorem gap_persists_via_embedding
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ)
    (P1 : ∀ {X Y : E →L[𝕂] E},
      IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    (ι : ∀ n, (F n) →L[𝕂] E)
    (T : ∀ n, (F n) →L[𝕂] (F n))
    {A : E →L[𝕂] E} (hA : IsSelfAdjoint A)
    (hTsa : ∀ n, IsSelfAdjoint (T n))
    {δ : ℝ} (hδpos : 0 < δ)
    (hGapLift : ∀ n, eigGap λ₁ λ₂ (liftThrough ι T n) ≥ δ)
    (hConvLift : ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖(liftThrough ι T n) - A‖ ≤ ε) :
    ∃ δ' > 0, eigGap λ₁ λ₂ A ≥ δ' :=
by
  -- Each lifted operator is self-adjoint:
  have hSA : ∀ n, IsSelfAdjoint (liftThrough ι T n) :=
    liftThrough_isSelfAdjoint ι T hTsa
  -- Apply P2 on the fixed ambient E with the lifted sequence:
  exact gap_persistence λ₁ λ₂ P1 hA hSA hδpos hGapLift hConvLift

end Embedding

end YM
