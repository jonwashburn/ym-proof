/-
P2 (Gap persistence in the limit)

Specialized to finite-dimensional operators on an inner-product space over ℝ or ℂ.

Preconditions (all are explicit in the statement below):
• 𝕂 is ℝ or ℂ (IsROrC 𝕂).
• E is a finite-dimensional inner-product space over 𝕂 with finrank ≥ 2.
• A : E →ₗ[𝕂] E and each Aseq n are self-adjoint.
• Uniform gap: for some δ > 0, (λ₁ (Aseq n) - λ₂ (Aseq n)) ≥ δ for all n.
• Operator-norm convergence: ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖Aseq n - A‖ ≤ ε.
• P1 (to be supplied by the user/project): for self-adjoint X,Y, each of λ₁, λ₂ is 1‑Lipschitz:
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖   and   |λ₂ X - λ₂ Y| ≤ ‖X - Y‖.

Conclusion:
• ∃ δ' > 0, (λ₁ A - λ₂ A) ≥ δ' (indeed one can take δ' = δ/2).

This file treats λ₁, λ₂ abstractly as functions (E →ₗ[𝕂] E) → ℝ that, on self-adjoint
operators, return the largest and second-largest eigenvalues in ℝ. You instantiate them
with your concrete definitions from P1 (e.g. min–max ordered eigenvalues).
-/

import Mathlib/Analysis/InnerProductSpace/Adjoint
import Mathlib/Analysis/NormedSpace/OperatorNorm
import Mathlib/Tactic

noncomputable section
open scoped Real
open Filter

variables {𝕂 : Type*} [IsROrC 𝕂]
variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]
variables [FiniteDimensional 𝕂 E] [Fact (1 < finrank 𝕂 E)]

/-- The spectral gap functional built from the top two (ordered) eigenvalue functionals. -/
def eigGap (λ₁ λ₂ : (E →ₗ[𝕂] E) → ℝ) (T : E →ₗ[𝕂] E) : ℝ :=
  λ₁ T - λ₂ T

/--
**P2 (Gap persistence in the limit).**

Let `(Aseq n)` be self-adjoint operators on a finite-dimensional inner-product space `E`
over `𝕂 ∈ {ℝ, ℂ}`, and suppose there is a **uniform** top-eigenvalue gap `δ > 0`:
`λ₁ (Aseq n) - λ₂ (Aseq n) ≥ δ` for all `n`. If `Aseq n → A` in operator norm, then
`A` has a positive gap as well; in fact, `λ₁ A - λ₂ A ≥ δ/2`.

This uses **P1** (Lipschitz stability of the ordered top two eigenvalues) as a hypothesis.
-/
theorem gap_persistence
    (λ₁ λ₂ : (E →ₗ[𝕂] E) → ℝ)
    (P1 : ∀ {X Y : E →ₗ[𝕂] E},
      IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    {A : E →ₗ[𝕂] E} {Aseq : ℕ → E →ₗ[𝕂] E}
    (hA  : IsSelfAdjoint A)
    (hAn : ∀ n, IsSelfAdjoint (Aseq n))
    {δ : ℝ} (hδpos : 0 < δ)
    (hGap  : ∀ n, eigGap λ₁ λ₂ (Aseq n) ≥ δ)
    (hConv : ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖Aseq n - A‖ ≤ ε) :
    ∃ δ' > 0, eigGap λ₁ λ₂ A ≥ δ' :=
by
  -- Pick N so that ‖Aseq N - A‖ ≤ δ/4.
  obtain ⟨N, hN⟩ := hConv (δ / 4) (by nlinarith)
  have hNnorm : ‖A - Aseq N‖ ≤ δ / 4 := by
    have := hN N (le_rfl)
    simpa [norm_sub_rev] using this

  -- From P1: control λ₁ and λ₂ at A via their values at Aseq N.
  have hλ1_abs : |λ₁ A - λ₁ (Aseq N)| ≤ ‖A - Aseq N‖ := (P1 hA (hAn N)).1
  have hλ2_abs : |λ₂ A - λ₂ (Aseq N)| ≤ ‖A - Aseq N‖ := (P1 hA (hAn N)).2
  have hλ1_low : λ₁ (Aseq N) - ‖A - Aseq N‖ ≤ λ₁ A :=
    (abs_sub_le_iff.mp hλ1_abs).1
  have hλ2_high : λ₂ A ≤ λ₂ (Aseq N) + ‖A - Aseq N‖ :=
    (abs_sub_le_iff.mp hλ2_abs).2

  -- Lower bound the gap of A in terms of the gap of Aseq N and ‖A - Aseq N‖.
  have h_main :
      (λ₁ (Aseq N) - ‖A - Aseq N‖) - (λ₂ (Aseq N) + ‖A - Aseq N‖)
        ≤ eigGap λ₁ λ₂ A :=
  by
    have := sub_le_sub hλ1_low hλ2_high
    simpa [eigGap, sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using this

  have gap_ge : eigGap λ₁ λ₂ A
                  ≥ (eigGap λ₁ λ₂ (Aseq N)) - 2 * ‖A - Aseq N‖ :=
  by
    -- (x - z) - (y + z) = (x - y) - 2 z
    have : (λ₁ (Aseq N) - ‖A - Aseq N‖) - (λ₂ (Aseq N) + ‖A - Aseq N‖)
            = (eigGap λ₁ λ₂ (Aseq N)) - 2 * ‖A - Aseq N‖ := by
      ring
    have := h_main
    simpa [this] using this

  -- Insert the uniform gap and the δ/4 proximity to get δ/2.
  have gap_ge_delta_minus : eigGap λ₁ λ₂ A ≥ δ - 2 * ‖A - Aseq N‖ := by
    -- combine gap_ge with hGap N
    linarith [gap_ge, hGap N]
  have gap_ge_half : eigGap λ₁ λ₂ A ≥ δ / 2 := by
    -- use ‖A - Aseq N‖ ≤ δ/4
    linarith [gap_ge_delta_minus, hNnorm]

  exact ⟨δ / 2, by nlinarith [hδpos], gap_ge_half⟩
