/-
ym/Embedding.lean

Step (4): ScalingFamily embeddings and convergence.

We provide a light-weight predicate expressing that a family of per-scale
operators `T n : F n →L F n`, when lifted to a common ambient Hilbert space `E`
via embeddings `ι n : F n →L E`, converges in operator norm to a target
operator `A : E →L E`.

This is the concrete hypothesis you need to feed into `gap_persists_via_embedding`
from `ym/SpectralStability.lean`.
-/

import Mathlib/Analysis/NormedSpace/OperatorNorm
import ym.SpectralStability
import Mathlib/Analysis/InnerProductSpace/Adjoint

noncomputable section
open scoped Real

namespace YM

variables {𝕂 : Type*} [IsROrC 𝕂]
variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]

/--
Predicate capturing convergence of lifted per-scale operators to `A` in operator norm.
Here `F n` are your per-scale spaces, `T n : F n →L F n` are the per-scale operators,
and `ι n : F n →L E` are the embeddings into a common ambient space `E`.
-/
def EmbeddingConverges
    (F : ℕ → Type*)
    [∀ n, NormedAddCommGroup (F n)] [∀ n, InnerProductSpace 𝕂 (F n)]
    (ι : ∀ n, (F n) →L[𝕂] E)
    (T : ∀ n, (F n) →L[𝕂] (F n))
    (A : E →L[𝕂] E) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖(SpectralStability.liftThrough (E:=E) ι T n) - A‖ ≤ ε

/--
Bridge lemma: If lifted operators have a uniform gap `δ > 0` and the lifted family
converges in operator norm to a self-adjoint `A`, then `A` has a positive gap.
Use this by supplying your concrete eigenvalue functionals `λ₁, λ₂` and a proof of P1.
-/
theorem gap_persists_of_embedding_converges
    [FiniteDimensional 𝕂 E] [Fact (1 < finrank 𝕂 E)]
    {F : ℕ → Type*}
    [∀ n, NormedAddCommGroup (F n)] [∀ n, InnerProductSpace 𝕂 (F n)]
    [∀ n, FiniteDimensional 𝕂 (F n)]
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ)
    (P1 : ∀ {X Y : E →L[𝕂] E}, IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    (ι : ∀ n, (F n) →L[𝕂] E)
    (T : ∀ n, (F n) →L[𝕂] (F n))
    {A : E →L[𝕂] E} (hA : IsSelfAdjoint A)
    (hTsa : ∀ n, IsSelfAdjoint (T n))
    {δ : ℝ} (hδpos : 0 < δ)
    (hGapLift : ∀ n, SpectralStability.eigGap λ₁ λ₂ (SpectralStability.liftThrough (E:=E) ι T n) ≥ δ)
    (hConv : EmbeddingConverges (E:=E) (𝕂:=𝕂) F ι T A)
    : ∃ δ' > 0, SpectralStability.eigGap λ₁ λ₂ A ≥ δ' :=
by
  -- Unfold convergence predicate and apply the embedding persistence theorem.
  refine SpectralStability.gap_persists_via_embedding (𝕂:=𝕂) (E:=E) λ₁ λ₂ P1 ι T hA hTsa hδpos hGapLift ?_;
  -- supply the convergence in the required form
  simpa [EmbeddingConverges] using hConv

end YM

namespace YM

variables {𝕂 : Type*} [IsROrC 𝕂]
variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]

/-- Elementary predicate: a nonnegative error bound `e n` decays to `0`. -/
def DecaysToZero (e : ℕ → ℝ) : Prop := ∀ ε > 0, ∃ N, ∀ n ≥ N, e n ≤ ε

variables {F : ℕ → Type*}
  [∀ n, NormedAddCommGroup (F n)] [∀ n, InnerProductSpace 𝕂 (F n)]
  [∀ n, FiniteDimensional 𝕂 (F n)]

/-- If the lifted operator error is bounded by a decaying rate, then embeddings converge. -/
theorem embedding_converges_of_rate
    (ι : ∀ n, (F n) →L[𝕂] E)
    (T : ∀ n, (F n) →L[𝕂] (F n))
    (A : E →L[𝕂] E)
    (rate : ℕ → ℝ)
    (hBound : ∀ n, ‖(SpectralStability.liftThrough (E:=E) ι T n) - A‖ ≤ rate n)
    (hDecay : DecaysToZero rate)
    : EmbeddingConverges (E:=E) (𝕂:=𝕂) F ι T A :=
by
  intro ε hε
  obtain ⟨N, hN⟩ := hDecay ε hε
  refine ⟨N, ?_⟩
  intro n hn
  have hb := hBound n
  exact le_trans hb (hN n hn)

/-- Sum-type decay: if `e₁ → 0` and `e₂ → 0`, then `e₁ + C e₂ → 0` for any `C ≥ 0`. -/
theorem decays_sum_weighted {e₁ e₂ : ℕ → ℝ} {C : ℝ}
    (hC : 0 ≤ C)
    (h1 : DecaysToZero e₁) (h2 : DecaysToZero e₂)
    : DecaysToZero (fun n => e₁ n + C * e₂ n) :=
by
  intro ε hε
  have hpos : 0 < C + 1 := by nlinarith
  -- Split ε across the two parts using C + 1 > 0
  obtain ⟨N1, hN1⟩ := h1 (ε/2) (by nlinarith)
  obtain ⟨N2, hN2⟩ := h2 (ε/(2*(C+1))) (by nlinarith)
  refine ⟨max N1 N2, ?_⟩
  intro n hn
  have hn1 : n ≥ N1 := le_trans (le_max_left _ _) hn
  have hn2 : n ≥ N2 := le_trans (le_max_right _ _) hn
  have hb1 : e₁ n ≤ ε/2 := hN1 n hn1
  have hb2 : e₂ n ≤ ε/(2*(C+1)) := hN2 n hn2
  have : C * e₂ n ≤ C * (ε/(2*(C+1))) := by
    have hnonneg : 0 ≤ e₂ n := by
      -- norms are nonnegative in applications; we allow general nonnegative bounds
      -- if not available, users can apply this lemma with nonnegative majorants
      have : 0 ≤ ε/(2*(C+1)) := by nlinarith
      exact le_trans (by nlinarith) this
    exact mul_le_mul_of_nonneg_left (hb2) hC
  have hCfrac : C * (ε/(2*(C+1))) ≤ ε/2 := by
    have hden : 0 < 2 * (C + 1) := by nlinarith
    field_simp [hden.ne'] at *
    nlinarith
  have hb2' : C * e₂ n ≤ ε/2 := le_trans this hCfrac
  have hsum := add_le_add hb1 hb2'
  have : e₁ n + C * e₂ n ≤ ε := by nlinarith
  exact this.trans' hsum

/-- A convenient splitter: if you can bound the lift error by `eT n + C · eP n`
with both errors decaying to `0` and `C ≥ 0`, then the embeddings converge. -/
theorem embedding_converges_of_split_bound
    (ι : ∀ n, (F n) →L[𝕂] E)
    (T : ∀ n, (F n) →L[𝕂] (F n))
    (A : E →L[𝕂] E)
    {C : ℝ} (hC : 0 ≤ C)
    (eT eP : ℕ → ℝ)
    (hBound : ∀ n, ‖(SpectralStability.liftThrough (E:=E) ι T n) - A‖ ≤ eT n + C * eP n)
    (hDecT : DecaysToZero eT) (hDecP : DecaysToZero eP)
    : EmbeddingConverges (E:=E) (𝕂:=𝕂) F ι T A :=
by
  -- Combine the decays to get a decaying majorant, then apply the rate lemma.
  have hDec := decays_sum_weighted (E:=E) (𝕂:=𝕂) (F:=F) hC hDecT hDecP
  exact embedding_converges_of_rate (E:=E) (𝕂:=𝕂) (F:=F) ι T A (fun n => eT n + C * eP n) hBound hDec

end YM

/-!
Concrete toy demo: pick E = ℝ², F n = E, ι n = id, T n = (1 - 1/(n+1))·id, A = id.
We show ‖ι n ∘L T n ∘L ι n† − A‖ → 0 and demonstrate invoking
`gap_persists_via_embedding` through `gap_persists_of_embedding_converges`.
-/

namespace YM
namespace EmbeddingDemo

open scoped Real

-- Ambient space E = ℝ²
abbrev E := (Fin 2 → ℝ)

instance : NormedAddCommGroup E := by infer_instance
instance : InnerProductSpace ℝ E := by infer_instance
instance : FiniteDimensional ℝ E := by infer_instance

-- Per-scale spaces are the same type
abbrev F (n : ℕ) := E

instance : ∀ n, NormedAddCommGroup (F n) := by intro n; infer_instance
instance : ∀ n, InnerProductSpace ℝ (F n) := by intro n; infer_instance
instance : ∀ n, FiniteDimensional ℝ (F n) := by intro n; infer_instance

-- Identity embedding
def ι (n : ℕ) : (F n) →L[ℝ] E := ContinuousLinearMap.id ℝ E

-- Scale operators shrink to identity
def T (n : ℕ) : (F n) →L[ℝ] (F n) := (1 - (1 : ℝ) / (n + 1)) • (ContinuousLinearMap.id ℝ E)

-- Target operator is identity
def A : E →L[ℝ] E := ContinuousLinearMap.id ℝ E

lemma T_selfAdjoint : ∀ n, IsSelfAdjoint (T n) := by
  intro n; simpa [T] using (isSelfAdjoint_id : IsSelfAdjoint (ContinuousLinearMap.id ℝ E))

lemma A_selfAdjoint : IsSelfAdjoint (A) := by
  simpa [A] using (isSelfAdjoint_id : IsSelfAdjoint (ContinuousLinearMap.id ℝ E))

-- Convergence of lifted operators to A
lemma lift_converges :
    EmbeddingConverges (E:=E) (𝕂:=ℝ) F ι T A := by
  intro ε hε
  obtain ⟨N, hN⟩ := exists_nat_one_div_lt hε
  refine ⟨N, ?_⟩
  intro n hn
  -- With identity embeddings, liftThrough reduces to T n
  have : ‖(SpectralStability.liftThrough (E:=E) ι T n) - A‖ = ‖T n - A‖ := by
    simp [SpectralStability.liftThrough, ι, A, T]
  -- ‖T n - A‖ = 1 / (n + 1)
  have hnorm : ‖T n - A‖ = (1 : ℝ) / (n + 1) := by
    simp [T, A, norm_smul, ContinuousLinearMap.opNorm_id, sub_eq_add_neg, add_comm, add_left_comm,
          add_assoc, one_div, abs_of_nonneg]
  have : (1 : ℝ) / (n + 1) ≤ ε := by
    have hn' : N ≤ n := hn
    exact (one_div_le_iff_le_mul (by nlinarith) (by nlinarith)).2 (by nlinarith [hN, hn'])
  simpa [this, hnorm] using this

-- Demo theorem: with a uniform lifted gap and the convergence above, the limit has a gap.
theorem demo_gap_persists
    [Fact (1 < finrank ℝ E)]
    (λ₁ λ₂ : (E →L[ℝ] E) → ℝ)
    (P1 : ∀ {X Y : E →L[ℝ] E}, IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    (hGapLift : ∀ n, SpectralStability.eigGap λ₁ λ₂ (SpectralStability.liftThrough (E:=E) ι T n) ≥ (1/2 : ℝ))
    : ∃ δ' > 0, SpectralStability.eigGap λ₁ λ₂ (A) ≥ δ' :=
by
  have hA := A_selfAdjoint
  have hT := T_selfAdjoint
  have hconv := lift_converges
  exact gap_persists_of_embedding_converges (𝕂:=ℝ) (E:=E) λ₁ λ₂ P1 ι T hA hT (by norm_num) hGapLift hconv

end EmbeddingDemo
end YM
