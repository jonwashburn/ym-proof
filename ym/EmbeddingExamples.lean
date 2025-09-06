import Mathlib
import ym.SpectralStability
import ym.Embedding

/-!
Toy finite-dimensional embedding example for step (4).

We take E = ℝ² with the identity as the target operator A, and define a simple
family F n = ℝ² with ι n = id and T n = (1 - 1/(n+1)) · id, so that
ι n ∘ T n ∘ ι n† = T n → A in operator norm. The gap persistence lemma applies
trivially once a uniform gap is assumed for the lifted family.
-/

noncomputable section

namespace YM
namespace Examples

open scoped Real

def E := (Fin 2 → ℝ)

instance : NormedAddCommGroup E := by infer_instance
instance : InnerProductSpace ℝ E := by infer_instance
instance : FiniteDimensional ℝ E := by infer_instance

abbrev F (n : ℕ) := E

instance : ∀ n, NormedAddCommGroup (F n) := by intro n; infer_instance
instance : ∀ n, InnerProductSpace ℝ (F n) := by intro n; infer_instance
instance : ∀ n, FiniteDimensional ℝ (F n) := by intro n; infer_instance

-- Embeddings are identities
def ι (n : ℕ) : (F n) →L[ℝ] E := ContinuousLinearMap.id ℝ E

-- Per-scale operators shrink towards the identity: T n = (1 - 1/(n+1)) · id
def T (n : ℕ) : (F n) →L[ℝ] (F n) :=
  (1 - (1 : ℝ) / (n + 1)) • (ContinuousLinearMap.id ℝ E)

-- Target operator on E is the identity
def A : E →L[ℝ] E := ContinuousLinearMap.id ℝ E

-- Self-adjointness facts
lemma T_selfAdjoint : ∀ n, IsSelfAdjoint (T n) := by
  intro n; simpa [T] using (isSelfAdjoint_id : IsSelfAdjoint (ContinuousLinearMap.id ℝ E))

lemma A_selfAdjoint : IsSelfAdjoint (A) := by
  simpa [A] using (isSelfAdjoint_id : IsSelfAdjoint (ContinuousLinearMap.id ℝ E))

-- Convergence of lifted operators to A
lemma lift_converges :
    EmbeddingConverges (E:=E) (𝕂:=ℝ) F ι T A := by
  -- Here liftThrough ι T n = T n since ι n = id and (ι n)† = id
  -- ‖T n - A‖ = |1 - 1/(n+1) - 1| * ‖id‖ = (1/(n+1)) * 1 → 0
  intro ε hε
  obtain ⟨N, hN⟩ := exists_nat_one_div_lt hε
  refine ⟨N, ?_⟩
  intro n hn
  -- bound: ‖T n - A‖ = 1 / (n + 1)
  have : ‖(SpectralStability.liftThrough (E:=E) ι T n) - A‖
        = ‖(T n) - A‖ := by
    -- liftThrough with identity embeddings reduces to T n
    simp [SpectralStability.liftThrough, ι, A, T]
  have : ‖(T n) - A‖ = (1 : ℝ) / (n + 1) := by
    -- operator norm of scalar • id minus id is |(1 - 1/(n+1)) - 1| = 1/(n+1)
    simp [T, A, norm_smul, ContinuousLinearMap.opNorm_id, sub_eq_add_neg, add_comm, add_left_comm,
          add_assoc, one_div, abs_of_nonneg]
  have hfrac : (1 : ℝ) / (n + 1) ≤ ε := by
    have hn' : N ≤ n := hn
    exact (one_div_le_iff_le_mul (by nlinarith) (by nlinarith)).2 (by nlinarith [hN, hn'])
  simpa [this] using hfrac

-- A toy assumption of uniform top gap for lifted operators at δ=1/2 and a consequence
theorem toy_gap_persists_via_embedding
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
  exact gap_persists_of_embedding_converges (𝕂:=ℝ) (E:=E)
    λ₁ λ₂ P1 ι T hA hT (by norm_num) hGapLift hconv

end Examples
end YM
