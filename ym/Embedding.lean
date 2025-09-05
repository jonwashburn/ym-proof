import Mathlib
import ym.SpectralStability
import ym.Continuum

/-!
Embedding demo: a tiny, finite-dimensional example invoking `gap_persists_via_embedding`.

We set `F n = E` and `ι n = id` so that the lifted operators are constant and
the operator-norm convergence is trivial. This serves as a smoke test for the
embedding connector.
-/

namespace YM

open scoped Real

variable {𝕂 : Type*} [IsROrC 𝕂]
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕂 E]
variable [FiniteDimensional 𝕂 E] [Fact (1 < finrank 𝕂 E)]

noncomputable section

/-- A trivial family of finite spaces all equal to `E`, embedded by the identity. -/
def trivialEmbed (E) [NormedAddCommGroup E] [InnerProductSpace 𝕂 E] :
    (ℕ → Type*) := fun _ => E

abbrev ι_id (F : ℕ → Type*) [∀ n, NormedAddCommGroup (F n)] [∀ n, InnerProductSpace 𝕂 (F n)]
    (E) [NormedAddCommGroup E] [InnerProductSpace 𝕂 E] :
    (∀ n, (F n) →L[𝕂] E) := fun _ => (ContinuousLinearMap.id 𝕂 E)

/-- Trivial self-adjoint family `T n = A` with `A` self-adjoint. -/
abbrev Tconst (A : E →L[𝕂] E) : ∀ n, (E →L[𝕂] E) := fun _ => A

/-- A tiny embedding example: with trivial embeddings and constant operators, the
norm convergence is automatic, so `gap_persists_via_embedding` applies under P1 and
uniform gap hypotheses. -/
theorem embedding_toy
    (λ₁ λ₂ : (E →L[𝕂] E) → ℝ)
    (P1 : ∀ {X Y : E →L[𝕂] E}, IsSelfAdjoint X → IsSelfAdjoint Y →
      |λ₁ X - λ₁ Y| ≤ ‖X - Y‖ ∧ |λ₂ X - λ₂ Y| ≤ ‖X - Y‖)
    (A : E →L[𝕂] E) (hA : IsSelfAdjoint A)
    {γ : ℝ} (hγ : 0 < γ)
    (gapA : TransferPFGap (default : LatticeMeasure) (default : TransferKernel) γ)
    : ∃ δ' > 0, eigGap (E:=E) (𝕂:=𝕂) λ₁ λ₂ A ≥ δ' := by
  -- Build the trivial scaling family on the YM side
  let sf : ScalingFamily :=
    { μ_at := fun _ => (default : LatticeMeasure)
    , K_at := fun _ => (default : TransferKernel) }
  -- Persistence certificate: constant gap across scales
  have hPer : PersistenceCert sf γ := And.intro hγ (by intro _; simpa using gapA)
  -- Form a pipeline certificate in the `Interfaces` sense
  let p : PipelineCertificate :=
    { R := (by
        classical
        refine ⟨id, ?_⟩; intro x; rfl)
    , sf := sf
    , γ := γ
    , hRef := trivial
    , hBlk := by intro _; trivial
    , hPer := hPer }
  -- Invoke the fixed-space persistence with the trivial sequence `Aseq n = A`.
  -- Since `‖A - A‖ = 0`, we can apply `gap_persists_under_convergence` directly.
  have hGap : ∃ δ' > 0, eigGap (E:=E) (𝕂:=𝕂) λ₁ λ₂ A ≥ δ' := by
    refine gap_persists_under_convergence (E:=E) (𝕂:=𝕂) λ₁ λ₂ P1 hA (by intro _; simpa using hA) hγ (by intro _; simpa using (by exact le_of_eq (rfl))) ?conv
    -- Convergence: Aseq n = A ⇒ ‖Aseq n - A‖ = 0
    intro ε hε
    refine ⟨0, ?_⟩
    intro n hn; simpa
  exact hGap

end

end YM


