/-!
Kato projector continuity for a simple, isolated eigenvalue (finite dimension).

We provide a placeholder-free continuity statement specialized to matrices over ℂ:
on the open set where `trace (adj (λ•I - A)) ≠ 0`, the map `(A,λ) ↦ P(A,λ)` is
continuous in operator norm. This suffices for the pipeline stability composition
with P1; quantitative Lipschitz constants can be supplied later by B4.
-/

import Mathlib
import Mathlib/LinearAlgebra/Matrix/Adjugate
import Mathlib/Topology/Basic
import Mathlib/Topology/Instances/Complex

noncomputable section

namespace YM.Kato

open Matrix Topology

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Raw numerator `Adj(λ•I - A)`. -/
@[simp] def katoNumerator (A : Matrix ι ι ℂ) (λ : ℂ) : Matrix ι ι ℂ :=
  adjugate (λ • (1 : Matrix ι ι ℂ) - A)

/-- Raw denominator `tr (Adj(λ•I - A))`. -/
@[simp] def katoDen (A : Matrix ι ι ℂ) (λ : ℂ) : ℂ := trace (katoNumerator A λ)

/-- Kato projector: zero if denominator vanishes, else normalized adjugate. -/
def katoProj (A : Matrix ι ι ℂ) (λ : ℂ) : Matrix ι ι ℂ :=
  if h : katoDen A λ = 0 then 0 else (katoDen A λ)⁻¹ • katoNumerator A λ

@[simp] lemma katoProj_eq (A : Matrix ι ι ℂ) (λ : ℂ) (h : katoDen A λ ≠ 0) :
    katoProj A λ = (katoDen A λ)⁻¹ • katoNumerator A λ := by
  simp [katoProj, h]

/-- Continuity of `katoProj` on the open set where the denominator is nonzero. -/
theorem continuousAt_katoProj
    (A : Matrix ι ι ℂ) (λ : ℂ) (hden : katoDen A λ ≠ 0) :
    ContinuousAt (fun p : Matrix ι ι ℂ × ℂ => katoProj p.1 p.2) (A, λ) := by
  classical
  -- On a neighborhood where the denominator stays nonzero, use the explicit formula.
  have hcontAdj : Continuous fun p : Matrix ι ι ℂ × ℂ =>
      katoNumerator p.1 p.2 := by
    -- adjugate is polynomial in entries; composition is continuous
    refine (Matrix.continuous_adjugate.comp ?_)
    exact (continuous_snd.smul continuous_const).sub continuous_fst
  have hcontDen : Continuous fun p : Matrix ι ι ℂ × ℂ => katoDen p.1 p.2 :=
    hcontAdj.trace
  have hopen : IsOpen {p : Matrix ι ι ℂ × ℂ | katoDen p.1 p.2 ≠ 0} := by
    have : Continuous fun p : Matrix ι ι ℂ × ℂ => katoDen p.1 p.2 := hcontDen
    simpa [isClosed_eq, isOpen_compl_iff] using
      (this.isClosed_preimage continuous_const isClosed_singleton).isOpen_compl
  have hmem : (A, λ) ∈ {p : Matrix ι ι ℂ × ℂ | katoDen p.1 p.2 ≠ 0} := by simpa using hden
  -- Compose continuous maps on this open set: (adj), (trace), inversion, and smul.
  have hcontInv :
      ContinuousAt (fun p : Matrix ι ι ℂ × ℂ => (katoDen p.1 p.2)⁻¹) (A, λ) := by
    refine (hcontDen.continuousAt.inv₀ ?_)
    simpa using hden
  -- Finally, use the explicit formula equality on the open set.
  have hloc : ContinuousAt (fun p => (katoDen p.1 p.2)⁻¹ • katoNumerator p.1 p.2) (A, λ) := by
    exact hcontInv.smul (hcontAdj.continuousAt)
  -- Since both sides agree on a neighborhood, their `ContinuousAt` coincide.
  refine ContinuousAt.congr_of_loc_eq hloc ?eqnhd
  -- Provide the local equality near (A,λ).
  have : {p : Matrix ι ι ℂ × ℂ | katoDen p.1 p.2 ≠ 0}.IsOpen := hopen
  have : ∀ᶠ p in 𝓝 (A, λ), katoDen p.1 p.2 ≠ 0 :=
    IsOpen.mem_nhds hopen hmem
  filter_upwards [this] with p hp
  simp [katoProj, hp]

end YM.Kato
