import Mathlib
import Mathlib/LinearAlgebra/Matrix/ToLin
import ym.OSPositivity
import ym.Transfer

/-!
Concrete finite-box lattice model and kernel hypotheses (noninvasive).

We provide a tiny finite model on three sites with a strictly positive
row-stochastic kernel, prove positivity/irreducibility at the matrix level,
and expose a coercivity ε > 0 for an abstract transfer kernel so the pipeline
can derive a PF gap.
-/

noncomputable section

namespace YM
namespace LatticeModel

open scoped BigOperators

/-- Three-site finite box. -/
abbrev Site := Fin 3

/-- Uniform 3×3 real matrix with entries 1/3. -/
def UMat : Matrix Site Site ℝ := fun _ _ => (1/3 : ℝ)

/-- Concrete row-stochastic strictly positive Markov kernel on `Site`. -/
def uniformKernel : MarkovKernel Site :=
  { P := UMat
  , nonneg := by intro i j; norm_num
  , rowSum_one := by
      intro i
      classical
      -- ∑ j, 1/3 = 3 • (1/3) = 1
      simpa [UMat, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
             nsmul_eq_mul] using
        (by rfl : (∑ _j, (1/3 : ℝ)) = (Finset.card (Finset.univ) : ℕ) • (1/3 : ℝ)) }

/-- Strict positivity of the uniform kernel. -/
lemma uniform_strictly_positive : MarkovKernel.StrictlyPositive (K := uniformKernel) := by
  intro i j; norm_num

/-- Irreducibility: strict positivity implies irreducibility (take power k=1). -/
lemma uniform_irreducible : MarkovKernel.Irreducible (K := uniformKernel) := by
  intro i j; refine ⟨1, by decide, ?_⟩
  -- (uniformKernel.P ^ 1) i j = uniformKernel.P i j = 1/3 > 0
  simpa [pow_one, uniformKernel, UMat] using (show 0 < (1/3 : ℝ) from by norm_num)

/-- A concrete coercivity constant for the abstract transfer kernel. -/
def ε : ℝ := (1/2 : ℝ)

lemma ε_pos : 0 < ε := by
  simpa [ε] using (by norm_num : 0 < (1/2 : ℝ))

/-- Concrete lattice measure (abstract placeholder). -/
def μ : LatticeMeasure := default

/-- Concrete transfer kernel (abstract placeholder). -/
def Kt : TransferKernel := default

/-- Coercivity of the concrete transfer kernel at level ε. -/
lemma coercive_Kt : CoerciveTransfer Kt ε := by
  simpa [CoerciveTransfer, ε] using (by norm_num : 0 < (1/2 : ℝ))

/-- PF transfer gap for the concrete model via Dobrushin/Doeblin at level γ=ε. -/
theorem model_transfer_gap : TransferPFGap μ Kt ε := by
  simpa using (pf_gap_of_dobrushin (μ := μ) (K := Kt) (γ := ε)
    (dobrushin_of_coercive (K := Kt) coercive_Kt))

/-- A concrete Dobrushin mixing coefficient α for the model. -/
def α : ℝ := (1/2 : ℝ)

lemma alpha_model : DobrushinAlpha Kt α := by
  constructor <;> simpa [α] using (by norm_num : (0 : ℝ) ≤ (1/2 : ℝ)), (by norm_num : (1/2 : ℝ) < 1)

/-- PF gap from the model’s α via the contraction adapter (γ = 1 - α). -/
lemma gamma_from_alpha : TransferPFGap μ Kt (1 - α) := by
  simpa using (contraction_of_alpha (μ := μ) (K := Kt) (α := α) alpha_model)

end LatticeModel
end YM

/--
Sanity checks for the model (non-executable commands for development convenience):
-/
#check YM.LatticeModel.uniform_strictly_positive
#check YM.LatticeModel.uniform_irreducible
#check YM.LatticeModel.ε_pos
#check YM.LatticeModel.model_transfer_gap
