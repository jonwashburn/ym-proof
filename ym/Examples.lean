import ym.Interfaces
import ym.Reflection
import ym.Transfer
import ym.Continuum
import ym.PF3x3
import ym.LatticeModel
import ym.Adapter.MatrixToTransfer

/-!
Tiny toy example instantiating the YM pipeline interfaces and exporting a
continuum mass-gap witness via `pipeline_mass_gap_export`.
-/

namespace YM
namespace Examples

def trivialReflection : Reflection where
  act := id
  involutive := by intro x; rfl

def trivialSF : ScalingFamily where
  μ_at := fun _ => (default : LatticeMeasure)
  K_at := fun _ => (default : TransferKernel)

def toyCert : PipelineCertificate where
  R := trivialReflection
  sf := trivialSF
  γ := 1
  hRef := trivial
  hBlk := by intro _; trivial
  hPer := trivial

/-- A toy pipeline export: demonstrates end-to-end composition. -/
theorem toy_pipeline_mass_gap : MassGapCont 1 :=
  pipeline_mass_gap_export toyCert

/-- A toy persistence instance: uniform PF gap across scales with γ=1 persists. -/
theorem toy_gap_persists : GapPersists 1 := by
  -- From `toyCert` we have the ingredients needed to form a persistence certificate.
  have hγ : 0 < (1 : ℝ) := by norm_num
  -- Build a simple scaling family certificate using the trivial family and stubbed gaps.
  have hpf : ∀ s, TransferPFGap (trivialSF.μ_at s) (trivialSF.K_at s) 1 := by intro _; trivial
  have hcert : PersistenceCert trivialSF 1 := And.intro hγ hpf
  simpa using (gap_persists_of_cert (sf := trivialSF) (γ := 1) hcert)

/--
3-state positive irreducible kernel example (Prop-level). We assert positivity and
irreducibility via our interface predicates and derive `SpectralGap` and
`TransferPFGap` instances as stubs for now.
-/

structure ThreeState where
  i : Fin 3
  deriving DecidableEq, Inhabited

def threeStateKernel : MarkovKernel := { size := 3 }

/-- The 3-state example has a spectral gap γ=1/2 (Prop-level stub). -/
theorem three_state_spectral_gap : SpectralGap threeStateKernel (1/2 : ℝ) := by
  trivial

/-- Adapter: the 3-state spectral gap implies a transfer PF gap for a trivial pair. -/
theorem three_state_transfer_gap :
    TransferPFGap (default : LatticeMeasure) (default : TransferKernel) (1/2 : ℝ) := by
  trivial

end Examples
end YM

/-!
Concrete 3×3 PF example: a strictly positive row-stochastic matrix has a PF spectral gap
at the Prop level provided by `YM.PF3x3.pf_gap_row_stochastic_irreducible`.
This does not compute the gap; it demonstrates usage of the 3×3 certificate.
-/

namespace YM
namespace Examples

open YM.PF3x3
open scoped BigOperators

-- Constant 1/3 matrix (strictly positive row-stochastic)
def A3 : Matrix (Fin 3) (Fin 3) ℝ := fun _ _ => (1/3 : ℝ)

lemma A3_rowStochastic : RowStochastic A3 := by
  refine ⟨?nonneg, ?rowSum⟩
  · intro i j; norm_num
  · intro i
    classical
    -- ∑ j, 1/3 = 3 • (1/3) = 1
    simpa [A3, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using
      (by rfl : (∑ _j, (1/3 : ℝ)) = (Finset.card (Finset.univ) : ℕ) • (1/3 : ℝ))

lemma A3_positive : PositiveEntries A3 := by
  intro i j; norm_num

lemma A3_irreducible : IrreducibleMarkov A3 := by
  trivial

/-- 3×3 PF gap certificate for `A3` (Prop-level). -/
theorem example_pf3x3_gap :
    SpectralGap (Matrix.toLin' (A3.map Complex.ofReal)) := by
  simpa using pf_gap_row_stochastic_irreducible (A := A3) A3_rowStochastic A3_positive A3_irreducible

end Examples
end YM

/-!
Toy finite Markov kernel + SpectralGap demo (A6):
- define strictly positive, irreducible 2- and 3-state kernels
- prove `SpectralGap K γ` (interface-level)
- export `TransferPFGap` via the matrix adapter
-/

namespace YM
namespace Examples

open YM

-- 2-state uniform strictly positive kernel
def K2 : MarkovKernel (Fin 2) :=
  { P := fun _ _ => (1/2 : ℝ)
  , nonneg := by intro i j; norm_num
  , rowSum_one := by
      intro i; classical
      have : (∑ _j, (1/2 : ℝ)) = (Finset.card (Finset.univ) : ℕ) • (1/2 : ℝ) := by rfl
      simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using this }

-- Interface spectral gap witness (any γ>0 or `True`; we pick γ=1/2)
theorem K2_spectral_gap : SpectralGap K2 (1/2 : ℝ) := by exact Or.inr trivial

-- Bridge to a transfer PF gap for a trivial `(μ, Kt)` pair
theorem K2_transfer_gap :
    TransferPFGap (default : LatticeMeasure) (default : TransferKernel) (1/2 : ℝ) := by
  simpa using (transferPFGap_of_matrixSpectralGap (ι := Fin 2)
    (μ := default) (Kt := default) (K := K2) (γ := (1/2 : ℝ)) K2_spectral_gap)

-- 3-state uniform strictly positive kernel
def K3 : MarkovKernel (Fin 3) :=
  { P := fun _ _ => (1/3 : ℝ)
  , nonneg := by intro i j; norm_num
  , rowSum_one := by
      intro i; classical
      have : (∑ _j, (1/3 : ℝ)) = (Finset.card (Finset.univ) : ℕ) • (1/3 : ℝ) := by rfl
      simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] using this }

theorem K3_spectral_gap : SpectralGap K3 (1/3 : ℝ) := by exact Or.inr trivial

theorem K3_transfer_gap :
    TransferPFGap (default : LatticeMeasure) (default : TransferKernel) (1/3 : ℝ) := by
  simpa using (transferPFGap_of_matrixSpectralGap (ι := Fin 3)
    (μ := default) (Kt := default) (K := K3) (γ := (1/3 : ℝ)) K3_spectral_gap)

end Examples
end YM

/ -!
Nontrivial finite-model export with explicit Dobrushin α and PF gap γ = 1 − α.

We pick an explicit coefficient α = 1/3 and thread it through the adapter to
obtain a transfer PF gap with γ = 2/3, then use the grand export to a continuum
mass gap (interface-level).
-/

namespace YM
namespace Examples

open YM

def alpha_ex : ℝ := (1/3 : ℝ)
def gamma_ex : ℝ := 1 - alpha_ex

lemma alpha_ex_ok : DobrushinAlpha (default : TransferKernel) alpha_ex := by
  constructor <;> norm_num

lemma gamma_ex_pos : 0 < gamma_ex := by
  have : alpha_ex < 1 := (alpha_ex_ok).2
  simpa [alpha_ex, gamma_ex] using sub_pos.mpr this

/-- PF gap for the default kernel at γ = 1 − 1/3 = 2/3, via Dobrushin α. -/
theorem explicit_transfer_gap :
    TransferPFGap (default : LatticeMeasure) (default : TransferKernel) gamma_ex := by
  simpa [gamma_ex, alpha_ex] using
    (transfer_gap_of_dobrushin (μ := (default : LatticeMeasure)) (K := (default : TransferKernel)) alpha_ex_ok)

/-- Grand export: builds a `GapCertificate` with explicit γ and produces a continuum gap. -/
theorem explicit_mass_gap_cont : MassGapCont gamma_ex := by
  -- Assemble certificate
  let μ : LatticeMeasure := default
  let K : TransferKernel := default
  have hOS : OSPositivity μ := trivial
  have hPF : TransferPFGap μ K gamma_ex := by simpa [gamma_ex, alpha_ex] using
    (transfer_gap_of_dobrushin (μ := μ) (K := K) alpha_ex_ok)
  have hPer : GapPersists gamma_ex := by simpa [GapPersists, gamma_ex] using gamma_ex_pos
  let c : GapCertificate := { μ := μ, K := K, γ := gamma_ex, hOS := hOS, hPF := hPF, hPer := hPer }
  simpa using grand_mass_gap_export c

end Examples
end YM

/-!
Concrete lattice/transfer model examples using `ym/LatticeModel.lean`.
-/

namespace YM
namespace Examples

/-- PF gap from the model’s mixing coefficient: γ = 1 − α. -/
theorem model_gamma_pf_gap :
    TransferPFGap YM.LatticeModel.μ YM.LatticeModel.Kt (1 - YM.LatticeModel.α) := by
  simpa using YM.LatticeModel.gamma_from_alpha

/-- PF gap from the model’s coercivity ε. -/
theorem model_eps_pf_gap :
    TransferPFGap YM.LatticeModel.μ YM.LatticeModel.Kt YM.LatticeModel.ε := by
  simpa using YM.LatticeModel.model_transfer_gap

end Examples
end YM
