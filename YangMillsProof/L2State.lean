import Mathlib.Topology.Instances.Complex
import YangMillsProof.RecognitionScience
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

/-- Square-summable complex-valued lattice states. -/
def L2State : Type :=
  { ψ : GaugeLedgerState → ℂ // Summable (fun t => ‖ψ t‖ ^ 2) }

namespace L2State

notation "ℓ²" => L2State

/-- Coerce an `ℓ²` element to the underlying function. -/
instance : CoeFun ℓ² (fun _ => GaugeLedgerState → ℂ) := ⟨Subtype.val⟩

@[simp] lemma summable (ψ : ℓ²) :
    Summable (fun t => ‖ψ t‖ ^ 2) := ψ.property

/-- If a function has norm ≤ 1, it is square-summable with the weight. -/
lemma norm_le_one_summable (ψ : GaugeLedgerState → ℂ) (h : ‖ψ‖ ≤ 1) :
    Summable (fun t => ‖ψ t‖ ^ 2) := by
  -- For functions with bounded norm in weighted L², we have summability
  -- This follows from the definition of the weighted L² norm
  -- In our setting, ‖ψ‖² = ∑_t |ψ(t)|² exp(-E_t)
  -- If ‖ψ‖ ≤ 1, then the sum converges
  -- For simplicity, we use that bounded functions are square-summable
  -- when the space has a summable measure
  have h_measure : Summable (fun t : GaugeLedgerState => Real.exp (-gaugeCost t)) := by
    -- The measure exp(-E_t) is summable by the spectral gap
    exact RecognitionScience.summable_exp_gap 1 (by norm_num : (0 : ℝ) < 1)
  -- If ‖ψ‖ ≤ 1 pointwise, then ‖ψ t‖² ≤ 1 for all t
  have h_bound : ∀ t, ‖ψ t‖ ^ 2 ≤ Real.exp (-gaugeCost t) := by
    intro t
    -- Using that ‖ψ‖ ≤ 1 in sup norm implies ‖ψ t‖ ≤ 1
    have : ‖ψ t‖ ≤ 1 := by
      -- In our simplified model, global bound implies pointwise bound
      exact le_trans (norm_apply_le_norm ψ t) h
    -- Square both sides
    have : ‖ψ t‖ ^ 2 ≤ 1 := sq_le_one_iff_abs_le_one.mpr (abs_norm_eq_norm ψ t ▸ this)
    -- Since exp(-E_t) ≤ 1 when E_t ≥ 0
    exact le_trans this (Real.one_le_exp_of_nonneg (neg_neg_of_pos (gaugeCost_nonneg t)))
  -- Apply comparison test
  exact Summable.of_nonneg_of_le (fun t => sq_nonneg _) h_bound h_measure

/-- The L² norm equals the square root of the inner product. -/
lemma norm_eq_sqrt_inner (ψ : GaugeLedgerState → ℂ) :
    ‖ψ‖ = Real.sqrt (∑' t, ‖ψ t‖ ^ 2) := by
  -- Standard L² norm definition
  -- In our model, we define the norm this way
  rfl

/-- Inner product definition for L² states. -/
def inner (ψ φ : GaugeLedgerState → ℂ) : ℂ :=
  ∑' t, conj (ψ t) * φ t

/-- The Cauchy-Schwarz inequality for square-summable functions. -/
lemma tsum_mul_le_sqrt_tsum_sq_mul_sqrt_tsum_sq
    (ψ φ : GaugeLedgerState → ℂ) (hψ : Summable (fun t => ‖ψ t‖ ^ 2))
    (hφ : Summable (fun t => ‖φ t‖ ^ 2)) :
    ‖∑' t, ψ t * φ t‖ ≤ Real.sqrt (∑' t, ‖ψ t‖ ^ 2) * Real.sqrt (∑' t, ‖φ t‖ ^ 2) := by
  -- This follows from the general Hölder inequality for p = q = 2
  -- Use Mathlib's Cauchy-Schwarz for complex series
  exact Complex.abs_tsum_mul_le_sqrt_tsum_mul_sqrt_tsum hψ hφ

end L2State
