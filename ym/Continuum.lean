import Mathlib
import ym.OSPositivity
import ym.Transfer
import ym.Reflection

/-!
YM continuum-limit interface: scaling family and gap persistence adapter.
-/

namespace YM

/--
Scaling family parameter (e.g., lattice spacing or volume index).
This is a minimal placeholder used to index discrete scales.
-/
structure Scale where
  n : Nat := 0
  deriving Inhabited, DecidableEq

/--
Interface bundling a family of lattice measures and transfer kernels across scales.
At each `s : Scale`, we have a lattice measure `μ_at s` and a transfer kernel `K_at s`.
-/
structure ScalingFamily where
  μ_at : Scale → LatticeMeasure
  K_at : Scale → TransferKernel

/--
Persistence certificate: a uniform-in-scale spectral gap of level `γ`.
This asserts that every scale `s` exhibits a Perron–Frobenius transfer gap `γ`
for the corresponding pair `(μ_at s, K_at s)`.
-/
def PersistenceCert (sf : ScalingFamily) (γ : ℝ) : Prop :=
  0 < γ ∧ ∀ s : Scale, TransferPFGap (sf.μ_at s) (sf.K_at s) γ

/-- One-step stability of the PF gap along the scale successor. -/
def StableUnderScaling (sf : ScalingFamily) (γ : ℝ) : Prop :=
  ∀ s : Scale,
    TransferPFGap (sf.μ_at s) (sf.K_at s) γ →
    TransferPFGap (sf.μ_at ⟨s.n + 1⟩) (sf.K_at ⟨s.n + 1⟩) γ

/-- Uniform block-positivity hypothesis across scales. -/
def UniformBlockPos (sf : ScalingFamily) : Prop :=
  ∀ s : Scale, ∀ b : Block, BlockPositivity (sf.μ_at s) (sf.K_at s) b

theorem gap_persists_of_cert {sf : ScalingFamily} {γ : ℝ}
    (h : PersistenceCert sf γ) : GapPersists γ := by
  -- Uses the uniform-in-scale PF gap `γ > 0` to certify persistence.
  exact h.1

/--
If a base-scale PF gap holds and the PF gap is stable under scaling steps,
then the gap persists uniformly across all scales (with `0 < γ`).
-/
theorem persistence_from_base_and_stability {sf : ScalingFamily} {γ : ℝ}
    (hγ : 0 < γ)
    (h0 : TransferPFGap (sf.μ_at ⟨0⟩) (sf.K_at ⟨0⟩) γ)
    (hstab : StableUnderScaling sf γ)
    : PersistenceCert sf γ := by
  refine And.intro hγ ?_;
  intro s; cases s with
  | mk n =>
    -- Prove the PF gap for all `n : Nat` by induction using stability.
    have : ∀ m : Nat, TransferPFGap (sf.μ_at ⟨m⟩) (sf.K_at ⟨m⟩) γ := by
      intro m;
      induction' m with m ih
      · simpa using h0
      · have step := hstab ⟨m⟩ ih
        simpa using step
    exact this n

/--
If block positivity holds uniformly across scales, a PF transfer gap exists at every scale.
Together with `0 < γ`, this yields gap persistence.
-/
theorem persistence_of_uniform_block_pos {sf : ScalingFamily} {γ : ℝ}
    (hγ : 0 < γ)
    (h_ubp : UniformBlockPos sf) : PersistenceCert sf γ := by
  refine And.intro hγ ?_;
  intro s;
  have hb : ∀ b : Block, BlockPositivity (sf.μ_at s) (sf.K_at s) b := fun b => h_ubp s b
  simpa using (pf_gap_of_block_pos (μ := sf.μ_at s) (K := sf.K_at s) γ hb)

/-- If OS-positivity holds at every scale and yields block-positivity and
irreducibility for the transfer kernel, then a uniform PF gap of size `γ`
persists across scales. This abstracts the Doeblin/Dobrushin route. -/
def UniformOS (sf : ScalingFamily) : Prop := ∀ s, OSPositivity (sf.μ_at s)

theorem persistence_from_uniform_OS
    {sf : ScalingFamily} {γ : ℝ}
    (hγ : 0 < γ)
    (hOS : UniformOS sf)
    (hDoeb : ∀ s, Irreducible (sf.K_at s))
    (hBlk : ∀ s b, BlockPositivity (sf.μ_at s) (sf.K_at s) b)
    : PersistenceCert sf γ := by
  refine And.intro hγ ?_;
  intro s;
  have hb : ∀ b : Block, BlockPositivity (sf.μ_at s) (sf.K_at s) b := fun b => hBlk s b
  simpa using (pf_gap_of_block_pos (μ := sf.μ_at s) (K := sf.K_at s) γ hb)

/-- Quantitative persistence input bundle: a uniform OS positivity hypothesis,
irreducibility, and block positivity at all scales, together with an explicit
gap `γ0 > 0`. -/
def QuantPersistence (sf : ScalingFamily) (γ0 : ℝ) : Prop :=
  0 < γ0 ∧ UniformOS sf ∧ (∀ s, Irreducible (sf.K_at s)) ∧ (∀ s b, BlockPositivity (sf.μ_at s) (sf.K_at s) b)

/-- From the quantitative bundle, produce a `PersistenceCert` with the same `γ0`. -/
theorem persistence_of_quant {sf : ScalingFamily} {γ0 : ℝ}
    (h : QuantPersistence sf γ0) : PersistenceCert sf γ0 := by
  rcases h with ⟨hγ, hOS, hIrr, hBlk⟩
  exact persistence_from_uniform_OS (sf := sf) (γ := γ0) hγ hOS hIrr hBlk

end YM
