import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Real.Sqrt
import Mathlib.Topology.Algebra.InfiniteSum.Basic
import Mathlib.Analysis.Normed.Group.InfiniteSum
import YangMillsProof.RSImport.BasicDefinitions

namespace YangMillsProof

-- Import key definitions from RSImport
open RSImport

/-- Colour residues are the three possible values of a ledger rung modulo 3. -/
abbrev ColourResidue := Fin 3

/-- A voxel position in 3D space -/
structure VoxelPos where
  x : ℤ
  y : ℤ
  z : ℤ

/-- A voxel face is identified by its rung, position, and orientation -/
structure VoxelFace where
  rung : ℤ
  position : VoxelPos
  orientation : Fin 6  -- 6 orientations for cube faces

/-- The colour residue of a voxel face -/
def colourResidue (f : VoxelFace) : ColourResidue :=
  (f.rung.natAbs % 3 : Fin 3)

/-- A gauge ledger entry represents gauge field components -/
structure GaugeLedgerEntry where
  debit : ℕ
  credit : ℕ

/-- The gauge ledger state -/
structure GaugeLedgerState where
  debit : VoxelFace → ℕ
  credit : VoxelFace → ℕ
  finite_support : Set.Finite {f | debit f ≠ 0 ∨ credit f ≠ 0}

/-- The vacuum state for gauge fields -/
def vacuumStateGauge : GaugeLedgerState where
  debit := fun _ => 0
  credit := fun _ => 0
  finite_support := by simp [Set.finite_empty]

/-- The gauge layer consists of states with specific colour residue patterns -/
def GaugeLayer : Set GaugeLedgerState :=
  {s | ∃ f, s.debit f + s.credit f > 0 ∧ colourResidue f ≠ 0}

/-- The gauge cost functional -/
noncomputable def gaugeCost (s : GaugeLedgerState) : ℝ :=
  ∑' f : VoxelFace, (s.debit f + s.credit f : ℝ) * (E_coh * phi ^ f.rung.natAbs)

/-- Key lemma: gauge layer states have non-zero colour residue faces -/
lemma gauge_layer_has_nonzero_residue_face (s : GaugeLedgerState) (hs : s ∈ GaugeLayer) :
  ∃ f, s.debit f + s.credit f > 0 ∧ colourResidue f ≠ 0 := hs

/-- The main gauge cost lower bound theorem -/
theorem gauge_cost_lower_bound (s : GaugeLedgerState) (hs : s ∈ GaugeLayer) :
  gaugeCost s ≥ E_coh * phi := by
  -- Get a face with non-zero residue and non-zero entry
  obtain ⟨f, ⟨hf_nonzero, hf_residue⟩⟩ := gauge_layer_has_nonzero_residue_face s hs

  -- Since s.debit f + s.credit f > 0, we have at least 1
  have h_sum_ge_one : s.debit f + s.credit f ≥ 1 := hf_nonzero

  -- Since colourResidue f ≠ 0, we have f.rung.natAbs ≥ 1
  have h_rung_ge_one : f.rung.natAbs ≥ 1 := by
    by_contra h_zero
    push_neg at h_zero
    have : f.rung.natAbs = 0 := Nat.lt_one_iff.mp h_zero
    have : colourResidue f = 0 := by
      simp [colourResidue, this]
    exact hf_residue this

  -- Since phi > 1, we have phi^n ≥ phi for n ≥ 1
  have h_phi_power : phi ^ f.rung.natAbs ≥ phi := by
    calc phi ^ f.rung.natAbs
      ≥ phi ^ 1 := pow_le_pow_right (le_of_lt phi_gt_one) h_rung_ge_one
      _ = phi := pow_one phi

  -- The contribution from face f is at least E_coh * phi
  have h_face_contrib : (s.debit f + s.credit f : ℝ) * (E_coh * phi ^ f.rung.natAbs) ≥ E_coh * phi := by
    calc
      (s.debit f + s.credit f : ℝ) * (E_coh * phi ^ f.rung.natAbs)
      = E_coh * ((s.debit f + s.credit f : ℝ) * phi ^ f.rung.natAbs) := by ring
      _ ≥ E_coh * (1 * phi ^ f.rung.natAbs) := by
        apply mul_le_mul_of_nonneg_left
        apply mul_le_mul_of_nonneg_right
        · exact_mod_cast h_sum_ge_one
        · exact pow_nonneg (le_of_lt phi_pos) _
        · exact le_of_lt E_coh_pos
      _ = E_coh * phi ^ f.rung.natAbs := by ring
      _ ≥ E_coh * phi := by
        apply mul_le_mul_of_nonneg_left h_phi_power
        exact le_of_lt E_coh_pos

  -- Show gaugeCost s ≥ E_coh * phi
  unfold gaugeCost
  apply le_trans h_face_contrib

  -- The key insight: the sum includes the f-th term and all terms are non-negative
  -- Therefore ∑' g ≥ the f-th term
  have h_nonneg : ∀ g : VoxelFace, 0 ≤ (s.debit g + s.credit g : ℝ) * (E_coh * phi ^ g.rung.natAbs) := by
    intro g
    apply mul_nonneg
    · exact_mod_cast add_nonneg (Nat.zero_le _) (Nat.zero_le _)
    · apply mul_nonneg
      · exact le_of_lt E_coh_pos
      · exact pow_nonneg (le_of_lt phi_pos) _

  -- The series is summable because of finite support
  have h_summable : Summable fun g => (s.debit g + s.credit g : ℝ) * (E_coh * phi ^ g.rung.natAbs) := by
    -- First show the support is finite
    have h_finite : (Function.support fun g => (s.debit g + s.credit g : ℝ) * (E_coh * phi ^ g.rung.natAbs)).Finite := by
      apply Set.Finite.subset s.finite_support
      intro g hg
      simp [Function.mem_support] at hg
      simp [Function.mem_support]
      by_contra h_zero
      push_neg at h_zero
      obtain ⟨h1, h2⟩ := h_zero
      have : (s.debit g : ℝ) + (s.credit g : ℝ) = 0 := by
        rw [h1, h2]
        simp
      rw [this] at hg
      simp at hg
    -- Use finite support to get summability
    exact summable_of_finite_support h_finite

  -- Apply the standard result that tsum ≥ any single term
  apply le_tsum h_summable
  intro j _
  exact h_nonneg j

end YangMillsProof
