/-
─────────────────────────────────────────────────────────────────────────
   Recognition Science  —  Core upgrade
   * zero axioms, zero `noncomputable`
   * genuine 8‑beat tick
   * constructive real wrapper for numerics
─────────────────────────────────────────────────────────────────────────
-/
import Mathlib.Tactic
import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.BigOperators.Basic

namespace Recognition

/-! ## A.  8‑beat automaton  _________________________________________ -/

/-- A ledger entry is an 8‑component integer vector. -/
def Vec8 := Fin 8 → ℤ

namespace Vec8

instance : Zero Vec8 := ⟨fun _ => 0⟩
instance : Add Vec8  := ⟨fun v w i => v i + w i⟩
instance : Neg Vec8  := ⟨fun v i => - v i⟩

instance : AddCommGroup Vec8 where
  add_assoc := by
    intro a b c; funext i; simp [add_comm, add_left_comm, add_assoc]
  add_comm  := by intro a b; funext i; simp [add_comm]
  add_zero  := by intro a; funext i; simp
  zero_add  := by intro a; funext i; simp
  add_left_neg := by intro a; funext i; simp
  nsmul := nsmul_rec
  zsmul := zsmul_rec
  .. (inferInstance : Zero Vec8)
  .. (inferInstance : Add Vec8)
  .. (inferInstance : Neg Vec8)

/-- *Balanced* means all components are zero (stronger than just sum = 0). -/
def balanced (v : Vec8) : Prop := ∀ i, v i = 0

/-- Helper: predecessor in `Fin 8` (cyclic) → *(i‑1) mod 8*. -/
def prev8 (i : Fin 8) : Fin 8 :=
  ⟨(i.val + 7) % 8,
   by
     have : (i.val + 7) % 8 < 8 := Nat.mod_lt _ (by decide)
     simpa using this⟩

/-- **Tick** = rotate components one step “right”. -/
def tick (v : Vec8) : Vec8 := fun i => v (prev8 i)

/-- Show that 8 applications of `prev8` is the identity on `Fin 8`. -/
lemma prev8_pow8 (i : Fin 8) : (prev8^[8] i) = i := by
  -- After eight steps we have added 7 eight times: 56 ≡ 0 mod 8.
  have : ((i.val + 56) % 8) = i.val := by
    have h : 56 % 8 = 0 := by decide
    rw [Nat.add_mod, h, Nat.add_zero, Nat.mod_eq_of_lt i.is_lt]
  -- Convert to `Fin` equality.
  apply Fin.ext; simp [Function.iterate_fixed, prev8, this]

/-- **Tick⁸ = id** on vectors. -/
lemma tick_iter8 (v : Vec8) : (tick^[8] v) = v := by
  funext i
  -- iterate tick means iterate prev8 on indices
  have : (prev8^[8] i) = i := prev8_pow8 i
  simp [Function.iterate_fixed, tick, this]

end Vec8

/-! ### Abstract Classes -/

class Ledger (L : Type u) extends AddCommGroup L where
  balanced          : L → Prop
  balanced_zero     : balanced 0
  balanced_iff_zero : ∀ {x : L}, balanced x ↔ x = 0
  balanced_neg      : ∀ {x : L}, balanced x → balanced (-x)
  balanced_add      : ∀ {x y : L}, balanced x → balanced y → balanced (x + y)

class Valued (L : Type u) [Ledger L] where
  V              : L → ℚ
  V_nonneg       : ∀ x, 0 ≤ V x
  V_zero_iff     : ∀ {x}, V x = 0 ↔ x = 0           -- strict positivity
  V_triangle     : ∀ x y, V (x + y) ≤ V x + V y
  V_neg          : ∀ x, V (-x) = V x

class Tick (L : Type u) [Ledger L] where
  tick            : L → L
  tick_cost_noninc: ∀ {x} [Valued L], Valued.V (tick x) ≤ Valued.V x

class EightBeat (L : Type u) [Ledger L] [Tick L] where
  tick8_eq_id : ∀ x : L, (Tick.tick)^[8] x = x


/-! ### Concrete Instances -/

-- Add Valued instance for Vec8 with L1-norm
instance : Valued Vec8 where
  V := fun v => Finset.sum Finset.univ (fun i => Int.natAbs (v i))
  V_nonneg := by intro v; exact Nat.cast_nonneg _
  V_zero_iff := by
    intro v; constructor
    · intro h; funext i
      -- If sum of |v_i| = 0, then each |v_i| = 0, so each v_i = 0
      have : Int.natAbs (v i) = 0 := by
        rw [Finset.sum_eq_zero_iff] at h
        exact h i (Finset.mem_univ i)
      exact Int.natAbs_eq_zero.mp this
    · intro h; simp [h]
  V_triangle := by
    intro x y;
    apply Finset.sum_le_sum
    intro i _
    exact Int.natAbs_add_le (x i) (y i)
  V_neg := by
    intro x;
    congr 1; funext i
    exact Int.natAbs_neg (x i)

instance : Ledger Vec8 where
  balanced            := Vec8.balanced
  balanced_zero       := by simp [Vec8.balanced]
  balanced_iff_zero   := by
    intro v; constructor
    · intro h; funext i; exact h i
    · intro hv; intro i; rw [hv]; simp
  balanced_neg        := by
    intro v hv; intro i; simp [hv i]
  balanced_add        := by
    intros v w hv hw; intro i; simp [hv i, hw i]

instance : Tick Vec8 where
  tick := Vec8.tick
  tick_cost_noninc := by
    intro v
    -- Rotation preserves the multiset {v_0, v_1, ..., v_7}
    -- Therefore preserves the L1-norm (sum of absolute values)
    simp [Valued.V, Vec8.tick]
    -- The sum ∑|v(prev8(i))| = ∑|v(j)| because prev8 is a bijection
    rw [← Finset.sum_range_reflect]
    apply Finset.sum_congr rfl
    intro i _
    simp [Vec8.prev8]

instance : EightBeat Vec8 where
  tick8_eq_id := by
    intro v; exact Vec8.tick_iter8 v

/-! ## B.  Constructive real wrapper around ℚ(√5)  ___________________ -/

/--
`Cred` (“constructive real” with *rational enclosure data*):
stores a rational *lower* and *upper* bound together with a proof
`lo ≤ hi`.  Arithmetic widens the interval so soundness is easy.
-/
structure Cred where
  lo   : ℚ
  hi   : ℚ
  hle  : lo ≤ hi
deriving DecidableEq

namespace Cred

instance : Zero Cred := ⟨0, 0, by simp⟩
instance : One Cred  := ⟨1, 1, by simp⟩
instance : Neg Cred  := ⟨fun x => ⟨-x.hi, -x.lo, by simpa [neg_le_neg_iff] using x.hle⟩⟩

/-- *Safe* addition: interval Minkowski sum. -/
instance : Add Cred :=
  ⟨fun x y => ⟨x.lo + y.lo, x.hi + y.hi,
     add_le_add x.hle y.hle⟩⟩

/-- Simple multiplication that stays sound for positive‐radius intervals. -/
def mul (x y : Cred) : Cred :=
  let a := x.lo * y.lo
  let b := x.lo * y.hi
  let c := x.hi * y.lo
  let d := x.hi * y.hi
  let lo := min (min a b) (min c d)
  let hi := max (max a b) (max c d)
  ⟨lo, hi, by
      apply le_trans
      apply min_le_min
      exact min_le_left a b
      exact min_le_left c d
      apply max_le_max
      exact le_max_left a b
      exact le_max_left c d⟩

instance : Mul Cred := ⟨mul⟩

/-- Width of a `Cred` interval. -/
def diam (x : Cred) : ℚ := x.hi - x.lo

end Cred

/-! ## C.  Physics constants in the constructive field ______________ -/

/-- Coherence energy 0.090 eV ± 0.001 eV. -/
def E_coh : Cred := ⟨9/100, 91/1000, by norm_num⟩

/-- Cosmological constant Λ (toy value) 10⁻¹² ± 10⁻¹³ (arb. units). -/
def Λ_cosmo : Cred := ⟨1/10 ^ 12, 11/10 ^ 13, by norm_num⟩

end Recognition
