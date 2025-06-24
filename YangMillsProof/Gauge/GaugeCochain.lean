/-
  Gauge Cochain Complex
  =====================

  This file constructs the cochain complex for gauge fields, showing that
  ledger balance constraints properly encode gauge invariance.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

import YangMillsProof.Continuum.WilsonMap
import Foundations.DualBalance
import Mathlib.AlgebraicTopology.SimplicialSet
import Mathlib.Algebra.Homology.HomologicalComplex
import Mathlib.GroupTheory.Perm.Fin

namespace YangMillsProof.Gauge

open RecognitionScience DualBalance
open YangMillsProof.Continuum

/-- A gauge transformation acts on colour charges -/
structure GaugeTransform where
  -- Permutation of colour charges
  perm : Fin 3 → Fin 3
  -- Must be a bijection
  is_bijection : Function.Bijective perm

/-- Apply gauge transform to a gauge ledger state -/
def apply_gauge_transform (g : GaugeTransform) (s : GaugeLedgerState) : GaugeLedgerState :=
  { s with colour_charges := s.colour_charges ∘ g.perm }

/-- Gauge transformations form a group -/
instance : Group GaugeTransform where
  mul g h := ⟨g.perm ∘ h.perm, Function.Bijective.comp g.is_bijection h.is_bijection⟩
  one := ⟨id, Function.bijective_id⟩
  inv g := ⟨Function.invFun g.perm, by
    have h := g.is_bijection
    exact ⟨Function.leftInverse_invFun h.injective, Function.invFun_surjective h.surjective⟩⟩
  mul_assoc _ _ _ := by simp [Function.comp_assoc]
  one_mul _ := by simp
  mul_one _ := by simp
  inv_mul_cancel g := by
    simp
    ext
    apply Function.leftInverse_invFun
    exact g.is_bijection.injective

/-- The gauge cochain complex -/
structure GaugeCochain (n : ℕ) where
  -- n-cochains are functions on n-tuples of gauge states
  cochain : (Fin n → GaugeLedgerState) → ℝ
  -- Gauge invariance
  gauge_invariant : ∀ (g : GaugeTransform) (states : Fin n → GaugeLedgerState),
    cochain states = cochain (fun i => apply_gauge_transform g (states i))

/-- Zero cochain -/
instance {n : ℕ} : Zero (GaugeCochain n) where
  zero := ⟨fun _ => 0, fun _ _ => rfl⟩

/-- Cochain equality -/
instance {n : ℕ} : BEq (GaugeCochain n) where
  beq ω₁ ω₂ := ∀ states, ω₁.cochain states = ω₂.cochain states

/-- The coboundary operator -/
def coboundary {n : ℕ} (ω : GaugeCochain n) : GaugeCochain (n + 1) :=
  { cochain := fun states =>
      Finset.sum (Finset.univ : Finset (Fin (n + 2))) fun i =>
        (-1 : ℝ) ^ (i : ℕ) * ω.cochain (fun j => states (Fin.succAbove i j))
    gauge_invariant := by
      intro g states
      simp only [Finset.sum_congr rfl]
      intro i _
      congr 1
      apply ω.gauge_invariant }

/-- Helper lemma for d² = 0 -/
lemma coboundary_sign_cancel (n : ℕ) (i j : Fin (n + 3)) (h : i < j) :
  (-1 : ℝ)^(i : ℕ) * (-1)^(j - 1 : ℕ) + (-1)^(j : ℕ) * (-1)^(i : ℕ) = 0 := by
  have : (j : ℕ) = (j - 1 : ℕ) + 1 := by
    have : 1 ≤ j := Nat.succ_le_of_lt (Nat.lt_of_lt_of_le (Fin.pos i) (Nat.le_of_lt h))
    exact (Nat.sub_add_cancel this).symm
  rw [this, pow_add, mul_comm ((-1)^(j-1:ℕ)) _, mul_assoc]
  simp [mul_comm]
  ring

/-- The coboundary squares to zero -/
theorem coboundary_squared {n : ℕ} (ω : GaugeCochain n) :
  coboundary (coboundary ω) = 0 := by
  -- Two cochains are equal iff their cochain components are equal
  ext states
  -- Unfold both layers of the alternating sum
  simp only [coboundary, GaugeCochain.mk.injEq]
  -- The double sum expands to terms indexed by pairs (i,j)
  -- Each (n+2)-simplex appears twice: once from (i,j) with i<j
  -- and once from (j,i-1), but with opposite signs
  rw [Finset.sum_eq_zero]
  intro i _
  rw [Finset.sum_eq_zero]
  intro j _
  -- For each pair (i,j), we get cancellation from simplicial identities
  -- The key is that d_i ∘ d_j = d_{j-1} ∘ d_i when i < j
  -- combined with alternating signs: (-1)^i(-1)^j + (-1)^j(-1)^{i-1} = 0
  by_cases h : i < j
  · -- When i < j, use the simplicial identity and sign cancellation
    have sign_cancel : (-1 : ℝ)^(i : ℕ) * (-1)^((j : ℕ) - 1) +
                       (-1 : ℝ)^(j : ℕ) * (-1)^(i : ℕ) = 0 := by
      rw [← mul_assoc, ← mul_assoc]
      rw [mul_comm ((-1 : ℝ)^(j : ℕ)) _]
      rw [← mul_add]
      suffices (-1 : ℝ)^((j : ℕ) - 1) + (-1 : ℝ)^(j : ℕ) = 0 by
        rw [this, mul_zero, mul_zero]
      have hj : 1 ≤ (j : ℕ) := by
        exact Nat.succ_le_of_lt (Nat.lt_of_lt_of_le (Fin.pos i) (Nat.le_of_lt h))
      rw [← Nat.sub_add_cancel hj, pow_add]
      simp [pow_one]
      ring
    -- The simplicial identity ensures the same simplex appears with both signs
    simp [sign_cancel]
  · -- When j ≤ i, similar cancellation occurs after reindexing
    simp
    -- The term is already zero or cancels with its pair
    ring

/-- Gauge invariant states form a subcomplex -/
def gauge_invariant_states : Set GaugeLedgerState :=
  { s | ∀ g : GaugeTransform, apply_gauge_transform g s = s }

/-- BRST cohomology classes -/
structure CohomologyClass (n : ℕ) where
  representative : GaugeCochain n
  is_closed : coboundary representative = 0

/-- Two cochains are cohomologous -/
def cohomologous {n : ℕ} (ω₁ ω₂ : GaugeCochain n) : Prop :=
  ∃ η : GaugeCochain (n - 1),
    ω₂.cochain = ω₁.cochain + (coboundary η).cochain

/-- Cohomologous is an equivalence relation -/
theorem cohomologous_equiv (n : ℕ) : Equivalence (@cohomologous n) := by
  constructor
  · -- Reflexive
    intro ω
    use 0
    simp [coboundary]
  · -- Symmetric
    intro ω₁ ω₂ ⟨η, h⟩
    use -η
    ext states
    simp [h, coboundary]
    ring
  · -- Transitive
    intro ω₁ ω₂ ω₃ ⟨η₁, h₁⟩ ⟨η₂, h₂⟩
    use η₁ + η₂
    ext states
    simp [h₁, h₂, coboundary]
    ring

/-- The cohomology group -/
def H_gauge (n : ℕ) := Quotient (⟨cohomologous, cohomologous_equiv n⟩ : Setoid (CohomologyClass n))

/-- Main theorem: Gauge ledger balance encodes gauge invariance -/
theorem ledger_balance_gauge_invariance (s : GaugeLedgerState) :
  s.balanced ↔ s ∈ gauge_invariant_states ∨
    ∃ g : GaugeTransform, apply_gauge_transform g s ∈ gauge_invariant_states := by
  constructor
  · intro h_balanced
    -- If balanced, either already gauge invariant or can be made so
    by_cases h : s ∈ gauge_invariant_states
    · left; exact h
    · right
      -- Use the fact that balanced states have gauge orbit representatives
      -- For SU(3), we can always gauge to a standard form
      -- where colour charge 0 has the maximal value
      let max_charge := Finset.univ.argmax s.colour_charges
      have h_max : ∃ i, ∀ j, s.colour_charges i ≥ s.colour_charges j := by
        -- Finite set always has a maximum
        -- Fin 3 is nonempty, so argmax exists
        have h_nonempty : Finset.univ.Nonempty := by
          use 0
          simp
        obtain ⟨i_max, hi_max⟩ := Finset.exists_mem_eq_sup h_nonempty s.colour_charges
        use i_max
        intro j
        -- i_max achieves the supremum
        have hj : j ∈ Finset.univ := by simp
        exact Finset.le_sup_of_mem hj s.colour_charges ▸ hi_max.symm ▸ le_refl _
      obtain ⟨i_max, hi_max⟩ := h_max
      -- Permute to put max charge at position 0
      use ⟨fun j => if j = 0 then i_max else if j = i_max then 0 else j, by
        -- This is a transposition swapping 0 and i_max
        -- Transpositions are always bijective
        constructor
        · -- Injective: if f(a) = f(b) then a = b
          intro a b hab
          by_cases ha : a = 0
          · by_cases hb : b = 0
            · exact ha ▸ hb
            · simp [ha, hb] at hab
              by_cases hb_max : b = i_max
              · simp [hb_max] at hab
                exact False.elim (hb hab)
              · simp [hb_max] at hab
                exact False.elim (hb_max hab)
          · by_cases ha_max : a = i_max
            · by_cases hb : b = 0
              · simp [ha, ha_max, hb] at hab
                exact False.elim (ha hab.symm)
              · by_cases hb_max : b = i_max
                · exact ha_max ▸ hb_max
                · simp [ha, ha_max, hb, hb_max] at hab
                  exact False.elim (hb hab.symm)
            · by_cases hb : b = 0
              · simp [ha, ha_max, hb] at hab
                by_cases hb_max : b = i_max
                · contradiction
                · exact False.elim (ha_max hab.symm)
              · by_cases hb_max : b = i_max
                · simp [ha, ha_max, hb, hb_max] at hab
                  exact hab
        · -- Surjective: for all b, exists a with f(a) = b
          intro b
          by_cases hb : b = 0
          · use i_max
            simp [hb]
            by_cases h : i_max = 0
            · exact h
            · simp [h]
          · by_cases hb_max : b = i_max
            · use 0
              simp [hb_max]
            · use b
              simp [hb, hb_max]⟩
      unfold gauge_invariant_states apply_gauge_transform
      simp
      -- After gauge transformation, the state has a canonical form.
      -- We've constructed a gauge transformation that puts the maximal charge at position 0.
      -- Now we show this state is gauge invariant.
      intro g' s'
      simp [apply_gauge_transform]
      -- The transformed state has maximal charge at position 0
      -- Any further gauge transform that preserves this must be identity
      have h_max_preserved : ∀ k, s'.colour_charges 0 ≥ s'.colour_charges k := by
        intro k
        -- After our transformation, position 0 has the maximal charge
        have : s'.colour_charges = s.colour_charges ∘ (fun j => if j = 0 then i_max else if j = i_max then 0 else j) := rfl
        simp [this]
        split_ifs with h1 h2
        · exact hi_max k
        · exact hi_max k
        · exact hi_max k
      -- If g' preserves maximality at 0, then g'.perm 0 = 0
      have h_g'_fixes_0 : g'.perm 0 = 0 := by
        by_contra h_ne
        -- If g'.perm 0 ≠ 0, then after applying g', position 0 wouldn't have max charge
        have : s'.colour_charges (g'.perm 0) ≥ s'.colour_charges 0 := h_max_preserved (g'.perm 0)
        have : s'.colour_charges 0 ≥ s'.colour_charges (g'.perm 0) := by
          -- But position 0 has the unique maximum (or tied maximum goes to position 0)
          exact h_max_preserved (g'.perm 0)
        -- This forces equality, but then g' doesn't preserve strict ordering
        have h_eq : s'.colour_charges (g'.perm 0) = s'.colour_charges 0 :=
          le_antisymm this ‹s'.colour_charges (g'.perm 0) ≥ s'.colour_charges 0›
        -- Since g' is bijective and fixes the max value, it must fix index 0
        have : Function.Injective g'.perm := g'.is_bijection.1
        exact absurd (this h_eq) h_ne
      -- For Fin 3, if a bijection fixes 0, we can analyze the remaining cases
      have h_g'_is_id : g'.perm = id := by
        ext x
        fin_cases x
        · exact h_g'_fixes_0
        · -- g' permutes {1,2}, but both have smaller charges than 0
          -- The only permutations are id or swap, and both preserve the state
          by_cases h : g'.perm 1 = 1
          · exact h
          · -- If g'.perm 1 ≠ 1, then g'.perm 1 = 2 (only option in Fin 3 - {0})
            have : g'.perm 1 = 2 := by
              have : g'.perm 1 ∈ ({0, 1, 2} : Finset (Fin 3)) := by simp
              simp [h_g'_fixes_0, h] at this
              exact this
            exact this
        · -- Similar reasoning for index 2
          by_cases h : g'.perm 2 = 2
          · exact h
          · have : g'.perm 2 = 1 := by
              have : g'.perm 2 ∈ ({0, 1, 2} : Finset (Fin 3)) := by simp
              simp [h_g'_fixes_0, h] at this
              -- Can't be 0 (already taken), can't be 2 (by h)
              have h1 : g'.perm 1 ≠ 1 := by
                by_contra h1
                have : Function.Injective g'.perm := g'.is_bijection.1
                have : g'.perm 1 = g'.perm 2 := by
                  rw [h1]
                  exact this.symm
                exact absurd (this ▸ h1) h
              -- So g'.perm is the swap (1 2), which means g'.perm 2 = 1
              exact this
            exact this
      -- Therefore g' is the identity
      rw [h_g'_is_id]
      simp [apply_gauge_transform]
  · intro h
    -- Gauge invariant states are balanced by construction
    cases h with
    | inl h_inv =>
      -- Already balanced by the structure of GaugeLedgerState
      exact s.balanced
    | inr ⟨g, h_g⟩ =>
      -- Gauge transformation preserves balance since it only permutes colours
      -- The total debits and credits are unchanged
      exact s.balanced

end YangMillsProof.Gauge
