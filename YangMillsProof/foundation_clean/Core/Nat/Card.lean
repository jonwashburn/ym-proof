/-
  Nat.Card
  --------
  Elementary counting lemmas for finite types.
  Self-contained implementation using only Lean 4 standard library.

  Author: Jonathan Washburn
  Recognition Science Institute
-/

import Mathlib.Logic.Function.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fin.Basic

namespace RecognitionScience.Nat.Card

/-- There is no injection from Fin (n+1) to Fin n -/
theorem no_inj_succ_to_self {n : Nat} (f : Fin (n + 1) → Fin n) : ¬Function.Injective f := by
  intro h_inj
  -- Use cardinality argument: Fin (n+1) has n+1 elements, Fin n has n elements
  -- An injection would require n+1 ≤ n, which is false
  have h_card_src : Fintype.card (Fin (n + 1)) = n + 1 := Fintype.card_fin (n + 1)
  have h_card_tgt : Fintype.card (Fin n) = n := Fintype.card_fin n
  have h_le : Fintype.card (Fin (n + 1)) ≤ Fintype.card (Fin n) :=
    Fintype.card_le_of_injective h_inj
  rw [h_card_src, h_card_tgt] at h_le
  -- h_le : n + 1 ≤ n, which contradicts Nat.lt_succ_self n
  exact Nat.not_succ_le_self n h_le

/-- If Fin n is in bijection with Fin m, then n = m -/
theorem bij_fin_eq {n m : Nat} (h : Fin n ≃ Fin m) : n = m := by
  -- Use cardinality: bijections preserve cardinality
  have h_card : Fintype.card (Fin n) = Fintype.card (Fin m) :=
    Fintype.card_of_bijective h.bijective
  rw [Fintype.card_fin, Fintype.card_fin] at h_card
  exact h_card

end RecognitionScience.Nat.Card
