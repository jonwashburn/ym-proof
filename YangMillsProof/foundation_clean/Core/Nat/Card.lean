/-
  Natural Number Cardinality
  =========================

  Simple lemma for finite type cardinality.
-/

import Mathlib.Data.Fintype.Card
import Mathlib.Logic.Function.Basic

namespace RS.Core.Nat

open Fintype Function

/-- If there's an injective function from A to B, then |A| ≤ |B| -/
theorem card_le_of_injective {A B : Type*} [Fintype A] [Fintype B]
  (f : A → B) (h_inj : Function.Injective f) :
  Fintype.card A ≤ Fintype.card B := by
  -- This is a standard result in combinatorics
  -- An injective function from A to B means A can be embedded in B
  -- Therefore the cardinality of A cannot exceed that of B
  exact Fintype.card_le_of_injective f h_inj

end RS.Core.Nat
