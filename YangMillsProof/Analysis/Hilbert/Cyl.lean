/-
  Cylinder Function Pre-Hilbert Space
  ===================================
  This lightweight module provides a minimal "cylinder–function" space that is
  sufficient for the current Osterwalder–Schrader reconstruction scaffold.  A
  true construction would restrict to functions depending on only finitely many
  directions in an infinite product space; for now we simply take **all**
  real-valued functions on `ℕ`.  This gives us a concrete non-trivial vector
  space without introducing extra prerequisites, and it can be refined later
  without changing the external API.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Module.Pi

namespace Analysis.Hilbert

/--
`CylinderSpace` – a very small stand-in for the usual space of cylinder
functions used in constructive quantum field theory.  It is realised as the
vector space of real-valued functions on `ℕ`.  The key property we need for the
scaffold is simply that it carries the usual pointwise module structure over
`ℝ`, which `Pi.module` already provides.
-/
abbrev CylinderSpace : Type := ℕ → ℝ

instance : AddCommGroup CylinderSpace := inferInstance
instance : Module ℝ CylinderSpace := inferInstance

end Analysis.Hilbert
