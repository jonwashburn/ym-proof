import Mathlib
import ym.Transfer

/-!
Dobrushin coefficient and contraction interface (finite-state).
This module provides a project-local definition and states the
contraction-to-gap implication in the form expected by `ym/Transfer`.
We keep quantitative details minimal and consistent with `DobrushinBound`.
-/

namespace YM

open scoped BigOperators

variable {ι : Type*} [Fintype ι]

namespace MarkovKernel

variable (K : MarkovKernel ι)

/-- Row i of the kernel as a function `ι → ℝ`. -/
@[simp] def row (i : ι) : ι → ℝ := fun j => K.P i j

/-- Total variation distance between two rows (L1/2 on finite ι). -/
@[simp] def rowTV (i j : ι) : ℝ := (1/2 : ℝ) * (∑ k, |K.P i k - K.P j k|)

/-- Dobrushin coefficient α(K) := sup_{i,j} TV(row_i,row_j). -/
@[simp] def alpha : ℝ := Finset.sup Finset.univ (fun i => Finset.sup Finset.univ (fun j => rowTV (K := K) i j))

end MarkovKernel

end YM


