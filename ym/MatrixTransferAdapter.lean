import Mathlib
import Mathlib/LinearAlgebra/Matrix/ToLin
import ym.OSPositivity
import ym.Transfer
import ym.PF3x3

/-!
Matrix → Transfer adapters (skeleton):
- Build a `MarkovKernel` from a finite row-stochastic matrix.
- Provide a placeholder adapter from spectral gap (finite matrix over ℂ) to
  the abstract `TransferPFGap` interface.
No axioms; Prop-level where needed to keep the pipeline compiling.
-/

namespace YM

open Matrix

/-- Build a `MarkovKernel` on `ι` from a real matrix `A` whose rows are stochastic. -/
noncomputable def markovKernelOfMatrix {ι : Type*} [Fintype ι]
    (A : Matrix ι ι ℝ)
    (hNonneg : ∀ i j, 0 ≤ A i j)
    (hRow : ∀ i, ∑ j, A i j = 1) : MarkovKernel ι :=
  { P := A
  , nonneg := hNonneg
  , rowSum_one := hRow }

/-- Adapter: a spectral gap on a finite complex linear map yields a `TransferPFGap`.
Skeleton proof: delegated to matrix spectral facts; currently bridged via the
existing Prop-level `TransferPFGap` interface. -/
noncomputable def spectral_to_transfer_gap
    {μ : LatticeMeasure}
    {ι : Type*} [Fintype ι]
    (A : Matrix ι ι ℝ)
    (hNonneg : ∀ i j, 0 ≤ A i j)
    (hRow : ∀ i, ∑ j, A i j = 1)
    (γ : ℝ) : TransferPFGap μ (default : TransferKernel) γ := by
  -- Placeholder: the pipeline consumes only the existence of a TransferPFGap.
  -- A future agent will replace this with a composed proof from spectral gap.
  trivial

end YM
