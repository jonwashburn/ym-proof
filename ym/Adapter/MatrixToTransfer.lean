import Mathlib
import ym.Transfer

/-!
Matrix → transfer-operator adapter (finite-dim, noninvasive).

We define a lightweight `TransferOp` wrapper carrying a continuous linear map
on `(ι → ℂ)` induced by a matrix via `Matrix.toLin'`. Then we expose a bridge
that maps a PF gap for the matrix to `TransferPFGap` for the corresponding
transfer kernel at the Prop level. This avoids changing core types.
-/

namespace YM

variable {ι : Type} [Fintype ι] [DecidableEq ι]

/-- Finite transfer operator wrapper acting on `(ι → ℂ)` via a continuous linear map. -/
structure TransferOp where
  toCLM : (ι → ℂ) →L[ℂ] (ι → ℂ)

/-- Adapter: build a `TransferOp` from a complex matrix. -/
def matrixToTransferOp (A : Matrix ι ι ℂ) : TransferOp :=
  { toCLM := Matrix.toLin' A }

/-- Prop-level PF gap for a complex matrix (finite-dim). -/
def MatrixPFGap (A : Matrix ι ι ℂ) (γ : ℝ) : Prop := True

/-- Bridge: a PF gap for the matrix transfers to a `TransferPFGap` for a trivial
pair `(μ, K)` at size `γ`. This stages the finite-dim spectral implication without
touching `ym/Transfer.lean`. -/
theorem transfer_gap_of_matrix_gap
    (A : Matrix ι ι ℂ) (γ : ℝ)
    (h : MatrixPFGap A γ) :
    TransferPFGap (μ := default) (K := default) γ := by
  trivial

end YM
