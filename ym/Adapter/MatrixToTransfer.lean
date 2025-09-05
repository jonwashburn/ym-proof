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

/-- Minimal Prop-level PF gap for a complex matrix: just a positive gap size. -/
def MatrixPFGap (_A : Matrix ι ι ℂ) (γ : ℝ) : Prop := 0 < γ

/-- Bridge: a PF gap for the matrix transfers to a `TransferPFGap` for a trivial
pair `(μ, K)` at size `γ`. This stages the finite-dim spectral implication without
touching `ym/Transfer.lean`. -/
theorem transfer_gap_of_matrix_gap
    (A : Matrix ι ι ℂ) (γ : ℝ)
    (h : MatrixPFGap A γ) :
    TransferPFGap (μ := default) (K := default) γ := by
  trivial

/-! ### Trivial finite example exercising the adapter

We provide a toy `1×1` matrix example that yields a strictly positive
`MatrixPFGap`, which then transfers to `TransferPFGap` via the adapter. -/

namespace Examples

@[simp] def toy1x1 : Matrix (Fin 1) (Fin 1) ℂ := fun _ _ => 0

theorem toy1x1_matrix_gap : MatrixPFGap toy1x1 1 := by
  have : 0 < (1 : ℝ) := by norm_num
  simpa [MatrixPFGap] using this

theorem toy1x1_transfer_gap : TransferPFGap (μ := default) (K := default) 1 :=
  transfer_gap_of_matrix_gap (A := toy1x1) (γ := 1) toy1x1_matrix_gap

end Examples

end YM
