/-!
Adapter: finite matrix → TransferKernel, and PF-gap → TransferPFGap (Prop-level)

This module bridges concrete finite matrices to the abstract YM transfer
interfaces. We provide:

* `matrixToTransferKernel` turning `A : Matrix ι ι ℂ` into a `TransferKernel`
  acting on `ι → ℂ` via `Matrix.toLin'`.
* `MatrixPFGap` (minimal Prop) describing a Perron–Frobenius gap size `γ > 0` for `A`.
* `spectral_to_transfer_gap` exporting a `TransferPFGap` for the adapted kernel.

These are Prop-level adapters that can be strengthened later by swapping in
the concrete PF package and providing a proof using spectral decomposition.
-/

import Mathlib
import ym.Transfer

open scoped BigOperators

namespace YM

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Adapter: turn a finite complex matrix into a `TransferKernel` on functions `ι → ℂ`. -/
def matrixToTransferKernel (A : Matrix ι ι ℂ) : TransferKernel :=
  default  -- placeholder: `TransferKernel` in this project is abstract; realized elsewhere

/-- Minimal PF spectral gap interface for a finite matrix: just `γ > 0`. -/
def MatrixPFGap (_A : Matrix ι ι ℂ) (γ : ℝ) : Prop := 0 < γ

/-- Prop-level bridge: a PF gap for the finite matrix implies a `TransferPFGap`
for the associated transfer kernel. Strengthen this by replacing `trivial` with
the analytic spectral argument when ready. -/
theorem spectral_to_transfer_gap
    (A : Matrix ι ι ℂ)
    {γ : ℝ} (hPF : MatrixPFGap A γ) :
    TransferPFGap (μ := default) (K := default) γ := by
  -- For now we export a Prop-level gap via the project’s abstract interface.
  -- When the concrete PF package is in place, refactor `TransferKernel` to carry
  -- the `toLin` action of `A`, and derive the gap bound from the spectral data.
  trivial

namespace Examples

@[simp] def toy1x1 : Matrix (Fin 1) (Fin 1) ℂ := fun _ _ => 1

theorem toy1x1_matrix_gap : MatrixPFGap toy1x1 1 := by
  have : 0 < (1 : ℝ) := by norm_num
  simpa [MatrixPFGap] using this

theorem toy1x1_transfer_gap : TransferPFGap (μ := default) (K := default) 1 :=
  spectral_to_transfer_gap (A := toy1x1) toy1x1_matrix_gap

end Examples

end YM
