import ym.OSPositivity
import ym.RGOneLoop
import ym.Reflection
import ym.Transfer
import ym.Continuum
import ym.Embedding
import ym.Adapter.MatrixToTransfer

/-!
YM Main assembly: exposes key wrapper theorems as public names for reporting.
Prop-level; no axioms.
-/

namespace YM

open scoped Classical

variable {μ : LatticeMeasure} {K : TransferKernel} {γ : ℝ}

/-- Public export: lattice mass gap from OS positivity and PF gap. -/
theorem lattice_mass_gap_export
    (hOS : OSPositivity μ) (hPF : TransferPFGap μ K γ) : MassGap μ γ :=
  mass_gap_of_OS_PF (μ:=μ) (K:=K) (γ:=γ) hOS hPF

/-- Public export: continuum mass gap from lattice gap and persistence. -/
theorem continuum_mass_gap_export
    (hGap : MassGap μ γ) (hPers : GapPersists γ) : MassGapCont γ :=
  mass_gap_continuum (μ:=μ) (γ:=γ) hGap hPers

/-- Public export: one-loop exactness from the discrete 8‑beat symmetry certificate. -/
theorem one_loop_exact_export (h : EightBeatSym) : ZeroHigherLoops :=
  one_loop_exact_of_clock h

end YM
