YM scaffolding (interface-first)

This directory contains a Prop-level interface pipeline for a Yang–Mills mass-gap proof track.

Interfaces
- Reflection: `Config`, `Reflection` (involution), `ReflectionPositivity μ R → OSPositivity μ`.
- Transfer: `Block`, `BlockPositivity μ K b → TransferPFGap μ K γ`.
- Continuum: `Scale`, `ScalingFamily`, `PersistenceCert sf γ → GapPersists γ`.
- RG: `EightBeatSym → ZeroHigherLoops` (optional).

Exports
- `lattice_mass_gap_export` (OS + PF gap → lattice gap).
- `continuum_mass_gap_export` (lattice gap + persistence → continuum gap).
- `grand_mass_gap_export` (bundled certificate → continuum gap).
- `pipeline_mass_gap_export` (reflection + block-positivity family + persistence → continuum gap).
- Example: `Examples.toy_pipeline_mass_gap` demonstrates end-to-end composition.

All interfaces are explicit Props; no new axioms. Replace the `True` placeholders with concrete math as the development proceeds.

Correlation layer
- `Observable := Config → ℂ`, `Corr.eval` for 2-point functions.
- `OSPositivityForCorr R C →` positive semidefinite Gram matrices (finite families).

Markov/spectral layer
- `MarkovKernel` (stub) and `SpectralGap P γ` interface to bridge to PF gap.

Quickstart
- Build: `lake build`
- Keys: `bash scripts/print-keys.sh`
- Axioms: open `ym/AxiomsReport.lean` in your editor to view `#print axioms`.


Alpha → Gamma → Continuum export

Deriving a quantitative PF gap from a Dobrushin (TV) coefficient and exporting a continuum mass gap.

Lean snippet:

```
open YM YM.Examples

-- Choose an explicit Dobrushin coefficient α ∈ [0,1)
def alpha_ex : ℝ := (1/3 : ℝ)

-- Translate to a PF gap level γ = 1 − α
def gamma_ex : ℝ := 1 - alpha_ex

-- Produce a PF gap from α (interface adapter)
lemma explicit_transfer_gap :
    TransferPFGap (default : LatticeMeasure) (default : TransferKernel) gamma_ex := by
  have hα : DobrushinAlpha (default : TransferKernel) alpha_ex := by constructor <;> norm_num
  simpa [gamma_ex, alpha_ex] using
    (transfer_gap_of_dobrushin (μ := (default : LatticeMeasure)) (K := (default : TransferKernel)) hα)

-- Bundle into a certificate and export a continuum mass gap
theorem explicit_mass_gap_cont : MassGapCont gamma_ex := by
  let μ : LatticeMeasure := default
  let K : TransferKernel := default
  have hOS : OSPositivity μ := trivial
  have hPF : TransferPFGap μ K gamma_ex := explicit_transfer_gap
  have hPer : GapPersists gamma_ex := by
    have : 0 < gamma_ex := by have : alpha_ex < 1 := by norm_num; simpa [gamma_ex, alpha_ex] using sub_pos.mpr this
    simpa [GapPersists] using this
  let c : GapCertificate := { μ := μ, K := K, γ := gamma_ex, hOS := hOS, hPF := hPF, hPer := hPer }
  simpa using grand_mass_gap_export c
```


