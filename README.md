# YM Proof (fresh scaffold)

Minimal Lean 4 project scaffold for a Yang–Mills mass-gap pipeline with
interface-first modules. Prop-level adapters are provided so the whole
pipeline composes cleanly; you can strengthen pieces incrementally.

## Layout

- `ym/OSPositivity.lean` – OS/reflection-positivity interfaces and wrappers
- `ym/Reflection.lean` – reflection map, sesquilinear RP, OS adapter
- `ym/Transfer.lean` – Markov/transfer interfaces; PF-gap adapters
- `ym/Continuum.lean` – scaling family; persistence connector
- `ym/SpectralStability.lean` – P2/P5: gap persistence and embedding glue
- `ym/Interfaces.lean` – `GapCertificate`, `PipelineCertificate`, exports
- `ym/Pipeline.lean` – end-to-end glue + examples
- `ym/Embedding.lean` – trivial embedding demo invoking persistence
- `ym/Adapter/MatrixToTransfer.lean` – `TransferOp` and matrix→transfer bridge
- `ym/PF3x3.lean`, `ym/PF3x3_Bridge.lean` – finite PF route (bridge draft)
- `ym/MatrixTransferAdapter.lean` – matrix↦transfer adapter (Prop-level)
- `ym/Examples.lean` – assorted toy examples

## Building

1) Ensure `lean-toolchain` and mathlib pin are compatible and run:
```
lake build
```
If using cache:
```
lake exe cache get
```

CI: GitHub Actions builds the project on pushes and PRs, fails on any `sorry`/`axiom`,
and prints an axioms report for `YM.mass_gap_final`.

## Running the pipeline examples

Open `ym/Pipeline.lean` and jump to the examples namespace:

- `YM.Examples.toy_end_to_end` – end-to-end export at γ = 1
- `YM.Examples.two_thirds_end_to_end` – end-to-end export at γ = 2/3
 - `YM.Examples.three_fourths_end_to_end` – end-to-end export at γ = 3/4

See also the embedding demo in `ym/Embedding.lean` for a tiny finite‑dimensional
example showing how to invoke the persistence connector with trivial embeddings.

These use Prop-level interfaces, so they compile without additional
dependencies. For stronger results, replace the Prop-level pieces with
concrete proofs (e.g., PF gap via Gershgorin) and the pipeline continues
to compose unchanged.

You can also use the higher-level export in `ym/Interfaces.lean`:

```lean
open YM

-- Given a PipelineCertificate `p`, we can export directly:
#check pipeline_mass_gap_export
-- example usage (replace `p` with your certificate):
-- have h : MassGapCont p.γ := pipeline_mass_gap_export p
```

## Final theorem and axioms report

To use the final wrapper in code:

```lean
import YM
open YM

-- Given a PipelineCertificate `p`:
have hfinal : MassGapCont p.γ := mass_gap_final p
```

To inspect axioms for the final theorem locally, add at the end of a scratch
Lean file (or use CI):

```lean
import YM
open YM
#print axioms YM.mass_gap_final
```

Final theorem wrapper (no placeholders):

```lean
open YM

-- Given a `PipelineCertificate p` built from concrete components:
#check mass_gap_final
-- have hfinal : MassGapCont p.γ := mass_gap_final p
```

Quantitative export (once reflection→Dobrushin and kernel model land):

```lean
open YM

-- Given p : PipelineCertificate at base scale and a quantitative PF gap:
have hQuant : ∃ γ : ℝ, 0 < γ ∧ TransferPFGap (p.sf.μ_at ⟨0⟩) (p.sf.K_at ⟨0⟩) γ :=
  pf_gap_via_reflection_blocks (μ := p.sf.μ_at ⟨0⟩) (K := p.sf.K_at ⟨0⟩) (R := p.R)
    p.hRef (p.hBlk)
have : ∃ γ, MassGapCont γ := pipeline_mass_gap_export_quant p hQuant
```

## Contributing (parallel agents)

- Do not modify shared core types in a feature branch. If needed, propose a
  tiny interface PR first, then rebase dependents.
- One branch per agent; edit only owned files; keep changes small.
- If a needed mathlib lemma is missing, log a succinct blocker and stop.

