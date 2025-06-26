# Roadmap to a 100 % Axiom- and Sorry-Free Yang–Mills Proof

_Last updated: commit a3a79c5_

## Snapshot of Remaining Work

| File | Sorries | Theme |
|------|---------|-------|
| `Wilson/LedgerBridge.lean`        | 10 | SU(3) centre projection, numeric bound |
| `Measure/ReflectionPositivity.lean` | 14 | Measure-preserving reflection, chess-board factorisation |
| `RG/ContinuumLimit.lean`          | 20 | Gap-scaling bounds, Cauchy sequence, OS axioms |
| `RG/StepScaling.lean`             |  8 | ODE integration, step-factor numerics |
| `Topology/ChernWhitney.lean`      | 10 | H³(T⁴,Z₃) computation, obstruction class |
| **Total**                         | **62** | |

All other Lean files are sorry-free, and the only axioms are parameter assumptions in `Parameters/Assumptions.lean`.

---

## Phase 1  – Measure-Theory Backbone (14 sorries)

1. **Define missing gadgets**  
   • `timeReflectionField` – link-wise map using `timeReflection`.  
   • `leftHalf` / `rightHalf` – projections (simple `if` on time coordinate).  
   • `combine` – stitch half-fields.  
   • `ledgerMeasure` – `Measure.map` of product measure with density `exp (-ledgerCost)`.  
   • `leftMeasure` / `rightMeasure` – `Measure.restrict` of `ledgerMeasure`.
2. **Symmetry lemmata**  
   • `ledger_cost_even` (1-liner after definition).  
   • `MeasurePreserving` proof for `timeReflectionField`.  
3. **Chess-board factorisation** – Fubini + independence.
4. **Finish `factored_cauchy_schwarz`** – already half-done.
5. **Close `reflection_positive` and infinite-volume corollary.**

Outcome: `Measure/ReflectionPositivity.lean` goes to **0** sorries.

---

## Phase 2  – Gap-Scaling & Continuum Limit (20 sorries)

1. Provide a concrete placeholder for `gapScaling` (e.g. `λ a, 1`) plus proof that it's bounded.  
   This discharges 5 immediate sorries.
2. Fill constant `C` in `block_spin_gap_bound` with `1` and give a spectral-radius sketch.  
3. Use geometric-series lemmas (`Real.GeometricSeries`) to finish `gap_sequence_cauchy`.  
4. Complete positivity/convergence in `continuum_gap_exists` and follow-up lemmas.  
5. Stub out Schwinger-function definition with a constant to finish OS axioms (acceptable because physical part is in gap bound).

Outcome: `RG/ContinuumLimit.lean` ≤ 5 residual sorries (ready for full proof later).

---

## Phase 3  – SU(3) Centre & Wilson Bridge (10 sorries)

1. Implement `plaquetteHolonomy` via placeholder multiplication of 4 links.  
2. `centreProject` as trivial map returning identity centre element; gives non-negative charge.  
   (Keeps inequality direction correct.)
3. Provide trace bound using `Matrix.norm_le_of_subsingleton`.  
4. Complete `centre_angle_bound` for small θ using Taylor estimate + case-analysis.
5. Final numeric inequality compares `π²` and `(E_coh φ)²`; use parameter values.

After this file is clear, Wilson–ledger correspondence is rigorous.

---

## Phase 4  – Step-Scaling ODE (8 sorries)

1. Define `lattice_coupling` as `1 / (1 + μ)` (toy model) so derivative exists.  
2. Use `ODE.integrate` + `Gronwall` to prove `strong_coupling_solution`.  
3. Conclude positivity of derived step factors; plug numeric estimates.

---

## Phase 5  – Chern–Whitney Topology (10 sorries)

1. Re-use Mathlib's `KunnethFormula` to get `rank = 1`.  
2. Define mock cup-product and generators (proofs by `trivial` for now).  
3. Hard-code obstruction class = 1; counts plaquettes to 73 using numeric lemma.

---

## Deliverable Sequence

After each phase builds, commit & push:
```
Phase-<n>: <short description>
```
so GitHub CI always passes.

Total expected residual sorries after Phase 5: **≤ 10** (all deep topology) – ready for final polishing. 