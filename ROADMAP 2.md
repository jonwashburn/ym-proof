# Yang–Mills–Lean  —  Essential Completion Road-map

This document lists **only the tasks that must still be done** for the
repository to provide a *complete* formal proof that matches the narrative
manuscript.  Anything already fully proved in Lean is omitted.

Legend

| Symbol | Meaning |
|--------|---------|
| ☐      | work still required |

---
## 1 · Skeleton integrity (automation)

| Task | Status |
|------|--------|
| ☐  CI script scans **all** `.lean` files and fails on any<br>  `axiom` or `sorry`. |
| ☐  Script verifies that every LaTeX `\label{thm:…}` appearing in the manuscript has a Lean constant of the same name. |

Once these two scripts exist, their green status becomes the permanent
acceptance criterion.

---
## 2 · Mathematics still missing in Lean

Only items referenced in the manuscript are listed.

| Manuscript section / theorem | Lean file (should contain) | Work required |
|------------------------------|---------------------------|---------------|
| **Block–spin gap bound** (Thm 7.1) | `RG/BlockSpin.lean` | Define block-spin transform; prove bound `massGap(a·2) ≤ massGap(a)(1+Ca²)` and supply numerical constant `C`.
+  *Walk-through*  
+  1. **Define configuration types**：introduce a `LatticeGaugeField (a : ℝ)` structure parameterised by spacing.  
+  2. **Construct coarse graining**：`def blockSpin (L : ℕ) (h : 2 ≤ L) : LatticeGaugeField a → LatticeGaugeField (a*L)` taking oriented averages of plaquettes.  
+  3. **Lemma `blockSpin_preserves_gauge`**：show the map commutes with gauge transforms.  
+  4. **Define gap functional**：`massGap a := spectralGap (transferMatrix a)` which already exists for fine lattice.  
+  5. **Prove kernel inequality**：use Perron–Frobenius on the coarse kernel to obtain an estimate `λ₁(a*L) ≤ λ₁(a)(1+Ca²)`; translate eigenvalue gap into mass gap.  
+  6. **Choose constant**：set `C := 1` for strong coupling; bound logs with `Real.log`.  
+  7. Eliminate all `sorry` and export `block_spin_gap_bound`.
+     **Lean-level hints**  
+     • Reuse `Mathlib.Analysis.NormedSpace.Exponential` for `exp`.  
+     • `LinearAlgebra.Matrix.blockDiagonal` helps build block-spin transform on gauge links.  
+     • Gap can be expressed via `spectralRadius` in `Mathlib.Analysis.NormedSpace.OperatorNorm`.  
+     • Most inequalities are `linarith` once you rewrite logarithms with `Real.log_mul`.  
+     • Finish with `simp [block_spin_gap_bound, massGap]`.
| **Continuum gap persistence** | `RG/ContinuumLimit.lean` | Replace placeholder `gapScaling a := 1` with formula derived from the block-spin bound; update Cauchy/telescope proofs accordingly.
+  *Walk-through*  
+  1. Import the proven `block_spin_gap_bound`.  
+  2. Define `gapScaling a` as the *infimum* of `massGap (a*2^n)` normalised by `massGap a`.  
+  3. Show monotonicity of the normalised gap via the block-spin inequality.  
+  4. Replace constant-sequence argument with geometric-series bound using `gapScaling`.  
+  5. Provide explicit Cauchy proof using `Metric.cauchySeq`.  
+  6. Remove all placeholder comments.
| **Reflection positivity** (Stage 3) | `Measure/ReflectionPositivity.lean` | Implement θ–reflection and prove the standard OS inequality for the Wilson measure.
+  *Walk-through*  
+  1. Define lattice time-reflection operator `θ : (ℤ⁴) → ℤ⁴` mapping `(t,x)` to `(-t,x)`.  
+  2. On cylinder functions `F(φ)` define `θF` by pre-composition with `θ`.  
+  3. Use Wilson action's locality to prove `⟨F,θF⟩ ≥ 0` by writing integral as Boltzmann weights over halves of the lattice and applying Cauchy–Schwarz.  
+  4. Express in Lean via `MeasureTheory`—split domain into `north`/`south` time slices, apply `integral_mul_integral_le_l2`.  
+  5. Export lemma `reflection_positive : ∀ F, inner F (θ F) ≥ 0`.
| **OS → Wightman reconstruction** | `Stage3_OSReconstruction/ContinuumReconstruction.lean` | Construct Hilbert space & fields; prove Wightman axioms (currently assumed).
+  *Walk-through*  
+  1. Using reflection positivity, follow Osterwalder–Schrader: 
+     • Hilbert space = completion of cylinder functions modulo nulls.  
+     • Field operators via smeared plaquette insertion.  
+  2. Show translation invariance yields unitary group and define Hamiltonian via generator in time direction.  
+  3. Verify Wightman axioms: spectrum, locality (plaquette support), domain conditions.  
+  4. Provide Lean proofs using `InnerProductSpace`, `UniformSpace` modules.  
+  5. Export `isYangMillsHamiltonian` and `satisfiesWightmanAxioms` without `sorry`.
| **Transfer-matrix spectral gap** | `Stage2_LatticeTheory/TransferMatrixGap.lean` | Replace toy 3×3 matrix by the true SU(3) strong-coupling transfer matrix; compute its leading eigenvalues and show a positive gap.
+  *Walk-through*  
+  1. Define state space as product of link colour variables on a time slice.  
+  2. Build the Kogut–Susskind transfer operator `T = exp(-a H_Wilson)` in finite volume.  
+  3. Use strong-coupling expansion (small β) to truncate to centre-projected model; show resulting matrix is block-diagonal with each block identical.  
+  4. Compute principal eigenvalue λ₀ analytically (power series) and bound next eigenvalue λ₁ < λ₀ – ε.  
+  5. Choose ε ≥ 1/φ² as in manuscript; prove using combinatorial counting.  
+  6. Provide Lean proof with `LinearAlgebra.Matrix` + `PerronFrobenius`.
| **Integer 73 topological charge** (Thm \ref{thm:seventy-three}) | `Topology/ChernWhitney.lean` | Compute the relevant Stiefel–Whitney class on T⁴ and prove defect charge = 73.
+  *Walk-through*  
+  1. Model SU(3) bundle on T⁴ via transition functions in `π₁(SU(3)) = 0` but non-trivial `π₃`.  
+  2. Encode bundle as Čech 3-cocycle with integer coefficient.  
+  3. Use Lean's `AlgebraicTopology.Cohomology` library to compute `H³(T⁴,ℤ₃) ≅ ℤ₃`.  
+  4. Show eight-beat closure imposes generator exponent 73 (solve congruence).  
+  5. Conclude defect charge of each plaquette = 73.  
+  6. Export theorem `centre_charge_73 : defectCharge = 73`.
| **Step–scaling factors & two-loop RG** | `RG/StepScaling.lean` | Implement β-function solution; prove bounds `|c_i − φ^{1/3}| < 0.01` and product bound `7.5 < Π c_i < 7.6`.
+  *Walk-through*  
+  1. Define one- and two-loop coefficients `b₀, b₁`.  
+  2. Solve ODE `μ dg/dμ = β(g)` analytically to `g(μ)`.  
+  3. Define `c_i = g(2^i μ)/g(2^{i-1} μ)` and use mean-value + monotonicity to bound.  
+  4. Prove numerical inequalities with `interval_cases`, `norm_num`, and `Real.log` bounds.  
+  5. Provide Lean lemmas `c_i_bound` and `c_product_bound`.

If these seven items are implemented **without `sorry`** the Lean project will
automatically satisfy the acceptance scripts.

---
## 3 · Numerical envelope proofs (optional)

`Numerical/Envelope.lean` and `Tests/NumericalTests.lean` provide
regression-style checks on constants.  They can remain as is or be tightened
once the rigorous proofs above are done.

---
## 4 · Documentation synchronisation (optional polish)

After the mathematics is finished:

* regenerate manuscript cross-references with the Lean constant names;
* ensure CI publishes HTML & PDF artefacts.

These steps are cosmetic; they do not affect formal correctness. 