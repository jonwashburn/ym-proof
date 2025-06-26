# Roadmap – Phase 2  (Filling Core Proof Gaps)

Now that all physical constants are parameters and skeleton files exist, Phase 2 focuses on turning the skeletons into complete Lean proofs and restoring a green build without `sorry`.

## 0  Key Build Goal
* `lake build` should succeed with **no** `sorry` **inside the core Yang-Mills chain**:
  `Parameters → Wilson/LedgerBridge → ReflectionPositivity → ContinuumLimit → Complete`.

Non-essential sub-projects (Navier–Stokes, gravity, extras) can keep `sorry` for now; they live in separate namespaces.

---

## 1  Priority Order

| Rank | File | Deliverable | Notes |
|------|------|-------------|-------|
| P1 | `Wilson/LedgerBridge.lean` | Proof that ledger cost ≤ Wilson action in strong-coupling window β ∈ (0,β₀). | Use standard expansion; may rely on Mathlib matrix identities. |
| P2 | `Measure/ReflectionPositivity.lean` | Chess-board proof of OS (RP). | Needs `MeasureTheory`, but only finite-volume case. |
| P3 | `RG/ContinuumLimit.lean` | Multi-scale induction showing gap limit exists & remains >0. | Follow Balaban-Fröhlich blueprint; acceptable to use `β→∞` monotonicity lemma. |
| P4 | `Topology/ChernWhitney.lean` | Compute H³(T⁴,ℤ₃) ≅ ℤ₃ and evaluate obstruction = q73. | Needs `Mathlib/AlgebraicTopology`. |
| P5 | `RG/StepScaling.lean` | Explicit derivation of six step factors, prove product≈7.55. | Numerical lemma acceptable so long as bounds are rational. |

---

## 2  Sub-task Breakdown

### 1. `Wilson/LedgerBridge`
1. Define plaquette angle: `θ_P : ℝ` via argument of trace.
2. Lemma: `1 − cos θ ≥ (2/π²) θ²` for |θ|≤π.
3. Show centre projection magnitude ≥ θ²/π².
4. Fix β₀ := π²/(6·E_coh φ).  Prove inequality for β<β₀.
5. Complete theorem.

### 2. `ReflectionPositivity`
1. Define time reflection `θ` on lattice sites.
2. Show ledger cost is even in first coordinate.
3. Factor measure into left / right halves, apply Cauchy–Schwarz.
4. Provide lemma `∫ f F(f)F(θf) ≥ 0`.

### 3. `ContinuumLimit`
1. Block-spin map B_L and spectral-gap monotone lemma.
2. Induction on scales a, aL.
3. Cauchy criterion for Schwinger functions.
4. Extract limit Δ and prove positivity.

### 4. `ChernWhitney`
1. Lean library: `K(ℤ_n,1)` for Z₃.
2. Decompose T⁴ = S¹×S¹×S¹×S¹, compute cup products.
3. Express SU(3) bundle via transition functions with non-trivial centre.
4. Evaluate third Stiefel–Whitney class, show value = 1 mod 3 on each plaquette ⇒ 72+1 total.

### 5. `StepScaling`
1. Define lattice coupling g(a) from transfer‐matrix eigenvalues.
2. Prove positivity & boundedness of β–function in strong coupling.
3. Integrate RG flow through six octaves (factor 2 scale steps).
4. Produce rational bounds for each cᵢ.
5. Prove product within [7.50,7.60].

---

## 3  Build & CI
* Add `scripts/check_sorry.py` to flag `sorry` inside `YangMillsProof/` but ignore `working/` and `gravity/` dirs.
* Update GitHub action to run `lake build` + the script.

---

## 4  Documentation
* Update `CONSTANTS_ROADMAP.md` status column as proofs replace parameters.
* Add appendix to manuscript summarising completed Lean theorem names.

---

## 5  Optional Quality-of-Life
* Provide `make doc` target that calls `lake exe cache get` then `doc-gen`.
* Introduce `#align` comments linking paper theorem numbers to Lean names.

---

## 6  Progress Tracker (tick when done)
- [ ] P1 Wilson bridge complete Ⓢ‐free
- [ ] P2 Reflection positivity Ⓢ‐free
- [ ] P3 Continuum limit Ⓢ‐free
- [ ] P4 Chern-Whitney 73 proof Ⓢ‐free
- [ ] P5 Step-scaling product Ⓢ‐free
- [ ] CI green (no `sorry` in core chain)
- [ ] Manuscript updated to match Lean proofs

*Last updated*: 

# Constants-to-Parameter Audit

- [ ] φ hard-coded definitions removed
- [ ] E_coh literals removed
- [ ] q73 literals removed
- [ ] λ_rec literals removed
- [ ] σ_phys literals removed
- [ ] β_critical literals removed
- [ ] a_lattice literals removed
- [ ] c6 literals removed
- [ ] transferMatrix uses RS.Param.φ
... 