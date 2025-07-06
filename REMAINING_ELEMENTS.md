# 📋 Roadmap to a Fully-Rigorous Yang–Mills Mass Gap Proof

*Last updated: <!-- date will be filled automatically by git timestamps -->*

---

## 0  Executive Goal

Deliver a Lean 4 project that:

1. **Introduces no new axioms** beyond the Lean kernel.
2. **Contains zero `sorry` statements**.
3. Provides a mathematically standard construction of **SU(3) Yang–Mills theory on ℝ⁴**.
4. Proves that the corresponding Hamiltonian possesses a **positive spectral gap** ("mass gap").

Everything below is organised so that completing every checkbox ⇨ repository meets the four criteria above.

---

## 1  Axiom Elimination Matrix

| File | Line(s) | Axiom Constant | Action | Linked Section |
|------|---------|----------------|--------|----------------|
| `Stage3_OSReconstruction/ContinuumReconstruction.lean` | 210-215 | `measure_inner_nonneg`, `wilson_measure_normalized`, `semiInner_*` | Replace with formal proofs using `Mathlib.MeasureTheory` APIs | §2.2, §2.3 |
| `RecognitionScience/BRST/Cohomology.lean` | 28-32 | Five BRST axioms | Construct BRST complex; port proofs from Mathlib's homological algebra | §3.2 |
| `ledger-foundation/Core/*` | misc | Physical constants (`k_B`, quark masses, …) | Either derive, or **move to `Parameters/Assumptions.lean` & mark as axioms** (max four allowed) | §2.5 |

**Deliverable:** `./verify_no_axioms.sh` prints `Files containing axiom declarations: 0`.

---

## 2  `sorry` Eradication Checklist

> Current `lake build` passes, but 6 `sorry`s remain.  Once these are gone, re-enable lint `assertNoSorry`.

### 2.1  OS Reconstruction (5 sorries)

- [ ] `fieldOperator` implementation (def)  
  *Define smeared gauge field via Wilson loops & test functions; use `ContinuousLinearMap`.*
- [ ] `timeEvolution` (def)  
  *Construct strongly-continuous one-parameter unitary group `e^{-i t H}` once `H` exists.*
- [ ] `hamiltonian_positive` (thm)  
  *Show positivity via reflection positivity ⇒ Osterwalder–Schrader reconstruction.*
- [ ] `hamiltonian_mass_gap` (thm)  
  *Spectral gap ≥ `E_coh·φ`.  Needs spectral theorem + cluster expansion bounds.*
- [ ] `wilson_cluster_decay` placeholder bound (in `Measure/Wilson.lean`)  
  *Prove exponential decay using standard correlation-length arguments.*

### 2.2  Measure Theory Proofs (replace axioms)

- [ ] `measure_inner_nonneg`  
  *Use `L2` positivity: `∫|f|² ≥ 0`.*
- [ ] `wilson_measure_normalized`  
  *Compute normalization constant via Gaussian integral or bound + limit.*
- [ ] `semiInner_*` add/scale lemmas  
  *Follow `InnerProductSpace` linearity proofs.*

### 2.3  BRST Cohomology Axioms

- [ ] Build chain complex `C•`, differential `Q`, & prove `Q²=0`.
- [ ] Define physical state space `H_phys = ker Q / im Q`.
- [ ] Prove positivity of BRST inner product (`⟨ψ,ψ⟩ ≥ 0`).

---

## 3  Placeholder → Concrete Implementations

| Construct | Current Status | Needed Work |
|-----------|----------------|-------------|
| **Hamiltonian `H`** | `0` operator | (a) define on cylinder functions via transfer matrix; (b) show essentially self-adjoint; (c) extend to completion. |
| **Field operators** | `0` operator | Construct gauge-invariant smeared fields; verify locality & covariance. |
| **Physical Hilbert `ℋ_phys`** | alias of `PreHilbert` | Complete metric space: use `UniformSpace.Completion`. |
| **Wilson Measure (`wilsonMeasure`)** | stub densities | Kolmogorov extension of finite-volume Gaussians; prove probability & reflection positivity. |

---

## 4  Yang–Mills-Specific Content Still Missing

1. **SU(3) Gauge Structure**  
   ‑ Formalise lattice gauge fields `U : Edges → SU(3)`; plaquette action.  
   ‑ Embed Recognition-Science ledger into gauge variables (§5 of paper).
2. **Continuum Limit (`a→0`)**  
   ‑ Use Osterwalder-Seiler block-spin RG or Bałaban's renormalisation flow.  
   ‑ Show plaquette expectation → `exp(-g² E[B²+E²])`.
3. **Confinement / Wilson Area Law**  
   ‑ Rigorously derive string tension `σ = massGap² /(8 E_coh)`; remove heuristic inequalities.
4. **Mass-Gap Lower Bound**  
   ‑ Combine spectral gap on lattice with continuity estimates to physical units.

---

## 5  Testing & CI

- [ ] Re-enable `AssertNoAxiom` linter (Mathlib).
- [ ] Re-enable `AssertNoSorry` linter.
- [ ] Add GitHub CI workflow: `lake build && ./verify_no_axioms.sh && leanproject lint`.
- [ ] Provide `lake doc` artefact upload.

---

## 6  Timeline (Suggested)

| Week | Milestone |
|------|-----------|
| 1 | Measure-theory proofs §2.2 complete; `wilsonInner` cluster bound proved |
| 2 | Implement `fieldOperator`, `timeEvolution`; eliminate 3 sorries |
| 3 | Define Hamiltonian; prove positivity (`hamiltonian_positive`) |
| 4 | Spectral analysis → eliminate `hamiltonian_mass_gap` sorry |
| 5 | Replace BRST axioms with proofs; zero axioms across repo |
| 6 | SU(3) lattice variables + transfer matrix gap proof |
| 7 | Continuum limit & area law rigor |
| 8 | Full CI green: 0 axioms, 0 sorries |

*(Adjust pacing as contributors join.)*

---

## 7  Reading List / References

1. **Glimm & Jaffe, "Quantum Physics: A Functional-Integral Point of View"** – OS reconstruction & cluster expansions.
2. **Bałaban's Papers (1980–1986)** – Non-perturbative renormalisation group for lattice gauge theories.
3. **Araujo, Fox, et al. (2024)** – Formalization of Gaussian measures in Lean.
4. **Thompson, "Renormalization Group"** – For gap preservation under RG flow.
5. **Mathlib Docs** for `MeasureTheory`, `InnerProductSpace.Quotient`, `UniformSpace.Completion`.

---

## 8  Contributor On-Boarding

* Clone repo & run `lake exe cache get` for fast build.
* See `SORRY_RESOLUTION_PLAN.md` for detailed proofs templates.
* Use issues labelled `good-first-proof` for manageable lemmas.
* Join the Zulip stream **#proj-yang-mills-lean** for discussion.

---

### Checklist Snapshot *(to be updated in PRs)*

- [ ] 0 Axioms  
- [ ] 0 Sorries  
- [ ] CI Green  
- [ ] OS Axioms verified  
- [ ] Mass-gap theorem proved against concrete Hamiltonian 