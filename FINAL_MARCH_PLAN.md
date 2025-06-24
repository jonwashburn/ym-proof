# Yang–Mills Lean Formalisation – Final March Plan (June 25 → July 02 2025)

## 0  Objective
Achieve a _fully rigorous, axiom-free_ Lean 4 formalisation of the Yang–Mills mass-gap proof (tagged **v48**), with

• **0 `sorry`** lines and **0 temporary placeholders** (`True` lemmas, etc.).  
• CI green on GitHub (`lake build`).  
• Release notes + preprint update.

---

## 1  Dependency Graph of Remaining Gaps
```
TransferMatrix spectral-gap  ─┬─►  cluster_property (InfiniteVolume)
                             │
                             ├─►  area-law + mass-gap lemma (OSFull)
                             │
                             ├─►  confinement_scale (RGFlow)
                             │
                             └─►  callan_symanzik  (RGFlow)
```
_Eliminating the analytic axioms in **TransferMatrix.lean** unlocks the proof of every remaining placeholder._

---

## 2  Task Breakdown

1. **TransferMatrix.lean** (currently: 0 user `sorry` + 5 analytic axioms)
   1. Import:
      ```lean
      import Mathlib.Analysis.SpectralRadius
      import Mathlib.Analysis.NormedSpace.OperatorNorm
      import Mathlib.Topology.Algebra.InfiniteSum
      ```
   2. Prove positivity & boundedness: ‖T‖ ≤ 1.
   3. Apply Perron–Frobenius ⇒ simple maximal eigenvalue λ₀ = 1.
   4. Schur test + kernel positivity ⇒ λ₁ ≤ exp(−Δ).
   5. Rename the 5 analytic axioms to lemmas and supply Lean proofs.

2. **InfiniteVolume.lean**
   * Replace `cluster_property := True` by real exponential-clustering proof using the transfer-matrix bound plus `Real.exp_neg_mul_le`.

3. **OSFull.lean** (7 sorries)
   * Use clustering to finish:
     * ledger quantum minimal-cost lemma (import from `PhysicalConstants`).
     * area-law inequality.
     * Cauchy–Schwarz inner-product bounds.
     * square-integrability proof.

4. **RGFlow.lean**
   * Restore original statements:
     * `confinement_scale` – prove divergence of `g_running μ` as μ ↓ Λ.
     * `callan_symanzik` – prove derivative identity using `gap_RGE`.

5. **WilsonCorrespondence.lean** (3 sorries)
   * Minimal‐excitation hypothesis: turn into explicit assumption or prove via ledger-quantum lemma.
   * Modular-phase arithmetic: use `Real.angleMod` & `Int.cast_mod` proofs.
   * Lattice → continuum bound via Taylor expansion `Real.cos_one_add_series`.

---

## 3  Timeline (replaces earlier punch-list)
| Date | Goal |
|------|------|
| **Jun 25** | Complete TransferMatrix spectral-gap lemma; CI passes. **Benchmark**: `grep -R "partition_function_le_one"` returns 0 and TransferMatrix has 0 axioms. |
| **Jun 26** | Prove clustering in InfiniteVolume & update OSFull area-law. **Benchmark**: `cluster_property` definition is non-`True`; OSFull `sorry` count ≤ 3. |
| **Jun 27** | Close all OSFull sorries. **Benchmark**: `grep -R "OSFull.*sorry"` returns 0. |
| **Jun 28** | Restore confinement_scale & callan_symanzik in RGFlow. **Benchmark**: `RGFlow` contains no `True` placeholders. |
| **Jun 29** | Finish WilsonCorrespondence proofs; workspace now **0 sorries**. **Benchmark**: global `grep -R "sorry"` count 0. |
| **Jun 30 – Jul 02** | Code clean-up, `grep "^axiom"` audit, final CI, tag v48, submit preprint. |

---

## 4  Success Checklist
- [ ] TransferMatrix lemmas proved
- [ ] All placeholders removed (`cluster_property`, `confinement_scale`, `callan_symanzik`)
- [ ] OSFull, WilsonCorrespondence compile without `sorry`
- [ ] `grep -R "sorry"` returns zero matches
- [ ] `grep -R "^axiom"` returns zero matches
- [ ] `lake build` & GitHub Actions green
- [ ] Release tag **v48** created

---

## 5  Immediate Next Action
Start with **TransferMatrix.lean**: implement positivity, Schur test, and Perron–Frobenius spectral-gap proof.

---

## 6  Session Progress Reports

### Session 1 (Dec 25, 2024)
**Status**: TransferMatrix work in progress  
**Sorries**: 14 total (was 10, added 4 in TransferMatrix)
- OSFull: 7
- WilsonCorrespondence: 3  
- TransferMatrix: 4 (new)
- RecognitionBounds: 0 ✓
- RGFlow: 0 ✓ (but has 2 placeholder `True` lemmas)
- InfiniteVolume: 0 ✓ (but has 1 placeholder `True` definition)

**Completed**:
- Created final march plan with benchmarks
- Added missing lemma definitions to TransferMatrix
- Proved `kernel_mul_psi_summable` without sorries
- Build passes

**Next steps**:
- Rethink TransferMatrix approach to avoid adding sorries
- Consider whether to accept modeling assumptions vs full proofs
- Focus on removing existing sorries in OSFull/WilsonCorrespondence

**Benchmark check**:
- ❌ `grep -R "partition_function_le_one"` returns 2 (definition + usage)
- ❌ TransferMatrix has 4 sorries (not 0)

_(Document created 2025-06-25)_ 