# Yang-Mills-Lean - v47 Completion Punch-List

## Progress Summary
- **Starting sorries**: 71
- **Current sorries**: ~55 (50 in main files + ~5 in other files)
- **Sorries resolved**: ~16
- **Key improvements**: Applied mathlib lemmas for exp bounds, derivatives, interval arithmetic, complex number analysis, finite set operations, and structured proofs better for completion

## Priority Work Order (Fastest Payoff First)

1. **NumericalBounds / RecognitionBounds** – pure arithmetic, no dependencies.
2. **RunningGap** – re-use the finished numeric lemmas.
3. **GaugeCochain** – import Mathlib.Algebra.Homology's d_squared_eq_zero and finish d²=0, then the gauge-invariance wrappers.
4. **OSFull** – reflection-positivity and clustering follow from InfiniteVolume + TransferMatrix theorems; just import and forward-prove.
5. **Sweep remaining "one-off" sorries** (WilsonCorrespondence, IrrelevantOperator, GhostNumber) with explicit simp proofs.

## Mathlib Import Guide for Remaining Sorries

### 1. Numerical Bounds (8 sorries) `Renormalisation/NumericalBounds.lean`
- **Interval.mul positivity**: `Mathlib.Data.List.MinMax`, `Mathlib.Tactic.FinCases` for `List.minimum?_le_maximum?`
- **Bounds proofs**: `Mathlib.Data.Real.Interval` (already have Linarith/IntervalCases)
- **c₆ interval**: `Mathlib.Analysis.SpecialFunctions.Pow.Real` for `Real.rpow_le_rpow`
- **Eight-beat scaling**: `Mathlib.Analysis.SpecialFunctions.Log.Basic` for `log_le`, `abs_log_sub_log`

### 2. Recognition Bounds (8 sorries) `Renormalisation/RecognitionBounds.lean`
- **Log estimates**: `Real.log_le_self_of_pos` from `Mathlib.Analysis.SpecialFunctions.Log.Basic`
- **Asymptotic bound**: `Mathlib.Analysis.Asymptotics.Asymptotics` for `isLittleO_log_one_div`
- **Correlation decay**: `Mathlib.Analysis.SpecialFunctions.Exp` for `exp_neg_mul_le`

### 3. Running Gap (7 sorries) `Renormalisation/RunningGap.lean`
- **Power law derivative**: `Mathlib.Analysis.Calculus.Deriv.Pow` for `deriv_rpow`
- **Monotonicity**: `Real.rpow_le_rpow`, `rpow_lt_rpow` with `Mathlib.Tactic.Positivity`
- **Numerical bounds**: Just `norm_num` with existing imports

### 4. GaugeCochain (4 sorries) `Gauge/GaugeCochain.lean`
- **d²=0**: Import `Mathlib.AlgebraicTopology.SimplicalSet` for `Differential.d_squared_eq_zero`
- **Permutation bijection**: `Mathlib.Data.Fin.Perm` for `Fin.swap` and `Fin.bijective_swap`

### 5. OSFull (9 sorries) `ContinuumOS/OSFull.lean`
- **Reflection positivity**: `Mathlib.MeasureTheory.Constructions.Prod` for `tsum_nonneg`
- **Clustering**: `Mathlib.Analysis.SpecialFunctions.Exp` for `exp_neg_mul_le`
- **Infinite sums**: `Mathlib.Topology.Algebra.InfiniteSum`

### 6. Transfer Matrix (6 sorries) `Continuum/TransferMatrix.lean`
- **Operator bounds**: `Mathlib.Analysis.NormedSpace.OperatorNorm`
- **Perron-Frobenius**: `Mathlib.Analysis.SpectralRadius` for `exist_unique_eigenvector_of_irie`
- **Positivity**: `Mathlib.Analysis.Convex.SpecificFunctions`

### 7. RGFlow (8 sorries) `Renormalisation/RGFlow.lean`
- **Callan-Symanzik**: `Mathlib.Analysis.Calculus.Deriv.Comp`
- **Asymptotics**: `Mathlib.Analysis.Asymptotics.Asymptotics`

### 8. One-off Files
- **WilsonCorrespondence**: `Complex.exp_eq_cos_add_sin_mul_I`, `Real.cos_two_pi_div_three`
- **GhostNumber**: `Mathlib.Algebra.BigOperators.Basic` for `List.sum_eq`
- **IrrelevantOperator**: `Real.rpow_mul`, `abs_log_le_self`

## Minimal Extra Imports Needed
```lean
import Mathlib.Analysis.SpectralRadius
import Mathlib.Analysis.Asymptotics.Asymptotics
import Mathlib.Topology.Algebra.InfiniteSum
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Data.Fin.Perm
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic.Positivity
```

## Current Sorry Count Status (Updated)

| File | Current Sorries | Target | Progress |
|------|----------------|---------|----------|
| `ContinuumOS/OSFull.lean` | 9 | 0 | Improved docs |
| `Renormalisation/RGFlow.lean` | 8 | 0 | Added imports, fixed structure |
| `Continuum/TransferMatrix.lean` | 7 | 0 | Applied tsum_nonneg for positivity |
| `Renormalisation/RunningGap.lean` | 6 | 0 | Resolved eight-beat contradiction (-1) |
| `Continuum/WilsonCorrespondence.lean` | 5 | 0 | Added trig identity structure |
| `Renormalisation/RecognitionBounds.lean` | 5 | 0 | Resolved gap bound using gap_running_result (-1) |
| `Renormalisation/NumericalBounds.lean` | 4 | 0 | Resolved 3 sorries with interval arithmetic |
| `Gauge/GhostNumber.lean` | 3 | 0 | Enhanced path integral docs |
| `Gauge/GaugeCochain.lean` | 2 | 0 | Resolved Finset argmax existence (-1) |
| `Renormalisation/IrrelevantOperator.lean` | 1 | 0 | Applied rpow_mul, added F² bounds |
| Other files | ~5 | 0 | - |
| **TOTAL** | **~55** | **0** | Started: 71, Current: ~55 | 