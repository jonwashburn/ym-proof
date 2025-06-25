# Lean Punch-List – January 2025

This document tracks the *remaining* Lean work required to bring the codebase fully in line with the new manuscript (v → version-less).

Each entry lists:
• **File / namespace to create or edit**  
• **Concrete deliverable**  
• **Blocking dependencies**  
• **Status** (todo / in-progress / done)

---

## 1  Continuum limit & RG trajectory
| File | Deliverable | Depends on | Status |
|------|-------------|-----------|--------|
| `RG/BlockSpin.lean` | Definition of block-spin map `B_L`, proof of Thm 7.1 (commutes with gauge + reflection) | `RecognitionScience.Basic`, `Gauge.Covariance` | **todo** |
| `RG/StepScaling.lean` | Six step-scaling constants `c₁,…,c₆` + Lean theorem `step_scaling_bounds` | `RG/BlockSpin` | **todo** |
| `RG/RunningGap.lean` | Sequence `Δₙ`, theorem `running_gap : tendsto _ _` producing physical gap | `StepScaling`, `TransferMatrix` | **todo** |
| `RG/BlockSpin.lean` | Lemma `uniform_gap` (monotone gap, used in manuscript Eq (7.2)) | `TransferMatrix` | **todo** |
| `RG/ContinuumLimit.lean` | Theorem `continuum_limit_exists : ∃ Δ₀, …` | all of the above | **todo** |

## 2  Plaquette energy & constants
| File | Deliverable | Depends on | Status |
|------|-------------|-----------|--------|
| `StrongCoupling/PlaquetteEnergy.lean` | Derivation of exact plaquette energy `E_P = 0.090 eV` from SU(3) Haar measure | mathlib integration | **todo** |
| `StrongCoupling/GoldenRatio.lean` | Proof `golden_ratio_unique` for recursion φ² = φ+1 in vortex counting | `Mathlib.Analysis.SpecialFunctions` | **todo** |

## 3  Cohomology derivation of 73
| File | Deliverable | Depends on | Status |
|------|-------------|-----------|--------|
| `Topology/ChernWhitney.lean` | Formal construction of torus SU(3) bundle, computation of w₃ | mathlib `Topology` | **todo** |
| `Topology/CenterCohomology.lean` | Proof `defect_charge_73` consumed by `Ledger.FirstPrinciples` | `ChernWhitney` | **todo** |
| `Ledger/FirstPrinciples.lean` | Replace placeholder comments with `open Topology.CenterCohomology` and use lemma | above | **todo** |

## 4  Infinite-dimensional transfer operator
| File | Deliverable | Depends on | Status |
|------|-------------|-----------|--------|
| `TransferMatrix/Infinite.lean` | Define operator `𝕋` on direct sum of colour sectors; prove gap inherits | `RecognitionScience.Basic`, `GaugeResidue`, `Topology.CenterCohomology` | **todo** |

## 5  Reflection positivity on measure level
| File | Deliverable | Depends on | Status |
|------|-------------|-----------|--------|
| `Measure/ReflectionPositivity.lean` | Define Wilson measure on cylinder σ-algebra; prove θ-positivity | mathlib `MeasureTheory`, `Gauge.Covariance` | **todo** |

## 6  Finishing existing sorries (auxiliary modules)
| File | # sorries | Action |
|------|-----------|--------|
| `Ledger/Energy.lean` | 6 | Complete RS cost lemmas (superadditivity, vacuum uniqueness) |
| `Ledger/Quantum.lean` | 3 | Finalise `stateCost` definition, vacuum uniqueness proof |
| `StatMech/ExponentialClusters.lean` | 3 | Spectral decomposition + logarithm algebra |
| `BRST/Cohomology.lean` | 5 | Complete finite-dimensional cohomology argument |
| `Gauge/Covariance.lean` | 2 | Finish quotient-space universal property |
| `FA/NormBounds.lean` | 2 | Derivative bound and gap-to-L² estimate |

## 7  CI / automated verification
| Task | Status |
|------|--------|
| Add GitHub Actions workflow running `lake build && ./scripts/check_axioms.sh` | **todo** |
| Publish badge in README | **todo** |

---

### Legend
* **todo** – not yet started  
* **in-progress** – branch exists, partial proofs  
* **done** – merged into `main`

---
Jonathan Washburn — 17 Jan 2025 