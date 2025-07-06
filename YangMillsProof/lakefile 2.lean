import Lake
open Lake DSL

package yang_mills_proof where
  moreLeanArgs := #["-DautoImplicit=false", "-DrelaxedAutoImplicit=false"]
  packagesDir := ".lake/packages"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

-- Remove ProofWidgets for now to avoid compatibility issues
-- require proofwidgets from git
--   "https://github.com/leanprover-community/ProofWidgets4.git" @ "v0.0.42"

-- Core modules
lean_lib Core where
  roots := #[`Core.Finite, `Core.MetaPrinciple, `Core.MetaPrincipleMinimal,
            `Core.EightFoundations, `Core.Nat.Card, `Core.Constants]

-- Foundation implementations
lean_lib Foundations where
  roots := #[`Foundations.DiscreteTime, `Foundations.DualBalance,
            `Foundations.PositiveCost, `Foundations.UnitaryEvolution,
            `Foundations.IrreducibleTick, `Foundations.SpatialVoxels,
            `Foundations.EightBeat, `Foundations.GoldenRatio]

-- Recognition Science main module
lean_lib RecognitionScience where
  roots := #[`RecognitionScience]

-- YangMillsProof submodules
lean_lib YangMillsProof.Parameters where
  roots := #[`Parameters.Constants, `Parameters.DerivedConstants,
            `Parameters.FromRS, `Parameters.Assumptions]

lean_lib YangMillsProof.Gauge where
  roots := #[`Gauge.GaugeCochain, `Gauge.BRST, `Gauge.GhostNumber,
            `Gauge.Lattice, `Gauge.SU3]

lean_lib YangMillsProof.Continuum where
  roots := #[`Continuum.WilsonMap, `Continuum.Continuum,
            `Continuum.TransferMatrix, `Continuum.WilsonCorrespondence]

lean_lib YangMillsProof.ContinuumOS where
  roots := #[`ContinuumOS.InfiniteVolume, `ContinuumOS.OSFull]

lean_lib YangMillsProof.Renormalisation where
  roots := #[`Renormalisation.RunningGap, `Renormalisation.IrrelevantOperator,
            `Renormalisation.RGFlow, `Renormalisation.NumericalBounds,
            `Renormalisation.RecognitionBounds]

lean_lib YangMillsProof.RG where
  roots := #[`RG.BlockSpin, `RG.ContinuumLimit, `RG.ExactSolution,
            `RG.StepScaling]

lean_lib YangMillsProof.Measure where
  roots := #[`Measure.ReflectionPositivity]

lean_lib YangMillsProof.Topology where
  roots := #[`Topology.ChernWhitney, `Topology.GaugeSector]

lean_lib YangMillsProof.Stage0_RS_Foundation where
  roots := #[`Stage0_RS_Foundation.ActivityCost,
            `Stage0_RS_Foundation.LedgerThermodynamics]

lean_lib YangMillsProof.Stage1_GaugeEmbedding where
  roots := #[`Stage1_GaugeEmbedding.GaugeToLedger,
            `Stage1_GaugeEmbedding.VoxelLattice]

lean_lib YangMillsProof.Stage2_LatticeTheory where
  roots := #[`Stage2_LatticeTheory.TransferMatrixGap]

lean_lib YangMillsProof.Stage3_OSReconstruction where
  roots := #[`Stage3_OSReconstruction.ContinuumReconstruction,
            `Stage3_OSReconstruction.FractionalActionRP]

lean_lib YangMillsProof.Stage5_Renormalization where
  roots := #[`Stage5_Renormalization.IrrelevantOperator]

lean_lib YangMillsProof.Stage6_MainTheorem where
  roots := #[`Stage6_MainTheorem.Complete]

lean_lib YangMillsProof.Numerical where
  roots := #[`Numerical.Constants, `Numerical.Envelope,
            `Numerical.Interval, `Numerical.PrintMassGap]

lean_lib YangMillsProof.Tests where
  roots := #[`Tests.FaithfulnessTests, `Tests.NumericalTests]

lean_lib YangMillsProof.Wilson where
  roots := #[`Wilson.LedgerBridge]

lean_lib YangMillsProof.Infrastructure where
  roots := #[`Infrastructure.PhysicalConstants]

lean_lib TrigExtras where
  roots := #[`Analysis.Trig.MonotoneCos]

lean_lib HilbertExtras where
  roots := #[`Analysis.Hilbert.Cyl]

-- Main target combining all modules
@[default_target]
lean_lib YangMillsProof where
  roots := #[`Main, `Complete, `PhysicalConstants, `GaugeLayer, `L2State, `TransferMatrix]
