import Lake
open Lake DSL

package ym_proof where
  srcDir := "."
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

-- Analysis library in root directory
lean_lib Analysis where
  srcDir := "Analysis"
  roots := #[`Trig.MonotoneCos]

-- RSImport library for Recognition Science definitions
lean_lib RSImport where
  srcDir := "RSImport"
  roots := #[`RSImport, `RSImport.BasicDefinitions]

-- Incrementally add working YangMillsProof modules
@[default_target]
lean_lib YangMillsProof where
  srcDir := "YangMillsProof"
  roots := #[
    `Stage0_RS_Foundation.ActivityCost,
    `Stage1_GaugeEmbedding.VoxelLattice,
    `Stage1_GaugeEmbedding.GaugeToLedger,
    `Stage2_LatticeTheory.TransferMatrixGap,
    `Stage2_LatticeTheory.FermionTransferMatrix,
    `Stage3_OSReconstruction.ContinuumReconstruction_Simple,
    `Stage3_OSReconstruction.FractionalActionRP,
    `Stage4_ContinuumLimit.MassGapPersistence,
    `Stage5_Renormalization.IrrelevantOperator,
    `foundation_clean.MinimalFoundation,
    `foundation_clean.RecognitionScience,
    `Stage0_RS_Foundation.LedgerThermodynamics
  ]
