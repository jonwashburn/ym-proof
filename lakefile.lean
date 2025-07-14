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

@[default_target]
lean_lib YangMillsProof where
  srcDir := "YangMillsProof"
  roots := #[
    -- Foundation layer (integrated from foundation_clean)
    `foundation_clean.MinimalFoundation, `foundation_clean.RecognitionScience,
    `foundation_clean.Main, `foundation_clean.Core, `foundation_clean.Foundations,
    `foundation_clean.Parameters,
    -- Main proof components
    `Foundations, `RecognitionScience, `Parameters,
    `Gauge, `Continuum, `ContinuumOS, `Renormalisation, `RG,
    `Measure, `Topology, `Stage0_RS_Foundation, `Stage1_GaugeEmbedding,
    `Stage2_LatticeTheory, `Stage3_OSReconstruction, `Stage4_ContinuumLimit,
    `Stage5_Renormalization, `Stage6_MainTheorem, `Numerical, `Tests, `Wilson,
    `Infrastructure, `Analysis, `Main, `Complete
  ]
