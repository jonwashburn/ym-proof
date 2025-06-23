import Lake
open Lake DSL

package «YangMillsProof» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

require RecognitionScience from git
  "https://github.com/jonwashburn/RecognitionScience.git" @ "main" / "recognition-framework"

@[default_target]
lean_lib «YangMillsProof» where
  globs := #[
    .submodules `YangMillsProof.Stage0_RS_Foundation,
    .submodules `YangMillsProof.Stage1_GaugeEmbedding,
    .submodules `YangMillsProof.Stage2_LatticeTheory,
    .submodules `YangMillsProof.Stage3_OSReconstruction,
    .submodules `YangMillsProof.Stage4_ContinuumLimit,
    .submodules `YangMillsProof.Stage5_Renormalization,
    .submodules `YangMillsProof.Stage6_MainTheorem,
    .submodules `YangMillsProof.Infrastructure,
    .submodules `YangMillsProof.Topology,
    .submodules `YangMillsProof.RSImport
  ]
