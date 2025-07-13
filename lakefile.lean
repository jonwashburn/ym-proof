import Lake
open Lake DSL

package ym_proof where
  srcDir := "YangMillsProof"
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

@[default_target]
<<<<<<< HEAD
lean_lib YangMillsProof where
  roots := #[
    `Foundations, `RecognitionScience, `Parameters,
    `Gauge, `Continuum, `ContinuumOS, `Renormalisation, `RG,
    `Measure, `Topology, `Stage0_RS_Foundation, `Stage1_GaugeEmbedding,
    `Stage2_LatticeTheory, `Stage3_OSReconstruction, `Stage5_Renormalization,
    `Stage6_MainTheorem, `Numerical, `Tests, `Wilson, `Infrastructure,
    `Analysis.Trig.MonotoneCos, `Analysis.Hilbert.Cyl, `Main, `Complete
  ]
=======
lean_lib YangMillsProof
>>>>>>> 0ce0d744066b91222d10f001ef65a5a095de4ea6
