import Lake
open Lake DSL

package «yang-mills-proof» where
  -- Standard Lean options
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

-- Mathlib dependency
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

-- Analysis library (for MonotoneCos)
lean_lib «Analysis» where
  srcDir := "Analysis"
  roots := #[`Trig.MonotoneCos]

-- Main library
@[default_target]
lean_lib «YangMillsProof» where
  -- All source files are in the YangMillsProof directory
  srcDir := "YangMillsProof"
  -- Specify the main entry point
  roots := #[`Main, `RecognitionScience, `PhysicalConstants, `MinimalFoundation,
             `Continuum, `Gauge, `Renormalisation, `ContinuumOS, `Foundations, `Parameters]

lean_lib «foundation_clean» where
  srcDir := "YangMillsProof/foundation_clean"
  roots := #[`RecognitionScience, `MinimalFoundation, `ZeroAxiomFoundation, `Fintype.Basic, `Core.Physics.MassGap, `Core.Physics.RungGap]
