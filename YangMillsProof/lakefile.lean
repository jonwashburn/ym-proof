import Lake
open Lake DSL

package «recognition-science» where
  -- Basic settings
  moreLeanArgs := #[
    "-DautoImplicit=false",
    "-DrelaxedAutoImplicit=false"
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

@[default_target]
lean_lib «Core» where
  -- Core modules
  roots := #[`Core.Finite, `Core.MetaPrincipleMinimal, `Core.MetaPrinciple, `Core.EightFoundations, `Core.Nat.Card]

lean_lib «Foundations» where
  -- Concrete foundation implementations
  roots := #[
    `Foundations.DiscreteTime,
    `Foundations.DualBalance,
    `Foundations.PositiveCost,
    `Foundations.UnitaryEvolution,
    `Foundations.IrreducibleTick,
    `Foundations.SpatialVoxels,
    `Foundations.EightBeat,
    `Foundations.GoldenRatio
  ]

lean_lib «RecognitionScience» where
  -- Main library combining everything
  roots := #[`RecognitionScience]

lean_exe «numerical_tests» where
  -- Numerical verification executable
  root := `Tests.NumericalTests
