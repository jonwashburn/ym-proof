import Lake
open Lake DSL

package «ym-proof» where
  srcDir := "YangMillsProof"

require mathlib from
  git "https://github.com/leanprover-community/mathlib4" @ "master"

lean_lib YMProof

targets := #[ ]
